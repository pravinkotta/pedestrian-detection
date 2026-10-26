"""Export the trained detector to ONNX for lightweight serving.

Run:
    python -m src.models.export_onnx                       # models/model.onnx
    python -m src.models.export_onnx --checkpoint models/model.pt --verify

Why ONNX
--------
The serving image has to carry a deep-learning runtime just to run
inference. Shipping PyTorch costs ~1 GB of image; ONNX Runtime costs
~15 MB and is typically *faster* on CPU because the graph is fused and
quantisation is straightforward. Same weights, same numbers, far less
weight dragged into every container pull.

The trade-off (stated honestly, because it matters for a review):
the graph is traced once, so input shapes are fixed unless
``--dynamic`` succeeds, and any change to the model architecture
requires re-exporting. That is the normal, acceptable contract for
deploying a frozen checkpoint.

Why a wrapper module
--------------------
Faster R-CNN returns ``list[dict]`` — not representable as a stable ONNX
graph. The wrapper flattens that into three plain tensors (boxes, scores,
labels) whose **first dimension is dynamic**: it is data-dependent, so it
is declared as ``num_dets`` rather than frozen. The serving layer slices
whatever it gets.

The subtle part is described in :func:`make_wrapper`: the TorchScript
tracer turns Python ints computed from tensor shapes into constants, so
every data-dependent count in torchvision's RPN/box head has to be pinned
down *before* tracing or the exported graph silently assumes the sizes it
happened to see.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# fixed canvas: torchvision's GeneralizedRCNNTransform pads to a multiple of
# 32, and a static shape keeps RoIAlign + FPN math exactly reproducible.
DEFAULT_H, DEFAULT_W = 192, 256

# Fixed number of RPN proposals handed to the box head. MUST stay <= the
# effective rpn post-NMS cap (build_model.POST_NMS_TOP_N_TEST, applied by
# apply_rpn_caps) — enforced by the guard in make_wrapper, because torch
# emitting MORE proposals than this would be silently truncated by the graph.
FIXED_PROPOSALS = 200


def make_wrapper(model, fixed_proposals: int = FIXED_PROPOSALS):
    """Adapt Faster R-CNN's ``list[dict]`` output to three plain tensors.

    Two graph-safety rules are enforced here, and both exist because the
    TorchScript tracer bakes **data-dependent Python integers** as constants:

    1. **The proposal count must be constant.**
       ``roi_heads.postprocess_detections`` does
       ``boxes_per_image = [b.shape[0] for b in proposals]`` and
       ``BoxCoder.decode`` does ``rel_codes.reshape(box_sum, -1)``. Trace once
       with 247 proposals and the graph is hard-wired to 247 forever — run it
       on a frame that yields 250 and the Reshape fails outright. The RPN
       emits anywhere between 0 and ``post_nms_top_n_test`` proposals, so
       ``_fixed_filter`` re-pads to exactly ``fixed_proposals`` using tensor
       ops only (no ``if``, no ``len()``), which keeps the traced constant
       correct for *every* input.

    2. **The output count must be left dynamic.**
       Padding detections to a fixed ``max_det`` would bake another
       data-dependent int (``n = out["boxes"].shape[0]``), so the outputs keep
       their natural, graph-native size and the caller slices what it needs.

    The score threshold is also lifted to 0.0 *in the graph* so the API can
    vary ``score_thresh`` per request without re-exporting.

    **Guard:** padding only *adds* rows if torch's own filter returns at most
    ``fixed_proposals``. If the RPN caps are not actually in effect (they are
    easy to lose — see ``build_model.apply_rpn_caps``), torch emits 208-355
    proposals while the graph keeps 200 and the two backends silently
    disagree on every low-score detection. That mismatch is checked here and
    raises instead of diverging.
    """
    import torch
    from torch import nn

    caps = getattr(model.rpn, "_post_nms_top_n", None)
    effective_cap = caps.get("testing") if isinstance(caps, dict) else None
    if effective_cap is None:
        raise RuntimeError(
            "cannot read rpn._post_nms_top_n['testing'] - cannot prove the "
            "proposal cap matches the ONNX graph. Apply build_model.apply_rpn_caps()."
        )
    if effective_cap > fixed_proposals:
        raise RuntimeError(
            f"RPN post-NMS cap ({effective_cap}) exceeds the ONNX proposal "
            f"canvas ({fixed_proposals}): torch would emit more proposals than "
            "the graph and the torch/onnx backends would disagree. Apply "
            "build_model.apply_rpn_caps() so both run at the same budget."
        )

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.fixed_proposals = fixed_proposals
            # keep the original (filtering) implementation to call first
            self._filter_proposals = model.rpn.filter_proposals
            # instance attribute: RegionProposalNetwork.forward calls
            # self.filter_proposals(...), which resolves here first.
            model.rpn.filter_proposals = self._pad_proposals
            # Filtering belongs to the serving layer, not the graph.
            self.model.roi_heads.score_thresh = 0.0

        def _pad_proposals(self, proposals, objectness, image_shapes,
                           num_anchors_per_level):
            """Run the real RPN filter, then pad/truncate to a fixed count.

            ``orig`` caps the result at ``post_nms_top_n()`` — which is only
            true when ``apply_rpn_caps`` actually wrote that cap into the
            ``_post_nms_top_n`` dict (a plain ``rpn.post_nms_top_n_test = 200``
            assignment is dead config torchvision never reads). The guard at
            the top of :func:`make_wrapper` proves it before tracing, so here
            the result is at most ``fixed_proposals`` and this step only ever
            *adds* rows: duplicates of real proposals (harmless — the box
            head's NMS removes them) plus a sentinel row when the image
            produced very few proposals. A sentinel at (0,0,0,0) decodes
            back to a zero-area box and is discarded by the box head's
            ``remove_small_boxes``, so it can never surface as a detection.

            ``n`` is computed with ``ones_like(...).sum()`` so it stays a
            *tensor*: converting it to a Python int is exactly what would
            bake a stale constant into the graph.
            """
            boxes, scores = self._filter_proposals(
                proposals, objectness, image_shapes, num_anchors_per_level
            )
            k = self.fixed_proposals
            out_boxes, out_scores = [], []

            for b, s in zip(boxes, scores):        # batch size is 1 -> baked, fine
                zero_b = torch.zeros(1, 4, dtype=b.dtype, device=b.device)
                zero_s = torch.zeros(1, dtype=s.dtype, device=s.device)
                # probe carries a spare row so the sentinel is always present
                probe_b = torch.cat([b, zero_b], 0)
                probe_s = torch.cat([s, zero_s], 0)
                # sentinel = first real proposal, or the zero row if there were none
                sentinel_b = probe_b[:1]
                sentinel_s = probe_s[:1]
                base_b = torch.cat([b, sentinel_b], 0)
                base_s = torch.cat([s, sentinel_s], 0)
                # int64: index_select rejects a float index, and `ones_like`
                # must override the dtype so the sum stays integral.
                n = torch.ones_like(base_s, dtype=torch.int64).sum()
                idx = torch.arange(k, dtype=torch.int64, device=b.device) % n
                out_boxes.append(base_b.index_select(0, idx))
                out_scores.append(base_s.index_select(0, idx))

            return out_boxes, out_scores

        def forward(self, images):
            # images: [1, 3, H, W] float32 in 0..1
            out = self.model(list(images))[0]
            return out["boxes"], out["scores"], out["labels"]

    return _Wrapper()


def export(checkpoint: Path, out_path: Path, dynamic: bool = False,
           height: int = DEFAULT_H, width: int = DEFAULT_W) -> dict:
    import torch

    from src.models.checkpoint import load_checkpoint

    model, meta = load_checkpoint(checkpoint, device="cpu")
    wrapper = make_wrapper(model).eval()

    # Trace on a REAL frame, not torch.rand. The tracer freezes Python ints
    # it observes, and on noise the RPN legitimately emits zero proposals —
    # which would bake "0" into every downstream split/reshape. Same reason
    # _verification_frame exists; trace and verify must agree.
    dummy_np, trace_source = _verification_frame(height, width)
    dummy = torch.from_numpy(dummy_np)

    # The detection count is data-dependent by nature: always declare it
    # dynamic. H/W are only offered via --dynamic because the FPN/RoIAlign
    # path often cannot be traced with changing shapes (they are padded to
    # a multiple of 32, so only multiples of 32 would ever be safe).
    dynamic_axes = {"boxes": {0: "num_dets"},
                    "scores": {0: "num_dets"},
                    "labels": {0: "num_dets"}}
    if dynamic:
        dynamic_axes["images"] = {2: "height", 3: "width"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes=dynamic_axes,
        opset_version=17,          # needed for ScatterND / RoIAlign paths
        do_constant_folding=True,
        # Torch 2.14 defaults to dynamo=True, and the dynamo exporter dies on
        # a data-dependent guard in torchvision's batched_nms:
        #     if boxes.numel() > (4000 if cpu else 100_000) and not tracing
        # Under the TorchScript tracer torch.jit.is_tracing() is True, so that
        # branch short-circuits and export succeeds. Legacy exporter is the
        # correct choice for torchvision detection models.
        dynamo=False,
    )
    elapsed = time.perf_counter() - t0

    # ONNX files carry no class names — write them alongside, and also
    # embed as a metadata prop so a single self-describing artifact ships.
    import onnx

    graph = onnx.load(str(out_path))
    onnx.helper.set_model_props(graph, {
        "classes": ",".join(meta["classes"]),
        "bbox_format": "xyxy",
        "source_checkpoint": checkpoint.name,
        "input_layout": "NCHW float32 0..1",
        "padded_to": "32",
    })
    onnx.checker.check_model(graph)
    onnx.save(graph, str(out_path))

    return {"path": str(out_path), "classes": meta["classes"],
            "shape": [1, 3, height, width], "fixed_proposals": FIXED_PROPOSALS,
            "traced_on": trace_source, "dynamic": dynamic,
            "export_seconds": round(elapsed, 2),
            "size_mb": round(out_path.stat().st_size / 1e6, 1)}


def _verification_frame(height: int, width: int) -> tuple[np.ndarray, str]:
    """Deterministic frame for the parity check.

    A REAL image is required, not random noise. The RPN applies
    ``score_thresh = 0.05`` to objectness, and a correctly trained proposal
    network looks at noise and concludes "no objects here" — it then emits
    zero proposals and the comparison degenerates into ``0 == 0``, which
    proves nothing. Random noise therefore reports ``vacuous`` rather than a
    pass. Falling back to noise (flagged as such) keeps this usable on a
    machine where the dataset has not been materialised yet.
    """
    from PIL import Image

    candidates = sorted((PROJECT_ROOT / "data" / "val").glob("*.png"))
    if candidates:
        with Image.open(candidates[0]) as im:
            im = im.convert("RGB").resize((width, height), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None], f"real image {candidates[0].name}"

    rng = np.random.default_rng(42)
    noise = rng.random((1, 3, height, width), dtype=np.float32)
    return noise, "RANDOM NOISE (no dataset found - parity result is weak)"


def verify(checkpoint: Path, onnx_path: Path,
           height: int = DEFAULT_H, width: int = DEFAULT_W) -> dict:
    """Compare ONNX Runtime against PyTorch on the same frame."""
    import onnxruntime as ort
    import torch

    from src.models.checkpoint import load_checkpoint

    model, _ = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    wrapper = make_wrapper(model).eval()

    arr, input_source = _verification_frame(height, width)
    x = torch.from_numpy(arr)

    with torch.no_grad():
        tb, ts, tl = wrapper(x)
        tb, ts, tl = tb.numpy(), ts.numpy(), tl.numpy()

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    ob, os_, ol = sess.run(None, {"images": arr})

    # Strict comparison: the graph no longer pads, so every row on both
    # sides is a real detection and nothing may be filtered out.
    t_b, t_s, t_l = tb, ts, tl
    o_b, o_s, o_l = ob, os_, ol

    report = {
        "input": input_source,
        "torch_detections": len(t_s),
        "onnx_detections": len(o_s),
        "count_match": len(t_s) == len(o_s),
        "score_max_abs_diff": None,
        "box_max_abs_diff": None,
        "label_match": True,
    }

    k = min(len(t_s), len(o_s))
    if k:
        report["score_max_abs_diff"] = float(np.abs(t_s[:k] - o_s[:k]).max())
        report["box_max_abs_diff"] = float(np.abs(t_b[:k] - o_b[:k]).max())
        report["label_match"] = bool(np.array_equal(t_l[:k], o_l[:k]))

    # tolerance: fp32 kernel-fusion differences, not algorithmic drift.
    # Zero detections on both sides is vacuous rather than wrong, so it must
    # not be reported as a pass — treat it as untested.
    vacuous = len(t_s) == 0 and len(o_s) == 0
    report["vacuous"] = bool(vacuous)
    ok = (not vacuous) and report["count_match"] and report["label_match"] \
        and (report["score_max_abs_diff"] or 0) < 1e-3 \
        and (report["box_max_abs_diff"] or 0) < 1.0
    report["passed"] = bool(ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export detector to ONNX")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dynamic", action="store_true",
                        help="attempt dynamic H/W (fails on some FPN paths)")
    parser.add_argument("--verify", action="store_true",
                        help="compare ONNX Runtime against PyTorch")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ckpt = (Path(args.checkpoint) if args.checkpoint
            else PROJECT_ROOT / cfg["paths"]["model_dir"] / "model.pt")
    if not ckpt.exists():
        print(f"[ONNX] ERROR — checkpoint not found: {ckpt}\n"
              "Train first:  python -m src.experiments.run_experiment")
        return 2

    out = (Path(args.out) if args.out
           else PROJECT_ROOT / cfg["paths"]["model_dir"] / "model.onnx")

    try:
        info = export(ckpt, out, dynamic=args.dynamic)
    except Exception as exc:                          # noqa: BLE001
        print(f"[ONNX] EXPORT FAILED ({type(exc).__name__}): {exc}")
        if args.dynamic:
            print("       Retry without --dynamic: FPN + RoIAlign often "
                  "cannot be traced with changing shapes.")
        return 1

    print(f"[ONNX] exported -> {info['path']}")
    print(f"       size {info['size_mb']} MB | input {info['shape']} | "
          f"proposals {info['fixed_proposals']} | "
          f"dynamic {info['dynamic']} | {info['export_seconds']}s")
    print(f"       traced on: {info['traced_on']}")
    print(f"       classes {info['classes']}")

    if args.verify:
        rep = verify(ckpt, out)
        print("\n[ONNX] parity check vs PyTorch:")
        for k, v in rep.items():
            print(f"       {k:24s} {v}")
        if not rep["passed"]:
            print("\n[ONNX] FAILED parity — do not ship this graph.")
            return 1
        print("\n[ONNX] parity OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
