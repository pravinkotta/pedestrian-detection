"""Model construction — Faster R-CNN (MobileNetV3-Large + FPN).

Promoted from the notebook (cell 25) unchanged, including the two RPN caps
that are easy to lose when refactoring:

    model.rpn.pre_nms_top_n_test  = 600
    model.rpn.post_nms_top_n_test = 200

Torchvision defaults are 1000 / 1000. Every proposal flows through the box
head, which dominates CPU latency. Dropping to 200 post-NMS proposals is what
keeps inference inside a plausible edge budget while preserving recall on
dense scenes (notebook: 734 ms -> and it is the documented reason the
MobileNetV3 backbone was chosen at all).

**Those two assignments are dead code.** ``RegionProposalNetwork`` exposes no
``pre_nms_top_n_test`` / ``post_nms_top_n_test`` *property* — the statement
above simply creates two plain instance attributes that nothing reads, while
``post_nms_top_n()`` keeps returning the constructor's 1000/1000. The caps
never took effect, and the mismatch eventually showed up as a real failure:
the ONNX graph bakes ``FIXED_PROPOSALS = 200`` (traced shapes must be static)
while torch ran 208-355 proposals, so the two backends only agreed while the
dropped proposals happened not to matter. ``apply_rpn_caps`` writes into the
``_pre_nms_top_n`` / ``_post_nms_top_n`` dicts that
``post_nms_top_n()`` actually consults, making one proposal budget true for
torch, for evaluation and for the graph.
"""

from __future__ import annotations

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# RPN proposal caps for test-time inference (edge-deployment optimisation)
PRE_NMS_TOP_N_TEST = 600
POST_NMS_TOP_N_TEST = 200


def apply_rpn_caps(model: torch.nn.Module,
                   pre: int = PRE_NMS_TOP_N_TEST,
                   post: int = POST_NMS_TOP_N_TEST) -> torch.nn.Module:
    """Make the proposal caps real, not decorative.

    torchvision reads ``rpn._pre_nms_top_n["testing"]`` and
    ``rpn._post_nms_top_n["testing"]`` (via the ``pre_nms_top_n()`` /
    ``post_nms_top_n()`` accessors). Writing ``rpn.post_nms_top_n_test = 200``
    instead only shadows the name on the instance, which is why the notebook's
    caps were silently inactive — see the module docstring.

    Training caps are left alone: proposal sampling there is free (the box
    head does not run) and more candidates only helps.
    """
    rpn = model.rpn
    rpn.pre_nms_top_n_test = pre          # kept so describe()/the notebook's
    rpn.post_nms_top_n_test = post        # declared values stay visible
    for private, value in (("_pre_nms_top_n", pre),
                           ("_post_nms_top_n", post)):
        caps = getattr(rpn, private, None)
        if isinstance(caps, dict) and "testing" in caps:
            caps["testing"] = int(value)
        else:  # torchvision changed its internals: fail loudly, not silently
            raise RuntimeError(
                f"torchvision's RegionProposalNetwork no longer exposes {private}"
                " - the proposal caps in build_model() are no longer applied "
                "and the ONNX graph would diverge from torch."
            )
    return model


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, pretrained: bool = True,
                device: str | torch.device = "auto",
                cfg: dict | None = None) -> torch.nn.Module:
    """Build Faster R-CNN with a fresh ``num_classes`` detection head.

    ``pretrained=True`` keeps the COCO-pretrained *backbone and FPN* (transfer
    learning — the part that actually matters) and re-initialises only the
    classification + box-regression heads. The RPN head is also COCO-trained
    and is kept, which is why the model converges in 3 epochs.

    ``cfg`` is the loaded ``configs/config.yaml``. It is optional, but when
    present the model honours the values declared there — they were written
    down for a reason:

    * ``train.image_min_size / image_max_size`` — the transform's resize.
      Left at torchvision's COCO defaults a 256x192 frame is *upscaled* to
      800x1088 inside the graph: 870k pixels, 484 ms/frame. Honouring the
      config's 480/800 gives 480x640 (307k px, 184 ms) with **identical**
      detections. Training and serving must agree here or the model sees a
      different image distribution than it was fitted on.
    * ``model.box_score_thresh / box_nms_thresh / box_detections_per_img``
      — the box head's operating point. These only apply to the torch
      backend; the ONNX wrapper deliberately zeroes ``score_thresh`` so the
      API can vary it per request.
    """
    if pretrained:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # the caps must be APPLIED, not merely assigned (see apply_rpn_caps)
    apply_rpn_caps(model)

    if cfg:
        train = cfg.get("train", {})
        head = cfg.get("model", {})
        if train.get("image_min_size"):
            model.transform.min_size = (int(train["image_min_size"]),)
        if train.get("image_max_size"):
            model.transform.max_size = int(train["image_max_size"])
        if head.get("box_score_thresh") is not None:
            model.roi_heads.score_thresh = float(head["box_score_thresh"])
        if head.get("box_nms_thresh") is not None:
            model.roi_heads.nms_thresh = float(head["box_nms_thresh"])
        if head.get("box_detections_per_img") is not None:
            model.roi_heads.detections_per_img = int(
                head["box_detections_per_img"]
            )

    dev = resolve_device(str(device))
    return model.to(dev)


def describe(model: torch.nn.Module) -> dict:
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "architecture": type(model).__name__,
        "params_total": n_params,
        "params_millions": round(n_params / 1e6, 1),
        "params_trainable": trainable,
        "device": str(next(model.parameters()).device),
        "rpn_pre_nms_top_n_test": model.rpn.pre_nms_top_n(),
        "rpn_post_nms_top_n_test": model.rpn.post_nms_top_n(),
        "image_min_size": list(model.transform.min_size),
        "image_max_size": model.transform.max_size,
        "box_detections_per_img": model.roi_heads.detections_per_img,
    }
