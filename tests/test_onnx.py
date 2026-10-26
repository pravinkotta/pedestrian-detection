"""ONNX export and backend parity (Step 6).

These tests exist because "the graph exported" is not the same claim as
"the graph is right". TorchScript tracing freezes Python ints it observes
at export time, so a graph can export cleanly and then crash — or silently
disagree — on a different frame. Every test here compares the two runtimes
rather than trusting either one.

Skipped (not failed) when the checkpoint, dataset or onnxruntime is absent,
matching the convention in the other test modules: CI clones a fresh VM and
runs the data pipeline first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "model.pt"
VAL_DIR = ROOT / "data" / "val"

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("onnx")


def _require_checkpoint():
    if not CHECKPOINT.exists():
        pytest.skip("no checkpoint yet - run: python -m src.experiments.run_experiment")
    if not VAL_DIR.exists():
        pytest.skip("data pipeline not run yet")


def _sample_frame():
    """A REAL frame. Noise makes the RPN return 0 proposals and every
    comparison degenerates into 0 == 0, which proves nothing."""
    from PIL import Image

    files = sorted(VAL_DIR.glob("*.png"))
    if not files:
        pytest.skip("data pipeline not run yet")
    with Image.open(files[0]) as im:
        im = im.convert("RGB").resize((256, 192), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], files[0].name


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def exported_graph(tmp_path_factory) -> Path:
    """Export once per module so the graph is provably current."""
    _require_checkpoint()
    pytest.importorskip("onnxruntime")

    from src.models.export_onnx import export

    out = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export(CHECKPOINT, out)
    return out


def test_export_produces_a_valid_self_describing_graph(exported_graph):
    import onnx

    assert exported_graph.exists()
    graph = onnx.load(str(exported_graph))
    onnx.checker.check_model(graph)

    names = [i.name for i in graph.graph.input]
    assert names == ["images"], "the serving contract is a single input"
    shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    assert shape == [1, 3, 192, 256]

    assert [o.name for o in graph.graph.output] == ["boxes", "scores", "labels"]
    # class names are not in the weights - they travel as metadata so the
    # API never has to guess label order
    props = {p.key: p.value for p in graph.metadata_props}
    assert props["classes"] == "pedestrian,cyclist,vehicle"


def test_export_declares_the_detection_count_dynamic(exported_graph):
    """n_dets is data-dependent; freezing it would be a lie in the graph
    contract that bites the moment two frames differ."""
    import onnx

    graph = onnx.load(str(exported_graph))
    out_dims = {
        o.name: [d.dim_param or d.dim_value
                 for d in o.type.tensor_type.shape.dim]
        for o in graph.graph.output
    }
    assert out_dims["boxes"][0] == "num_dets"
    assert out_dims["scores"][0] == "num_dets"
    assert out_dims["labels"][0] == "num_dets"


# --------------------------------------------------------------------------
# parity: this is the test that would have caught the traced-int bug
# --------------------------------------------------------------------------
def test_onnx_matches_pytorch_on_a_real_frame(exported_graph):
    from src.models.export_onnx import verify

    report = verify(CHECKPOINT, exported_graph)

    assert report["vacuous"] is False, (
        "both sides returned 0 detections - the check proves nothing; "
        f"input was {report['input']}"
    )
    assert report["passed"] is True, report
    assert report["torch_detections"] > 0
    assert report["count_match"] is True
    assert report["label_match"] is True
    assert report["score_max_abs_diff"] < 1e-3
    assert report["box_max_abs_diff"] < 1.0


def test_onnx_graph_survives_a_different_frame(exported_graph):
    """The traced-int bug only shows up on frames whose proposal count
    differs from the export-time frame."""
    import onnxruntime as ort

    from src.models.checkpoint import load_checkpoint
    from src.models.export_onnx import make_wrapper

    files = sorted(VAL_DIR.glob("*.png"))
    if len(files) < 3:
        pytest.skip("need at least 3 validation frames")

    model, _ = load_checkpoint(CHECKPOINT, device="cpu")
    model.eval()
    wrapper = make_wrapper(model).eval()
    sess = ort.InferenceSession(str(exported_graph),
                                providers=["CPUExecutionProvider"])

    counts = []
    for p in files[:3]:
        from PIL import Image

        with Image.open(p) as im:
            im = im.convert("RGB").resize((256, 192), Image.BILINEAR)
            arr = (np.asarray(im, dtype=np.float32) / 255.0
                   ).transpose(2, 0, 1)[None]

        with torch.no_grad():
            _tb, ts, _ = wrapper(torch.from_numpy(arr))
        _ob, os_, _ = sess.run(None, {"images": arr})

        # a different proposal count per frame must not break the graph
        assert len(ts) == len(os_), f"{p.name}: torch {len(ts)} vs onnx {len(os_)}"
        counts.append(len(ts))

    assert min(counts) > 0, "no detections on any frame"


# --------------------------------------------------------------------------
# serving: the API must be able to swap runtime without changing answers
# --------------------------------------------------------------------------
def test_service_backends_agree(monkeypatch, exported_graph):
    from api.service import ModelService

    monkeypatch.setenv("SERVE_BACKEND", "torch")
    monkeypatch.setenv("ONNX_MODEL_PATH", str(exported_graph))
    monkeypatch.delenv("MODEL_URI", raising=False)
    torch_svc = ModelService.from_env(device="cpu")
    assert torch_svc.loaded, torch_svc.error
    assert torch_svc.backend == "torch"

    monkeypatch.setenv("SERVE_BACKEND", "onnx")
    onnx_svc = ModelService.from_env(device="cpu")
    assert onnx_svc.loaded, onnx_svc.error
    assert onnx_svc.backend == "onnx"

    from PIL import Image

    files = sorted(VAL_DIR.glob("*.png"))
    with Image.open(files[0]) as im:
        im = im.convert("RGB")

    t_det, _ = torch_svc.detect(im, 0.01)
    o_det, _ = onnx_svc.detect(im, 0.01)

    assert len(t_det) == len(o_det) > 0
    assert [d["label"] for d in t_det] == [d["label"] for d in o_det]
    for t, o in zip(t_det, o_det):
        assert abs(t["score"] - o["score"]) < 1e-3
        assert max(abs(a - b) for a, b in zip(t["bbox"], o["bbox"])) < 0.1


def test_service_onnx_backend_reports_a_missing_graph(monkeypatch, tmp_path):
    """A slim image with no graph must surface a clear startup error, not
    an exception that kills /health."""
    from api.service import ModelService

    monkeypatch.setenv("SERVE_BACKEND", "onnx")
    monkeypatch.setenv("ONNX_MODEL_PATH", str(tmp_path / "nope.onnx"))

    svc = ModelService.from_env(device="cpu")
    assert svc.loaded is False
    assert svc.error and "no graph" in svc.error
    assert svc.info()["status"] == "error"


def test_service_rejects_an_unknown_backend(monkeypatch):
    from api.service import ModelService

    monkeypatch.setenv("SERVE_BACKEND", "cuda-magic")
    svc = ModelService.from_env(device="cpu")
    assert svc.loaded is False
    assert svc.error and "SERVE_BACKEND" in svc.error


def test_registry_uri_cannot_be_served_by_onnx(monkeypatch):
    """The MLflow pytorch flavor has no ONNX artifact - fail loudly rather
    than silently serving something else."""
    from api.service import ModelService

    monkeypatch.setenv("MODEL_URI", "models:/PedestrianDetector@champion")
    monkeypatch.setenv("SERVE_BACKEND", "onnx")
    svc = ModelService.from_env(device="cpu")
    assert svc.loaded is False
    assert "MODEL_URI" in (svc.error or "")


def test_backend_is_visible_in_model_info(monkeypatch, exported_graph):
    from api.service import ModelService

    monkeypatch.setenv("SERVE_BACKEND", "onnx")
    monkeypatch.setenv("ONNX_MODEL_PATH", str(exported_graph))
    monkeypatch.delenv("MODEL_URI", raising=False)
    svc = ModelService.from_env(device="cpu")
    if not svc.loaded:
        pytest.skip(svc.error or "onnx graph not built yet")

    info = svc.info()
    assert info["backend"] == "onnx"
    assert info["status"] == "loaded"
    assert info["classes"] == ["pedestrian", "cyclist", "vehicle"]
