"""API tests (Step 5) — run against the real FastAPI app with a real image.

The lifespan handler loads an actual checkpoint, so these exercise the same
code path the deployed container will, not a mock.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "data" / "val"


def _sample_image_bytes() -> bytes:
    for p in sorted(VAL_DIR.glob("*.png")):
        return p.read_bytes()
    pytest.skip("no validation images present")


@pytest.fixture(scope="module")
def client():
    from api.main import app

    # `with` triggers the lifespan handler -> model actually loads
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# ops endpoints — ECS depends on these
# --------------------------------------------------------------------------
def test_health_is_reachable(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["model_loaded"], bool)


def test_health_stays_200_even_when_model_fails_to_load(client):
    """A container that cannot load must still answer /health, or the ALB
    can never distinguish 'starting up' from 'broken'.

    NOTE: we mutate ``app.state`` on the SHARED module-scoped client and
    restore it. Opening a nested TestClient here would re-run the lifespan
    handler and leave ``service = None`` for every subsequent test.
    """
    from api.main import app
    from api.service import ModelService

    original = app.state.service
    try:
        svc = ModelService()
        svc.error = "simulated failure"
        app.state.service = svc

        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "degraded", "model_loaded": False}

        # ...and the real reason is still visible one endpoint over
        res = client.get("/model")
        assert res.status_code == 200
        assert res.json()["status"] == "error"
    finally:
        app.state.service = original


def test_model_endpoint_reports_classes_and_device(client):
    body = client.get("/model").json()
    assert body["status"] == "loaded"
    assert body["classes"] == ["pedestrian", "cyclist", "vehicle"]
    assert body["device"] == "cpu"
    assert body["model_source"]
    assert body["architecture"]


def test_model_endpoint_reports_which_code_is_serving(client):
    """/model must name the build, or "did my deploy land?" is unanswerable.

    CI stamps GIT_SHA into the image; outside a container build the field
    stays at its documented default instead of breaking the response.
    """
    import os

    body = client.get("/model").json()
    assert body["git_sha"] == os.environ.get("GIT_SHA", "unknown")
    assert body["git_sha"]  # never empty


def test_root_lists_docs(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert "version" in body


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def test_detect_returns_valid_detections(client):
    res = client.post(
        "/detect",
        files={"file": ("frame.png", _sample_image_bytes(), "image/png")},
    )
    assert res.status_code == 200, res.text
    body = res.json()

    # image geometry must be reported as the UPLOADED size, not a resized one
    assert body["image"] == {"width": 256, "height": 192}
    assert body["classes"] == ["pedestrian", "cyclist", "vehicle"]
    assert body["inference_ms"] > 0
    assert body["threshold"] == pytest.approx(0.30)

    for det in body["detections"]:
        assert set(det) == {"bbox", "label", "score"}
        x1, y1, x2, y2 = det["bbox"]
        assert len(det["bbox"]) == 4
        assert x2 >= x1 and y2 >= y1, "bbox must be xyxy with positive extent"
        assert 0 <= x1 <= 256 and x2 <= 256 + 1, "bbox outside image width"
        assert 0 <= y1 <= 192 and y2 <= 192 + 1, "bbox outside image height"
        assert det["label"] in ("pedestrian", "cyclist", "vehicle")
        assert 0.0 <= det["score"] <= 1.0
        assert det["score"] >= body["threshold"], "below-threshold det leaked"


def test_detect_respects_score_threshold(client):
    payload = _sample_image_bytes()

    high = client.post("/detect?score_thresh=0.95",
                       files={"file": ("f.png", payload, "image/png")}).json()
    low = client.post("/detect?score_thresh=0.01",
                      files={"file": ("f.png", payload, "image/png")}).json()

    assert len(high["detections"]) <= len(low["detections"]), (
        "raising the threshold must never add detections"
    )
    assert high["threshold"] == 0.95
    for d in high["detections"]:
        assert d["score"] >= 0.95


def test_detect_rejects_out_of_range_threshold(client):
    res = client.post("/detect?score_thresh=1.5",
                      files={"file": ("f.png", _sample_image_bytes(), "image/png")})
    assert res.status_code == 422   # pydantic/FastAPI validation


def test_detect_rejects_non_image_payload(client):
    res = client.post("/detect",
                      files={"file": ("f.png", b"this is not an image", "image/png")})
    assert res.status_code == 400
    assert "could not decode" in res.json()["detail"]


def test_detect_rejects_empty_file(client):
    res = client.post("/detect", files={"file": ("f.png", b"", "image/png")})
    assert res.status_code == 400
    assert res.json()["detail"] == "empty file"


def test_detect_returns_503_when_model_unavailable(client):
    from api.main import app
    from api.service import ModelService

    original = app.state.service
    try:
        svc = ModelService()
        svc.error = "no checkpoint on disk"
        app.state.service = svc

        res = client.post("/detect",
                          files={"file": ("f.png", _sample_image_bytes(),
                                          "image/png")})
        assert res.status_code == 503
        assert "no checkpoint on disk" in res.json()["detail"]
    finally:
        app.state.service = original


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------
def test_openapi_schema_exposes_all_endpoints(client):
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"])
    assert {"/", "/health", "/model", "/detect", "/monitor/features"} <= paths


def test_upload_accepts_jpeg_too(client):
    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(_sample_image_bytes())).convert("RGB").save(buf, "JPEG")
    res = client.post("/detect",
                      files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")})
    assert res.status_code == 200, res.text


# --------------------------------------------------------------------------
# Step 9 — input-feature capture for the drift monitor
# --------------------------------------------------------------------------
def test_detect_captures_features_when_capture_enabled(client, tmp_path,
                                                        monkeypatch):
    """One served frame -> one feature row (aggregates only, no pixels)."""
    from src.monitoring.features import read_rows

    log = tmp_path / "features.jsonl"
    monkeypatch.setenv("FEATURES_LOG", str(log))

    res = client.post("/detect",
                      files={"file": ("f.png", _sample_image_bytes(),
                                      "image/png")})
    assert res.status_code == 200, res.text

    rows = read_rows(log)
    assert len(rows) == 1
    row = rows[0]
    assert {"brightness", "contrast", "sharpness",
            "width", "height"} <= set(row)
    assert row["width"] == 256 and row["height"] == 192
    assert row["inference_ms"] > 0
    assert "ts" in row and "n_detections" in row

    # ...and the drift job's pull endpoint sees exactly that row
    body = client.get("/monitor/features").json()
    assert body["capture"] == "on"
    assert body["count"] == 1
    assert body["rows"][0]["brightness"] == pytest.approx(row["brightness"])


def test_monitor_endpoint_says_capture_is_off_by_default(client, monkeypatch):
    """'no drift' and 'not watching' must be distinguishable."""
    monkeypatch.delenv("FEATURES_LOG", raising=False)
    body = client.get("/monitor/features").json()
    assert body == {"capture": "off", "source": None, "count": 0, "rows": []}


def test_failed_capture_never_breaks_inference(client, tmp_path, monkeypatch):
    """FEATURES_LOG points somewhere unwritable -> /detect still serves."""
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("i am a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("FEATURES_LOG", str(blocker / "features.jsonl"))

    res = client.post("/detect",
                      files={"file": ("f.png", _sample_image_bytes(),
                                      "image/png")})
    assert res.status_code == 200, res.text
    assert res.json()["detections"] is not None
