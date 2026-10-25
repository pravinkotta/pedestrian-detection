"""FastAPI application — the serving layer (Step 5).

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    # then open http://127.0.0.1:8000/docs  (auto-generated OpenAPI)

Endpoints
---------
GET  /health   liveness probe — AWS ECS / ALB target groups poll this
GET  /model    which model, which classes, which device (traceability)
POST /detect   image in -> bounding boxes out

Design notes
------------
* The model is loaded **once** in the lifespan handler, not per request.
  Loading a 76 MB checkpoint on every call would blow the latency budget.
* ``/health`` returns 200 even if the model failed to load, but reports
  ``model_loaded: false``. A container that can't load its model should be
  replaced rather than left to fail every request — but the *reason* must
  stay visible, so ``/model`` returns ``status: "error"`` with the message.
* Score threshold is a request parameter because the operating point is a
  deployment decision, not a training decision: the notebook chose
  score >= 0.71 for the safety operating point, and a tuner may want to
  change it without retraining.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from api.schemas import (
    DetectResponse,
    Detection,
    HealthResponse,
    ImageInfo,
    ModelInfo,
)
from api.service import DEFAULT_THRESHOLD, ModelNotLoadedError, ModelService

TITLE = "Pedestrian Detection API"
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, release it at shutdown."""
    device = os.getenv("INFERENCE_DEVICE", "cpu")
    app.state.service = ModelService.from_env(device=device)
    svc = app.state.service
    if svc.loaded:
        print(f"[api] model loaded: {svc.source} | classes={svc.classes} "
              f"| device={svc.device}")
    else:
        # Do NOT crash the container: /health must stay answerable so the
        # failure is observable, and the orchestrator can act on it.
        print(f"[api] MODEL FAILED TO LOAD: {svc.error}")
    yield
    app.state.service = None


app = FastAPI(
    title=TITLE,
    version=VERSION,
    description=(
        "Serves a Faster R-CNN pedestrian / cyclist / vehicle detector.\n\n"
        "Deployed on AWS ECS Fargate behind an ALB; `/health` is the "
        "target-group health check."
    ),
    lifespan=lifespan,
)


def _service() -> ModelService:
    svc = getattr(app.state, "service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="service is starting up")
    return svc


@app.get("/", tags=["meta"], summary="Service metadata")
def root() -> dict:
    return {"service": TITLE, "version": VERSION, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["ops"],
         summary="Liveness probe for ECS / ALB")
def health() -> HealthResponse:
    svc = _service()
    return HealthResponse(
        status="ok" if svc.loaded else "degraded",
        model_loaded=svc.loaded,
    )


@app.get("/model", response_model=ModelInfo, tags=["ops"],
         summary="Which model is serving, and with which classes")
def model_info() -> ModelInfo:
    return ModelInfo(**_service().info())


@app.post("/detect", response_model=DetectResponse, tags=["inference"],
          summary="Detect pedestrians, cyclists and vehicles in an image")
async def detect(
    file: UploadFile = File(..., description="PNG or JPEG image"),
    score_thresh: float = Query(
        DEFAULT_THRESHOLD, ge=0.0, le=1.0,
        description="Minimum confidence. Raise to reduce false positives; "
                    "the notebook's safety operating point is 0.71.",
    ),
) -> DetectResponse:
    svc = _service()

    if not svc.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"model not loaded: {svc.error}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = ModelService.read_image(data)
    except Exception as exc:                         # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"could not decode image: {type(exc).__name__}",
        ) from exc

    try:
        detections, elapsed_ms = svc.detect(image, score_thresh)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:                      # tensor shape errors etc.
        raise HTTPException(
            status_code=500, detail=f"inference failed: {exc}"
        ) from exc

    return DetectResponse(
        detections=[Detection(**d) for d in detections],
        image=ImageInfo(width=image.width, height=image.height),
        threshold=score_thresh,
        inference_ms=elapsed_ms,
        model_source=svc.source,
        model_version=svc.model_version,
        classes=svc.classes,
    )
