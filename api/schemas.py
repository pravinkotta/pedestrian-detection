"""Pydantic schemas — the API's public contract.

Everything the service returns is declared here, so the OpenAPI schema at
``/docs`` is generated rather than hand-maintained. Change a field and the
docs, validation and client typings all move together.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    """One detected object, in ORIGINAL image pixel coordinates."""

    bbox: list[float] = Field(
        description="[x1, y1, x2, y2] top-left and bottom-right corners, "
                    "in pixels of the uploaded image (not resized/normalised)."
    )
    label: str = Field(description="Class name, e.g. 'pedestrian'.")
    score: float = Field(ge=0.0, le=1.0,
                         description="Confidence in [0, 1].")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "bbox": [49.0, 116.0, 65.0, 161.0],
                "label": "pedestrian",
                "score": 0.93,
            }]
        }
    )


class ImageInfo(BaseModel):
    width: int
    height: int


class DetectResponse(BaseModel):
    detections: list[Detection]
    image: ImageInfo
    threshold: float = Field(description="score threshold actually applied.")
    inference_ms: float
    model_source: str = Field(description="Checkpoint path or MLflow model URI.")
    model_version: str | None = None
    classes: list[str]


class ModelInfo(BaseModel):
    status: Literal["loaded", "error"]
    model_source: str
    model_version: str | None
    classes: list[str]
    architecture: str | None = None
    git_sha: str = Field(
        default="unknown",
        description="Commit the serving image was built from (CI passes "
                    "--build-arg GIT_SHA). 'unknown' outside a container "
                    "build — and the field that proves a deploy landed.",
    )
    backend: Literal["torch", "onnx", "none"] = Field(
        default="none",
        description="Which runtime is executing the model: PyTorch or "
                    "ONNX Runtime. Both serve the same graph.",
    )
    device: str
    default_threshold: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
