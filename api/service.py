"""Model loading and inference for the serving layer.

Separated from ``main.py`` on purpose: the FastAPI file owns HTTP concerns
(status codes, request parsing), this one owns ML concerns (where the model
comes from, how to run it). That split lets Step 6's Docker image, Step 9's
drift monitor and any batch job reuse ``ModelService`` without an HTTP
server in the way.

Two supported sources, selected by environment variable:

    MODEL_URI   an MLflow model URI, e.g. models:/PedestrianDetector@champion
    MODEL_PATH  a local checkpoint,           e.g. models/model.pt

``MODEL_URI`` wins when both are set — production should serve whatever the
registry says is champion, while local development serves whatever was last
trained.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD = 0.30


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested before/without a usable model."""


def _resolve_source() -> tuple[str, str]:
    """Return (kind, value) where kind is 'mlflow' or 'checkpoint'."""
    uri = os.getenv("MODEL_URI", "").strip()
    if uri:
        return "mlflow", uri
    path = os.getenv(
        "MODEL_PATH", str(PROJECT_ROOT / "models" / "model.pt")
    ).strip()
    return "checkpoint", path


class ModelService:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.classes: list[str] = []
        self.architecture: str | None = None
        self.source_kind = ""
        self.source = ""
        self.model_version: str | None = None
        self.error: str | None = None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, device: str = "cpu") -> "ModelService":
        svc = cls(device=device)
        svc.load()
        return svc

    def load(self) -> "ModelService":
        from src.models.checkpoint import load_checkpoint

        kind, value = _resolve_source()
        self.source_kind, self.source = kind, value

        try:
            if kind == "mlflow":
                import mlflow.pytorch
                os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
                from src.registry.model_uri import describe_uri
                tracking = os.getenv("MLFLOW_TRACKING_URI")
                if not tracking:
                    from src.experiments.tracking import PROJECT_ROOT as ROOT
                    mlflow.set_tracking_uri(
                        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
                    )
                self.model = mlflow.pytorch.load_model(value, device=self.device)
                self.model_version = describe_uri(value)
            else:
                model, meta = load_checkpoint(value, device=self.device)
                self.model = model
                self.classes = meta["classes"]
                self.model_version = Path(value).stem

            self.model.eval()
            self.architecture = type(self.model).__name__

            # The MLflow pytorch flavor does not carry our custom metadata
            # dict back on load, so recover class names from the head width
            # when the checkpoint path did not supply them.
            if not self.classes:
                self.classes = self._classes_from_head()

        except FileNotFoundError as exc:
            self.error = str(exc)
        except Exception as exc:                      # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"

        return self

    def _classes_from_head(self) -> list[str]:
        n = self.model.roi_heads.box_predictor.cls_score.out_features
        if n == 4:
            return ["pedestrian", "cyclist", "vehicle"]
        return [f"class_{i}" for i in range(1, n)]

    @property
    def loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def detect(self, image: Image.Image, score_thresh: float) -> tuple[list, float]:
        """Run detection. Returns (detections, inference_ms)."""
        if self.model is None:
            raise ModelNotLoadedError(self.error or "model not loaded")

        from torchvision.transforms import ToTensor

        tensor = ToTensor()(image.convert("RGB")).to(self.device)

        t0 = time.perf_counter()
        out = self.model([tensor])[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        boxes = out["boxes"].cpu().tolist()
        labels = out["labels"].cpu().tolist()
        scores = out["scores"].cpu().tolist()

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            if score < score_thresh:
                continue
            idx = int(label) - 1           # 0 is background
            name = self.classes[idx] if 0 <= idx < len(self.classes) else f"unknown_{label}"
            detections.append({
                "bbox": [round(float(v), 2) for v in box],
                "label": name,
                "score": round(float(score), 4),
            })

        detections.sort(key=lambda d: -d["score"])
        return detections, round(elapsed_ms, 1)

    @staticmethod
    def read_image(data: bytes) -> Image.Image:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.convert("RGB")

    def info(self) -> dict:
        return {
            "status": "loaded" if self.loaded else "error",
            "model_source": self.source or "unresolved",
            "model_version": self.model_version,
            "classes": self.classes,
            "architecture": self.architecture,
            "device": self.device,
            "default_threshold": DEFAULT_THRESHOLD,
        }
