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

Two supported runtimes, selected by ``SERVE_BACKEND``:

    torch   load ``MODEL_PATH``/``MODEL_URI`` and run PyTorch  (default)
    onnx    load ``ONNX_MODEL_PATH`` and run ONNX Runtime
    auto    prefer ``torch``, fall back to ONNX if torch is unavailable

ONNX is the deployment path: the graph is 76 MB, needs no PyTorch in the
image at all (the slim build passes ``--build-arg INSTALL_TORCH=0``), and
matches PyTorch to ~1e-6 (see ``src/models/export_onnx.py --verify``).
``MLFLOW``-sourced models are always torch — the MLflow pytorch flavor has
no ONNX counterpart — so ONNX serving reads a checkpoint file instead.

Preprocessing is deliberately **backend-independent**: every frame is
resized to the model's declared input (256x192, taken from the ONNX graph
when serving ONNX) and detections are scaled back into the uploaded image's
coordinate space. Without that, swapping the runtime would also swap the
geometry and the A/B comparison would be meaningless.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD = 0.30
# Training frames are 256x192 (see configs/config.yaml / the dataset), and
# the exported graph has a static input of exactly this size.
DEFAULT_INPUT_WIDTH = 256
DEFAULT_INPUT_HEIGHT = 192


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


def _resolve_backend(source_kind: str) -> str:
    """Pick the runtime. Never silently degrades an explicit request."""
    want = os.getenv("SERVE_BACKEND", "auto").strip().lower() or "auto"
    if want not in ("auto", "torch", "onnx"):
        raise ValueError(
            f"SERVE_BACKEND={want!r} is not one of auto|torch|onnx"
        )
    if source_kind == "mlflow":
        if want == "onnx":
            raise ValueError(
                "SERVE_BACKEND=onnx cannot serve MODEL_URI: the MLflow "
                "pytorch flavor has no ONNX artifact. Export one "
                "(python -m src.models.export_onnx) and point "
                "ONNX_MODEL_PATH at it, or unset MODEL_URI."
            )
        return "torch"
    if want == "torch":
        return "torch"
    onnx_path = Path(
        os.getenv("ONNX_MODEL_PATH",
                  str(PROJECT_ROOT / "models" / "model.onnx"))
    )
    if want == "onnx":
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"SERVE_BACKEND=onnx but no graph at {onnx_path}\n"
                "Export one:  python -m src.models.export_onnx --verify"
            )
        return "onnx"
    # auto: torch unless torch cannot be imported (slim image)
    try:
        import torch  # noqa: F401
    except ImportError:
        if onnx_path.exists():
            return "onnx"
        raise FileNotFoundError(
            "no torch in this image and no ONNX graph at "
            f"{onnx_path} - build with --build-arg INSTALL_TORCH=1, or "
            "export a graph: python -m src.models.export_onnx --verify"
        ) from None
    return "torch"


class ModelService:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None                 # torch backend
        self.session = None               # onnx backend
        self.onnx_input_name: str | None = None
        self.classes: list[str] = []
        self.architecture: str | None = None
        self.backend = ""
        self.source_kind = ""
        self.source = ""
        self.model_version: str | None = None
        self.error: str | None = None
        self.input_width = DEFAULT_INPUT_WIDTH
        self.input_height = DEFAULT_INPUT_HEIGHT

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, device: str = "cpu") -> ModelService:
        svc = cls(device=device)
        svc.load()
        return svc

    def load(self) -> ModelService:
        kind, value = _resolve_source()
        self.source_kind, self.source = kind, value

        try:
            self.backend = _resolve_backend(kind)
            if self.backend == "onnx":
                self._load_onnx(Path(
                    os.getenv("ONNX_MODEL_PATH",
                              str(PROJECT_ROOT / "models" / "model.onnx"))
                ))
            elif kind == "mlflow":
                self._load_mlflow(value)
            else:
                self._load_torch(Path(value))
        except FileNotFoundError as exc:
            self.error = str(exc)
            self.backend = ""
        except Exception as exc:                      # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.backend = ""

        return self

    def _load_torch(self, path: Path) -> None:

        from src.models.checkpoint import load_checkpoint

        model, meta = load_checkpoint(str(path), device=self.device)
        self.model = model
        self.model.eval()
        self.classes = meta["classes"]
        self.model_version = path.stem
        self.architecture = type(model).__name__
        # The operating point belongs to the request (`score_thresh` query
        # param), and the ONNX graph bakes 0.0 — match it here so the two
        # runtimes return the same candidate set. Otherwise torch would
        # pre-filter at the config's 0.05 and the backends could disagree.
        self.model.roi_heads.score_thresh = 0.0
        if not self.classes:
            self.classes = self._classes_from_head()

    def _load_mlflow(self, uri: str) -> None:
        import mlflow.pytorch

        from src.registry.model_uri import describe_uri

        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        tracking = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking:
            from src.experiments.tracking import PROJECT_ROOT as ROOT
            mlflow.set_tracking_uri(
                f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
            )

        self.model = mlflow.pytorch.load_model(uri, device=self.device)
        self.model.eval()
        self.architecture = type(self.model).__name__
        self.model_version = describe_uri(uri)
        self.model.roi_heads.score_thresh = 0.0
        # the pytorch flavor does not carry our metadata dict back on load,
        # so class names are recovered from the head width
        self.classes = self._classes_from_head()

    def _load_onnx(self, path: Path) -> None:
        """Load the graph. No PyTorch on this path — that is the point."""
        import onnxruntime as ort

        if not path.exists():
            raise FileNotFoundError(
                f"ONNX graph not found: {path}\n"
                "Export one:  python -m src.models.export_onnx --verify"
            )

        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

        # The graph's declared input is the source of truth for geometry.
        shape = self.session.get_inputs()[0].shape      # [1, 3, H, W]
        self.onnx_input_name = self.session.get_inputs()[0].name
        self.input_height, self.input_width = int(shape[2]), int(shape[3])

        # class names are not in the graph unless the exporter put them in
        # the metadata — export_onnx.py writes them there.
        meta = self.session.get_modelmeta().custom_metadata_map
        if meta.get("classes"):
            self.classes = [c for c in meta["classes"].split(",") if c]
        if not self.classes:
            self.classes = ["pedestrian", "cyclist", "vehicle"]

        self.model_version = path.stem
        self.architecture = "FasterRCNN_MobileNet_V3_Large_FPN (ONNX)"
        # /model must name the file actually being executed, not the
        # checkpoint MODEL_PATH points at
        self.source = str(path)

    def _classes_from_head(self) -> list[str]:
        n = self.model.roi_heads.box_predictor.cls_score.out_features
        if n == 4:
            return ["pedestrian", "cyclist", "vehicle"]
        return [f"class_{i}" for i in range(1, n)]

    @property
    def loaded(self) -> bool:
        return self.model is not None or self.session is not None

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def detect(self, image: Image.Image, score_thresh: float) -> tuple[list, float]:
        """Run detection. Returns (detections, inference_ms)."""
        if not self.loaded:
            raise ModelNotLoadedError(self.error or "model not loaded")

        rgb = image.convert("RGB")
        orig_w, orig_h = rgb.size

        # One preprocessing path for both runtimes: resize to the model's
        # declared canvas, run, then map boxes back to the uploaded image.
        if rgb.size != (self.input_width, self.input_height):
            rgb = rgb.resize((self.input_width, self.input_height),
                             Image.BILINEAR)
        arr = (np.asarray(rgb, dtype=np.float32) / 255.0
                ).transpose(2, 0, 1)[None]              # -> [1, 3, H, W]

        t0 = time.perf_counter()
        if self.session is not None:
            # the wrapper exports single-image tensors: shapes are
            # (N,4), (N,) and (N,) — there is no batch dimension to drop
            boxes, scores, labels = self.session.run(
                None, {self.onnx_input_name: arr}
            )
        else:
            import torch

            x = torch.from_numpy(arr)
            with torch.no_grad():
                out = self.model([x[0]])[0]
            boxes = out["boxes"].numpy()
            scores = out["scores"].numpy()
            labels = out["labels"].numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # boxes live in the resized canvas -> back into uploaded coordinates
        sx = orig_w / self.input_width
        sy = orig_h / self.input_height

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            if score < score_thresh:
                continue
            idx = int(label) - 1           # 0 is background
            name = self.classes[idx] if 0 <= idx < len(self.classes) else f"unknown_{label}"
            detections.append({
                "bbox": [round(float(box[0]) * sx, 2),
                         round(float(box[1]) * sy, 2),
                         round(float(box[2]) * sx, 2),
                         round(float(box[3]) * sy, 2)],
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
            "backend": self.backend or "none",
            "device": self.device,
            "default_threshold": DEFAULT_THRESHOLD,
        }
