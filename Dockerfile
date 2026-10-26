# syntax=docker/dockerfile:1
# =============================================================================
#  Pedestrian Detection API — production image
#
#  Build:  docker build -t pedestrian-api .
#  Run:    docker run -p 8000:8000 -e MODEL_PATH=/app/models/model.pt \
#                  -v $(pwd)/models:/app/models:ro pedestrian-api
#
#  Design:
#   * Two stages -> no compiler, no pip cache, no build junk in the runtime.
#   * CPU-only torch wheels (the default PyPI wheel drags in ~2.5 GB of CUDA;
#     the cpu index is ~190 MB). Overridable via --build-arg TORCH_INDEX.
#   * Non-root user: a container escape should not land in uid 0.
#   * Weights are NOT baked in (.dockerignore excludes them) — the container
#     pulls them at startup via MODEL_URI (MLflow registry) or MODEL_PATH.
#   * HEALTHCHECK is stricter than GET /health: it fails when the model did
#     not load, so a broken container gets replaced instead of serving 503s.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — build the virtualenv
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

# The +cpu local tag is what forces the small wheel: PyPI only ships CUDA
# builds of torch for linux, and `torch==2.14.1` (no tag) would resolve to
# ~2.5 GB of CUDA libraries. Pinning `==2.14.1+cpu` makes the CPU wheel the
# only possible match, while --extra-index-url lets deps (flit_core,
# typing_extensions, ...) resolve normally from PyPI.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PYPI_INDEX=https://pypi.org/simple
ARG TORCH_SPEC=torch==2.14.1+cpu
ARG TORCHVISION_SPEC=torchvision==0.29.1+cpu
# INSTALL_TORCH=0 builds the ONNX-only serving image: the runtime then
# carries onnxruntime instead of torch (~190 MB of wheels + CUDA-free deps).
#   docker build --build-arg INSTALL_TORCH=0 -t pedestrian-api:slim .
# Selecting the runtime at RUN time is a build arg, not an env var, so an
# image can never end up with a backend it does not have the libraries for.
ARG INSTALL_TORCH=1

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# build-essential only in this stage; the runtime never sees it
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch FIRST (when requested), so the runtime requirements file sees it as
# already satisfied and never tries to re-resolve it against PyPI.
# NOTE: requirements-runtime.txt (serving only), NOT requirements.txt
# (the full MLflow + Evidently dev/CI toolchain) — see that file's header.
COPY requirements-runtime.txt .
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install --no-cache-dir \
            --index-url ${TORCH_INDEX} --extra-index-url ${PYPI_INDEX} \
            ${TORCH_SPEC} ${TORCHVISION_SPEC}; \
    fi \
    && pip install --no-cache-dir -r requirements-runtime.txt \
    && pip check

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

ARG SERVE_BACKEND=auto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    INFERENCE_DEVICE=cpu \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    SERVE_BACKEND=${SERVE_BACKEND} \
    ONNX_MODEL_PATH=/app/models/model.onnx

# libgomp1: torch's OpenMP runtime (missing from slim -> cryptic .so errors)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# application code only — no data, no weights (see .dockerignore)
COPY src/    ./src/
COPY api/    ./api/
COPY configs/ ./configs/

USER app

EXPOSE 8000

# Start-up grace: loading a 76 MB checkpoint on a cold container takes a few
# seconds, and we must not have the orchestrator kill it during that window.
HEALTHCHECK --interval=30s --timeout=8s --start-period=90s --retries=3 \
    CMD python -c "import json,sys,urllib.request as u; \
d=json.load(u.urlopen('http://127.0.0.1:8000/health',timeout=6)); \
sys.exit(0 if d.get('model_loaded') else 1)"

# --workers 1: the model is loaded once per worker. More workers would load
# more copies of a 76 MB model into memory for no benefit at this scale.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
