# Pedestrian Detection — MLOps Pipeline

Faster R-CNN (MobileNetV3-Large + FPN) detecting **pedestrians / cyclists / vehicles**,
promoted from notebook research code to a production ML pipeline deployed on **AWS ECS Fargate**.

**Headline metric:** *Recall @ FPR ≤ 1%* — a safety metric, not mAP. A missed pedestrian
costs far more than a false alarm, so recall is optimised subject to a low false-positive ceiling.

## Pipeline

```
Feature Engineering + Gold Layer  ->  MLflow Experiment Tracking  ->  MLflow Model Registry
                                                                        |
FastAPI Endpoint  ->  Docker Container  ->  GitHub Actions CI/CD  ->  AWS ECS (Fargate)
                                                                        |
Evidently Drift Monitor  ->  HITL Gate (high-risk)  ->  Retrain Trigger
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Stage 1 — Data layers (Bronze / Silver / Gold)

| Layer | Command | Output |
|-------|---------|--------|
| Bronze | `python -m src.data.make_bronze` | `data/_bronze_manifest.json` |
| Silver | `python -m src.data.convert_to_coco` | `data/interim/annotations/{train,val,test}.json` |
| Gold | `python -m src.data.make_gold` | `data/processed/annotations/*.json` + `dataset_stats.json` |
| Verify | `pytest -q` | 8 checks |

Verified baseline — **380 images, 1389 boxes**:

| Split | Images | Boxes | pedestrian | cyclist | vehicle |
|-------|-------:|------:|-----------:|--------:|--------:|
| train | 240 | 881 | 503 | 129 | 249 |
| val | 80 | 288 | 143 | 43 | 102 |
| test | 60 | 220 | 137 | 29 | 54 |

## Dataset schema

```
data/
├── meta.json                 # {"train": [{"file": ..., "annotations": [{"bbox":[x1,y1,x2,y2], "label": 1}]}]}
├── train/train_0000.png      # 256x192 synthetic street-scene frames
├── val/   ...
└── test/  ...
```

Labels: `1 = pedestrian`, `2 = cyclist`, `3 = vehicle`.
Silver converts `[x1,y1,x2,y2]` → COCO `[x,y,w,h]` and validates every box against image bounds.

## Config

Every path, hyperparameter and the ship/no-ship gate lives in
[`configs/config.yaml`](configs/config.yaml). Change values there, never in code.

### Evaluation gate (`gate:` in config)

| Key | Value | Meaning |
|-----|------:|---------|
| `fpr_budget` | 0.01 | false-positive-rate ceiling (1%) |
| `min_recall_at_fpr` | 0.85 | recall must clear this at the FPR budget |
| `min_ap` | 0.75 | 11-point AP, all classes |
| `min_pedestrian_recall` | 0.85 | safety-critical class, gated separately |
| `max_inference_ms` | 800 | p95 latency budget on CI CPU |

CI fails the deploy if any gate value is missed → that is the *automatic* half of the
HITL gate. High-risk changes additionally require human approval.

## Stage 2 — Training

```powershell
python -m src.models.train --smoke    # ~15 s: imports, shapes, gradients, one batch
python -m src.models.train            # full run: 3 epochs, ~15-25 min on CPU
python -m src.models.train --epochs 5 # override config
pytest -q                             # 40 tests total
```

Output: `models/model.pt` — a checkpoint containing the weights, class list,
training history, metrics and the config used to produce it.

> `--smoke` is what CI runs on every push: it catches broken imports, bad
> tensor shapes and dataset regressions in seconds instead of 25 minutes.
> Set `train.eval_each_epoch: false` in config if you want to skip the
> end-of-run evaluation.

## Stage 3 — Experiment tracking (MLflow)

```powershell
python -m src.experiments.run_experiment --smoke     # ~30 s: full tracking path
python -m src.experiments.run_experiment             # real run, ~15-25 min CPU
python -m src.experiments.run_experiment --run-name "baseline-v1" --tags dataset=v1
python -m src.experiments.run_experiment --no-track  # train without MLflow

mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
```

> **The UI is a separate process — nothing in the pipeline starts it.**
> Training writes straight to `mlflow.db`, so runs keep tracking fine while
> `http://127.0.0.1:5000` refuses connections. Start the viewer yourself
> (from the repo root, so artifacts resolve):
>
> ```powershell
> mlflow ui --backend-store-uri sqlite:///mlflow.db
> # or, with explicit artifact storage:
> python -m mlflow server --backend-store-uri sqlite:///mlflow.db `
>                         --artifacts-destination mlruns --port 5000
> ```

| Logged | Contents |
|--------|----------|
| **params** (25) | every `train.*`, `model.*`, `gate.*` value + label map |
| **metrics** (23) | `recall_at_fpr`, `ap_11point`, `pedestrian_recall_at_op`, `precision/fpr/fppi_at_op`, per-class recall, 5 loss terms per epoch |
| **tags** | `git_sha`, `git_dirty`, `config_sha`, `data_manifest_sha`, `dataset_stats_sha`, `gate_passed` |
| **artifacts** | `config/`, `data/` (bronze manifest + gold stats), `model/` (checkpoint) |

> MLflow 3.x retired the plain `./mlruns` filesystem backend, so metadata lives in
> **SQLite** (`mlflow.db`) while artifacts stay in `mlruns/`. Both gitignored.

`git_dirty: True` on a run means it was built from **uncommitted code** — that run
is not reproducible, and MLflow says so out loud.

## Stage 4 — Evaluation gate + Model Registry

```powershell
# Gate — the script CI calls. Exit 0 = ship, 1 = block, 2 = broken.
python -m src.registry.gate                       # models/model.pt on val
python -m src.registry.gate --split test --json   # hold-out, machine-readable

# Registry — refuses any run whose gate did not pass
python -m src.registry.register --run <run_id>                  # -> @challenger
python -m src.registry.register --run <run_id> --alias champion # -> production
python -m src.registry.register --run <run_id> --dry-run        # inspect only
```

| Exit code | Meaning |
|---:|---|
| `0` | **PASS** — every gate metric cleared its threshold |
| `1` | **FAIL** — model rejected, do not deploy |
| `2` | **ERROR** — missing checkpoint/data/config (a bug, not a bad model) |

Keeping `1` and `2` distinct lets CI tell *"this model is not good enough"*
apart from *"something is broken."*

**Gate thresholds** (`gate:` in config) — CI fails the run if any is missed:

| Metric | Required |
|---|---:|
| `recall_at_fpr` | ≥ 0.85 @ FPR ≤ 1% |
| `ap_11point` | ≥ 0.75 |
| `pedestrian_recall_at_op` | ≥ 0.85 |

The registry additionally refuses runs tagged `gate_passed: false`
(`--force` overrides, loudly). Registering also verifies the model's class
list matches `config.dataset.label_map` — a mismatch would ship a model that
confidently calls cyclists "vehicles."

> **MLflow 3 note:** stages (None/Staging/Production) are replaced by
> **aliases** — `models:/PedestrianDetector@champion` points at the model
> serving traffic, `@challenger` at the candidate. Promoting a retrain is a
> pointer flip, so rollback is instant and no version history is lost.

## Stage 5 — FastAPI serving

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
# interactive docs at http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness probe — the **ECS / ALB target-group check** |
| `GET /model` | which model, classes, device (traceability) |
| `POST /detect?score_thresh=0.3` | image in → `[{bbox, label, score}]` out |

```python
import httpx
r = httpx.post("http://127.0.0.1:8000/detect",
               params={"score_thresh": 0.3},
               files={"file": ("frame.png", open("frame.png","rb").read(), "image/png")})
r.json()   # {detections: [...], image: {width, height}, inference_ms, model_source, ...}
```

**Model source** (resolved in `api/service.py`), highest priority first:

| Env var | Example | Used by |
|---|---|---|
| `MODEL_URI` | `models:/PedestrianDetector@champion` | production (serves whatever the registry says) |
| `MODEL_PATH` | `models/model.pt` | local development |
| `ONNX_MODEL_PATH` | `models/model.onnx` | the graph behind `SERVE_BACKEND=onnx` |
| `SERVE_BACKEND` | `auto` (default) / `torch` / `onnx` | which runtime executes the model |
| `INFERENCE_DEVICE` | `cpu` (default) / `cuda` | torch backend only |

`/health` returns **200 even if the model failed to load** (`model_loaded: false`),
so the ALB can distinguish *starting up* from *broken* while `/model` still
reports the actual error. `score_thresh` is a query parameter because the
operating point is a deployment decision — the notebook's safety point is 0.71 —
and it must be tunable **without retraining**.

Measured on this machine after training (`baseline-v1`), 256×192 PNG:

| runtime | median inference | detections @ 0.30 |
|---|---|---|
| PyTorch (`SERVE_BACKEND=torch`) | ~224 ms | 5 — identical |
| ONNX Runtime (`SERVE_BACKEND=onnx`) | ~208 ms | 5 — identical |

Same boxes, same scores (max delta **0**). The two backends are A/B-testable
precisely because preprocessing is shared — see `tests/test_onnx.py`.

Earlier in the project this measured ~648 ms. Two things changed: the config's
`train.image_min_size: 480` / `image_max_size: 800` were finally applied to
`model.transform` (they had been declared but never wired up), and the model
was actually trained instead of shipping the untrained placeholder.

## Stage 6 — Docker + ONNX serving

One Dockerfile, two images, selected at **build** time so an image can never
end up calling a runtime it does not have:

| image | build command | size | backends available |
|---|---|---|---|
| `pedestrian-api:local` | `docker build -t pedestrian-api:local .` | **1.73 GB** | `torch` and `onnx` |
| `pedestrian-api:slim` | `docker build --build-arg INSTALL_TORCH=0 -t pedestrian-api:slim .` | **702 MB** | `onnx` only |

```powershell
docker compose up --build                # SERVE_BACKEND=onnx by default
curl http://localhost:8000/health        # {"status":"ok","model_loaded":true}
curl http://localhost:8000/model         # "backend": "onnx"
```

Measured **inside** containers (256×192 PNG, 5 warm runs, both images serving
the same weights):

| image | backend | min | median | detections |
|---|---|---|---|---|
| `pedestrian-api:slim` | `onnx` (no torch in the image) | **142 ms** | 152 ms | 5 |
| `pedestrian-api:local` | `torch` | 189 ms | 199 ms | 5 |

Labels identical, **score delta 0, bbox delta 0 px**, `docker inspect` →
`health: healthy`, and `import torch` **fails** in the slim image — it has none.

### Why the container can drop PyTorch

```powershell
python -m src.models.export_onnx --verify     # torch vs ONNX Runtime, same frame
```

The export writes a wrapper that satisfies torchvision's two traps and then
proves the result: **80 detections on both sides, score delta 5.6e-06,
box delta 0.0002 px, labels identical** (`tests/test_onnx.py` asserts this
on every CI run). Once that holds, PyTorch is only needed to *produce* the
graph, not to *run* it — and production should ship the runtime it verified,
not the training framework.

Two details that make the graph safe to run:

* **proposal count pinned to 200** with tensor-only arithmetic — the
  TorchScript tracer freezes data-dependent Python ints (`boxes_per_image`,
  `box_sum`) at export time, so a graph traced on one frame otherwise breaks
  on the next.
* **`n_dets` left dynamic** in `dynamic_axes`, because padding detections to
  a fixed count would bake another such int.

### Where the 702 MB goes (site-packages: 374 MB)

| package | MB | why it is there |
|---|---|---|
| `sympy` | 57 | required by `onnxruntime` |
| `onnxruntime` | 52 | the serving runtime |
| `numpy` (+libs) | 68 | arrays in and out of the graph |
| `mlflow-skinny` + `databricks-sdk` | ~60 | `MODEL_URI=models:/PedestrianDetector@champion` pulls |
| `onnx` | 30 | graph export / `onnx.checker` (`src/models/export_onnx.py`) |
| rest of the ML stack | ~107 | fastapi, uvicorn, pydantic, pillow, pyyaml |
| Python 3.10 base + stdlib | ~330 | OS layer, `libgomp1`, `curl` |

Everything else in `requirements.txt` (Evidently, pandas, pyarrow, pytest,
ruff, the MLflow server) is deliberately excluded — see the header of
`requirements-runtime.txt`.

Build context is **0.1 MB**, not 147 MB: `.dockerignore` excludes `data/`,
`mlruns/`, `models/` and weights, which are mounted or fetched at runtime.

## Repository policy (for GitHub Actions)

CI clones a **fresh, empty VM** — it only ever sees committed files.

**Committed:** `src/`, `configs/`, `tests/`, `api/`, `requirements*.txt`, `Dockerfile`, `docker-compose.yml`, `.github/`.

**Never committed** (see `.gitignore`): `data/`, `mlruns/`, `models/*.pt`, `.venv/`, `.env`.
AWS credentials live in **GitHub Secrets**, not in the repo.

> The synthetic dataset is regenerable from the notebook's `make_scene()` generator,
> so CI can rebuild it without bloating the repo.

## Roadmap

- [x] Step 0 — project scaffold
- [x] Step 1 — data layers (Bronze / Silver / Gold) + 8 passing tests
- [x] Step 2 — training loop (promoted from notebook) + 11 model tests
- [x] Step 3 — MLflow experiment tracking (SQLite backend) + 8 tracking tests
- [x] Step 4 — evaluation gate (exit 0/1/2) + MLflow registry with aliases + 13 tests
- [x] Step 5 — FastAPI serving (health / model / detect) + 12 API tests
- [x] Step 6 — Docker + ONNX serving (two image variants, parity tests) + 9 ONNX tests
- [ ] Step 7 — GitHub Actions CI/CD
- [ ] Step 8 — AWS ECS Fargate
- [ ] Step 9 — Evidently drift monitor
- [ ] Step 10 — HITL gate + retrain trigger
