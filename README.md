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

## Repository policy (for GitHub Actions)

CI clones a **fresh, empty VM** — it only ever sees committed files.

**Committed:** `src/`, `configs/`, `tests/`, `requirements.txt`, `Dockerfile`, `.github/`.

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
- [ ] Step 5 — FastAPI serving
- [ ] Step 4 — MLflow model registry + evaluation gate
- [ ] Step 5 — FastAPI serving
- [ ] Step 6 — Docker
- [ ] Step 7 — GitHub Actions CI/CD
- [ ] Step 8 — AWS ECS Fargate
- [ ] Step 9 — Evidently drift monitor
- [ ] Step 10 — HITL gate + retrain trigger
