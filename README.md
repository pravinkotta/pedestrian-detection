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
pytest -q                             # 19 tests
```

Output: `models/model.pt` — a checkpoint containing the weights, class list,
training history, metrics and the config used to produce it.

> `--smoke` is what CI runs on every push: it catches broken imports, bad
> tensor shapes and dataset regressions in seconds instead of 25 minutes.
> Set `train.eval_each_epoch: false` in config if you want to skip the
> end-of-run evaluation.

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
- [ ] Step 3 — MLflow experiment tracking
- [ ] Step 4 — MLflow model registry + evaluation gate
- [ ] Step 5 — FastAPI serving
- [ ] Step 6 — Docker
- [ ] Step 7 — GitHub Actions CI/CD
- [ ] Step 8 — AWS ECS Fargate
- [ ] Step 9 — Evidently drift monitor
- [ ] Step 10 — HITL gate + retrain trigger
