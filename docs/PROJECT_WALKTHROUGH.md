# Project Walkthrough — Notebook → Monitored, Self-Deploying Model

> This document is the **full step-by-step story** of this repository: what was
> built at each stage, *why* it was built that way, **every failure and error
> encountered and how it was fixed**, and the evidence behind each claim.
> It is written so it can be read end-to-end or walked through in an interview.
>
> **One-paragraph summary:** A Faster R-CNN pedestrian/cyclist/vehicle detector
> was promoted from research notebook code into a production MLOps pipeline:
> versioned data layers → MLflow experiment tracking → an evaluation gate whose
> *exit code* decides ship/no-ship → FastAPI serving → Docker + ONNX → a 5-job
> GitHub Actions CI/CD pipeline → deployment to AWS EC2 with **zero stored
> secrets** → an Evidently drift monitor reading live production traffic → a
> human-approval (HITL) gate → a drift-triggered automatic retrain. Today 99
> tests run in CI on every push, and pushing to `main` puts a gate-checked
> model on a public endpoint in under 15 minutes without anyone touching a
> keyboard.

**Repo:** https://github.com/pravinkotta/pedestrian-detection (public)
**Live endpoint:** `http://3.26.48.54:8000` (`/health`, `/model`, `/detect`,
`/monitor/features`) — *ephemeral IP: it moves on every stop/start; look it up
with `aws ec2 describe-instances --instance-ids i-084fbb0fb3cb86547 --query
"Reservations[0].Instances[0].PublicIpAddress" --output text` and keep the
`LIVE_API_URL` repo variable in sync* · **Headline metric:** *Recall @ FPR ≤ 1%* (safety
metric, not mAP — a missed pedestrian costs more than a false alarm).

> 📖 **Reading this from the repo's front page?** This file *is* the tour —
> skip to [§4 The failure log](#4-the-failure-log) for the debugging stories,
> or [§8](#8-likely-follow-up-questions-answered) for interview-style Q&A.

---

## Contents

1. [The assignment: the mentor's 10-step board](#1-the-assignment-the-mentors-10-step-board)
2. [The system as it stands today](#2-the-system-as-it-stands-today)
3. [Stage-by-stage: what was built, why, and what broke](#3-stage-by-stage-what-was-built-why-and-what-broke)
4. [The failure log (every error, root cause, fix, lesson)](#4-the-failure-log)
5. [Key decisions and trade-offs](#5-key-decisions-and-trade-offs)
6. [Verification: how each claim is proven](#6-verification-how-each-claim-is-proven)
7. [The numbers](#7-the-numbers)
8. [Likely follow-up questions, answered](#8-likely-follow-up-questions-answered)
9. [What I would do next](#9-what-i-would-do-next)
10. [Cheat sheet: running everything yourself](#10-cheat-sheet)

---

## 1. The assignment: the mentor's 10-step board

The goal was a standard MLOps board, applied to a real detector:

```
Feature Engineering + Gold Layer  ->  MLflow Experiment Tracking  ->  MLflow Model Registry
                                                                        |
FastAPI Endpoint  ->  Docker Container  ->  GitHub Actions CI/CD  ->  AWS
                                                                        |
Evidently Drift Monitor  ->  HITL Gate (high-risk)  ->  Retrain Trigger
```

How each box maps to the repository:

| # | Board says | What exists | Where |
|---|---|---|---|
| 0 | scaffold | package layout, config, lint/test tooling | `src/`, `configs/`, `ruff.toml`, `pytest.ini` |
| 1 | Feature Eng + Gold | Bronze/Silver/Gold layers with validation; synthetic scene **generator promoted out of the notebook** | `src/data/` |
| 2 | training | Faster R-CNN training loop + evaluation | `src/models/` |
| 3 | MLflow tracking | runs, params, metrics, provenance tags (SQLite backend + `mlruns/` artifacts) | `src/experiments/tracking.py` |
| 4 | Model Registry | evaluation **gate** (exit 0/1/2) + alias promotion `@champion` / `@challenger` | `src/registry/` |
| 5 | FastAPI | `/health` `/model` `/detect` `/monitor/features` | `api/` |
| 6 | Docker | two image variants + ONNX export with parity proof | `Dockerfile`, `Dockerfile.deploy`, `src/models/export_onnx.py` |
| 7 | GitHub Actions | 5 jobs: lint, tests, docker, train+gate, publish | `.github/workflows/ci.yml` |
| 8 | AWS | t3.micro on EC2 + auto-pull deploy loop from GHCR | `deploy/user-data.sh` |
| 9 | Evidently drift | features from live traffic, K-S per column, JSON/HTML report | `src/monitoring/` |
| 10 | HITL → retrain | GitHub environment approval gate + drift-triggered retrain workflow | `.github/workflows/monitor.yml`, `retrain.yml` |

**Two deviations from the board, both deliberate:** AWS target is **EC2**, not
ECS/Fargate (cost + permissions, §5), and image distribution is **GHCR**, not
ECR (permissions, §5). Both are documented in the README where the board is
quoted.

---

## 2. The system as it stands today

### The three loops

```
A. DELIVERY (every push to main)
   git push ─▶ CI: lint | tests | docker | train+gate │ (parallel, ~8-9 min)
                          └─ gate EXIT CODE decides ─▶ publish ghcr.io:…:latest
                                                          │
        EC2 box, every 5 min: docker pull ────────────────┘
        new image id? replace container   (≤5 min later: live)

B. MONITORING (daily 05:00 UTC, or manually)
   live box /monitor/features ─▶ monitor.yml ─▶ Evidently K-S vs train baseline
                                    │ verdict via exit code: 0 OK / 1 DRIFT / 2 ERROR
                                    └─ DRIFT ─▶ auto-dispatch retrain.yml

C. RETRAIN (after drift, or manually)
   retrain.yml: job `approval` targets environment `hitl`
        ⏸ WAITS for a human to click Approve  ← the HITL gate
        │ approved
        ▼
   train → gate (exit code!) → ONNX parity → register @challenger → publish
        │ PASS                                                     │
        └──────────────────────────────────────────────────────────┘
                          EC2 box pulls the new model (≤5 min)
```

### Facts

| | |
|---|---|
| Model | Faster R-CNN (MobileNetV3-Large + FPN), classes: pedestrian / cyclist / vehicle |
| Data | synthetic street scenes (generator promoted from the notebook into `src/data/generate.py`, so a fresh CI VM can rebuild it byte-for-byte) |
| Serving | FastAPI + ONNX Runtime, CPU, ~266 ms inference on the t3.micro |
| CI | 5 jobs, green in ~8–9 min; **99 tests**; zero stored secrets |
| Deploy | GHCR public package, **anonymous pull** by the instance |
| Monitor | Evidently 0.7, 9 input features, K-S p<0.05 per column, dataset verdict when >50% of features drift |
| HITL | GitHub environment `hitl`, `required_reviewers` rule, self-review allowed (single reviewer) |
| Cost | ~$9.6/mo while the instance runs (free inside AWS's 12-month free tier) |

---

## 3. Stage-by-stage: what was built, why, and what broke

### Stage 0 — Scaffold

Standard layout (`src/`, `api/`, `tests/`, `configs/`), one central
`configs/config.yaml` so no path is ever hardcoded in a script, `ruff` for
lint, `pytest` for tests. `pytest.ini` carries `pythonpath = .` — added
*after* CI failed (see F4), so imports never depend on how pytest was invoked.

### Stage 1 — Data layers (Bronze / Silver / Gold)

Raw frames + `meta.json` → validated Bronze manifest → normalised COCO
Silver annotations → Gold splits + `dataset_stats.json` (the drift baseline
lives here). Each layer has validation tests.

**What broke:** the scene generator (`make_scene()`) existed *only inside the
notebook*, which is deliberately not in the repo (the notebook is the
research source of truth, kept outside). CI clones a clean VM and therefore
could not build the dataset at all. **Fix:** promoted the generator into
`src/data/generate.py` as a proper module with deterministic output — CI now
rebuilds the dataset from nothing on every run. **Lesson:** a pipeline step
that only exists in a researcher's laptop is not a pipeline step; CI is what
proves ownership of your own code.

### Stage 2 — Training

Training loop, dataset/dataloader, evaluation promoted from the notebook with
the same hyperparameters (seed 42, `num_workers: 0` for Windows).

**What broke:** CI's test job initially ran a `--smoke` training pass, which
by design writes **no checkpoint** (`train.py` guards it so a plumbing check
can never clobber a real checkpoint) — so every API test 503'd. **Fix:**
the test job now trains a genuine 1-epoch model so tests exercise a real
checkpoint (commit `5dedb61`). **Lesson:** design your guard rails
(no-smoke-writes) *and* the workflow around them; the mismatch between them
was the bug.

### Stage 3 — MLflow experiment tracking

Runs record params, metrics, git SHA, dataset stats — provenance tags so you
can always answer "which code + which data produced this number". SQLite
backend (MLflow 3.x retired the flat `./mlruns` filesystem backend),
artifacts stay in `mlruns/`.

### Stage 4 — Evaluation gate + Model Registry ⚠️ *the big bug lived here*

`src/registry/gate.py` evaluates and **returns an exit code**:
`0 PASS · 1 FAIL · 2 ERROR`. CI depends on that contract; distinguishing
"bad model" (1) from "broken tooling" (2) keeps a crashed script from being
misread as "model rejected". The registry promotes aliases (`@champion`,
`@challenger`) — nothing self-promotes to `@champion`.

**What broke (F2, the most instructive failure of the project):**
the RPN's evaluation-time proposal caps (`_pre_nms_top_n` /
`_post_nms_top_n["testing"]`) were **dead configuration** — a dict key
nothing ever wrote, so torch silently evaluated at torchvision's *default*
budget while the exported ONNX graph was built around a fixed
`FIXED_PROPOSALS=200` canvas. Two models that were supposed to be identical
were being asked *different questions*, and the first CI run failed its ONNX
parity check.

**Fix:** `apply_rpn_caps()` in `build_model.py` now writes the testing caps
(600 pre / 200 post) explicitly at model construction, and the exporter
(`export_onnx.py --verify`) **refuses to export** if torch's live RPN cap
exceeds the graph's canvas — the failure mode cannot recur silently. After
the fix: gate PASS under the real budget (*recall@FPR 0.8854, AP 0.8167,
pedestrian recall 0.8531, threshold 0.0585*), ONNX parity **80 == 80**
detections, max score delta **5.6e-06**.

**What broke (F5):** the MLflow `baseline-v1` run still carried metrics
logged *before* that fix, so the registry advertised numbers no code path
produced any more. **Fix:** `gate --mlflow-run <id>` re-mirrors the corrected
metrics onto the existing run (chosen over creating a new run so the
champion's identity and history stay intact). Tracking is a *record*, never
the verdict — the mirror failure path prints a warning and never changes the
gate's exit code.

### Stage 5 — FastAPI serving

`/health` (200 even when the model failed to load — but reporting
`model_loaded: false`, because a broken container should be *replaced*, and
the reason must stay visible on `/model`), `/model` (traceability: which
model, which classes, which code — `git_sha`), `/detect` (score threshold as
a *request* parameter, because the operating point is a deployment decision).

### Stage 6 — Docker + ONNX

Two-stage build (builder → runtime), CPU-only torch wheels (PyPI's default
wheel drags ~2.5 GB of CUDA; the CPU index is ~190 MB), non-root user,
`HEALTHCHECK` that is stricter than `/health` (fails if the model did not
load, so an orchestrator replaces it instead of serving 503s). Two image
variants: full and **slim** (ONNX only, *proven* torch-free by a CI test).
Weights are excluded from every build context by `.dockerignore`; the deploy
image (`Dockerfile.deploy`) gets its own context rules so *that one build*
may see `models/model.onnx` while no other build can.

**What broke (F3):** a transient **`502 Bad Gateway` from
registry-1.docker.io** while pulling `python:3.10-slim` killed the first CI
run — an upstream outage, not a repo problem. **Fix:** three attempts with
backoff around every `docker build`. **Lesson:** classify failures before
"fixing" your own code — retrying an upstream hiccup is correct, retrying a
real bug hides it (which is exactly why the gate's ERROR state exists).

### Stage 7 — GitHub Actions CI/CD

Five jobs; wall clock = the slowest job, not the sum. `train-and-gate` does
the real work (full training → gate exit code → ONNX verify → register
`@challenger`); `deploy` publishes to GHCR only on green pushes to `main`.
The gate report is uploaded as an artifact on every run.

**What broke:**
- **F1** — first-ever CI run failed (ONNX parity from F2 + the DockerHub 502
  of F3). Fixed both, plus the smoke-checkpoint issue (F-Stage-2).
- **F4** — `ModuleNotFoundError: No module named 'src'` when CI invoked bare
  `pytest` (run `40271189827`). Locally the import had been resolved by
  *incidental* path behaviour; on a clean runner it was not. **Fix:**
  `pytest.ini` with `pythonpath = .` — the import contract is now declared,
  not accidental. **Lesson:** "works on my machine" is usually an undeclared
  dependency on your machine's state.
- **F10** — the four action pins (`checkout@v4` etc.) began emitting
  *Node 20 deprecated* annotations: GitHub was force-running them on Node 24
  via a compatibility layer. **Fix:** read all four changelogs first (the
  majors had real breaking changes: `checkout`'s `pull_request_target` change,
  `download-artifact`'s by-ID path change and v8 digest enforcement — none of
  which touch this repo's usage), then bump to `checkout@v7`,
  `setup-python@v7`, `upload-artifact@v7`, `download-artifact@v8`.
  Verified the annotations were gone and CI stayed green.

### Stage 8 — Deploy on AWS (EC2 + GHCR)

The mentor's board says *AWS ECS / Kubernetes*. The account's IAM policy had
other ideas, and the restrictions **are** the architecture (§5 has the full
table): no `iam:CreateRole` → no instance profile → the instance can never
authenticate to ECR (and Fargate's task role is impossible too); no `s3:*` →
nowhere to stash weights; no `ec2:CreateKeyPair` → no new SSH keys.

**Design that resulted:**
- **EC2 t3.micro** (cheapest path, ~$9.6/mo vs ~$33/mo for Fargate+ALB),
  launched with the existing `AI-LLM` key and a security group opening only
  8000 + 22.
- **GHCR instead of ECR:** the repo is public → the package is public → the
  instance pulls **anonymously with zero credentials** (verified with a
  credential-less token fetch *before* launching: HTTP 200), CI pushes with
  its automatic `GITHUB_TOKEN`.
- **Weights baked into the image** (`Dockerfile.deploy`) — a deployed API
  must not depend on a training-time service or an object store it cannot
  reach.
- **`deploy/user-data.sh`** bootstraps docker and runs a systemd *poller*
  (`pedestrian-deployer`): every 5 minutes, pull `:latest`; if the image id
  changed, replace the container. Cloud-init runs user-data once, so the loop
  lives in systemd, not in the boot script.

**Proving observability:** `/model` now reports `git_sha`, stamped at image
build via `--build-arg GIT_SHA` — the answer to "which code is this endpoint
running?" became one `curl` instead of a guess. The deploy loop was then
observed end-to-end **four times** (`unknown → b522351 → 65a14d0 → c05db6f →
… → 23ea913`), each with zero manual action.

**Other failures in this stage:** an empty ECR repository (created when ECR
was still the plan) was deleted once GHCR won; the public IP is *ephemeral*
(no `ec2:AllocateAddress` permission for an Elastic IP) — documented as a
caveat, and the `LIVE_API_URL` repo variable must be updated if the box
restarts.

### Stage 9 — Evidently drift monitor

Because the detector serves *images*, and drift tools compare *columns*, each
frame is summarised into a row of 9 numbers (`src/monitoring/features.py`):
brightness, contrast, Laplacian sharpness, per-channel means, geometry.
**Only aggregates are stored — never the image** — so the capture log is ~9
floats per request, safe to keep where raw frames would not be.

Flow: `POST /detect` → `_capture()` appends a row (opt-in via
`FEATURES_LOG`, self-rotating at 5 MB, **failures swallowed — monitoring may
never break inference**) → `GET /monitor/features` exposes the batch over
plain HTTP → `python -m src.monitoring.drift --url …` compares it against the
cached feature baseline of the gold **train** split and writes JSON + Evidently
HTML reports. Same exit-code contract as the gate: `0 OK / 1 DRIFT / 2 ERROR`.

**What broke / what we caught:**
- **F6 — the Evidently API had moved.** `requirements.txt` pinned
  `evidently>=0.4.20`, but the code was written against the **0.7** API
  (`evidently.presets.DataDriftPreset`; `Report.run()` returns a Run whose
  `.json()` carries per-column K-S p-values). The 0.4 line has a completely
  different API — a resolver change would have broken CI silently. **Fix:**
  pinned `evidently>=0.7.23,<0.8` with a comment saying why the floor is a
  compatibility boundary, not a suggestion.
- **Design bug caught in review before push:** an ad-hoc
  `--reference-dir` (compare against *any* baseline folder) would still have
  read the **default cache**, silently comparing the wrong baselines.
  **Fix:** ad-hoc baselines use `cache=None` — a one-off baseline can neither
  read nor poison the real one.
- **Statistical honesty:** columns that are *constant* in the reference
  (every frame is 256×192, so width/height/aspect) are dropped with a note —
  there is no distribution to compare, and judging them would manufacture
  both false alarms and false all-clears. Fewer than `min_rows` (10) current
  samples → **ERROR (2), not a verdict**: a monitor that cries wolf on 2
  samples is worse than one that stays quiet.
- **YAML + `python -c` indentation trap:** a multi-line `python -c` block in
  the workflow would have handed Python leading indentation
  (`IndentationError`) because YAML block scalars preserve relative indent.
  Rewritten as one line *before* pushing. (see F8 for the family of these.)

**Verified both ways:**
```
$ python -m src.monitoring.drift --current data/val
[DRIFT] OK — 0/6 features drifted (share 0.000 vs limit 0.5)        exit 0

$ python -m src.monitoring.drift --current <40 frames darkened to 25%>
[DRIFT] DRIFT — 6/6 features drifted (share 1.000 vs limit 0.5)     exit 1
    DRIFT  brightness  p=4e-49   DRIFT  contrast  p=4e-49   …

$ python -m src.monitoring.drift --url http://3.26.48.54:8000/monitor/features
[DRIFT] OK — 1/6 features drifted (share 0.167 vs limit 0.5)        exit 0
    (18 real requests pulled live from the box; contrast alone p=0.047 —
     exactly the case the >50% dataset rule exists for)
```

### Stage 10 — HITL gate + retrain trigger

Two workflows close the mentor's loop:

- **`monitor.yml`** — daily cron + manual; rebuilds the baseline, pulls live
  features, writes the verdict into the run summary, and **on DRIFT
  dispatches `retrain.yml`**. The dispatch uses only the built-in
  `GITHUB_TOKEN` (scoped `actions: write`): GitHub deliberately *blocks*
  event cascades from `GITHUB_TOKEN` to prevent recursion — **except
  `workflow_dispatch`/`repository_dispatch`**, which is precisely what makes
  a zero-secret automatic trigger possible.
- **`retrain.yml`** — first job targets the **`hitl` environment**, which has
  a `required_reviewers` protection rule, so the run parks in *Waiting* until
  a human approves. On approval the proven chain runs: train → gate exit
  code → ONNX parity → register `@challenger` → publish `:latest` → the box
  re-pulls. **Approval is not a quality gate** — even after a human opens the
  door, `gate.py`'s exit code still decides what may ship.

A `simulate_night=true` input demonstrates the entire chain on demand by
comparing against darkened frames (guaranteed DRIFT).

**What broke while building it (F9):**
- GitHub's API rejected the environment creation with **HTTP 422** —
  `-f reviewers[0][type]=User` on the `gh` CLI builds a JSON *object*, not an
  *array*. Fixed by piping proper JSON via `--input -`.
- Three of my own workflow-structure tests failed on YAML semantics, not on
  the workflows: `schedule:` parses as a **list** of `{cron: …}` maps; a
  single `needs:` parses as a **scalar**; and the zero-secrets test matched
  the English word "secrets." inside a *comment*. Fixed the tests (skip
  comments, normalise scalars) — the assertions themselves were right.

**End-to-end proof (recorded runs):**

| # | What happened | Evidence |
|---|---|---|
| 1 | Monitor detected DRIFT (simulated night frames) | run `40290399140` conclusion **failure** (red = alarm) |
| 2 | Auto-trigger fired | `Retrain trigger` step **success** → run `40290513804` created |
| 3 | **HITL held** | job `HITL approval (human decides)` status **`waiting`** — no training started |
| 4 | Human approved (via pending-deployments API) | deployment `7327000268` approved with comment |
| 5 | Chain ran | approval 2s → train+gate **7m45s** → publish **1m24s** |
| 6 | New model published | GHCR `:latest` config digest `088b26c9…` **≠** CI's build of the same commit `17d6da0e…`, and **=** `:retrain-40290513804` |
| 7 | Box deployed it | capture log reset to 0 rows (container replaced) + `/health` → `model_loaded: true` |

---

## 4. The failure log

Every error encountered, grouped by kind. Each entry: *what happened → root
cause → fix → lesson*.

### A. Pipeline / product bugs

| # | Failure | Root cause | Fix | Lesson |
|---|---|---|---|---|
| F1 | First CI run failed (run `40269995723`) | ONNX parity mismatch (F2) **+** Docker Hub 502 (F3) | fixed both, not just the symptom | classify before fixing |
| F2 | ONNX parity ≠ torch | RPN `["testing"]` caps were **dead config** — nothing wrote the key, so eval used torchvision defaults while the graph assumed 200 proposals | `apply_rpn_caps()` sets 600/200; exporter now *refuses* to export if torch's cap exceeds the canvas | config that nothing sets is not config, it's a rumour — and make the failure mode unrepresentable |
| F3 | `502 Bad Gateway` pulling `python:3.10-slim` | transient Docker Hub outage | 3-attempt retry with backoff in CI | upstream flakes ≠ your bug; retry only the retryable |
| F4 | `ModuleNotFoundError: src` (run `40271189827`) | bare `pytest` relied on incidental local path behaviour | `pytest.ini` → `pythonpath = .` | declare import contracts; "works locally" is undeclared state |
| F5 | Registry showed stale metrics on `baseline-v1` | metrics were logged before F2's fix | `gate --mlflow-run` re-mirrors corrected metrics onto the same run | records must be repairable; verdicts must stay independent of records |
| F6 | `evidently>=0.4.20` pin vs 0.7-only code | API rewritten between majors | pin `>=0.7.23,<0.8` with an explanatory comment | version floors are compatibility boundaries |
| F7 | Ad-hoc `--reference-dir` would read the default cache | cache key didn't track the baseline source | ad-hoc baselines bypass the cache entirely | review your own design for silent-wrong-answer paths, not just crashes |
| F8 | Caught pre-push: multi-line `python -c` in YAML would `IndentationError`; a `--build-arg INSTALL_Torch` typo | YAML block scalars preserve indent; case-sensitive build args | single-line `python -c`; corrected arg | cheap failures caught before the commit are free |
| F9 | Env-creation HTTP 422; 3 of my own workflow tests failed | `gh -f` builds objects not arrays; `schedule` is a list; single `needs` is a scalar; test matched the word "secrets." in a comment | JSON via `--input -`; normalised YAML semantics in tests | read the parser's output shape, not your assumption of it |
| F10 | Node 20 deprecation annotations on every run | action pins targeted an EOL runtime GitHub now force-runs on Node 24 | read 4 changelogs for breakage, then bump to v7/v7/v7/v8; annotations gone, CI green | warnings are deadlines with no visible timer |

### B. Cloud / IAM walls (these *shaped the architecture*)

| Denied | Consequence | What we did instead |
|---|---|---|
| `iam:CreateRole` | no instance profile → box can't auth to ECR or SSM; **Fargate's task role impossible too** | GHCR public package, **anonymous pull** (verified HTTP 200 before launch) |
| `s3:*` | nowhere to put the weights | bake `model.onnx` into `Dockerfile.deploy` |
| `ec2:CreateKeyPair` | no new SSH keys | cloud-init bootstrap; existing `AI-LLM` key for human debugging only |
| `ec2:CreateTags` | can't name the instance | identify by instance ID `i-084fbb0fb3cb86547` |
| `ec2:AllocateAddress` | no Elastic IP | document the ephemeral IP; keep `LIVE_API_URL` repo var current |
| `ec2:StopInstances` **and** `ec2:StartInstances` | the `admin` user cannot stop the box to stop billing (both verified `UnauthorizedOperation` on 2026-10-26) | documented honestly in the README instead of pretending the command works; needs root console or a policy update. Stop/start itself needs **no manual container work** — systemd `enable` + `--restart always` are on disk |
| (allowed) `RunInstances`, SGs | launch + open 8000/22 | how the service exists at all |

ECR repo was created while ECR was still the plan, then **deleted** once GHCR
won — no dead resources left in the account.

### C. Environment & tooling quirks (small, but they cost time)

| Symptom | Cause | Fix |
|---|---|---|
| `Unexpected token '??'` | PowerShell 5.1 has null-coalescing | rewrite without `??` |
| `git commit -m @'…'@` → `pathspec 'the' did not match` | PS here-string mangled across the native call; **nothing was committed** | write message to a file, `git commit -F file` |
| `﻿BOM` in commit subjects | `Out-File -Encoding utf8` writes a BOM | `[IO.File]::WriteAllText` (subjects `b522351`, `65a14d0` still show it — cosmetic) |
| `--jq ".tag_name"` → "term not recognized" | PS interpolated the jq expression's quotes | fetch JSON, parse with `ConvertFrom-Json` |
| `curl -F "score_thresh=…"` ignored by the API | it's a **query** param, not a form field | `?score_thresh=…` |
| `--html ""` → "expected one argument" | PS drops the empty argument | pass `--html=` |
| `gh variable set` → "not a git repository" | `gh` resolves the repo from cwd | run inside the repo |
| one-off `read tcp … failed to respond` from api.github.com | transient network blip | retry — same class as F3 |
| **`aws ec2 wait instance-stopped` hung for 10 min** (2026-10-26) | the `stop-instances` call *before* it had **failed on IAM** — `wait` polls for a state that will never arrive and never re-checks why | check `$LASTEXITCODE` after the operation you're waiting on, *before* calling `wait`; and bound every wait with a timeout (`timeout-minutes`, `--cli-read-timeout`) |

### D. Lint / hygiene (ruff kept us honest)

Unused `noqa` directive, redundant `int()` casts, **NaN-check via `p != p`**
(flagged as "name compared with itself" → replaced with `math.isNaN`
equivalents), import ordering, `typing.Iterable` → `collections.abc`.
All fixed; `ruff check .` is clean and enforced as CI's first job.

---

## 5. Key decisions and trade-offs

| Decision | Alternative | Why this one |
|---|---|---|
| **EC2 t3.micro** | ECS Fargate + ALB (~$33/mo) | ~$9.6/mo; **and** Fargate needs an execution/task role → `iam:CreateRole` denied made it impossible anyway. Container is still portable to Fargate later — that's the point of shipping a container |
| **GHCR public package** | ECR | instance has no identity (no role) → ECR unusable; public GHCR = anonymous pull = zero credentials in the whole chain |
| **Weights baked into the image** | pull from S3/MLflow at boot | `s3:*` denied, no MLflow server reachable; also: a deployed API must not depend on a training-time service |
| **Zero stored secrets** | AWS keys / PATs in GitHub Secrets | IAM denied most things anyway; `GITHUB_TOKEN` + anonymous pull covers everything. Enforced by test F-`no_stored_secrets` so it can't rot |
| **Exit codes 0/1/2 as the contract** | parse logs / human reads | machines gate on exit codes; separating ERROR(2) from FAIL(1) stops tooling crashes being read as "model rejected" (and drift ERROR vs DRIFT the same way) |
| **Image features, not pixels** | raw frame archive / embeddings | 9 floats/request, no privacy exposure, explainable ("brightness drifted") instead of "column 48213 drifted" |
| **K-S per column + >50% share rule** | single omnibus test | per-column tells you *what* moved; share rule stops one borderline p-value (contrast p=0.047 in the live check) from crying wolf |
| **HITL = GitHub environment protection** | custom approval bot, manual `workflow_dispatch` | native, auditable (who/when/reason), zero code, zero secrets; blocked *before* any compute is spent |
| **Approval ≠ promotion** | human decides quality | humans open doors; **metrics decide** what walks through — gate still runs after approval |
| **`retrain.yml` duplicates CI's jobs** | refactor into a reusable workflow | restructuring a green, reviewed ship gate is a riskier change than honest duplication; flagged in the file header as the first thing to extract later |
| **SQLite MLflow tracking** | hosted MLflow server | no server to operate/secure; registry semantics (aliases) preserved. Swap the tracking URI when there's a server |

---

## 6. Verification: how each claim is proven

| Claim | Evidence |
|---|---|
| Gate is green under the real budget | recall@FPR **0.8854** (≤1% FPR), AP 0.8167, pedestrian recall 0.8531, thr 0.0585 |
| ONNX == PyTorch | parity **80 == 80** detections, max score delta **5.6e-06** (`export_onnx --verify` in every CI run) |
| CI is trustworthy | 99 tests; 10 CI runs: 2 early failures root-caused & fixed, **8 green since** (runs `40271924968` … `40289450998`) |
| Slim image has no torch | CI step asserts `find_spec('torch') is None && find_spec('onnxruntime')` inside the image |
| The instance needs no credentials | anonymous manifest fetch → **HTTP 200** with a credential-less token, checked *before* launch |
| Deploy loop works unattended | `/model.git_sha` observed flipping `unknown → b522351 → 65a14d0 → c05db6f → 23ea913 → 1f68bbe`, no human action |
| Drift monitor discriminates | val → OK (exit 0); darkened frames → DRIFT 6/6, p=4e-49 (exit 1); live 18 rows via `--url` → OK with 1/6 flagged |
| Capture can't break serving | test posts an unwritable `FEATURES_LOG` path → `/detect` still 200 |
| HITL actually blocks | run `40290513804` sat in **`waiting`**; only the approval API call started training |
| Retrain actually shipped | `:latest` digest `088b26c9…` = `:retrain-40290513804` ≠ CI's `17d6da0e…`; box's capture log reset to 0 (container replaced), `/health` ok |
| Zero secrets | `secrets.` references in workflows = only `GITHUB_TOKEN`, asserted by `tests/test_workflows.py` |

---

## 7. The numbers

**Model / evaluation**
- Recall @ FPR ≤ 1%: **0.8854** · AP: **0.8167** · pedestrian recall: **0.8531** · operating threshold: **0.0585** (gate), 0.3 default served, 0.71 notebook safety point
- ONNX parity: 80 == 80 detections, Δscore ≤ **5.6e-06**
- Inference: **~266 ms** CPU on t3.micro (CI budget: p95 ≤ 800 ms, baseline ~734 ms)

**Pipeline**
- **99 tests** across 9 files in `tests/` (API, data pipeline, generator,
  model, tracking, registry, ONNX, monitoring, workflows), ruff clean
- CI: 5 jobs, wall clock **~8–9 min**; `train-and-gate` ~7 min dominates
- **17 commits** of structured history (each step + each fix is its own commit); **12+ workflow runs** (10 CI, 1 monitor, 1 retrain)
- Deploy latency: push → live **≤ ~13 min** (CI ~9 + poller ≤5)

**Ops / cost**
- t3.micro ap-southeast-2: **~$9.6/mo** running, **$0 stopped (free-tier eligible 750 h/mo)
- GHCR storage: free (public) · ECR: deleted · secrets stored anywhere: **0**

---

## 8. Likely follow-up questions, answered

**Why not Kubernetes/ECS like the board said?**
Two reasons: cost (Fargate + ALB ≈ $33/mo vs $9.6/mo for one micro instance)
and capability — `iam:CreateRole` is denied in this account, so Fargate's
task role (and ECR auth, and SSM) are impossible, not merely expensive. The
image is portable: moving to Fargate/EKS later is a `run` command, not a
rewrite. The deviation is documented in the README where the board is quoted.

**Isn't one reviewer a rubber stamp?**
Yes — and the config says so: `prevent_self_review: false`, single reviewer.
The *mechanism* is what matters: an auditable, non-bypassable-by-code pause
before compute is spent. Hardening is one API call
(`prevent_self_review: true` + a second reviewer for four-eyes).

**Synthetic data — does any of this transfer to real data?**
The model quality numbers don't; **every pipeline property does**: the layers,
the gate contract, the registry, CI, the deploy loop, drift detection. The
generator was promoted into the repo precisely so CI can rebuild data from
nothing — swapping in a real dataset is a change to `src/data/` only.

**What happens when the box's IP changes or it dies?**
Stop/start moves the ephemeral IP (no `AllocateAddress` permission) — update
the `LIVE_API_URL` repo variable; monitoring then keeps working. **Note that
this `admin` user cannot do the stop itself**: both `ec2:StopInstances` and
`ec2:StartInstances` were verified denied (`UnauthorizedOperation`), so
stopping means the root console or a policy update. Once stopped and started
by someone with permission, **no container is started by hand** — docker, the
`pedestrian-deployer` poller and the container's `--restart always`
all live on the EBS disk and come back at boot (the poller even recreates a
missing container); only the IP needs re-pointing. If the box dies,
`/health` fails, nothing depends on it for training/CI, and `user-data.sh`
rebuilds it from scratch (it has no state worth saving).

**How do you know a deploy landed?**
`curl /model` → `git_sha` is the commit the image was built from. For the
retrain (same commit, new weights) the image digest differs —
`:latest` = `088b26c9…` vs CI's `17d6da0e…`.

**Why exit codes instead of dashboards for the gate?**
Because a pipeline should stop without a human reading charts. Exit 0/1/2 is
consumed by CI today and would be consumed by any orchestrator tomorrow;
dashboards are for humans reviewing *why*, which is why the JSON/HTML reports
exist alongside.

**Is the API on port 8000 with no auth a problem?**
For a demo, no — and it's documented in the README as a caveat: put it behind
a reverse proxy/ALB with auth before real traffic. The endpoint only runs
inference on images and returns boxes; capture stores aggregates, not images.

**Why is the drift monitor a separate workflow with a lean requirements file?**
A scheduled job shouldn't download 200 MB of torch to compute means. It also
isolates failure: monitor red ≠ CI red.

---

## 9. What I would do next

1. **Alerting** — on scheduled DRIFT, open a GitHub Issue (still zero
   secrets) so a red 05:00 run isn't the only signal.
2. **Four-eyes + wait timer** on `hitl`; branch protection with required
   reviews on `main`.
3. **Prediction drift** — `n_detections`/`mean_score` are already captured;
   wire them as a second monitor (they answer "did the *model's world* move",
   input drift answers "did the *camera's world* move").
4. **Statistical hardening** — Benjamini–Hochberg correction across columns
   (9 simultaneous K-S tests inflate false positives), and a per-column
   baseline of historical p-values rather than a single-shot comparison.
5. **Stable address** — Elastic IP / Route53 the moment the account allows it.
6. **Supply chain** — image scanning (Trivy), SBOM, pinned digests instead of
   `:latest` on the box.
7. **Extract the reusable workflow** (noted in `retrain.yml`'s header) once
   the two copies have drifted enough to justify the refactor risk.
8. **Cost** — stop the instance when not demoing. Blocked *today*: the `admin`
   user is denied `ec2:StopInstances`/`StartInstances` (§4B), so this needs
   the root console or ~2 lines added to the IAM policy — worth doing, since
   it's the one thing standing between the project and "stop paying".

---

## 10. Cheat sheet

```powershell
# ---- local ----
pip install -r requirements.txt
python -m src.data.generate                 # rebuild the synthetic dataset
python -m src.experiments.run_experiment    # train + log to MLflow
python -m src.registry.gate                 # exit 0/1/2 = PASS/FAIL/ERROR
python -m src.registry.gate --mlflow-run <id>   # correct a run's metrics
python -m src.models.export_onnx --verify   # parity proof
pytest -q && ruff check .                   # what CI runs first

# ---- serving ----
uvicorn api.main:app --port 8000
curl -F "file=@data/val/val_0000.png" http://127.0.0.1:8000/detect

# ---- drift (Step 9) ----
python -m src.monitoring.drift --current data/val                 # OK   -> 0
python -m src.monitoring.drift --current <frames>                 # may DRIFT -> 1
python -m src.monitoring.drift --url http://3.26.48.54:8000/monitor/features

# ---- automation (Step 10) ----
gh workflow run monitor.yml -f simulate_night=true   # demo the whole chain
gh workflow run retrain.yml -f "reason=manual"       # starts at the HITL gate
gh run list; gh run watch <id>

# ---- AWS ----
aws ec2 describe-instances --instance-ids i-084fbb0fb3cb86547
# stop/start: correct commands, but the `admin` user is DENIED both actions —
# run them from the root console (or after a policy update), then re-point:
#   gh variable set LIVE_API_URL --body http://<new-ip>:8000
aws ec2 stop-instances   --instance-ids i-084fbb0fb3cb86547   # stop paying
aws ec2 start-instances  --instance-ids i-084fbb0fb3cb86547   # self-redeploys
```

---

### Appendix: 60-second interview pitch

> "I took a pedestrian-detection notebook and made it production-shaped. Data
> lives in validated bronze/silver/gold layers that CI rebuilds from scratch.
> Training logs to MLflow, and a **gate with an exit code** — not a dashboard —
> decides ship/no-ship, with a fixed bug I'm proud of: the RPN's eval proposal
> caps were dead config, so torch and the exported ONNX graph were answering
> different questions; I made the caps explicit and made the exporter refuse to
> build in that state. Serving is FastAPI + ONNX in a two-variant Docker image,
> shipped by a 5-job GitHub Actions pipeline to an EC2 box that **pulls
> updates every 5 minutes with zero stored secrets** — the account's IAM
> denials forced GHCR-over-ECR and weights-baked-into-image, and turned out to
> be better engineering anyway. Drift monitoring summarises live requests into
> 9 features and runs Evidently K-S tests against the training baseline with
> the same 0/1/2 exit contract, and a detected drift auto-dispatches a retrain
> that **stops dead at a GitHub-environment approval gate** until a human says
> go — after approval the same gate still decides whether the model ships.
> 99 tests, two early CI failures fully root-caused, everything documented."

### Appendix: glossary (plain definitions)

- **Data drift** — the *inputs* your model sees no longer follow the
  distribution it was trained on. The model didn't change; the world did.
- **K-S (Kolmogorov–Smirnov) test** — compares two samples of a numeric
  column; small p-value (< 0.05) ⇒ they likely come from different
  distributions ⇒ that column "drifted".
- **drift share** — fraction of monitored columns that drifted; the *dataset*
  is flagged when it exceeds the limit (0.5 here), so one borderline column
  can't trigger a retrain.
- **FPR budget / Recall @ FPR ≤ 1%** — recall measured while allowing at most
  1% false alarms; a safety-oriented way to score a detector, because missed
  pedestrians are the costly error.
- **HITL** — human-in-the-loop: a machine proposes, a person approves before
  anything consequential happens.
- **`@champion` / `@challenger`** — MLflow registry aliases: *champion* is
  what serves, *challenger* is the candidate waiting to beat it.
- **ONNX parity** — proof that the exported graph computes the same outputs
  as the original PyTorch model (here: identical 80 detections, Δ ≤ 5.6e-06).
- **Exit-code contract** — agreed meaning of a program's return value
  (`0/1/2`) that other systems build decisions on.
