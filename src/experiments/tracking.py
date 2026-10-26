"""Experiment tracking (Step 3) — MLflow.

Everything about *what* produced a model is recorded here, so Step 4 can
promote a specific run to the Model Registry and a retrain triggered by
drift can be compared against the run it replaced.

What gets logged per run
------------------------
params   — every value from config that could change the result
metrics  — recall@FPR, AP, pedestrian recall, precision, per-class recall
           (flattened, because MLflow metrics must be scalars)
tags     — git SHA + dirty flag, data manifest hash, python/torch versions,
           config hash: enough to reproduce or *discredit* a run
artifacts— model checkpoint, dataset_stats.json, bronze manifest, config.yaml
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

# MLflow 3.x prints an agent hint on import; irrelevant for a pipeline
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def git_info() -> dict:
    """Current commit SHA + whether the working tree is dirty.

    A run built from uncommitted code is *not reproducible* — tagging that
    explicitly is the difference between debugging and guessing later.
    """
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                text=True, timeout=10, check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:                      # noqa: BLE001
            # git provenance is a nice-to-have: a tarball or CI checkout
            # without .git must not break tracking
            return None

    sha = _run("rev-parse", "HEAD")
    if sha is None:
        return {"git_sha": "unknown", "git_dirty": "unknown"}
    status = _run("status", "--porcelain") or ""
    return {"git_sha": sha, "git_dirty": str(bool(status.strip()))}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict -> flat {dotted.key: scalar} (MLflow params are flat)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, prefix=f"{key}."))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = str(v)
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(str(x) for x in v)
    return out


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def start_run(cfg: dict, run_name: str | None = None,
              experiment: str | None = None):
    """Open a run against the SQLite-backed tracking store.

    Artifacts go to ``mlflow.artifact_root``; run metadata goes to the
    SQLite file. Splitting them means the database stays small (fast UI)
    while checkpoints and stats remain plain files CI can cache.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    m = cfg.get("mlflow", {})
    db_path = PROJECT_ROOT / m.get("sqlite", "mlflow.db")
    artifact_root = (PROJECT_ROOT / m.get("artifact_root", "mlruns")).resolve()

    # sqlite:/// needs an ABSOLUTE path (3 slashes + drive letter)
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    exp_name = experiment or m.get("experiment", "pedestrian-detection")
    mlflow.set_experiment(exp_name)

    # pin the artifact location so runs are findable from a known folder
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is not None and not exp.artifact_location:
        client.set_experiment(
            exp.experiment_id,
            artifact_location=artifact_root.as_uri(),
        )

    tags = dict(git_info())
    tags.update({
        "python_version": sys.version.split()[0],
        "config_sha": file_hash(PROJECT_ROOT / "configs" / "config.yaml"),
        "data_manifest_sha": file_hash(PROJECT_ROOT / cfg["paths"]["bronze_manifest"]),
        "dataset_stats_sha": file_hash(PROJECT_ROOT / cfg["paths"]["gold_stats"]),
    })
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_params(mlflow, cfg: dict) -> None:
    """Log every config value that could change the outcome."""
    payload = {
        **flatten(cfg["project"], "project."),
        **flatten(cfg["model"], "model."),
        **flatten(cfg["train"], "train."),
        **flatten(cfg["gate"], "gate."),
        "dataset.label_map": ",".join(
            f"{k}:{v}" for k, v in cfg["dataset"]["label_map"].items()
        ),
        "dataset.min_box_area": str(cfg["dataset"]["min_box_area"]),
        "dataset.splits": ",".join(cfg["dataset"]["splits"]),
    }
    # MLflow rejects values > 500 chars
    payload = {k: (v[:500] if len(str(v)) > 500 else str(v))
               for k, v in payload.items()}
    mlflow.log_params(payload)


def flatten_scalars(metrics: dict | None) -> dict:
    """Flatten the metric bundle to scalars, including per-class recall.

    Shared with ``src.registry.gate``: whatever the gate measures has to be
    written into MLflow in exactly the shape the training run used, or the
    registry would show two different spellings of the same number.
    """
    if not metrics:
        return {}
    flat = {}
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            for sub, val in v.items():
                if isinstance(val, (int, float)):
                    flat[f"{k}/{sub}"] = float(val)
        elif isinstance(v, (int, float)):
            flat[k] = float(v)
    return flat


def log_metrics(mlflow, metrics: dict | None) -> None:
    """Flatten the metric bundle to scalars, including per-class recall."""
    flat = flatten_scalars(metrics)
    if flat:
        mlflow.log_metrics(flat)


def log_artifacts(mlflow, cfg: dict, checkpoint: Path | None) -> None:
    root = PROJECT_ROOT
    # config + data lineage
    mlflow.log_artifact(str(root / "configs" / "config.yaml"), "config")
    for rel in (cfg["paths"]["bronze_manifest"], cfg["paths"]["gold_stats"]):
        path = root / rel
        if path.exists():
            mlflow.log_artifact(str(path), "data")
    # model — stored under `model/` so Step 4 can register runs:/<id>/model
    if checkpoint and checkpoint.exists():
        mlflow.log_artifact(str(checkpoint), "model")
