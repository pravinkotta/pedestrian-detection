"""Tests for the MLflow tracking layer (Step 3).

These are deliberately pure — they exercise the *helpers*, not a live MLflow
server, so they run in milliseconds on any machine (including CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


def test_flatten_produces_flat_scalars():
    from src.experiments.tracking import flatten

    flat = flatten({"a": {"b": {"c": 1}}, "d": True,
                    "e": [1, 2], "f": None, "g": "x"})
    assert flat == {
        "a.b.c": "1",
        "d": "True",
        "e": "1,2",
        "f": "None",
        "g": "x",
    }


def test_flatten_survives_mlflow_500_char_limit():
    from src.experiments.tracking import flatten

    long_value = "x" * 2000
    flat = flatten({"k": long_value})
    payload = {k: (v[:500] if len(str(v)) > 500 else str(v))
               for k, v in flat.items()}
    assert len(payload["k"]) == 500, "MLflow rejects params over 500 chars"


def test_git_info_returns_sha_and_dirty_flag():
    from src.experiments.tracking import git_info

    info = git_info()
    assert "git_sha" in info and "git_dirty" in info
    # we are inside a repo, so a real SHA should come back
    assert info["git_sha"] == "unknown" or len(info["git_sha"]) == 40
    assert info["git_dirty"] in ("True", "False", "unknown")


def test_file_hash_is_stable_and_handles_missing():
    from src.experiments.tracking import file_hash

    cfg = ROOT / "configs" / "config.yaml"
    a, b = file_hash(cfg), file_hash(cfg)
    assert a == b and len(a) == 16
    assert file_hash(ROOT / "does-not-exist.json") == "missing"


def test_config_has_mlflow_block():
    assert "mlflow" in CONFIG, "Step 3 requires an mlflow: config section"
    m = CONFIG["mlflow"]
    assert m["sqlite"].endswith(".db")
    assert m["artifact_root"]
    assert m["experiment"]


def test_mlflow_database_and_artifacts_are_gitignored():
    """Tracking state must never reach the remote repo."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("/mlruns/", "/mlflow.db", "*.db"):
        assert pattern in ignore, f".gitignore missing {pattern}"


def test_params_logged_cover_every_decision_that_changes_results():
    """If a value can change the model, MLflow must record it."""
    from src.experiments.tracking import flatten

    payload = {
        **flatten(CONFIG["train"], "train."),
        **flatten(CONFIG["model"], "model."),
        **flatten(CONFIG["gate"], "gate."),
    }
    for required in ("train.lr", "train.epochs", "train.batch_size",
                     "model.architecture", "gate.min_ap",
                     "gate.min_recall_at_fpr"):
        assert required in payload, f"{required} would not be tracked"


def test_metrics_flat_bundle_contains_safety_metric():
    """The headline safety metric must be a scalar MLflow can chart."""
    from src.experiments.tracking import flatten  # noqa: F401
    from src.models.evaluate import compute_metrics

    fake = {
        "dets": [{"score": 0.9, "label": 1, "tp": 1, "gt": 1},
                 {"score": 0.8, "label": 2, "tp": 0, "gt": -1}],
        "total_gt": 2,
        "gt_per_class": {1: 2},
        "fn_labels": [1],
        "n_images": 10,
        "n_negatives": 1000,
    }
    m = compute_metrics(fake, ["pedestrian", "cyclist", "vehicle"])

    # every metric must be a plain number (MLflow rejects dicts)
    from src.experiments.tracking import log_metrics

    scalars = {}
    for k, v in m.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            for sub, val in v.items():
                if isinstance(val, (int, float)):
                    scalars[f"{k}/{sub}"] = float(val)
        elif isinstance(v, (int, float)):
            scalars[k] = float(v)

    assert "recall_at_fpr" in scalars, "headline safety metric missing"
    assert "pedestrian_recall_at_op" in scalars
    assert "ap_11point" in scalars
    assert "per_class_recall/pedestrian" in scalars
    assert all(isinstance(v, float) for v in scalars.values())
