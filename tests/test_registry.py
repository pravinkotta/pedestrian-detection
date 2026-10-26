"""Tests for the evaluation gate and model registry (Step 4).

Exit codes are the contract CI relies on, so they are asserted directly:

    0 PASS | 1 FAIL (bad model) | 2 ERROR (missing/broken input)

Keeping 1 and 2 distinct lets a pipeline tell "this model is not good
enough" apart from "something is broken" — a distinction that matters
when you are paging someone at 3am.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        encoding="utf-8", errors="replace", check=False,
    )


# --------------------------------------------------------------------------
# checkpoint format
# --------------------------------------------------------------------------
def _tiny_checkpoint(tmp_path: Path) -> Path:
    from src.models.build_model import build_model
    from src.models.checkpoint import save_checkpoint

    model = build_model(4, pretrained=False, device="cpu")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model,
                    classes=["pedestrian", "cyclist", "vehicle"],
                    history=[], metrics=None,
                    config=CONFIG["gate"], model_info={})
    return path


def test_checkpoint_roundtrip_preserves_label_map(tmp_path):
    from src.models.checkpoint import load_checkpoint

    path = _tiny_checkpoint(tmp_path)
    model, meta = load_checkpoint(path)

    # the label map is the contract between train and serve — losing it
    # means a model that confidently calls cyclists "vehicles"
    assert meta["classes"] == ["pedestrian", "cyclist", "vehicle"]
    assert meta["num_classes"] == 4

    head = model.roi_heads.box_predictor.cls_score.out_features
    assert head == meta["num_classes"], "head width must match the class list"


def test_checkpoint_contract_reads_metadata_without_building(tmp_path):
    from src.models.checkpoint import checkpoint_contract

    path = _tiny_checkpoint(tmp_path)
    meta = checkpoint_contract(path)
    assert meta["classes"] == ["pedestrian", "cyclist", "vehicle"]
    assert meta["has_metrics"] is False


def test_checkpoint_rejects_a_file_without_required_keys(tmp_path):
    import torch

    from src.models.checkpoint import load_checkpoint

    bad = tmp_path / "not_a_checkpoint.pt"
    torch.save({"weights": [1, 2, 3]}, bad)

    with pytest.raises(KeyError, match="missing"):
        load_checkpoint(bad)


def test_checkpoint_missing_file_gives_actionable_error(tmp_path):
    from src.models.checkpoint import load_checkpoint

    with pytest.raises(FileNotFoundError, match="Train one first"):
        load_checkpoint(tmp_path / "absent.pt")


# --------------------------------------------------------------------------
# gate exit codes
# --------------------------------------------------------------------------
def test_gate_exits_2_when_checkpoint_missing(tmp_path):
    """ERROR (2) must not be confused with a model that merely failed (1)."""
    res = _run_cli("src.registry.gate",
                   "--checkpoint", str(tmp_path / "nope.pt"))
    assert res.returncode == 2, res.stdout + res.stderr
    assert "checkpoint not found" in (res.stdout + res.stderr)


def test_gate_exits_2_when_annotations_missing(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path)
    res = _run_cli("src.registry.gate",
                   "--checkpoint", str(ckpt),
                   "--split", "val",
                   "--config", str(tmp_path / "no_config.yaml"))
    assert res.returncode == 2
    assert "ERROR" in res.stdout


def test_gate_report_is_written_when_it_runs():
    """The report from a previous real gate run must be well-formed."""
    report = ROOT / "outputs" / "gate_report.json"
    if not report.exists():
        pytest.skip("no gate report yet — run: python -m src.registry.gate")
    data = json.loads(report.read_text(encoding="utf-8"))

    assert data["passed"] in (True, False)
    assert data["split"] in ("train", "val", "test")
    assert isinstance(data["failures"], list)
    assert "recall_at_fpr" in data["metrics"]
    assert data["checkpoint_classes"] == list(
        CONFIG["dataset"]["label_map"].values()
    )
    if not data["passed"]:
        assert data["failures"], "a failing report must say why"


# --------------------------------------------------------------------------
# register
# --------------------------------------------------------------------------
def test_register_exits_2_for_unknown_run():
    res = _run_cli("src.registry.register", "--run", "does-not-exist")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "not found" in (res.stdout + res.stderr)


def test_register_rejects_bad_alias():
    res = _run_cli("src.registry.register", "--run", "x",
                   "--alias", "production")
    assert res.returncode != 0
    assert "invalid choice" in (res.stdout + res.stderr).lower()


def _sqlite_uri() -> str:
    """sqlite:/// + ABSOLUTE posix path (as_uri() would yield file:/// ...)."""
    return f"sqlite:///{(ROOT / CONFIG['mlflow']['sqlite']).as_posix()}"


def _find_failed_run_with_model():
    """Return (client, run_id) for a gate-failed run that logged a model.

    Not every failed run has a model (e.g. ``--no-model`` smoke runs), and
    register correctly rejects those with exit 2 — so the test must pick one
    that reaches the gate check, which is the behaviour under test.
    """
    import mlflow
    from mlflow.models import get_model_info

    mlflow.set_tracking_uri(_sqlite_uri())
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(CONFIG["mlflow"]["experiment"])
    if exp is None:
        return None, None, None

    runs = client.search_runs(exp.experiment_id,
                              filter_string="tags.gate_passed = 'false'")
    for run in runs:
        run_id = run.info.run_id
        try:
            get_model_info(f"runs:/{run_id}/model")
        except Exception:                      # noqa: BLE001, S112
            continue
        return client, run_id, exp
    return client, None, exp


def test_register_refuses_run_that_failed_gate():
    """The core safety rule: a failing model cannot reach the registry."""
    _client, run_id, _ = _find_failed_run_with_model()
    if not run_id:
        pytest.skip("no gate-failed run WITH a model recorded yet")

    res = _run_cli("src.registry.register", "--run", run_id)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "REFUSED" in res.stdout


def test_register_dry_run_changes_nothing():
    client, run_id, _ = _find_failed_run_with_model()
    if not run_id:
        pytest.skip("no gate-failed run WITH a model recorded yet")

    def versions() -> set:
        if not _model_exists(client):
            return set()
        return {v.name for v in
                client.search_model_versions("name='PedestrianDetector'")}

    before = versions()

    res = _run_cli("src.registry.register", "--run", run_id,
                   "--force", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "dry-run" in res.stdout

    assert versions() == before, "dry-run must not create model versions"


def _model_exists(client) -> bool:
    try:
        client.get_registered_model("PedestrianDetector")
        return True
    except Exception:                          # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# config contract
# --------------------------------------------------------------------------
def test_registry_uses_mlflow_aliases_not_stages():
    """MLflow 3 replaced Staging/Production stages with aliases."""
    src = (ROOT / "src" / "registry" / "register.py").read_text(encoding="utf-8")
    assert "set_registered_model_alias" in src, "use MLflow 3 aliases"
    assert "challenger" in src and "champion" in src


def test_gate_distinguishes_fail_from_error_in_docstring():
    src = (ROOT / "src" / "registry" / "gate.py").read_text(encoding="utf-8")
    assert "EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2" in src
