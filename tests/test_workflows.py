"""Workflow structure tests (Step 10) — the HITL chain is code, so it is tested.

These read the actual YAML the Actions service will execute. They are cheap
(~ms) and catch the two ways this stage breaks silently:

  * someone reorders `retrain.yml` so training happens BEFORE the approval
    job — the gate would become decorative;
  * someone adds a stored secret, breaking the pipeline's zero-secrets
    property that the whole EC2/GHCR design was built around.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WF_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load(name: str) -> dict:
    data = yaml.safe_load((WF_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{name} is not a mapping"
    return data


def _triggers(workflow: dict) -> dict:
    # YAML 1.1 parses the bare key `on:` as boolean True — the well-known
    # GitHub Actions gotcha; accept either spelling.
    return workflow.get("on") or workflow.get(True) or {}


def test_monitor_runs_on_a_schedule_and_on_demand():
    triggers = _triggers(_load("monitor.yml"))
    assert "schedule" in triggers, "an unattended drift monitor must be scheduled"
    # schedule is a LIST of {cron: ...} mappings
    assert "cron" in triggers["schedule"][0]
    assert "workflow_dispatch" in triggers
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert "url" in inputs and "simulate_night" in inputs


def test_monitor_dispatches_the_retrain_only_on_drift():
    """The trigger step must be conditional on DRIFT — a green check must
    never spend retraining compute."""
    text = (WF_DIR / "monitor.yml").read_text(encoding="utf-8")
    assert "gh workflow run retrain.yml" in text
    assert "steps.monitor.outputs.verdict == 'DRIFT'" in text
    # dispatch is allowed to run after the (deliberately failed) monitor step
    assert "!cancelled()" in text


def test_monitor_needs_only_actions_write_permissions():
    """Auto-dispatch must not require a PAT: permissions are scoped, and the
    only token involved is the built-in GITHUB_TOKEN."""
    wf = _load("monitor.yml")
    perms = wf.get("permissions", {})
    assert perms.get("actions") == "write"
    assert perms.get("contents") == "read"


def test_retrain_approval_job_blocks_everything_else():
    """THE HITL invariant: no retraining step may run before a human
    approves, i.e. train-and-gate must need approval."""
    jobs = _load("retrain.yml")["jobs"]
    assert jobs["approval"]["environment"]["name"] == "hitl"
    needs = jobs["train-and-gate"]["needs"]
    needs = needs if isinstance(needs, list) else [needs]  # scalar shortcut
    assert needs == ["approval"], (
        "training must sit BEHIND the HITL approval job"
    )
    publish_needs = jobs["publish"]["needs"]
    publish_needs = (publish_needs if isinstance(publish_needs, list)
                     else [publish_needs])
    assert publish_needs == ["train-and-gate"]


def test_retrain_keeps_the_ship_gate_contract():
    """An approved retrain is still a gated retrain: same checks as CI."""
    text = (WF_DIR / "retrain.yml").read_text(encoding="utf-8")
    assert "src.registry.gate" in text            # exit code decides
    assert "src.models.export_onnx --verify" in text  # ONNX parity proof
    assert "alias challenger" in text             # registry promotion


def test_no_stored_secrets_anywhere_in_the_pipeline():
    """Zero-secrets is a design property, not a vibe — enforce it.

    The only token any workflow may reference is GitHub's automatic
    GITHUB_TOKEN. (It appears in `secrets.` context and as an env var name.)
    """
    for path in sorted(WF_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # comments are not executed — prose may say "secrets."
            if "secrets." in stripped:
                assert "secrets.GITHUB_TOKEN" in stripped, (
                    f"{path.name} references a stored secret: {stripped}"
                )
