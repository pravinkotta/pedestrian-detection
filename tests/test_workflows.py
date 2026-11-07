"""Workflow structure tests (Step 10) — the HITL chain is code, so it is tested.

These read the actual YAML the Actions service will execute. They are cheap
(~ms) and catch the two ways this stage breaks silently:

  * someone reorders `retrain.yml` so training happens BEFORE the approval
    job — the gate would become decorative;
  * someone adds a stored secret, breaking the pipeline's zero-secrets
    property that the whole EC2/GHCR design was built around.
"""

from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
# Deployment-bootstrap properties — the day-2 contract (start -> API live,
# stop -> nothing runs) encoded as test failures instead of as documentation.
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
_SKIP_DIRS = {".git", "mlruns", "data", "models", "outputs", "node_modules",
              "__pycache__", ".pytest_cache"}


def test_container_restart_policy_is_always():
    """The API must come back on every boot, even after a manual `docker stop`.

    `unless-stopped` would honour a deliberate stop across a reboot — for an
    unattended API that means "someone debugged it once and it stayed down".
    Cost is gated by *instance* state, not by this flag: a stopped instance
    runs nothing, a started one must serve unattended.
    """
    text = (REPO / "deploy" / "user-data.sh").read_text(encoding="utf-8")
    run_lines = [ln for ln in text.splitlines() if "docker run" in ln]
    assert run_lines, "user-data.sh must create the container"
    assert "--restart always" in run_lines[0], (
        "the deployed container must use --restart always"
    )
    # and the loop that creates it must itself be enabled at boot
    assert "systemctl enable --now pedestrian-deployer" in text


def test_shell_scripts_are_lf_only():
    """cloud-init pipes user-data.sh straight into bash on the EC2 box.

    A CRLF checkout (core.autocrlf=true is the Windows default) would upload a
    script whose lines end in `\\r` → the bootstrap dies with
    `\\r: command not found` before docker is even installed. `.gitattributes`
    pins eol=lf; this test fails if that protection ever disappears.
    """
    scripts = [p for p in REPO.rglob("*.sh") if not (set(p.parts) & _SKIP_DIRS)]
    assert scripts, "expected at least deploy/user-data.sh"
    for path in scripts:
        assert b"\r\n" not in path.read_bytes(), (
            f"{path.relative_to(REPO)} has CRLF line endings"
        )


# ---------------------------------------------------------------------------
# DuckDNS properties — a *permanent* domain name for the ephemeral-public-IP
# instance. The token only ever lives on the box; these tests keep it that way.
# ---------------------------------------------------------------------------

DUCKDNS_DIR = REPO / "deploy"
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{12}\b"
)


def test_no_duckdns_token_is_committed():
    """The DuckDNS token authorises anyone holding it to repoint our DNS.

    It exists ONLY in /etc/duckdns/config on the box (mode 600) — never in
    user-data.sh, never in a workflow, never in this test itself. The regex
    also catches the real-world failure of pasting a token into a doc.
    """
    for path in REPO.rglob("*"):
        if not path.is_file() or set(path.parts) & _SKIP_DIRS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        match = _UUID_RE.search(text)
        assert not match, (
            f"{path.relative_to(REPO)} contains a UUID that looks like a "
            f"DuckDNS token ('{match.group(0)[:8]}...') — tokens live on the "
            "box only"
        )


def test_user_data_ships_only_empty_duckdns_placeholders():
    """First-boot self-install must be opt-in and secret-free.

    An operator filling the two values in their LOCAL user-data copy before
    `run-instances` gets a self-installing updater; the committed file must
    keep both empty so `git push` can never leak a token.
    """
    text = (DUCKDNS_DIR / "user-data.sh").read_text(encoding="utf-8")
    assert 'DUCKDNS_DOMAIN=""' in text, "placeholder must be empty in the repo"
    assert 'DUCKDNS_TOKEN=""' in text, "placeholder must be empty in the repo"
    assert ("if [ -n \"$DUCKDNS_DOMAIN\" ] && [ -n \"$DUCKDNS_TOKEN\" ]; then"
            in text), "install must be skipped while placeholders are empty"


def test_duckdns_updater_sends_the_explicit_instance_ip():
    """If the updater ever trusted auto-detection, calling it from a laptop
    (or from a CI runner) would silently repoint the domain to THAT machine's
    IP. The canonical source is the instance metadata service (IMDSv2), sent
    explicitly on every update.
    """
    text = (DUCKDNS_DIR / "duckdns.sh").read_text(encoding="utf-8")
    assert "169.254.169.254" in text, "IP must come from the instance metadata service"
    assert "X-aws-ec2-metadata-token" in text, "IMDSv2 (not the legacy v1 hop)"
    assert "ip=${IP}" in text, "the update request must carry the explicit IP"

    setup = (DUCKDNS_DIR / "duckdns-setup.sh").read_text(encoding="utf-8")
    assert "chmod 600 /etc/duckdns/config" in setup, (
        "the token file must be root-only"
    )

