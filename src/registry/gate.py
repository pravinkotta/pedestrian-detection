"""Evaluation gate — decide ship / no-ship from measured metrics.

Run:
    python -m src.registry.gate                         # models/model.pt on val
    python -m src.registry.gate --split test            # hold-out evaluation
    python -m src.registry.gate --checkpoint models/old.pt --json
    python -m src.registry.gate --mlflow-run <run_id>   # mirror metrics to MLflow

Exit codes (this is the contract CI depends on):

    0  PASS  — every gate metric cleared its threshold
    1  FAIL  — at least one metric below threshold -> do NOT deploy
    2  ERROR — missing checkpoint/data/broken config (a bug, not a bad model)

Distinguishing 1 from 2 matters: a *bad model* should stop the pipeline with a
clear message, but a *crashed script* must not be mistaken for "model rejected".
GitHub Actions treats both as failure, but a human reading the log needs to
know which happened.

This is the automated half of your mentor's HITL gate. High-risk changes
(manual approval) are Step 10 — this is the threshold check that runs on
every push.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def _log_to_mlflow(cfg: dict, run_id: str, metrics: dict, report: dict) -> None:
    """Mirror the gate's measurements onto an existing MLflow run.

    The Model Registry describes a run by the metrics *logged on it*, so when
    the evaluation contract changes the run's numbers must be corrected too —
    otherwise ``@champion`` is advertised by figures no code path produces
    any more. (What triggered this: the RPN proposal caps were dead config
    until ``apply_rpn_caps``; with them effective the eval runs at 200
    proposals instead of 1000 and six metrics moved.)

    Tracking is a record, never the verdict: a logging failure prints a
    warning and leaves the gate's exit code untouched.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from src.experiments.tracking import flatten_scalars

        db = PROJECT_ROOT / cfg.get("mlflow", {}).get("sqlite", "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
        client = MlflowClient()
        client.get_run(run_id)  # surfaces a bad id with MLflow's own message
        # module-level API: MlflowClient has no plural log_metrics(), and the
        # active-run context does not apply to an arbitrary run_id
        flat = flatten_scalars(metrics)
        if flat:
            mlflow.log_metrics(flat, run_id=run_id)
        for key, value in {
            "gate/verdict": "PASS" if report["passed"] else "FAIL",
            "gate/split": report["split"],
            "gate/evaluated_utc": report["generated_utc"],
        }.items():
            client.set_tag(run_id, key, value)
        print(f"  mirrored onto MLflow run {run_id}")
    except Exception as exc:  # noqa: BLE001 — reporting must not change the verdict
        print(f"  [warn] MLflow mirror skipped ({run_id}): {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation gate (ship / no-ship)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None,
                        help="defaults to <paths.model_dir>/model.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default="outputs/gate_report.json")
    parser.add_argument("--json", action="store_true",
                        help="print the full report as JSON")
    parser.add_argument("--mlflow-run", default=None, metavar="RUN_ID",
                        help="also write the measured metrics onto this MLflow "
                             "run, so registry numbers always match the eval "
                             "config actually in force")
    args = parser.parse_args()

    try:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        ckpt = (Path(args.checkpoint) if args.checkpoint
                else PROJECT_ROOT / cfg["paths"]["model_dir"] / "model.pt")

        if not ckpt.exists():
            print(f"[GATE] ERROR — checkpoint not found: {ckpt}\n"
                  "Train first:  python -m src.experiments.run_experiment")
            return EXIT_ERROR

        import torch
        torch.set_num_threads(torch.get_num_threads() or 4)

        from src.models.checkpoint import load_checkpoint
        from src.models.dataset import make_loader
        from src.models.evaluate import check_gate, evaluate

        p = cfg["paths"]
        ann = PROJECT_ROOT / p["coco_dir"] / f"{args.split}.json"
        if not ann.exists():
            print(f"[GATE] ERROR — annotations missing: {ann}\n"
                  "Run:  python -m src.data.convert_to_coco")
            return EXIT_ERROR

        loader = make_loader(ann, PROJECT_ROOT / p["raw_images"] / args.split,
                             batch_size=cfg["train"]["batch_size"] * 2,
                             shuffle=False, num_workers=0)
        model, meta = load_checkpoint(ckpt, device=args.device)

        classes = meta["classes"]
        if list(loader.dataset.classes) != classes:
            print(f"[GATE] ERROR — checkpoint classes {classes} do not match "
                  f"dataset {list(loader.dataset.classes)}. "
                  "The label map changed between train and serve.")
            return EXIT_ERROR

        print(f"[GATE] evaluating {ckpt.name} on '{args.split}' "
              f"({len(loader.dataset)} images, classes={classes})")
        metrics = evaluate(model, loader, args.device, classes,
                           fpr_budget=cfg["gate"]["fpr_budget"])
        metrics.pop("_raw", None)
        passed, failures = check_gate(metrics, cfg["gate"])

    except FileNotFoundError as exc:
        print(f"[GATE] ERROR — {exc}")
        return EXIT_ERROR
    except KeyError as exc:
        print(f"[GATE] ERROR — {exc}")
        return EXIT_ERROR

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt),
        "split": args.split,
        "passed": passed,
        "failures": failures,
        "gate": cfg["gate"],
        "metrics": metrics,
        "checkpoint_classes": classes,
    }

    out = PROJECT_ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.mlflow_run:
        _log_to_mlflow(cfg, args.mlflow_run, metrics, report)

    print()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        rows = [
            ("recall@FPR", "recall_at_fpr", "min_recall_at_fpr"),
            ("AP (11-pt)", "ap_11point", "min_ap"),
            ("ped recall", "pedestrian_recall_at_op", "min_pedestrian_recall"),
        ]
        for label, mk, gk in rows:
            actual, need = metrics.get(mk), cfg["gate"].get(gk)
            mark = "OK " if (actual is not None and actual >= need) else "FAIL"
            print(f"  [{mark}] {label:12s} {actual}  (>= {need})")

        print(f"\n  threshold @ FPR<={cfg['gate']['fpr_budget']}: "
              f"{metrics['operating_threshold']:.3f}  "
              f"(precision {metrics['precision_at_op']}, "
              f"FPR {metrics['fpr_at_op']})")
        print(f"  per-class recall: {metrics['per_class_recall']}")
        print(f"  report -> {out.relative_to(PROJECT_ROOT)}")

    print(f"\n[GATE] {'PASS' if passed else 'FAIL'}")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return EXIT_PASS if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
