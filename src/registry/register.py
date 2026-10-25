"""Promote a training run's model into the MLflow Model Registry.

Run:
    python -m src.registry.register --run <run_id>                 # -> @challenger
    python -m src.registry.register --run <run_id> --alias champion # -> production
    python -m src.registry.register --run <run_id> --dry-run        # inspect only

Safety rule
-----------
A run can only be registered if its recorded gate verdict is ``true``.
An untrained or degraded model reaching "Production" is exactly the failure
mode the whole pipeline exists to prevent, so this is enforced here and
again in CI (``gate.py``) rather than left to human discipline.

MLflow 3 note
-------------
Older MLflow used *stages* (None/Staging/Production). MLflow 3 deprecates
those in favour of **aliases** — mutable pointers to a version:

    models:/PedestrianDetector@champion     -> the model serving traffic
    models:/PedestrianDetector@challenger   -> the candidate being evaluated

Aliases are better here: promoting a retrain is a pointer flip, so rollback
is instant and never loses a version's history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

DEFAULT_MODEL_NAME = "PedestrianDetector"
ALLOWED_ALIASES = ("champion", "challenger")


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a run's model")
    parser.add_argument("--run", required=True, help="MLflow run id")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--alias", default="challenger",
                        choices=ALLOWED_ALIASES)
    parser.add_argument("--force", action="store_true",
                        help="register even if the gate did not pass")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen, change nothing")
    args = parser.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    m = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(
        f"sqlite:///{(PROJECT_ROOT / m.get('sqlite', 'mlflow.db')).as_posix()}"
    )
    client = MlflowClient()

    # ---- 1. the run must exist and actually have a model ---------------
    try:
        run = client.get_run(args.run)
    except Exception as exc:
        print(f"[REGISTER] ERROR — run '{args.run}' not found: {exc}")
        return 2

    if run.info.status != "FINISHED":
        print(f"[REGISTER] ERROR — run status is {run.info.status}, "
              "only FINISHED runs can be registered.")
        return 2

    artifacts = {a.path for a in client.list_artifacts(args.run)}
    uri = f"runs:/{args.run}/model"

    # MLflow 3 stores logged models in a *separate* model store
    # (mlruns/<exp>/models/m-*), so they do NOT appear in list_artifacts().
    # Resolving the URI is the only reliable existence check.
    from mlflow.models import get_model_info
    try:
        info = get_model_info(uri)
    except Exception:
        info = None

    if info is None:
        print(f"[REGISTER] ERROR — run {args.run} logged no 'model' artifact.\n"
              f"           (run-level artifacts present: {sorted(artifacts) or 'none'})\n"
              "Re-run training so the experiment logs the model.")
        return 2

    logged_classes = (info.metadata or {}).get("classes")
    if logged_classes:
        cfg_labels = list(cfg["dataset"]["label_map"].values())
        if logged_classes != cfg_labels:
            print(f"[REGISTER] ERROR — model classes {logged_classes} do not "
                  f"match config label map {cfg_labels}.\n"
                  "           Registering it would serve mislabelled detections.")
            return 2

    # ---- 2. gate verdict ------------------------------------------------
    tags = run.data.tags
    gate_passed = tags.get("gate_passed", "").lower() == "true"
    gate_failures = tags.get("gate_failures", "")

    print(f"[REGISTER] run      : {args.run}")
    print(f"[REGISTER] metrics  : recall@FPR "
          f"{run.data.metrics.get('recall_at_fpr')} | "
          f"AP {run.data.metrics.get('ap_11point')} | "
          f"ped recall {run.data.metrics.get('pedestrian_recall_at_op')}")
    print(f"[REGISTER] gate     : {'PASS' if gate_passed else 'FAIL'}"
          f"{' — ' + gate_failures if gate_failures else ''}")
    print(f"[REGISTER] alias    : {args.name}@{args.alias}")

    if not gate_passed and not args.force:
        print("\n[REGISTER] REFUSED — the evaluation gate did not pass.\n"
              "           Fix the model, or override deliberately with --force.")
        return 1

    if not gate_passed and args.force:
        print("\n[REGISTER] WARNING — registering a model that FAILED the gate "
              "(--force).")

    if args.dry_run:
        print("\n[REGISTER] dry-run — nothing written.")
        return 0

    # ---- 3. register + alias -------------------------------------------
    version = mlflow.register_model(uri, args.name)
    print(f"\n[REGISTER] created {args.name} v{version.version}")

    client.set_registered_model_alias(args.name, args.alias, version.version)
    client.set_model_version_tag(args.name, version.version,
                                 "source_run", args.run)
    client.set_model_version_tag(args.name, version.version,
                                 "gate_passed", str(gate_passed).lower())

    resolved = client.get_model_version_by_alias(args.name, args.alias)
    print(f"[REGISTER] alias {args.alias} -> v{resolved.version}")
    print(f"\nLoad it with:\n"
          f"  mlflow.pytorch.load_model('models:/{args.name}@{args.alias}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
