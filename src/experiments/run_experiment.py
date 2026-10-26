"""Run one MLflow-tracked training experiment.

Run:
    python -m src.experiments.run_experiment --smoke      # CI: full tracking path, ~30 s
    python -m src.experiments.run_experiment              # real run, ~15-25 min CPU
    python -m src.experiments.run_experiment --run-name "baseline-v1"

Then inspect:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    -> http://127.0.0.1:5000

This is a thin orchestration layer: ``src.models.train`` owns the training
logic and knows nothing about MLflow. Keeping them separate means you can
train without tracking (CI, quick experiments) and swap MLflow for something
else without touching the model code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def main() -> int:
    import mlflow

    from src.experiments.tracking import (
        log_artifacts,
        log_metrics,
        log_params,
        start_run,
    )
    from src.models.train import load_config, train

    parser = argparse.ArgumentParser(description="Tracked training run")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="2 training batches + tiny eval: exercises the "
                             "whole tracking path in seconds")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-track", action="store_true",
                        help="train without MLflow")
    parser.add_argument("--tags", nargs="*", default=[],
                        help="extra tags, key=value")
    parser.add_argument("--no-model", action="store_true",
                        help="skip logging the registerable model artifact")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.no_track:
        result = train(cfg, epochs=args.epochs, smoke=args.smoke)
        print(json.dumps(result["metrics"], indent=2, default=str))
        return 0

    extra_tags = {}
    for kv in args.tags:
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra_tags[k] = v

    run_name = args.run_name or ("smoke-test" if args.smoke else None)

    with start_run(cfg, run_name=run_name) as run:
        print(f"\nMLflow run: {run.info.run_id}\n")

        # 1. what we are about to run
        log_params(mlflow, cfg)
        if extra_tags:
            mlflow.set_tags(extra_tags)
        if args.epochs:
            mlflow.log_param("train.epochs_override", args.epochs)
        if args.smoke:
            mlflow.set_tag("smoke", "true")

        # 2. train (prints progress; returns history + metrics)
        result = train(cfg, epochs=args.epochs, smoke=args.smoke)

        # 3. what we got
        log_metrics(mlflow, result["metrics"])

        # loss curve as a per-epoch metric series is more useful than an artifact
        for rec in result["history"]:
            mlflow.log_metrics(
                {"loss/total": rec["total"],
                 "loss/classifier": rec["loss_classifier"],
                 "loss/box_reg": rec["loss_box_reg"],
                 "loss/rpn_objectness": rec["loss_objectness"],
                 "loss/rpn_box_reg": rec["loss_rpn_box_reg"],
                 "epoch_time_s": rec["time_s"]},
                step=int(rec["epoch"]),
            )

        # 4. what to keep
        checkpoint = PROJECT_ROOT / cfg["paths"]["model_dir"] / "model.pt"
        log_artifacts(mlflow, cfg, None if args.smoke else checkpoint)

        # 4b. registerable model artifact — `runs:/<id>/model` is what
        #     src.registry.register promotes into the Model Registry.
        #     `metadata` carries the label map so the serving layer never
        #     has to guess class order.
        if result.get("model") is not None and not args.no_model:
            import mlflow.pytorch
            # MLflow 3 defaults serialization_format='pt2' (torch.compile
            # traced graph), which requires a fixed input_example. A detector
            # takes variable-size images and returns list[dict], so it cannot
            # be traced — 'pickle' is the correct format here.
            mlflow.pytorch.log_model(
                result["model"],
                name="model",
                serialization_format="pickle",
                metadata={
                    "classes": result["classes"],
                    "architecture": cfg["model"]["architecture"],
                    "task": "object-detection",
                    "metric": "recall_at_fpr",
                },
                pip_requirements=["torch", "torchvision", "pillow", "numpy"],
            )
            print(f"  logged model artifact -> runs:/{run.info.run_id}/model")

        # 5. did it pass the gate?  (recorded, not yet enforced — Step 4 enforces)
        from src.models.evaluate import check_gate
        if result["metrics"]:
            passed, failures = check_gate(result["metrics"], cfg["gate"])
            mlflow.set_tag("gate_passed", str(passed).lower())
            if failures:
                mlflow.set_tag("gate_failures", "; ".join(failures))
                print("\n[GATE] would FAIL:\n" +
                      "\n".join(f"  - {f}" for f in failures))
            else:
                print("\n[GATE] would PASS")

        print(f"\nMLflow run finished: {run.info.run_id}")
        print(f"  status={run.info.status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
