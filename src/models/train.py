"""Training loop — promoted from notebook cell 29.

Run:
    python -m src.models.train                 # full training (3 epochs, ~15-25 min CPU)
    python -m src.models.train --smoke         # 2 batches only — plumbing check for CI
    python -m src.models.train --epochs 5      # override config

Design notes
------------
* **Hyperparameters come from ``configs/config.yaml``**, not constants —
  so Step 3 (MLflow) can log exactly what was run, and a retrain triggered
  by drift can pass different values without editing code.
* **``--smoke`` mode** trains on 2 batches then exits. This is what CI runs
  on every push: it catches broken imports, bad tensor shapes and dataset
  regressions in ~1 minute instead of 25.
* **Seeded** with ``project.seed`` so two runs on the same data are
  comparable — without this, experiment tracking compares noise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_loaders(cfg: dict, val_limit: int | None = None):
    """Build train + val DataLoaders.

    ``val_limit`` caps the validation set — used by ``--smoke`` so CI still
    exercises the *entire* code path (inference -> matching -> metrics ->
    MLflow logging) without paying for a full 80-image evaluation.
    """
    from torch.utils.data import DataLoader, Subset

    from .dataset import CocoDetectionDataset, collate

    p = cfg["paths"]
    nw = cfg["train"]["num_workers"]

    def dataset(split: str) -> CocoDetectionDataset:
        return CocoDetectionDataset(
            PROJECT_ROOT / p["coco_dir"] / f"{split}.json",
            PROJECT_ROOT / p["raw_images"] / split,
        )

    train_ds = dataset("train")
    val_ds = dataset("val")
    if val_limit is not None:
        val_ds = Subset(val_ds, range(min(val_limit, len(val_ds))))

    train = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                       shuffle=True, collate_fn=collate, num_workers=nw)
    val = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"] * 2,
                     shuffle=False, collate_fn=collate, num_workers=nw)
    return train, val, train_ds.classes


def train(cfg: dict, epochs: int | None = None, smoke: bool = False,
          log=print) -> dict:
    from .build_model import build_model, describe, resolve_device
    from .evaluate import evaluate

    t_cfg = cfg["train"]
    epochs = epochs if epochs is not None else int(t_cfg["epochs"])
    if smoke:
        epochs = 1

    set_seed(cfg["project"]["seed"])
    device = resolve_device("auto")

    # In smoke mode, evaluate on only a handful of val images so the full
    # metrics path runs cheaply.
    train_loader, val_loader, classes = build_loaders(
        cfg, val_limit=4 if smoke else None
    )
    num_classes = len(classes) + 1      # + background

    model = build_model(num_classes, pretrained=True, device=device)
    info = describe(model)
    log(f"device={info['device']} | classes={classes} "
        f"({num_classes} incl. background) | params={info['params_millions']}M")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        momentum=float(t_cfg["momentum"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )

    history: list[dict] = []
    best: dict = {}
    limit = 2 if smoke else len(train_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        run: defaultdict = defaultdict(float)
        t0 = time.time()

        for it, (imgs, targets) in enumerate(train_loader, 1):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(v for v in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in loss_dict.items():
                run[k] += v.item()
            run["total"] += loss.item()

            if it % max(t_cfg["log_every"], 1) == 0:
                log(f"  epoch {epoch} [{it}/{len(train_loader)}] "
                    f"loss {run['total']/it:.4f}")
            if smoke and it >= limit:
                break

        n_iter = min(limit, len(train_loader)) if smoke else len(train_loader)
        rec = {k: v / n_iter for k, v in run.items()}
        rec["epoch"], rec["time_s"] = epoch, round(time.time() - t0, 1)
        rec["iterations"] = n_iter

        log(f"epoch {epoch}: total {rec['total']:.4f} "
            f"(rpn-cls {rec['loss_objectness']:.4f}, "
            f"rpn-box {rec['loss_rpn_box_reg']:.4f}, "
            f"cls {rec['loss_classifier']:.4f}, "
            f"box {rec['loss_box_reg']:.4f}) [{rec['time_s']}s]")
        history.append(rec)

        if smoke:
            metrics = evaluate(model, val_loader, device, classes,
                               fpr_budget=cfg["gate"]["fpr_budget"])
            metrics.pop("_raw", None)
            best = metrics
            log(f"  smoke val: AP {metrics['ap_11point']} | "
                f"recall@FPR {metrics['recall_at_fpr']} | "
                f"ped recall {metrics['pedestrian_recall_at_op']}")
            break

        if t_cfg.get("eval_each_epoch", False) and epoch == epochs:
            metrics = evaluate(model, val_loader, device, classes,
                               fpr_budget=cfg["gate"]["fpr_budget"])
            metrics.pop("_raw", None)
            log(f"  val: AP {metrics['ap_11point']} | "
                f"recall@FPR {metrics['recall_at_fpr']} | "
                f"ped recall {metrics['pedestrian_recall_at_op']}")
            best = metrics

    checkpoint = {
        "model_state": model.state_dict(),
        "classes": classes,
        "num_classes": num_classes,
        "history": history,
        "metrics": best or None,
        "config": {k: cfg[k] for k in ("model", "train", "gate")},
        "torch_version": torch.__version__,
        "model_info": info,
        "smoke": smoke,
    }

    if not smoke:
        out_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "model.pt"
        torch.save(checkpoint, out_path)
        log(f"checkpoint -> {out_path.relative_to(PROJECT_ROOT)} "
            f"({out_path.stat().st_size/1e6:.1f} MB)")

    return {"history": history, "metrics": best or None,
            "model_info": info, "classes": classes}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN detector")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="train on 2 batches only (CI plumbing check)")
    parser.add_argument("--json", action="store_true",
                        help="print a machine-readable summary at the end")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    torch.set_num_threads(os.cpu_count() or 4)

    result = train(cfg, epochs=args.epochs, smoke=args.smoke)

    if args.json:
        print(json.dumps({
            "history": result["history"],
            "metrics": result["metrics"],
            "classes": result["classes"],
            "model_info": result["model_info"],
        }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
