"""Checkpoint save/load — one format used by train, gate, serve and CI.

A checkpoint is not just weights. Shipping weights alone means the serving
layer has to *guess* how many classes there are and in what order, and a
label-map change silently produces a model that calls cyclists "vehicles".
So we bundle the contract alongside the state dict.
"""

from __future__ import annotations

from pathlib import Path

import torch

REQUIRED_KEYS = ("model_state", "classes", "num_classes")


def save_checkpoint(path: Path, model, classes: list[str],
                    history: list, metrics: dict | None,
                    config: dict, model_info: dict, smoke: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "num_classes": len(classes) + 1,
        "history": history,
        "metrics": metrics,
        "config": config,
        "model_info": model_info,
        "torch_version": torch.__version__,
        "smoke": smoke,
    }, path)
    return path


def load_checkpoint(path: Path, device: str = "cpu", strict: bool = True):
    """Rebuild the detector from a checkpoint. Returns (model, meta)."""
    from .build_model import build_model, resolve_device

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n"
            "Train one first:\n"
            "  python -m src.experiments.run_experiment"
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise KeyError(f"{path} is not a valid checkpoint — missing {missing}")

    dev = resolve_device(device)
    # Restore the training-time geometry/head settings from the checkpoint's
    # stored config so serving sees exactly what training saw. This is why
    # save_checkpoint bundles config alongside the weights.
    model = build_model(ckpt["num_classes"], pretrained=False, device=dev,
                        cfg=ckpt.get("config"))
    model.load_state_dict(ckpt["model_state"], strict=strict)
    model.eval()

    return model, ckpt


def checkpoint_contract(path: Path) -> dict:
    """Small metadata read without building the model (cheap, for CI)."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    for k in REQUIRED_KEYS:
        if k not in ckpt:
            raise KeyError(f"{path} missing {k}")
    return {
        "classes": ckpt["classes"],
        "num_classes": ckpt["num_classes"],
        "torch_version": ckpt.get("torch_version"),
        "has_metrics": ckpt.get("metrics") is not None,
        "smoke": ckpt.get("smoke", False),
    }
