"""__init__.py for the model package."""

from .build_model import build_model, describe, resolve_device
from .dataset import CocoDetectionDataset, collate, make_loader

__all__ = [
    "CocoDetectionDataset",
    "build_model",
    "collate",
    "describe",
    "make_loader",
    "resolve_device",
]
