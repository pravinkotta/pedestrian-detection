"""__init__.py for the model package."""

from .build_model import build_model, describe, resolve_device
from .dataset import CocoDetectionDataset, collate, make_loader

__all__ = [
    "build_model",
    "describe",
    "resolve_device",
    "CocoDetectionDataset",
    "collate",
    "make_loader",
]
