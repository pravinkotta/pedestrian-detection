"""Smoke tests for the data layers (Steps 1a-1c).

Run:
    pytest -q

These are cheap on purpose — they run in CI on every push and catch the
"dataset silently got corrupted" class of bug before it wastes a GPU hour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


def _load(rel: str) -> dict:
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not generated yet — run the data pipeline first")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Bronze
# --------------------------------------------------------------------------
def test_bronze_manifest_exists_and_matches_config():
    manifest = _load(CONFIG["paths"]["bronze_manifest"])
    label_map = CONFIG["dataset"]["label_map"]

    assert manifest["label_map"] == {str(k): v for k, v in label_map.items()}
    assert manifest["totals"]["images"] == 380, "dataset size changed unexpectedly"
    # 881 + 288 + 220
    assert manifest["totals"]["annotations"] == 1389


def test_every_split_present_in_bronze():
    manifest = _load(CONFIG["paths"]["bronze_manifest"])
    for split in CONFIG["dataset"]["splits"]:
        assert manifest["splits"][split]["images"] > 0, f"{split} split is empty"


# --------------------------------------------------------------------------
# Silver (COCO)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_coco_is_valid(split: str):
    coco = _load(f"{CONFIG['paths']['coco_dir']}/{split}.json")

    cat_ids = {c["id"] for c in coco["categories"]}
    assert 0 not in cat_ids, "COCO category id 0 is reserved for background"

    img_ids = {i["id"] for i in coco["images"]}
    assert len(img_ids) == len(coco["images"]), "duplicate image ids"

    ann_ids = set()
    for ann in coco["annotations"]:
        assert ann["image_id"] in img_ids, "annotation points at unknown image"
        assert ann["category_id"] in cat_ids, "annotation has unknown category"
        assert ann["id"] not in ann_ids, "duplicate annotation id"
        ann_ids.add(ann["id"])

        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0, f"non-positive box size: {ann['bbox']}"
        assert abs(ann["area"] - w * h) < 1e-6, "area does not match bbox"
        assert x >= 0 and y >= 0, "box starts outside the image"

        # box must fit inside its image
        img = next(i for i in coco["images"] if i["id"] == ann["image_id"])
        assert x + w <= img["width"] + 1e-6, "box overflows image width"
        assert y + h <= img["height"] + 1e-6, "box overflows image height"


def test_coco_counts_match_bronze():
    manifest = _load(CONFIG["paths"]["bronze_manifest"])
    total = 0
    for split in CONFIG["dataset"]["splits"]:
        coco = _load(f"{CONFIG['paths']['coco_dir']}/{split}.json")
        assert len(coco["images"]) == manifest["splits"][split]["images"], (
            f"{split}: image count drifted from the Bronze manifest"
        )
        total += len(coco["annotations"])
    assert total == manifest["totals"]["annotations"]


# --------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------
def test_gold_stats_baseline_exists():
    stats = _load(CONFIG["paths"]["gold_stats"])
    splits = {s["split"] for s in stats["splits"]}
    assert {"train", "val"} <= splits, "Gold must produce train AND val stats"

    train = next(s for s in stats["splits"] if s["split"] == "train")
    assert set(stats["label_map"] if "label_map" in stats else
               {c["name"] for c in stats["categories"]}) == set(
        CONFIG["dataset"]["label_map"].values()
    ), "category set drifted from config"
    assert train["num_annotations"] > 0


def test_gate_thresholds_are_achievable():
    """Sanity-check the ship/no-ship gate is not set to an impossible value."""
    gate = CONFIG["gate"]
    assert 0 < gate["fpr_budget"] < 1
    assert 0 < gate["min_recall_at_fpr"] <= 1
    assert 0 < gate["min_ap"] <= 1
    assert gate["min_pedestrian_recall"] >= gate["min_recall_at_fpr"], (
        "pedestrian recall gate should be at least as strict — it is the "
        "safety-critical class"
    )
