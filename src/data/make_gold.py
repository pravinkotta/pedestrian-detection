"""Gold layer: final training-ready splits + dataset statistics.

Run:
    python -m src.data.make_gold

Reads  : data/interim/annotations/{train,val,test}.json   (Silver)
Writes : data/processed/annotations/{train,val,test}.json (Gold, splits locked)
         data/processed/dataset_stats.json                (goes to MLflow)

Two jobs:

1. **Deterministic split.**  CityPersons ships train *and* val annotations, so
   normally this is just a copy — but if ``val.json`` is missing, we carve a
   validation set out of train using a fixed seed, so the split is reproducible
   across machines and CI runs.

2. **Dataset statistics.**  Class counts, box-size histogram, aspect ratios.
   These numbers are what the drift monitor (Evidently) will later compare
   *live traffic* against. Without a recorded baseline there is nothing to
   detect drift *from*.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_coco(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[GOLD] missing {path} — run the Silver step first:\n"
                         "        python -m src.data.convert_citypersons --split train")
    return json.loads(path.read_text(encoding="utf-8"))


def split_train_val(train: dict, val_fraction: float, seed: int) -> tuple[dict, dict]:
    """Hold out ``val_fraction`` of images (stratified by 'has small boxes')."""
    rng = random.Random(seed)
    img_ids = [im["id"] for im in train["images"]]
    rng.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * val_fraction))
    val_ids = set(img_ids[:n_val])

    def subset(ids: set[int]) -> dict:
        return {
            "images": [im for im in train["images"] if im["id"] in ids],
            "annotations": [a for a in train["annotations"] if a["image_id"] in ids],
            "categories": train["categories"],
        }

    return subset(set(img_ids) - val_ids), subset(val_ids)


def box_size_bucket(area: float) -> str:
    if area < 64**2:
        return "tiny"
    if area < 128**2:
        return "small"
    if area < 256**2:
        return "medium"
    return "large"


def compute_stats(coco: dict, name: str) -> dict:
    classes = {c["id"]: c["name"] for c in coco["categories"]}
    per_class: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    aspect_ratios: list[float] = []
    crowd = 0

    for a in coco["annotations"]:
        if a["iscrowd"]:
            crowd += 1
            continue
        per_class[classes.get(a["category_id"], str(a["category_id"]))] += 1
        buckets[box_size_bucket(a["area"])] += 1
        w, h = a["bbox"][2], a["bbox"][3]
        if h > 0:
            aspect_ratios.append(round(w / h, 3))

    aspect_ratios.sort()
    p50 = aspect_ratios[len(aspect_ratios) // 2] if aspect_ratios else 0.0

    return {
        "split": name,
        "num_images": len(coco["images"]),
        "num_annotations": len(coco["annotations"]),
        "num_crowd_ignored": crowd,
        "boxes_per_class": dict(per_class),
        "box_size_buckets": {k: buckets.get(k, 0)
                             for k in ("tiny", "small", "medium", "large")},
        "aspect_ratio_p50": p50,
        "aspect_ratio_mean": round(sum(aspect_ratios) / len(aspect_ratios), 3)
        if aspect_ratios else 0.0,
    }


def main() -> int:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    p = cfg["paths"]

    interim = PROJECT_ROOT / p["coco_dir"]
    processed = PROJECT_ROOT / p["processed"] / "annotations"
    processed.mkdir(parents=True, exist_ok=True)

    train = load_coco(interim / "train.json")
    val_path = interim / "val.json"

    if val_path.exists():
        val = load_coco(val_path)
        print(f"[GOLD] using provided val split ({len(val['images'])} images)")
    else:
        frac = float(cfg["dataset"].get("val_fraction", 0.15))
        print(f"[GOLD] no val.json — carving {frac:.0%} "
              f"from train with seed {cfg['project']['seed']}")
        train, val = split_train_val(train, frac, cfg["project"]["seed"])

    (processed / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (processed / "val.json").write_text(json.dumps(val), encoding="utf-8")

    stats = {
        "seed": cfg["project"]["seed"],
        "categories": train["categories"],
        "min_box_area": cfg["dataset"]["min_box_area"],
        "splits": [compute_stats(train, "train"), compute_stats(val, "val")],
    }

    stats_path = PROJECT_ROOT / p["gold_stats"]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("[GOLD] OK")
    for s in stats["splits"]:
        print(f"  {s['split']:5s}: {s['num_images']:5d} images, "
              f"{s['num_annotations']:6d} boxes  {s['boxes_per_class']}")
        print(f"         sizes: {s['box_size_buckets']}")
    print(f"  stats -> {stats_path.relative_to(PROJECT_ROOT)}")
    print("\n  (This stats file is the baseline Evidently compares live "
          "traffic against in Step 9.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
