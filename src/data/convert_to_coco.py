"""Silver layer: convert ``data/meta.json`` -> COCO JSON per split.

Run:
    python -m src.data.convert_to_coco

Input  : data/meta.json + data/{train,val,test}/*.png
Output : data/interim/annotations/{train,val,test}.json

Why COCO?  torchvision's detection models, ``pycocotools`` evaluation and
every MLOps tool in this pipeline speak COCO. Converting once, here, means
training, evaluation, CI and the drift monitor all share one format.

Conversion detail
-----------------
``meta.json`` stores boxes as ``[x1, y1, x2, y2]`` (corner format).
COCO stores them as ``[x, y, w, h]`` (origin + size). The conversion is::

    w = x2 - x1
    h = y2 - y1

and ``area = w * h``. Boxes are clipped to the image and anything smaller
than ``dataset.min_box_area`` is dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# COCO category ids are 1..N (0 is reserved for "background" in detection models)
XYXY_TO_XYWH = True


def convert_split(split: str, entries: list, label_map: dict[int, str],
                  img_dir: Path, min_area: float) -> dict:
    from PIL import Image

    images, annotations = [], []
    img_id, ann_id = 0, 0
    dropped_small = dropped_oob = 0

    for entry in entries:
        file = entry["file"]
        img_path = img_dir / file
        if not img_path.exists():
            raise SystemExit(f"[SILVER] image missing: {img_path}")

        with Image.open(img_path) as im:
            width, height = im.size

        images.append({"id": img_id, "file_name": file,
                       "height": height, "width": width})

        for ann in entry.get("annotations", []):
            x1, y1, x2, y2 = (float(v) for v in ann["bbox"])

            # clip to image bounds
            x1c, y1c = max(x1, 0.0), max(y1, 0.0)
            x2c, y2c = min(x2, float(width)), min(y2, float(height))
            w, h = x2c - x1c, y2c - y1c

            if w <= 0 or h <= 0:
                dropped_oob += 1
                continue
            if w * h < min_area:
                dropped_small += 1
                continue

            cat_id = int(ann["label"])
            if cat_id not in label_map:
                dropped_oob += 1
                continue

            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,          # 1..3, matches meta.json labels
                "bbox": [x1c, y1c, w, h] if XYXY_TO_XYWH else [x1, y1, x2, y2],
                "area": w * h,
                "iscrowd": 0,
            })
        img_id += 1

    categories = [{"id": cid, "name": name}
                  for cid, name in sorted(label_map.items())]

    return {"images": images, "annotations": annotations,
            "categories": categories,
            "_dropped": {"small": dropped_small, "out_of_bounds": dropped_oob}}


def main() -> int:
    parser = argparse.ArgumentParser(description="meta.json -> COCO converter")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    p, ds = cfg["paths"], cfg["dataset"]

    meta_path = PROJECT_ROOT / p["raw_meta"]
    if not meta_path.exists():
        raise SystemExit(f"[SILVER] FAILED — {meta_path} not found. "
                         "Run `python -m src.data.make_bronze` first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_map = {int(k): v for k, v in ds["label_map"].items()}
    min_area = float(ds.get("min_box_area", 0))

    out_dir = PROJECT_ROOT / p["coco_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SILVER] reading {meta_path.name} "
          f"({', '.join(f'{s}={len(meta.get(s, []))}' for s in ds['splits'])})")

    for split in ds["splits"]:
        entries = meta.get(split, [])
        img_dir = PROJECT_ROOT / p["raw_images"] / split
        coco = convert_split(split, entries, label_map, img_dir, min_area)

        if not coco["images"]:
            print(f"[SILVER] FAILED — split '{split}' produced 0 images")
            return 1

        out_path = out_dir / f"{split}.json"
        out_path.write_text(json.dumps(coco), encoding="utf-8")

        dropped = coco.pop("_dropped")
        print(f"  {split:5s}: {len(coco['images']):4d} images, "
              f"{len(coco['annotations']):4d} boxes "
              f"(dropped {dropped['small']} small, "
              f"{dropped['out_of_bounds']} invalid)")
        print(f"          -> {out_path.relative_to(PROJECT_ROOT)}")

    print("[SILVER] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
