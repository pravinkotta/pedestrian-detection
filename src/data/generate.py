"""Synthetic dataset generator — the source of truth for ``data/``.

Why this module exists
----------------------
``data/`` is gitignored (never commit a dataset), but CI clones a fresh VM
that starts with an empty checkout. Until this generator lived in the repo,
the only copy of the scene renderer was ``av_pedestrian_detection_faster_rcnn.ipynb``
— a file that is not committed either, so *nothing* could rebuild the data and
every data/model test silently skipped in CI.

This is a faithful promotion of the notebook's section-2 generator cell
(``draw_person`` / ``draw_cyclist`` / ``draw_vehicle`` / ``make_scene`` /
``generate_split``). The arithmetic is deliberately untouched: the split
seeds are ``1000 + SEED + index`` exactly as in the notebook, so a
regeneration reproduces ``data/meta.json`` byte-for-byte and the published
dataset statistics (380 frames, 1389 boxes) still hold.

Usage
-----
    python -m src.data.generate            # rebuild data/ (a few seconds)
    python -m src.data.generate --check    # verify it matches what is on disk

Then run the rest of the layers: ``make_bronze`` -> ``convert_to_coco`` ->
``make_gold``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
IMG_W, IMG_H = 256, 192                      # divisible by 32 (FPN-friendly)
CLASSES = ["background", "pedestrian", "cyclist", "vehicle"]
SPLITS = {"train": 240, "val": 80, "test": 60}

SKY_TOP, SKY_BOT = (135, 180, 220), (205, 228, 242)
ROAD_COLOR, SIDEWALK = (62, 64, 68), (152, 150, 146)
CLOTHING = [(200, 60, 60), (60, 90, 200), (70, 150, 80), (200, 150, 40),
            (150, 70, 170), (40, 160, 160), (230, 120, 60)]
VEH_COLORS = [(210, 210, 210), (180, 40, 40), (40, 80, 180), (240, 240, 240),
              (60, 60, 70), (230, 170, 30)]
SKIN = [(240, 200, 170), (215, 175, 140), (190, 150, 115), (245, 215, 185)]


def draw_person(d: ImageDraw.ImageDraw, cx: int, y_top: int, h: int,
                cloth: tuple[int, int, int], skin: tuple[int, int, int]) -> tuple:
    # round() already returns an int on py3 (kept exactly as the notebook had it)
    head_r = max(2, round(h * 0.11))
    torso_h = int(h * 0.42)
    leg_h = max(4, h - 2 * head_r - torso_h)
    torso_w = max(4, int(h * 0.28))
    head_cy = y_top + head_r
    d.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
              fill=skin)
    t0 = head_cy + head_r - 1
    d.rounded_rectangle([cx - torso_w // 2, t0, cx + torso_w // 2, t0 + torso_h],
                        radius=2, fill=cloth)
    lw = max(1, torso_w // 4)
    d.line([cx - torso_w // 4, t0 + torso_h, cx - torso_w // 4, t0 + torso_h + leg_h],
           fill=(40, 40, 50), width=lw)
    d.line([cx + torso_w // 4, t0 + torso_h, cx + torso_w // 4, t0 + torso_h + leg_h],
           fill=(40, 40, 50), width=lw)
    ay = t0 + max(1, torso_h // 5)
    al = max(2, h // 8)
    d.line([cx - torso_w // 2, ay, cx - torso_w // 2 - al, ay + leg_h // 2],
           fill=cloth, width=max(1, lw - 1))
    d.line([cx + torso_w // 2, ay, cx + torso_w // 2 + al, ay + leg_h // 2],
           fill=cloth, width=max(1, lw - 1))
    m = max(torso_w // 2, head_r) + 2
    return (cx - m, y_top, cx + m, t0 + torso_h + leg_h)


def draw_cyclist(d: ImageDraw.ImageDraw, cx: int, y_bottom: int, h: int,
                 cloth: tuple[int, int, int], skin: tuple[int, int, int]) -> tuple:
    wr = max(3, int(h * 0.16))
    wcy = y_bottom - wr
    gap = 2 * wr + max(2, int(h * 0.14))
    w1x, w2x = cx - gap // 2, cx + gap // 2
    for wx in (w1x, w2x):
        d.ellipse([wx - wr, wcy - wr, wx + wr, wcy + wr],
                  outline=(20, 20, 20), width=max(1, h // 24))
    fr, fw = (120, 120, 120), max(1, h // 30)
    apex = wcy - int(h * 0.30)
    d.line([w1x, wcy, cx, apex], fill=fr, width=fw)
    d.line([cx, apex, w2x, wcy], fill=fr, width=fw)
    d.line([cx, apex, cx + int(h * 0.20), apex + int(h * 0.05)], fill=fr, width=fw)
    rider_h = int(h * 0.52)
    head_r = max(2, int(rider_h * 0.17))
    top = apex - rider_h
    d.ellipse([cx - head_r, top, cx + head_r, top + 2 * head_r], fill=skin)
    d.line([cx, top + 2 * head_r, cx, apex], fill=cloth,
           width=max(2, int(h * 0.10)))
    d.line([cx, top + int(rider_h * 0.55), w2x - int(h * 0.06), wcy - int(h * 0.10)],
           fill=cloth, width=max(1, int(h * 0.07)))
    d.line([cx, top + int(rider_h * 0.55), w1x + int(h * 0.06), wcy - int(h * 0.10)],
           fill=cloth, width=max(1, int(h * 0.07)))
    return (w1x - wr - 1, top - 1, w2x + wr + 1, y_bottom + 1)


def draw_vehicle(d: ImageDraw.ImageDraw, cx: int, y_bottom: int, w: int, h: int,
                 color: tuple[int, int, int]) -> tuple:
    x0, x1 = cx - w // 2, cx + w // 2
    y0, y1 = y_bottom - h, y_bottom
    body_y0 = y0 + int(h * 0.35)
    d.rounded_rectangle([x0, body_y0, x1, y1], radius=3, fill=color)
    if x1 - int(w * 0.18) > x0 + int(w * 0.18):
        d.rounded_rectangle([x0 + int(w * 0.18), y0, x1 - int(w * 0.18), body_y0 + 2],
                            radius=3, fill=color)
    win_y0, win_y1 = y0 + 3, body_y0 - 2
    if win_y1 > win_y0 and x1 - int(w * 0.24) > x0 + int(w * 0.24):
        d.rectangle([x0 + int(w * 0.24), win_y0, x1 - int(w * 0.24), win_y1],
                    fill=(160, 190, 210))
    wr = max(2, int(h * 0.16))
    for wx in (x0 + int(w * 0.20), x1 - int(w * 0.20)):
        d.ellipse([wx - wr, y1 - 2 * wr, wx + wr, y1], fill=(15, 15, 15))
    return (x0 - 1, y0 - 1, x1 + 1, y1 + 1)


def make_scene(rng: np.random.Generator) -> tuple[Image.Image, list[dict]]:
    """Return ``(PIL image, [{'bbox': [x1,y1,x2,y2], 'label': int}])``."""
    while True:  # regenerate rare frames that end up empty after clipping
        img = Image.new("RGB", (IMG_W, IMG_H))
        d = ImageDraw.Draw(img)
        horizon = int(IMG_H * 0.58)
        for y in range(horizon):                       # sky gradient
            t = y / max(1, horizon - 1)
            d.line([(0, y), (IMG_W, y)],
                   fill=tuple(int(SKY_TOP[i] + (SKY_BOT[i] - SKY_TOP[i]) * t)
                             for i in range(3)))
        d.ellipse([IMG_W - 70, 8, IMG_W - 40, 38], fill=(255, 240, 180))
        x = 0                                          # buildings
        while x < IMG_W:
            bw = int(rng.integers(30, 70))
            bh = int(rng.integers(30, horizon - 70))
            d.rectangle([x, horizon - bh, x + bw, horizon],
                        fill=tuple(int(c) for c in rng.integers(95, 165, 3)))
            x += bw + int(rng.integers(2, 10))
        d.rectangle([0, horizon, IMG_W, horizon + 14], fill=SIDEWALK)
        d.rectangle([0, horizon + 14, IMG_W, IMG_H], fill=ROAD_COLOR)
        for lx in range(10, IMG_W, 36):                # lane dashes
            d.rectangle([lx, (horizon + IMG_H) // 2, lx + 18,
                         (horizon + IMG_H) // 2 + 3], fill=(230, 230, 230))
        objs = []
        for _ in range(int(rng.integers(1, 7))):
            r = rng.random()
            cls = 1 if r < 0.55 else (2 if r < 0.70 else 3)
            # base y encodes depth: further down the frame = closer to camera
            cy = int(horizon + 20 + (IMG_H - horizon - 40) * rng.random())
            if cls == 1:
                h = int(np.clip(cy * 0.42 * rng.uniform(0.55, 1.15), 24, 150))
                cx = int(rng.integers(14, IMG_W - 14))
                box = draw_person(d, cx, cy - h, h,
                                  CLOTHING[int(rng.integers(0, len(CLOTHING)))],
                                  SKIN[int(rng.integers(0, len(SKIN)))])
            elif cls == 2:
                h = int(np.clip(cy * 0.50 * rng.uniform(0.50, 1.00), 26, 130))
                cx = int(rng.integers(20, IMG_W - 20))
                box = draw_cyclist(d, cx, cy, h,
                                   CLOTHING[int(rng.integers(0, len(CLOTHING)))],
                                   SKIN[int(rng.integers(0, len(SKIN)))])
            else:
                w = int(np.clip(cy * 0.90 * rng.uniform(0.50, 1.10), 26, 170))
                h = int(w * rng.uniform(0.38, 0.50))
                cx = int(rng.integers(w // 2 + 4, IMG_W - w // 2 - 4))
                box = draw_vehicle(d, cx, cy, w, h,
                                   VEH_COLORS[int(rng.integers(0, len(VEH_COLORS)))])
            objs.append((cls, box))
        if len(objs) >= 2 and rng.random() < 0.35:      # occlusion pass
            cls, (x0, y0, x1, y1) = objs[int(rng.integers(0, len(objs)))]
            w = int((x1 - x0) * rng.uniform(0.8, 1.4))
            h = int(w * 0.45)
            cx = int((x0 + x1) // 2 + rng.integers(-10, 10))
            cy = min(IMG_H - 4, y1 + 2)
            draw_vehicle(d, cx, cy, w, h,
                         VEH_COLORS[int(rng.integers(0, len(VEH_COLORS)))])
        anns = []
        for cls, (x0, y0, x1, y1) in objs:              # clip to the frame
            x0 = max(0, min(IMG_W - 2, int(x0)))
            x1 = max(1, min(IMG_W - 1, int(x1)))
            y0 = max(0, min(IMG_H - 2, int(y0)))
            y1 = max(1, min(IMG_H - 1, int(y1)))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                anns.append({"bbox": [x0, y0, x1, y1], "label": int(cls)})
        if anns:
            return img, anns


def generate_split(split: str, n: int, seed: int, out_dir: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    d = out_dir / split
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in range(n):
        img, anns = make_scene(rng)
        fname = f"{split}_{i:04d}.png"
        img.save(d / fname)
        recs.append({"file": fname, "annotations": anns})
    return recs


def generate(out_dir: Path) -> dict[str, list[dict]]:
    """Write ``<out_dir>/<split>/*.png`` + ``<out_dir>/meta.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {s: generate_split(s, n, 1000 + SEED + k, out_dir)
            for k, (s, n) in enumerate(SPLITS.items())}
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh)
    return meta


def stats(meta: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for split, recs in meta.items():
        counts: dict[str, int] = defaultdict(int)
        for r in recs:
            for a in r["annotations"]:
                counts[CLASSES[a["label"]]] += 1
        total = sum(counts.values())
        rows.append({"split": split, "frames": len(recs), "objects": total,
                     "objs_per_frame": round(total / len(recs), 2),
                     "pedestrian": counts["pedestrian"],
                     "cyclist": counts["cyclist"],
                     "vehicle": counts["vehicle"]})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="src.data.generate",
        description="Regenerate the synthetic dataset (the notebook's "
                    "make_scene() generator, promoted into the repo so CI can "
                    "rebuild data/ from an empty checkout).",
    )
    p.add_argument("--out", default=None,
                   help="output directory (default: <repo>/data)")
    p.add_argument("--check", action="store_true",
                   help="generate into a temp dir and report whether it "
                        "matches the dataset already on disk")
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "data"

    if args.check:
        import tempfile

        if not (out_dir / "meta.json").exists():
            print(f"[GENERATE] nothing to compare against at {out_dir}")
            return 2
        with open(out_dir / "meta.json") as fh:
            existing = json.load(fh)
        with tempfile.TemporaryDirectory() as tmp:
            fresh = generate(Path(tmp))
        same = fresh == existing
        print(f"[GENERATE] regeneration matches meta.json: {same}")
        if not same:
            for s in SPLITS:
                print(f"  {s}: existing {len(existing.get(s, []))} frames, "
                      f"fresh {len(fresh.get(s, []))} frames")
        return 0 if same else 1

    t0 = time.time()
    meta = generate(out_dir)
    for row in stats(meta):
        print(f"[GENERATE] {row['split']:<6} {row['frames']:>3} frames  "
              f"{row['objects']:>4} boxes  "
              f"(ped {row['pedestrian']}, cyc {row['cyclist']}, veh {row['vehicle']})")
    print(f"[GENERATE] wrote {sum(len(v) for v in meta.values())} frames to "
          f"{out_dir} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
