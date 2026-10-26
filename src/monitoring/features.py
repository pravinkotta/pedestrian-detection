"""Image-level features — the drift monitor's vocabulary (Step 9).

Evidently compares *tabular columns*, not pictures, so every frame is
summarised into one row of numbers:

    brightness   mean luminance — did the world get darker / brighter?
    contrast     std luminance  — fog, overcast, washed-out sensors
    sharpness    variance of the Laplacian — defocused / dirty lens
    r/g/b_mean   per-channel means — white-balance or colour cast shift
    width/height/aspect_ratio     — resolution or cropping changed

Why features and not pixels: a feature row is ~9 floats. The serving API
appends one row per request to a JSONL log, which is

  * cheap     (microseconds, no image is stored — aggregates only, so the
    log is safe to keep even where raw frames would not be),
  * portable  (the drift job reads it over plain HTTP from anywhere),
  * explainable ("brightness drifted, p=1e-12" tells an operator *what*
    changed; "column 48213 drifted" does not).

This module deliberately imports only numpy + PIL: it runs inside the slim
serving image, which has neither pandas nor evidently (those live in the
dev/CI toolchain — see requirements-runtime.txt).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Columns compared by default; configs/config.yaml `monitor.features` may
# narrow this. Order defines the report's column order.
FEATURE_COLUMNS = [
    "brightness",
    "contrast",
    "sharpness",
    "r_mean",
    "g_mean",
    "b_mean",
    "width",
    "height",
    "aspect_ratio",
]

# Capture log bound: rotation keeps the file from growing forever on a busy
# endpoint (the log is a *recent window* of traffic, not an archive).
DEFAULT_MAX_BYTES = 5 * 1024 * 1024


def extract_features(image: Image.Image) -> dict[str, float]:
    """Summarise one PIL image as a flat row of numeric features."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)          # (H, W, 3)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # BT.601 luma — the standard perceptual grey.
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    width, height = rgb.size
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": _laplacian_variance(gray),
        "r_mean": float(r.mean()),
        "g_mean": float(g.mean()),
        "b_mean": float(b.mean()),
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width) / float(height) if height else float("nan"),
    }


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian: 0 for a flat image, large for
    a detailed/sharp one. The classic focus measure, without OpenCV."""
    padded = np.pad(gray, 1, mode="edge")
    lap = (padded[:-2, 1:-1] + padded[2:, 1:-1]
           + padded[1:-1, :-2] + padded[1:-1, 2:]
           - 4.0 * gray)
    return float(lap.var())


# ---------------------------------------------------------------------------
# JSONL capture log — written by the API, read by the drift job
# ---------------------------------------------------------------------------
def append_row(path: str | Path, row: dict[str, Any],
               max_bytes: int = DEFAULT_MAX_BYTES) -> None:
    """Append one feature row, rotating the file when it outgrows ``max_bytes``.

    Rotation keeps the newest half. A monitoring side-effect must never be
    able to fill a disk; it is a sliding window of recent traffic by design.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if p.exists() and p.stat().st_size > max_bytes:
            lines = p.read_text(encoding="utf-8").splitlines()
            p.write_text("\n".join(lines[len(lines) // 2:]) + "\n",
                         encoding="utf-8")
    except OSError:
        pass  # rotation is best-effort; losing old rows beats failing a request
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def read_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    """Read the capture log, newest rows last; corrupt lines are skipped."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    if limit is not None and len(rows) > limit:
        rows = rows[-limit:]
    return rows


def write_rows(path: str | Path, rows: Iterable[dict]) -> None:
    """Replace the file with these rows (used by tests and demos)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
