"""Drift monitor tests (Step 9) — features, statistical verdict, CLI contract.

All batches are synthetic images written to tmp_path, so these tests never
depend on the repo's dataset being present (CI's test job happens to build
one, but a monitoring test that silently skipped would be worthless).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.monitoring.drift import (
    CONFIG_PATH,
    EXIT_DRIFT,
    EXIT_ERROR,
    EXIT_OK,
    DriftError,
    compare,
    main,
)
from src.monitoring.features import (
    FEATURE_COLUMNS,
    append_row,
    extract_features,
    read_rows,
    write_rows,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _frame(width: int = 64, height: int = 48, mean: float = 120.0,
           sigma: float = 25.0, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(mean, sigma, (height, width, 3)),
                  0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _batch(directory: Path, n: int, mean: float = 120.0,
           seed: int = 0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _frame(mean=mean, seed=seed + i).save(directory / f"frame_{i:03d}.png")
    return directory


def _cfg() -> dict:
    return {
        "features": list(FEATURE_COLUMNS),
        "stat_test_threshold": 0.05,
        "drift_share": 0.5,
        "min_rows": 10,
    }


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------
def test_extract_features_returns_exactly_the_configured_columns():
    row = extract_features(_frame())
    assert set(row) == set(FEATURE_COLUMNS)
    assert all(isinstance(v, float) for v in row.values())
    assert all(math.isfinite(v) for v in row.values()), "no NaN in normal features"


def test_features_are_deterministic_and_bounded():
    img = _frame(seed=7)
    a, b = extract_features(img), extract_features(img)
    assert a == b, "same image must produce the same row"

    assert 0.0 <= a["brightness"] <= 255.0
    assert a["contrast"] >= 0.0
    assert a["width"] == 64.0 and a["height"] == 48.0
    assert a["aspect_ratio"] == pytest.approx(64 / 48)


def test_sharpness_separates_flat_from_textured():
    """A flat frame has no edges: the focus measure must say so."""
    flat = Image.new("RGB", (64, 48), (128, 128, 128))
    sharp = _frame(sigma=60.0, seed=3)
    assert extract_features(flat)["sharpness"] < extract_features(sharp)["sharpness"]


# --------------------------------------------------------------------------
# capture log (API writes it, drift job reads it)
# --------------------------------------------------------------------------
def test_capture_log_roundtrip(tmp_path):
    log = tmp_path / "features.jsonl"
    for i in range(3):
        row = extract_features(_frame(seed=i))
        row["n_detections"] = i
        append_row(log, row)
    rows = read_rows(log)
    assert len(rows) == 3
    assert rows[0]["n_detections"] == 0
    assert set(FEATURE_COLUMNS) <= set(rows[-1])


def test_capture_log_rotates_instead_of_growing_forever(tmp_path):
    """A busy endpoint must not be able to fill the disk (5 MB default)."""
    log = tmp_path / "features.jsonl"
    for i in range(50):
        append_row(log, {"i": i, "pad": "x" * 100}, max_bytes=1000)
    assert log.stat().st_size < 5000
    rows = read_rows(log)
    assert rows, "rotation must keep the NEWEST rows, not delete everything"
    assert rows[-1]["i"] == 49
    assert [r["i"] for r in rows] == sorted(r["i"] for r in rows)


def test_read_rows_skips_corrupt_lines(tmp_path):
    log = tmp_path / "features.jsonl"
    log.write_text('{"brightness": 1.0}\nnot json at all\n\n{"contrast": 2.0}\n',
                   encoding="utf-8")
    rows = read_rows(log)
    assert [r.get("brightness") for r in rows] == [1.0, None]
    assert rows[1]["contrast"] == 2.0
    assert read_rows(tmp_path / "missing.jsonl") == []


# --------------------------------------------------------------------------
# the verdict itself
# --------------------------------------------------------------------------
def test_same_distribution_is_not_drift(tmp_path):
    import pandas as pd

    ref = pd.DataFrame([extract_features(_frame(seed=i)) for i in range(40)])
    cur = pd.DataFrame([extract_features(_frame(seed=100 + i)) for i in range(40)])
    report, _ = compare(ref, cur, _cfg(), source_ref="ref", source_cur="cur")
    assert report["verdict"] == "OK"
    assert report["drift"]["drifted_share"] <= 0.5


def test_darkened_batch_is_drift_and_names_the_column(tmp_path):
    """Night frames vs a daylight baseline: the report must say WHICH
    column moved, not just 'drift happened'."""
    import pandas as pd

    ref = pd.DataFrame([extract_features(_frame(seed=i)) for i in range(40)])
    cur = pd.DataFrame(
        [extract_features(_frame(mean=40.0, seed=100 + i)) for i in range(40)])
    report, _ = compare(ref, cur, _cfg(), source_ref="ref", source_cur="cur")

    assert report["verdict"] == "DRIFT"
    by_name = {c["column"]: c for c in report["drift"]["columns"]}
    assert by_name["brightness"]["drifted"] is True
    assert by_name["brightness"]["p_value"] < 0.05
    # same size everywhere -> geometry has no distribution to compare
    assert set(report["reference"]["dropped_constant"]) == {
        "width", "height", "aspect_ratio"}


def test_constant_reference_columns_are_never_judged():
    """Comparing against a constant would make Evidently report drift (or
    all-clear) on an unanswerable question — the monitor drops them."""
    import pandas as pd

    ref = pd.DataFrame([extract_features(_frame(seed=i)) for i in range(30)])
    cur = pd.DataFrame([extract_features(_frame(seed=900 + i)) for i in range(30)])
    report, _ = compare(ref, cur, _cfg(), source_ref="ref", source_cur="cur")
    used = report["reference"]["columns_used"]
    assert "width" not in used and "height" not in used
    assert report["drift"]["columns"]  # still judged on the real features


def test_too_few_rows_is_an_error_not_a_verdict(tmp_path):
    import pandas as pd

    ref = pd.DataFrame([extract_features(_frame(seed=i)) for i in range(30)])
    cur = pd.DataFrame([extract_features(_frame(seed=900 + i)) for i in range(3)])
    with pytest.raises(DriftError, match="too few to judge"):
        compare(ref, cur, _cfg(), source_ref="ref", source_cur="cur")


# --------------------------------------------------------------------------
# CLI exit-code contract (0 OK / 1 DRIFT / 2 ERROR)
# --------------------------------------------------------------------------
def test_cli_exit_ok_on_matched_batch(tmp_path):
    ref = _batch(tmp_path / "ref", 30, mean=120.0, seed=0)
    cur = _batch(tmp_path / "cur", 30, mean=120.0, seed=500)
    code = main(["--config", str(CONFIG_PATH),
                 "--reference-dir", str(ref),
                 "--current", str(cur),
                 "--report", str(tmp_path / "report.json"),
                 "--html="])
    assert code == EXIT_OK


def test_cli_exit_drift_on_shifted_batch(tmp_path):
    ref = _batch(tmp_path / "ref", 30, mean=120.0, seed=0)
    cur = _batch(tmp_path / "cur", 30, mean=40.0, seed=500)
    report_path = tmp_path / "report.json"
    code = main(["--config", str(CONFIG_PATH),
                 "--reference-dir", str(ref),
                 "--current", str(cur),
                 "--report", str(report_path),
                 "--html="])
    assert code == EXIT_DRIFT
    body = json.loads(report_path.read_text(encoding="utf-8"))
    assert body["verdict"] == "DRIFT"
    assert body["drift"]["drifted_count"] >= 1


def test_cli_exit_error_on_missing_current(tmp_path):
    assert main(["--config", str(CONFIG_PATH),
                 "--current", str(tmp_path / "nope")]) == EXIT_ERROR


def test_cli_exit_error_without_any_input():
    assert main(["--config", str(CONFIG_PATH)]) == EXIT_ERROR


def test_cli_exit_error_on_tiny_batch(tmp_path):
    ref = _batch(tmp_path / "ref", 30, seed=0)
    cur = _batch(tmp_path / "cur", 3, seed=9)   # min_rows is 10
    assert main(["--config", str(CONFIG_PATH),
                 "--reference-dir", str(ref),
                 "--current", str(cur),
                 "--report", str(tmp_path / "r.json"),
                 "--html="]) == EXIT_ERROR


def test_jsonl_current_is_accepted(tmp_path):
    """The API's capture log is a first-class --current source."""
    ref = _batch(tmp_path / "ref", 30, mean=120.0, seed=0)
    log = tmp_path / "served.jsonl"
    write_rows(log, [extract_features(_frame(mean=40.0, seed=i))
                     for i in range(30)])
    assert main(["--config", str(CONFIG_PATH),
                 "--reference-dir", str(ref),
                 "--current", str(log),
                 "--report", str(tmp_path / "r.json"),
                 "--html="]) == EXIT_DRIFT
