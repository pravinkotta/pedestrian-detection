"""The dataset generator itself (Step 7).

CI rebuilds ``data/`` from an empty checkout with this module, so it has to be
correct *before* anything downstream runs. These tests need no dataset on
disk — they generate a handful of frames directly — which means they always
run, unlike the pipeline tests that skip when ``data/`` is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.generate import CLASSES, IMG_H, IMG_W, SPLITS, generate_split, make_scene


def test_make_scene_returns_a_labelled_frame():
    img, anns = make_scene(np.random.default_rng(0))

    assert img.size == (IMG_W, IMG_H)
    assert anns, "a scene with no objects cannot be trained on"
    assert 1 <= len(anns) <= 8

    for a in anns:
        assert set(a) == {"bbox", "label"}
        assert a["label"] in (1, 2, 3), f"label outside the class map: {a['label']}"
        x0, y0, x1, y1 = a["bbox"]
        # inside the frame, and not degenerate (a zero-area box is unlearnable)
        assert 0 <= x0 < x1 <= IMG_W - 1, a
        assert 0 <= y0 < y1 <= IMG_H - 1, a
        assert x1 - x0 >= 4 and y1 - y0 >= 4, a


def test_scenes_vary_with_the_seed():
    """Two seeds must not produce two copies of the same frame — otherwise
    '240 training images' would really mean 'one image, 240 times'."""
    a, _ = make_scene(np.random.default_rng(1))
    b, _ = make_scene(np.random.default_rng(2))
    assert np.asarray(a).tobytes() != np.asarray(b).tobytes()


def test_generate_split_writes_the_expected_files(tmp_path):
    recs = generate_split("val", 3, 1042, tmp_path)

    files = sorted(p.name for p in (tmp_path / "val").glob("*.png"))
    assert files == ["val_0000.png", "val_0001.png", "val_0002.png"]
    assert len(recs) == 3
    for r in recs:
        assert r["file"] in files
        assert r["annotations"]


def test_split_sizes_match_the_published_dataset():
    """240 / 80 / 60 = the 380 frames quoted in the README."""
    assert SPLITS == {"train": 240, "val": 80, "test": 60}
    assert sum(SPLITS.values()) == 380
    assert CLASSES == ["background", "pedestrian", "cyclist", "vehicle"]


def test_meta_json_round_trips(tmp_path):
    """The Silver layer reads data/meta.json — the shape must survive JSON."""
    recs = generate_split("train", 2, 1042, tmp_path)
    meta = {"train": recs}
    back = json.loads(json.dumps(meta))
    assert back == meta


def test_existing_dataset_matches_the_generator(tmp_path):
    """If data/ is already on disk, it must be exactly what this module
    produces: same seeds as the notebook, so a regeneration is a no-op.

    Skipped on a fresh checkout (CI generates it first, then runs pytest —
    at that point data/ exists and this assertion runs).
    """
    from src.data.generate import PROJECT_ROOT, generate

    existing = PROJECT_ROOT / "data" / "meta.json"
    if not existing.exists():
        pytest.skip("data pipeline not run yet")

    with open(existing) as fh:
        on_disk = json.load(fh)

    fresh = generate(tmp_path)
    assert fresh == on_disk, (
        "data/ no longer matches src/data/generate.py - the dataset was "
        "produced by something other than the committed generator"
    )


@pytest.mark.parametrize("split", list(SPLITS))
def test_every_split_has_all_three_classes(split):
    from src.data.generate import PROJECT_ROOT

    meta_path = PROJECT_ROOT / "data" / "meta.json"
    if not meta_path.exists():
        pytest.skip("data pipeline not run yet")

    with open(meta_path) as fh:
        meta = json.load(fh)
    labels = {a["label"] for r in meta[split] for a in r["annotations"]}
    assert labels == {1, 2, 3}, f"{split} is missing a class: {labels}"
