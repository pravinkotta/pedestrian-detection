"""Unit tests for the model package (Step 2).

Run:
    pytest tests/test_model.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


def _loader(split: str, batch_size: int = 2):
    from src.models.dataset import make_loader

    ann = ROOT / CONFIG["paths"]["coco_dir"] / f"{split}.json"
    img = ROOT / CONFIG["paths"]["raw_images"] / split
    if not ann.exists():
        pytest.skip("data pipeline not run yet")
    return make_loader(ann, img, batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------------
# coordinate conversion — the silent-failure trap
# --------------------------------------------------------------------------
def test_xywh_to_xyxy_roundtrip_is_exact():
    from src.models.dataset import xywh_to_xyxy

    coco = torch.tensor([[10.0, 20.0, 30.0, 40.0]])   # x, y, w, h
    xyxy = xywh_to_xyxy(coco)
    assert xyxy.tolist() == [[10.0, 20.0, 40.0, 60.0]]


def test_dataset_boxes_are_xyxy_not_xywh():
    """Catches the classic bug: training on xywh without converting."""
    loader = _loader("train")
    imgs, targets = next(iter(loader))

    for img, tgt in zip(imgs, targets):
        boxes = tgt["boxes"]
        h, w = img.shape[-2], img.shape[-1]
        # x2 must be >= x1 (xyxy). If someone forgets the conversion,
        # x2 = x and every box collapses to zero/negative width.
        assert (boxes[:, 2] >= boxes[:, 0]).all(), "boxes look like xywh, not xyxy"
        assert (boxes[:, 3] >= boxes[:, 1]).all(), "boxes look like xywh, not xyxy"
        # and they must sit inside the frame
        assert boxes[:, 0].min() >= 0
        assert boxes[:, 1].min() >= 0
        assert boxes[:, 2].max() <= w + 1e-4
        assert boxes[:, 3].max() <= h + 1e-4
        # image tensor must be 0..1 float (ToTensor), CHW
        assert img.dtype == torch.float32 and img.shape[0] == 3
        assert 0.0 <= img.min() and img.max() <= 1.0


def test_dataset_labels_match_config():
    from src.models.dataset import CocoDetectionDataset

    ann = ROOT / CONFIG["paths"]["coco_dir"] / "train.json"
    if not ann.exists():
        pytest.skip("data pipeline not run yet")
    ds = CocoDetectionDataset(ann, ROOT / CONFIG["paths"]["raw_images"] / "train")

    expected = [CONFIG["dataset"]["label_map"][k]
                for k in sorted(int(k) for k in CONFIG["dataset"]["label_map"])]
    assert ds.classes == expected, "class order drifted from config"
    assert ds.num_classes == len(expected) + 1, "background class missing"


def test_model_num_classes_matches_dataset():
    from src.models.build_model import build_model
    from src.models.dataset import CocoDetectionDataset

    ann = ROOT / CONFIG["paths"]["coco_dir"] / "train.json"
    if not ann.exists():
        pytest.skip("data pipeline not run yet")
    ds = CocoDetectionDataset(ann, ROOT / CONFIG["paths"]["raw_images"] / "train")

    model = build_model(ds.num_classes, pretrained=False, device="cpu")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    n_logits = model.roi_heads.box_predictor.cls_score.out_features
    assert n_logits == ds.num_classes, (
        f"head predicts {n_logits} classes but dataset has {ds.num_classes}"
    )
    assert in_feats > 0


def test_rpn_proposal_caps_preserved():
    """The edge-deployment optimisation from the notebook must not be lost."""
    from src.models.build_model import build_model

    model = build_model(4, pretrained=False, device="cpu")
    assert model.rpn.pre_nms_top_n_test == 600
    assert model.rpn.post_nms_top_n_test == 200


# --------------------------------------------------------------------------
# forward pass (the shape bugs show up here)
# --------------------------------------------------------------------------
def test_training_forward_pass_produces_losses():
    loader = _loader("train")
    imgs, targets = next(iter(loader))

    from src.models.build_model import build_model
    model = build_model(4, pretrained=False, device="cpu")
    model.train()

    loss_dict = model(list(imgs), [dict(t) for t in targets])
    assert set(loss_dict) == {"loss_objectness", "loss_rpn_box_reg",
                              "loss_classifier", "loss_box_reg"}
    total = sum(loss_dict.values())
    assert torch.isfinite(total), "loss diverged"
    total.backward()   # confirms gradients flow


def test_inference_returns_boxes_scores_labels():
    loader = _loader("val")
    imgs, _ = next(iter(loader))

    from src.models.build_model import build_model
    model = build_model(4, pretrained=False, device="cpu")
    model.eval()
    with torch.no_grad():
        outs = model(list(imgs))

    for out in outs:
        assert out["boxes"].shape[0] == out["scores"].shape[0]
        assert out["boxes"].shape[0] == out["labels"].shape[0]
        if len(out["scores"]):
            assert out["scores"].min() >= 0 and out["scores"].max() <= 1
            assert out["labels"].min() >= 1, "label 0 is reserved for background"


# --------------------------------------------------------------------------
# evaluation metrics
# --------------------------------------------------------------------------
def test_box_iou_identity_and_disjoint():
    from src.models.evaluate import box_iou

    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)
    # boxes [0,0,10,10] and [5,0,15,10]: intersection 5*10=50,
    # union 100+100-50=150 -> IoU = 50/150 = 1/3
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_recall_at_fpr_prefers_within_budget():
    from src.models.evaluate import recall_at_fpr

    dets = ([{"score": 0.9, "label": 1, "tp": 1, "gt": 1}] * 5
            + [{"score": 0.8, "label": 1, "tp": 0, "gt": -1}] * 100)

    loose, _ = recall_at_fpr(dets, total_gt=5, negatives=1000, budget=0.5)
    tight, _ = recall_at_fpr(dets, total_gt=5, negatives=1000, budget=0.01)
    assert tight <= loose, "tighter FPR budget must not yield higher recall"
    assert tight == pytest.approx(1.0)   # all 5 TPs come before any FP


def test_ap_11point_perfect_detector_scores_one():
    import numpy as np

    from src.models.evaluate import ap_11point
    rec = np.linspace(0, 1, 11)
    prec = np.ones(11)
    assert ap_11point(rec, prec) == pytest.approx(1.0)


def test_gate_blocks_model_below_threshold():
    from src.models.evaluate import check_gate

    gate = CONFIG["gate"]
    bad = {"recall_at_fpr": 0.5, "ap_11point": 0.4,
           "pedestrian_recall_at_op": 0.5}
    passed, failures = check_gate(bad, gate)
    assert not passed
    assert len(failures) == 3, f"expected 3 failures, got: {failures}"

    good = {"recall_at_fpr": 0.95, "ap_11point": 0.9,
            "pedestrian_recall_at_op": 0.95}
    passed, failures = check_gate(good, gate)
    assert passed, f"expected pass, got: {failures}"
