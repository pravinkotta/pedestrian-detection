"""Evaluation engine — the project's headline metric is Recall @ FPR.

Promoted from notebook cells 33 (``evaluate``, ``pr_points``, ``ap_11point``)
and 39 (``metrics_at``, ``recall_at_fpr_budget``), with one structural change:
the "negatives" denominator is computed from each image's actual shape rather
than the hardcoded ``IMG_H``/``IMG_W`` constants, so it keeps working if the
frame size ever changes.

Why this metric and not mAP
---------------------------
The two error types have asymmetric cost in an AV perception stack:

* **False negative** (missed pedestrian) -> no braking -> possible collision.
* **False positive** (phantom pedestrian) -> unnecessary hard braking.

So we maximise recall *subject to* FPR <= 1%. A model with great mAP that
misses 1 pedestrian in 10 is unsafe and must not ship — which is exactly what
the ``gate:`` block in ``configs/config.yaml`` enforces.

Matching protocol (unchanged from the notebook): greedy, score-ordered,
IoU >= 0.5, same-class only, each ground-truth box matched at most once.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

IOU_THR = 0.5
MIN_SCORE = 0.05
NEG_CELL = 16   # negative regions counted as (H // 16) * (W // 16) grid cells


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def box_iou(a, b) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    denom = aa + bb - inter
    return inter / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------
# inference + matching
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_detections(model, loader, device, iou_thr: float = IOU_THR,
                        min_score: float = MIN_SCORE) -> dict:
    """Run the model over ``loader`` and greedy-match against ground truth."""
    model.eval()

    dets: list[dict] = []
    fn_labels: list[int] = []
    gt_per_class: defaultdict = defaultdict(int)
    total_gt = n_images = n_negatives = 0

    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        outs = model(imgs)
        n_images += len(imgs)

        for img_tensor, tgt, out in zip(imgs, targets, outs):
            h, w = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])
            n_negatives += (h // NEG_CELL) * (w // NEG_CELL)

            gt_boxes = tgt["boxes"].cpu().numpy()
            gt_labels = tgt["labels"].cpu().numpy()
            total_gt += len(gt_boxes)
            n_negatives -= len(gt_boxes)
            for c in np.unique(gt_labels):
                gt_per_class[int(c)] += int((gt_labels == c).sum())

            boxes = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()

            keep = scores >= min_score
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            used: set[int] = set()
            for oi in np.argsort(-scores):
                best_iou, best_gi = iou_thr, -1
                for gi in np.where(gt_labels == labels[oi])[0]:
                    if gi in used:
                        continue
                    v = box_iou(boxes[oi], gt_boxes[gi])
                    if v > best_iou:
                        best_iou, best_gi = v, gi
                if best_gi >= 0:
                    used.add(best_gi)
                    dets.append({"score": float(scores[oi]),
                                 "label": int(labels[oi]),
                                 "tp": 1, "gt": int(gt_labels[best_gi])})
                else:
                    dets.append({"score": float(scores[oi]),
                                 "label": int(labels[oi]),
                                 "tp": 0, "gt": -1})

            for gi in range(len(gt_boxes)):
                if gi not in used:
                    fn_labels.append(int(gt_labels[gi]))

    return {
        "dets": dets,
        "total_gt": total_gt,
        "gt_per_class": dict(gt_per_class),
        "fn_labels": fn_labels,
        "n_images": n_images,
        "n_negatives": max(n_negatives, 1),
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def pr_points(dets: list[dict], n_pos: int):
    """Score-ordered precision/recall curve."""
    rows = sorted(dets, key=lambda r: -r["score"])
    if not rows:
        return [], np.array([1.0]), np.array([0.0]), np.array([]), np.array([])
    tp = np.cumsum([r["tp"] for r in rows])
    fp = np.cumsum([1 - r["tp"] for r in rows])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(n_pos, 1)
    return rows, prec, rec, tp, fp


def ap_11point(rec, prec) -> float:
    """VOC-style 11-point average precision."""
    if len(rec) == 0:
        return 0.0
    ap = 0.0
    for t in np.arange(0.0, 1.01, 0.1):
        mask = rec >= t - 1e-9
        ap += prec[mask].max() if mask.any() else 0.0
    return ap / 11.0


def recall_at_fpr(dets: list[dict], total_gt: int, negatives: int,
                  budget: float) -> tuple[float, float]:
    """(max recall, threshold) achievable while holding FPR <= budget."""
    rows = sorted(dets, key=lambda r: -r["score"])
    tp = fp = 0
    best_rec, best_thr = 0.0, 1.0
    for r in rows:
        tp += r["tp"]
        fp += 1 - r["tp"]
        if fp / max(negatives, 1) <= budget:
            rec = tp / max(total_gt, 1)
            if rec >= best_rec:
                best_rec, best_thr = rec, r["score"]
    return best_rec, best_thr


def compute_metrics(result: dict, classes: list[str],
                    fpr_budget: float = 0.01) -> dict:
    """Full metric bundle — this is what gets logged to MLflow in Step 3."""
    dets = result["dets"]
    total_gt = result["total_gt"]
    negatives = result["n_negatives"]
    n_images = result["n_images"]

    _, prec, rec, _, _ = pr_points(dets, total_gt)
    ap_all = ap_11point(rec, prec)

    # pedestrian class specifically — the safety-critical one
    ped_id = classes.index("pedestrian") + 1 if "pedestrian" in classes else 1
    ped_pos = result["gt_per_class"].get(ped_id, 0)
    _, prec_ped, rec_ped, _, _ = pr_points(
        [d for d in dets if d["label"] == ped_id], ped_pos
    )
    ap_ped = ap_11point(rec_ped, prec_ped)

    best_rec, best_thr = recall_at_fpr(dets, total_gt, negatives, fpr_budget)

    # metrics at the chosen operating point
    kp = [r for r in dets if r["score"] >= best_thr]
    tp_at = sum(r["tp"] for r in kp)
    fp_at = len(kp) - tp_at
    ped_tp = sum(r["tp"] for r in kp if r["gt"] == ped_id)

    # per-class recall
    per_class_recall = {}
    for cid, name in enumerate(classes, start=1):
        gt_n = result["gt_per_class"].get(cid, 0)
        if gt_n:
            tp_c = sum(r["tp"] for r in dets if r["gt"] == cid)
            per_class_recall[name] = round(tp_c / gt_n, 4)

    return {
        "ap_11point": round(float(ap_all), 4),
        "ap_pedestrian": round(float(ap_ped), 4),
        "recall_at_fpr": round(float(best_rec), 4),
        "fpr_budget": fpr_budget,
        "operating_threshold": round(float(best_thr), 4),
        "precision_at_op": round(tp_at / len(kp), 4) if kp else 1.0,
        "recall_at_op": round(tp_at / max(total_gt, 1), 4),
        "pedestrian_recall_at_op": round(ped_tp / max(ped_pos, 1), 4),
        "fpr_at_op": round(fp_at / negatives, 6),
        "fppi_at_op": round(fp_at / max(n_images, 1), 4),
        "total_gt": total_gt,
        "total_detections": len(dets),
        "num_images": n_images,
        "num_negatives": negatives,
        "per_class_recall": per_class_recall,
    }


def evaluate(model, loader, device, classes: list[str],
             fpr_budget: float = 0.01) -> dict:
    """One call: run inference, match, and return the metric bundle."""
    result = collect_detections(model, loader, device)
    metrics = compute_metrics(result, classes, fpr_budget)
    metrics["_raw"] = result
    return metrics


def check_gate(metrics: dict, gate: dict) -> tuple[bool, list[str]]:
    """Apply the ship/no-ship gate from config. Returns (passed, failures)."""
    failures: list[str] = []

    checks = [
        ("recall_at_fpr", "min_recall_at_fpr",
         f"recall@FPR<={gate.get('fpr_budget', 0.01):.0%}"),
        ("ap_11point", "min_ap", "11-point AP"),
        ("pedestrian_recall_at_op", "min_pedestrian_recall",
         "pedestrian recall"),
    ]
    for metric_key, gate_key, label in checks:
        if gate_key not in gate:
            continue
        actual, required = metrics.get(metric_key), gate[gate_key]
        if actual is None:
            failures.append(f"{label}: metric missing from results")
        elif actual < required:
            failures.append(f"{label}: {actual} < required {required}")

    return (not failures), failures
