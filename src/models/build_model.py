"""Model construction — Faster R-CNN (MobileNetV3-Large + FPN).

Promoted from the notebook (cell 25) unchanged, including the two RPN caps
that are easy to lose when refactoring:

    model.rpn.pre_nms_top_n_test  = 600
    model.rpn.post_nms_top_n_test = 200

Torchvision defaults are 1000 / 1000. Every proposal flows through the box
head, which dominates CPU latency. Dropping to 200 post-NMS proposals is what
keeps inference inside a plausible edge budget while preserving recall on
dense scenes (notebook: 734 ms -> and it is the documented reason the
MobileNetV3 backbone was chosen at all).
"""

from __future__ import annotations

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# RPN proposal caps for test-time inference (edge-deployment optimisation)
PRE_NMS_TOP_N_TEST = 600
POST_NMS_TOP_N_TEST = 200


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, pretrained: bool = True,
                device: str | torch.device = "auto") -> torch.nn.Module:
    """Build Faster R-CNN with a fresh ``num_classes`` detection head.

    ``pretrained=True`` keeps the COCO-pretrained *backbone and FPN* (transfer
    learning — the part that actually matters) and re-initialises only the
    classification + box-regression heads. The RPN head is also COCO-trained
    and is kept, which is why the model converges in 3 epochs.
    """
    if pretrained:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.rpn.pre_nms_top_n_test = PRE_NMS_TOP_N_TEST
    model.rpn.post_nms_top_n_test = POST_NMS_TOP_N_TEST

    dev = resolve_device(str(device) if not isinstance(device, torch.device) else str(device))
    return model.to(dev)


def describe(model: torch.nn.Module) -> dict:
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "architecture": type(model).__name__,
        "params_total": n_params,
        "params_millions": round(n_params / 1e6, 1),
        "params_trainable": trainable,
        "device": str(next(model.parameters()).device),
        "rpn_pre_nms_top_n_test": model.rpn.pre_nms_top_n_test,
        "rpn_post_nms_top_n_test": model.rpn.post_nms_top_n_test,
    }
