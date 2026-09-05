"""Deformable DETR 损失构建入口。"""

from ..config import (
    BBOX_COST,
    BBOX_LOSS_COEF,
    CLASS_COST,
    CLASS_LOSS_COEF,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    GIOU_COST,
    GIOU_LOSS_COEF,
    NUM_CLASSES,
)
from .criterion import SetCriterion, sigmoid_focal_loss
from .matcher import HungarianMatcher


def build_criterion() -> SetCriterion:
    matcher = HungarianMatcher(
        cost_class=CLASS_COST,
        cost_bbox=BBOX_COST,
        cost_giou=GIOU_COST,
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
    )
    return SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict={
            "loss_ce": CLASS_LOSS_COEF,
            "loss_bbox": BBOX_LOSS_COEF,
            "loss_giou": GIOU_LOSS_COEF,
        },
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
    )


__all__ = [
    "HungarianMatcher",
    "SetCriterion",
    "build_criterion",
    "sigmoid_focal_loss",
]
