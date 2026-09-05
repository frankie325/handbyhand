"""Deformable DETR 的一对一匈牙利匹配。"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """用分类 Focal 代价、框 L1 代价和 GIoU 代价寻找最优匹配。"""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if cost_class == cost_bbox == cost_giou == 0:
            raise ValueError("至少需要启用一种匹配代价")
        if min(cost_class, cost_bbox, cost_giou) < 0:
            raise ValueError("匹配代价权重不能为负数")
        if not 0 <= focal_alpha <= 1:
            raise ValueError("focal_alpha 必须位于 [0, 1]")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]):
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        batch_size, num_queries, num_classes = logits.shape
        device = logits.device
        if len(targets) != batch_size:
            raise ValueError(
                f"targets 数量 {len(targets)} 与 batch_size {batch_size} 不一致"
            )
        target_sizes = [len(target["labels"]) for target in targets]

        if sum(target_sizes) == 0:
            return [
                (
                    torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.int64, device=device),
                )
                for _ in targets
            ]

        probabilities = logits.flatten(0, 1).sigmoid()
        predicted_boxes = boxes.flatten(0, 1)
        target_labels = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets])
        if target_labels.min() < 0 or target_labels.max() >= num_classes:
            raise ValueError(
                f"target label 必须位于 [0, {num_classes - 1}]，"
                f"实际范围为 [{target_labels.min().item()}, {target_labels.max().item()}]"
            )

        # Focal loss 的正样本代价减去负样本代价。概率越像目标类别，代价越低。
        eps = 1e-8
        negative_cost = (
            (1 - self.focal_alpha)
            * probabilities.pow(self.focal_gamma)
            * (-(1 - probabilities + eps).log())
        )
        positive_cost = (
            self.focal_alpha
            * (1 - probabilities).pow(self.focal_gamma)
            * (-(probabilities + eps).log())
        )
        class_cost = (
            positive_cost[:, target_labels] - negative_cost[:, target_labels]
        )
        bbox_cost = torch.cdist(predicted_boxes, target_boxes, p=1)
        giou_cost = -generalized_box_iou(
            box_cxcywh_to_xyxy(predicted_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )

        cost_matrix = (
            self.cost_class * class_cost
            + self.cost_bbox * bbox_cost
            + self.cost_giou * giou_cost
        ).reshape(batch_size, num_queries, -1)
        cost_matrix = cost_matrix.cpu()

        indices = []
        for image_index, image_cost in enumerate(cost_matrix.split(target_sizes, -1)):
            if target_sizes[image_index] == 0:
                prediction_indices = target_indices = torch.empty(
                    0, dtype=torch.int64, device=device
                )
            else:
                prediction_array, target_array = linear_sum_assignment(
                    image_cost[image_index].numpy()
                )
                prediction_indices = torch.as_tensor(
                    prediction_array, dtype=torch.int64, device=device
                )
                target_indices = torch.as_tensor(
                    target_array, dtype=torch.int64, device=device
                )
            indices.append((prediction_indices, target_indices))
        return indices
