"""Deformable DETR 的集合匹配损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """对每个 query/类别计算 sigmoid focal loss。"""
    if num_boxes <= 0:
        raise ValueError("num_boxes 必须大于 0")
    probabilities = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    target_probability = probabilities * targets + (1 - probabilities) * (
        1 - targets
    )
    loss = ce_loss * (1 - target_probability).pow(gamma)
    if alpha >= 0:
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_factor * loss
    return loss.mean(dim=1).sum() / num_boxes


class SetCriterion(nn.Module):
    """先一对一匹配，再计算分类、L1 和 GIoU 损失。"""

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict[str, float],
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_indices = torch.cat(
            [
                torch.full_like(prediction_indices, batch_index)
                for batch_index, (prediction_indices, _) in enumerate(indices)
            ]
        )
        query_indices = torch.cat(
            [prediction_indices for prediction_indices, _ in indices]
        )
        return batch_indices, query_indices

    def loss_labels(self, outputs, targets, indices, num_boxes):
        logits = outputs["pred_logits"]
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"分类头输出 {logits.shape[-1]} 类，但 criterion 配置为 "
                f"{self.num_classes} 类"
            )
        matched_index = self._get_src_permutation_idx(indices)
        target_one_hot = torch.zeros_like(logits)

        matched_labels = torch.cat(
            [
                target["labels"][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ]
        )
        if matched_labels.numel() > 0:
            target_one_hot[
                matched_index[0], matched_index[1], matched_labels
            ] = 1

        loss_ce = sigmoid_focal_loss(
            logits,
            target_one_hot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        # mean(dim=1) 已除以 num_queries；乘回来与官方实现的尺度一致。
        return {"loss_ce": loss_ce * logits.shape[1]}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        matched_index = self._get_src_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][matched_index]
        target_boxes = torch.cat(
            [
                target["boxes"][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ]
        )

        if source_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0
            return {"loss_bbox": zero, "loss_giou": zero}

        loss_bbox = F.l1_loss(source_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(source_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        return {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }

    def _compute_losses(self, outputs, targets, num_boxes):
        indices = self.matcher(outputs, targets)
        losses = self.loss_labels(outputs, targets, indices, num_boxes)
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        if not targets:
            raise ValueError("targets 不能为空")
        main_outputs = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        num_boxes = max(sum(len(target["labels"]) for target in targets), 1)
        losses = self._compute_losses(main_outputs, targets, float(num_boxes))

        for layer_index, auxiliary_outputs in enumerate(outputs.get("aux_outputs", [])):
            layer_losses = self._compute_losses(
                auxiliary_outputs, targets, float(num_boxes)
            )
            losses.update(
                {f"{name}_{layer_index}": value for name, value in layer_losses.items()}
            )

        if "enc_outputs" in outputs:
            # 两阶段 Encoder 做的是目标/背景候选筛选，类别统一映射为前景类 0。
            binary_targets = [
                {**target, "labels": torch.zeros_like(target["labels"])}
                for target in targets
            ]
            encoder_losses = self._compute_losses(
                outputs["enc_outputs"], binary_targets, float(num_boxes)
            )
            losses.update(
                {f"{name}_enc": value for name, value in encoder_losses.items()}
            )
        return losses

    def total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """按基础损失名给主输出、辅助层和 Encoder 输出统一加权。"""
        weighted = []
        for name, value in losses.items():
            base_name = next(
                (
                    key
                    for key in self.weight_dict
                    if name == key or name.startswith(f"{key}_")
                ),
                None,
            )
            if base_name is not None:
                weighted.append(value * self.weight_dict[base_name])
        if not weighted:
            raise ValueError("losses 中没有可加权的损失项")
        return torch.stack(weighted).sum()
