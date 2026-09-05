"""边界框格式转换、IoU 和 GIoU。"""

from __future__ import annotations

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ],
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x_min, y_min, x_max, y_max = boxes.unbind(-1)
    return torch.stack(
        [
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            x_max - x_min,
            y_max - y_min,
        ],
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return size[:, 0] * size[:, 1]


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    intersection_size = (right_bottom - left_top).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = area1[:, None] + area2 - intersection
    iou = intersection / union.clamp(min=torch.finfo(union.dtype).eps)
    return iou, union


def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """计算两组 xyxy 框两两之间的 GIoU，返回 [len(boxes1), len(boxes2)]。"""
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError("boxes1 中存在 x_max < x_min 或 y_max < y_min 的框")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError("boxes2 中存在 x_max < x_min 或 y_max < y_min 的框")

    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    enclosing_size = (right_bottom - left_top).clamp(min=0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp(
        min=torch.finfo(enclosing_area.dtype).eps
    )
