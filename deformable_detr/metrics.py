"""RAOD 检测结果后处理与 AP@0.5 评估。"""

from __future__ import annotations

import math

import torch

from .datasets.raod import RaodDetection
from .utils.box_ops import box_cxcywh_to_xyxy


def outputs_to_results(
    outputs: dict,
    targets: list[dict],
    score_threshold: float = 0.0,
) -> list[dict]:
    """将归一化 cxcywh 输出转换为原图像素坐标下的检测结果。"""
    probabilities = outputs["pred_logits"].sigmoid()
    scores, labels = probabilities.max(dim=-1)
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
    results: list[dict] = []

    for image_scores, image_labels, image_boxes, target in zip(
        scores, labels, boxes, targets
    ):
        original_height, original_width = target["orig_size"].tolist()
        scale = image_boxes.new_tensor(
            [original_width, original_height, original_width, original_height]
        )
        image_boxes = image_boxes * scale
        image_boxes[:, 0::2].clamp_(0, original_width)
        image_boxes[:, 1::2].clamp_(0, original_height)

        keep = image_scores >= score_threshold
        for score, label, box in zip(
            image_scores[keep].tolist(),
            image_labels[keep].tolist(),
            image_boxes[keep].tolist(),
        ):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            results.append(
                {
                    "image_id": str(target["image_id"]),
                    "label": int(label),
                    "bbox": [float(coordinate) for coordinate in box],
                    "score": float(score),
                }
            )
    return results


def _box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_empty(0)
    intersection_min = torch.maximum(box[:2], boxes[:, :2])
    intersection_max = torch.minimum(box[2:], boxes[:, 2:])
    intersection_size = (intersection_max - intersection_min).clamp(min=0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (
        boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps)


def voc2007_ap(recalls: list[float], precisions: list[float]) -> float:
    """VOC2007 的 11 点插值 AP。"""
    average_precision = 0.0
    for threshold_index in range(11):
        recall_threshold = threshold_index / 10
        precision = max(
            (
                precision
                for recall, precision in zip(recalls, precisions)
                if recall >= recall_threshold
            ),
            default=0.0,
        )
        average_precision += precision / 11
    return min(average_precision, 1.0)


def evaluate_raod(
    results: list[dict],
    dataset: RaodDetection,
    image_ids: list[str] | None = None,
    iou_threshold: float = 0.5,
) -> dict:
    """在指定图片上计算 VOC2007 风格的 AP@0.5。"""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold 必须位于 (0, 1] 范围")
    evaluated_ids = (
        list(dataset.ids)
        if image_ids is None
        else list(dict.fromkeys(image_ids))
    )
    annotations = {
        image_id: dataset.get_ground_truth(image_id) for image_id in evaluated_ids
    }

    per_class_ap: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for class_index, class_name in enumerate(dataset.classes):
        ground_truth = {}
        positive_count = 0
        for image_id, annotation in annotations.items():
            class_mask = annotation["labels"] == class_index
            class_boxes = annotation["boxes"][class_mask]
            difficult = annotation["difficult"][class_mask]
            ground_truth[image_id] = {
                "boxes": class_boxes,
                "difficult": difficult,
                "detected": torch.zeros(len(class_boxes), dtype=torch.bool),
            }
            positive_count += int((~difficult).sum().item())

        detections = sorted(
            (
                result
                for result in results
                if int(result["label"]) == class_index
                and str(result["image_id"]) in ground_truth
            ),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        true_positives: list[float] = []
        false_positives: list[float] = []

        for detection in detections:
            image_ground_truth = ground_truth[str(detection["image_id"])]
            if image_ground_truth["boxes"].numel() == 0:
                true_positives.append(0.0)
                false_positives.append(1.0)
                continue

            detection_box = torch.as_tensor(detection["bbox"], dtype=torch.float32)
            overlaps = _box_iou(detection_box, image_ground_truth["boxes"])
            best_overlap, best_index_tensor = overlaps.max(dim=0)
            best_index = int(best_index_tensor.item())
            if float(best_overlap.item()) < iou_threshold:
                true_positives.append(0.0)
                false_positives.append(1.0)
            elif bool(image_ground_truth["difficult"][best_index].item()):
                continue
            elif not bool(image_ground_truth["detected"][best_index].item()):
                image_ground_truth["detected"][best_index] = True
                true_positives.append(1.0)
                false_positives.append(0.0)
            else:
                true_positives.append(0.0)
                false_positives.append(1.0)

        if positive_count == 0:
            per_class_ap[class_name] = math.nan
            per_class_recall[class_name] = math.nan
            continue

        cumulative_tp = 0.0
        cumulative_fp = 0.0
        recalls: list[float] = []
        precisions: list[float] = []
        for true_positive, false_positive in zip(true_positives, false_positives):
            cumulative_tp += true_positive
            cumulative_fp += false_positive
            recalls.append(cumulative_tp / positive_count)
            precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1.0))

        per_class_ap[class_name] = voc2007_ap(recalls, precisions)
        per_class_recall[class_name] = recalls[-1] if recalls else 0.0

    finite_aps = [value for value in per_class_ap.values() if math.isfinite(value)]
    mean_ap = sum(finite_aps) / len(finite_aps) if finite_aps else math.nan
    return {
        "iou_threshold": iou_threshold,
        "ap_method": "VOC2007 11-point",
        "num_images": len(evaluated_ids),
        "per_class_ap": per_class_ap,
        "per_class_recall": per_class_recall,
        "map": mean_ap,
    }
