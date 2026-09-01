from __future__ import annotations

import math

import torch
from tqdm import tqdm

from .config import BATCH_SIZE, NUM_CLASSES, DATASET_TYPE
from .datasets import VocDetection, RaodDetection, DATASET_REGISTRY
from .datasets.build import build_dataloader
from .loss.criterion import SetCriterion
from .loss.matcher import HungarianMatcher
from .model.build import build_model
from .utils.bos_ops import box_cxcywh_to_xyxy
from .utils.common import get_device


def outputs_to_voc_results(outputs, targets) -> list[dict]:
    """把模型输出转换为 VOC 评估使用的原图像素 xyxy 检测结果。"""
    probabilities = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
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

        for score, label, box in zip(
            image_scores.tolist(), image_labels.tolist(), image_boxes.tolist()
        ):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            results.append(
                {
                    "image_id": str(target["image_id"]),
                    "label": int(label),
                    "bbox": box,
                    "score": float(score),
                }
            )
    return results


def _box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """计算一个 xyxy 框与一组 xyxy 框的 IoU。"""
    if boxes.numel() == 0:
        return boxes.new_empty((0,))

    intersection_min = torch.maximum(box[:2], boxes[:, :2])
    intersection_max = torch.minimum(box[2:], boxes[:, 2:])
    intersection_size = (intersection_max - intersection_min).clamp(min=0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps)


def voc2007_ap(recalls: list[float], precisions: list[float]) -> float:
    """计算 VOC2007 官方的 11 点插值 AP。"""
    average_precision = 0.0
    for recall_threshold in (index / 10 for index in range(11)):
        precision = max(
            (
                precision
                for recall, precision in zip(recalls, precisions)
                if recall >= recall_threshold
            ),
            default=0.0,
        )
        average_precision += precision / 11
    return average_precision


def evaluate_voc(
    results: list[dict],
    dataset: VocDetection | RaodDetection,
    iou_threshold: float = 0.5,
) -> dict:
    """按 VOC2007 规则计算每类 AP 和 mAP。

    difficult GT 不计入正样本数量；达到 IoU 阈值且匹配 difficult GT 的预测
    会被忽略，既不是 TP 也不是 FP。
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold 必须位于 (0, 1] 范围")

    annotations = {
        image_id: dataset.get_ground_truth(image_id) for image_id in dataset.ids
    }
    per_class_ap: dict[str, float] = {}
    for class_index, class_name in enumerate(dataset.classes):
        ground_truth: dict[str, dict[str, torch.Tensor]] = {}
        positive_count = 0
        for image_id in dataset.ids:
            annotation = annotations[image_id]
            class_mask = annotation["labels"] == class_index
            boxes = annotation["boxes"][class_mask]
            difficult = annotation["difficult"][class_mask]
            ground_truth[image_id] = {
                "boxes": boxes,
                "difficult": difficult,
                "detected": torch.zeros(len(boxes), dtype=torch.bool),
            }
            positive_count += int((~difficult).sum().item())

        detections = sorted(
            (result for result in results if int(result["label"]) == class_index),
            key=lambda result: float(result["score"]),
            reverse=True,
        )
        true_positives: list[float] = []
        false_positives: list[float] = []

        for detection in detections:
            image_id = str(detection["image_id"])
            image_ground_truth = ground_truth.get(image_id)
            if image_ground_truth is None or image_ground_truth["boxes"].numel() == 0:
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
                # VOC 官方语义：命中 difficult 目标的检测不进入 PR 曲线。
                continue
            elif not bool(image_ground_truth["detected"][best_index].item()):
                image_ground_truth["detected"][best_index] = True
                true_positives.append(1.0)
                false_positives.append(0.0)
            else:
                # 同一非 difficult GT 只能匹配一次，重复检测记为 FP。
                true_positives.append(0.0)
                false_positives.append(1.0)

        if positive_count == 0:
            per_class_ap[class_name] = math.nan
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

    finite_aps = [value for value in per_class_ap.values() if math.isfinite(value)]
    mean_ap = sum(finite_aps) / len(finite_aps) if finite_aps else math.nan
    return {
        "iou_threshold": iou_threshold,
        "per_class_ap": per_class_ap,
        "map": mean_ap,
    }


@torch.inference_mode()
def validate(model, criterion, dataloader, device):
    model.eval()
    results: list[dict] = []
    loss_sums: dict[str, float] = {}

    for images, padding_mask, targets in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        padding_mask = padding_mask.to(device)
        targets = [
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        outputs = model(images, padding_mask)
        loss_dict = criterion(outputs, targets)
        for name, value in loss_dict.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + value.item()
        results.extend(outputs_to_voc_results(outputs, targets))

    average_losses = {
        name: total / len(dataloader) for name, total in loss_sums.items()
    }
    return average_losses, results


def main() -> None:
    device = get_device()
    dataloader = build_dataloader(
        "val",
        batch_size=BATCH_SIZE,
        dataset_cls=DATASET_REGISTRY[DATASET_TYPE],
        shuffle=False,
    )
    model = build_model(False, device)
    criterion = SetCriterion(
        NUM_CLASSES,
        HungarianMatcher(),
        eos_coef=0.1,
    ).to(device)

    losses, results = validate(model, criterion, dataloader, device)
    print("\n验证损失:")
    for name, value in losses.items():
        print(f"{name}: {value:.4f}")
    print(f"total: {sum(losses.values()):.4f}")

    metrics = evaluate_voc(results, dataloader.dataset)
    print(f"\n{DATASET_TYPE.upper()} AP@0.5 (11-point):")
    for class_name in dataloader.dataset.classes:
        average_precision = metrics["per_class_ap"][class_name]
        display_value = (
            "N/A" if math.isnan(average_precision) else f"{average_precision:.4f}"
        )
        print(f"{class_name:>12}: {display_value}")
    print(f"mAP@0.5: {metrics['map']:.4f}")


if __name__ == "__main__":
    main()
