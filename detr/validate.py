from __future__ import annotations
import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from detr.datasets.build import build_dataloader
from detr.loss.criterion import SetCriterion
from detr.loss.matcher import HungarianMatcher
from detr.utils.bos_ops import box_cxcywh_to_xyxy
from .model.build import build_model
from .utils.common import get_device
from .config import (
    BATCH_SIZE,
    NUM_CLASSES,
)


def outputs_to_coco_results(outputs, targets, category_ids):
    """把归一化 cxcywh 转为 COCO 需要的像素 xywh。"""
    probabilities = outputs["pred_logits"].softmax(dim=-1)

    # 只在COCO真实存在的80个category_id中选择类别，
    # 自动排除no-object、类别0和COCO类别ID的空缺。
    category_ids = torch.as_tensor(
        category_ids,
        device=probabilities.device,
    ) #
    scores, category_positions = probabilities.index_select(
        dim=-1,
        index=category_ids,
    ).max(dim=-1)
    labels = category_ids[category_positions]

    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
    results = []

    for image_scores, image_labels, image_boxes, target in zip(
        scores, labels, boxes, targets
    ):
        original_height, original_width = target["orig_size"].tolist()

        scale = image_boxes.new_tensor(
            [
                original_width,
                original_height,
                original_width,
                original_height,
            ]
        )
        image_boxes = image_boxes * scale
        image_boxes[:, 0::2].clamp_(0, original_width)
        image_boxes[:, 1::2].clamp_(0, original_height)

        # xyxy → COCO xywh
        image_boxes[:, 2:] -= image_boxes[:, :2]
        image_id = int(target["image_id"].item())

        for score, label, box in zip(
            image_scores.tolist(),
            image_labels.tolist(),
            image_boxes.tolist(),
        ):
            if box[2] <= 0 or box[3] <= 0:
                continue

            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box,
                    "score": float(score),
                }
            )

    return results


@torch.inference_mode()
def validate(model, criterion, dataloader, device):
    model.eval()
    coco_results = []
    loss_sums: dict[str, float] = {}
    category_ids = dataloader.dataset.category_ids

    for images, padding_mask, targets in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        padding_mask = padding_mask.to(device)
        # targets: list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]
        targets = [
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        outputs = model(images, padding_mask)
        loss_dict = criterion(outputs, targets)
        # 分别累计每一项损失
        for name, value in loss_dict.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + value.item()

        coco_results.extend(outputs_to_coco_results(outputs, targets, category_ids))

    average_losses = {
        name: total / len(dataloader) for name, total in loss_sums.items()
    }

    return average_losses, coco_results


def main():
    device = get_device()

    dataloader = build_dataloader(
        "val",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_model(False, device)

    criterion = SetCriterion(
        NUM_CLASSES,
        HungarianMatcher(),
        eos_coef=0.1,
    ).to(device)

    losses, results = validate(
        model,
        criterion,
        dataloader,
        device,
    )

    print("\n验证损失:")
    for name, value in losses.items():
        print(f"{name}: {value:.4f}")
    print(f"total: {sum(losses.values()):.4f}")

    # COCO AP评估
    coco_ground_truth = dataloader.dataset.coco
    coco_predictions = coco_ground_truth.loadRes(results)

    evaluator = COCOeval(
        coco_ground_truth,
        coco_predictions,
        iouType="bbox",
    )
    evaluator.params.imgIds = dataloader.dataset.ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()
