from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from detr.config import (
    BATCH_SIZE,
    D_FF,
    D_MODEL,
    DROPOUT,
    MODELS_DIR,
    N_HEAD,
    N_LAYER,
    NUM_CLASSES,
    NUM_QUERIES,
)
from detr.datasets.build import DEFAULT_DATA_ROOT, build_dataloader
from detr.loss.criterion import SetCriterion
from detr.loss.matcher import HungarianMatcher
from detr.model.detr import Detr
from detr.utils.bos_ops import box_cxcywh_to_xyxy


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device) -> Detr:
    # checkpoint 已包含 backbone 权重，因此不再下载预训练 ResNet。
    model = Detr(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        d_model=D_MODEL,
        N=N_LAYER,
        d_ff=D_FF,
        n_head=N_HEAD,
        dropout=DROPOUT,
        pretrained_backbone=False,
    )

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    return model.to(device).eval()


def targets_to_device(targets, device):
    return [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def outputs_to_coco_results(outputs, targets, category_ids):
    """把归一化 cxcywh 转为 COCO 需要的像素 xywh。"""
    probabilities = outputs["pred_logits"].softmax(dim=-1)

    # 只在COCO真实存在的80个category_id中选择类别，
    # 自动排除no-object、类别0和COCO类别ID的空缺。
    category_ids = torch.as_tensor(
        category_ids,
        device=probabilities.device,
    )
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
    loss_totals = defaultdict(float)
    coco_results = []
    number_of_batches = 0
    category_ids = dataloader.dataset.category_ids

    for images, padding_mask, targets in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        padding_mask = padding_mask.to(device)
        targets = targets_to_device(targets, device)

        outputs = model(images, padding_mask)
        loss_dict = criterion(outputs, targets)

        for name, value in loss_dict.items():
            loss_totals[name] += value.item()

        coco_results.extend(
            outputs_to_coco_results(outputs, targets, category_ids)
        )
        number_of_batches += 1

    average_losses = {
        name: value / number_of_batches
        for name, value in loss_totals.items()
    }

    return average_losses, coco_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODELS_DIR / "best.pth",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="使用较小输入尺寸，只用于快速检查",
    )
    args = parser.parse_args()

    device = select_device()
    print("设备:", device)

    dataloader = build_dataloader(
        "val",
        root=args.data_root,
        batch_size=args.batch_size,
        shuffle=False,
        debug=args.debug,
    )

    model = load_model(args.checkpoint, device)
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