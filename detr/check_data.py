"""VOC2007 数据管线检查脚本；运行：uv run python -m detr.check_data。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from detr.config import NUM_CLASSES
from detr.datasets import VOC_CLASSES, build_dataloader, build_dataset
from detr.datasets.build import DEFAULT_DATA_ROOT
from detr.loss.criterion import SetCriterion
from detr.loss.matcher import HungarianMatcher
from detr.model.detr import Detr


def check_targets(targets: list[dict]) -> None:
    for target in targets:
        boxes = target["boxes"]
        labels = target["labels"]
        assert boxes.ndim == 2 and boxes.shape[-1] == 4
        assert torch.isfinite(boxes).all()
        assert ((boxes >= 0) & (boxes <= 1)).all()
        assert labels.dtype == torch.int64
        assert ((labels >= 0) & (labels < len(VOC_CLASSES))).all()
        assert len(boxes) == len(labels) == len(target["area"])
        assert isinstance(target["image_id"], str)


def check_padding_mask(
    padding_mask: torch.Tensor,
    targets: list[dict],
) -> None:
    assert padding_mask.dtype == torch.bool
    for index, target in enumerate(targets):
        height, width = target["size"].tolist()
        assert not padding_mask[index, :height, :width].any()
        if height < padding_mask.shape[1]:
            assert padding_mask[index, height:, :].all()
        if width < padding_mask.shape[2]:
            assert padding_mask[index, :height, width:].all()


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 DETR VOC2007 数据管线")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--full-resolution",
        action="store_true",
        help="使用论文默认 800/1333 尺度；默认使用较小尺度加快检查",
    )
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    debug = not args.full_resolution
    train_dataset = build_dataset("train", root=args.data_root, debug=debug)
    val_dataset = build_dataset("val", root=args.data_root, debug=debug)
    assert len(train_dataset) == 2501
    assert len(val_dataset) == 2510
    assert train_dataset.classes == val_dataset.classes == VOC_CLASSES
    assert len(train_dataset.class_to_idx) == NUM_CLASSES == 20
    assert set(train_dataset.ids).isdisjoint(val_dataset.ids)

    first_image_id = train_dataset.ids[0]
    first_ground_truth = train_dataset.get_ground_truth(first_image_id)
    assert first_ground_truth["image_id"] == first_image_id
    assert first_ground_truth["difficult"].dtype == torch.bool
    assert len(first_ground_truth["boxes"]) == len(first_ground_truth["difficult"])
    assert (first_ground_truth["area"] > 0).all()

    image, target = val_dataset[0]
    assert image.ndim == 3 and image.shape[0] == 3
    check_targets([target])

    loader = build_dataloader(
        "val",
        root=args.data_root,
        batch_size=args.batch_size,
        shuffle=False,
        debug=debug,
    )
    images, padding_mask, targets = next(iter(loader))
    assert images.ndim == 4
    assert padding_mask.shape == images.shape[:1] + images.shape[-2:]
    check_targets(targets)
    check_padding_mask(padding_mask, targets)

    print(f"train: {len(train_dataset)} 张，val: {len(val_dataset)} 张")
    print(f"classes: {len(VOC_CLASSES)}")
    print(
        f"difficult: train={train_dataset.num_difficult}, "
        f"val={val_dataset.num_difficult}"
    )
    print(f"images: {tuple(images.shape)}")
    print(f"padding_mask: {tuple(padding_mask.shape)}")

    if not args.skip_model:
        model = Detr(
            num_classes=NUM_CLASSES,
            num_queries=100,
            d_model=64,
            N=1,
            d_ff=128,
            n_head=8,
            pretrained_backbone=False,
            train_backbone=False,
        ).eval()
        with torch.no_grad():
            outputs = model(images, padding_mask)
            losses = SetCriterion(
                NUM_CLASSES,
                HungarianMatcher(),
                eos_coef=0.1,
            )(outputs, targets)
        assert outputs["pred_logits"].shape == (images.shape[0], 100, 21)
        assert outputs["pred_boxes"].shape == (images.shape[0], 100, 4)
        assert all(torch.isfinite(loss).all() for loss in losses.values())
        print(f"pred_logits: {tuple(outputs['pred_logits'].shape)}")
        print(f"pred_boxes: {tuple(outputs['pred_boxes'].shape)}")
        print(f"loss: {sum(loss.item() for loss in losses.values()):.4f}")

    print("数据管线检查通过")


if __name__ == "__main__":
    main()
