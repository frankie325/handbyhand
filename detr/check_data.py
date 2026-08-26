"""数据管线检查脚本；运行：conda run -n handbyhand python -m detr.check_data。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from detr.datasets import build_dataloader, build_dataset
from detr.datasets.build import DEFAULT_DATA_ROOT
from detr.model.detr import Detr


def check_targets(targets: list[dict[str, torch.Tensor]]) -> None:
    for target in targets:
        boxes = target["boxes"]
        assert boxes.ndim == 2 and boxes.shape[-1] == 4
        assert torch.isfinite(boxes).all()
        assert ((boxes >= 0) & (boxes <= 1)).all()
        assert target["labels"].dtype == torch.int64
        assert (target["labels"] >= 1).all()
        assert (target["labels"] <= 90).all()
        assert not target["iscrowd"].bool().any()


def check_padding_mask(
    padding_mask: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
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
    parser = argparse.ArgumentParser(description="检查 DETR Tiny COCO 数据管线")
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
    assert len(train_dataset) == len(train_dataset.coco.getImgIds()) == 200
    assert len(val_dataset) == len(val_dataset.coco.getImgIds()) == 50
    assert len(train_dataset.category_ids) == len(train_dataset.coco.getCatIds()) == 80

    first_image_id = train_dataset.ids[0]
    first_image_info = train_dataset.coco.loadImgs([first_image_id])[0]
    first_annotation_ids = train_dataset.coco.getAnnIds(imgIds=[first_image_id])
    first_annotations = train_dataset.coco.loadAnns(first_annotation_ids)
    assert first_image_info["id"] == first_image_id
    assert all(annotation["image_id"] == first_image_id for annotation in first_annotations)

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
    print(f"pycocotools categories: {len(train_dataset.coco.getCatIds())}")
    print(f"images: {tuple(images.shape)}")
    print(f"padding_mask: {tuple(padding_mask.shape)}")

    if not args.skip_model:
        model = Detr(
            num_classes=91,
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
        assert outputs["pred_logits"].shape == (images.shape[0], 100, 92)
        assert outputs["pred_boxes"].shape == (images.shape[0], 100, 4)
        print(f"pred_logits: {tuple(outputs['pred_logits'].shape)}")
        print(f"pred_boxes: {tuple(outputs['pred_boxes'].shape)}")

    print("数据管线检查通过")


if __name__ == "__main__":
    main()
