"""Tiny COCO 数据集和 DataLoader 构建入口。"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .coco import CocoDetection
from .misc import collate_fn
from .transforms import make_coco_transforms


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def build_dataset(
    image_set: str,
    root: str | Path = DEFAULT_DATA_ROOT,
    debug: bool = False,
) -> CocoDetection:
    """构建 train 或 val 数据集。"""
    if image_set not in {"train", "val"}:
        raise ValueError(f"image_set 应为 'train' 或 'val'，收到: {image_set!r}")

    root = Path(root)
    year = "2014"
    # train -> data/train2014 + instances_train2014.json
    # val   -> data/val2014   + instances_val2014.json
    image_dir = root / f"{image_set}{year}"
    annotation_file = root / "annotations" / f"instances_{image_set}{year}.json"

    # CocoDetection 负责读取图片/标注；make_coco_transforms 决定该划分
    # 使用训练随机增强还是验证确定性预处理。
    return CocoDetection(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transforms=make_coco_transforms(image_set, debug=debug),
    )


def build_dataloader(
    image_set: str,
    root: str | Path = DEFAULT_DATA_ROOT,
    batch_size: int = 2,
    shuffle: bool | None = None,
    num_workers: int = 0,
    debug: bool = False,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """构建能直接送入 Detr.forward 的 DataLoader。"""
    dataset = build_dataset(image_set=image_set, root=root, debug=debug)
    if shuffle is None:
        # 训练集默认打乱；验证集保持固定顺序，便于稳定复现评估结果。
        shuffle = image_set == "train"

    # collate_fn 会把不同尺寸图片 padding 到当前 batch 的最大尺寸，最终每轮返回：
    # images、padding_mask、targets，可直接调用 model(images, padding_mask)。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
