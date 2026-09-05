"""RAOD 数据集和 DataLoader 构建函数。"""

from __future__ import annotations

from torch.utils.data import DataLoader

from ..config import DATA_ROOT, TRAIN_RATIO
from .misc import collate_fn
from .raod import RaodDetection
from .transforms import make_detection_transforms


def build_dataset(image_set: str, debug: bool = False) -> RaodDetection:
    return RaodDetection(
        root=DATA_ROOT,
        image_set=image_set,
        transforms=make_detection_transforms(image_set, debug=debug),
        train_ratio=TRAIN_RATIO,
    )


def build_dataloader(
    image_set: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool | None = None,
    debug: bool = False,
) -> DataLoader:
    dataset = build_dataset(image_set, debug=debug)
    if shuffle is None:
        shuffle = image_set == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
