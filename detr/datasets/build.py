"""VOC2007 数据集和 DataLoader 构建入口。"""

from __future__ import annotations


from torch.utils.data import DataLoader, Dataset

from ..config import DATASET_ROOT

from .misc import collate_fn
from .transforms import make_detection_transforms
from .voc import VocDetection


def build_dataset(
    image_set: str,
    dataset_cls: type[Dataset] = VocDetection,
    debug: bool = False,
) -> Dataset:
    """构建 train 或 val 数据集。"""
    if image_set not in {"train", "val"}:
        raise ValueError(f"image_set 应为 'train' 或 'val'，收到: {image_set!r}")

    transforms = make_detection_transforms(image_set, debug=debug)
    return dataset_cls(
        root=DATASET_ROOT,
        image_set=image_set,
        transforms=transforms,
    )


def build_dataloader(
    image_set: str,
    dataset_cls: type[Dataset] = VocDetection,
    batch_size: int = 2,
    shuffle: bool | None = None,
    num_workers: int = 0,
    debug: bool = False,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """构建能直接送入 Detr.forward 的 DataLoader。"""
    dataset = build_dataset(image_set=image_set, dataset_cls=dataset_cls, debug=debug)
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
