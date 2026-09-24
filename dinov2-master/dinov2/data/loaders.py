"""根据配置创建 Dataset、Sampler 和 PyTorch DataLoader。"""

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler

from .datasets import ImageFolderDataset, ImageNet, ImageNet22k
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    """训练入口可选择的采样策略。"""

    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_sample_transform(
    image_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """把独立的 image/target transform 合并成 torchvision 接口。

    当前 Dataset 基类已通过 ``VisionDataset`` 完成同样的组合，因此该工具
    主要保留给外部 Dataset 适配时使用。
    """

    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    """解析 ``Name:key=value:key=value`` 格式的数据集描述。

    例如：
    ``ImageNet:split=TRAIN:root=D:/imagenet:extra=D:/imagenet-extra``。
    路径本身不要包含冒号分隔符；Windows 盘符场景更推荐在调用前直接
    构造 Dataset，或把路径改成项目支持的统一配置字段。
    """

    tokens = dataset_str.split(":")
    name = tokens[0]
    kwargs = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"invalid dataset option: {token!r}")
        key, value = token.split("=", maxsplit=1)
        if key not in ("root", "extra", "split"):
            raise ValueError(f"unsupported dataset option: {key!r}")
        kwargs[key] = value

    if name == "ImageNet":
        dataset_class = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        dataset_class = ImageNet22k
    elif name == "ImageFolder":
        dataset_class = ImageFolderDataset
    else:
        raise ValueError(f"unsupported dataset {name!r}")

    return dataset_class, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """根据字符串创建数据集，并挂载图像/标签变换。"""

    logger.info('using dataset: "%s"', dataset_str)
    dataset_class, kwargs = _parse_dataset_str(dataset_str)
    dataset = dataset_class(transform=transform, target_transform=target_transform, **kwargs)
    logger.info("# of dataset samples: %s", f"{len(dataset):,d}")
    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    """只创建索引流，不读取图片。"""

    sample_count = len(dataset)
    if type == SamplerType.INFINITE:
        if size > 0:
            raise ValueError("INFINITE sampler does not accept a finite size")
        return InfiniteSampler(sample_count=sample_count, shuffle=shuffle, seed=seed, advance=advance)

    if type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        if size > 0:
            raise ValueError("SHARDED_INFINITE sampler does not accept a finite size")
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=type == SamplerType.SHARDED_INFINITE_NEW,
        )

    if type == SamplerType.EPOCH:
        if advance > 0:
            raise NotImplementedError("EpochSampler does not support advance")
        return EpochSampler(
            size=size if size > 0 else sample_count,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )

    if type == SamplerType.DISTRIBUTED:
        if size > 0 or advance > 0:
            raise ValueError("DistributedSampler does not accept size/advance here")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """构造训练 DataLoader。

    shuffle 已由自定义 sampler 实现，所以不能再把 ``shuffle=True`` 直接
    传给 DataLoader，否则 PyTorch 会认为 sampler 与 shuffle 冲突。
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    # persistent_workers 只有在 num_workers > 0 时合法。
    keep_workers = persistent_workers and num_workers > 0
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=keep_workers,
        collate_fn=collate_fn,
    )
    return data_loader
