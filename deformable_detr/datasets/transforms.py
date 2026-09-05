"""RAOD 检测图片与边界框的同步变换。"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torchvision.transforms import functional as F

from ..utils.box_ops import box_xyxy_to_cxcywh

Target = dict[str, object]


def _resize_shape(
    image_size: tuple[int, int],
    short_side: int,
    max_size: int,
) -> tuple[int, int]:
    """按短边缩放并限制长边，返回 torchvision 使用的 (height, width)。"""
    width, height = image_size
    min_side = min(width, height)
    max_side = max(width, height)
    if max_side / min_side * short_side > max_size:
        short_side = int(round(max_size * min_side / max_side))

    if width < height:
        new_width = short_side
        new_height = int(round(height * short_side / width))
    else:
        new_height = short_side
        new_width = int(round(width * short_side / height))
    return new_height, new_width


def resize(
    image: Image.Image,
    target: Target | None,
    short_side: int,
    max_size: int,
) -> tuple[Image.Image, Target | None]:
    output_size = _resize_shape(image.size, short_side, max_size)
    resized_image = F.resize(image, output_size)

    if target is None:
        return resized_image, None

    old_width, old_height = image.size
    new_width, new_height = resized_image.size
    ratio_width = new_width / old_width
    ratio_height = new_height / old_height

    target = target.copy()
    boxes = target["boxes"]
    assert isinstance(boxes, torch.Tensor)
    scale = boxes.new_tensor(
        [ratio_width, ratio_height, ratio_width, ratio_height]
    )
    target["boxes"] = boxes * scale
    area = target["area"]
    assert isinstance(area, torch.Tensor)
    target["area"] = area * ratio_width * ratio_height
    target["size"] = torch.tensor([new_height, new_width], dtype=torch.int64)
    return resized_image, target


def horizontal_flip(
    image: Image.Image,
    target: Target | None,
) -> tuple[Image.Image, Target | None]:
    flipped_image = F.hflip(image)
    if target is None:
        return flipped_image, None

    width, _ = image.size
    target = target.copy()
    boxes = target["boxes"]
    assert isinstance(boxes, torch.Tensor)
    target["boxes"] = boxes[:, [2, 1, 0, 3]] * boxes.new_tensor(
        [-1, 1, -1, 1]
    ) + boxes.new_tensor([width, 0, width, 0])
    return flipped_image, target


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, image: Image.Image, target: Target | None):
        if random.random() < self.probability:
            return horizontal_flip(image, target)
        return image, target


class RandomResize:
    def __init__(self, sizes: Sequence[int], max_size: int):
        if not sizes:
            raise ValueError("sizes 不能为空")
        self.sizes = tuple(sizes)
        self.max_size = max_size

    def __call__(self, image: Image.Image, target: Target | None):
        return resize(image, target, random.choice(self.sizes), self.max_size)


class ToTensor:
    def __call__(self, image: Image.Image, target: Target | None):
        return F.to_tensor(image), target


class Normalize:
    """标准化图片，并把绝对 xyxy 框转为归一化 cxcywh。"""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = tuple(mean)
        self.std = tuple(std)

    def __call__(self, image: torch.Tensor, target: Target | None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None

        target = target.copy()
        boxes = target["boxes"]
        assert isinstance(boxes, torch.Tensor)
        height, width = image.shape[-2:]
        scale = boxes.new_tensor([width, height, width, height])
        target["boxes"] = (box_xyxy_to_cxcywh(boxes) / scale).clamp(0, 1)
        return image, target


def make_detection_transforms(image_set: str, debug: bool = False) -> Compose:
    """构建 RAOD 训练/验证变换。

    RAOD 目标普遍较小，因此训练增强只使用水平翻转和尺度变化，不做可能把
    小目标完全裁掉的随机裁剪。
    """
    if debug:
        train_sizes = [256, 288, 320]
        validation_size = 320
        max_size = 533
    else:
        train_sizes = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        validation_size = 800
        max_size = 1333

    normalize = Compose(
        [
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if image_set == "train":
        return Compose(
            [
                RandomHorizontalFlip(),
                RandomResize(train_sizes, max_size=max_size),
                normalize,
            ]
        )
    if image_set in {"val", "test"}:
        return Compose(
            [
                RandomResize([validation_size], max_size=max_size),
                normalize,
            ]
        )
    raise ValueError(f"image_set 应为 'train'、'val' 或 'test'，收到: {image_set!r}")
