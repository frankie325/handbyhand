"""DINOv2 数据增强中可复用的基础变换。

该模块不负责生成多视图；它只提供图像模糊、Tensor 转换、ImageNet
标准化，以及下游分类任务常用的训练/验证变换。
"""

from typing import Sequence

import torch
from torchvision import transforms


# DINOv2 处理普通 RGB 图像时沿用 ImageNet 的通道统计量。
# Normalize 执行 (x - mean) / std，其中 ToTensor 已将像素缩放到 [0, 1]。
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class GaussianBlur(transforms.RandomApply):
    """以概率 ``p`` 对 PIL 图像执行高斯模糊。

    DINO 的不同视图使用不同的模糊概率：第一个全局视图最强，第二个
    全局视图较弱，局部视图介于两者之间。sigma 会在给定范围内随机
    采样，因此每次调用的模糊强度也可能不同。
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        blur = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        # RandomApply 的 p 就是“执行内部变换”的概率。
        super().__init__(transforms=[blur], p=p)


class MaybeToTensor(transforms.ToTensor):
    """输入不是 Tensor 时才执行 ``ToTensor``。

    普通 RGB 解码器返回 PIL.Image，需要转换；多通道科学图像解码器可能
    已返回 Tensor，此时重复 ToTensor 会报错或改变数据含义。
    """

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    """构造按通道标准化变换，允许下游任务覆盖 mean/std。"""

    return transforms.Normalize(mean=mean, std=std)


def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    """构造下游分类训练变换，不用于 DINO 多裁剪预训练。"""

    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend([MaybeToTensor(), make_normalize_transform(mean=mean, std=std)])
    return transforms.Compose(transforms_list)


def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    """构造确定性的下游分类验证变换。"""

    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
