"""把不同尺寸的检测图片补齐成一个 batch。"""

from __future__ import annotations

import torch


def pad_images(images: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """返回补齐后的图片和 padding mask；mask 中 True 表示补齐区域。"""
    if not images:
        raise ValueError("images 不能为空")
    if any(image.ndim != 3 for image in images):
        raise ValueError("每张图片必须是 [channels, height, width] Tensor")

    channels = images[0].shape[0]
    if any(image.shape[0] != channels for image in images):
        raise ValueError("同一个 batch 中的图片通道数必须一致")

    batch_size = len(images)
    max_height = max(image.shape[-2] for image in images)
    max_width = max(image.shape[-1] for image in images)
    padded = images[0].new_zeros(batch_size, channels, max_height, max_width)
    padding_mask = torch.ones(
        batch_size,
        max_height,
        max_width,
        dtype=torch.bool,
        device=images[0].device,
    )

    for index, image in enumerate(images):
        height, width = image.shape[-2:]
        padded[index, :, :height, :width].copy_(image)
        padding_mask[index, :height, :width] = False

    return padded, padding_mask


def collate_fn(batch):
    """DataLoader 组批：boxes 数量不同，因此 targets 保持为 list。"""
    images, targets = zip(*batch)
    padded_images, padding_mask = pad_images(list(images))
    return padded_images, padding_mask, list(targets)
