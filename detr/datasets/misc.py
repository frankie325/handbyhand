"""将不同尺寸图片组成 DETR batch。"""

from __future__ import annotations

import torch


def pad_images(images: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """把不同尺寸图片补成一个 batch，并生成 True 表示 padding 的 mask。

    输入中每张图片为 [channels, height, width]；输出 images 为
    [batch_size, channels, max_height, max_width]，padding_mask 为
    [batch_size, max_height, max_width]。
    """
    if not images:
        raise ValueError("images 不能为空")
    if any(image.ndim != 3 for image in images):
        raise ValueError("每张图片都必须是 [channels, height, width] 的 Tensor")

    channels = images[0].shape[0]
    if any(image.shape[0] != channels for image in images):
        raise ValueError("同一个 batch 中的图片通道数必须一致")

    batch_size = len(images)
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    # 每个 batch 只补到该 batch 的最大高度和宽度，避免固定到全数据集最大尺寸。
    # 补充值为 0；这些位置随后会被 padding_mask 屏蔽，不参与 Transformer 注意力。
    padded_images = images[0].new_zeros(
        (batch_size, channels, max_height, max_width)
    )
    # 先把整张 mask 初始化为 True（padding），复制真实图像后再把对应区域设为 False。
    padding_mask = torch.ones(
        (batch_size, max_height, max_width),
        dtype=torch.bool,
        device=images[0].device,
    )

    for index, image in enumerate(images):
        _, height, width = image.shape
        # 图片放在左上角，padding 只会出现在右侧和下方。
        padded_images[index, :, :height, :width].copy_(image)
        padding_mask[index, :height, :width] = False

    return padded_images, padding_mask


def collate_fn(batch):
    """DataLoader 的组批函数，返回 images、padding_mask 和 targets。"""
    # 每张图片中的目标数量不同，targets 无法堆叠成规则 Tensor，因此保留为
    # 长度等于 batch_size 的 list；只有图片和 mask 会被堆叠。
    images, targets = zip(*batch)
    padded_images, padding_mask = pad_images(list(images))
    return padded_images, padding_mask, list(targets)
