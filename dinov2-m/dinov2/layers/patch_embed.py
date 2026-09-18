import torch.nn as nn
from torch import Tensor
from typing import Callable


class PatchEmbed(nn.Module):

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ):

        image_HW = (img_size, img_size)
        patch_HW = (patch_size, patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]  # 16 * 16 = 196

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_HW,
            stride=patch_HW,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x.shape = (batch_size, channels, H, W)
        _, _, h, w = x.shape
        patch_H, patch_W = self.patch_size

        assert h % patch_H == 0, f"输入图片高度 {h} 不能整除patch：{patch_H}"
        assert w % patch_W == 0, f"输入图片高度 {w} 不能整除patch：{patch_W}"

        x = self.proj(x)  #  (batch_size, channels, 224 / 16, 224 / 16)
        h, w = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, 196, 768)
        x = self.norm(x)

        if not self.flatten_embedding:
        # 不展平
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
