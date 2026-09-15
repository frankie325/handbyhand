import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        # 输入3*224*224的图片
        self.img_size = (img_size, img_size)
        # patch块的大小3*16*16
        self.patch_size = (patch_size, patch_size)

        # 水平和垂直方向 224 / 16 = 14个patch
        # 14 * 14 = 196个patch
        self.num_patches = (img_size // patch_size) ** 2

        # 经过卷积层 768*14*14
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def forward(self, x):
        # x.shape = (batch_size, 3, 224, 224)

        x = self.proj(x)  # (batch_size, 768, 14, 14)

        # (batch_size, 768, 14*14) -> (batch_size, 196, 768)
        x = x.flatten(2).transpose(1, 2)

        return x
