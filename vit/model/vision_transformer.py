import math
import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding
from transformers import EncoderLayer
from transformers import MultiHeadedAttention


# 前馈层
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# 编码器块
class Block(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        drop,
        attn_drop,
        mlp_ratio=4.0,
    ):
        super(Block, self).__init__()
        attn = MultiHeadedAttention(num_heads, embed_dim, attn_drop)
        feed_forward = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)
        self.encoder = EncoderLayer(embed_dim, attn, feed_forward, drop)

    def forward(self, x):
        # vit不需要传padding_mask
        return self.encoder(x, None)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=20,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,  # 前馈层隐藏层的比例
        # qkv_bias=False,
        # qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)

        # 可学习的分类标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, 768)

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )  # (1, 197, 768)

        # dropout正则化
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList(
            [
                Block(num_heads, embed_dim, drop, attn_drop, mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # 检测头
        self.head = nn.Linear(embed_dim, num_classes)

        # 对整个模型的所有子模块递归调用init_weights
        self.apply(self.init_weights)
        # 带边界约束的正态初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x.shape = (batch_size, 3, 224, 224)

        x = self.patch_embed(x)  # (batch_size, 196, 768)

        batch_size = x.shape[0]
        x = torch.cat(
            [self.cls_token.expand(batch_size, -1, -1), x], dim=1
        )  # (batch_size, 197, 768)

        # 融合位置编码
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)
        # (batch_size, 197, 768)
        x = self.norm(x)

        # 取cls_token作为分类检测头 (batch_size, num_classes)
        output = self.head(x[:, 0])

        return output

    @staticmethod
    def init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_in = module.in_channels * math.prod(module.kernel_size)
            nn.init.trunc_normal_(module.weight, std=math.sqrt(1.0 / fan_in))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
