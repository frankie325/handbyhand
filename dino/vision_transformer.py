# 一份干净、自包含、适配 DINO 多裁剪自监督训练的 VisionTransformer 实现。
#
# 主要特点（相对你 vit/ 下的旧版）：
#   1. 不依赖 transformers 库，纯用 PyTorch 标准模块，结构清晰、易读；
#   2. 自带 interpolate_pos_encoding：按实际输入分辨率对位置编码做双三次插值，
#      因此同一份权重能处理 224 的 global crop 和 96 的 local crop（DINO 多裁剪必需）；
#   3. forward 返回 CLS token 特征 [B, D]，配合 dino/utils.py 的 MultiCropWrapper + DINOHead 使用；
#   4. 附带 DINOHead（带 weight_norm、冻结 last_layer 尺度）与 build_model 工厂函数。
#
# 代码主体基于 facebookresearch/dino 官方实现（timm 风格）改写，并加了中文注释。

import math
from functools import partial
from typing import Any, Callable

import torch
import torch.nn as nn

from .utils import trunc_normal_


# ---------------------------------------------------------------------------
# 随机深度 DropPath（Stochastic Depth）：训练时按概率把整条残差路径「丢弃」
# ---------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # 只对 batch 维做随机掩码，其它维保持 [1]，便于广播
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化：>1 留下，<1 丢弃
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# MLP：Transformer 里每个 block 的前馈层
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# 自注意力：标准的 QKV -> softmax(QK^T)V，纯手写以便理解
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5  # 1/sqrt(d_head)，缩放点积

        # 一次线性把输入映射成拼接的 Q、K、V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # 拆成 3 份 Q/K/V，并reshape成 [B, heads, N, d_head]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数 = QK^T / sqrt(d_head)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 用注意力分数对 V 加权求和，再合并回 [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


# ---------------------------------------------------------------------------
# Transformer block：Pre-LN 结构（先 LayerNorm 再子层，再残差）
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        # 注意力分支
        x = x + self.drop_path(y)
        # 前馈分支
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch Embedding：用 stride=patch_size 的卷积把图切成 patch 并投影到 embed_dim
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 224/16=14 -> 196 个 patch

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, 3, H, W] -> conv -> [B, D, H/p, W/p] -> flatten -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# VisionTransformer 主干
# ---------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token：一个可学习的「全局聚合」向量，和每个 patch 拼在一起
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码：[1, num_patches+1, D]，其中 +1 是 CLS 的位置
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 各 block 的随机深度概率按层线性递增（浅层丢弃少、深层丢弃多）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # 分类头：DINO 自监督时 num_classes=0，head 就是 Identity（不分类，
        # 由外部的 DINOHead 接在 CLS 特征上做原型投影）。
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        让一份固定分辨率（224）学好的位置编码（positional embedding），
        能适配任意输入分辨率，尤其是 DINO 多裁剪里的 96 局部 crop
        把固定尺寸学到的 pos_embed 双三次插值到实际输入分辨率（支持多裁剪）。

        x: [B, 1 + 36, D]，
        w/h: 原始图片宽高 96。
        """
        npatch = x.shape[1] - 1  # 实际 patch 数 36
        N = self.pos_embed.shape[1] - 1  # 注册时 patch 数（224 下为 196）
        if npatch == N and w == h:
            return self.pos_embed  # 尺寸一致，直接用

        class_pos_embed = self.pos_embed[:, 0]  # (1, D) CLS 的位置编码
        patch_pos_embed = self.pos_embed[:, 1:]  # (1, 196, D) 各 patch 的位置编码
        dim = x.shape[-1]

        # 实际 / 注册时的 patch 网格边长: 96 / 16 = 6
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # 加 0.1 避免插值时的浮点误差（见 dino issue #8）
        w0, h0 = w0 + 0.1, h0 + 0.1
        side = int(math.sqrt(N))  # 注册时是正方形网格（14）

        # 把 (1, 196, 768) reshape 成 (1, 768, 14, 14) 后做双三次插值到 (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, side, side, dim).permute(0, 3, 1, 2),
            scale_factor=(
                w0 / math.sqrt(N),
                h0 / math.sqrt(N),
            ),  # scale_factor ≈ (6.1/14, 6.1/14) ≈ (0.4357, 0.4357) 把边长 14 的位置网格整体缩小到约 6，使输出变成 6×6
            mode="bicubic",
        )  # (1, 768, 6, 6)
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )

        # (1, 6, 6, 768) -> (1, 36, 768)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # 拼回 CLS 位置编码 -> (1, 37, 768)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch 投影 -> [B, N, D]

        # 给每个样本补上 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + N, D]

        # 加位置编码（按实际 w/h 插值）
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # 取 CLS token 作为整张图的表征；num_classes=0 时 self.head 是 Identity，原样返回
        return self.head(x[:, 0])

    # ---- 下面两个方法是官方 DINO 里用来可视化/取中间层特征的，训练用不到 ----
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


# ---------------------------------------------------------------------------
# DINOHead：接在 CLS 特征上的投影头，输出「原型」分布
#   in_dim -> (2048 -> 2048 ->) 256 -> L2 归一化 -> weight_norm(256 -> out_dim)
# 关键点：last_layer 用 weight_norm 且冻结 weight_g=1，只学方向，防止坍塌。
# ---------------------------------------------------------------------------
class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        # 3 层 MLP：768→2048→2048→256，最后输出 [batch_size, 256]
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # 权重归一化（Weight Normalization）：把线性层权重 w（形状 [out, in]）重参数化为
        #   有效权重 w = weight_g * (weight_v / ||weight_v||)
        # - weight_g（形状 [out, 1]）：尺度/模长，每个输出单元一个标量
        # - weight_v（形状 [out, in]）：方向向量
        # 作用：把权重的「大小」与「方向」解耦。DINO 进一步冻结 weight_g=1，
        # 使最后一层只学方向，配合 DINOLoss 的 centering 稳定训练、防止表征坍塌。
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)  # [batch_size, bottleneck_dim=256]
        x = nn.functional.normalize(x, dim=-1, p=2)  # L2 归一化到单位向量
        x = self.last_layer(x)  # [batch_size, out_dim]
        return x


# ---------------------------------------------------------------------------
# 工厂函数：按名字快速构造不同规模的 ViT
# ---------------------------------------------------------------------------
def vit_tiny_patch16(**kwargs: Any) -> VisionTransformer:
    return VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_small_patch16(**kwargs: Any) -> VisionTransformer:
    return VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base_patch16(**kwargs: Any) -> VisionTransformer:
    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base_patch8(**kwargs: Any) -> VisionTransformer:
    return VisionTransformer(
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


MODEL_BUILDERS: dict[str, Callable[..., VisionTransformer]] = {
    "vit_t_16": vit_tiny_patch16,
    "vit_s_16": vit_small_patch16,
    "vit_b_16": vit_base_patch16,
    "vit_b_8": vit_base_patch8,
}


def build_model(args: Any) -> VisionTransformer:
    """根据 args 创建 ViT，接口与 dino/train.py 的 Namespace 兼容。"""
    model_name = getattr(args, "model", "vit_b_16")
    model_name = model_name.lower().replace("-", "_").replace("/", "_")
    if model_name not in MODEL_BUILDERS:
        available = ", ".join(MODEL_BUILDERS)
        raise ValueError(f"不支持的模型: {model_name}，可选: {available}")

    model_kwargs = {
        "img_size": args.img_size,
        "num_classes": args.num_classes,
    }
    for name in ("in_chans", "drop", "attn_drop", "drop_path_rate"):
        if hasattr(args, name):
            model_kwargs[name] = getattr(args, name)

    return MODEL_BUILDERS[model_name](**model_kwargs)
