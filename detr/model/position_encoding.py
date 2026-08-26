import math

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """DETR 官方二维正弦位置编码，兼容带 mask 的变长特征图

    公式（对每个坐标轴 pos = y 或 x）:
        PE(pos, 2i)     = sin(pos / 10000^(2i/num_pos_feats))
        PE(pos, 2i + 1) = cos(pos / 10000^(2i/num_pos_feats))
    行坐标 y 和列坐标 x 各编码 num_pos_feats 维，最后沿通道维拼接得到 d_model 维。
    """

    def __init__(
        self, num_pos_feats=128, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 每个坐标轴占的通道数 = d_model / 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize=True 时才允许传 scale")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        # x:    [batch, channels, height, width]  (backbone 输出的特征图)
        # mask: [batch, height, width]  True 表示 padding 区域
        b, _, h, w = x.shape

        if mask is None:
            # 没有 mask 时直接按坐标计算
            y_embed = (
                torch.arange(1, h + 1, dtype=torch.float32, device=x.device)
                .view(1, h, 1)
                .repeat(b, 1, w)
            )
            x_embed = (
                torch.arange(1, w + 1, dtype=torch.float32, device=x.device)
                .view(1, 1, w)
                .repeat(b, h, 1)
            )
        else:
            # ?反正 attention mask 会在注意力里把它们全部屏蔽，那位置编码还需要mask吗
            # 答案是：不是多余的。attention mask 确实屏蔽了 padding 对注意力的影响，但位置编码里的 mask 还有一个它替代不了的职责——保证归一化分母正确，让有效区域的位置编码铺满 [0, 2π]，分辨率最大
            not_mask = ~mask  # [b, h, w]
            y_embed = not_mask.cumsum(
                1, dtype=torch.float32
            )  # 行坐标, padding 处不累计
            x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 列坐标

        if self.normalize:
            eps = 1e-6
            # 除以最后一个有效位置（最大值）进行缩放, 缩放到 [0, 2π]
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        """
        频率表: dim_t[i] = 10000^(2*(i//2)/num_pos_feats), 相邻两维共享频率
        dim_t.shape [num_pos_feats]
        dim_t（输入）	[0, 1, 2, 3, 4, 5, ...]
        dim_t // 2	   [0, 0, 1, 1, 2, 2, ...]
        """
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # x_embed.shape: [b, h, w]
        # y_embed.shape: [b, h, w]
        #  x_embed[..., None]表示在最后一维增加一个维度，形状变为 [b, h, w, 1]
        # 右边的 dim_t 形状 [num_pos_feats] 先被当作 [1, 1, 1, num_pos_feats]
        # 与左边的 [b, h, w, 1] 对齐后广播 → [b, h, w, num_pos_feats]
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        # 偶通道取 sin, 奇通道取 cos, 再交错拼接 -> [sin, cos, sin, cos, ...]
        """
        sin部分:  [s0, s1, s2]
        cos部分:  [c0, c1, c2]

        stack(..., dim=-1) → 
        [
            [s0, c0],
            [s1, c1],
            [s2, c2],
        ]
        形状: [3, 2]
        """

        # stack后实际pos_x 和 pos_y 的形状是 [b, h, w, num_pos_feats / 2, 2]
        # flatten(3) 把最后两维展平 [b, h, w, num_pos_feats]
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
        ).flatten(3)

        # 先 y 后 x 拼接: [b, h, w, d_model]  ->permute-> [b, d_model, h, w]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
