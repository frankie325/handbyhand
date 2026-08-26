import torch
import torch.nn as nn
from ...utils import clones
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # ============================================================
        # 笔记：先归一化(Pre-LN) vs 后归一化(Post-LN)
        # ============================================================
        # 本实现是 Pre-LN（先归一化）：
        #     y = x + Dropout(sublayer(LayerNorm(x)))
        # 即先对输入做 LayerNorm，再进 sublayer（注意力/前馈），最后与原始 x 残差相加。
        #
        # 对应的 Post-LN（原 Transformer 论文写法，后归一化）：
        #     y = LayerNorm(x + Dropout(sublayer(x)))
        # 即先残差相加，再对整体做 LayerNorm。
        #
        # 两者核心区别：
        #   1. 归一化位置：Pre-LN 在 sublayer 之前；Post-LN 在残差加法之后。
        #   2. 梯度/稳定性：Pre-LN 的残差路径 x 不经过 norm，恒等映射直接透传，
        #      梯度更稳，不易梯度消失/爆炸，深层网络（几十~上百层）也能稳定训练，
        #      对学习率和 warmup 不那么敏感。
        #   3. 性能：Post-LN 调好超参（尤其需要 warmup）后通常略高，但训练不稳；
        #      Pre-LN 更易训练、更稳，现代大模型（GPT-2/3、LLaMA 等）普遍采用 Pre-LN。
        #   4. 语义：Pre-LN 每个 sublayer 看到的是归一化后的“干净”分布；
        #      Post-LN 的 sublayer 拿到残差累加的原始信号，最后统一归一化。
        # ============================================================
        return x + self.dropout(sublayer(self.norm(x)))
