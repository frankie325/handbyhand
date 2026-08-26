import torch
import torch.nn as nn
import math
from ...utils import clones


# 注意力公式计算
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # Q@K^T / sqrt(d_k)
    # 得分矩阵 [batch_size, n_head, seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask为1的位置保留原始分数，为0的位置替换为一个极小值
        # 经过softmax后，被屏蔽位置的注意力权重接近于0
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)
    # dropout 防止过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最终注意力输出context [batch_size, n_head, seq_len, d_k]
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    # n_head: 多头注意力的头数: 8
    # d_model: 模型输入的维度: 512
    # dropout: 注意力权重dropout的概率
    def __init__(self, n_head, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head  # 64
        self.n_head = n_head
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.linears = clones(nn.Linear(d_model, d_model), 4)

    # query, key 和 value传入的是和位置编码融合后的词向量，是同一个，形状为: [batch_size, seq_len, d_model]
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # 批次数
        """
          1. 首先，将输入经过线性层，映射为3个新的query, key, value
          x(query)  -> linear  -> query  [batch_size, seq_len, d_model]
          x(key)    -> linear  -> key    [batch_size, seq_len, d_model]
          x(value)  -> linear  -> value  [batch_size, seq_len, d_model]
          2. 再将d_model维度拆分成 n_head 个头，每个头维度为 d_k
          query  -> view  -> query  [batch_size, n_head, seq_len, d_k]
          key    -> view  -> key    [batch_size, n_head, seq_len, d_k]
          value  -> view  -> value  [batch_size, n_head, seq_len, d_k]
        """
        query, key, value = [
            lin(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, [query, key, value])
        ]

        # 计算注意力得分
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 沿着n_head维度进行拼接
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)

        # 释放内存
        del query, key, value

        # 经过线性层映射回原始维度
        # [batch_size, seq_len, d_model]
        return self.linears[-1](x)
