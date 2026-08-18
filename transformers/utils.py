import copy
import torch
import torch.nn as nn


# PyTorch 中的 ModuleList 和 Sequential: 区别和使用场景：https://zhuanlan.zhihu.com/p/64990232
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# 因果掩码
def subsequent_mask(size):
    attn_shape = (1, size, size)
    # torch.triu: 返回上三角矩阵
    # diagonal=1表示对角线（不包括对角线）以上元素为1，对角线及以下元素为0
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


# 因果掩码 + padding_mask 作为解码器的自注意力计算过程中的掩码
# !注意：解码器的交叉注意力不需要因果掩码
def make_std_mask(tgt, padding_idx):
    # tgt: [batch_size, tgt_len]
    # tgt_mask: [batch_size, tgt_len] -> [batch_size, 1, tgt_len]
    tgt_mask = (tgt != padding_idx).unsqueeze(-2)

    tgt_mask = tgt_mask & subsequent_mask(tgt_mask.size(-1)).to(
        device=tgt_mask.device,
        dtype=tgt_mask.dtype,
    )
    return tgt_mask
