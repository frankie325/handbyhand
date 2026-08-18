import torch.nn as nn
from torch.nn.functional import log_softmax


# 最终的预测层
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        # vocab: 词表大小
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [batch_size, tgt_len, d_model]
        # 输出 [batch_size, tgt_len, vocab_size]
        #
        # ============================================================
        # 笔记：log_softmax vs softmax
        # ============================================================
        # 数学关系：log_softmax(x) = log(softmax(x))，但实现上不是先
        # softmax 再 log，而是直接一步算出（用 logsumexp 技巧），更稳定。
        #
        # 1. 公式：
        #    softmax(x_i)     = e^{x_i} / Σ_j e^{x_j}             → 概率，范围 (0,1)
        #    log_softmax(x_i) = x_i - log(Σ_j e^{x_j})           → 对数域，范围 (-∞,0]
        #
        # 2. 数值稳定性：
        #    softmax 里的 e^x 在 x 很大时溢出为 inf，很小时下溢为 0，
        #    导致 log(softmax) 得到 nan/-inf。
        #    log_softmax 内部用 x_i - max - log(Σ e^{x_j - max}) 恒等式，
        #    所有指数项 ≤ 1，不会溢出。
        #
        # 3. 用途：
        #    - softmax：输出概率分布（可视化、采样）
        #    - log_softmax：计算对数似然，配合 NLLLoss 训练
        #    - CrossEntropyLoss 内部就是 log_softmax + NLLLoss，
        #      直接传原始 logits 即可，不要手动 softmax 后再算损失。
        #
        # 4. 本模型用 log_softmax 的原因：
        #    这是训练时输出词表上对数分布的标准做法（配合 NLLLoss），
        #    比先 softmax 再 log 数值更稳定，梯度回传更简单。
        # ============================================================
        return log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # src: 原始的输入序列 [batch_size, src_len]
        # src_mask: 编码器的padding_mask [batch_size, src_len]
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory: 编码器的输出 [batch_size, src_len, d_model]
        # src_mask: 编码器的padding_mask [batch_size, src_len]
        # tgt: 解码器的输入序列 [batch_size, tgt_len]
        # tgt_mask: 解码器的因果掩码 [batch_size, tgt_len]
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
