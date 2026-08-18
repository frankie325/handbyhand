import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # 乘以 math.sqrt(d_model) 可以让词嵌入的数值尺度更合适，避免位置编码对词嵌入的影响过强。这是原始 Transformer 论文中的做法
        # [batch_size, seq_len, d_model]
        return self.lut(x) * math.sqrt(self.d_model)
