import torch.nn as nn
from transformers.utils import clones
from transformers.model.layer.sub_layer_connection import SublayerConnection


# 编码器层，每个编码器层都有两个子层：一个是多头自注意力，另一个是前馈网络
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
