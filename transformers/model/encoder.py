import torch.nn as nn
from transformers.utils import clones
from transformers.model.layer.layer_norm import LayerNorm


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # 编码器总共N层
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
