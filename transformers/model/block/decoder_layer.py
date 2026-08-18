import torch.nn as nn
from transformers.utils import clones
from transformers.model.layer.sub_layer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory: 编码器的输出
        # src_mask: padding_mask，防止模型看到padding的信息
        # tgt_mask: 因果掩码（+ padding mask），用于防止模型看到未来的信息

        # 计算Masked自注意力子层
        # [batch_size, tgt_len, d_model]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 计算交叉注意力子层，memory是编码器的输出，src_mask是编码器的padding_mask
        # !交叉注意力不需要因果掩码：它是对编码器输出（整个源句子）做注意力，Q 来自解码器，K/V 来自 memory。解码器当前位置理应能看到源句子的全部内容（翻译时每个目标词都基于整句原文），所以只屏蔽源句的 padding（src_mask），不屏蔽未来
        # [batch_size, tgt_len, d_model]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))

        # 计算前馈网络子层
        # [batch_size, tgt_len, d_model]
        return self.sublayer[2](x, self.feed_forward)
