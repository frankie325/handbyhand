import torch
import torch.nn as nn
from transformers import (
    LayerNorm,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    SublayerConnection,
)
from transformers.utils import clones
import copy


# 为输入添加位置编码，detr中给query和key添加位置编码，value不添加位置编码
def with_pos_embed(tensor: torch.Tensor, pos) -> torch.Tensor:
    return tensor if pos is None else tensor + pos


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.layer_norm = LayerNorm(size)

    # def forward(self, src, pos, mask):
    #     # query和key添加位置编码
    #     query = self.layer_norm(with_pos_embed(src, pos))
    #     key = self.layer_norm(with_pos_embed(src, pos))
    #     x = self.sublayer[0](src, lambda x: self.self_attn(query, key, x, mask))
    #     return self.sublayer[1](x, self.feed_forward)
    def forward(self, src, pos, mask):
        # query和key添加位置编码
        x = self.sublayer[0](
            src,
            lambda x: self.self_attn(
                with_pos_embed(x, pos),
                with_pos_embed(x, pos),
                x,
                mask,
            ),
        )
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, src, pos, mask):
        # 编码器总共N层
        x = src
        for layer in self.layers:
            x = layer(x, pos, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.layer_norm = LayerNorm(size)

    # def forward(self, tgt, memory, pos, query_pos, src_mask, tgt_mask=None):
    #     # memory: 编码器的输出
    #     # src_mask: padding_mask，防止模型看到padding的信息
    #     # tgt_mask: Detr不需要因果掩码，因为query_objects是并行计算的

    #     # query和key添加位置编码
    #     query = self.layer_norm(with_pos_embed(tgt, query_pos))
    #     key = self.layer_norm(with_pos_embed(tgt, query_pos))
    #     # 计算自注意力子层
    #     # [batch_size, tgt_len, d_model]
    #     tgt = self.sublayer[0](tgt, lambda x: self.self_attn(query, key, x, None))

    #     # query和key添加位置编码, key,value来自编码器
    #     tgt = with_pos_embed(tgt, query_pos)
    #     key = with_pos_embed(memory, pos)
    #     # 计算交叉注意力子层，memory是编码器的输出，src_mask是编码器的padding_mask
    #     # !交叉注意力不需要因果掩码：它是对编码器输出（整个源句子）做注意力，Q 来自解码器，K/V 来自 memory。解码器当前位置理应能看到源句子的全部内容（翻译时每个目标词都基于整句原文），所以只屏蔽源句的 padding（src_mask），不屏蔽未来
    #     # [batch_size, tgt_len, d_model]
    #     x = self.sublayer[1](tgt, lambda x: self.src_attn(x, key, memory, src_mask))

    #     # 计算前馈网络子层
    #     # [batch_size, tgt_len, d_model]
    #     return self.sublayer[2](x, self.feed_forward)

    def forward(self, tgt, memory, pos, query_pos, src_mask, tgt_mask=None):
        # memory: 编码器的输出
        # src_mask: padding_mask，防止模型看到padding的信息
        # tgt_mask: Detr不需要因果掩码，因为query_objects是并行计算的

        # query和key添加位置编码
        # 计算自注意力子层
        # [batch_size, tgt_len, d_model]
        tgt = self.sublayer[0](
            tgt,
            lambda x: self.self_attn(
                with_pos_embed(x, query_pos),
                with_pos_embed(x, query_pos),
                x,
                None,
            ),
        )

        # query和key添加位置编码, key,value来自编码器
        # 计算交叉注意力子层，memory是编码器的输出，src_mask是编码器的padding_mask
        # !交叉注意力不需要因果掩码：它是对编码器输出（整个源句子）做注意力，Q 来自解码器，K/V 来自 memory。解码器当前位置理应能看到源句子的全部内容（翻译时每个目标词都基于整句原文），所以只屏蔽源句的 padding（src_mask），不屏蔽未来
        # [batch_size, tgt_len, d_model]
        x = self.sublayer[1](
            tgt,
            lambda x: self.src_attn(
                with_pos_embed(x, query_pos),
                with_pos_embed(memory, pos),
                memory,
                src_mask,
            ),
        )

        # 计算前馈网络子层
        # [batch_size, tgt_len, d_model]
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, tgt, memory, pos, query_pos, mask, tgt_mask=None):
        # 解码器总共N层
        x = tgt
        # 保存每一层解码器的输出
        intermediate_outputs = []

        for layer in self.layers:
            x = layer(x, memory, pos, query_pos, mask, tgt_mask=None)
            intermediate_outputs.append(self.norm(x))
        return torch.stack(
            intermediate_outputs
        )  # [num_layers, batch_size, tgt_len, d_model]


class DetrTransformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, pos, query_pos, mask, tgt_mask=None):
        return self.decode(
            self.encode(src, pos, mask), tgt, pos, query_pos, mask, tgt_mask
        )

    def encode(self, src, pos, mask):
        # src: 原始的输入序列 [batch_size, src_len]
        # src_mask: 编码器的padding_mask [batch_size, src_len]
        return self.encoder(src, pos, mask)

    def decode(
        self,
        memory,
        tgt,
        pos,
        query_pos,
        mask,
        tgt_mask=None,
    ):
        # memory: 编码器的输出 [batch_size, src_len, d_model]
        # src_mask: 编码器的padding_mask [batch_size, src_len]
        # tgt: 解码器的输入序列 [batch_size, tgt_len]
        # tgt_mask: 解码器的因果掩码 [batch_size, tgt_len]
        return self.decoder(tgt, memory, pos, query_pos, mask, tgt_mask)


def build_transformer(
    N=6,  # 编码器和解码器的层数
    d_model=512,  # 序列维度
    d_ff=2048,  # 前馈网络的中间隐藏层大小
    n_head=8,
    dropout=0.1,
):
    attn = MultiHeadedAttention(n_head, d_model, dropout)
    ff = PositionwiseFeedForward(
        d_model,
        d_ff,
        dropout,
    )
    c = copy.deepcopy
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    model = DetrTransformer(encoder, decoder)

    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
