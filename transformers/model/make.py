import copy
import torch
import torch.nn as nn
from transformers.model.encoder import Encoder
from transformers.model.decoder import Decoder
from transformers.model.block.encoder_layer import EncoderLayer
from transformers.model.block.decoder_layer import DecoderLayer
from transformers.model.embedding.embedding import Embeddings
from transformers.model.embedding.position_encoding import PositionalEncoding
from transformers.model.layer.multi_headed_attention import MultiHeadedAttention
from transformers.model.layer.position_wise_feed_forward import PositionwiseFeedForward
from transformers.model.transformer import Transformer, Generator


def make_model(
    src_vocab,  # 源语言词表大小
    tgt_vocab,  # 目标语言词表大小
    N=6,  # 编码器和解码器的层数
    d_model=512,  # 序列维度
    d_ff=2048,  # 前馈网络的中间隐藏层大小
    n_head=8,
    dropout=0.1,
    max_len=5000,
):
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(
            Embeddings(d_model, src_vocab), c(position)
        ),  # 输入序列融合位置编码
        nn.Sequential(
            Embeddings(d_model, tgt_vocab), c(position)
        ),  # 目标序列融合位置编码
        Generator(d_model, tgt_vocab),
    )

    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
