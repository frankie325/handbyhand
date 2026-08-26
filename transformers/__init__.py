"""Transformers 模型组件包。

从包顶层导出模型相关的可复用类，支持 `from transformers import Transformer` 直接导入。
数据与分词类（TranslationDataset、Tokenizer 等）、函数与配置常量请从原模块导入。
"""

# 模型主体
from .model.transformer import Transformer, Generator
from .model.encoder import Encoder
from .model.decoder import Decoder

# 模块组件
from .model.block.encoder_layer import EncoderLayer
from .model.block.decoder_layer import DecoderLayer
from .model.embedding.embedding import Embeddings
from .model.embedding.position_encoding import PositionalEncoding

# 基础层
from .model.layer.layer_norm import LayerNorm
from .model.layer.multi_headed_attention import MultiHeadedAttention
from .model.layer.position_wise_feed_forward import PositionwiseFeedForward
from .model.layer.sub_layer_connection import SublayerConnection

__all__ = [
    # 模型主体
    "Transformer", "Generator", "Encoder", "Decoder",
    # 模块组件
    "EncoderLayer", "DecoderLayer", "Embeddings", "PositionalEncoding",
    # 基础层
    "LayerNorm", "MultiHeadedAttention", "PositionwiseFeedForward", "SublayerConnection",
]
