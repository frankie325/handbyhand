---
name: transformers 包顶层导出可复用类
overview: 在 transformers/__init__.py 中集中导出该包内所有可复用的类，使用户可以直接 `from transformers import Transformer, ChineseTokenizer` 等，函数与常量保持原样。
todos:
  - id: write-top-level-exports
    content: 在 transformers/__init__.py 中编写全部 12 个模型类的相对导入与 __all__ 定义，仅导出类
    status: completed
  - id: verify-imports
    content: 运行 python 验证 12 个类均可从 transformers 顶层导入且原有深层导入路径不受影响
    status: completed
    dependencies:
      - write-top-level-exports
---

## 需求概述

将 `/Users/frank/code/ai/handbyhand/transformers` 包中模型相关可复用的类从各子模块提升到包顶层导出，使用户可以直接通过 `from transformers import ClassName` 导入，无需再深入子包路径。

## 核心功能

- 在 `transformers/__init__.py`（当前为空文件）中导出全部 12 个可复用类：
- 模型主体：`Transformer`、`Generator`、`Encoder`、`Decoder`
- 模块组件：`EncoderLayer`、`DecoderLayer`、`Embeddings`、`PositionalEncoding`
- 基础层：`LayerNorm`、`MultiHeadedAttention`、`PositionwiseFeedForward`、`SublayerConnection`
- 定义 `__all__` 列表，支持 `from transformers import *` 通配导入
- 导出范围仅限类：数据与分词类（`TranslationDataset`、`BaseTokenizer`、`ChineseTokenizer`、`EnglishTokenizer`）以及函数（`make_model`、`collate_fn`、`get_dataloader`、`clones`、`attention`、`predict` 等）与配置常量（`config.py` 中的路径和超参）保持原样，仍从原模块导入
- 保持现有子模块内部导入路径（如 `from transformers.utils import clones`）不受影响，原有深层导入方式继续有效

## 技术栈

- Python 3.14（项目现有环境，uv 管理依赖，非可安装包）
- 使用 Python 标准相对导入机制，无新增依赖

## 实现方案

在 `transformers/__init__.py` 中通过显式相对导入（`from .model.transformer import Transformer` 等）聚合导出模型相关全部可复用类，并定义按模块分组排列的 `__all__` 列表。不改动任何现有子模块文件的内部代码，仅新增一个聚合导出入口。

## 关键技术决策

- **仅修改 `__init__.py` 一个文件**：导出是纯增量操作，不触碰任何子模块逻辑，回归风险为零
- **不补建子目录 `__init__.py`**：`model/`、`model/layer/` 等子目录当前依赖 PEP 420 namespace package 机制（存在 .pyc 证明可正常工作），顶层 regular package 与其配合无冲突，无需额外改动
- **显式列出而非通配导入**：每个类从明确来源导入，避免隐式依赖，便于追踪与维护
- **模块间依赖链已完整**：子模块内已存在相互相对导入，`__init__.py` 仅做聚合导出，不存在循环导入风险

## 性能与可靠性

- 导入开销可忽略：均为轻量模块（torch 为既有重依赖，非本次新增）
- 无运行时性能影响，仅影响导入路径的可见性

## 验证方式

- 运行 `python -c "from transformers import Transformer, Encoder, Decoder, Generator, EncoderLayer, DecoderLayer, Embeddings, PositionalEncoding, LayerNorm, MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection"` 确认全部 12 个类可顶层导入
- 运行 `python -c "import transformers; print(transformers.__all__)"` 确认 `__all__` 定义完整
- 确认原有深层导入（如 `from transformers.model.transformer import Transformer`）仍正常

## 架构设计

本次改动为单文件聚合导出，结构简单，无需系统架构图。数据流：顶层包 `transformers/__init__.py` → 相对导入各子模块 → 暴露 12 个类于包命名空间。

## 目录结构

```
transformers/
└── __init__.py  # [MODIFY] 包顶层导出入口。当前为空文件，需写入：
                 # 1) 12 个可复用类的显式相对导入（来自 .model.transformer、
                 #    .model.encoder、.model.decoder、.model.block.*、.model.embedding.*、.model.layer.*）
                 # 2) 按模块分组、含全部 12 个类名的 __all__ 列表
                 # 注意：不修改任何其他子模块文件，不导入数据与分词类、函数与配置常量
```

## 关键代码结构

`transformers/__init__.py` 的核心内容结构（示意）：

```python
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
```