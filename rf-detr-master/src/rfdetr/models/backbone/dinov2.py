from __future__ import annotations

import json
import math
import os
import types
from typing import Any, cast

from torch import Tensor, nn
import torch.nn.functional as F  # noqa: N812
from transformers import Dinov2Config, AutoBackbone
from typing import TYPE_CHECKING, Literal, Mapping, TextIO
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
)
from rfdetr.utilities.logger import get_logger

logger = get_logger()

size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

size_to_config = {
    "small": "dinov2_small.json",
    "base": "dinov2_base.json",
    "large": "dinov2_large.json",
}

size_to_config_with_registers = {
    "small": "dinov2_with_registers_small.json",
    "base": "dinov2_with_registers_base.json",
    "large": "dinov2_with_registers_large.json",
}


def get_config(size: str, use_registers: bool) -> dict[str, Any]:
    config_dict = size_to_config_with_registers if use_registers else size_to_config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(current_dir, "dinov2_configs")
    config_path = os.path.join(configs_dir, config_dict[size])
    with open(config_path) as f:
        dino_config: dict[str, Any] = json.load(f)
    return dino_config


def compute_window_block_indexes(
    out_feature_indexes: list[int], window_block_indexes: list[int] | None = None
) -> list[int]:
    """Derive which 0-indexed encoder blocks use windowed self-attention (the rest use global).

    When `window_block_indexes` is provided explicitly, it is returned unchanged — this is the
    override path for callers that want to pin an exact routing (e.g. to match a published
    architecture spec) without disturbing the derived default used by existing configs/checkpoints.
    This override is only wired through `ModelDefaults` (`rfdetr/models/_defaults.py`) or direct
    `Backbone`/`DinoV2` construction — not the public pydantic `RFDETR*Config` classes, which set
    `ConfigDict(extra="forbid")` and raise on unknown fields.

    When omitted, the windowed set is derived as the complement of `out_feature_indexes` (treated
    as raw block indices) within `range(0, out_feature_indexes[-1] + 1)`. This mirrors the
    historical behavior relied upon by all released RF-DETR checkpoints; it does not renumber
    `out_feature_indexes` (1-indexed HF stage numbers) to 0-indexed block indices, so it is not
    guaranteed to match any particular paper-specified layer schedule. The derivation assumes
    `out_feature_indexes` is given in ascending order: it uses `out_feature_indexes[-1]` (not
    `max(out_feature_indexes)`) as the upper bound, so an unsorted list whose maximum is not last
    would under-cover the intended range.

    Args:
        out_feature_indexes: Encoder stage numbers whose feature maps are exported to the decoder.
        window_block_indexes: Explicit windowed-block override. `None` derives from
            `out_feature_indexes` using the legacy formula.

    Returns:
        0-indexed encoder block positions that should run windowed attention.

    Examples:
        >>> compute_window_block_indexes([3, 6, 9, 12])
        [0, 1, 2, 4, 5, 7, 8, 10, 11]
        >>> compute_window_block_indexes([3, 6, 9, 12], window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10])
        [0, 1, 3, 4, 6, 7, 9, 10]
    """
    if window_block_indexes is not None:
        return window_block_indexes
    window_block_indexes_set = set(range(out_feature_indexes[-1] + 1))
    window_block_indexes_set.difference_update(out_feature_indexes)
    return sorted(window_block_indexes_set)


class DinoV2(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int] = (640, 640),
        out_feature_indexes: list[int] | None = None,  # 输出指定的特征层数
        size: str = "base",
        use_registers: bool = True,
        use_windowed_attn: bool = True,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,  # 窗口注意力分快数
        positional_encoding_size: int = 37,
        drop_path_rate: float = 0.0,
        window_block_indexes: list[int] | None = None,  # 使用窗口注意力的层数
    ) -> None:
        super().__init__()

        if out_feature_indexes is None:
            out_feature_indexes = [2, 4, 5, 9]
        # 加载的dinov2模型结构
        name = (
            f"facebook/dinov2-with-registers-{size}"
            if use_registers
            else f"facebook/dinov2-{size}"
        )

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows

        # 预训练权重目录：作为 HuggingFace 的下载缓存目录。
        # 权重已存在时会自动复用本地副本，无需重新下载；缺失时才联网下载。
        weights_dir = "/Users/frank/code/ai/handbyhand/rf-detr-master/weights"

        if not use_windowed_attn:
            assert (
                not gradient_checkpointing
            ), "Gradient checkpointing is not supported for non-windowed attention"
            assert (
                load_dinov2_weights
            ), "Using non-windowed attention requires loading dinov2 weights from hub"
            if drop_path_rate > 0.0:
                logger.warning(
                    "drop_path_rate > 0.0 is not supported for non-windowed DinoV2 backbones."
                    " drop_path will be ignored."
                )

            self.encoder = AutoBackbone.from_pretrained(  # type: ignore[no-untyped-call]
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
                cache_dir=weights_dir,
            )

        else:
            # 获取配置文件
            dino_config = get_config(size, use_registers)

            # encoder层数
            num_hidden_layers = dino_config["num_hidden_layers"]
            if not out_feature_indexes:
                raise ValueError("out_feature_indexes must be non-empty.")
            if window_block_indexes is not None and (
                len(set(window_block_indexes)) != len(window_block_indexes)
                or any(
                    not (0 <= idx < num_hidden_layers) for idx in window_block_indexes
                )
            ):
                raise ValueError(
                    f"window_block_indexes entries must be unique and within "
                    f"[0, {num_hidden_layers}); got {window_block_indexes}."
                )

            # 计算哪些encoder_layer使用窗口注意力机制
            window_block_indexes = compute_window_block_indexes(
                out_feature_indexes, window_block_indexes
            )

            dino_config["return_dict"] = False
            dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]
            dino_config["drop_path_rate"] = drop_path_rate

            implied_resolution = positional_encoding_size * patch_size

            # 检查位置编码网格和图像大小是否匹配，官方预训练的位置编码为（37×37）
            if implied_resolution != dino_config["image_size"]:
                logger.warning(
                    "Using a different number of positional encodings than DINOv2, which means"
                    " we're not loading DINOv2 backbone weights. This is not a problem if"
                    " finetuning a pretrained RF-DETR model."
                )
                dino_config["image_size"] = implied_resolution
                load_dinov2_weights = False  # 如果不匹配，则不使用官方预训练权重

            # 检查patch_size是否匹配，DINOv2 预训练用的是 patch_size=14
            if patch_size != 14:
                logger.warning(
                    f"Using patch size {patch_size} instead of 14, which means we're not loading"
                    " DINOv2 backbone weights. This is not a problem if finetuning a pretrained"
                    " RF-DETR model."
                )
                dino_config["patch_size"] = patch_size
                load_dinov2_weights = False

            if use_registers:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    num_register_tokens=0,
                    gradient_checkpointing=gradient_checkpointing,
                )
            self.encoder = (
                WindowedDinov2WithRegistersBackbone.from_pretrained(
                    name,
                    config=windowed_dino_config,
                    cache_dir=weights_dir,
                )
                if load_dinov2_weights
                else WindowedDinov2WithRegistersBackbone(windowed_dino_config)
            )

        # 如果是base 模型：则4个输出特征图的d_model维度为[768, 768, 768, 768]
        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    """
       训练完成、要拿去部署时，才显式调一次 model.export()，把模型序列化成一个确定性静态计算图。
       图导出器（ONNX/TensorRT）遇到这种"尺寸随输入变"的插值算子，要么导出失败，要么退化成动态节点、无法做常量折叠和优化。
       部署时要的就是一个固定分辨率的干净静态图
    """

    def export(self) -> None:
        if self._export:
            return
        self._export = True
        shape = self.shape

        def make_new_interpolated_pos_encoding(
            position_embeddings: Tensor, patch_size: int, height: int, width: int
        ) -> Tensor:

            num_positions = position_embeddings.shape[1] - 1
            dim = position_embeddings.shape[-1]
            height = height // patch_size
            width = width // patch_size

            class_pos_embed = position_embeddings[:, 0]
            patch_pos_embed = position_embeddings[:, 1:]

            # Reshape and permute
            patch_pos_embed = patch_pos_embed.reshape(
                1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

            # Use bicubic interpolation, disabling antialias only on MPS devices
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=patch_pos_embed.device.type != "mps",
            )

            # Reshape back
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        # If the shape of self.encoder.embeddings.position_embeddings
        # matches the shape of your new tensor, use copy_:
        with torch.no_grad():
            new_positions = make_new_interpolated_pos_encoding(
                self.encoder.embeddings.position_embeddings,
                self.encoder.config.patch_size,
                shape[0],
                shape[1],
            )
        # Create a new Parameter with the new size
        old_interpolate_pos_encoding = self.encoder.embeddings.interpolate_pos_encoding

        def new_interpolate_pos_encoding(
            self_mod: Any, embeddings: Tensor, height: int, width: int
        ) -> Tensor:
            num_patches = embeddings.shape[1] - 1
            num_positions = self_mod.position_embeddings.shape[1] - 1
            # The precomputed table is valid only for this exact static export grid.
            if num_patches == num_positions and (height, width) == shape:
                return cast(Tensor, self_mod.position_embeddings)
            return cast(Tensor, old_interpolate_pos_encoding(embeddings, height, width))

        self.encoder.embeddings.position_embeddings = nn.Parameter(new_positions)
        self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
            new_interpolate_pos_encoding, self.encoder.embeddings
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        # x.shape = (batch_size, c, h, w)
        block_size = self.patch_size * self.num_windows
        # 检查输入图像大小是否是block_size的整数倍
        assert (
            x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0
        ), f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        x = self.encoder(x)

        return list(x[0])
