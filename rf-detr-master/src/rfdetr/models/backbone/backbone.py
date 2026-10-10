import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import NestedTensor
from rfdetr.models.backbone.projector import MultiScaleProjector


class Backbone(nn.Module):

    def __init__(
        self,
        name: str,
        pretrained_encoder: str | None = None,
        window_block_indexes: list[int] | None = None,
        drop_path: float = 0.0,
        out_channels: int = 256,
        out_feature_indexes: list[int] | None = None,
        projector_scale: list[str] | None = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 0,
        dual_projector: bool = False,
    ) -> None:
        super().__init__()

        # an example name here would be "dinov2_base" or "dinov2_registers_windowed_base"
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        # name_parts[-1]
        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert (
            len(name_parts) == 2
        ), "name should be dinov2, then either registers, windowed, both, or none, then the size"

        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
            window_block_indexes=window_block_indexes,
        )

        # 是否冻结骨干网
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        """
        MultiScaleProjector 的作用是把 DINOv2 输出的多层、同分辨率特征，转换成适合检测的多尺度特征
        P4 → scale 1.0（原生分辨率，stride≈patch_size）
        P3 → scale 2.0（上采样 2×，更高分辨率，利于小目标）
        P5 → scale 0.5（下采样 2×，更低分辨率，利于大目标）
        P6 → scale 0.25（再下采样，靠 max pool）
        """
        self.projector_scale = projector_scale
        assert self.projector_scale is not None and len(self.projector_scale) > 0
        # x[0]
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "only support projector scale P3/P4/P5/P6 in ascending order."
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        # 必须是 ["P3","P4","P5","P6"] 的升序子集
        # [2.0, 1.0, 0.5, 0.25]
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

    def forward(
        self, tensor_list: NestedTensor
    ) -> tuple[list[NestedTensor], list[NestedTensor] | None]:
        # (H, W, B, C)
        raw_feats = self.encoder(tensor_list.tensors)
        feats = self.projector(raw_feats)
