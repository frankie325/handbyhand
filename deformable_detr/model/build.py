from pathlib import Path

import torch

from .deformable_detr import DeformableDetr
from ..config import (
    AUX_LOSS,
    NUM_CLASSES,
    NUM_QUERIES,
    D_MODEL,
    N_ENCODER_LAYER,
    N_DECODER_LAYER,
    ENCODER_N_POINTES,
    DECODER_N_POINTES,
    D_FF,
    N_HEAD,
    DROPOUT,
    MODELS_DIR,
    TWO_STAGE,
    WITH_BOX_REFINE,
)


def build_model(
    train: bool = True,
    device: torch.device | str = torch.device("cpu"),
    checkpoint_path: Path | str | None = None,
    pretrained_backbone: bool = True,
    train_backbone: bool = True,
) -> DeformableDetr:
    """构建模型，并可选加载训练检查点。

    ``train=False`` 时默认加载 ``weights/best.pth``；显式传入
    ``checkpoint_path`` 时，无论 train 取值如何都会加载该检查点。
    """
    device = torch.device(device)

    model = DeformableDetr(
        NUM_CLASSES,
        NUM_QUERIES,
        d_model=D_MODEL,
        n_heads=N_HEAD,
        num_encoder_layers=N_ENCODER_LAYER,
        num_decoder_layers=N_DECODER_LAYER,
        dim_feedforward=D_FF,
        dropout=DROPOUT,
        enc_n_pointer=ENCODER_N_POINTES,
        dec_n_pointer=DECODER_N_POINTES,
        aux_loss=AUX_LOSS,
        with_box_refine=WITH_BOX_REFINE,
        two_stage=TWO_STAGE,
        pretrained_backbone=pretrained_backbone,
        train_backbone=train_backbone,
    )

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is None and not train:
        checkpoint = MODELS_DIR / "best.pth"

    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"权重不存在: {checkpoint}。请先运行 python -m deformable_detr.train。"
            )
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model_state = state["model"] if "model" in state else state
        try:
            model.load_state_dict(model_state)
        except RuntimeError as error:
            raise RuntimeError(
                f"权重 {checkpoint} 与当前 RAOD Deformable DETR 配置不兼容，"
                "请检查类别数、d_model、Decoder 层数和两阶段配置。"
            ) from error

    return model.to(device)
