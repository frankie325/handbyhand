import torch
from .detr import Detr
from ..config import (
    NUM_CLASSES,
    NUM_QUERIES,
    D_MODEL,
    N_LAYER,
    D_FF,
    N_HEAD,
    DROPOUT,
    MODELS_DIR,
)


def build_model(train: bool = True, device=torch.device("cpu")):

    model = Detr(
        NUM_CLASSES,
        NUM_QUERIES,
        D_MODEL,
        N_LAYER,
        D_FF,
        N_HEAD,
        DROPOUT,
        pretrained_backbone=train,
    ).to(device)

    if not train:
        checkpoint = MODELS_DIR / "best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"VOC2007 权重不存在: {checkpoint}。请先运行 python -m detr.train。"
            )
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state["model"])
        except RuntimeError as error:
            raise RuntimeError(
                f"权重 {checkpoint} 与当前 VOC20 模型不兼容。"
                "旧的 92 维分类头不能直接加载，请重新训练 VOC2007 模型。"
            ) from error

    return model
