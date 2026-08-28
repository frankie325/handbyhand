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
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)

    return model
