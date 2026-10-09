import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class Backbone(nn.Module):

    def __init__(self, model_name: str, feature_layers: list[int], freeze: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
