"""Backbone used by the Deformable-DETR model.

The original DETR backbone returned only the output of ``layer4``. A
Deformable-DETR encoder consumes a feature pyramid instead, so this module
returns the outputs of ResNet ``layer2``, ``layer3`` and ``layer4``. Every
feature has a matching padding mask and the module exposes the metadata used
by the input projections (``strides`` and ``num_channels``).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter

from ..config import MODELS_DIR


# Keep torchvision's cache inside this project, as the previous backbone did.
torch.hub.set_dir(str(MODELS_DIR))


@dataclass
class NestedTensor:
    """A feature tensor and its padding mask.

    ``mask[b, y, x] == True`` means that the position is padding and must not
    be attended to by the transformer. This local implementation keeps the
    backbone independent from the top-level DETR package while retaining the
    interface used by the reference Deformable-DETR implementation.
    """

    tensors: Tensor
    mask: Optional[Tensor]

    def decompose(self) -> Tuple[Tensor, Optional[Tensor]]:
        return self.tensors, self.mask

    def to(self, *args, **kwargs) -> "NestedTensor":
        return NestedTensor(
            self.tensors.to(*args, **kwargs),
            None if self.mask is None else self.mask.to(*args, **kwargs),
        )


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed statistics and affine parameters.

    ImageNet pretrained ResNet checkpoints contain BatchNorm buffers. Keeping
    those statistics frozen is the behavior used by the official
    Deformable-DETR implementation and avoids small-batch instability.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        prefix: str,
        local_metadata: Mapping[str, object],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        # Older torchvision checkpoints have this extra BatchNorm buffer.
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + self.eps).rsqrt()
        return x * scale + (bias - running_mean * scale)


class BackboneBase(nn.Module):
    """Wrap a torchvision backbone and expose selected intermediate layers."""

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        return_interm_layers: bool = True,
    ) -> None:
        super().__init__()

        # Match Deformable-DETR: the stem and layer1 stay frozen while the
        # deeper ResNet stages can be fine-tuned when requested.
        for name, parameter in backbone.named_parameters():
            if not train_backbone or not any(
                stage in name for stage in ("layer2", "layer3", "layer4")
            ):
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = self._stage_channels(backbone, return_layers)
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [self._stage_channels(backbone, return_layers)[-1]]

        self.return_interm_layers = return_interm_layers
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    @staticmethod
    def _stage_channels(
        backbone: nn.Module, return_layers: Mapping[str, str]
    ) -> List[int]:
        """Infer channels so ResNet-18/34 and bottleneck ResNets all work."""

        channels: List[int] = []
        for stage in return_layers:
            block = getattr(backbone, stage)[-1]
            if hasattr(block, "conv3"):  # Bottleneck (ResNet-50/101/152)
                channels.append(int(block.conv3.out_channels))
            elif hasattr(block, "conv2"):  # BasicBlock (ResNet-18/34)
                channels.append(int(block.conv2.out_channels))
            else:  # pragma: no cover - defensive for a custom torchvision model
                raise ValueError(f"Cannot infer output channels for {stage}")
        return channels

    @staticmethod
    def _mask_for(
        padding_mask: Optional[Tensor],
        size: Tuple[int, int],
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if padding_mask is None:
            return torch.zeros(
                (batch_size, size[0], size[1]), dtype=torch.bool, device=device
            )
        if padding_mask.ndim != 3:
            raise ValueError(
                "padding_mask must have shape [batch, height, width], "
                f"got {tuple(padding_mask.shape)}"
            )
        if padding_mask.shape[0] != batch_size:
            raise ValueError("padding_mask batch size must match images")
        return F.interpolate(
            padding_mask[:, None].to(dtype=torch.float32),
            size=size,
            mode="nearest",
        )[:, 0].to(torch.bool)

    def forward(
        self,
        images: Union[Tensor, NestedTensor],
        padding_mask: Optional[Tensor] = None,
    ) -> "OrderedDict[str, NestedTensor]":
        """Return an ordered mapping of multiscale ``NestedTensor`` features.

        ``images`` may be either a raw ``[B, 3, H, W]`` tensor or a
        :class:`NestedTensor`. The raw tensor form is convenient for this
        project; the nested form mirrors the reference implementation.
        """

        if isinstance(images, NestedTensor):
            if padding_mask is not None:
                raise ValueError("padding_mask must be omitted with NestedTensor")
            image_tensor = images.tensors
            padding_mask = images.mask
        else:
            image_tensor = images

        if image_tensor.ndim != 4:
            raise ValueError(
                "images must have shape [batch, channels, height, width], "
                f"got {tuple(image_tensor.shape)}"
            )

        outputs: "OrderedDict[str, NestedTensor]" = OrderedDict()
        for name, feature in self.body(image_tensor).items():
            mask = self._mask_for(
                padding_mask,
                feature.shape[-2:],
                image_tensor.shape[0],
                feature.device,
            )
            outputs[name] = NestedTensor(feature, mask)
        return outputs

    def forward_features(
        self, images: Tensor, padding_mask: Optional[Tensor] = None
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Return ``(features, masks)`` lists for simple model integrations."""

        outputs = self(images, padding_mask)
        return [x.tensors for x in outputs.values()], [x.mask for x in outputs.values()]


class Backbone(BackboneBase):
    """ResNet backbone configured for Deformable-DETR feature levels.

    ``return_interm_layers=True`` produces the standard three levels (strides
    8, 16 and 32). Set it to ``False`` to retain a single ``layer4`` level.
    The ``pretrained_backbone`` keyword is kept for compatibility with the
    earlier project backbone.
    """

    def __init__(
        self,
        name: str = "resnet50",
        train_backbone: bool = True,
        return_interm_layers: bool = True,
        dilation: bool = False,
        pretrained_backbone: bool = True,
    ) -> None:
        # The old local API accepted Backbone(pretrained_backbone, train_backbone)
        # positionally. Interpret that form without weakening the new API.
        if isinstance(name, bool):
            old_pretrained = name
            old_train = train_backbone
            name = "resnet50"
            pretrained_backbone = old_pretrained
            train_backbone = old_train

        if name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
            backbone = resnet50(
                weights=weights,
                replace_stride_with_dilation=[False, False, dilation],
                norm_layer=FrozenBatchNorm2d,
            )
        else:
            # Import lazily so the common resnet50 path stays explicit while
            # allowing resnet101/152/18/34 through the same interface.
            import torchvision.models as models

            try:
                constructor = getattr(models, name)
            except AttributeError as exc:
                raise ValueError(f"Unsupported torchvision backbone: {name}") from exc
            weight_enum_name = {
                "resnet18": "ResNet18_Weights",
                "resnet34": "ResNet34_Weights",
                "resnet101": "ResNet101_Weights",
                "resnet152": "ResNet152_Weights",
            }.get(name)
            weights = None
            if pretrained_backbone and weight_enum_name is not None:
                weights = getattr(models, weight_enum_name).DEFAULT
            backbone = constructor(
                weights=weights,
                replace_stride_with_dilation=[False, False, dilation],
                norm_layer=FrozenBatchNorm2d,
            )
            if pretrained_backbone and weight_enum_name is None:
                raise ValueError(
                    f"pretrained_backbone=True is not configured for {name}; "
                    "set it to False or add a torchvision weight enum"
                )

        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            # Replacing the final stride with dilation changes layer4 from
            # stride 32 to stride 16.
            self.strides[-1] //= 2


class PositionEmbeddingSine(nn.Module):
    """Sine/cosine image position encoding used by the reference model."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        if scale is not None and not normalize:
            raise ValueError("normalize must be True when scale is provided")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        x = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None:
            mask = torch.zeros(
                x.shape[0], x.shape[-2], x.shape[-1],
                dtype=torch.bool, device=x.device,
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class PositionEmbeddingLearned(nn.Module):
    """Learned absolute row/column position encoding."""

    def __init__(self, num_pos_feats: int = 128, max_size: int = 256) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(max_size, num_pos_feats)
        self.col_embed = nn.Embedding(max_size, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        x = tensor_list.tensors
        height, width = x.shape[-2:]
        if height > self.row_embed.num_embeddings or width > self.col_embed.num_embeddings:
            raise ValueError(
                "learned position embedding supports at most "
                f"{self.row_embed.num_embeddings}x{self.col_embed.num_embeddings} "
                f"features, got {height}x{width}"
            )
        i = torch.arange(width, device=x.device)
        j = torch.arange(height, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(height, 1, 1),
                y_emb.unsqueeze(1).repeat(1, width, 1),
            ],
            dim=-1,
        )
        return pos.permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)


class Joiner(nn.Sequential):
    """Combine the multiscale backbone with one position encoding module."""

    def __init__(self, backbone: BackboneBase, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(
        self,
        images: Union[Tensor, NestedTensor],
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[List[NestedTensor], List[Tensor]]:
        xs = self[0](images, padding_mask)
        features = list(xs.values())
        positions = [self[1](feature).to(feature.tensors.dtype) for feature in features]
        return features, positions


def build_position_encoding(args) -> nn.Module:
    """Build the position encoding from a Deformable-DETR argument namespace."""

    hidden_dim = int(getattr(args, "hidden_dim", 256))
    num_pos_feats = hidden_dim // 2
    encoding = getattr(args, "position_embedding", "sine")
    if encoding in ("sine", "v2"):
        return PositionEmbeddingSine(num_pos_feats, normalize=True)
    if encoding in ("learned", "v3"):
        return PositionEmbeddingLearned(num_pos_feats)
    raise ValueError(f"Unsupported position embedding: {encoding}")


def build_backbone(args) -> Joiner:
    """Build a Deformable-DETR backbone and its position encoding."""

    name = getattr(args, "backbone", "resnet50")
    train_backbone = getattr(args, "lr_backbone", 1.0) > 0
    return_interm_layers = bool(
        getattr(args, "masks", False)
        or getattr(args, "num_feature_levels", 4) > 1
    )
    dilation = bool(getattr(args, "dilation", False))
    pretrained = bool(getattr(args, "pretrained_backbone", True))
    backbone = Backbone(
        name=name,
        train_backbone=train_backbone,
        return_interm_layers=return_interm_layers,
        dilation=dilation,
        pretrained_backbone=pretrained,
    )
    return Joiner(backbone, build_position_encoding(args))


__all__ = [
    "Backbone",
    "BackboneBase",
    "FrozenBatchNorm2d",
    "Joiner",
    "NestedTensor",
    "PositionEmbeddingLearned",
    "PositionEmbeddingSine",
    "build_backbone",
    "build_position_encoding",
]
