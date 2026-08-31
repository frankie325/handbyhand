"""DETR 数据集加载与批处理工具。"""

from .build import build_dataloader, build_dataset
from .misc import collate_fn, pad_images
from .voc import VOC_CLASSES, VocDetection
from .raod import RAOD_CLASSES, RaodDetection

DATASET_REGISTRY = {
    "voc": VocDetection,
    "raod": RaodDetection,
}

__all__ = [
    "VOC_CLASSES",
    "VocDetection",
    "RAOD_CLASSES",
    "RaodDetection",
    "DATASET_REGISTRY",
    "build_dataloader",
    "build_dataset",
    "collate_fn",
    "pad_images",
]
