"""DINOv2 标准 RGB 预训练数据集。"""

from .image_net import ImageNet
from .image_net_22k import ImageNet22k
from .image_folder import ImageFolderDataset

__all__ = ["ImageFolderDataset", "ImageNet", "ImageNet22k"]
