"""基于预生成 numpy 索引读取 ImageNet-1k 图片。"""

from enum import Enum
import os
from typing import Callable, Optional

import numpy as np

from .extended import ExtendedVisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        return {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }[self]

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        """根据官方 ImageNet 命名规则恢复图片相对路径。"""

        dirname = self.value if class_id is None else os.path.join(self.value, class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")


class ImageNet(ExtendedVisionDataset):
    """读取已经展开为目录结构的 ImageNet-1k。

    ``extra`` 目录中的结构化 numpy 数组充当轻量数据库，避免每次启动时
    都扫描一百多万文件。数组使用 mmap 只读打开，不会整体载入内存。
    """

    Split = _Split

    def __init__(
        self,
        *,
        split: "ImageNet.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        self._entries = None

    @property
    def split(self):
        return self._split

    def _load_extra(self, filename: str) -> np.ndarray:
        path = os.path.join(self._extra_root, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ImageNet metadata not found: {path}. Generate entries-*.npy with the full DINOv2 data utility first."
            )
        return np.load(path, mmap_mode="r")

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(f"entries-{self._split.value.upper()}.npy")
        return self._entries

    def get_image_data(self, index: int) -> bytes:
        entry = self._get_entries()[index]
        actual_index = int(entry["actual_index"])
        class_id = None if self.split == _Split.TEST else str(entry["class_id"])
        relative_path = self.split.get_image_relpath(actual_index, class_id)
        with open(os.path.join(self.root, relative_path), "rb") as file:
            return file.read()

    def get_target(self, index: int):
        if self.split == _Split.TEST:
            return None
        return int(self._get_entries()[index]["class_index"])

    def get_targets(self):
        return None if self.split == _Split.TEST else self._get_entries()["class_index"]

    def __len__(self) -> int:
        entries = self._get_entries()
        # 自定义裁剪版数据集可能没有官方长度，若需要支持可移除此断言。
        if len(entries) != self.split.length:
            raise ValueError(f"expected {self.split.length} samples, found {len(entries)}")
        return len(entries)
