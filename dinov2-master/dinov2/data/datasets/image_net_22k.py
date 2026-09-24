"""通过 numpy 元数据和 mmap 随机读取 ImageNet-22k tar 文件。"""

from functools import lru_cache
from gzip import GzipFile
from io import BytesIO
from mmap import ACCESS_READ, mmap
import os
from typing import Callable, Optional
import warnings

import numpy as np

from .extended import ExtendedVisionDataset


_DEFAULT_MMAP_CACHE_SIZE = 16


def _make_mmap_tarball(tarballs_root: str, mmap_cache_size: int):
    """创建带 LRU 缓存的 tar mmap 获取函数。

    ImageNet-22k 按类别拆成许多 tar。缓存最近访问的 mmap 可避免频繁
    打开/关闭文件，但缓存过大也会耗尽操作系统文件描述符。
    """

    @lru_cache(maxsize=mmap_cache_size)
    def _mmap_tarball(class_id: str) -> mmap:
        path = os.path.join(tarballs_root, f"{class_id}.tar")
        with open(path, "rb") as file:
            return mmap(fileno=file.fileno(), length=0, access=ACCESS_READ)

    return _mmap_tarball


class ImageNet22k(ExtendedVisionDataset):
    """使用 tar 内 byte offset 做 O(1) 随机访问的 ImageNet-22k。"""

    # 官方数据中这些条目的 payload 还包了一层 gzip。
    _GZIPPED_INDICES = {
        841_545,
        1_304_131,
        2_437_921,
        2_672_079,
        2_795_676,
        2_969_786,
        6_902_965,
        6_903_550,
        6_903_628,
        7_432_557,
        7_432_589,
        7_813_809,
        8_329_633,
        10_296_990,
        10_417_652,
        10_492_265,
        10_598_078,
        10_782_398,
        10_902_612,
        11_203_736,
        11_342_890,
        11_397_596,
        11_589_762,
        11_705_103,
        12_936_875,
        13_289_782,
    }

    def __init__(
        self,
        *,
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mmap_cache_size: int = _DEFAULT_MMAP_CACHE_SIZE,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._entries = self._load_extra("entries.npy")
        self._class_ids = self._load_extra("class-ids.npy")
        self._mmap_tarball = _make_mmap_tarball(self.root, mmap_cache_size)

    def _load_extra(self, filename: str) -> np.ndarray:
        path = os.path.join(self._extra_root, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ImageNet-22k metadata not found: {path}. Generate entries.npy/class-ids.npy before training."
            )
        return np.load(path, mmap_mode="r")

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        class_id = str(entry["class_id"])
        tar_data = self._mmap_tarball(class_id)
        start_offset = int(entry["start_offset"])
        end_offset = int(entry["end_offset"])

        try:
            # 每个 tar member 首块的 512 bytes 是 header，真正图片从其后开始。
            data = tar_data[start_offset:end_offset][512:]
            if len(data) >= 2 and data[:2] == b"\x1f\x8b":
                if index not in self._GZIPPED_INDICES:
                    warnings.warn(f"unexpected gzip payload at sample {index}")
                with GzipFile(fileobj=BytesIO(data)) as gzip_file:
                    data = gzip_file.read()
            return data
        except Exception as error:
            raise RuntimeError(f"cannot retrieve sample {index} from {class_id}.tar") from error

    def get_target(self, index: int) -> int:
        return int(self._entries[index]["class_index"])

    def get_targets(self) -> np.ndarray:
        return self._entries["class_index"]

    def find_class_id(self, class_index: int) -> str:
        return str(self._class_ids[class_index])

    def __getitem__(self, index):
        # 部分历史 ImageNet-22k 图片含不影响解码的元数据警告。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self._entries)
