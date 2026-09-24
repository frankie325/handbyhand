"""适用于大规模、自监督和分布式训练的索引采样器。"""

import itertools
from typing import Optional

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import dinov2.distributed as distributed


class EpochSampler(Sampler):
    """每个逻辑 epoch 产生固定数量的样本索引。

    当 ``size`` 大于数据集长度时先重复索引，再从中采样；start/step
    默认等于全局 rank/world size，使多个进程从同一序列取互不重叠的切片。
    """

    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        repeat_count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), repeat_count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            indices = rng.choice(tiled_indices, self._size, replace=False)
        else:
            indices = tiled_indices[: self._size]

        # rank 0 取 0, world_size, 2*world_size...；rank 1 从 1 开始。
        yield from itertools.islice(indices, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch):
        self._epoch = epoch


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """逐项产生随机排列，避免必须长期保存另一个 Python 索引列表。"""

    dtype = torch.int32 if size <= 2**31 else torch.int64
    permutation = torch.arange(size, dtype=dtype)
    for index in range(size):
        swap_index = torch.randint(index, size, size=(1,), generator=generator).item()
        value = permutation[swap_index].item()
        permutation[swap_index] = permutation[index].item()
        permutation[index] = value
        yield value


class InfiniteSampler(Sampler):
    """无限循环数据集；训练何时结束由 max_iter 决定。"""

    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        iterator = self._shuffled_iterator() if self._shuffle else self._ordered_iterator()
        # 恢复训练时可跳过已经消费的本 rank 样本。
        yield from itertools.islice(iterator, self._advance, None)

    def _ordered_iterator(self):
        while True:
            yield from itertools.islice(range(self._sample_count), self._start, None, self._step)

    def _shuffled_iterator(self):
        # 在 __iter__ 中创建 generator，使 Sampler 本身可以被多进程 pickle。
        generator = torch.Generator().manual_seed(self._seed)
        while True:
            permutation = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(permutation, self._start, None, self._step)


class ShardedInfiniteSampler(Sampler):
    """先全局打乱，再给每个 rank 分片，并在每轮内重新打乱各自分片。

    该策略适合 DINOv2 的多卡迭代式训练：各 rank 不会读取同一索引，
    同时每个 rank 的局部顺序也会随循环次数变化。
    """

    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance
        self._iteration = 0
        # 保留参数以兼容官方配置；学习版统一使用 randperm 实现。
        self._use_new_shuffle_tensor_slice = use_new_shuffle_tensor_slice

    def __iter__(self):
        completed_rounds, remaining_advance = divmod(self._advance, self._sample_count)
        self._iteration += completed_rounds
        iterator = self._shuffled_iterator() if self._shuffle else self._ordered_iterator()
        yield from itertools.islice(iterator, remaining_advance, None)

    def _ordered_iterator(self):
        while True:
            yield from itertools.islice(range(self._sample_count), self._start, None, self._step)

    def _shuffled_iterator(self):
        generator = torch.Generator().manual_seed(self._seed)
        dtype = torch.int32 if self._sample_count <= 2**31 else torch.int64
        global_permutation = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        # 丢掉无法被 world_size 整除的尾部，与官方实现的等长分片语义一致。
        local_indices = global_permutation[self._start :: self._step]
        while True:
            local_seed = self._seed + self._start + (self._iteration << 24)
            generator.manual_seed(local_seed)
            local_order = torch.randperm(len(local_indices), generator=generator)
            yield from local_indices[local_order].tolist()
            self._iteration += 1
