"""为 iBOT patch 预测任务生成连续的二维块状遮挡。"""

import math
import random

import numpy as np


class MaskingGenerator:
    """在 patch 网格上反复绘制随机矩形，直到接近目标遮挡数。

    相比逐 patch 独立伯努利采样，矩形 mask 会遮掉连续区域，迫使模型
    更多地利用上下文语义，而不是简单插值相邻像素。
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        # 在 log 空间均匀采样宽高比，可对“瘦高”和“扁宽”矩形保持对称。
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        return (
            f"MaskingGenerator(grid={self.height}x{self.width}, "
            f"block={self.min_num_patches}..{self.max_num_patches})"
        )

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        """尝试加入一个矩形，返回这次真正新增的遮挡 patch 数。"""

        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))

            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)
                window = mask[top : top + height, left : left + width]
                already_masked = int(window.sum())
                newly_masked = height * width - already_masked

                # 不允许单个矩形让总遮挡量越过本轮剩余额度。
                if 0 < newly_masked <= max_mask_patches:
                    window[:] = True
                    delta = newly_masked
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        """返回形状为 ``[grid_h, grid_w]`` 的 numpy bool mask。"""

        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            remaining = num_masking_patches - mask_count
            block_budget = min(remaining, self.max_num_patches)
            delta = self._mask(mask, block_budget)
            if delta == 0:
                # 随机尝试无法再放入合法矩形时提前停止，因此实际遮挡数量
                # 可能略小于 num_masking_patches。
                break
            mask_count += delta
        return mask
