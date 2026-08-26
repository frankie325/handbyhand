"""与原版 DETR 一致的图像和检测框联合变换。

凡是改变图片几何位置或尺寸的操作，都必须对 target["boxes"] 做相同操作，
否则图片中的物体与监督框就会错位。
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import torch
from PIL import Image
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as F


Target = dict[str, torch.Tensor]


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """把 [x_min, y_min, x_max, y_max] 转换为 [center_x, center_y, width, height]。"""
    x_min, y_min, x_max, y_max = boxes.unbind(-1)
    return torch.stack(
        (
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            x_max - x_min,
            y_max - y_min,
        ),
        dim=-1,
    )


def crop(
    image: Image.Image,
    target: Target | None,
    region: tuple[int, int, int, int],
) -> tuple[Image.Image, Target | None]:
    """裁剪图片，并同步平移、裁剪检测框。"""
    cropped_image = F.crop(image, *region)
    if target is None:
        return cropped_image, None

    target = target.copy()
    top, left, height, width = region
    target["size"] = torch.tensor([height, width])

    boxes = target["boxes"]
    # 裁剪区域的左上角会成为新图片的 (0, 0)，所以所有框先减去
    # [left, top]，然后再限制到新图片的宽高范围内。
    offset = torch.tensor([left, top, left, top], dtype=boxes.dtype)
    max_size = torch.tensor([width, height], dtype=boxes.dtype)
    cropped_boxes = boxes - offset
    cropped_boxes = torch.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    cropped_boxes = cropped_boxes.reshape(-1, 4)

    # 完全落在裁剪区域外的框会退化成零宽或零高，需要同步删除。
    # 随机裁剪后 target 可能为空，这是合法训练样本。
    box_sizes = cropped_boxes[:, 2:] - cropped_boxes[:, :2]
    keep = (box_sizes[:, 0] > 0) & (box_sizes[:, 1] > 0)

    target["boxes"] = cropped_boxes[keep]
    target["labels"] = target["labels"][keep]
    target["iscrowd"] = target["iscrowd"][keep]
    target["area"] = (box_sizes[:, 0] * box_sizes[:, 1])[keep]
    return cropped_image, target


def hflip(
    image: Image.Image,
    target: Target | None,
) -> tuple[Image.Image, Target | None]:
    """水平翻转图片和检测框。"""
    flipped_image = F.hflip(image)
    if target is None:
        return flipped_image, None

    target = target.copy()
    width, _ = image.size
    boxes = target["boxes"]
    # 水平翻转后的横坐标为 width - x；纵坐标保持不变。
    # 重新排列 x_max/x_min 后即可得到新的 [x_min, y_min, x_max, y_max]。
    boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor(
        [-1, 1, -1, 1], dtype=boxes.dtype
    )
    boxes = boxes + torch.tensor([width, 0, width, 0], dtype=boxes.dtype)
    target["boxes"] = boxes
    return flipped_image, target


def _get_size_with_aspect_ratio(
    image_size: tuple[int, int],
    size: int,
    max_size: int | None = None,
) -> tuple[int, int]:
    width, height = image_size

    if max_size is not None:
        # 先以短边 size 缩放；若按该比例缩放后长边超过 max_size，
        # 就反算短边尺寸，使长边恰好不超过 max_size。
        min_original_size = float(min(width, height))
        max_original_size = float(max(width, height))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (width <= height and width == size) or (height <= width and height == size):
        return height, width

    if width < height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)
    return new_height, new_width


def resize(
    image: Image.Image,
    target: Target | None,
    size: int | tuple[int, int],
    max_size: int | None = None,
) -> tuple[Image.Image, Target | None]:
    """保持宽高比缩放图片，并同步缩放检测框。"""
    if isinstance(size, int):
        output_size = _get_size_with_aspect_ratio(image.size, size, max_size)
    else:
        output_size = size

    resized_image = F.resize(image, output_size)
    if target is None:
        return resized_image, None

    target = target.copy()
    old_width, old_height = image.size
    new_width, new_height = resized_image.size
    ratio_width = new_width / old_width
    ratio_height = new_height / old_height

    # x 坐标乘宽度比例，y 坐标乘高度比例；area 乘两个比例的乘积。
    boxes = target["boxes"]
    scale = torch.tensor(
        [ratio_width, ratio_height, ratio_width, ratio_height],
        dtype=boxes.dtype,
    )
    target["boxes"] = boxes * scale
    target["area"] = target["area"] * ratio_width * ratio_height
    target["size"] = torch.tensor([new_height, new_width])
    return resized_image, target


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(
        self,
        image: Image.Image | torch.Tensor,
        target: Target | None,
    ) -> tuple[Image.Image | torch.Tensor, Target | None]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, image: Image.Image, target: Target | None):
        if random.random() < self.probability:
            return hflip(image, target)
        return image, target


class RandomResize:
    def __init__(self, sizes: Sequence[int], max_size: int | None = None):
        if not sizes:
            raise ValueError("sizes 不能为空")
        self.sizes = tuple(sizes)
        self.max_size = max_size

    def __call__(self, image: Image.Image, target: Target | None):
        size = random.choice(self.sizes)
        return resize(image, target, size, self.max_size)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        if min_size > max_size:
            raise ValueError("min_size 不能大于 max_size")
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image: Image.Image, target: Target | None):
        width = random.randint(self.min_size, min(image.width, self.max_size))
        height = random.randint(self.min_size, min(image.height, self.max_size))
        region = RandomCrop.get_params(image, [height, width])
        return crop(image, target, region)


class RandomSelect:
    """以 probability 选择第一个变换，否则选择第二个变换。"""

    def __init__(self, first: Callable, second: Callable, probability: float = 0.5):
        self.first = first
        self.second = second
        self.probability = probability

    def __call__(self, image: Image.Image, target: Target | None):
        if random.random() < self.probability:
            return self.first(image, target)
        return self.second(image, target)


class ToTensor:
    def __call__(self, image: Image.Image, target: Target | None):
        # PIL: [height, width, channels]、0~255
        # Tensor: [channels, height, width]、float32、0~1
        return F.to_tensor(image), target


class Normalize:
    """归一化图片，并把绝对 xyxy 框转成 [0, 1] 范围的 cxcywh。"""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, target: Target | None):
        # 使用 ImageNet 均值和标准差标准化，和预训练 ResNet-50 的输入分布一致。
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None

        target = target.copy()
        height, width = image.shape[-2:]
        # DETR 预测的是相对当前图片尺寸的 [center_x, center_y, width, height]，
        # 因此这里是检测框进入模型损失前的最后一次格式转换。
        boxes = box_xyxy_to_cxcywh(target["boxes"])
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        # 贴近边界的坐标可能因浮点舍入得到 1.0000001，限制回模型约定的范围。
        target["boxes"] = (boxes / scale).clamp(min=0, max=1)
        return image, target


def make_coco_transforms(image_set: str, debug: bool = False) -> Compose:
    """创建 DETR 论文使用的训练或验证变换。"""
    normalize = Compose(
        [
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if debug:
        # 只用于快速检查代码，处理逻辑与论文配置相同，但图片尺寸更小。
        scales = [256, 288, 320]
        crop_resize_scales = [224, 256, 288]
        crop_min_size, crop_max_size = 192, 320
        validation_size = 320
        max_size = 533
    else:
        # 原论文尺度增强：短边从 480~800 中随机选择，长边最多为 1333。
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        crop_resize_scales = [400, 500, 600]
        crop_min_size, crop_max_size = 384, 600
        validation_size = 800
        max_size = 1333

    if image_set == "train":
        # 训练流程：
        # 1. 以 0.5 概率水平翻转；
        # 2. 再以 0.5 概率二选一：直接随机缩放，或先缩放、随机裁剪、再缩放；
        # 3. 转成 Tensor、标准化图片，并把 boxes 归一化为 cxcywh。
        return Compose(
            [
                RandomHorizontalFlip(),
                RandomSelect(
                    RandomResize(scales, max_size=max_size),
                    Compose(
                        [
                            RandomResize(crop_resize_scales),
                            RandomSizeCrop(crop_min_size, crop_max_size),
                            RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        # 验证时不使用随机增强，只把短边缩放到 800（debug 时为 320），
        # 这样同一张图片每次评估都得到完全相同的输入。
        return Compose(
            [
                RandomResize([validation_size], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"不支持的数据集划分: {image_set!r}，应为 'train' 或 'val'")
