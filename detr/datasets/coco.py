"""基于 pycocotools 的 COCO 目标检测数据集。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision.datasets import CocoDetection as TorchvisionCocoDetection


class ConvertCocoAnnotations:
    """把 COCO 原始标注整理为可和图片一起做增强的 DETR target。

    此处输出的 boxes 仍是像素尺度的 xyxy；完成缩放、裁剪等几何增强后，
    Normalize 才会把它转换成模型训练所需的归一化 cxcywh。
    """

    def __call__(
        self,
        image: Image.Image,
        target: dict,
    ) -> tuple[Image.Image, dict[str, torch.Tensor]]:
        width, height = image.size
        image_id = torch.tensor([target["image_id"]], dtype=torch.int64)

        # crowd 标注通常表示一群难以逐个分开的目标。原版 DETR 的目标检测训练
        # 不把它们作为普通实例监督，因此在这里过滤掉 iscrowd=1 的标注。
        annotations = [
            annotation
            for annotation in target["annotations"]
            if annotation.get("iscrowd", 0) == 0
        ]

        # 每个原始 bbox 为 [x, y, width, height]，先组成 [num_objects, 4]。每个物体一个 [x, y, w, h]）
        boxes = torch.as_tensor(
            [annotation["bbox"] for annotation in annotations],
            dtype=torch.float32,
        ).reshape(-1, 4)
        if boxes.numel() > 0:
            # ?为什么需要裁剪COCO 的标注框有时会稍微超出图片边界（标注者画的框不总是完全在图像内）
            # COCO 的 bbox 是 [x, y, width, height]，先转成绝对坐标 xyxy。
            boxes[:, 2:] += boxes[
                :, :2
            ]  # [x , y, width + x, height + y] -> [x1, y1, x2, y2]
            boxes[:, 0::2].clamp_(min=0, max=width)  # x1, x2 限制在 [0, width]
            boxes[:, 1::2].clamp_(min=0, max=height)  # y1, y2 限制在 [0, height]

        # 保留 COCO 原始 category_id（1~90，中间存在空缺），不重新映射到 0~79。
        labels = torch.as_tensor(
            [annotation["category_id"] for annotation in annotations],
            dtype=torch.int64,
        )
        area = torch.as_tensor(
            [
                annotation.get(
                    "area",
                    annotation["bbox"][2] * annotation["bbox"][3],
                )
                for annotation in annotations
            ],
            dtype=torch.float32,
        )
        iscrowd = torch.as_tensor(
            [annotation.get("iscrowd", 0) for annotation in annotations],
            dtype=torch.int64,
        )

        # 裁剪后可能出现 x1 == x2 或 y1 == y2（宽度或高度变成 0）
        # 宽度或高度 ≤ 0 的框没有有效面积，会被 boxes[keep]、labels[keep] 等删除
        # label、area、iscrowd 一起删除，确保这些字段的第一维始终对齐。
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        keep = (box_sizes[:, 0] > 0) & (
            box_sizes[:, 1] > 0
        )  # tensor([True, True, False, ...])  宽度和高度是否都 > 0

        # orig_size 永远记录原图尺寸；size 会在后续 resize/crop 时持续更新。
        # 二者都使用 [height, width]，而不是 PIL 的 [width, height] 顺序。
        size = torch.tensor([height, width], dtype=torch.int64)
        converted_target = {
            "boxes": boxes[keep],  # [num_objects, 4]
            "labels": labels[keep],  # [num_objects] 类别id集合
            "image_id": image_id,
            "area": area[keep],
            "iscrowd": iscrowd[keep],
            "orig_size": size.clone(),
            "size": size,
        }
        return image, converted_target


class CocoDetection(TorchvisionCocoDetection):
    """使用 pycocotools 索引标注，并返回经过联合变换的图片和 target。"""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        transforms: Callable | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"图片目录不存在: {self.image_dir}")
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_file}")

        # 第 1 步：TorchVision 内部创建 pycocotools.coco.COCO。
        # self.coco 保存图片、标注和类别索引；self.ids 保存排序后的 image_id。
        super().__init__(
            root=self.image_dir,
            annFile=str(self.annotation_file),
        )
        self._transforms = transforms
        self.prepare = ConvertCocoAnnotations()
        self.categories = {
            category_id: category["name"]
            for category_id, category in self.coco.cats.items()
        }

        missing_images = [
            image["file_name"]
            for image in self.coco.imgs.values()
            if not (self.image_dir / image["file_name"]).is_file()
        ]
        if missing_images:
            examples = ", ".join(missing_images[:3])
            raise FileNotFoundError(
                f"标注中有 {len(missing_images)} 张图片不存在，例如: {examples}"
            )

    @property
    def category_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.coco.getCatIds()))

    def __getitem__(self, index: int):
        # 第 2 步：父类根据 self.ids[index] 查询文件名和标注。
        # pycocotools 负责查询 JSON 索引；TorchVision 使用 PIL 打开 JPEG。
        # 此时 image 是 PIL.Image，annotations 是该图片对应的原始标注列表。
        image, annotations = super().__getitem__(index)
        image_id = self.ids[index]
        target = {
            "image_id": image_id,
            "annotations": annotations,
        }
        # 第 3 步：过滤 crowd，将 bbox 从 COCO xywh 转为像素坐标 xyxy，
        # 并整理出 boxes、labels、area、尺寸等训练所需字段。
        image, target = self.prepare(image, target)

        if self._transforms is not None:
            # 第 4 步：图片和 boxes 必须同步翻转、缩放、裁剪；最后再转 Tensor、
            # 标准化图片，并把 boxes 变为 [0, 1] 范围的 cxcywh。
            image, target = self._transforms(image, target)

        # iamge.shape [3, height, width]
        return image, target
