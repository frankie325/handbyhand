"""RAOD-benchmark 道路遗落物检测数据集。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

# RAOD 数据集只有一类：遗落物 (Abandoned Object)
RAOD_CLASSES = ("abandoned_object",)


class RaodDetection(Dataset):
    """读取 RAOD YOLO 格式标注，并返回适用于 DETR 的图片和 target。"""

    classes = RAOD_CLASSES
    class_to_idx = {name: index for index, name in enumerate(RAOD_CLASSES)}

    def __init__(
        self,
        root: Path,
        image_set: str,
        transforms: Callable | None = None,
        train_ratio: float = 0.8,
    ) -> None:
        self.root = root / "RAOD-benchmark"
        self.image_set = image_set
        self._transforms = transforms

        self.image_dir = self.root / "select_1000_images"
        self.label_dir = self.root / "select_1000_txts"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"RAOD 图片目录不存在: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"RAOD 标注目录不存在: {self.label_dir}")

        # 获取所有图片 ID（文件名去掉后缀）
        all_ids = sorted([f.stem for f in self.image_dir.glob("*.jpg")])
        if not all_ids:
            raise ValueError(f"RAOD 图片目录为空: {self.image_dir}")

        # 简单按比例划分训练集和验证集
        num_train = int(len(all_ids) * train_ratio)
        if image_set == "train":
            self.ids = tuple(all_ids[:num_train])
        elif image_set == "val":
            self.ids = tuple(all_ids[num_train:])
        else:
            raise ValueError(f"image_set 应为 'train' 或 'val'，收到: {image_set!r}")

        # 预解析标注
        self._records = {
            image_id: self._parse_annotation(image_id) for image_id in self.ids
        }

    def _parse_annotation(self, image_id: str) -> dict:
        label_file = self.label_dir / f"{image_id}.txt"
        image_file = self.image_dir / f"{image_id}.jpg"

        with Image.open(image_file) as img:
            width, height = img.size

        boxes: list[list[float]] = []
        labels: list[int] = []

        if label_file.is_file():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    class_id = int(parts[0])
                    # YOLO 格式: class_id, x_center, y_center, w, h (均为归一化坐标)
                    cx, cy, w, h = map(float, parts[1:])

                    # 转换为绝对 xyxy 坐标
                    xmin = (cx - w / 2) * width
                    ymin = (cy - h / 2) * height
                    xmax = (cx + w / 2) * width
                    ymax = (cy + h / 2) * height

                    # 标注可能因四舍五入超出图片边界，例如 xmin=-1。
                    # 将框裁剪到有效图像区域 [0, width] × [0, height]。
                    xmin = max(0.0, min(xmin, float(width)))
                    ymin = max(0.0, min(ymin, float(height)))
                    xmax = max(0.0, min(xmax, float(width)))
                    ymax = max(0.0, min(ymax, float(height)))

                    # 裁剪后没有有效面积的框直接丢弃，避免产生无效监督目标。
                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
            boxes_tensor[:, 3] - boxes_tensor[:, 1]
        )

        return {
            "image_id": image_id,
            "filename": f"{image_id}.jpg",
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "area": areas,
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "difficult": torch.zeros(len(boxes), dtype=torch.bool),
        }

    def get_ground_truth(self, image_id: str) -> dict:
        """返回原图坐标下的完整 GT。"""
        try:
            record = self._records[image_id]
        except KeyError as error:
            raise KeyError(f"RAOD 划分中不存在图片 ID: {image_id}") from error

        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in record.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        record = self._records[image_id]

        image_path = self.image_dir / record["filename"]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        # DETR 期望的 target 格式
        target = {
            "boxes": record["boxes"].clone(),
            "labels": record["labels"].clone(),
            "image_id": image_id,
            "area": record["area"].clone(),
            "orig_size": record["orig_size"].clone(),
            "size": record["orig_size"].clone(),
        }

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target
