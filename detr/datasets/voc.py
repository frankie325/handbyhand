"""原生 PASCAL VOC 目标检测数据集。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from xml.etree import ElementTree

import torch
from PIL import Image
from torch.utils.data import Dataset


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def _required_text(element: ElementTree.Element, path: str, annotation_file: Path) -> str:
    value = element.findtext(path)
    if value is None or not value.strip():
        raise ValueError(f"标注 {annotation_file} 缺少字段: {path}")
    return value.strip()


class VocDetection(Dataset):
    """读取 VOC XML，并返回适用于 DETR 的图片和 target。

    返回给损失函数的 target 会排除 ``difficult=1`` 的目标。完整原始标注
    保存在内部，并可通过 :meth:`get_ground_truth` 供 VOC 评估器使用。
    """

    classes = VOC_CLASSES
    class_to_idx = {name: index for index, name in enumerate(VOC_CLASSES)}

    def __init__(
        self,
        root: str | Path,
        image_set: str,
        transforms: Callable | None = None,
    ) -> None:
        if image_set not in {"train", "val"}:
            raise ValueError(f"image_set 应为 'train' 或 'val'，收到: {image_set!r}")

        self.root = Path(root)
        self.image_set = image_set
        self.image_dir = self.root / "JPEGImages"
        self.annotation_dir = self.root / "Annotations"
        self.split_file = self.root / "ImageSets" / "Main" / f"{image_set}.txt"
        self._transforms = transforms

        for directory in (self.image_dir, self.annotation_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"VOC 数据目录不存在: {directory}")
        if not self.split_file.is_file():
            raise FileNotFoundError(f"VOC 划分文件不存在: {self.split_file}")

        ids = [line.strip() for line in self.split_file.read_text().splitlines() if line.strip()]
        if not ids:
            raise ValueError(f"VOC 划分文件为空: {self.split_file}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"VOC 划分文件包含重复图片 ID: {self.split_file}")

        self.ids = tuple(ids)
        self._records = {image_id: self._parse_annotation(image_id) for image_id in self.ids}
        self.num_difficult = sum(
            int(record["difficult"].sum().item()) for record in self._records.values()
        )

    def _parse_annotation(self, image_id: str) -> dict:
        annotation_file = self.annotation_dir / f"{image_id}.xml"
        if not annotation_file.is_file():
            raise FileNotFoundError(f"VOC 标注文件不存在: {annotation_file}")

        root = ElementTree.parse(annotation_file).getroot()
        filename = _required_text(root, "filename", annotation_file)
        expected_filename = f"{image_id}.jpg"
        if filename != expected_filename:
            raise ValueError(
                f"标注 {annotation_file} 的 filename={filename!r}，"
                f"与图片 ID 对应的 {expected_filename!r} 不一致"
            )

        image_file = self.image_dir / filename
        if not image_file.is_file():
            raise FileNotFoundError(f"VOC 图片不存在: {image_file}")

        width = int(_required_text(root, "size/width", annotation_file))
        height = int(_required_text(root, "size/height", annotation_file))
        if width <= 0 or height <= 0:
            raise ValueError(f"标注 {annotation_file} 的图片尺寸无效: {width}x{height}")

        boxes: list[list[float]] = []
        labels: list[int] = []
        difficult: list[bool] = []
        for object_element in root.findall("object"):
            class_name = _required_text(object_element, "name", annotation_file)
            if class_name not in self.class_to_idx:
                raise ValueError(f"标注 {annotation_file} 包含未知类别: {class_name!r}")

            xmin = int(_required_text(object_element, "bndbox/xmin", annotation_file))
            ymin = int(_required_text(object_element, "bndbox/ymin", annotation_file))
            xmax = int(_required_text(object_element, "bndbox/xmax", annotation_file))
            ymax = int(_required_text(object_element, "bndbox/ymax", annotation_file))
            if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
                raise ValueError(
                    f"标注 {annotation_file} 的边界框越界: "
                    f"({xmin}, {ymin}, {xmax}, {ymax}) / {width}x{height}"
                )

            # VOC 坐标是 1-based inclusive。转换为连续 xyxy 后，整张宽 W 的
            # 图片范围是 [0, W]，因此只需将左上角减 1。
            boxes.append([float(xmin - 1), float(ymin - 1), float(xmax), float(ymax)])
            labels.append(self.class_to_idx[class_name])
            difficult.append(
                bool(int(_required_text(object_element, "difficult", annotation_file)))
            )

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        difficult_tensor = torch.tensor(difficult, dtype=torch.bool)
        areas = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        )
        size = torch.tensor([height, width], dtype=torch.int64)
        return {
            "image_id": image_id,
            "filename": filename,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "difficult": difficult_tensor,
            "area": areas,
            "orig_size": size,
        }

    def get_ground_truth(self, image_id: str) -> dict:
        """返回原图坐标下的完整 GT，包括 difficult 目标。"""
        try:
            record = self._records[image_id]
        except KeyError as error:
            raise KeyError(f"VOC 划分中不存在图片 ID: {image_id}") from error

        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in record.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        ground_truth = self.get_ground_truth(image_id)

        image_path = self.image_dir / ground_truth["filename"]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        annotated_height, annotated_width = ground_truth["orig_size"].tolist()
        if image.size != (annotated_width, annotated_height):
            raise ValueError(
                f"图片 {image_path} 的实际尺寸 {image.size} 与 XML 中的 "
                f"{(annotated_width, annotated_height)} 不一致"
            )

        # difficult 目标仍留在图像中，但不作为 DETR 损失的监督目标。
        keep = ~ground_truth["difficult"]
        size = ground_truth["orig_size"].clone()
        target = {
            "boxes": ground_truth["boxes"][keep],
            "labels": ground_truth["labels"][keep],
            "image_id": image_id,
            "area": ground_truth["area"][keep],
            "orig_size": size.clone(),
            "size": size,
        }

        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target
