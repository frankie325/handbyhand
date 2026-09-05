"""RAOD-benchmark 道路遗落物检测数据集。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

RAOD_CLASSES = ("abandoned_object",)


class RaodDetection(Dataset):
    """读取 RAOD 的 YOLO 标注，返回 DETR 所需的 image 和 target。"""

    classes = RAOD_CLASSES

    def __init__(
        self,
        root: Path | str,
        image_set: str,
        transforms: Callable | None = None,
        train_ratio: float = 0.8,
    ) -> None:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio 必须位于 (0, 1) 范围")

        self.root = Path(root)
        self.image_set = image_set
        self.transforms = transforms
        self.image_dir = self.root / "select_1000_images"
        self.label_dir = self.root / "select_1000_txts"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"RAOD 图片目录不存在: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"RAOD 标注目录不存在: {self.label_dir}")

        all_ids = sorted(path.stem for path in self.image_dir.glob("*.jpg"))
        if not all_ids:
            raise ValueError(f"RAOD 图片目录为空: {self.image_dir}")

        split_index = int(len(all_ids) * train_ratio)
        if image_set == "train":
            self.ids = tuple(all_ids[:split_index])
        elif image_set in {"val", "test"}:
            # RAOD-benchmark 没有独立 test 标注，test 复用固定的留出验证集。
            self.ids = tuple(all_ids[split_index:])
        else:
            raise ValueError(
                f"image_set 应为 'train'、'val' 或 'test'，收到: {image_set!r}"
            )

        self._records = {
            image_id: self._parse_annotation(image_id) for image_id in self.ids
        }

    def _parse_annotation(self, image_id: str) -> dict[str, object]:
        image_file = self.image_dir / f"{image_id}.jpg"
        label_file = self.label_dir / f"{image_id}.txt"
        with Image.open(image_file) as image:
            width, height = image.size

        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_file.is_file():
            with label_file.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    fields = line.split()
                    if not fields:
                        continue
                    if len(fields) != 5:
                        raise ValueError(
                            f"{label_file}:{line_number} 应包含 5 列，实际为 {len(fields)}"
                        )

                    class_id = int(fields[0])
                    if class_id != 0:
                        raise ValueError(
                            f"{label_file}:{line_number} 出现未知类别 {class_id}；"
                            "RAOD 配置只允许类别 0"
                        )
                    center_x, center_y, box_width, box_height = map(
                        float, fields[1:]
                    )
                    x_min = (center_x - box_width / 2) * width
                    y_min = (center_y - box_height / 2) * height
                    x_max = (center_x + box_width / 2) * width
                    y_max = (center_y + box_height / 2) * height

                    # 少量标注因小数舍入会越过边界，进入训练前统一裁剪。
                    x_min = min(max(x_min, 0.0), float(width))
                    y_min = min(max(y_min, 0.0), float(height))
                    x_max = min(max(x_max, 0.0), float(width))
                    y_max = min(max(y_max, 0.0), float(height))
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (box_tensor[:, 2] - box_tensor[:, 0]) * (
            box_tensor[:, 3] - box_tensor[:, 1]
        )
        size = torch.tensor([height, width], dtype=torch.int64)
        return {
            "image_id": image_id,
            "filename": image_file.name,
            "boxes": box_tensor,
            "labels": label_tensor,
            "area": area,
            "orig_size": size,
            "difficult": torch.zeros(len(box_tensor), dtype=torch.bool),
        }

    def get_ground_truth(self, image_id: str) -> dict[str, object]:
        try:
            record = self._records[image_id]
        except KeyError as error:
            raise KeyError(f"当前 RAOD 划分中不存在图片 ID: {image_id}") from error
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in record.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        record = self._records[image_id]
        image_path = self.image_dir / str(record["filename"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        target = {
            "boxes": record["boxes"].clone(),
            "labels": record["labels"].clone(),
            "image_id": image_id,
            "area": record["area"].clone(),
            "orig_size": record["orig_size"].clone(),
            "size": record["orig_size"].clone(),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
