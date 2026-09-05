"""Convert the local RAOD YOLO labels to COCO JSON for official RT-DETR.

The official PyTorch RT-DETR trainer consumes torchvision's CocoDetection
dataset. RAOD stores one class of objects in YOLO ``cx cy w h`` text files, so
this small, deterministic preparation step keeps the upstream training and
evaluation code unchanged while producing the COCO format it expects.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


CLASS_ID = 0
CLASS_NAME = "abandoned_object"


def _parse_yolo_boxes(label_path: Path, width: int, height: int) -> tuple[list[list[float]], int, int]:
    boxes: list[list[float]] = []
    clipped = 0
    discarded = 0

    if not label_path.is_file():
        return boxes, clipped, discarded

    for line_number, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}")

        class_id = int(fields[0])
        if class_id != CLASS_ID:
            raise ValueError(
                f"{label_path}:{line_number}: expected class {CLASS_ID}, got {class_id}"
            )

        cx, cy, box_width, box_height = (float(value) for value in fields[1:])
        if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height)):
            raise ValueError(f"{label_path}:{line_number}: non-finite box value")

        xmin = (cx - box_width / 2) * width
        ymin = (cy - box_height / 2) * height
        xmax = (cx + box_width / 2) * width
        ymax = (cy + box_height / 2) * height

        original_box = (xmin, ymin, xmax, ymax)
        xmin = max(0.0, min(xmin, float(width)))
        ymin = max(0.0, min(ymin, float(height)))
        xmax = max(0.0, min(xmax, float(width)))
        ymax = max(0.0, min(ymax, float(height)))
        if (xmin, ymin, xmax, ymax) != original_box:
            clipped += 1

        if xmax <= xmin or ymax <= ymin:
            discarded += 1
            continue

        boxes.append([xmin, ymin, xmax, ymax])

    return boxes, clipped, discarded


def _build_split(
    image_paths: list[Path],
    image_ids: dict[str, int],
    label_dir: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    stats = {"images": 0, "boxes": 0, "clipped": 0, "discarded": 0}
    annotation_id = 1

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        image_id = image_ids[image_path.stem]
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        stats["images"] += 1

        boxes, clipped, discarded = _parse_yolo_boxes(
            label_dir / f"{image_path.stem}.txt", width, height
        )
        stats["clipped"] += clipped
        stats["discarded"] += discarded
        for xmin, ymin, xmax, ymax in boxes:
            box_width = xmax - xmin
            box_height = ymax - ymin
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CLASS_ID,
                    "bbox": [xmin, ymin, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            stats["boxes"] += 1

    return (
        {
            "info": {"description": "RAOD-benchmark converted from YOLO labels"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": CLASS_ID, "name": CLASS_NAME, "supercategory": "object"}
            ],
        },
        stats,
    )


def prepare(dataset_root: Path, output_dir: Path, train_ratio: float) -> None:
    image_dir = dataset_root / "select_1000_images"
    label_dir = dataset_root / "select_1000_txts"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"RAOD image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"RAOD label directory does not exist: {label_dir}")

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"RAOD image directory is empty: {image_dir}")
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    # Match the existing handwritten RAOD loader: lexicographic order and an
    # 80/20 deterministic split. The preparation command is therefore
    # repeatable and does not silently reshuffle the benchmark.
    num_train = int(len(image_paths) * train_ratio)
    splits = {"train": image_paths[:num_train], "val": image_paths[num_train:]}
    image_ids = {path.stem: index + 1 for index, path in enumerate(image_paths)}
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_paths in splits.items():
        coco, stats = _build_split(split_paths, image_ids, label_dir)
        output_path = output_dir / f"instances_{split_name}.json"
        output_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n")
        print(
            f"{split_name}: images={stats['images']} boxes={stats['boxes']} "
            f"clipped={stats['clipped']} discarded={stats['discarded']} -> {output_path}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_dataset_root = project_root / "datasets" / "RAOD-benchmark"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument(
        "--output-dir", type=Path, default=default_dataset_root / "annotations"
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    prepare(args.dataset_root, args.output_dir, args.train_ratio)


if __name__ == "__main__":
    main()
