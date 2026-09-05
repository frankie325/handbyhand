"""使用训练好的 Deformable DETR 预测任意图片。

运行示例：
    python -m deformable_detr.predict image.jpg
    python -m deformable_detr.predict images/ --score-threshold 0.1
    python -m deformable_detr.predict images/ --recursive --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps

from .config import MODELS_DIR, OUTPUTS_DIR
from .datasets import RAOD_CLASSES
from .datasets.transforms import make_detection_transforms
from .metrics import outputs_to_results
from .model.build import build_model
from .utils.common import get_device

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
BOX_COLOR = (255, 64, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict images with the RAOD Deformable DETR model"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="一张或多张图片，也可以传入图片目录",
    )
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "best.pth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "predictions",
        help="画框图片和 predictions.json 的保存目录",
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="每张图片最多保留的高分预测数量",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 mps")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归查找输入目录中的图片",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="将图片短边缩放到 320，便于快速调试；正常预测默认使用 800",
    )
    return parser.parse_args()


def collect_image_paths(inputs: list[Path], recursive: bool) -> list[Path]:
    """展开文件和目录，返回去重并排序后的图片路径。"""
    image_paths: set[Path] = set()
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"输入路径不存在: {input_path}")

        if input_path.is_file():
            if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"不支持的图片格式: {input_path}")
            image_paths.add(input_path.resolve())
            continue

        if not input_path.is_dir():
            raise ValueError(f"输入路径既不是图片也不是目录: {input_path}")
        candidates = input_path.rglob("*") if recursive else input_path.iterdir()
        image_paths.update(
            path.resolve()
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not image_paths:
        raise ValueError("输入路径中没有找到支持的图片")
    return sorted(image_paths, key=lambda path: path.as_posix())


def _class_name(label: int) -> str:
    if 0 <= label < len(RAOD_CLASSES):
        return RAOD_CLASSES[label]
    return f"class_{label}"


def _draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """在原始分辨率图片上绘制检测框、类别和置信度。"""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    line_width = max(2, round(min(result.size) / 400))

    for detection in detections:
        x_min, y_min, x_max, y_max = detection["bbox"]
        draw.rectangle(
            (x_min, y_min, x_max, y_max),
            outline=BOX_COLOR,
            width=line_width,
        )

        text = f"{detection['class_name']} {detection['score']:.2f}"
        text_box = draw.textbbox((0, 0), text)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = min(
            max(0.0, x_min),
            max(0.0, result.width - text_width - 6),
        )
        text_y = max(0.0, y_min - text_height - 6)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + 6, text_y + text_height + 6),
            fill=BOX_COLOR,
        )
        draw.text((text_x + 3, text_y + 3), text, fill="white")

    return result


def _make_output_path(
    image_path: Path,
    output_dir: Path,
    used_paths: set[Path],
) -> Path:
    """生成输出文件名，并避免同名输入图片在本次预测中互相覆盖。"""
    suffix = image_path.suffix.lower()
    candidate = output_dir / f"{image_path.stem}_pred{suffix}"
    index = 2
    while candidate in used_paths:
        candidate = output_dir / f"{image_path.stem}_pred_{index}{suffix}"
        index += 1
    used_paths.add(candidate)
    return candidate


@torch.inference_mode()
def predict_image(
    model,
    image_path: Path,
    transform,
    device: torch.device,
    score_threshold: float,
    top_k: int,
) -> tuple[Image.Image, list[dict]]:
    """预测一张图片，返回原图以及像素坐标下的检测结果。"""
    with Image.open(image_path) as source:
        original_image = ImageOps.exif_transpose(source).convert("RGB")

    image_tensor, _ = transform(original_image, None)
    images = image_tensor.unsqueeze(0).to(device)
    padding_mask = torch.zeros(
        (1, image_tensor.shape[-2], image_tensor.shape[-1]),
        dtype=torch.bool,
        device=device,
    )

    outputs = model(images, padding_mask)
    original_width, original_height = original_image.size
    target = {
        "image_id": str(image_path),
        "orig_size": torch.tensor([original_height, original_width]),
    }
    results = outputs_to_results(
        outputs,
        [target],
        score_threshold=score_threshold,
    )
    results.sort(key=lambda result: result["score"], reverse=True)

    detections = []
    for result in results[:top_k]:
        label = int(result["label"])
        detections.append(
            {
                "label": label,
                "class_name": _class_name(label),
                "score": float(result["score"]),
                "bbox": [float(value) for value in result["bbox"]],
            }
        )
    return original_image, detections


def main() -> None:
    args = parse_args()
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("score-threshold 必须位于 [0, 1]")
    if args.top_k <= 0:
        raise ValueError("top-k 必须大于 0")

    image_paths = collect_image_paths(args.inputs, args.recursive)
    device = get_device(args.device)
    model = build_model(
        train=False,
        device=device,
        checkpoint_path=args.checkpoint,
        pretrained_backbone=False,
    )
    model.eval()
    transform = make_detection_transforms("val", debug=args.debug)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_output_paths: set[Path] = set()
    prediction_records = []
    total_detections = 0

    print(f"设备: {device}")
    print(f"权重: {args.checkpoint}")
    for image_index, image_path in enumerate(image_paths, start=1):
        original_image, detections = predict_image(
            model=model,
            image_path=image_path,
            transform=transform,
            device=device,
            score_threshold=args.score_threshold,
            top_k=args.top_k,
        )
        output_path = _make_output_path(
            image_path, args.output_dir, used_output_paths
        )
        _draw_detections(original_image, detections).save(output_path)

        width, height = original_image.size
        prediction_records.append(
            {
                "image": str(image_path),
                "width": width,
                "height": height,
                "output_image": str(output_path.resolve()),
                "detections": detections,
            }
        )
        total_detections += len(detections)
        print(
            f"[{image_index}/{len(image_paths)}] {image_path.name}: "
            f"{len(detections)} 个目标 -> {output_path}"
        )

    json_path = args.output_dir / "predictions.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "score_threshold": args.score_threshold,
        "top_k": args.top_k,
        "num_images": len(image_paths),
        "num_detections": total_detections,
        "predictions": prediction_records,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成：{len(image_paths)} 张图片，共 {total_detections} 个目标")
    print(f"JSON 结果: {json_path}")


if __name__ == "__main__":
    main()
