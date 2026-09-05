from __future__ import annotations

import colorsys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .model.build import build_model
from .config import DATASET_TYPE
from .datasets import DATASET_REGISTRY
from .datasets.transforms import make_detection_transforms
from .utils.bos_ops import box_cxcywh_to_xyxy
from .utils.common import get_device
from .config import OUTPUTS_DIR


def _class_color(class_index: int) -> tuple[int, int, int]:
    """为每个类别生成稳定且容易区分的颜色。"""
    # 使用黄金比例遍历色相，类别数量增加时也不会依赖固定长度的颜色表。
    hue = (class_index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return round(red * 255), round(green * 255), round(blue * 255)


def _load_label_font(image_size: tuple[int, int]) -> ImageFont.FreeTypeFont:
    """加载较醒目的粗体；不同系统找不到字体时回退到 PIL 默认字体。"""
    width, height = image_size
    font_size = max(16, round(min(width, height) / 45))
    font_candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    )
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


@torch.inference_mode()
def predict(image_paths: list[Path]) -> None:
    threshold = 0.5

    device = get_device()
    model = build_model(False, device)
    model.eval()

    # 预测使用和验证集相同的确定性预处理。
    transform = make_detection_transforms("val")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        original_image = Image.open(image_path).convert("RGB")
        original_width, original_height = original_image.size

        image_tensor, _ = transform(original_image, None)
        images = image_tensor.unsqueeze(0).to(device)

        # 单张图片没有padding，因此mask全部为False。
        padding_mask = torch.zeros(
            (1, image_tensor.shape[1], image_tensor.shape[2]),
            dtype=torch.bool,
            device=device,
        )

        outputs = model(images, padding_mask)

        # 最后一列是 no-object，只在 VOC20 个真实类别中选择最高分。
        probabilities = outputs["pred_logits"][0].softmax(dim=-1)[..., :-1]
        scores, labels = probabilities.max(dim=-1)

        # boxes是相对图片归一化的坐标
        boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][0])
        # 这里进行还原，画在真实图片上，必须乘以图片的宽高
        boxes = boxes * boxes.new_tensor(
            [
                original_width,
                original_height,
                original_width,
                original_height,
            ]
        )

        # boxes不能超出原始图片区域
        boxes[:, 0::2].clamp_(0, original_width)
        boxes[:, 1::2].clamp_(0, original_height)

        keep = scores >= threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        # 按置信度从高到低排列。
        order = scores.argsort(descending=True)
        scores = scores[order].cpu()
        labels = labels[order].cpu()
        boxes = boxes[order].cpu()

        result_image = original_image.copy()
        drawer = ImageDraw.Draw(result_image)
        label_font = _load_label_font(result_image.size)

        for score, label, box in zip(scores, labels, boxes):
            x_min, y_min, x_max, y_max = box.tolist()
            dataset_cls = DATASET_REGISTRY[DATASET_TYPE]
            category_name = dataset_cls.classes[int(label)]
            text = f"{category_name} {score.item():.2f}"
            color = _class_color(int(label))

            drawer.rectangle(
                (x_min, y_min, x_max, y_max),
                outline=color,
                width=max(2, round(min(original_width, original_height) / 360)),
            )

            # 给文字加彩色底框和黑色描边，避免文字和道路/物体纹理混在一起。
            text_bbox = drawer.textbbox(
                (0, 0),
                text,
                font=label_font,
                stroke_width=1,
            )
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            padding = 4
            label_left = max(0, x_min)
            label_top = y_min - text_height - padding * 2
            if label_top < 0:
                label_top = min(
                    y_max,
                    original_height - text_height - padding * 2,
                )
            label_top = max(0, label_top)
            label_right = min(
                original_width,
                label_left + text_width + padding * 2,
            )
            label_bottom = min(
                original_height,
                label_top + text_height + padding * 2,
            )
            drawer.rounded_rectangle(
                (label_left, label_top, label_right, label_bottom),
                radius=4,
                fill=color,
            )
            drawer.text(
                (
                    label_left + padding - text_bbox[0],
                    label_top + padding - text_bbox[1],
                ),
                text,
                font=label_font,
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )

        output_path = OUTPUTS_DIR / image_path.name
        result_image.save(output_path)

        print(f"{image_path}: 检测到 {len(boxes)} 个目标")
        print("结果保存到:", output_path)


def main() -> None:
    # TODO: 在这里手动填写要预测的图片路径
    image_dir = Path("/Users/frank/code/ai/yolo_data/VOC2007/JPEGImages/000622.jpg")
    image_paths: list[Path] = [image_dir]
    predict(image_paths)


if __name__ == "__main__":
    main()
