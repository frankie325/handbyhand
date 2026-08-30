from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .model.build import build_model
from detr.datasets import VOC_CLASSES
from detr.datasets.transforms import make_detection_transforms
from detr.utils.bos_ops import box_cxcywh_to_xyxy
from .utils.common import get_device
from .config import OUTPUTS_DIR


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

        for score, label, box in zip(scores, labels, boxes):
            x_min, y_min, x_max, y_max = box.tolist()
            category_name = VOC_CLASSES[int(label)]
            text = f"{category_name} {score.item():.2f}"

            drawer.rectangle(
                (x_min, y_min, x_max, y_max),
                outline="red",
                width=3,
            )
            drawer.text(
                (x_min, max(0, y_min - 12)),
                text,
                fill="red",
            )

        output_path = OUTPUTS_DIR / image_path.name
        result_image.save(output_path)

        print(f"{image_path}: 检测到 {len(boxes)} 个目标")
        print("结果保存到:", output_path)


def main() -> None:
    # TODO: 在这里手动填写要预测的图片路径
    image_dir = Path("/Users/frank/code/ai/yolo_data/VOC2007/JPEGImages/000590.jpg")
    image_paths: list[Path] = [image_dir]
    predict(image_paths)


if __name__ == "__main__":
    main()
