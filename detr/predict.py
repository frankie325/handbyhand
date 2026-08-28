from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from .model.build import build_model
from detr.datasets.transforms import make_coco_transforms
from detr.utils.bos_ops import box_cxcywh_to_xyxy
from .utils.common import get_device
from .config import ROOT_DIR, OUTPUTS_DIR


@torch.inference_mode()
def predict(image_paths: list[Path]) -> None:
    threshold = 0.5
    annotations = ROOT_DIR / "datasets/coco/annotations/instances_val2017.json"

    device = get_device()
    model = build_model(False, device)
    model.eval()

    coco = COCO(str(annotations))
    categories = {
        category_id: category["name"] for category_id, category in coco.cats.items()
    }

    """
    COCO 的 80 个类别 ID 并不连续
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90])
    """
    category_ids = torch.as_tensor(sorted(categories), device=device)

    # 预测使用和验证集相同的确定性预处理。
    transform = make_coco_transforms("val")

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

        """
        pred_logits，最后一维类别数，92
        [0, 1, 2, ..., 92]
        """
        probabilities = outputs["pred_logits"][0].softmax(
            dim=-1
        )  # probabilities.shape = [100, 92]

        # index_select 先把 92 列压缩成 80 列（只留真实类别）
        scores, positions = probabilities.index_select(
            dim=-1,
            index=category_ids,
        ).max(dim=-1)
        # scores = [100] ，positions = [100] score是最高置信度分数，position是最高置信度分数对应的类别ID在category_ids中的位置
        labels = category_ids[
            positions
        ]  # [100] 用 positions 里的每个整数，逐一去 category_ids 中按下标取元素，拿到coco类别id

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
            category_name = categories[int(label)]
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
    # TODO: 在这里手动填写要预测的图片路S径
    image_paths: list[Path] = []
    predict(image_paths)


if __name__ == "__main__":
    main()
