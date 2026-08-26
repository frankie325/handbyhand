from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from detr.config import MODELS_DIR
from detr.datasets.build import DEFAULT_DATA_ROOT
from detr.datasets.transforms import make_coco_transforms
from detr.utils.bos_ops import box_cxcywh_to_xyxy
from detr.validate import load_model, select_device


@torch.inference_mode()
def predict(
    image_path: Path,
    checkpoint: Path,
    output_path: Path,
    annotations: Path,
    threshold: float,
):
    device = select_device()
    model = load_model(checkpoint, device)

    coco = COCO(str(annotations))
    categories = {
        category_id: category["name"] for category_id, category in coco.cats.items()
    }

    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size

    # 预测使用和验证集相同的确定性预处理。
    transform = make_coco_transforms("val")
    image_tensor, _ = transform(original_image, None)

    images = image_tensor.unsqueeze(0).to(device)

    # 单张图片没有padding，因此mask全部为False。
    padding_mask = torch.zeros(
        (1, image_tensor.shape[1], image_tensor.shape[2]),
        dtype=torch.bool,
        device=device,
    )

    outputs = model(images, padding_mask)

    probabilities = outputs["pred_logits"][0].softmax(
        dim=-1
    )  # probabilities.shape = [100, 92]

    category_ids = torch.as_tensor(
        sorted(categories),
        device=device,
    )  # COCO 类别 ID 分布在 1~90 之间，中间有跳号，如缺 12、26、29 等），tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, ... 88, 89, 90])

    scores, positions = probabilities.index_select(
        dim=-1,
        index=category_ids,
    ).max(dim=-1)
    # scores = [100] ，positions = [100] score是最高置信度分数，position是最高置信度分数对应的类别ID在category_ids中的位置
    labels = category_ids[positions]  # [100] 拿到coco类别id

    # boxes是相对图片归一化的坐标
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][0])
    # 这里进行还愿，画在真实图片上，必须乘以图片的宽高
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(output_path)

    print(f"检测到 {len(boxes)} 个目标")
    print("结果保存到:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODELS_DIR / "best.pth",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_DATA_ROOT / "annotations" / "instances_val2014.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prediction.jpg"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predict(
        image_path=args.image,
        checkpoint=args.checkpoint,
        output_path=args.output,
        annotations=args.annotations,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
