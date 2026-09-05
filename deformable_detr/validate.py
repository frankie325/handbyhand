"""计算 RAOD 留出验证集的损失和 AP@0.5。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .config import BATCH_SIZE, MODELS_DIR, NUM_WORKERS
from .datasets import build_dataloader
from .engine import validate
from .loss import build_criterion
from .model.build import build_model
from .utils.common import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Deformable DETR on RAOD")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "best.pth")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size 必须大于 0，num-workers 不能小于 0")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max-batches 必须大于 0")

    device = get_device(args.device)
    dataloader = build_dataloader(
        "val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        debug=args.debug,
    )
    model = build_model(
        train=False,
        device=device,
        checkpoint_path=args.checkpoint,
        pretrained_backbone=False,
    )
    criterion = build_criterion().to(device)
    losses, metrics, _ = validate(
        model, criterion, dataloader, device, max_batches=args.max_batches
    )

    print(f"设备: {device}")
    print(f"验证图片: {metrics['num_images']}")
    print("验证损失:")
    for name in ("loss", "loss_ce", "loss_bbox", "loss_giou"):
        print(f"  {name}: {losses[name]:.4f}")
    print(f"AP 算法: {metrics['ap_method']}")
    for class_name, average_precision in metrics["per_class_ap"].items():
        value = (
            "N/A"
            if math.isnan(average_precision)
            else f"{average_precision:.4f}"
        )
        print(f"  {class_name} AP@0.5: {value}")
    print(f"mAP@0.5: {metrics['map']:.4f}")


if __name__ == "__main__":
    main()
