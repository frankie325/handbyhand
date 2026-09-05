"""在 RAOD 的 20% 留出集上测试并导出预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import BATCH_SIZE, MODELS_DIR, NUM_WORKERS, OUTPUTS_DIR
from .datasets import build_dataloader
from .engine import predict
from .model.build import build_model
from .utils.common import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Deformable DETR on RAOD")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "best.pth")
    parser.add_argument(
        "--output", type=Path, default=OUTPUTS_DIR / "test_predictions.json"
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("score-threshold 必须位于 [0, 1]")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size 必须大于 0，num-workers 不能小于 0")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max-batches 必须大于 0")

    device = get_device(args.device)
    dataloader = build_dataloader(
        "test",
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
    metrics, all_results = predict(
        model, dataloader, device, max_batches=args.max_batches
    )
    exported_results = [
        result for result in all_results if result["score"] >= args.score_threshold
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": "RAOD held-out 20%",
        "score_threshold_for_export": args.score_threshold,
        "metrics": metrics,
        "predictions": exported_results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("测试划分: RAOD 固定的 20% 留出集（数据集无独立 test 标注）")
    print(f"测试图片: {metrics['num_images']}")
    print(f"mAP@0.5 ({metrics['ap_method']}): {metrics['map']:.4f}")
    print(f"导出预测数 (score >= {args.score_threshold}): {len(exported_results)}")
    print(f"预测文件: {args.output}")


if __name__ == "__main__":
    main()
