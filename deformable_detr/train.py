"""在 RAOD-benchmark 上训练 Deformable DETR。

运行示例：
    python -m deformable_detr.train
    python -m deformable_detr.train --debug --epochs 1 --max-train-batches 2
    python -m deformable_detr.train --resume
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from .config import (
    BACKBONE_LR,
    BATCH_SIZE,
    CLIP_MAX_NORM,
    EPOCHS,
    LOG_DIR,
    LR,
    LR_DROP,
    MODELS_DIR,
    NUM_WORKERS,
    SEED,
    VALIDATE_EVERY,
    WEIGHT_DECAY,
)
from .datasets import build_dataloader
from .engine import train_one_epoch, validate
from .loss import build_criterion
from .model.build import build_model
from .utils.common import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Deformable DETR on RAOD")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lr-drop", type=int, default=LR_DROP)
    parser.add_argument("--clip-max-norm", type=float, default=CLIP_MAX_NORM)
    parser.add_argument("--validate-every", type=int, default=VALIDATE_EVERY)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 mps")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=str(MODELS_DIR / "last.pth"),
        default=None,
        help="从检查点恢复；不跟路径时使用 weights/last.pth",
    )
    parser.add_argument("--debug", action="store_true", help="使用较小输入尺寸")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def _check_positive(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} 必须大于 0")


def _build_optimizer(model, args: argparse.Namespace) -> torch.optim.Optimizer:
    main_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "backbone" not in name and parameter.requires_grad
    ]
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "backbone" in name and parameter.requires_grad
    ]
    parameter_groups = [{"params": main_parameters, "lr": args.lr, "name": "main"}]
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": args.backbone_lr,
                "name": "backbone",
            }
        )
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def _load_resume_state(
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "optimizer" not in checkpoint or "scheduler" not in checkpoint:
        raise ValueError(f"{checkpoint_path} 只有模型权重，不能用于 --resume")
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_map = float(checkpoint.get("best_map", -math.inf))
    return start_epoch, global_step, best_map


def main() -> None:
    args = parse_args()
    for name in ("epochs", "batch_size", "lr_drop", "validate_every"):
        _check_positive(name, getattr(args, name))
    if args.num_workers < 0:
        raise ValueError("num_workers 不能小于 0")
    _check_positive("max_train_batches", args.max_train_batches)
    _check_positive("max_val_batches", args.max_val_batches)

    seed_everything(args.seed)
    device = get_device(args.device)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备: {device}")

    train_loader = build_dataloader(
        "train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        debug=args.debug,
    )
    val_loader = build_dataloader(
        "val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        debug=args.debug,
    )
    print(f"RAOD: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    resume_path = Path(args.resume) if args.resume else None
    model = build_model(
        train=True,
        device=device,
        checkpoint_path=resume_path,
        pretrained_backbone=not args.no_pretrained_backbone and resume_path is None,
        train_backbone=not args.freeze_backbone,
    )
    criterion = build_criterion().to(device)
    optimizer = _build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_drop, gamma=0.1
    )

    start_epoch = 1
    global_step = 0
    best_map = -math.inf
    if resume_path is not None:
        start_epoch, global_step, best_map = _load_resume_state(
            resume_path, optimizer, scheduler, device
        )
        print(f"从 {resume_path} 的第 {start_epoch} 轮继续训练")

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=LOG_DIR / run_name)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_losses, global_step = train_one_epoch(
                model=model,
                criterion=criterion,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                clip_max_norm=args.clip_max_norm,
                writer=writer,
                global_step=global_step,
                max_batches=args.max_train_batches,
            )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"train_loss={train_losses['loss']:.4f}"
            )

            metrics = None
            if epoch % args.validate_every == 0 or epoch == args.epochs:
                val_losses, metrics, _ = validate(
                    model=model,
                    criterion=criterion,
                    dataloader=val_loader,
                    device=device,
                    max_batches=args.max_val_batches,
                )
                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"val_loss={val_losses['loss']:.4f}, "
                    f"mAP@0.5={metrics['map']:.4f}"
                )
                writer.add_scalar("val/loss", val_losses["loss"], epoch)
                writer.add_scalar("val/map_50", metrics["map"], epoch)

            scheduler.step()
            for group in optimizer.param_groups:
                writer.add_scalar(
                    f"learning_rate/{group.get('name', 'group')}",
                    group["lr"],
                    epoch,
                )

            is_best = metrics is not None and metrics["map"] > best_map
            if is_best:
                best_map = float(metrics["map"])
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "best_map": best_map,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(checkpoint, MODELS_DIR / "last.pth")
            if is_best:
                torch.save(checkpoint, MODELS_DIR / "best.pth")
                print(f"保存最佳权重: {MODELS_DIR / 'best.pth'}")
            writer.flush()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
