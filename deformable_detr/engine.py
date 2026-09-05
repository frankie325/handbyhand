"""训练、验证和纯推理循环。"""

from __future__ import annotations

import math
from itertools import islice

import torch
from tqdm import tqdm

from .metrics import evaluate_raod, outputs_to_results


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def _average(values: dict[str, float], batch_count: int) -> dict[str, float]:
    if batch_count == 0:
        raise ValueError("DataLoader 没有产生任何 batch")
    return {name: value / batch_count for name, value in values.items()}


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    device: torch.device,
    epoch: int,
    clip_max_norm: float,
    writer=None,
    global_step: int = 0,
    max_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    model.train()
    criterion.train()
    sums: dict[str, float] = {}
    batch_count = 0
    total = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    batches = islice(dataloader, max_batches) if max_batches else dataloader

    for images, padding_mask, targets in tqdm(
        batches, total=total, desc=f"Train {epoch}"
    ):
        images = images.to(device)
        padding_mask = padding_mask.to(device)
        targets = move_targets_to_device(targets, device)

        outputs = model(images, padding_mask)
        loss_dict = criterion(outputs, targets)
        total_loss = criterion.total_loss(loss_dict)
        loss_value = float(total_loss.detach().item())
        if not math.isfinite(loss_value):
            detached = {
                name: float(value.detach().item())
                for name, value in loss_dict.items()
            }
            raise FloatingPointError(f"损失出现 NaN/Inf: total={loss_value}, {detached}")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_max_norm
        )
        optimizer.step()

        batch_count += 1
        global_step += 1
        sums["loss"] = sums.get("loss", 0.0) + loss_value
        for name, value in loss_dict.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().item())

        if writer is not None:
            writer.add_scalar("train/loss", loss_value, global_step)
            for name in ("loss_ce", "loss_bbox", "loss_giou"):
                writer.add_scalar(f"train/{name}", loss_dict[name].item(), global_step)
            writer.add_scalar(
                "train/gradient_norm", float(grad_norm), global_step
            )

    return _average(sums, batch_count), global_step


@torch.inference_mode()
def validate(
    model,
    criterion,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict, list[dict]]:
    model.eval()
    criterion.eval()
    sums: dict[str, float] = {}
    results: list[dict] = []
    image_ids: list[str] = []
    batch_count = 0
    total = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    batches = islice(dataloader, max_batches) if max_batches else dataloader

    for images, padding_mask, targets in tqdm(batches, total=total, desc="Validate"):
        images = images.to(device)
        padding_mask = padding_mask.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = model(images, padding_mask)
        loss_dict = criterion(outputs, targets)
        total_loss = criterion.total_loss(loss_dict)

        batch_count += 1
        sums["loss"] = sums.get("loss", 0.0) + float(total_loss.item())
        for name, value in loss_dict.items():
            sums[name] = sums.get(name, 0.0) + float(value.item())
        results.extend(outputs_to_results(outputs, targets))
        image_ids.extend(str(target["image_id"]) for target in targets)

    metrics = evaluate_raod(results, dataloader.dataset, image_ids=image_ids)
    return _average(sums, batch_count), metrics, results


@torch.inference_mode()
def predict(
    model,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict, list[dict]]:
    model.eval()
    results: list[dict] = []
    image_ids: list[str] = []
    batch_count = 0
    total = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    batches = islice(dataloader, max_batches) if max_batches else dataloader

    for images, padding_mask, targets in tqdm(batches, total=total, desc="Test"):
        images = images.to(device)
        padding_mask = padding_mask.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = model(images, padding_mask)
        results.extend(outputs_to_results(outputs, targets))
        image_ids.extend(str(target["image_id"]) for target in targets)
        batch_count += 1

    if batch_count == 0:
        raise ValueError("DataLoader 没有产生任何 batch")
    metrics = evaluate_raod(results, dataloader.dataset, image_ids=image_ids)
    return metrics, results
