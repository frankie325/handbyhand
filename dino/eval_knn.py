"""使用冻结的 DINO backbone 在 Imagenette 验证集上做加权 k-NN 评估。"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from .vision_transformer import build_model


def get_args_parser():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        "DINO weighted k-NN evaluation",
        description=(
            "用 train split 建立冻结特征库，在 val split 上报告 Top-1/Top-5。"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "weights" / "vits16" / "checkpoint.pth",
        help="DINO checkpoint 路径。",
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("teacher", "student"),
        default="teacher",
        help="评估 teacher 或 student；DINO 通常评估 teacher。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="默认从 checkpoint 的 args.model 自动读取；仅在 checkpoint 未保存参数时指定。",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root.parent / "datasets" / "imagenette2-160",
        help="同时包含 train/ 和 val/ 的 ImageFolder 数据集根目录。",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=20, help="近邻数量。")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="近邻余弦相似度的温度系数。",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="评估设备，例如 cuda、cuda:1；正式评估建议使用 CUDA。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="结果 JSON 路径；默认写入 checkpoint 同目录的 knn_eval.json。",
    )
    return parser


def _checkpoint_backbone_state(checkpoint, checkpoint_key):
    if checkpoint_key not in checkpoint:
        available = ", ".join(sorted(checkpoint))
        raise KeyError(f"checkpoint 中没有 {checkpoint_key!r}，可用字段为: {available}")

    backbone_state = {}
    for name, value in checkpoint[checkpoint_key].items():
        name = name.removeprefix("module.")
        if name.startswith("backbone."):
            backbone_state[name.removeprefix("backbone.")] = value

    if not backbone_state:
        raise ValueError(f"checkpoint[{checkpoint_key!r}] 中没有找到 backbone.* 参数")
    return backbone_state


def load_backbone(checkpoint_path, checkpoint_key="teacher", model_name=None):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    saved_args = checkpoint.get("args")
    if model_name is None and saved_args is not None:
        model_name = getattr(saved_args, "model", None)
    if model_name is None:
        raise ValueError("无法从 checkpoint 推断模型，请通过 --model 显式指定")

    img_size = getattr(saved_args, "img_size", 224) if saved_args is not None else 224
    model = build_model(Namespace(model=model_name, img_size=img_size, num_classes=0))
    backbone_state = _checkpoint_backbone_state(checkpoint, checkpoint_key)
    model.load_state_dict(backbone_state, strict=True)
    model.eval()
    return model, checkpoint, model_name, img_size


@torch.inference_mode()
def extract_features(model, data_loader, device, description):
    features = []
    labels = []
    for images, targets in tqdm(data_loader, desc=description):
        images = images.to(device, non_blocking=True)
        batch_features = F.normalize(model(images), dim=-1)
        features.append(batch_features.cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def weighted_knn(
    train_features,
    train_labels,
    val_features,
    val_labels,
    num_classes,
    k=20,
    temperature=0.07,
    query_batch_size=1024,
    device=torch.device("cuda"),
):
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    k = min(k, len(train_features))
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    top1_correct = 0
    top5_correct = 0
    top5_width = min(5, num_classes)

    for start in tqdm(
        range(0, len(val_features), query_batch_size), desc="weighted k-NN"
    ):
        end = start + query_batch_size
        queries = val_features[start:end].to(device)
        targets = val_labels[start:end].to(device)

        similarities = queries @ train_features.T
        top_similarities, top_indices = similarities.topk(k, dim=1)
        neighbor_labels = train_labels[top_indices]
        neighbor_weights = (top_similarities / temperature).exp()

        class_scores = torch.zeros(
            len(queries), num_classes, device=device, dtype=neighbor_weights.dtype
        )
        class_scores.scatter_add_(1, neighbor_labels, neighbor_weights)
        predictions = class_scores.topk(top5_width, dim=1).indices

        matches = predictions.eq(targets.unsqueeze(1))
        top1_correct += matches[:, :1].sum().item()
        top5_correct += matches.any(dim=1).sum().item()

    total = len(val_features)
    return {
        "top1": 100.0 * top1_correct / total,
        "top5": 100.0 * top5_correct / total,
        "k": k,
        "temperature": temperature,
    }


def main(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {args.checkpoint}")
    train_dir = args.data_path / "train"
    val_dir = args.data_path / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"数据目录必须同时包含 train/ 和 val/: {args.data_path}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请在 GPU 环境运行正式 k-NN evaluation")

    model, checkpoint, model_name, img_size = load_backbone(
        args.checkpoint, args.checkpoint_key, args.model
    )
    model.to(device)

    resize_size = round(img_size * 256 / 224)
    transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train 和 val 的类别映射不一致")

    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, **loader_options)
    val_loader = DataLoader(val_dataset, **loader_options)

    train_features, train_labels = extract_features(
        model, train_loader, device, "train feature bank"
    )
    val_features, val_labels = extract_features(
        model, val_loader, device, "val features"
    )
    metrics = weighted_knn(
        train_features,
        train_labels,
        val_features,
        val_labels,
        num_classes=len(train_dataset.classes),
        k=args.k,
        temperature=args.temperature,
        device=device,
    )
    metrics.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_key": args.checkpoint_key,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "model": model_name,
            "feature_dim": train_features.shape[1],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "classes": len(train_dataset.classes),
        }
    )

    output_path = args.output or args.checkpoint.parent / "knn_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"k-NN evaluation: Top-1 {metrics['top1']:.2f}% | "
        f"Top-5 {metrics['top5']:.2f}% | k={metrics['k']} | "
        f"T={metrics['temperature']}"
    )
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
