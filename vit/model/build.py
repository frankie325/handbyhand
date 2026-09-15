from collections.abc import Callable
from typing import Any

from .vision_transformer import VisionTransformer


def vit_b_16(num_classes: int = 1000, **kwargs: Any) -> VisionTransformer:
    """创建 ViT-Base/16。"""
    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


def vit_b_32(num_classes: int = 1000, **kwargs: Any) -> VisionTransformer:
    """创建 ViT-Base/32。"""
    return VisionTransformer(
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


def vit_l_16(num_classes: int = 1000, **kwargs: Any) -> VisionTransformer:
    """创建 ViT-Large/16。"""
    return VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


def vit_l_32(num_classes: int = 1000, **kwargs: Any) -> VisionTransformer:
    """创建 ViT-Large/32。"""
    return VisionTransformer(
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


def vit_h_14(num_classes: int = 1000, **kwargs: Any) -> VisionTransformer:
    """创建 ViT-Huge/14。"""
    return VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


MODEL_BUILDERS: dict[str, Callable[..., VisionTransformer]] = {
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
    "vit_h_14": vit_h_14,
}


def build_model(args: Any) -> VisionTransformer:
    """根据命令行参数创建 ViT，默认使用 ViT-B/16。"""
    model_name = getattr(args, "model", "vit_b_16")
    model_name = model_name.lower().replace("-", "_").replace("/", "_")

    if model_name not in MODEL_BUILDERS:
        available_models = ", ".join(MODEL_BUILDERS)
        raise ValueError(
            f"不支持的模型: {model_name}，可选模型为: {available_models}"
        )

    model_kwargs = {
        "img_size": args.img_size,
        "num_classes": args.num_classes,
    }
    for name in ("in_chans", "drop", "attn_drop"):
        if hasattr(args, name):
            model_kwargs[name] = getattr(args, name)

    return MODEL_BUILDERS[model_name](**model_kwargs)
