import os
import torch
import torch.nn as nn
import sys
import random
import colorsys
import numpy as np
from .utils import get_device
from argparse import ArgumentParser, Namespace
from pathlib import Path
from dino.vision_transformer import build_model
from PIL import Image
from torchvision import transforms as pth_transforms
import torchvision
import skimage.io
from skimage.measure import find_contours
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = (
            image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        )
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(
    image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5
):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis("off")
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect="auto")
    fig.savefig(fname)
    plt.close(fig)
    print(f"{fname} saved.")
    return


if __name__ == "__main__":

    file_path = Path(__file__)
    root = file_path.parent
    parser = ArgumentParser("Visualize DINO self-attention")
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=root / "weights" / "vits16" / "checkpoint.pth",
    )
    parser.add_argument(
        "--image-path", type=Path, default=root / "images" / "fish.png"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument(
        "--checkpoint-key", choices=("teacher", "student"), default="teacher"
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    cli_args = parser.parse_args()

    args = Namespace(
        pretrained_weights=str(cli_args.pretrained_weights),
        output_dir=str(cli_args.output_dir),
        model=None,  # 从 checkpoint 的训练参数自动读取，避免模型结构不匹配
        img_size=224,
        num_classes=0,
        checkpoint_key=cli_args.checkpoint_key,
        image_path=str(cli_args.image_path),
        threshold=cli_args.threshold,
    )

    if not os.path.isfile(args.pretrained_weights):
        print("请设置正确的pretrained_weights")
        sys.exit(1)

    checkpoint = torch.load(
        args.pretrained_weights, map_location="cpu", weights_only=False
    )
    checkpoint_args = checkpoint.get("args")
    if checkpoint_args is None or not hasattr(checkpoint_args, "model"):
        print("checkpoint 中没有 args.model，无法自动判断模型结构")
        sys.exit(1)
    args.model = checkpoint_args.model
    args.img_size = getattr(checkpoint_args, "img_size", args.img_size)

    device = get_device()
    model = build_model(args)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    if args.checkpoint_key is not None and args.checkpoint_key in checkpoint:
        state_dict = checkpoint[args.checkpoint_key]
        # print(
        #     state_dict["teacher"].keys()
        # )  # ['backbone.cls_token', 'backbone.pos_embed', 'backbone.patch_embed.proj.weight', 'backbone.patch_embed.proj.bias', 'backbone.blocks.0.norm1.weight', 'backbone.blocks.0.norm1.bias', 'backbone.blocks.0.attn.qkv.weight', 'backbone.blocks.0.attn.qkv.bias', 'backbone.blocks.0.attn.proj.weight', 'backbone.blocks.0.attn.proj.bias', 'backbone.blocks.0.norm2.weight', 'backbone.blocks.0.norm2.bias', 'backbone.blocks.0.mlp.fc1.weight', 'backbone.blocks.0.mlp.fc1.bias', 'backbone.blocks.0.mlp.fc2.weight',...
        # print(state_dict.keys())
        print(f"Take key {args.checkpoint_key} in provided checkpoint dict")

        # 移除state_dict中 `module.` 前缀(来自 DDP 的 module.)
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        # 只加载 backbone，并严格检查所有参数，避免 strict=False 静默忽略结构错误
        state_dict = {
            k.removeprefix("backbone."): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        msg = model.load_state_dict(state_dict, strict=True)
        print("权重位于 {} 加载信息为: {}".format(args.pretrained_weights, msg))
    else:
        print(f"checkpoint 中不存在字段: {args.checkpoint_key}")
        sys.exit(1)

    with open(args.image_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

    # 归一化成tensor
    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(args.img_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = transform(img)  # [3, 224, 224]

    patch_size = model.patch_embed.patch_size
    # print(f"Patch size: {patch_size}")

    # 把图片裁剪成「能被 patch 尺寸整除」的大小，保证 ViT 切出来的 patch 网格是规整的整型二维图
    w, h = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )  # (224, 224)
    img = img[:, :w, :h].unsqueeze(0)  # [1, 3, 224, 224]

    # print(f"Image size after padding: {w}x{h}")

    # 模型输出的特征图的尺寸
    w_featmap = img.shape[-2] // patch_size  # 14
    h_featmap = img.shape[-1] // patch_size  # 14

    attentions = model.get_last_selfattention(
        img.to(device)
    )  # （1, num_heads, seq_len, seq_len)
    # print(attentions.shape) # (1, 12, 197, 197)
    nh = attentions.shape[1]  # number of head

    attentions = attentions[0, :, 0, 1:].reshape(
        nh, -1
    )  # （num_heads, 1, 196） -> (num_heads, 196)
    # print(attentions.shape) # (12, 196)
    # 取出每个 head 上 cls_token 对所有 特征图像素点 的注意力权重
    # attentions[0, :, 0, 1:]
    #            │  │  │  │
    #            │  │  │  └─ 1:  → key 维保留「第 2 个起到最后」，即排除第 0 位(CLS 自己)，只留 patch token
    #            │  │  └──── 0  → query 维取「第 0 个 token」，即 CLS token（看 CLS 怎么关注别人）
    #            │  └────── :  → 保留所有 head（维 1）
    #            └──────── 0  → 取 batch 里第 1 张（也是唯一一张）图
    if args.threshold is not None:
        """
        注意力:    0.1   0.5   0.2   0.9   0.3
        排序后
            val = [0.1, 0.2, 0.3, 0.5, 0.9]     # 排序后的值
            idx = [  0,   2,   4,   1,   3]     # 每个值在「原数组」里的位置
        归一化
            val = [0.05, 0.1, 0.15, 0.25, 0.45]
        累加质量
            th_attn = [0.05, 0.15, 0.3, 0.55, 1]
        取质量前40%（1-0.6）的注意力得分，将其置为True，得到掩码。相当于进行排名，选择前40%的部分
            th_attn = [False, False, False, True, True]
        idx2为idx的逆置换，可以将排序后的th_attn进行还原
            idx2= [  0,   3,   1,   4,   2]
            val[idx2] = [0.1 ,0.5 ,0.2 ,0.9 ,0.3]
            th_attn[idx2] = [False, True, False, True, False] 得到原图像的二值掩码

        True=被 CLS 高度关注的 patch（即模型认为的「物体区域」
        """
        #  每个 head 的注意力从小到大排序
        val, idx = torch.sort(attentions)  # (num_heads, 196)
        #  按head行进行归一化，使每个 head 的注意力总质量=1（变成概率分布）
        val /= torch.sum(val, dim=1, keepdim=True)
        # 从小到大累加质量。例如 [0.01, 0.03, …]→[0.01, 0.04, …] 逐渐逼近 1
        cumval = torch.cumsum(val, dim=1)
        # 取质量前40%（1-0.6）的注意力得分，这些位置被标记为True
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(
            nh, w_featmap, h_featmap
        ).float()  # (num_heads, 14, 14)
        # 用最近邻把每张小掩码放大 patch_size 倍回原图分辨率
        th_attn = (
            nn.functional.interpolate(
                th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )  # (num_heads, 224, 224)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)  # (num_heads, 14, 14)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )  # (num_heads, 224, 224)

    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        os.path.join(args.output_dir, "img.png"),
    )
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format="png")
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(
                image,
                th_attn[j],
                fname=os.path.join(
                    args.output_dir,
                    "mask_th" + str(args.threshold) + "_head" + str(j) + ".png",
                ),
                blur=False,
            )
