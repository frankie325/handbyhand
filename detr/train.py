import torch
from torch.utils.tensorboard import SummaryWriter
from .config import (
    MODELS_DIR,
    BATCH_SIZE,
    D_MODEL,
    LOG_DIR,
    NUM_CLASSES,
    NUM_QUERIES,
    N_LAYER,
    D_FF,
    DROPOUT,
    N_HEAD,
    BACKBONE_LR,
    LR,
    EPOCHS,
)
import time
from .model.detr import Detr
from .datasets.build import build_dataloader
from tqdm import tqdm
from .loss.criterion import SetCriterion
from detr.loss.matcher import HungarianMatcher
import math
from .utils.common import get_device

global_step = 0


def train_one_epoch(model, dataloader, optimizer, criterion, device, writer):
    global global_step
    model.train()
    total_loss: float = 0.0
    for images, padding_mask, targets in tqdm(dataloader, desc="Training"):
        images = images.to(device)  # [batch_size, 3, height, width]
        padding_mask = padding_mask.to(device)  # [batch_size, height, width]

        # targets: list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]
        targets = [
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        # print(images)
        # print(padding_mask)
        # targets = targets.to(device) #

        outputs = model(images, padding_mask)
        loss_dict = criterion(outputs, targets)

        # 计算总损失
        losses = torch.stack(list(loss_dict.values())).sum()

        optimizer.zero_grad()
        losses.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.1,
        )

        optimizer.step()

        total_loss += losses.item()

        writer.add_scalar(
            "Loss/total",
            losses.item(),
            global_step,
        )

        writer.add_scalar(
            "Loss/class",
            loss_dict["loss_ce"].detach().item(),
            global_step,
        )

        writer.add_scalar(
            "Loss/bbox_l1",
            loss_dict["loss_bbox"].detach().item(),
            global_step,
        )

        writer.add_scalar(
            "Loss/giou",
            loss_dict["loss_giou"].detach().item(),
            global_step,
        )

        writer.add_scalar(
            "Gradient/global_norm",
            grad_norm.item(),
            global_step,
        )

        global_step += 1
    return total_loss / len(dataloader)


def train():
    device = get_device()

    print("设备:", device)

    # tensorboard 记录，终端切换到当前目录下，输入tensorboard --logdir=logs
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    dataloader = build_dataloader("train", batch_size=BATCH_SIZE, shuffle=True)

    model = Detr(NUM_CLASSES, NUM_QUERIES, D_MODEL, N_LAYER, D_FF, N_HEAD, DROPOUT).to(
        device
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "backbone" not in name and parameter.requires_grad
                ],
                "lr": LR,
            },
            # 主干网络使用1e-5的学习率
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "backbone" in name and parameter.requires_grad
                ],
                "lr": BACKBONE_LR,
            },
        ],
        weight_decay=1e-4,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=math.floor(EPOCHS * 0.7),
        gamma=0.1,
    )

    # 计算代价矩阵，进行匈牙利匹配算法
    matcher = HungarianMatcher()

    criterion = SetCriterion(
        NUM_CLASSES,
        matcher,
        eos_coef=0.1,
    ).to(device)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, writer
        )
        lr_scheduler.step()
        # tensorboard 记录
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # 记录学习率
        writer.add_scalar(
            "LearningRate/main",
            optimizer.param_groups[0]["lr"],
            global_step,
        )

        # 记录主干网习率
        writer.add_scalar(
            "LearningRate/backbone",
            optimizer.param_groups[1]["lr"],
            global_step,
        )
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch, EPOCHS, avg_loss))

        writer.flush()
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODELS_DIR / "best.pth")
            print("模型已保存到:", MODELS_DIR / "best.pth")

    writer.close()


if __name__ == "__main__":
    train()
