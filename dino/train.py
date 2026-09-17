import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from argparse import Namespace
from .vision_transformer import build_model, DINOHead
from .utils import (
    init_distributed_mode,
    fix_random_seeds,
    has_batchnorms,
    get_world_size,
    get_params_group,
    get_device,
    cosine_scheduler,
    clip_gradients,
    cancel_gradients_last_layer,
    save_on_master,
    restart_from_checkpoint,
    MultiCropWrapper,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import subprocess
import sys
from .augmention import AugmentationDino
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
import datetime
import os
import math


def _denorm_to_pil(tensor, mean, std):
    """把归一化张量(≈[-1,1])反归一化并转回 PIL Image。"""
    t = tensor.cpu().clone() * std + mean
    t = t.clamp(0, 1)
    return transforms.ToPILImage()(t)


def _pil_grid(images, nrow, padding=4, bg=(255, 255, 255)):
    """把一组 PIL 图拼成 nrow 列的网格，返回一张大图。"""
    w, h = images[0].size
    n = len(images)
    ncol = min(nrow, n)
    nrow_real = (n + ncol - 1) // ncol
    grid_w = ncol * w + (ncol + 1) * padding
    grid_h = nrow_real * h + (nrow_real + 1) * padding
    canvas = Image.new("RGB", (grid_w, grid_h), bg)
    for idx, im in enumerate(images):
        r, c = divmod(idx, ncol)
        x = padding + c * (w + padding)
        y = padding + r * (h + padding)
        canvas.paste(im, (x, y))
    return canvas


def visualize_augmentation(img, transform, out_path, nrow=5, open_image=True):
    """对比 DINO 数据增强前后：左边原图，右边增强得到的多视图网格。

    Args:
        img: 原始 PIL.Image（来自 dataset[i]）
        transform: AugmentationDino 实例
        out_path: 对比图保存路径
        nrow: crop 网格每行数量（默认 5，10 个 crop 排成 2 行）
        open_image: macOS 下用 Preview 打开
    Returns:
        保存的对比图路径
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 原图（统一缩到 224，方便和 crop 对齐）
    orig = img.convert("RGB").resize((224, 224))

    # 增强后的多个 crop：反归一化转回 PIL
    crops = [_denorm_to_pil(t, mean, std) for t in transform(img)]
    grid = _pil_grid(crops, nrow)

    # 并排拼接：左原图 + 右 crop 网格
    pad = 16
    total_w = orig.width + pad + grid.width
    total_h = max(orig.height, grid.height)
    comp = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    comp.paste(orig, (0, 0))
    comp.paste(grid, (orig.width + pad, 0))

    out_path = Path(out_path)
    comp.save(out_path)
    info = f"原图 {orig.size} | {len(crops)} 个 crop，网格 {grid.size}"
    print(f"增强对比图已保存: {out_path}  ({info})")

    if open_image and sys.platform == "darwin":
        subprocess.run(["open", str(out_path)])
    return out_path


def train_dino():
    device = get_device()

    file_path = Path(__file__)
    root = file_path.parent
    # build_model 通过 args.model / args.img_size
    # 因此这里用 Namespace（带属性的对象）而非 dict
    args = Namespace(
        img_size=224,
        num_classes=0,
        model="vit_s_16",
        num_workers=4,
        out_dim=8192,
        use_bn_in_head=False,
        norm_last_layer=True,
        drop_path_rate=0.1,
        global_crops_scale=(0.4, 1.0),  # global_crop的裁剪比例
        local_crops_scale=(0.05, 0.4),  # local_crop的裁剪比例
        local_crops_number=8,  # local_crop的数量
        epochs=300,
        lr=0.0005,
        min_lr=1e-6,
        warmup_epochs=10,  # 预热轮数
        weight_decay=0.04,
        weight_decay_end=0.4,
        # 小 batch 下让 teacher 更新得更平稳，避免过快追随 student 的噪声
        # 官方建议小 batch 使用更高的 EMA momentum；可以单独测试 0.999 或 0.9995
        momentum_teacher=0.9995,
        use_fp16=True,
        warmup_teacher_temp=0.04,
        # 300 epoch 配置：先把 teacher 温度从 0.04 预热到 0.07，避免目标过早变成近似 one-hot
        teacher_temp=0.07,
        warmup_teacher_temp_epochs=30,
        clip_grad=3.0,  # 梯度裁剪限制范围
        freeze_last_layer=1,
        dist_url="env://",
        seed=0,
        batch_size_per_gpu=64,
        # 保留旧的 ViT-B checkpoint；新结构必须从头训练并写入独立目录
        output_dir=f"{root}/weights/vits16",
        saveckp_freq=20,  # 隔多少轮保存一次权重
    )

    init_distributed_mode(args)
    fix_random_seeds(
        args.seed
    )  # 每个进程都会执行一次，各自把本进程的随机源设为同一个 seed

    # ============ 准备数据 ... ============
    # path = Path(__file__)
    dataset_dir = (
        Path(__file__).parent.parent / "datasets" / "imagenette2-160" / "train"
    )

    transform = AugmentationDino(
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=args.local_crops_number,
    )
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)

    """
    dataset =  list( (views, label) )
    views：transform(img) 的返回值,含 2 + N 个张量，2 个 global crop：shape = [3, 224, 224]
    N 个 local crop：shape = [3, 96, 96]（默认 local_crops_number=8 → 共 10 个）
    label：int（类别索引，0~9，因为 imagenette 有 10 类）
    """
    # print("images:", dataset[0][0])
    # print("label:", dataset[0][1])
    # print(dataset)
    # print(len(dataset[0][0]))

    # dataset[i] 返回 (PIL.Image, label) 元组
    # img, label = dataset[0]
    # print("class:", dataset.classes[label], "| label:", label)
    # print("image size:", img.size)

    # 增强前后对比：原图 vs 多视图 crop 网格
    # visualize_augmentation(
    #     img,
    #     transform,
    #     out_path=Path(__file__).parent / "augment_compare_0.jpg",
    # )

    # ============ 构建student和teacher网络 ... ============
    student = build_model(args)
    # stochastic depth 只用于 student；teacher 与官方实现一致，保持 drop_path=0
    teacher_args = Namespace(**vars(args))
    teacher_args.drop_path_rate = 0.0
    teacher = build_model(teacher_args)
    # 采样器确保各GPU进程读取不重叠的数据切片
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=get_world_size(),  # 按照GPU的数量为间隔进行采样
        shuffle=True,
    )
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    embed_dim = student.embed_dim

    student = MultiCropWrapper(
        student,
        DINOHead(
            embed_dim,
            args.out_dim,
            args.use_bn_in_head,
            args.norm_last_layer,
        ),
    )

    teacher = MultiCropWrapper(
        teacher,
        DINOHead(
            embed_dim,
            args.out_dim,
            args.use_bn_in_head,
        ),
    )

    # 将网络移动至GPU
    student, teacher = student.cuda(), teacher.cuda()

    if has_batchnorms(student):
        # 默认 BN 仅在单卡内计算均值 / 方差，小 batch 训练场景下，单卡BN统计误差大；开启 SyncBatchNorm 可跨 GPU 同步 BN 全局统计量，提升精度：
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = (
            teacher.module
        )  #  teacher 的参数更新是原地 EMA（.data.mul_/.add_），不经过反向传播、也不需要 DDP 的梯度通信。这里直接拿teacher_without_ddp原地修改
    else:
        teacher_without_ddp = teacher

    # !3.使用DistributedDataParallel包装一层
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # 初始时加载和student一样的权重参数
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # !teacher 不参与反向传播（没有梯度），它只作为"目标"被 EMA 更新
    for p in teacher.parameters():
        p.requires_grad = False

    # ============ 准备优化器 ... ============
    params_group = get_params_group(student)
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(params_group)

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ 学习率调度 ============
    lr_schedule = cosine_scheduler(
        # DINO（以及大多数大规模视觉训练）里经典的 Linear Scaling Rule（线性学习率缩放规则），目的是让学习率跟着"实际总 batch size"走，保证不同 GPU 数量下训练动态一致
        # args.lr × (总 batch / 256)
        args.lr * (args.batch_size_per_gpu * get_world_size()) / 256.0,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )

    momentum_schedule = cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )

    # ============ 从检查点恢复训练 ============
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    # 确保输出目录存在，否则 torch.save 会报 Parent directory does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):

        # !4 设置采样器的epoch，保证每个epoch的数据集是不同的
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            device,
            epoch,
            fp16_scaler,
            args,
        )

        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "dino_loss": dino_loss.state_dict(),
        }

        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        print(f"Epoch: {epoch}/{args.epochs}, loss: {train_stats['loss']:.4f}")
    total_time: float = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # print(f"Data loaded: there are {len(dataset)} images.")


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    device,
    epoch,
    fp16_scaler,
    args,
):
    total_loss = 0
    logs = {}
    for it, (images, labels) in enumerate(tqdm(data_loader)):
        # images = [
        #     tensor[B, 3, 224, 224],   # global crop 1
        #     tensor[B, 3, 224, 224],   # global crop 2
        #     tensor[B, 3, 96, 96],     # local crop 1
        #     ... ×8
        # ]
        # labels = tensor[B]   #batch_size个类别索引
        it = len(data_loader) * epoch + it  # 全局迭代次数
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            # 非偏置的参数添加权重衰减
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]

        images = [image.to(device) for image in images]

        with torch.amp.autocast("cuda", enabled=fp16_scaler is not None):
            teacher_output = teacher(
                images[:2]
            )  # 前两个 global crop送给教师网络学习 [batch_size, 65536]
            student_output = student(images)  # [batch_size, 65536]
            loss = dino_loss(student_output, teacher_output, epoch)

        # 检查是否梯度爆炸
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = clip_gradients(student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = clip_gradients(student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # 教师网络的指数滑动平均（EMA）更新
        with torch.no_grad():
            m = momentum_schedule[it]
            # param_k = 教师参数，param_q = 学生参数。
            for param_q, param_k in zip(
                student.module.parameters(), teacher_without_ddp.parameters()
            ):
                # 先原地算 param_k = m * param_k
                # 再加上 (1 - m) * param_q，实现指数滑动平均
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # 阻塞 CPU 线程，等到 GPU 上所有已提交的核函数（kernel）全部执行完，再继续往下走
        torch.cuda.synchronize()
        total_loss += loss.item()

    logs = {
        "loss": total_loss / len(data_loader),
    }
    return logs


"""
① 单维独大（one-hot 坍塌）
指不管输入什么图，softmax 概率都堆到同一个维度上，网络等于「无视图像内容，永远输出同一个尖峰」：
样本A(猫):  [0.95, 0.03, 0.01, 0.01]     ← 第0维占 95%
样本B(狗):  [0.95, 0.03, 0.01, 0.01]     ← 还是第0维占 95%
样本C(车):  [0.95, 0.03, 0.01, 0.01]     ← 完全一样
这就是退化解：所有图表征相同，模型「偷懒」成功，但表征毫无用处。
成因：softmax 温度太低（如 teacher_temp 压到 0.01），logits/T 被放大，分布极度尖锐 → 接近 one-hot

中心化（centering）的作用：
计算批次均值，并不直接使用当前批次的均值，而是维护这些均值的指数移动平均，这个动态更新的平均值被称为“中心”向量，用 c 表示
教师网络的输出向量被送入 softmax 函数之前，需要先减去这个“中心”向量 c，减去这个均值，正是为削平这种单维坍缩
虽然它解决了单一维度主导的问题，但它反而会鼓励第二种坍塌形式：过度均匀

② 过度均匀
指softmax 概率被摊得几乎一样平，没有任何维度有信息量：
样本A(猫):  [0.27, 0.25, 0.25, 0.23]
样本B(狗):  [0.27, 0.25, 0.25, 0.23]
样本C(车):  [0.27, 0.25, 0.25, 0.23]

看似「没坍塌」（不是 one-hot），但所有维度都半斤八两，模型同样没学到区分性特征；学生去匹配这种「近乎均匀」的目标，得到的梯度信号极弱，学不动

温度锐化（Sharpening）：

logits / temp
temp 越小 → 除以小数 → logits 被放大 → 分布越尖
temp 越大 → 除以大数 → logits 被压平 → 分布越平

用较低温度让分布保持「尖而有意义」（学生侧 student_temp=0.1，教师侧 0.04–0.07，都偏低）
0.04–0.07 是实验调出来的超参
太低（→0.01）：分布极尖 → 近似 one-hot → 单维独大坍塌
太高（>0.07）：分布太均匀 → 过度均匀坍塌 / 训练不稳

中心化和锐化就像一对互补的搭档，共同防止了两种形式的坍塌


把 head 输出看成「把 CLS 表征映射到 65536 个『槽位』上的概率分布」。维度越大：

坍塌的「坑」更少：若 out_dim 很小（比如 64），模型很容易把所有图都映射到同一个常数分布（槽位少，恒等解太容易落到局部最优）；维度涨到 65536 后，可能的分布数量指数级爆炸，常数/one-hot 这种退化解不再是一个有吸引力的局部极小，模型被迫用不同维度编码不同语义内容。
和 centering 协同：centering 是逐维减均值（teacher_output - center）。维度越多，单维独大（单维独大坍塌）越难发生——65536 个维度里想让某一维恒定霸占概率，会被 center 更稳地削平。维度少时 centering 来不及兜底。
类似「大量 prototype」：可以类比 SwAV/DeepCluster 的几千~上万个 prototype 码本——槽位越多，表征越细、越不容易全挤成一个簇。
3. 论文的实证依据
DINO 论文专门做了 ablation：

out_dim 太小（如 4096 甚至更小）时，即使有 centering，训练仍会坍塌或效果骤降；
调到很大的值（ViT-B 用 65536、更小的 ViT-S 用 16384）才稳定且精度最高。
所以 65536 是「试出来」的经验最优区间，不是理论算出的。65536=2¹⁶ 只是个齐整的大数，方便显存对齐，本身没特殊含义。
"""


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,  # 65536
        ncrops,  # 10
        warmup_teacher_temp,  # 0.04
        teacher_temp,  # 0.04
        warmup_teacher_temp_epochs,  # 0
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))  # [1, 65536]
        # 温度系数
        """
        假设 warmup_teacher_temp=0.04、teacher_temp=0.07、warmup_teacher_temp_epochs=30、nepochs=100：
        第 0–29 轮：linspace(0.04, 0.07, 30) → 0.04, 0.041, ..., 0.07（逐轮爬升）
        第 30–99 轮：0.07, 0.07, ..., 0.07（恒定）
        数组为 [0.04, 0.041, … 0.07, 0.07×70个]，长度 100。
        """

        # ?softmax 里分母是 temp，所以：
        # ?温度越大 → 分布越平滑（软）；温度越小 → 越尖锐（峰突出）
        """
        teacher：承载更多「跨类的暗知识」，作为信息丰富的目标。
        student：逼自己给出确定预测，再去匹配教师的软分布——这个温差就是学习驱动力
        实际调参建议：
        直接沿用默认：warmup=0.04, teacher=0.04（无 warmup），多数情况能训。
        想要论文效果：把 warmup_teacher_temp_epochs 设成 30（虽然 default=0，但 DINO 论文 ViT-S 用的是 30 轮 warmup），即从 0.04 线性升到 0.04——此时 warmup 段是平的，等于没升温；真正有用的是当你想让 teacher_temp 略高于 warmup 时。
        只小幅上调 teacher_temp：比如 0.04 → 0.05，并观察 loss；绝不超过 0.07。
        loss 不降 → 调小 warmup_teacher_temp（如 0.04 → 0.02）：让早期目标更尖、信号更强。
        """
        self.teacher_temp_schedule = np.concatenate(
            [
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            ]
        )

    def forward(self, student_output, teacher_output, epoch):
        # student_output: [10*B, 65536]
        # teacher_output: [2*B, 65536]

        # 0.1 让学生的 logit 放大，softmax 后分布更尖锐（峰值更突出），逼学生给出确定性的预测
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(
            self.ncrops
        )  # 拆分学生的多个crop [(B, 65536), (B, 65536), ..., (B, 65536)]

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(
            2
        )  # 拆分教师的两个crop [(B, 65536), (B, 65536)]

        total_loss = 0
        n_loss_terms = 0

        # 教师的两个global crop和学生的10个crop进行损失的计算
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                # !跳过teacher和student前两个global crop，因为如果使用完全相同裁剪的图像计算损失，那么学生可能会学到捷径
                if v == iq:
                    continue

                # q teacher 用 softmax 得到 q
                # student_out[v] (B, 65536) -> (B)
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    """
    每步用教师输出的 batch 均值，以 center_momentum=0.9 做指数滑动平均更新 center。
    """

    @torch.no_grad()
    def update_center(self, teacher_output):
        # teacher_output.shape = [2*B, 65536]

        # batch_center.shape = [1, 65536]
        # 当前这一张卡上，对 teacher 输出沿 batch 维求和
        batch_center = torch.sum(
            teacher_output, dim=0, keepdim=True
        )  # 形状 [1, out_dim]

        # 跨所有卡求和 -> 变成"全局 batch 的总和"
        dist.all_reduce(batch_center)

        # 除以"本地样本数 × 卡数"= 全局总样本数 -> 得到全局平均中心
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # 用 EMA 把全局平均中心平滑进 self.center
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


if __name__ == "__main__":
    train_dino()
