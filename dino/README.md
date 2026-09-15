# 手写 DINO

这个目录是用于学习的手写版 DINO（Self-Distillation with No Labels）实现。当前训练脚本使用 Imagenette 数据集，并且只支持 CUDA/NCCL。

## 1. 数据集目录

训练脚本默认读取：

```text
datasets/imagenette2-160/train/
├── n01440764/
├── n02102040/
└── ...
```

也就是仓库根目录下的 `datasets/imagenette2-160/train`。每个类别需要单独放在一个子目录中，以便 `torchvision.datasets.ImageFolder` 读取。

## 2. 单卡训练

必须先进入项目根目录，再使用模块方式启动：

```bash
cd /Users/frank/code/ai/handbyhand
.venv/bin/python -m dino.train
```

如果使用 `uv` 管理环境，也可以运行：

```bash
cd /Users/frank/code/ai/handbyhand
uv run python -m dino.train
```

不要运行 `python dino/train.py`，因为训练脚本使用了包内相对导入。

## 3. 单机多卡训练

下面的例子使用 4 张 GPU：

```bash
cd /Users/frank/code/ai/handbyhand
.venv/bin/torchrun --standalone --nproc-per-node=4 --module dino.train
```

把 `4` 改成实际使用的 GPU 数量。`batch_size_per_gpu` 是每张 GPU 的 batch size，因此全局 batch size 为：

```text
全局 batch size = batch_size_per_gpu × GPU 数量
```

## 4. 当前默认配置

当前参数直接写在 [`train.py`](train.py) 的 `Namespace` 中，训练脚本暂时没有命令行参数解析。

| 参数 | 默认值 |
| --- | --- |
| 模型 | `vit_b_16` |
| 每张 GPU 的 batch size | `8` |
| 训练轮数 | `30` |
| DINO 输出维度 | `8192` |
| local crop 数量 | `4` |
| 混合精度 | 开启 |

如果显存不足，可以优先减小 `batch_size_per_gpu`，或者将模型改为 `vit_s_16` / `vit_t_16`。

## 5. 训练输出

训练结果默认保存在：

```text
dino/outputs/
```

其中：

- `checkpoint.pth`：最近一次保存的训练状态，可用于断点续训。
- `checkpointXXXX.pth`：按保存周期生成的历史检查点。

每个 epoch 的 loss 当前直接打印到终端，脚本还没有将日志写入文件。

再次启动训练时，如果 `checkpoint.pth` 存在，脚本会自动恢复模型、优化器、学习率调度器和 DINO loss 状态。

## 6. 运行前检查

```bash
cd /Users/frank/code/ai/handbyhand
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

开始训练前确认：

- `CUDA` 输出为 `True`。
- `GPU count` 至少为 `1`。
- `datasets/imagenette2-160/train` 已存在并包含类别子目录。

## 7. 常见问题

### 相对导入失败

如果出现 `attempted relative import with no known parent package`，请回到仓库根目录并使用：

```bash
.venv/bin/python -m dino.train
```

### CUDA 显存不足

在 [`train.py`](train.py) 中减小 `batch_size_per_gpu`。如果仍然不足，再减小 `local_crops_number` 或换用更小的 ViT。

### 多卡进程数与 GPU 数量不一致

确保 `--nproc-per-node` 不大于当前机器可用的 GPU 数量。
