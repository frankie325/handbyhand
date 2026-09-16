# 手写 DINO

这个目录是用于学习的手写版 DINO（Self-Distillation with No Labels）实现。当前训练脚本使用 Imagenette 数据集，并且只支持 CUDA/NCCL。

## 1. 数据集目录

训练使用 `train`，k-NN evaluation 使用 `train` 建立特征库并在 `val` 上评估：

```text
datasets/imagenette2-160/train/
├── n01440764/
├── n02102040/
└── ...

datasets/imagenette2-160/val/
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
| 模型 | `vit_s_16` |
| 每张 GPU 的 batch size | `16`（显存允许时可增至 `32`） |
| 训练轮数 | `300` |
| DINO 输出维度 | `8192` |
| local crop 数量 | `8` |
| student drop path | `0.1` |
| teacher momentum | `0.9995 → 1.0` |
| teacher temperature | `0.04 → 0.07`，前 30 轮预热 |
| 混合精度 | 开启 |

如果显存不足，可以优先减小 `batch_size_per_gpu`，或者将模型改为 `vit_t_16`。

## 5. 训练输出

训练结果默认保存在：

```text
dino/weights/vits16/
```

其中：

- `checkpoint.pth`：最近一次保存的训练状态，可用于断点续训。
- `checkpointXXXX.pth`：按保存周期生成的历史检查点。

每个 epoch 的 loss 当前直接打印到终端，脚本还没有将日志写入文件。

再次启动训练时，如果 `checkpoint.pth` 存在，脚本会自动恢复模型、优化器、学习率调度器和 DINO loss 状态。

原来的 ViT-B checkpoint 仍保留在 `dino/weights/checkpoint.pth`，新配置不会覆盖它。

## 6. 正式 k-NN evaluation

评估脚本冻结 DINO backbone，使用完整 Imagenette `train` split 建立归一化特征库，再在完整 `val` split 上执行余弦相似度加权 k-NN，报告 Top-1 和 Top-5。

评估新 ViT-S checkpoint：

```bash
cd /Users/frank/code/ai/handbyhand
.venv/bin/python -m dino.eval_knn
```

评估原来的 ViT-B checkpoint：

```bash
.venv/bin/python -m dino.eval_knn \
  --checkpoint dino/weights/checkpoint.pth
```

默认参数为 `k=20`、温度 `0.07`、评估 teacher。模型结构会从 checkpoint 自动读取。结果同时打印到终端并保存为 checkpoint 同目录下的 `knn_eval.json`。

常用参数：

```bash
.venv/bin/python -m dino.eval_knn \
  --checkpoint dino/weights/vits16/checkpoint.pth \
  --checkpoint-key teacher \
  --batch-size 128 \
  --k 20 \
  --temperature 0.07 \
  --device cuda
```

## 7. 注意力可视化

可视化脚本会从 checkpoint 自动读取模型结构，避免把 ViT-S 权重加载到 ViT-B。新 checkpoint 训练完成后运行：

```bash
.venv/bin/python -m dino.visualize
```

可视化旧 ViT-B checkpoint：

```bash
.venv/bin/python -m dino.visualize \
  --pretrained-weights dino/weights/checkpoint.pth \
  --image-path dino/images/fish.png
```

这些输出是最后一层 CLS self-attention，不是监督分割结果；应使用 k-NN Top-1/Top-5 作为主要定量指标。

## 8. 运行前检查

```bash
cd /Users/frank/code/ai/handbyhand
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

开始训练前确认：

- `CUDA` 输出为 `True`。
- `GPU count` 至少为 `1`。
- `datasets/imagenette2-160/train` 已存在并包含类别子目录。

## 9. 常见问题

### 相对导入失败

如果出现 `attempted relative import with no known parent package`，请回到仓库根目录并使用：

```bash
.venv/bin/python -m dino.train
```

### CUDA 显存不足

在 [`train.py`](train.py) 中减小 `batch_size_per_gpu`。如果仍然不足，再减小 `local_crops_number` 或换用更小的 ViT。

### 多卡进程数与 GPU 数量不一致

确保 `--nproc-per-node` 不大于当前机器可用的 GPU 数量。
