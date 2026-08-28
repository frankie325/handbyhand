# DETR + PASCAL VOC2007

本项目直接读取 PASCAL VOC2007 的 JPEG、XML 标注和 train/val 划分，
不需要转换为其他标注格式。

## 数据目录

默认数据根目录是仓库根目录下的 `datasets/VOC2007`：

```text
datasets/VOC2007/
├── Annotations/          # 000001.xml ...
├── ImageSets/Main/
│   ├── train.txt         # 2501 张
│   └── val.txt           # 2510 张
└── JPEGImages/           # 000001.jpg ...
```

数据集使用标准 VOC20 类，按照以下顺序映射到连续标签 `0..19`：

```text
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor
```

模型分类头输出 21 个值：前 20 个是真实类别，最后一个是 no-object。
VOC XML 的 `difficult=1` 目标不参与训练；验证时它们作为忽略目标处理。

## 安装与数据检查

在仓库根目录运行：

```bash
uv sync
uv run python -m detr.check_data
```

`check_data` 会检查数据数量、类别和边界框，构建一个 batch，并执行轻量模型
前向和损失计算。只检查数据时可以运行：

```bash
uv run python -m detr.check_data --skip-model
```

如需使用其他 VOC2007 根目录：

```bash
uv run python -m detr.check_data --data-root /path/to/VOC2007
```

## 训练

```bash
uv run python -m detr.train
```

训练配置位于 `detr/config.py`。最佳权重保存到 `detr/weights/best.pth`。
旧的 92 维分类头权重与 VOC20 模型不兼容，切换数据集后必须重新训练。

## 验证

```bash
uv run python -m detr.validate
```

验证会输出各项损失、20 个类别的 AP，以及 VOC2007 官方口径的
`mAP@0.5`（IoU 0.5、11 点插值）。每个非 difficult GT 只能匹配一次；
匹配 difficult GT 的预测不会计为 TP 或 FP。

## 预测

在 `detr/predict.py` 的 `main()` 中填写待预测图片路径，然后运行：

```bash
uv run python -m detr.predict
```

默认只绘制置信度不低于 0.5 的预测，结果写入 `detr/outputs/`。
