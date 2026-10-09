# 05：教学版与官方 RF-DETR 的差异

本项目保留了理解 RF-DETR 所需的检测主链路，但没有追求官方权重和部署兼容。

| 项目 | 教学版 | 官方 RF-DETR |
|---|---|---|
| 骨干 | HF DINOv2 或轻量 ViT，并按层切换窗口/全局注意力 | 定制窗口化 DINOv2 |
| 多尺度投影 | 与官方相同的逐层缩放、通道拼接、C2f、LayerNorm2d 主流程，固定输出 P3/P4/P5 | 同一主流程，并支持更多工程配置 |
| Deformable Attention | 纯 PyTorch `grid_sample` | 优化内核与多后端路径 |
| 训练框架 | 原生 PyTorch | PyTorch Lightning 完整训练栈 |
| 任务 | 目标检测 | 检测、分割、关键点 |
| NAS | 分辨率、queries、Decoder 深度 | 更大的搜索空间和大规模训练 |
| 导出 | 无 | ONNX、TensorRT、CoreML 等 |
| 官方检查点 | 不兼容 | 兼容发布权重 |

教学版现在复现了窗口层与全局层交替的关键数据流，并将多尺度投影器对齐为官方的逐层缩放、通道拼接和 C2f 融合。官方代码仍在自定义 DINOv2 内部维护窗口布局，并包含发布模型、额外尺度、导出和训练性能优化；教学版为已经加载的 Hugging Face block 添加清晰的窗口包装器，并固定输出 P3/P4/P5。两者不能直接交换检查点。

阅读本项目后，推荐在相邻官方仓库中按以下顺序继续：

```text
models/backbone/backbone.py
→ models/backbone/projector.py
→ models/transformer.py
→ models/lwdetr.py
→ models/matcher.py
→ models/criterion.py
```

先对照相同概念，再处理官方代码中的性能、兼容性和多任务分支，会比直接从高层 API 向下读容易得多。
