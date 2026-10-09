# 04：共享权重架构搜索

本项目把最大模型当作 supernet，小模型直接复用它的参数前缀：

```text
最大模型：512 分辨率、300 queries、4 层 Decoder
├── 384、100 queries、3 层 Decoder
├── 320、 50 queries、2 层 Decoder
└── 其他组合
```

训练时，`sample_active_subnet()` 随机选择分辨率、query 数量和 Decoder 深度。不同子网络共享骨干、投影器、检测头及 Decoder 前几层的权重。

搜索时不再训练权重，而是逐个激活候选配置，测量验证 AP 与单张推理延迟。若没有另一个候选同时更快且 AP 更高，该候选就位于 Pareto 前沿。

## 与论文 NAS 的差异

论文还搜索 patch size、窗口注意力配置等结构。本项目使用外部预训练 DINOv2；修改 patch projection 会破坏预训练权重形状，因此只搜索可以安全共享的三个维度。

## 结果解释

`nas_results.json` 包含全部候选，`pareto_frontier.json` 只包含非支配候选。延迟依赖当前设备、PyTorch 版本和 batch size，不能直接和论文中的 TensorRT T4 延迟比较。

## 练习

1. 固定 queries，只搜索分辨率和深度。
2. 比较随机子网训练和只训练最大子网得到的 Pareto 前沿。
3. 分别在 CPU 和 CUDA 测量延迟，解释排序变化。

