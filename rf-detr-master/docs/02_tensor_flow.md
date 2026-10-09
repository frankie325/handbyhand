# 02：张量完整流程

以下以 `batch_size=2`、输入 224、patch size 16、`hidden_size=128`、50 queries 为例。

## 输入和骨干

```text
images        [2, 3, 224, 224]
padding_mask  [2, 224, 224]
patch tokens  [2, 196, backbone_hidden_size]
feature maps  4 × [2, backbone_hidden_size, 14, 14]
```

当 `num_windows=2` 时，窗口注意力层把 `14×14` 网格切成四个 `7×7` 窗口，每次 attention 只处理 49 个 patch token；全局注意力层处理全部 196 个 patch token。窗口层输出会按原坐标拼回 `14×14`，所以切换注意力类型不会改变特征图形状。

padding mask 中 `False` 表示真实图像，`True` 表示填充区域。注意力和候选选择都不会把填充位置当成有效图像。

## 多尺度投影

四个骨干特征会分别变换到每个目标尺度，然后在通道维拼接。以轻量骨干的 192 通道为例，P3 分支中的每个转置卷积把通道降到 96，拼接后为 384 通道；P4、P5 拼接后为 768 通道。各分支再通过 C2f 压缩到配置中的 `hidden_size=128`。

```text
P3  [2, 128, 28, 28]
P4  [2, 128, 14, 14]
P5  [2, 128,  7,  7]
```

展平并拼接后：

```text
memory [2, 28×28 + 14×14 + 7×7, 128]
       [2, 1029, 128]
```

## 两阶段 Query

每个 memory 位置对应一个规则网格 anchor。分类头给 1029 个位置打分，选出 50 个候选：

```text
query           [2, 50, 128]
reference_boxes [2, 50, 4]
enc_outputs:
  pred_logits   [2, 50, num_classes]
  pred_boxes    [2, 50, 4]
```

## 多尺度可变形交叉注意力

每个 query、每个 attention head、每个尺度预测若干二维偏移：

```text
sampling_offsets
[batch_size, num_queries, num_heads, num_levels, num_points, 2]
```

reference box 给出中心和宽高，offset 决定实际采样位置。`grid_sample` 从每个尺度进行双线性采样，权重归一化后求和。

## Decoder 输出

每层根据当前 query 预测框增量：

```text
new_box = sigmoid(inverse_sigmoid(old_box) + delta)
```

最后一层放在 `pred_logits`、`pred_boxes`，前面各层放在 `aux_outputs`，候选生成结果放在 `enc_outputs`。

## 练习

1. 将 `num_points` 从 2 改为 4，写出 sampling_offsets 的新形状。
2. 在 `deformable_attention_core` 中打印三个尺度的 grid 范围。
3. 构造全为 `True` 的 padding mask，观察 top-k 候选为何失去意义。
