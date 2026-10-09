# 03：训练、损失与验证

## 一对一匹配

模型固定输出许多 query，而一张 RAOD 图片通常只有少量目标。Hungarian Matcher 使用三种代价寻找全局最优一对一对应：

```text
matching_cost = 2 × focal_class_cost
              + 5 × L1_box_cost
              + 2 × GIoU_cost
```

匹配只决定“哪个 query 学哪个 GT”，本身不参与反向传播。

## 损失

- sigmoid focal loss：所有未匹配 query 都是隐式背景。
- L1：约束归一化 `cxcywh` 的坐标距离。
- GIoU：约束预测框与 GT 的几何重叠。
- auxiliary loss：监督每个 Decoder 中间层。
- encoder loss：监督两阶段候选生成器。

## 优化

骨干网络和检测头使用不同学习率。DINOv2 通常使用更小学习率；本地配置直接冻结它。训练支持梯度累积、梯度裁剪、CUDA AMP、warmup 和 cosine decay。

## 验证

验证将归一化框恢复到原图坐标，写成 COCO prediction JSON，再通过 pycocotools 计算：

- AP50:95：主指标。
- AP50、AP75：不同定位严格程度。
- APS、APM、APL：小、中、大目标表现。

对于道路遗落物，还应在项目后续加入视频级召回率、每小时误报数和告警延迟；单帧 COCO AP 不能完整代表真实部署表现。

## 调试顺序

1. 运行 `check_data.py`，先确认标注和类别映射。
2. 使用 16 张图的 `local_smoke.yaml`，确认损失可以下降。
3. 检查 `loss_bbox` 和 `loss_giou`，避免只看总损失。
4. 可视化训练集预测，确认框坐标恢复正确。
5. 再切换 DINOv2 和完整 RAOD。

## 练习

1. 去掉 encoder loss，比较收敛速度。
2. 把 GIoU 权重设为 0，观察预测框形状。
3. 从 `last.pth` 恢复训练，确认 metrics.csv 的 epoch 连续。

