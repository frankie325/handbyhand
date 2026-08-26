import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from detr.utils.bos_ops import box_cxcywh_to_xyxy, generalized_box_iou


# DETR 的最优二分匹配：匈牙利算法

# $$\hat{\sigma}= \underset{\sigma \in \mathfrak{S}_N}{\operatorname{argmin}} \sum_{i=1}^{N}\mathcal{L}_{\mathrm{match}}\left(y_i, \hat{y}_{\sigma(i)}\right)$$
#
# 论文中的单对匹配代价：

# $$ \mathcal{L}_{\mathrm{match}} \left(y_i, \hat{y}_{\sigma(i)}\right) = -\mathbb{1}_{\{c_i \neq \varnothing\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\mathrm{box}} \left(b_i, \hat{b}_{\sigma(i)}\right) $$
#
# 当前实现只把真实目标送入匹配器，因此指示函数
# $\mathbb{1}_{\{c_i \neq \varnothing\}}$ 恒为 1。
# 对第 q 个预测和第 j 个真实目标，实际构造的代价矩阵元素为： $$ C_{qj} = \lambda_{\mathrm{class}}\left[-p_q(c_j)\right] + \lambda_{\mathrm{bbox}}\left\|b_q-b_j\right\|_1 + \lambda_{\mathrm{giou}} \left[-\operatorname{GIoU}(b_q,b_j)\right] $$
# 默认权重为 1、5、2。no-object 不进入代价矩阵；未匹配的 query 会再计算
# 分类损失时被标记为 no-object，并且不计算边界框损失。


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class=1,
        cost_bbox=5,
        cost_giou=2,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        # pred_logits: [batch_size, num_queries, num_classes + 1]
        # pred_boxes:  [batch_size, num_queries, 4]

        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # 展平是为了并行计算所有预测与所有真实目标之间的代价
        probabilities = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes + 1]
        predicted_boxes = outputs["pred_boxes"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        # targets: list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]
        # 合并target
        target_labels = torch.cat(
            [target["labels"] for target in targets]
        )  # [batch_num_objects]  batch_num_objects 表示一批样本的真实框合并在一起的个数
        target_boxes = torch.cat(
            [target["boxes"] for target in targets]
        )  # [batch_num_objects, 4] 真实框坐标

        # 数组索引取值，找到真实类别在预测类别中的预测概率，”-“号表示预测概率越高，匹配成本越低
        class_cost = -probabilities[
            :, target_labels
        ]  # [batch_size * num_queries, batch_num_objects]

        # 归一化 cxcywh 框之间的 L1 距离
        """
        x1：形状 [m, d]
        x2：形状 [n, d]
        p：距离范数（p=1 曼哈顿距离，p=2 欧氏距离，默认 2）
        返回：[m, n]，其中 result[i][j] = x1 第 i 行 与 x2 第 j 行的距离

        匈牙利匹配需要一张代价矩阵：每个预测 query 和每个真实目标之间的"匹配代价"。：
                    目标1   目标2   目标3   ...
        预测query1  [d11    d12    d13   ...]
        预测query2  [d21    d22    d23   ...]
        预测query3  [d31    d32    d33   ...]
        ...

        """
        bbox_cost = torch.cdist(
            predicted_boxes,
            target_boxes,
            p=1,  # 计算L1距离
        )  # [batch_size * num_queries, batch_num_objects]

        # 计算GIoU前，需要把 cxcywh 转换为 xyxy
        giou_cost = -generalized_box_iou(
            box_cxcywh_to_xyxy(predicted_boxes),  # [batch_size * num_queries, 4]
            box_cxcywh_to_xyxy(target_boxes),  # [batch_num_objects, 4]
        )  # [batch_size * num_queries, batch_num_objects]

        cost_matrix = (
            self.cost_class * class_cost  # 类别代价（权重1）
            + self.cost_bbox * bbox_cost  # 框L1距离代价（权重5）
            + self.cost_giou * giou_cost  # GIoU代价（权重2）
        )  # 计算代价矩阵

        cost_matrix = cost_matrix.view(
            batch_size,
            num_queries,
            -1,
        ).cpu()  # [batch_size, num_queries, batch_num_objects]

        target_sizes = [
            len(target["boxes"]) for target in targets
        ]  # [每个样本的真实框的个数]

        indices = []
        for image_index, image_cost in enumerate(
            cost_matrix.split(
                target_sizes, dim=-1
            )  # 最后一维按照每个样本进行分割  list[ [batch_size, num_queries, num_objects] ]
        ):
            # image_cost: [batch_size, num_queries, num_objects] 每个样本的代价矩阵，送入匈牙利算法
            # 匈牙利算法返回两个一维张量，分别表示每个样本的预测框和真实框的索引

            """
            cost = [[0.4, 0.9],   # query0 对 目标0/1 的代价
                    [1.0, 0.3],   # query1
                    [0.8, 0.7]]   # query2

            prediction_indices, target_indices = linear_sum_assignment(cost)
            # prediction_indices = [0, 1]     ← 行：query 0、query 1
            # target_indices     = [0, 1]     ← 列：目标 0、目标 1
            匹配结果：
            query 0 ↔ 目标 0（代价 0.4）
            query 1 ↔ 目标 1（代价 0.3）
            query 2 不匹配（成为 no-object/背景）
            """
            prediction_indices, target_indices = linear_sum_assignment(
                image_cost[image_index]
            )
            indices.append(
                (
                    torch.as_tensor(prediction_indices, dtype=torch.int64),
                    torch.as_tensor(target_indices, dtype=torch.int64),
                )
            )
            # indices: list[ (prediction_indices, target_indices) ] 每个样本配对的预测框索引和匹配框索引
            # prediction_indices: [0,1] target_indices: [1,3] 表示query0-第3个真实框配对， query1-第4个真实框配对
        return indices
