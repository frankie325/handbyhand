# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        """对 teacher 输出做 Sinkhorn-Knopp 归一化（DINO / SwAV 中用于生成教师目标分布）。

        作用：
            把 teacher 的原始 logits 转换成一张“平滑且平衡”的分配目标分布，
            供 student 网络通过交叉熵去匹配。对应配置 centering: sinkhorn_knopp
            （如 vitl14.yaml）；默认配置 ssl_default_config.yaml 用的是
            softmax_center_teacher，由 centering 字段选择其一。

        为什么需要 Sinkhorn-Knopp（防止表征塌缩）：
            若直接用 teacher 输出做目标，所有样本可能都预测成少数几个 prototype，
            导致 student “偷懒”/塌缩。Sinkhorn-Knopp 通过交替行列归一化，强制得到
            一张近似“双随机”的分配矩阵：
              - 每个 prototype（行）平均分到 1/K 权重（没有原型被冷落）；
              - 每个样本（列）平均分到 1/B 权重（没有样本占满）。
            从而把 batch 内样本均匀、分散地铺满所有 prototype。

        算法步骤（n_iterations 默认 3 次，已足够接近平衡）：
            1. 温度缩放 + exp：Q = exp(teacher_output / teacher_temp)
            2. 转置为 K×B（K = prototype 数，B = 样本数，对应论文记号）
            3. 全局归一，使整个矩阵和为 1
            4. 反复交替（默认 3 轮）：
                 - 行归一化 -> 每行和 = 1/K
                 - 列归一化 -> 每列和 = 1/B
            5. 乘 B 并转置回去：每列（每个样本）和为 1，得到合法分配概率。

        与 softmax_center_teacher 的区别：
            后者只做“减中心 + softmax 锐化”，而本方法用 Sinkhorn 生成双随机分配，
            更强调 batch 内分配的均衡性。两者都由配置 centering 字段选择其一。
        """
        teacher_output = teacher_output.float() #  (2*B, 投影头输出维度)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # 样本数
        K = Q.shape[0]  # how many prototypes = 教师输出空间里的 K 个伪类别/视觉概念簇，训练目标就是让样本在它们上面均衡、分散地分配

        # 全局归一：让整个矩阵 Q 的所有元素之和 = 1
        sum_Q = torch.sum(Q)  # 计算 Q 中所有元素的总和
        if dist.is_initialized():  # 多卡(DDP)训练时
            dist.all_reduce(sum_Q)  # 把所有卡上的 sum_Q 累加，得到全局总和
        Q /= sum_Q  # 除以总和，完成全局归一化（此时 Q 各项之和为 1）

        for it in range(n_iterations):  # 交替行列归一化，迭代 n_iterations 次（默认 3）
            # 行归一化：每个 prototype（行）的总权重应 = 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)  # 按行求和，得到每个 prototype 的权重总和
            if dist.is_initialized():  # 多卡时
                dist.all_reduce(sum_of_rows)  # 跨卡累加各行和，保证全局一致
            Q /= sum_of_rows  # 每行除以其和，使每行和 = 1
            Q /= K  # 再除以 K，使每行和 = 1/K（每个 prototype 平均分配到权重）

            # 列归一化：每个样本（列）的总权重应 = 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)  # 按列求和并除以，使每列和 = 1
            Q /= B  # 再除以 B，使每列和 = 1/B（每个样本平均分配到权重）

        Q *= B  # 乘回 B：使每列和重新变回 1，Q 才是一张合法的“分配矩阵”
        return Q.t()  # 转置回 B×K 返回（每行对应一个样本的分配概率分布）

    # student的8个local_crops和teacher的2个global_crops计算损失（cls_token损失计算，dino_local_crops_loss）
    # student的2个global_crops和teacher的2个global_crops计算损失（cls_token损失计算，dino_global_crops_loss）
    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
