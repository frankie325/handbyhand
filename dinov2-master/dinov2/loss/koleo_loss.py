# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.distributed as dist


logger = logging.getLogger("dinov2")

"""
 KoLeoLoss只对student 的全局裁剪 cls token 起作用
 让一批样本中的cls_toekn分布均匀：
 自监督训出来的 cls_token，本质上是给每张图发的一张"身份证"。后面做分类、检索、检测时，都是拿这张身份证去比来比去、或者接个小分类头。所以有一个最基本的要求：
 不同的图，要有不同的身份证；而且身份证的"编码空间"要被充分利用

 实现原理（Kozachenko-Leonenko 熵估计 / Spreading vectors）：
 - KOLEO 本质是一个"防抱团"的熵正则：通过最小化「每个特征向量到其最近邻距离的负对数」，
   迫使 batch 内的 cls_token 在单位球面上尽量均匀散开，从而最大化分布熵、防止特征坍缩。
 - 计算三步：
   1) 归一化到单位球面：把每个表征长度压成 1、只保留方向（见 forward 中的 F.normalize）。
      归一化后「点积越大 = 距离越近」，所以用点积就能找最近邻，无需真算距离。
   2) 找最近邻（pairwise_NNs_inner）：用 x @ x.t() 算两两内积得到 n×n 表；
      对角线是「自己和自己的点积=1」（永远最大），必须填 -1 排除自己，
      再对每行取 argmax 得到真正的最近邻索引 I。
   3) 算惩罚（forward）：用 nn.PairwiseDistance(2) 逐行算「每个点到其最近邻的真实欧氏距离」，
      再取 -log(距离) 平均。距离越近（点挤在一起）惩罚越重，逼着模型把点推散。
 - 为什么是 batch 内约束：无法对全量数据算全局分布，但每个 batch 内部都尽量均匀，
   拼起来整体就均匀了——batch 级正则是全局均匀的廉价近似。
 - 与 DINO 的 centering/sharpening 互补：teacher 的 centering 负责宏观防「某几个原型被垄断」，
   KOLEO 负责微观上把 batch 内的点「撑开」，双保险避免坍缩。
"""


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        # 每个点和每个点两两算点积，得到 n×n 的表，dots[i][j] 就是第 i 个点和第 j 个点的点积。
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        # dots 是个 n×n 的方阵，平铺成一维数组后，对角线元素的位置正好是每隔 n+1 个出现一个
        dots.view(-1)[:: (n + 1)].fill_(-1)  # 对角填 -1 排除自己和自己计算的点积
        # max inner prod -> min distance
        # 再取每行最大，得到的就是真正的"别人里最近的那个"， I[i] = 第 i 个点的最近邻的索引
        _, I = torch.max(dots, dim=1)  # noqa: E741 I.shape = (n,)
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            # (batch size, embed_dim)
            student_output = F.normalize(
                student_output, eps=eps, p=2, dim=-1
            )  # 先归一化到单位球面
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(
                distances + eps
            ).mean()  # 负对数距离取平均，距离越大，loss越小
        return loss
