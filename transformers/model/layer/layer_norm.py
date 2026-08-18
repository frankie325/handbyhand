import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # features: 最后一个维度，即d_model
        super(LayerNorm, self).__init__()
        # 学习参数：让模型可以学习在归一化后的基础上进行适当的调整，保证归一化不会限制模型的表示能力
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  # 防止除以0

    def forward(self, x):
        # 对最后一个维度求均值, 保持维度不变
        mean = x.mean(-1, keepdim=True)
        # 对最后一个维度求标准差, 保持维度不变
        std = x.std(-1, keepdim=True)

        # 标准化变化
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
