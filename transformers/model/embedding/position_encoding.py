import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        _2i = torch.arange(0, dim_model, 2, dtype=torch.float)  # [dim_model/2, ]
        div_term = torch.pow(10000, _2i / dim_model)  # [dim_model/2,]

        sins = torch.sin(pos / div_term)  # 广播之后，形状变为 [max_len, dim_model/2]
        coss = torch.cos(pos / div_term)  # 广播之后，形状变为 [max_len, dim_model/2]

        pe = torch.zeros([max_len, dim_model])  # [最大序列长度, 分词维度]
        # pe[:, 0::2] 表示取出pe的第0列，步长为2的所有列，pe[:, 1::2] 表示取出pe的第1列，步长为2的所有列
        pe[:, 0::2] = sins
        pe[:, 1::2] = coss

        pe = pe.unsqueeze(0)  # 添加batch维度 [1, max_len, dim_model]
        # 将pe注册为模型的参数，注册到buffer的张量，模型调用to device方法的时候会跟随同步到对应的设备上
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.shape: [batch_size, seq_len, dim_model]

        seq_len = x.shape[1]  # 当前这批序列的长度

        # requires_grad_(False)固定位置编码不需要学习
        part_pe = self.pe[:, :seq_len].requires_grad_(
            False
        )  # 从位置编码中取出当前这批序列长度的位置编码

        # part_pe.shape: [1, seq_len, dim_model]， 下面的相加会进行广播 -> 变成[batch_size, seq_len, dim_model]进行相加
        result = x + part_pe
        return self.dropout(result)
