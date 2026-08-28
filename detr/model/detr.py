import torch
import torch.nn as nn

from detr.model.transformer import build_transformer
from .backbone import Backbone
from .position_encoding import PositionEmbeddingSine


class MLP(nn.Module):
    """4层全连接网络，用于预测bbox"""

    def __init__(
        self,
        input_size: int,  # 输入尺寸，默认256
        hidden_size: int,  # 隐藏层大小
        output_size: int,  # 输出尺寸固定为4，x,y,w,h
        num_layers: int,  # 全连接网络层数
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        # input_size=256, hidden_size=256, output_size=4, num_layers=3
        # sizes = [256, 256, 256, 4]
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        """
         [256, 256, 256] [256, 256, 4] 配对
          Linear1: 256 → 256   （第 0 层）
          Linear2: 256 → 256   （第 1 层）
          Linear3: 256 → 4     （第 2 层）
        """
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, output_dim)
                for input_dim, output_dim in zip(sizes[:-1], sizes[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.relu(x)
        return x  # 最后一层不加激活函数，输出 [batch_size, 100, 4]


class Detr(nn.Module):
    def __init__(
        self,
        num_classes=20,  # 真实类别数，标签使用连续的 0..num_classes-1
        num_queries: int = 100,  # 解码器的query数，默认100个query
        d_model=256,
        N=6,  # 编码器和解码器的层数
        d_ff=2048,  # 前馈网络的中间隐藏层大小
        n_head=8,
        dropout=0.1,
        pretrained_backbone=True,  # 是否加载backbone的预训练权重
        train_backbone=True,  # 是否冻结backbone的权重，冻结了则backbone不参与训练
    ):
        super(Detr, self).__init__()

        # 主干网络，输出形状 [batch_size, 2048, height, width]
        self.backbone = Backbone(pretrained_backbone, train_backbone)

        # 经过1*1的卷积，将通道数变为d_model  [batch_size, d_model, height, width]
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, 1)

        # 二维位置编码
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=d_model // 2,
            normalize=True,
        )

        # object_query
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.transformer = build_transformer(N, d_model, d_ff, n_head, dropout)

        # 最后一个输出位置 num_classes 专门表示 no-object。
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self.init_weights()

    def forward(self, x, padding_mask=None):
        # 输入x形状 [batch_size, 3, height, width]
        # 输入mask形状 [batch_size, height, width]
        x, feature_mask = self.backbone(
            x, padding_mask
        )  # x.shape [batch_size, 2048, height/32, width/32] 下面的height和width都是/32后的值
        # feature_mask.shape [batch_size, height/32, width/32]

        x = self.input_proj(x)  # [batch_size, d_model, height, width]

        pos = self.position_embedding(
            x, feature_mask
        )  # [batch_size, d_model, height, width]
        # flatten(2)表示从第2维height之后进行展平 # [batch_size, d_model, height*width]
        # transpose(1, 2)表示将 第1维d_model和第2维height*width进行转置
        # 最终得到[batch_size, height*width, d_model]
        x = x.flatten(2).transpose(1, 2)
        pos = pos.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]
        # !weight存储的是100个可学习的向量，query_pos负责学习空间位置信息
        # weight [100, d_model] -> [100, 1, d_model] -> [batch_size, 100, d_model]
        query_pos = self.query_embed.weight.unsqueeze(0).expand(
            batch_size,  # 这一维扩展到batch_size
            -1,  # -1表示不扩展这个维度
            -1,  # -1表示不扩展这个维度
        )

        # tgt 初始化为0，作为输入内容
        tgt = torch.zeros_like(query_pos)

        if feature_mask is not None:
            # feature_mask的True 表示 padding，False 表示有效区域，传入transformer前先取反
            # [batch_size, 1, height*width(相当于是seq_len)]
            feature_mask = (~feature_mask).flatten(1).unsqueeze(1)

        # hidden_states.shape [num_layers, batch_size, 100, d_model]
        hidden_states = self.transformer(
            x,
            tgt,
            pos,
            query_pos,
            feature_mask,
        )

        output_classes = self.class_embed(
            hidden_states
        )  # [num_layers, batch_size, 100, num_classes + 1]
        output_boxes = self.bbox_embed(
            hidden_states
        ).sigmoid()  # 将预测x,y,w,h归一化到[0,1] [num_layers, batch_size, 100, 4]

        return {
            "pred_logits": output_classes[-1],  # [batch_size, 100, num_classes + 1]
            "pred_boxes": output_boxes[-1],  # [batch_size, 100, 4]
            # 每一层decoder的输出
            "aux_outputs": [
                {
                    "pred_logits": output_class,  # [batch_size, 100, num_classes + 1]
                    "pred_boxes": output_box,  # [batch_size, 100, 4]
                }
                for output_class, output_box in zip(
                    output_classes[:-1], output_boxes[:-1] # :-1表示不取最后一层
                )
            ],
        }

    def init_weights(self):
        modules = [
            self.input_proj,
            self.class_embed,
            self.bbox_embed,
        ]
        for module in modules:
            for parameter in module.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
