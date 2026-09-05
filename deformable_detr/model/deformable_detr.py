import torch
import torch.nn.functional as F
from torch import nn
from .backbone import Backbone
from .position_encoding import PositionEmbeddingSine
from .deformable_transformer import build_deforamble_transformer
import math

from ..utils.common import clones, inverse_sigmoid


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDetr(nn.Module):
    def __init__(
        self,
        num_classes,
        num_queries,
        # num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        d_model=256,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        enc_n_pointer=4,
        dec_n_pointer=4,
        pretrained_backbone=True,
        train_backbone=True,
    ):
        super(DeformableDetr, self).__init__()

        if two_stage and not with_box_refine:
            raise ValueError(
                "two_stage=True 时必须同时设置 with_box_refine=True，"
                "因为两阶段 Encoder 需要 bbox_embed 生成候选框。"
            )

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.num_feature_levels = 4  # 主干网这里固定输出4个特征图
        # 主干网
        self.backbone = Backbone(
            pretrained_backbone=pretrained_backbone,
            train_backbone=train_backbone,
        )

        # 位置编码
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=d_model // 2, normalize=True
        )
        self.transformer = build_deforamble_transformer(
            hidden_dim=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_feature_levels=self.num_feature_levels,
            enc_n_points=enc_n_pointer,
            dec_n_points=dec_n_pointer,
            two_stage=two_stage,
            num_queries=num_queries,
        )  # 构建 Deformable Transformer
        hidden_dim = d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        input_proj_list = []
        nums_feature = len(self.backbone.num_channels)

        # 将主干网输出的3个特征图进行1*1卷积，转换成通道数为 d_model 的特征图
        for i in range(nums_feature):
            in_channels = self.backbone.num_channels[i]
            input_proj = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                # 分成32组；d_model=256时每组8个通道，改善小 batch 下的训练稳定性
                nn.GroupNorm(32, hidden_dim),
            )
            input_proj_list.append(input_proj)

        # 主干网生成的最后一个特征图，再进行一次3*3, stride=2的卷积，下采样64倍，让模型看到更粗粒度、更大感受野的特征，适合大目标
        input_proj_list.append(
            nn.Sequential(
                nn.Conv2d(
                    self.backbone.num_channels[-1],
                    hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(32, hidden_dim),
            )
        )

        self.input_proj = nn.ModuleList(input_proj_list)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, d_model * 2)  # [num_queries, 512]

        # 官方初始化：类别偏置先验、bbox_head 全零、input_proj 用 xavier
        # 必须在 with_box_refine 克隆之前执行，否则 bbox 偏置前两位(dx,dy)会是随机值
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if two_stage
            else self.transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = clones(self.class_embed, num_pred)
            self.bbox_embed = clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, images, padding_mask=None):

        # images.shape = [batch_size, 3, height, width]
        features = self.backbone(images, padding_mask)

        srcs = []
        masks = []
        pos = []

        for i, (feature, mask) in enumerate(features):
            srcs.append(
                # [batch_size, d_model, height/下采样倍数, weight/下采样倍数]
                self.input_proj[i](feature)
            )
            masks.append(mask)
            pos.append(self.position_embedding(feature, mask))
        feature, _ = features[-1]
        last_feature = self.input_proj[-1](
            feature
        )  # [batch_size, d_model, height/64, weight/64]
        srcs.append(last_feature)
        last_mask = self.backbone.get_mask(padding_mask, last_feature)
        masks.append(last_mask)
        pos.append(self.position_embedding(last_feature, last_mask))

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, pos, query_embeds)

        # hs.shape = [batch_size, num_queries, d_model] 或者 [num_decoder_layers, batch_size, num_queries, d_model]
        # inter_references.shape = [batch_size, num_queries, 4] 或者 [num_decoder_layers, batch_size, num_queries, 4]

        # 官方代码这里只对返回所有中间层的输出做了处理
        outputs_classes = []
        outputs_coords = []
        # 遍历每一层 Decoder
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # 第 0 层使用初始 reference point：
                reference = init_reference
            else:
                # 后面的层使用上一层更新后的 reference point：
                reference = inter_references[lvl - 1]
            # 反 sigmoid 将 [0, 1] 范围的参考坐标转换到 logit 空间，便于与预测偏移量相加
            reference = inverse_sigmoid(reference)

            # !预测类别分支
            outputs_class = self.class_embed[lvl](
                hs[lvl]
            )  # [batch_size, num_queries, num_classes]
            # !预测边界框偏移量
            tmp = self.bbox_embed[lvl](hs[lvl])  # [batch_size, num_queries, 4]
            # 如果reference的最后一维是4，说明是x,y,w,h，则直接加上层参考位置reference
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                # 如果是x,y，则加上reference的xy部分，
                # !单阶段只有初始参考点eference只有x,y。后面的层就都是输出x,y,w,h了
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            # 保存每一层的预测
            outputs_classes.append(
                outputs_class
            )  # list(batch_size, num_queries, num_classes)
            outputs_coords.append(outputs_coord)  #  list(batch_size, num_queries, 4)
        outputs_class = torch.stack(
            outputs_classes
        )  # [num_decoder_layers, batch_size, num_queries, num_classes]
        outputs_coord = torch.stack(
            outputs_coords
        )  # [num_decoder_layers, batch_size, num_queries, 4]

        # 取最后一层作为输出
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        # 辅助损失，每一层都进行保存
        """
        [
            {   pred_logits: [batch_size, num_queries, num_classes],
                pred_boxes: [batch_size, num_queries, 4]
            }
        ...
        ]

        """
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        """
            两阶段 Encoder 的额外输出,还会返回 Encoder 阶段的粗略预测：
            enc_outputs_class: [batch_size, ∑(h*w), num_classes]
            enc_outputs_coord_unact: [batch_size, ∑(h*w), 4]
        """
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):

        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
