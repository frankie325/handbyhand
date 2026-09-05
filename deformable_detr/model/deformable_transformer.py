import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.common import clones, inverse_sigmoid
from torch.nn.init import xavier_uniform_, constant_, normal_
import math
from transformers import PositionwiseFeedForward, MultiHeadedAttention


# 为输入添加位置编码，detr中给query和key添加位置编码，value不添加位置编码
def with_pos_embed(tensor: torch.Tensor, pos) -> torch.Tensor:
    return tensor if pos is None else tensor + pos


def ms_deform_attn_core_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    # value.shape = [batch_size, ∑(h*w), n_heads, d_k]
    # sampling_locations.shape = [batch_size,  ∑(h*w), n_heads, n_levels, n_points, 2]
    # attention_weights.shape = [batch_size,  ∑(h*w), n_heads, n_levels, n_points]
    batch_size, num_spatial_tokens, n_heads, d_k = value.shape
    _, num_query, n_heads, n_levels, n_points, _ = sampling_locations.shape

    # 按特征层拆分 [ (batch_size, h*w, n_heads, d_k), ... ]
    value_list = value.split(
        [feat_h * feat_w for feat_h, feat_w in value_spatial_shapes], dim=1
    )

    # 将预测坐标转换到 grid_sample 的范围 [-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    # 循环处理每个特征层
    for level_id, (feat_h, feat_w) in enumerate(value_spatial_shapes):

        # [batch, h*w, n_heads, d_k]
        # -> [batch, h*w, n_heads*d_k] -> [batch, n_heads*d_k, h*w]
        # -> [batch*n_heads, d_k, feat_h, feat_w]
        value_level = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * n_heads, d_k, feat_h, feat_w)
        )
        # [batch, ∑(h*w), n_heads, n_points, 2]
        # -> [batch, n_heads, ∑(h*w), n_points, 2]
        # -> [batch*n_heads, ∑(h*w), n_points, 2]
        sampling_grid_level = (
            sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        )

        # !双线性插值采样，得到每个预测点位对应的特征值
        # [batch*n_heads, d_k, ∑(h*w), n_points]
        sampling_value_level = F.grid_sample(
            value_level,  # [batch*n_heads, d_k, feat_h, feat_w]映射回原特征图
            sampling_grid_level,  # [batch*n_heads, ∑(h*w), n_points, 2] 最后一维是必须是 (x, y)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # sampling_value_level.shape = [batch*n_heads, d_k, ∑(h*w), n_points] n_points就是通过双线性插值获取的原特征图的像素值
        sampling_value_list.append(sampling_value_level)
    # [batch, ∑(h*w), n_heads, n_levels, n_points]
    # -> [batch, n_heads, ∑(h*w), n_levels*n_points]
    # -> [batch*n_heads, 1, ∑(h*w), n_levels*n_points]
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * n_heads, 1, num_query, n_levels * n_points
    )

    # torch.stack(sampling_value_list, dim=-2) -> [batch*n_heads, d_k, ∑(h*w), n_levels, points]
    # flatten(-2) -> [batch*n_heads, d_k, ∑(h*w), n_levels*points]
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)  # 将n_levels*points加权求和 [batch*n_heads, d_k, ∑(h*w)]
        .view(batch_size, n_heads * d_k, num_query)
    )  # 输出 [batch_size, d_model( n_heads * d_k), ∑(h*w)]

    # !最终输出 [batch_size, ∑(h*w), d_model( n_heads * d_k)]
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels  # 多尺度
        self.n_heads = n_heads  # 多头
        self.n_points = n_points  # 预测的偏移点位
        self.d_k = d_model // n_heads

        self.value_proj = nn.Linear(d_model, d_model)  # 将输入的特征映射成value

        # 为每个特征预测n_points个偏移量，多头多尺度，n_points个点位，每个点位有2个偏移量 dx, dy
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # 最后再经过一个线性层
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )  # thetas.shape = [n_heads]
        grid_init = torch.stack(
            [thetas.cos(), thetas.sin()], -1
        )  # shape = [n_heads, 2] 8个头对应8个方向
        grid_init = (
            (
                grid_init / grid_init.abs().max(-1, keepdim=True)[0]
            )  # 保持维度不变，进行归一化 [n_heads, 2]
            .view(self.n_heads, 1, 1, 2)  # [n_heads,1, 1, 2]
            .repeat(
                1, self.n_levels, self.n_points, 1
            )  # [n_heads, n_levels, n_points, 2]
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (
                i + 1
            )  # 每个点位偏移量都不同，i+1 表示偏移量越大，点位越远
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))  # [256]
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        src_flatten,
        spatial_shapes,
        level_start_index,
        mask_flatten,  # [batch_size, ∑(h*w)]
    ):
        N, len_q, _ = query.shape
        N, len_in, _ = src_flatten.shape

        # 检查多尺度特征图所有空间位置的总数，是否等于 src_flatten 中的特征数量
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == len_in
        value = self.value_proj(src_flatten)  # [batch_size, ∑(h*w), d_model]
        if mask_flatten is not None:
            # padding位置都设置为0
            value = value.masked_fill(mask_flatten[..., None], float(0))

        value = value.view(
            N, len_in, self.n_heads, self.d_k
        )  # [batch_size, ∑(h*w), n_heads, d_k]

        # 预测偏移量 [batch_size, len_q, n_heads, n_levels, n_points, 2]
        sampling_offset = self.sampling_offsets(query).view(
            N, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # 注意力权重 [batch_size, len_q, n_heads, n_levels*n_points]
        attention_weights = self.attention_weights(query).view(
            N, len_q, self.n_heads, self.n_levels * self.n_points
        )
        # 同一个 query、同一个 head 的所有采样权重之和为 1
        # [batch_size, len_q, n_heads, n_levels, n_points]
        attention_weights = F.softmax(attention_weights, -1).view(
            N, len_q, self.n_heads, self.n_levels, self.n_points
        )

        # reference_points.shape = [batch_size, ∑(h*w), n_levels, 4或2]
        if reference_points.shape[-1] == 2:

            # 将spatial_shapes最后一位的H,W对调下位置 -> W,H  因为预测值是dx, dy
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )  # [n_levels, 2]

            # reference_points[:, :, None, :, None, :]会广播到 [batch_size,  ∑(h*w), n_heads, n_levels, n_points, 2]
            # offset_normalizer[None, None, None, :, None, :]会广播到 [batch_size,  ∑(h*w), n_heads, n_levels, n_points, 2]
            # 先对偏移量进行归一化，再加到reference_points得到预测4个点位的归一化坐标
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offset / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:

            # 抵消初始化sampling_offsets时"半径 = 1..K"带来的尺度放大，让 (w, h) × 0.5 这个调制因子能直接按"框的半宽半高"来解释
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offset
                / self.n_points
                * reference_points[
                    :, :, None, :, None, 2:
                ]  # 乘上 (w, h) 之后，偏移变成相对框的比例
                * 0.5
            )
        else:
            raise ValueError(
                "reference_points 的最后一维必须是 2（中心点 cxcy）或 4（边界框 cxcywh），"
                f"当前为 {reference_points.shape[-1]}"
            )
        # [batch_size, ∑(h*w), d_model]
        output = ms_deform_attn_core_pytorch(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
        )

        return self.output_proj(output)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forawrd = PositionwiseFeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        src_flatten,
        level_pos_embed_flatten,
        reference_points,
        spatial_shapes,
        level_start_index,
        mask_flatten,
    ):
        src2 = self.self_attn(
            with_pos_embed(src_flatten, level_pos_embed_flatten),  # query融合位置编码
            reference_points,
            src_flatten,
            spatial_shapes,
            level_start_index,
            mask_flatten,
        )  # [batch_size,  ∑(h*w), d_model]

        # 跨层连接
        src = src_flatten + self.dropout(src2)
        src = self.norm1(src)

        # 前馈层（PositionwiseFeedForward 内部不含残差，需要在这里补上）
        src2 = self.feed_forawrd(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src  # [batch_size,  ∑(h*w), d_model]


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # spatial_shapes = [(28, 40), (14, 20), (7, 10), (4, 5)]
        # valid_ratios.shape = [batch_size, 4, 2]
        reference_points_list = []
        for level, (h, w) in enumerate(spatial_shapes):
            # 高度方向采样点[0.5, 1.5, 2.5, ..., h-0.5] 总计h个采样点
            h_points = torch.linspace(
                0.5, h - 0.5, h, dtype=torch.float32, device=device
            )
            # 宽度方向采样点[0.5, 1.5, 2.5, ..., w-0.5] 总计w个采样点
            x_points = torch.linspace(
                0.5, w - 0.5, w, dtype=torch.float32, device=device
            )

            # ref_y.shape = [h, w] 每个值表示该网格中心在高度方向上的坐标
            # ref_x.shape = [h, w] 每个值表示该网格中心在宽度方向上的坐标
            ref_y, ref_x = torch.meshgrid(h_points, x_points, indexing="ij")

            # ref_y.reshape(-1)[None] = [h, w] -> [1, h*w]
            # valid_ratios[:, None, level, 1] 分开理解：先取出对应特征图的有效区域高度比率变成 [batch_size],None然后再加一维 [batch_size, 1]
            # valid_ratios[:, None, level, 1] * h 表示特征图真实的有效高度
            # 相除进行广播 -> [batch_size, h*w]
            # 也就是每个高度方向上的坐标相对于有效区域进行归一化处理，并且模型可以通过坐标大于 1 识别出它位于有效区域之外
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * h)
            # ref_x同理
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * w)
            # ref.shape = [batch_size, h*w, 2]
            # h*w可以理解为将网格水平方向进行展开，每个网格包含两个归一化坐标（x, y）
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(
            reference_points_list, 1
        )  # [batch_size, ∑(h*w), 2]
        """
        reference_points[:, :, None] -> [batch_size, ∑(h*w), 1, 2]
        valid_ratios[:, None] -> [batch_size, 1, 4, 2]
        相乘进行广播 -> [batch_size, ∑(h*w), 4, 2]
        比如说要映射的特征图是4*4，其中3*3是有效区域
        4 * 3/4（有效区域比率） * 有效区域的归一化值 = 采样点在4*4的参考坐标
        !参考点的坐标会限制在有效区域内
        """
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        """
        [batch_size, ∑(h*w), 4, 2]
        B：每张图片
        ∑(h*w)：每个 Encoder 特征点
        4：特征层数
        2：(x, y) 坐标
        含义：每个 Encoder token 都有一组对应所有特征层的参考坐标
        """
        return reference_points

    def forward(
        self,
        src_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        level_pos_embed_flatten,
        mask_flatten,
    ):
        output = src_flatten
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, src_flatten.device
        )

        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                level_pos_embed_flatten,
                reference_points,
                spatial_shapes,
                level_start_index,
                mask_flatten,
            )
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈层的 dropout 与 LayerNorm，FFN 本体复用 feed_forawrd
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.feed_forawrd = PositionwiseFeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
    ):
        # tgt.shape = [batch_size, num_queries, d_model]  编码器预测的object_queries
        # query_pos.shape = [batch_size, num_queries, d_model] 作为解码器的object_queries位置编码
        # reference_points.shape = [batch_size, num_queries, n_levels, 4或者2]
        # src.shape = [batch_size,  ∑(h*w), d_model] 编码器的输出
        # src_spatial_shapes = [(28, 40), (14, 20), (7, 10), (4, 5)]
        # src_level_start_index = [0, 28*40, 28*40+14*20, 28*40+14*20+7*10]
        # src_padding_mask.shape = [batch_size,  ∑(h*w)] 编码器输出的memory掩码

        q = k = with_pos_embed(tgt, query_pos)  # [batch_size, num_queries, d_model]

        # !解码器的自注意力计算不需要掩码
        tgt2 = self.self_attn(q, k, tgt)  # [batch_size, num_queries, d_model]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 交叉注意力计算
        # !query 同样要带上 query_pos：可变形注意力的采样点与注意力权重都由 query 预测，
        # !没有位置编码的话每个 query 缺少"我负责图上哪块区域"的身份信息，所有 query 会退化成同一份预测
        tgt2 = self.cross_attn(
            with_pos_embed(tgt, query_pos),  # [batch_size, num_queries, d_model]
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )  # [batch_size, num_queries, d_model]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 前馈层（PositionwiseFeedForward 内部不含残差，需要在这里补上）
        tgt2 = self.feed_forawrd(tgt)  # [batch_size, num_queries, d_model]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt  # [batch_size, num_queries, d_model]


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        # tgt,
        # reference_points,
        # memory,
        # spatial_shapes,
        # level_start_index,
        # valid_ratios,
        # query_embed,
        # mask_flatten,

    """
    根据当前 reference point 从多尺度特征中采样；
    更新 query 特征；
    预测并修正边界框；
    将修正后的框作为下一层的 reference point。
    """

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
    ):

        # tgt.shape = [batch_size, num_queries, d_model] 编码器预测的object_queries
        # reference_points.shape = [batch_size, num_queries, 4或者2]
        # src.shape = [batch_size,  ∑(h*w), d_model] 编码器的输出
        # src_spatial_shapes = [(28, 40), (14, 20), (7, 10), (4, 5)]
        # src_level_start_index = [0, 28*40, 28*40+14*20, 28*40+14*20+7*10]
        # src_valid_ratios.shape = [batch_size, n_levels, 2] 每个预测特征图的有效尺寸
        # query_pos.shape = [batch_size, num_queries, d_model] 作为解码器的object_queries位置编码
        # src_padding_mask.shape = [batch_size,  ∑(h*w)] 编码器输出的memory掩码

        # !在两阶段模型中，tgt 来自 Encoder 筛选出的 Top-K proposal；
        # !在一阶段模型中，tgt 来自可学习的 query embedding。
        output = tgt
        # 每层的 query 特征
        intermediate = []
        # 每层更新后的 reference points
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # 两阶段
            if reference_points.shape[-1] == 4:

                # reference_points归一化的参考范围是“有效区域”，相对于有效区域坐标 * 有效区域占特征图的比例 = 相对于整个特征图坐标
                # [batch_size, num_queries, 4] -> [batch_size, num_queries, 1, 4]
                # cat -> [batch_size, n_levels, 4] ->[:,None]->[batch_size, 1, n_levels, 4]
                # 广播 [batch_size, num_queries, n_levels, 4]
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                # 同理  [batch_size, num_queries, n_levels, 2]
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )  # [batch_size, num_queries, d_model]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                # 用 bbox_embed[lid] 预测修正量，更新 reference_points，每一层都在前一层的框基础上再修一点
                tmp = self.bbox_embed[lid](output)  # [batch_size, num_queries, 4]
                if reference_points.shape[-1] == 4:
                    # 先反转 sigmoid 得到修正量，再加到 reference_points 上
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    # 再归一化
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    """
                    单阶段在第 1 层之后，reference_points 就从 2 维变成了 4 维。
                    之后循环进入 shape[-1] == 4 分支，走和两阶段完全相同的路径。
                    这个"2D 中心点补全成 4D 框"的转变是单阶段第一次精修时自动完成的。

                    单阶段初始参考点只有中心点 (x, y)，没有宽高，所以：
                    前 2 维（中心点）：加上参考点的逆 sigmoid，做增量精修 → 有基准
                    后 2 维（宽高）：tmp 的预测值直接保留 → 没有基准，等于从零预测
                    """
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    # 新参考点 = 预测偏移 + 参考点
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                # detach 之后，每一层的框预测只对自己那一层的 aux loss 负责，层与层之间解耦
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # 中间层的结果全部返回
        if self.return_intermediate:

            # [num_decoder_layers, batch_size, num_queries, d_model]
            # [num_decoder_layers, batch_size, num_queries, 4]
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        # output.shape = [batch_size, num_queries, d_model]
        # reference_points.shape = [batch_size, num_queries, 4]
        return output, reference_points


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=4,
        enc_n_points=4,
        dec_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
    ):
        super().__init__()
        self.d_model = d_model  # get_proposal_pos_embed 需要使用
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.two_stage = two_stage
        self.two_stage_num_proposals = (
            two_stage_num_proposals  # 两阶段筛选出的候选框数量
        )
        # !在多尺度特征中，位于不同特征层的特征点可能拥有相同的(h,w)坐标，他们的位置编码是一样的，使用level_embed区分特征层，告诉模型位置编码对应第几层特征图
        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model)
        )  # [4, 256]

        # 编码器：先用单层模板，再 clones 出 num_encoder_layers 层
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        # 解码器：同理 clones 出 num_decoder_layers 层
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        # return_intermediate_dec=True 时返回所有中间层输出，供 aux loss 使用
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec
        )

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        level_embed 用 nn.Parameter(torch.Tensor(...)) 创建，内容是未初始化的脏内存，必须显式重置。
        """
        # 先把所有 2 维及以上的参数做 xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        # !上面的 xavier 会覆盖 MSDeformAttn 里精心设计的初始化（attention_weights 置 0、
        # !sampling_offsets 按 8 个方向 + 1..K 半径排布），必须再跑一遍把它们改回来
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        # 多尺度层级编码用正态分布初始化
        normal_(self.level_embed)

    """
    会计算每个特征层中有效区域的比例
    valid_ratio_w = 有效宽度 / 总宽度
    valid_ratio_h = 有效高度 / 总高度
    """

    def get_valid_ratio(self, mask):
        # mask.shape =[batch_size, h, w]
        _, H, W = mask.shape
        # !padding的像素点为0，
        valid_H = torch.sum(~mask[:, :, 0], 1)  # 在高度方向上求和 [batch_size]
        valid_W = torch.sum(~mask[:, 0, :], 1)  # 在宽度方向上求和 [batch_size]
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # [batch_size, 2]
        return valid_ratio

    # ?两阶段 Deformable DETR 中由 Encoder 特征生成的初始候选框,给 Decoder 提供初始位置参考
    # ?两阶段 Deformable DETR 是从 Faster R-CNN 的 RPN 借来的思想：先用第一阶段的稠密预测筛出少量高质量候选，再交给第二阶段精修
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        # memory.shape = [batch_size, ∑(h*w), d_model]
        # memory_padding_mask.shape = [batch_size, ∑(h*w)]
        batch_size, seq_len, d_model = memory.shape
        base_scale = 4.0
        current = 0  # 用来从序列中取出特征层
        proposals = []
        for level, (height, width) in enumerate(spatial_shapes):
            # 从序列中取出当前特征层并还原成特征图的形状
            mask_flatten = memory_padding_mask[
                :, current : (current + height * width)
            ].view(batch_size, height, width, 1)
            # [batch_size, h, w, 1]

            valid_H = torch.sum(
                ~mask_flatten[:, :, 0, 0], 1
            )  # 计算有效区域高度 [batch_size]
            valid_W = torch.sum(
                ~mask_flatten[:, 0, :, 0], 1
            )  # 计算有效区域宽度 [batch_size]

            # 为每个位置生成中心坐标
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0,
                    height - 1,
                    height,
                    dtype=torch.float32,
                    device=memory.device,
                ),  # [0, 1, 2, ..., height-1]
                torch.linspace(
                    0,
                    width - 1,
                    width,
                    dtype=torch.float32,
                    device=memory.device,
                ),  # [0, 1, 2, ..., width-1]
                indexing="ij",
            )
            # grid_y.shape = [h, w]
            # grid_x.shape = [h, w]
            # 构建成网格坐标[h, w, 2]
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [h, w, 2]
            # [batch_size, 2(valid_w,valid_h)] -> [batch_size, 1, 1, 2]
            scale = torch.stack([valid_W, valid_H], -1).view(batch_size, 1, 1, 2)
            # 除以有效区域将grid进行归一化
            # [batch_size, h, w, 2]
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            # 初始化宽高，不同特征层的宽高不同，通过level进行缩放
            # level越大，特征图尺寸越小，宽高越大
            # !指数缩放：level=0 时倍率为 1，若写成 2.0 * level 则 level=0 的宽高会变成 0，全部被判为无效
            wh = torch.ones_like(grid) * 0.05 * (2.0**level)  # [batch_size, h, w, 2]
            # 拼接中心坐标和宽高: [batch_size, h, w, 4(x,y,w,h)] ->[batch_size, h*w, 4]
            proposal = torch.cat((grid, wh), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            current += height * width

        # [batch_size, ∑(h*w), 4]
        output_proposals = torch.cat(proposals, 1)
        # 中心或宽高太接近边界，就认为它无效
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(
            -1, keepdim=True
        )  # [batch_size, ∑(h*w), 1]
        # 反Sigmoid，将归一化坐标还原成原始坐标
        output_proposals = torch.log(
            output_proposals / (1 - output_proposals)
        )  # [batch_size, ∑(h*w), 4]

        # 添加掩码
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )  # [batch_size, ∑(h*w), 4]

        # 无效位置也添加掩码
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )  # [batch_size, ∑(h*w), 4]

        output_memory = memory  # [batch_size, ∑(h*w), d_model]
        # 掩码设置为0，防止下面的线性计算得出nan
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )

        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

        # 本质上就是一个 RPN 区域候选网络，接一个独立的轻量检测头，而不是直接拿 encoder 的输出去预测
        # !这里走 Linear 生成新张量，也避免了修改 memory 这个非叶子节点触发 autograd 的 modified by an inplace operation 报错
        output_memory = self.enc_output_norm(
            self.enc_output(output_memory)
        )  # [batch_size, ∑(h*w), d_model]

        # output_memory.shape = [batch_size, ∑(h*w), d_model]
        # output_proposals.shape = [batch_size, ∑(h*w), 4]
        return output_memory, output_proposals

    """
    候选框的几何 (x, y, w, h)进行位置编码
    目的是把 top-k 个 proposal 的框信息，编码成 decoder 的初始 query
    decoder 的 query 不再是随机初始化的（单阶段 DETR 那样），
    而是由图像内容驱动生成、且携带明确几何先验的
    每个 query 一上来就知道"我负责 (x,y,w,h) 这个框"。所以两阶段收敛更快、小目标更好
    """

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128):
        # proposals.shape = [batch_size, topk, 4]
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )  # [num_pos_feats]
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # 缩放到[0, 2*pi]   [batch_size, topk, 4]
        proposals = proposals.sigmoid() * scale
        # [batch_size, topk, 4, num_pos_feats]
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        """
        sin/cos 交错后展平
        [ x 的 128 维 | y 的 128 维 | w 的 128 维 | h 的 128 维 ]
            ↑ 每 128 维内部是 sin(θ₀),cos(θ₀),sin(θ₁),cos(θ₁),... 交错
        """
        # [batch_size, topk, 4, num_pos_feats/2, 2] -> [batch_size, topk, 4*num_pos_feats/2*2]
        return pos

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # !h, w是经过下采样的特征图尺寸，每个特征图都不一样
        # 单个src.shape = [batch_size, d_model, h, w]
        # 单个mask.shape = [batch_size, h, w]
        # 单个pos_embed.shape = [batch_size, d_model, h, w]
        # query_embed.shape = [batch_size, num_queries, d_model]
        src_flatten = []
        mask_flatten = []
        level_pos_embed_flatten = []
        spatial_shapes = []

        # !三个列表一一对应，必须一起遍历：特征图、掩码、位置编码
        for level, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(
                1, 2
            )  # 展平特征图 [batch_size, h*w, d_model]
            mask = mask.flatten(1)  # 展平mask [batch_size, h*w]
            pos_embed = pos_embed.flatten(2).transpose(
                1, 2
            )  # 展平位置编码 [batch_size, h*w, d_model]

            # self.level_embed[level].view(1, 1, -1): [d_model] -> [1, 1, d_model] ->广播-> [batch_size, h*w, d_model]
            level_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            level_pos_embed_flatten.append(level_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # 沿着h*w维度进行拼接
        src_flatten = torch.cat(src_flatten, 1)  # [batch_size,   ∑(h*w), d_model]
        mask_flatten = torch.cat(mask_flatten, 1)  # [batch_size, ∑(h*w)]
        level_pos_embed_flatten = torch.cat(
            level_pos_embed_flatten, 1
        )  # [batch_size, ∑(h*w), d_model]

        """
        spatial_shapes: 4个特征图的尺寸
        假设为：
        spatial_shapes = [
            (28, 40),
            (14, 20),
            (7, 10),
            (4, 5),
        ]
        """
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )

        """
            spatial_shapes.new_zeros((1, )) = [0] 输出一个为0的标量
            spatial_shapes.prod(1) = [28*40, 14*20, 7*10, 4*5] 按照第1维计算乘积
            .cumsum(0)累加求和，[1120, 1400, 1470, 1490]
            [:-1] 去掉最后一个 [1120, 1400, 1470]
            torch.cat(([0], [1120, 1400, 1470])) = [0, 1120, 1400, 1470]
            !level_start_index的作用就是记住各层特征图在这个长序列中的起始位置
        """
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        valid_ratios = torch.stack(
            [self.get_valid_ratio(mask) for mask in masks], 1
        )  # [batch_size, n_levels, 2]

        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            level_pos_embed_flatten,
            mask_flatten,
        )  # [batch_size,  ∑(h*w), d_model]
        bs, _, c = memory.shape

        if self.two_stage:
            # !class_embed / bbox_embed 需要各 num_decoder_layers + 1 个（多出来的 1 个给 Encoder 阶段打分用），
            # !它们依赖 num_classes，按官方设计由 DeformableDetr 建好后注入到 decoder 上
            assert (
                self.decoder.class_embed is not None
                and self.decoder.bbox_embed is not None
            ), (
                "两阶段模式需要外部注入 decoder.class_embed 和 decoder.bbox_embed，"
                f"各需 {self.num_decoder_layers + 1} 个"
            )
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            # output_memory.shape = [batch_size, ∑(h*w), d_model] 编码器输出
            # output_proposals.shape = [batch_size, ∑(h*w), 4] 编码器预测的候选框

            # 实现两阶段 Deformable DETR
            # [batch_size, ∑(h*w), num_classes]
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
                output_memory
            )
            # batch_size, ∑(h*w), 4]，4是x,y,w,h，预测的原始坐标
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
                + output_proposals
            )

            # 需要筛选出成为query的数量
            topk = self.two_stage_num_proposals
            # 从类别维度中取出第 0 个类别的分数，用这个分数给 525 个候选位置排序
            # enc_outputs_class[..., 0]取出第0个类别的分数 [batch_size, ∑(h*w)]
            # topk取出分数最高的 topk 个位置
            # torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1] 表示只取 torch.topk 返回结果中的索引
            topk_proposals_indices = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[
                1
            ]  # [batch_size, topk]

            # 取出前topk个位置的坐标
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact,
                1,
                topk_proposals_indices.unsqueeze(-1).repeat(1, 1, 4),
            )  # [batch_size, topk, 4]
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = (
                topk_coords_unact.sigmoid()
            )  # 归一化 # [batch_size, topk, 4]
            init_reference_out = reference_points

            # !通过线性层，将位置特征转换成 Decoder 所需的 query 表示
            # [batch_size, topk, d_model * 2]
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(
                    self.get_proposal_pos_embed(topk_coords_unact, self.d_model // 2)
                )
            )

            # 沿着第2维度进行分割，分成query_embed和tgt
            # !query_embed作为解码器的位置编码，回答"我是谁、我负责图上哪块区域"
            # !tgt作为解码器的query，逐层更新，是累积出来的"内容/语义"表示回答"我这块区域里有什么东西
            # query_embed.shape = [batch_size, topk, d_model]
            # tgt.shape = [batch_size, topk, d_model]
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # 单阶段Deformable DETR
            # 单阶段直接用query_embed作为解码器的位置编码，回答"我是谁、我负责图上哪块区域"
            # query_embed.shape = [num_queries, d_model*2] -> [num_queries, d_model]
            # tgt.shape = [num_queries, d_model]
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(
                bs, -1, -1
            )  # [batch_size, num_queries, d_model]
            tgt = tgt.unsqueeze(0).expand(
                bs, -1, -1
            )  # [batch_size, num_queries, d_model]

            # 用 query_embed 预测初始参考点 [batch_size, num_queries, 2] 作为decoder的初始参考点
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        # hs.shape = [batch_size, num_queries, d_model] 或者是所有中间层的输出 [num_decoder_layers, batch_size, num_queries, d_model]
        # inter_references.shape = [batch_size, num_queries, 4] 或者是所有中间层的输出 [num_decoder_layers, batch_size, num_queries, d_model]
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
        )

        inter_references_out = inter_references
        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        return hs, init_reference_out, inter_references_out, None, None


def build_deforamble_transformer(
    hidden_dim,
    n_heads,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout,
    activation,
    num_feature_levels,
    enc_n_points,
    dec_n_points,
    two_stage,
    num_queries,
):
    return DeformableTransformer(
        d_model=hidden_dim,
        nhead=n_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        enc_n_points=enc_n_points,
        dec_n_points=dec_n_points,
        two_stage=two_stage,
        two_stage_num_proposals=num_queries,
    )
