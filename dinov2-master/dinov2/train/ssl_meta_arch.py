# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import (
    get_fsdp_wrapper,
    ShardedGradScaler,
    get_fsdp_modules,
    reshard_fsdp_model,
)

from dinov2.models.vision_transformer import BlockChunk
import dinov2.distributed as distributed


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 仅在启用分布式(FSDP)时才创建 ShardedGradScaler 做fp16梯度缩放。
        # 单机单卡未走FSDP、模型为纯float32时不能创建它：否则 backprop_loss
        # 会用它放大 loss 后 backward，而 do_train 的单卡分支(fp16_scaler=None)
        # 不会 unscale，导致梯度被放大数万倍、训练发散/NaN。
        self.fp16_scaler = (
            ShardedGradScaler()
            if cfg.compute_precision.grad_scaler and distributed.is_enabled()
            else None
        )

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(
                f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}"
            )
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(
                f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}"
            )
            logger.info(
                f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}"
            )
            logger.info(
                f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}"
            )
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(
            f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}"
        )
        logger.info(
            f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}"
        )
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert (
                max(cfg.ibot.mask_ratio_min_max) > 0
            ), "please provide a positive mask ratio tuple for ibot"
            assert (
                cfg.ibot.mask_sample_probability > 0
            ), "please provide a positive mask probability for ibot"
            self.ibot_out_dim = (
                cfg.ibot.head_n_prototypes
                if self.ibot_separate_head
                else cfg.dino.head_n_prototypes
            )
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(
                    f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}"
                )
                logger.info(
                    f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}"
                )
                logger.info(
                    f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}"
                )
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(
            f"Student and Teacher are built: they are both {cfg.student.arch} network."
        )

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        # =====================================================================
        # DINOv2 单步训练：学生-教师(EMA) + DINO对比损失 + iBOT掩码预测损失
        #
        # 整体思路（DINO / iBOT 自监督）：
        #   - 同一张图做多次不同裁剪：2 个“全局裁剪”(大图) + n 个“局部裁剪”(小图)。
        #   - teacher(动量编码器) 只看全局裁剪，产出“目标”特征(不回传梯度)。
        #   - student 看全部裁剪，预测这些目标；通过让 student 逼近 teacher 来学习表征。
        #   - DINO 损失：对比 cls token（整图表征）。
        #   - iBOT 损失：在 teacher 随机掩码的 patch 上，让 student 还原 patch 表征(MIM)。
        # images: 一个 batch 的 collated 裁剪数据(字典，含 global/local crops、mask 等)。
        # {
        #     "collated_global_crops": collated_global_crops.to(
        #         dtype
        #     ),  # [n_global_crops * batch_size, C, H, W]。
        #     "collated_local_crops": collated_local_crops.to(
        #         dtype
        #     ),  # [n_local_crops * batch_size, C, h, w]
        #     "collated_masks": collated_masks,  # (n_global_crops * batch_size, grid_h * grid_w)
        #     "mask_indices_list": mask_indices_list,  # (masked_token的数量) 当前批次所有mask中展平之后，非0位置（masked_token）的索引
        #     "masks_weight": masks_weight,  # (n_global_crops * batch_size, grid_h * grid_w)
        #     "upperbound": upperbound,  #  当前batch所有特征图遮挡的masked token总数的作为保守上界，模型用它提前分配 masked-token buffer，避免运行时反复申请显存
        #     "n_masked_patches": torch.full(
        #         (1,), mask_indices_list.numel(), dtype=torch.long
        #     ),  #n_masked_patches = [masked_token的数量]
        # }
        # teacher_temp: 当前 step 的 teacher 温度(随训练退火)。
        # =====================================================================

        # DINO 固定用 2 个全局裁剪（A、B 互为“视图对”，用于交叉对比）
        n_global_crops = 2
        assert n_global_crops == 2
        # 局部裁剪数量（如 8 或 10），由配置决定
        n_local_crops = self.cfg.crops.local_crops_number

        # ---- 把 collated 数据搬上 GPU（non_blocking 异步拷贝）----
        global_crops = images["collated_global_crops"].cuda(
            non_blocking=True
        )  # 2 个全局裁剪
        local_crops = images["collated_local_crops"].cuda(
            non_blocking=True
        )  # n 个局部裁剪

        # iBOT 掩码相关：teacher 在 masked patch 上做预测，student 还原
        masks = images["collated_masks"].cuda(non_blocking=True)  # 掩码本身(0/1)
        mask_indices_list = images["mask_indices_list"].cuda(
            non_blocking=True
        )  # 被掩 patch 的展平索引
        n_masked_patches_tensor = images["n_masked_patches"].cuda(
            non_blocking=True
        )  # masked_token的数量
        n_masked_patches = mask_indices_list.shape[0]  # 本 batch 掩码 patch 总数
        upperbound = images["upperbound"]  # patch 数上界(用于预分配 buffer)
        masks_weight = images["masks_weight"].cuda(non_blocking=True)  # 各 mask 的权重

        # ---- 损失项数（用于把总损失归一化为“平均到每个对比项”）----
        # 局部裁剪 vs 全局裁剪：每个局部裁剪要和每个全局裁剪配对 -> n_local * n_global 项
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        # 全局裁剪之间：g 个全局视图两两配对(非对称) -> g*(g-1) = 2 项
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino  # 是否算 DINO(cls) 损失
        do_ibot = self.do_ibot  # 是否算 iBOT(patch) 损失

        # loss scales
        # 每个全局裁剪对应的 iBOT 损失只占 1/n_global_crops（两个全局视图平均）
        ibot_loss_scale = 1.0 / n_global_crops

        # ============================ Teacher 前向 ============================
        # teacher 是“动量编码器”(EMA 更新的 student 副本)。@torch.no_grad() 不计算梯度。
        # 关键：teacher 只看【全局裁剪】，作为“目标”。每个 batch 只跑一次。
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            # 用 teacher 主干得到全局裁剪的特征（2 个全局视图拼在一起前向）
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            # 取出 cls token（整图全局表征）。
            # 形状：(2*B, embed_dim)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            # [(B, embed_dim), (B, embed_dim)]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # 注意：逆序拼接 [视图B, 视图A] —— 这样 DINO 全局损失里“学生视图A 匹配 教师视图B”，
            # 即交叉对比（避免模型走捷径：同一视图自匹配）。
            # (2*B, embed_dim)
            teacher_cls_tokens = torch.cat(
                (teacher_cls_tokens[1], teacher_cls_tokens[0])
            )
            # 取出 patch token（用于 iBOT 的掩码 patch 预测）。
            # (2*B, num_patchs, embed_dim)
            ibot_teacher_patch_tokens = teacher_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            _dim = ibot_teacher_patch_tokens.shape[
                -1
            ]  # 最后一维 = embedding dimension 特征维度
            n_cls_tokens = teacher_cls_tokens.shape[0]  # 2 * B

            # ---- 处理 iBOT 目标：teacher 在【被掩码的 patch】上给目标表征 ----
            if do_ibot and not self.ibot_separate_head:
                # !cls 与 patch 共用一个 head（dino_head），所以拼到一个 buffer 一起过 head。
                # buffer 形状：(patch 总数上界 + cls token 总数, embedding dimension)
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                    upperbound + n_cls_tokens, _dim
                )
                # 前 cls token 总数 行写入 cls 表征，形状：(cls token 总数, embedding dimension)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                # 用 mask_indices_list 从展平的 patch 中选出被掩码的那些，填到 buffer 后面。
                # 展平后 patch 形状：(2*B*num_patchs, embed_dim)
                # 选出的被掩 patch 形状：(本 batch 被掩 patch 总数, embed_dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(
                        0, 1
                    ),  # (2*B*num_patchs, embed_dim)
                    dim=0,
                    index=mask_indices_list,  # (masked_token的数量)
                    out=buffer_tensor_teacher[
                        n_cls_tokens : n_cls_tokens + n_masked_patches
                    ],  # 把筛选结果就地写入这个张量
                )
                # 一次性过投影头；tokens_after_head 形状：(buffer 行数, 投影头输出维度=类别原型数)
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                # 拆分：cls 部分形状 (2*B, 投影头输出维度)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                # 被掩 patch 部分形状 (n_masked_patches, 投影头输出维度)
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                # !iBOT 用独立 head（ibot_head），cls 走 dino_head，patch 走 ibot_head。
                # buffer 形状：(patch 总数上界, embedding dimension)
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                    upperbound, _dim
                )
                # 选出被掩 patch，形状 (本 batch 被掩 patch 总数, embedding dimension)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                # !teacher_cls_tokens(2*B, embed_dim) 过 dino_head -> (2*B, 投影头输出维度)
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                # patch 过 ibot_head -> (patch 总数上界, iBOT 头输出维度)，取前 本 batch 被掩 patch 总数 行
                # !(n_masked_patches, iBOT 头输出维度)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(
                    buffer_tensor_teacher
                )[:n_masked_patches]
            else:
                # 不算 iBOT，只取 cls 过 head
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                masked_teacher_ibot_softmaxed_centered = None

            # ---- 对 teacher 输出做“软化 + 中心化”，得到最终监督目标 ----
            # centering：防止表征坍缩（所有样本聚成一类），用滑动平均的均值做中心化。
            if self.cfg.train.centering == "centering":
                # softmax(温度) 后再减去中心 -> 目标分布。
                # !结果 reshape 为 (n_global_crops, batch size, 投影头输出维度) —— 按全局视图分组
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.softmax_center_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:],
                    )
                )
                # 用本步输出更新中心（动量更新均值）
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = (
                        masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    )
                    # (1, n_masked_patches, iBOT 头输出维度)
                    masked_teacher_ibot_softmaxed_centered = (
                        self.ibot_patch_loss.softmax_center_teacher(
                            masked_teacher_patch_tokens_after_head[
                                :, :n_masked_patches
                            ],
                            teacher_temp=teacher_temp,
                        )
                    )
                    # !(n_masked_patches, iBOT 头输出维度)
                    masked_teacher_ibot_softmaxed_centered = (
                        masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    )
                    self.ibot_patch_loss.update_center(
                        masked_teacher_patch_tokens_after_head[:n_masked_patches]
                    )

            elif self.cfg.train.centering == "sinkhorn_knopp":
                # 另一种防坍缩方式：Sinkhorn-Knopp 把目标分布“均匀化”。
                # !结果同样 reshape 为 (n_global_crops, batch size, 投影头输出维度)
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.sinkhorn_knopp_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:],
                    )
                )
                if do_ibot:
                    # !(n_masked_patches, iBOT 头输出维度)
                    masked_teacher_ibot_softmaxed_centered = (
                        self.ibot_patch_loss.sinkhorn_knopp_teacher(
                            teacher_output=masked_teacher_patch_tokens_after_head,
                            teacher_temp=teacher_temp,
                            n_masked_patches_tensor=n_masked_patches_tensor,
                        )
                    )

            else:
                raise NotImplementedError

            # 返回：teacher 的 DINO 目标(按 2 个全局视图分组) + iBOT 目标(若启用)
            return (
                teacher_dino_softmaxed_centered_list,
                masked_teacher_ibot_softmaxed_centered,
            )

        # 执行 teacher 前向（不追踪梯度）
        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = (
            get_teacher_output()
        )
        # FSDP：teacher 前向临时 gather 出的完整参数，用完立即释放回分片，省显存
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop  # 累计总损失，最后统一反传

        # ============================ Student 前向 ============================
        # student 看【全部裁剪】（全局 + 局部）。传入的是“列表”[global, local]，
        # 因为两种裁剪分辨率不同，走之前讲过的 forward_features_list / NestedTensorBlock 变长路径。
        # masks=[masks, None]：全局裁剪做 iBOT 掩码，局部裁剪不过掩码。
        student_global_backbone_output_dict, student_local_backbone_output_dict = (
            self.student.backbone(
                [global_crops, local_crops], masks=[masks, None], is_training=True
            )
        )

        # 把要过投影头(dino_head)的张量收集成一个 list，稍后合并成一次高效前向
        inputs_for_student_head_list = []

        # 1a: 局部裁剪的 cls token（整图表征），加一维方便放进 list。
        # 形状：(n_local_crops * batch_size , embed_dim)-> unsqueeze 后 (1, n_local_crops * batch_size , embed_dim)
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: 全局裁剪的 cls token。
        # 形状：(n_global_crops × batch size, embed_dim) -> unsqueeze 后 (1, n_global_crops × batch size, embed_dim)
        student_global_cls_tokens = student_global_backbone_output_dict[
            "x_norm_clstoken"
        ]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: 全局裁剪的【被掩码 patch】token（iBOT 用）
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[
                -1
            ]  # 最后一维 = embed_dim
            # 形状：(n_global_crops * batch_size ,n_patches, embed_dim)
            ibot_student_patch_tokens = student_global_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            # 预分配 buffer，形状 (patch 总数上界, embed_dim)
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                upperbound, _dim
            )
            # !「按掩码索引取出来的被掩 patch 表征」，形状为 (n_masked_patches, embed_dim)；它会被写进一个形状为 (upperbound, embed_dim) 的 buffer 的前 n_masked_patches 行（剩余行是 0 填充）
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_student_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                # 与 cls 共用 dino_head：一起送进 head 列表
                inputs_for_student_head_list.append(
                    buffer_tensor_patch_tokens.unsqueeze(0)
                )
            else:
                # 用独立的 ibot_head 先算好
                # !(n_masked_patches, iBOT 头输出维度)
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_tensor_patch_tokens
                )[:n_masked_patches]

        # inputs_for_student_head_list = [
        # (1, n_local_crops * batch_size , embed_dim),  局部裁剪 cls
        # (1, n_global_crops × batch size, embed_dim),  全局裁剪 cls
        # (1, upperbound, embed_dim),  被掩 patch（按需）
        # ]
        # 2: 把 list 里的若干组 token 拼成一个“块对角”大张量，过一次 dino_head（高效）。
        #    from_tensor_list 构造 BlockDiagonalMask：各组之间互不注意力（这里只是借用其拼接/拆分）。
        #    cat_inputs 形状：(1, 三组 token 总数之和, embedding dimension)
        #      —— 三组依次为 局部裁剪 cls / 全局裁剪 cls / 被掩 patch（按需），沿序列维拼接。
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            inputs_for_student_head_list
        )
        # dino_head 输出形状 (1, 三组 token 总数之和, 投影头输出维度)；split 后 outputs_list 为
        # 与输入分组一一对应的张量列表，每组形状 (该组 token 数量, 投影头输出维度)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: 取出 局部裁剪 cls 过 head 后的结果（squeeze 去掉第 0 维）。
        # 形状：(局部裁剪数 × 原始 batch size, 投影头输出维度)
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: 取出 全局裁剪 cls 过 head 后的结果。
        # 形状：(两个全局裁剪数 × 原始 batch size, 投影头输出维度)
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: 取出 全局裁剪被掩码 patch 过 head 后的结果（仅共用 head 时在这取）。
        # 形状：(本 batch 被掩 patch 总数, 投影头输出维度)
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(
                0
            )[:n_masked_patches]

        # ============================ 计算各项损失 ============================
        # ---- DINO 损失（局部裁剪 vs 全局裁剪的 teacher）----
        # !局部裁剪数 n_local，每个都要和 2 个全局视图的 teacher 对比 -> n_local 项
        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                # 局部裁剪 cls 按 局部裁剪数 拆开，每个元素形状 (n_local_crops, 原始 batch size, 投影头输出维度) 作为 student 输出
                student_output_list=student_local_cls_tokens_after_head.chunk(
                    n_local_crops
                ),
                # teacher 目标形状 (两个全局裁剪数=2, 原始 batch size, 投影头输出维度)
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (
                n_global_crops_loss_terms + n_local_crops_loss_terms
            )  # 平均到每个对比项

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss（乘以 DINO 权重后累计进总损失）
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        # 全局裁剪成对处理，会有 2 个交叉对比项，故 loss_scales=2 做补偿
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # !compute loss（全局裁剪 student vs 全局裁剪 teacher，交叉匹配 A<->B）
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[
                        student_global_cls_tokens_after_head
                    ],  # 形状  (global_crops × batch size, 投影头输出维度)
                    teacher_out_softmaxed_centered_list=[
                        # flatten(0,1) 后形状 (global_crops × batch size, 投影头输出维度)；逆序拼接过，即 A<->B 交叉配对
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # 之前逆序拼接过，这里展平即 A<->B 交叉配对
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            # !KOLEO 损失：鼓励 cls 表征在球面上均匀分布（防坍缩的额外正则）
            # (n_global_crops × batch size, embed_dim)
            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # 同一张图的两种全局裁剪之间不算 koleo（避免无意义约束）
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # 仅用于显示时还原到之前的尺度

        if do_ibot:
            # ---- iBOT 损失（掩码 patch 预测）----
            # !student 在 masked patch 上的表征，要匹配 teacher 在同位置的“目标”表征
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,  # (n_masked_patches, iBOT 头输出维度)
                    masked_teacher_ibot_softmaxed_centered,  # (n_masked_patches, iBOT 头输出维度)
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # 用累计的总损失统一反传；FSDP 下 backprop_loss 内部处理梯度分片/同步
        self.backprop_loss(loss_accumulator)

        # FSDP 多流同步：把各 backbone/head 的 CUDA stream 对齐，避免异步读写错位
        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        # 单机单卡（未启用 FSDP）时，普通模块没有 _streams 属性，
        # 跳过即可，否则会触发 AttributeError。
        if not distributed.is_enabled():
            self.need_to_synchronize_fsdp_streams = False
            return
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = self.teacher.dino_head._streams = (
                self.student.backbone._streams
            ) = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                student_fsdp_modules = get_fsdp_modules(self.student[k])
                teacher_fsdp_modules = get_fsdp_modules(self.teacher[k])
                if student_fsdp_modules and teacher_fsdp_modules:
                    # 多卡 FSDP：按分片单元对齐 student/teacher 参数做 EMA
                    for ms, mt in zip(student_fsdp_modules, teacher_fsdp_modules):
                        student_param_list += ms.params
                        teacher_param_list += mt.params
                else:
                    # 单机单卡（未启用 FSDP）：普通模块没有分片单元，
                    # 直接用 .parameters() 做 EMA 更新，否则 teacher 永远不会更新。
                    student_param_list += list(self.student[k].parameters())
                    teacher_param_list += list(self.teacher[k].parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # 未启用分布式（单机单卡、或未初始化进程组）时，跳过 FSDP 包裹，
        # 直接用普通 nn.Module。原因：
        #   1) FSDP 构造时必须存在进程组，否则抛
        #      "Default process group has not been initialized"；
        #   2) Windows 仅支持 gloo 后端，而 gloo 在 Windows 上不支持 CUDA
        #      张量的集合通信，强行初始化进程组后 all_gather 仍会失败。
        # 单机单卡无需分片，普通模型即可正常运行前向/反向。
        if not distributed.is_enabled():
            logger.info(
                "Distributed disabled -- skip FSDP wrapping (single-device mode)"
            )
            return
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(
                student_model_cfg, modules_to_wrap={BlockChunk}
            )(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(
                teacher_model_cfg, modules_to_wrap={BlockChunk}
            )(self.teacher[k])
