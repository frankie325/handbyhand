import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from detr.utils.bos_ops import box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
        }
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)  # 92 个 1
        empty_weight[-1] = self.eos_coef  # 最后一位（背景）设为 0.1
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        # indices: list[ (prediction_indices, target_indices) ] 每个样本配对的预测框索引和匹配框索引
        # prediction_indices: [0,1] target_indices: [1,3] 表示query0-第3个真实框配对， query1-第4个真实框配对

        # 输出为[0,0,1,1,2,2,2,...]  0,0 表示第一个样本有两个预测框, 1,1表示第二个样本有两个预测框, 2,2,2表示第三个样本有三个预测框
        batch_idx = torch.cat(
            [
                torch.full_like(prediction_indices, i)
                for i, (prediction_indices, _) in enumerate(indices)
            ]
        )
        # 输出[样本0的第一个预测框索引, 样本0的第二个预测框索引, 样本1的第一个预测框索引, 样本1的第二个预测框索引, 样本2的第一个预测框索引, 样本2的第二个预测框索引, 样本2的第三个预测框索引]
        src_idx = torch.cat([prediction_indices for (prediction_indices, _) in indices])

        # 两个张量组合起来可以直接定位pred_logits: pred_logits[batch_indices, query_indices]
        return batch_idx, src_idx

    # 计算分类交叉熵，包括 no-object
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        # indices: list[ (prediction_indices, target_indices) ] 每个样本配对的预测框索引和匹配框索引
        # prediction_indices: [0,1] target_indices: [1,3] 表示query0-第3个真实框配对， query1-第4个真实框配对

        src_logits = outputs["pred_logits"]  # [batch_size, 100, num_classes + 1]
        idx = self._get_src_permutation_idx(indices)

        # target["labels"] = [真实框0, 真实框1, 真实框2, ....]
        # 通过高级索引拿到匈牙利算法匹配的真实框 target["labels"][target_indices] = [真实框0, 真实框2]
        # 然后将每个样本的真实框拼接成一个数组张量
        target_classes_output = torch.cat(
            [
                target["labels"][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ]
        )
        """
        把全部 query 标记为 no-object, target_classes: [batch_size, 100]
        举例：batch_size = 2, num_queries = 5
        target_classes = tensor([
            [91, 91, 91, 91, 91],  # 图片0
            [91, 91, 91, 91, 91],  # 图片1
        ])
        """
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,  # 填充值为91，即no-object
            dtype=torch.int64,
            device=src_logits.device,
        )

        """
        batch_idx = tensor([0, 0, 1])
        src_idx = tensor([1, 4, 0])
        target_classes_output = tensor([3, 18, 2])
        等价于逐个执行：
        target_classes[0, 1] = 3
        target_classes[0, 4] = 18
        target_classes[1, 0] = 2
        最终得到
        target_classes = tensor([
            [91,  3, 91, 91, 18],
            [ 2, 91, 91, 91, 91],
        ])
        91都是表示未匹配的query
        """
        target_classes[idx] = target_classes_output

        # CrossEntropy第一个维度必须是类别数，empty_weight使用加权分类损失
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce * self.weight_dict["loss_ce"]}
        return losses

    # 计算预测框-真实框的损失，只计算配对成功的
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)

        # 拿到配对预测框的坐标
        src_boxes = outputs["pred_boxes"][idx]  # [所有样本的匹配上的query, 4]

        # [所有样本的匹配的真实框, 4]
        target_boxes = torch.cat(
            [
                target["boxes"][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ]
        )

        losses = {"loss_bbox": None, "loss_giou": None}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses["loss_bbox"] = (loss_bbox.sum() / num_boxes) * self.weight_dict[
            "loss_bbox"
        ]

        # generalized_box_iou会计算所有每个预测框和每个真实框的iou，然后取对角线上的值，即配对预测框-真实框的iou
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )

        losses["loss_giou"] = (loss_giou.sum() / num_boxes) * self.weight_dict[
            "loss_giou"
        ]

        return losses

    def get_loss(self, outputs, targets, indices, num_boxes):
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        return losses

    def forward(self, outputs, targets):
        """
        outputs:{
            "pred_logits": output_classes[-1],  # [batch_size, 100, num_classes + 1]
            "pred_boxes": output_boxes[-1],  # [batch_size, 100, 4]
            # 每一层decoder的输出
            "aux_outputs": [
                {
                    "pred_logits": output_class,  # [batch_size, 100, num_classes + 1]
                    "pred_boxes": output_box,  # [batch_size, 100, 4]
                }
            ],
        }
        targets: list[ { 'labels': [num_objects], 'boxes': [num_objects, 4]} ]
        """

        outputs_without_aux = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }

        # indices: list[ (prediction_indices, target_indices) ]
        indices = self.matcher(outputs_without_aux, targets)
        # num_boxes: 一批中所有目标框的数量
        num_boxes = sum(len(target["labels"]) for target in targets)
        # 转成Tensor
        num_boxes = torch.tensor(
            [num_boxes],
            dtype=torch.float32,
            device=outputs["pred_logits"].device,
        )
        # 避免后面出现：0除以0
        num_boxes = max(num_boxes.item(), 1)

        losses = self.get_loss(outputs, targets, indices, num_boxes)

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                l_dict = self.get_loss(aux_outputs, targets, indices, num_boxes)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        """
            输出：
            {
                # 最后一层
                "loss_ce": ...,
                "loss_bbox": ...,
                "loss_giou": ...,

                # Decoder中间层0
                "loss_ce_0": ...,
                "loss_bbox_0": ...,
                "loss_giou_0": ...,

                # Decoder中间层1
                "loss_ce_1": ...,
                "loss_bbox_1": ...,
                "loss_giou_1": ...,

                # Decoder中间层2
                "loss_ce_2": ...,
                "loss_bbox_2": ...,
                "loss_giou_2": ...,
            }
        """
        return losses
