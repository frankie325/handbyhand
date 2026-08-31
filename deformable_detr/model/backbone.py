import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torch
from ..config import MODELS_DIR

# 设置预训练模型保存路径
torch.hub.set_dir(str(MODELS_DIR))


class Backbone(nn.Module):
    def __init__(
        self,
        pretrained_backbone=True,  # 是否加载backbone的预训练权重
        train_backbone=True,  # 是否冻结backbone的权重，冻结了则backbone不参与训练
    ) -> None:
        super(Backbone, self).__init__()

        # 使用ResNet50作为主干网络，将图像下采样32倍
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        resnet = resnet50(weights=weights)
        self.num_channels = [512, 1024, 2048]
        # ResNet 里的 avgpool 和 fc 会把空间维度压成一个向量，DETR 需要
        # 保留 H_feature × W_feature 的网格，所以这里只取卷积部分。
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if not train_backbone:
            # 冻结 backbone 参数，不参与训练
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def get_mask(self, padding_mask, size):
        if padding_mask is not None:
            # ?最近邻插值法：https://www.bilibili.com/video/BV1BM4y1t7tf/?spm_id_from=333.337.search-card.all.click&vd_source=23a9782e838dd1b9be9e5295dd2ccf9e
            # mask 是离散的 bool 值，使用 最近邻插值法将padding_mask缩小到[height/32, width/32]的尺寸，像素值避免产生 0.3、0.7的小数
            # 之类没有明确含义的中间值。
            feature_mask = F.interpolate(
                padding_mask[:, None].float(),  # [B, 1, H, W]
                size=size,  # [H, W]
                mode="nearest",
            )[:, 0].to(
                torch.bool
            )  # [:, 0]表示取第0维第一个数据：[B, H/32, W/32]
        else:
            feature_mask = None
        return feature_mask

    def forward(self, images, padding_mask=None):
        # images.shape = [batch_size, 3, height, width]
        # padding_mask.shape = [batch_size, height, width]

        x = self.stem(images)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)  # [batch_size, 512, height/8, width/8]
        layer3 = self.layer3(layer2)  # [batch_size, 1024, height/16, width/16]
        layer4 = self.layer4(layer3)  # [batch_size, 2048, height/32, width/32]

        features = [
            (layer2, self.get_mask(padding_mask, layer2.shape[-2:])),
            (layer3, self.get_mask(padding_mask, layer3.shape[-2:])),
            (layer4, self.get_mask(padding_mask, layer4.shape[-2:])),
        ]
        return features


if __name__ == "__main__":
    from torchsummary import summary

    # model = resnet18(pretrained=True)
    # print(model)
    # print(weights_dir)
    summary(Backbone(), (3, 224, 224), batch_size=1)
