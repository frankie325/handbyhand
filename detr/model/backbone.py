import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torch
from pathlib import Path

# 设置预训练模型保存路径
weights_dir = Path(__file__).resolve().parents[1] / "weights"
torch.hub.set_dir(str(weights_dir))


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
        self.num_channels = 2048
        # ResNet 离得的 avgpool 和 fc 会把空间维度压成一个向量，DETR 需要
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

    def forward(self, images, padding_mask=None):
        # images.shape = [batch_size, 3, height, width]
        # padding_mask.shape = [batch_size, height, width]

        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        # features.shape = [batch_size, 2048, height/32, width/32]

        feature_mask = None
        if padding_mask is not None:
            # ?最近邻插值法：https://www.bilibili.com/video/BV1BM4y1t7tf/?spm_id_from=333.337.search-card.all.click&vd_source=23a9782e838dd1b9be9e5295dd2ccf9e
            # mask 是离散的 bool 值，使用 最近邻插值法将padding_mask缩小到[height/32, width/32]的尺寸，像素值避免产生 0.3、0.7的小数
            # 之类没有明确含义的中间值。
            feature_mask = F.interpolate(
                padding_mask[:, None].float(),  # [B, 1, H, W]
                size=features.shape[-2:],  # [B, 1, H/32, W/32]
                mode="nearest",
            )[:, 0].to(
                torch.bool
            )  # [:, 0]表示取第0维第一个数据：[B, H/32, W/32]
        return features, feature_mask


if __name__ == "__main__":
    from torchsummary import summary

    # model = resnet18(pretrained=True)
    # print(model)
    # print(weights_dir)
    summary(Backbone(), (3, 224, 224), batch_size=1)
