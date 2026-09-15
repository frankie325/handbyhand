from torchvision import transforms
from .utils import GaussianBlur, Solarization
from PIL import Image

""" 
DINO 数据增强策略
一张图最终会被变成 2 + N 个 crop：

2 个 global crop：大图（224×224），承载全局语义；
N 个 local crop：小图（96×96），承载局部细节，N 默认 8。

这正是 DINO 高效又有效的关键 —— student 看全部 2+N 个视图，teacher 只看 2 个 global 视图。
"""


class AugmentationDino:
    #
    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
    ):

        # 三种随机增强串联
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),  # 80% 概率随机改亮度/对比度/饱和度/色相
                transforms.RandomGrayscale(p=0.2),  # 20% 概率随机灰度化
            ]
        )

        # 转张量 + 标准化
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                # ImageNet 数据集统计出来的: RGB 三通道的均值（mean）和 标准差（std）
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # 第一个 global crop
        self.global_transfo1 = transforms.Compose(
            [
                # 从原图随机裁一块再缩放到 224，裁取比例由 global_crops_scale=(0.4, 1.) 控制（默认约 (0.14, 1.0)，即裁较大的区域
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(
                    1.0
                ),  # 高斯模糊，如果不加模糊，模型容易走捷径——靠纹理、高频噪点等「表面细节」来区分类别
                normalize,
            ]
        )

        # 第二个 global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )

        # N 个 local crop，默认8个
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                # 裁更小的区域（默认 local_crops_scale≈(0.05,0.14)）缩到 96×96
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
