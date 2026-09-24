"""DINOv2 自监督预训练使用的多裁剪数据增强。"""

import logging

from torchvision import transforms

from .transforms import GaussianBlur, make_normalize_transform


logger = logging.getLogger("dinov2")


class DataAugmentationDINO:
    """把一张 RGB 图片转换成两个全局视图和若干局部视图。

    teacher 只看全局视图；student 同时看全局和局部视图。不同视图来自
    同一原图，但裁剪位置、颜色扰动、模糊等随机量彼此独立。
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("Using DINO data augmentation parameters:")
        logger.info("global_crops_scale=%s", global_crops_scale)
        logger.info("local_crops_scale=%s", local_crops_scale)
        logger.info("local_crops_number=%s", local_crops_number)
        logger.info("global_crops_size=%s", global_crops_size)
        logger.info("local_crops_size=%s", local_crops_size)

        # 几何增强先决定“看原图的哪一块”，再统一缩放到模型要求的尺寸。
        # scale 表示裁剪区域相对原图面积的范围，而不是边长比例。
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # 所有视图共享颜色扰动策略，但每次调用都会独立抽样。
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        # 两个全局视图故意使用不对称增强，防止模型依赖某一种低层视觉特征。
        global_view_1_extra = GaussianBlur(p=1.0)
        global_view_2_extra = transforms.Compose(
            [GaussianBlur(p=0.1), transforms.RandomSolarize(threshold=128, p=0.2)]
        )
        local_view_extra = GaussianBlur(p=0.5)

        # ToTensor: [H,W,C] PIL → [C,H,W] float；随后按 ImageNet 统计量标准化。
        normalize = transforms.Compose([transforms.ToTensor(), make_normalize_transform()])

        self.global_transform_1 = transforms.Compose([color_jittering, global_view_1_extra, normalize])
        self.global_transform_2 = transforms.Compose([color_jittering, global_view_2_extra, normalize])
        self.local_transform = transforms.Compose([color_jittering, local_view_extra, normalize])

    def __call__(self, image):
        """生成一个样本的全部视图，返回供 collate 函数消费的字典。"""

        # 两个全局视图必须分别执行几何增强，才能观察原图的不同区域。
        global_base_1 = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transform_1(global_base_1)

        global_base_2 = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transform_2(global_base_2)
        global_crops = [global_crop_1, global_crop_2]

        # 每个 local crop 也独立裁剪，而不是从某个 global crop 中继续裁剪。
        local_crops = [
            self.local_transform(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]

        return {
            "global_crops": global_crops, # [(3,224,224), (3,224,224)]
            # teacher 看同一组未做 token mask 的全局图像；EMA 和 stop-gradient
            # 发生在模型中，而不是图像增强阶段。
            "global_crops_teacher": global_crops,
            "local_crops": local_crops,  # [(3,98,98), (3,98,98)]
            # 保留该字段以兼容支持裁剪位置监督的其他实现。
            "offsets": (),
        }
