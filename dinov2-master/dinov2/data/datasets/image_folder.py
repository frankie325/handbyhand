"""读取 ``root/类别名/图片`` 形式的小型图像数据集。"""

from torchvision.datasets import ImageFolder


class ImageFolderDataset(ImageFolder):
    """DINOv2 学习版使用的通用文件夹数据集。

    ``torchvision.datasets.ImageFolder`` 会把根目录下的每个子目录当成
    一个类别，并自动生成 ``类别名 -> 整数标签`` 的映射。图片在
    ``__getitem__`` 中才会按需读取，因此创建 Dataset 时不会把全部图片
    载入内存。

    目录示例::

        root/
          n01440764/
            image_1.JPEG
          n02102040/
            image_2.JPEG

    ImageFolder 默认使用 PIL 把图片转换为 RGB，然后继续调用传入的
    ``transform``。这使 Imagenette、Tiny ImageNet 或自制分类文件夹都能
    直接复用 DINOv2 的多裁剪增强。
    """

    pass
