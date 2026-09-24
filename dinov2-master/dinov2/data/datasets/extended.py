"""把“字节读取、图像解码、数据增强”分层的 Dataset 基类。"""

from typing import Any, Tuple, Type

from torchvision.datasets import VisionDataset

from .decoders import Decoder, ImageDataDecoder, TargetDecoder


class ExtendedVisionDataset(VisionDataset):
    """DINOv2 数据集的模板方法实现。

    子类只关心如何通过 index 找到图片 bytes 和 target；本类统一处理
    解码、异常包装以及 torchvision transforms，从而让文件夹图片和 tar
    图片共享完全相同的增强流程。
    """

    def __init__(self, *args, **kwargs) -> None:
        self._image_decoder_class: Type[Decoder] = kwargs.pop("image_decoder_class", ImageDataDecoder)
        self._decoder_params = kwargs.pop("image_decoder_params", {})
        super().__init__(*args, **kwargs)

    def get_image_data(self, index: int) -> bytes:
        """子类实现：返回一张仍处于 JPEG/PNG 编码状态的图片字节。"""

        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        """子类实现：返回标签；自监督预训练随后通常会丢弃它。"""

        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self._image_decoder_class(image_data, **self._decoder_params).decode()
        except Exception as error:
            # 附加样本 index，定位大规模数据中的坏图比只看到 PIL 异常更容易。
            raise RuntimeError(f"cannot read image for sample {index}") from error

        target = TargetDecoder(self.get_target(index)).decode()

        # VisionDataset 会把 transform 和 target_transform 合并到 self.transforms。
        # 对 DINOv2 而言，image 会在这里变成包含多个 crop 的字典。
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
