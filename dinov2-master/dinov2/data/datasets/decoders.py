"""将 Dataset 读取到的原始字节解码为模型增强所需的图像对象。"""

from io import BytesIO
from typing import Any

from PIL import Image


class Decoder:
    """所有解码器的最小接口。"""

    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    """把 JPEG/PNG 等编码字节解码为三通道 RGB PIL 图片。"""

    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image.Image:
        # BytesIO 让 PIL 像读取文件一样读取内存中的 bytes，无需创建临时文件。
        stream = BytesIO(self._image_data)
        # convert("RGB") 统一灰度、RGBA 等输入，保证网络收到 3 个通道。
        # load() 强制在 BytesIO 生命周期内完成像素解码，避免惰性读取问题。
        image = Image.open(stream).convert("RGB")
        image.load()
        return image


class TargetDecoder(Decoder):
    """标准分类标签不需要解码，原样返回以统一接口。"""

    def __init__(self, target: Any) -> None:
        self._target = target

    def decode(self) -> Any:
        return self._target
