# pycocotools 加载 COCO 数据集教程

> 本文以本仓库的实际数据（`detr/data/` 下的 train2014 / val2014）为例，
> 讲解 `pycocotools` 如何加载、查询、可视化 COCO 数据集，并做 COCO AP 评估。
> 所有示例都可以在 `detr` 项目根目录下运行。

## 1. 简介

[pycocotools](https://github.com/cocodataset/cocoapi) 是 COCO 官方提供的 Python API，
核心功能：

- `COCO` 类：加载并索引标注 JSON，提供图片 / 标注 / 类别的查询接口。
- `COCOeval` 类：计算 mAP / mAR 等评测指标（目标检测、分割、关键点）。
- `maskUtils`：处理分割掩码。

本项目中的实际使用位置：

- `detr/datasets/coco.py`：数据集类，用 `pycocotools.coco.COCO` 索引标注。
- `detr/predict.py`：加载 COCO 类别名用于推理可视化。
- `detr/validate.py`：用 `COCOeval` 做 COCO AP 评估。

## 2. 安装

```bash
uv add pycocotools
```

macOS 上如果编译报错，通常是缺少 C 编译器，先安装：

```bash
brew install gcc
```

## 3. COCO 标注文件结构

一个实例标注文件（如 `data/annotations/instances_val2014.json`）是一个大 JSON，包含 3 个顶层字段：

```jsonc
{
  "images": [
    {
      "id": 324158,              // 图片唯一 ID（跨 train/val 全局唯一）
      "file_name": "COCO_val2014_000000324158.jpg",
      "width": 640,
      "height": 480
    }
    // ...更多图片
  ],
  "annotations": [
    {
      "id": 1768,                // 标注唯一 ID
      "image_id": 324158,        // 属于哪张图片
      "category_id": 3,          // 类别 ID（1~90，中间有跳号）
      "bbox": [100, 200, 50, 60],// [x, y, width, height]，左上角 + 宽高，像素单位
      "area": 3000.0,            // 面积，通常 = w * h
      "iscrowd": 0,              // 1 表示群体标注，训练时一般过滤
      "segmentation": [[...], ...] // 多边形或 RLE（图片较大时）
    }
    // ...更多标注
  ],
  "categories": [
    {
      "id": 1,                   // 类别 ID
      "name": "person",          // 类别名
      "supercategory": "person"  // 父类
    }
    // ...共 80 个类别（ID 1~90 中有空缺，如 12、26、29）
  ]
}
```

> 注意：`bbox` 是 `[x, y, w, h]`（左上角 + 宽高），而 DETR 训练使用的是
> 归一化的 `cxcywh`（中心点 + 宽高），两者需要显式转换，见第 6 节。

## 4. 加载数据集

```python
from pycocotools.coco import COCO

# 传入标注 JSON 路径即可
coco = COCO("data/annotations/instances_val2014.json")
```

加载后 `COCO` 会在内部建立 3 个索引表，供所有查询使用：

| 属性 | 内容 |
| --- | --- |
| `coco.imgs` | `{image_id: 图片信息字典}` |
| `coco.anns` | `{annotation_id: 标注字典}` |
| `coco.cats` | `{category_id: 类别字典}` |

## 5. 核心查询 API

### 5.1 图片查询

```python
# 全部图片 ID（可传 imgIds 过滤）
all_ids = coco.getImgIds()
cat_ids = coco.getImgIds(catIds=[3])          # 包含 person 的图片
img_ids = coco.getImgIds(imgIds=[324158])     # 指定图片

# 根据 ID 取图片信息
img_infos = coco.loadImgs([324158])
img = img_infos[0]
print(img["file_name"], img["width"], img["height"])
```

### 5.2 标注查询

```python
# 某张图片的标注 ID 列表（可传 catIds 过滤类别）
ann_ids = coco.getAnnIds(imgIds=[324158])
ann_ids = coco.getAnnIds(imgIds=[324158], catIds=[3], iscrowd=0)

# 根据标注 ID 取标注内容
anns = coco.loadAnns(ann_ids)
for ann in anns:
    x, y, w, h = ann["bbox"]
    print(ann["category_id"], ann["area"], ann["iscrowd"])
```

### 5.3 类别查询

```python
# 所有类别 ID
coco.getCatIds()                       # [1, 2, 3, ..., 90]，中间有跳号

# 指定类别名的 ID（英文名）
coco.getCatIds(catNms=["person"])

# ID -> 名字 的映射（本项目 predict.py 就是这么用的）
categories = {
    category_id: category["name"]
    for category_id, category in coco.cats.items()
}
print(categories[3])                   # 'person'
```

### 5.4 一句话总结

`get*Ids()` 返回 **ID 列表**（用于过滤），`load*()` 返回 **详细信息列表**（用于读取内容），
两者配合使用：

```python
img_ids = coco.getImgIds()                  # 先查 ID
imgs    = coco.loadImgs(img_ids)            # 再取详情
```

## 6. 结合本项目：读取一张图并画框

下面的代码与 `detr/predict.py` 的可视化思路一致，但不经过模型，直接画 GT 框：

```python
from pathlib import Path

from PIL import Image, ImageDraw
from pycocotools.coco import COCO

coco = COCO("data/annotations/instances_val2014.json")
image_dir = Path("data/val2014")

image_id = coco.getImgIds()[0]
img_info = coco.loadImgs([image_id])[0]
image = Image.open(image_dir / img_info["file_name"]).convert("RGB")

categories = {cid: c["name"] for cid, c in coco.cats.items()}
drawer = ImageDraw.Draw(image)

for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
    x, y, w, h = ann["bbox"]
    drawer.rectangle((x, y, x + w, y + h), outline="red", width=3)
    drawer.text((x, max(0, y - 12)),
                f"{categories[ann['category_id']]}",
                fill="red")

image.save("gt_visualization.jpg")
```

> 坐标换算速查：
> - COCO `bbox` 是像素 `xywh`（左上角 + 宽高）→ 画框时 `(x, y, x+w, y+h)`。
> - DETR 模型的输出是归一化 `cxcywh` → 先 `box_cxcywh_to_xyxy` 再乘图片宽高
>   （见 `detr/predict.py` 第 68-77 行）。

## 7. 结合本项目：封装数据集类

本项目 `detr/datasets/coco.py` 的 `CocoDetection` 直接继承了
`torchvision.datasets.CocoDetection`，后者内部就是用它创建 `COCO` 索引：

```python
from torchvision.datasets import CocoDetection as TorchvisionCocoDetection

class CocoDetection(TorchvisionCocoDetection):
    def __init__(self, image_dir, annotation_file, transforms=None):
        # TorchVision 内部创建 pycocotools.coco.COCO
        # self.coco 保存图片、标注和类别索引；self.ids 保存排序后的 image_id
        super().__init__(root=image_dir, annFile=str(annotation_file))
        # ...
```

因此可以直接从 `dataset.coco` 访问索引：

```python
dataset.coco.getImgIds()        # 所有图片 ID
dataset.coco.getCatIds()        # 所有类别 ID
dataset.coco.imgs / .anns / .cats
```

## 8. 结合本项目：COCO AP 评估

完整代码见 `detr/validate.py`。核心流程三步：

```python
from pycocotools.cocoeval import COCOeval

# 1) 收集模型预测，格式必须是 dict 列表
results = [
    {"image_id": 324158, "category_id": 3, "bbox": [x, y, w, h], "score": 0.92},
    # ...
]

# 2) 把 GT 和预测包装成 COCO 对象
gt = COCO("data/annotations/instances_val2014.json")
pred = gt.loadRes(results)      # loadRes: 从检测结果构建预测对象

# 3) 评估
evaluator = COCOeval(gt, pred, iouType="bbox")
evaluator.params.imgIds = gt.getImgIds()   # 限定评估的图片范围
evaluator.evaluate()                        # 计算所有 IoU 匹配
evaluator.accumulate()                      # 累加统计 TP/FP
evaluator.summarize()                       # 打印 AP/AR 汇总表
```

> `evaluator.summarize()` 打印的 12 行结果依次是：
> `AP@[.5:.95]`、`AP@.5`、`AP@.75`、`AP small/medium/large`、
> `AR@1`、`AR@10`、`AR@100`、`AR small/medium/large`。

## 9. 常见坑

1. **`bbox` 越界**：COCO 的标注框偶尔会超出图片边界，读取后建议 clamp 到
   `[0, width]` / `[0, height]`（见 `datasets/coco.py` 第 47-48 行）。
2. **`iscrowd=1` 的标注**：表示群体目标，DETR 目标检测训练会过滤掉
   （见 `datasets/coco.py` 第 30-34 行）。
3. **类别 ID 不连续**：COCO 的 category_id 在 1~90 之间但有跳号，做
   `index_select` 时要用真实 ID 列表，不能假设 `range(80)`。
4. **`loadRes` 的输入**：必须是 `[{"image_id", "category_id", "bbox", "score"}]`
   的 dict 列表，`bbox` 必须是**像素 xywh**，且宽高 > 0 的框才会被评估。
5. **图片路径拼错**：`img_info["file_name"]` 才是真实文件名，`image_id` 不是文件名。
6. **2014 与 2017 数据**：两者标注格式一致，但类别/图片 ID 不同，混用会导致
   查询结果为空或评估错误。
