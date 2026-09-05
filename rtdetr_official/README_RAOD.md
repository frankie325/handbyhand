# Train official RT-DETR on RAOD-benchmark

This directory contains the official PyTorch implementation from
[`lyuwenyu/RT-DETR`](https://github.com/lyuwenyu/RT-DETR), pinned locally at
commit `068dfde65f2667ad6555883c69d73de886518cad`. The only source changes are
compatibility aliases for the current torchvision `tv_tensors` API; the model,
criterion, solver, and optimizer remain the upstream implementation.

RAOD stores YOLO labels, while the official trainer consumes COCO annotations.
Generate the two deterministic COCO split files first:

```bash
cd /Users/frank/code/ai/handbyhand/rtdetr_official
conda activate handbyhand
python tools/prepare_raod.py
```

Then train the one-class `abandoned_object` detector:

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_raod.yml \
  --seed 42
```

The first run downloads the official ResNet-50 vd pretrained backbone because
`PResNet.pretrained` is enabled. Checkpoints and logs are written to
`rtdetr_official/output/rtdetr_r50vd_6x_raod/`.

To evaluate a saved checkpoint:

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_raod.yml \
  -r output/rtdetr_r50vd_6x_raod/checkpoint.pth \
  --test-only
```
