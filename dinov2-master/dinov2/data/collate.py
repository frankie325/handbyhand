"""把多视图样本拼成 batch，并为 iBOT 生成 patch mask。"""

import random

import torch


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
):
    """
    输入：batch_size个样本
    [
        ({
        "global_crops": [Tensor[C, 224, 224], Tensor[C, 224, 224]],
        "global_crops_teacher": [Tensor[C, 224, 224], Tensor[C, 224, 224]],
        "local_crops": [Tensor[C, 98, 98], ...],  # 共 local_crops_number 个
        "offsets": (),
        }),
        ...
    ]
    只有global_crops需要添加mask
    """
    if not samples_list:
        raise ValueError("samples_list must not be empty")
    if n_tokens is None or mask_generator is None:
        raise ValueError("n_tokens and mask_generator are required")

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    # 采用“crop 类型优先”的排列：先放 batch 内所有 global-0，再放
    # 所有 global-1。形状为 [n_global_crops * batch_size, C, H, W]。
    collated_global_crops = torch.stack(
        [
            sample[0]["global_crops"][crop_index]
            for crop_index in range(n_global_crops)
            for sample in samples_list
        ]
    )

    # 形状为 [n_local_crops * batch_size, C, h, w]。local crop 数为 0
    # 时，标准预训练配置不会走到这里；学习版显式报错便于发现配置问题。
    if n_local_crops == 0:
        raise ValueError("DINO multi-crop training expects at least one local crop")
    collated_local_crops = torch.stack(
        [
            sample[0]["local_crops"][crop_index]
            for crop_index in range(n_local_crops)
            for sample in samples_list
        ]
    )

    global_batch_size = len(collated_global_crops)
    n_samples_masked = int(global_batch_size * mask_probability)

    # 将 mask 比例区间均分为 n_samples_masked 个小区间，每个样本从自己的
    # 区间抽样。这样一个 batch 内会同时包含轻度、中度和重度遮挡，而不是所有样本碰巧集中在某个比例附近
    # mask_ratio_tuple: (0.1, 0.5)
    # 0.1 ~ 0.5 均分成 n_samples_masked 个区间
    ratio_boundaries = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    masks_list = []
    upperbound = 0  # 当前batch所有特征图遮挡的masked token总数的作为保守上界，模型用它提前分配 masked-token buffer，避免运行时反复申请显存
    for sample_index in range(n_samples_masked):
        ratio_min = float(ratio_boundaries[sample_index])
        ratio_max = float(ratio_boundaries[sample_index + 1])
        ratio = random.uniform(ratio_min, ratio_max)
        # 一张特征图的masked token数量
        target_mask_count = int(n_tokens * ratio)
        masks_list.append(
            torch.as_tensor(mask_generator(target_mask_count), dtype=torch.bool)
        )
        # 按区间上界累计，用于模型侧提前分配足够大的 buffer。
        upperbound += int(n_tokens * ratio_max)

    # 剩余 global views 完全不遮挡。最后打乱列表，避免总是只有 batch
    # 前半部分样本被遮挡。
    for _ in range(n_samples_masked, global_batch_size):
        masks_list.append(torch.as_tensor(mask_generator(0), dtype=torch.bool))
    random.shuffle(masks_list)

    # masks_list [(grid_h, grid_w),... ] 总共n_global_crops * batch_size个mask

    # (n_global_crops * batch_size, grid_h, grid_w) → (n_global_crops * batch_size, grid_h * grid_w)
    collated_masks = torch.stack(masks_list).flatten(1)

    # 将二维 batch-token 坐标压成一维索引。模型可先把所有 patch token
    # 展平，再一次性抽取参与 iBOT loss 的位置。
    #  (非masked_token的数量)
    # [10, 11, 82, ...] 当前批次所有mask中展平之后，非0位置的索引
    mask_indices_list = collated_masks.flatten().nonzero(as_tuple=False).flatten()

    # 如果某个样本遮挡了 K 个 patch，那么每个 patch 权重为：1 / K，这样每个被遮挡样本对 loss 的总权重大约都是1，不会因为遮挡patch更多而占据更大权重
    # (n_global_crops * batch_size, grid_h * grid_w)
    per_sample_weight = 1 / collated_masks.sum(-1).clamp(
        min=1.0
    )  # (n_global_crops * batch_size)

    """
    (n_global_crops * batch_size, grid_h * grid_w)
    每个样本的权重
    tensor([[1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
        [0.0333, 0.0333, 0.0333,  ..., 0.0333, 0.0333, 0.0333],
        [1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
        ...,
        [0.0098, 0.0098, 0.0098,  ..., 0.0098, 0.0098, 0.0098],
        [0.0122, 0.0122, 0.0122,  ..., 0.0122, 0.0122, 0.0122],
        [0.0149, 0.0149, 0.0149,  ..., 0.0149, 0.0149, 0.0149]])
    """
    masks_weight = per_sample_weight.unsqueeze(-1).expand_as(collated_masks)[
        collated_masks
    ]

    return {
        # 只转换图片 dtype；mask 和索引必须保持 bool/long 才能正确索引。
        "collated_global_crops": collated_global_crops.to(
            dtype
        ),  # [n_global_crops * batch_size, C, H, W]。
        "collated_local_crops": collated_local_crops.to(
            dtype
        ),  # [n_local_crops * batch_size, C, h, w]
        "collated_masks": collated_masks,  # (n_global_crops * batch_size, grid_h * grid_w)
        "mask_indices_list": mask_indices_list,  # (masked_token的数量) 当前批次所有mask中展平之后，非0位置（masked_token）的索引
        "masks_weight": masks_weight,  # (n_global_crops * batch_size, grid_h * grid_w)
        "upperbound": upperbound,  #  当前batch所有特征图遮挡的masked token总数的作为保守上界，模型用它提前分配 masked-token buffer，避免运行时反复申请显存
        "n_masked_patches": torch.full(
            (1,), mask_indices_list.numel(), dtype=torch.long
        ),  #n_masked_patches = [masked_token的数量]
    }
