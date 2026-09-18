# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Any

import torch
import dinov2.distributed as distributed
from functools import partial
from fvcore.common.checkpoint import Checkpointer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp._runtime_utils import _reshard


# =============================================================================
# 本模块是 FSDP（Fully Sharded Data Parallel，全分片数据并行）的封装工具集。
#
# 作用：把 PyTorch 原生 torch.distributed.fsdp 包一层，方便在 dinov2 里用
#       FSDP 做多卡/多机大模型训练（显存按卡分片，能训超出单卡显存的模型）。
#
# 典型用法：
#   1) 构造包装器：
#        fsdp_wrapper = get_fsdp_wrapper(model_cfg, modules_to_wrap={Block})
#        model = fsdp_wrapper(model)          # 把模型包成 FSDP
#   2) 训练时用 ShardedGradScaler 做混合精度：
#        scaler = ShardedGradScaler()
#   3) 存/读 checkpoint：
#        ckpt = FSDPCheckpointer(model, save_dir=...)
#        ckpt.save("checkpoint") / ckpt.load(...)
#   4) 临时用完完整参数后回收显存：
#        reshard_fsdp_model(model)
# =============================================================================


def get_fsdp_wrapper(model_cfg, modules_to_wrap=set()):
    # 分片策略：把配置里的字符串映射成 FSDP 枚举
    #   NO_SHARD      : 不分片（等价于 DDP，但走 FSDP 框架）
    #   SHARD_GRAD_OP : 参数+梯度分片，优化器状态不分
    #   FULL_SHARD    : 参数+梯度+优化器状态全部分片（最省显存）
    sharding_strategy_dict = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    }

    # 混合精度数据类型映射（fp32 / fp16 / bf16）
    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    # 混合精度配置：分别指定 参数 / 归约通信 / 缓冲区 的计算精度
    #   param_dtype  : 计算时参数用的精度
    #   reduce_dtype : AllGather/ReduceScatter 通信用的精度（通常 fp32 保精度）
    #   buffer_dtype : 缓冲区（如 BN 的 running stats）的精度
    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[model_cfg.mixed_precision.param_dtype],
        reduce_dtype=dtype_dict[model_cfg.mixed_precision.reduce_dtype],
        buffer_dtype=dtype_dict[model_cfg.mixed_precision.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[model_cfg.sharding_strategy]

    # 本机内当前进程的 rank（用来绑定到对应 GPU）
    local_rank = distributed.get_local_rank()

    # 用 partial 预生成一个 FSDP 构造器，固化常用参数：
    #   device_id           : 绑定到 local_rank 对应的卡
    #   sync_module_states  : 各卡初始参数从 rank0 broadcast，保证一致
    #   use_orig_params     : 保留原始 nn.Parameter，便于接入优化器
    #   auto_wrap_policy    : 自动包裹策略，决定“哪些子模块单独成为分片单元”
    fsdp_wrapper = partial(
        FSDP,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        # ModuleWrapPolicy(modules_to_wrap)：
        #   - 接收一组“模块类”，递归遍历模型时，凡是类型在集合里的子模块，
        #     就被单独包成一个 FSDP 单元（其参数被分片，前向/反向时按需
        #     AllGather 出完整参数、用完即释放）。
        #   - 包裹粒度 = 分片/聚合的粒度：粒度越细（如每个 Block 单独包）
        #     显存峰值越低但通信越频繁；粒度越粗则相反。
        #   - 在 dinov2 里通常传入 transformer 的 Block 类，使每个 Block
        #     成为独立分片单元，在显存与通信间取得平衡。
        #   - 区别于 size_based_wrap_policy（按参数量阈值自动包裹）。
        auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap),
    )
    return fsdp_wrapper


# 判断对象是否为 FSDP 实例
def is_fsdp(x):
    return isinstance(x, FSDP)


# 判断是否为“真正分片”的 FSDP（排除 NO_SHARD 这种名义 FSDP 但没分片的情况）
def is_sharded_fsdp(x):
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD


# 把分片参数重新“拆解”回分片状态，释放临时聚合出的完整参数所占显存。
# 某些操作（如打印/推理）会临时 AllGather 出完整参数，用完调用它回收显存。
def free_if_fsdp(x):
    if is_sharded_fsdp(x):
        handles = x._handles
        true_list = [True for h in handles]
        _reshard(x, handles, true_list)


# 取模型里所有 FSDP 子模块（用于逐个重分片）
def get_fsdp_modules(x):
    return FSDP.fsdp_modules(x)


# 遍历模型内所有 FSDP 子模块并回收其显存（配合 free_if_fsdp 使用）
def reshard_fsdp_model(x):
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


# 生成 "rank_{N}" 后缀，用于让每个进程存/读自己那份分片 checkpoint
def rankstr():
    return f"rank_{distributed.get_global_rank()}"


# 继承 fvcore 的 Checkpointer，针对 FSDP 做“分片存取”改造。
#
# 背景：普通 DDP 每卡持有完整模型，通常只让 rank0 存一份完整权重即可；
#       而 FSDP 下每卡只持有自己那份参数分片，谁手里都没有完整模型，
#       因此必须让“每个 rank 各自存/读自己的分片”，恢复时再把所有分片拼回。
#
# 核心机制：用 FSDP.state_dict_type(..., LOCAL_STATE_DICT) 上下文取/载 state_dict，
#           只处理本 rank 的本地分片；文件名带 rank 后缀（如 ckpt.rank_0.pth），互不覆盖。
#
# 与“恢复中断训练”的关系：
#   - 真正“找到上次断点并加载”的编排逻辑来自父类 fvcore 的 resume_or_load()，
#     它会依次调用本类的 has_checkpoint() -> get_checkpoint_file() -> load()。
#   - 本类只是把这套“查找/加载”改成 FSDP 分片安全的版本（按 rank 定位、按分片读取）。
#   - 注意：仅恢复模型权重 ≠ 完整续训。要恢复优化器动量、步数/epoch、学习率调度器等，
#     需由上层把 optimizer / scheduler / iteration 等注册为 checkpointables 或塞进 kwargs，
#     由 save() 一并写入（见下方 checkpointables 与 kwargs 处理），load() 时才能完整续训。
#
# 结论：FSDPCheckpointer 不是 FSDP 核心训练（前向/反向/优化）的必需品，
#       而是“分片模型持久化 + 支持断点续训”的封装助手。
class FSDPCheckpointer(Checkpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        # 在 LOCAL_STATE_DICT 上下文中取 state_dict：只取本 rank 分到的本地分片
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            data["model"] = self.model.state_dict()

        # 一并保存其它可序列化对象（checkpointables，如 optimizer/scheduler/iteration）：
        # 这些是“完整续训”所必需的，否则只剩模型权重、丢失优化器与进度。
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        # 每个进程存自己的分片文件，如 checkpoint.rank_0.pth
        basename = f"{name}.{rankstr()}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, *args, **kwargs):
        # 同样在 LOCAL_STATE_DICT 下加载本 rank 的分片，再交给父类逻辑
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        # 按 rank 判断是否存在“最近一次 checkpoint”标记文件
        # （被父类 resume_or_load() 调用，用于决定是否能断点续训）
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        # pyre-fixme[6]: For 2nd param expected `Union[PathLike[str], str]` but got
        #  `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        # 写标记文件前用 barrier 同步各进程，保证所有 rank 都存完再标记
        if distributed.is_enabled():
            torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


# 重导出 FSDP 的梯度缩放器：用于混合精度（fp16）训练时做 loss scaling，
# 与 FSDP 的分片梯度协同工作（普通 GradScaler 不感知分片梯度）。
ShardedGradScaler = ShardedGradScaler
