# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from collections import defaultdict, deque
import datetime
import json
import logging
import time

import torch

import dinov2.distributed as distributed


logger = logging.getLogger("dinov2")


# =============================================================================
# 训练日志与指标平滑工具
#
# 作用：在训练循环里统一管理各类标量指标（loss、lr、准确率等），做滑动窗口平滑，
#       多卡(DDP/FSDP)下跨进程 all_reduce 同步成全局值，并按固定频率打印进度、
#       估算剩余时间(ETA)、把指标追加写入 JSONL 文件。
#
# 两个核心类：
#   - MetricLogger : 一组指标的“容器 + 进度打印机”，内部每个指标是一个 SmoothedValue。
#   - SmoothedValue: 单个指标的滑动窗口平滑 + 全局均值，支持分布式同步。
# =============================================================================


class MetricLogger(object):
    """指标容器：聚合多个 SmoothedValue，并提供 log_every 进度打印/落盘。"""

    def __init__(self, delimiter="\t", output_file=None):
        # meters: 指标名 -> SmoothedValue；delimiter 用于拼接打印；output_file 为 JSONL 路径
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file

    def update(self, **kwargs):
        # 批量更新指标；若传入的是单元素张量则先取 .item() 转成 Python 标量
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        # 支持用 logger.loss 这种点号语法直接访问某个指标(SmoothedValue)
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        # 把所有指标按 delimiter 拼成一行可读字符串，用于日志打印
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        # 让每个指标跨进程同步（多卡下把各卡的 count/total 做 all_reduce 求全局值）
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        # 手动注册一个自定义 SmoothedValue（一般指标用 update 自动创建即可）
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        # 仅主进程把本步指标以 JSONL 形式追加写入 output_file（iter_time/data_time 传的是 avg）
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")
        pass

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        """包装一个 iterable：每 print_freq 步打印一次进度，并 yield 出每个元素供训练使用。

        这是训练主循环的核心“进度条”：统计每步耗时(iter_time)与取数据耗时(data_time)，
        估算 ETA，定期把指标落盘并 logger.info 打印（含 GPU 峰值显存）。
        """
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        # 用两个 SmoothedValue 平滑“步耗时”与“数据加载耗时”
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        # 拼装日志模板
        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                # 到打印步：落盘 + 按全局平均步耗时估算剩余时间 ETA
                self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {} ({:.6f} s / it)".format(header, total_time_str, total_time / n_iterations))


class SmoothedValue:
    """跟踪一串数值，提供“滑动窗口内”的统计量（median/avg/max）与“全局”均值。

    内部维护一个定长 deque（窗口）用于局部平滑，同时累计 total/count 用于全局平均。
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        # deque 只保留最近 window_size 个值，用于算窗口 median/avg/max
        self.deque = deque(maxlen=window_size)
        self.total = 0.0   # 所有值的累加和（跨窗口）
        self.count = 0     # 值的个数（支持加权 num）
        self.fmt = fmt     # 打印格式

    def update(self, value, num=1):
        # 放入一个新值（num 表示该值的样本数，用于加权平均）
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        分布式同步：把各进程的 count/total 做 all_reduce 求和，得到全局指标。
        注意：deque（窗口）不参与同步，只同步全局 total/count。
        """
        if not distributed.is_enabled():
            return
        # 用 float64 在 GPU 上把 [count, total] 求和，避免精度损失
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        # 窗口内值的中位数
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        # 窗口内值的均值（最近 window_size 步的平均）
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # 全局均值 = 总和 / 总个数（跨整个训练，不限于窗口）
        return self.total / self.count

    @property
    def max(self):
        # 窗口内最大值
        return max(self.deque)

    @property
    def value(self):
        # 最近一次更新的值
        return self.deque[-1]

    def __str__(self):
        # 按 fmt 输出，例如 "0.1234 (0.1256)"
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
