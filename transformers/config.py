# config.py

from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 数据路径
RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR: Path = ROOT_DIR / "data" / "processed"

# 模型和日志路径
MODELS_DIR = ROOT_DIR / "weights"
LOG_DIR = ROOT_DIR / "logs"


# 训练参数
SEQ_LEN = 64  # 输入序列长度
BATCH_SIZE = 64  # 批大小
MAX_SEQ_LEN = 64  # 最大预测序列长度
LEARNING_RATE = 1e-3  # 学习率

D_MODEL = 128  # 模型输入维度
N_HEAD = 4  # 多头注意力头数
N_LAYER = 2  # 编码器层数
D_FF = 1024  # 前馈层网络维度
DROPOUT = 0.1  # 注意力权重dropout概率
EPOCHS = 30
WARMUP_STEPS = 4000  # Noam 学习率调度：warmup 步数
CLIP_GRAD_NORM = 1.0  # 梯度裁剪阈值（None 表示不裁剪）