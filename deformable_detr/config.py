# config.py
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 数据路径
DATA_ROOT: Path = ROOT_DIR.parent / "datasets" / "VOC2007"
RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR: Path = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# 模型和日志路径
MODELS_DIR = ROOT_DIR / "weights"
LOG_DIR = ROOT_DIR / "logs"

LR = 1e-4  # 学习率
BACKBONE_LR = 1e-5  # 主干网络学习率

BATCH_SIZE = 2  # 批大小
NUM_CLASSES = 20  # VOC20 类，不包括 no-object
NUM_QUERIES = 100
D_MODEL = 128  # 模型输入维度
N_HEAD = 4  # 多头注意力头数
N_LAYER = 4  # 编码器层数
D_FF = 1024 # 前馈层网络维度
DROPOUT = 0.1  # 注意力权重dropout概率
EPOCHS = 200
