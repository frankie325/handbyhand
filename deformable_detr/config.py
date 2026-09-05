# config.py
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# RAOD-benchmark 数据集根目录，其中包含 select_1000_images 和
# select_1000_txts。数据按文件名排序后固定划分为 80% 训练、20% 验证。
DATA_ROOT: Path = ROOT_DIR.parent / "datasets" / "RAOD-benchmark"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# 模型和日志路径
MODELS_DIR = ROOT_DIR / "weights"
LOG_DIR = ROOT_DIR / "logs"

LR = 1e-4  # 学习率
BACKBONE_LR = 1e-5  # 主干网络学习率
WEIGHT_DECAY = 1e-4
LR_DROP = 40
CLIP_MAX_NORM = 0.1

BATCH_SIZE = 2  # 批大小
NUM_WORKERS = 0
NUM_CLASSES = 1  # RAOD 只有 abandoned_object 一类，不包括 no-object
NUM_QUERIES = 50  # object_queries的数量
D_MODEL = 128  # 模型输入维度
N_HEAD = 4  # 多头注意力头数
N_ENCODER_LAYER = 4  # 编码器层数
N_DECODER_LAYER = 4  # 解码器层数
ENCODER_N_POINTES = 4  # 编码器预测偏移点数量
DECODER_N_POINTES = 4  # 解码器预测偏移点数量
D_FF = 1024  # 前馈层网络维度
DROPOUT = 0.1  # 注意力权重dropout概率
AUX_LOSS = True
WITH_BOX_REFINE = False
TWO_STAGE = False

# 匈牙利匹配和损失权重（与 Deformable DETR 的常用配置一致）
CLASS_COST = 2.0
BBOX_COST = 5.0
GIOU_COST = 2.0
CLASS_LOSS_COEF = 2.0
BBOX_LOSS_COEF = 5.0
GIOU_LOSS_COEF = 2.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

TRAIN_RATIO = 0.8
EPOCHS = 50
VALIDATE_EVERY = 1
SEED = 42
