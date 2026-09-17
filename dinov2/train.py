from .logging import setup_logging
from .distributed import is_main_process, get_global_size
import logging
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).parent
    log_outputs = str(root / "outputs/train.log")
    setup_logging(output=log_outputs, name="dinov2", level=logging.INFO)
    logger = logging.getLogger("dinov2")
    logger.info(f"是否是主进程：{is_main_process()}")
    logger.info(f"总进程（GPU）数量：{get_global_size()}")
    logger.info("测试警告信息")
