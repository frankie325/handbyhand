import argparse
import logging
import os

from pathlib import Path
from dinov2.logging import setup_logging
from dinov2.distributed import is_main_process, get_global_size
from dinov2.utils.config import setup


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument(
        "--config-file", default="dinov2/configs/train/vitl14.yaml", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="outputs",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def main(args):
    cfg = setup(args)
    os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    root = Path(__file__).parent
    log_outputs = str(root / "outputs/train.log")
    setup_logging(output=log_outputs, name="dinov2", level=logging.INFO)
    logger = logging.getLogger("dinov2")
    logger.info(f"是否是主进程：{is_main_process()}")
    logger.info(f"总进程（GPU）数量：{get_global_size()}")
    logger.info("测试警告信息")

    args = get_args_parser(add_help=True).parse_args()

    main(args)
