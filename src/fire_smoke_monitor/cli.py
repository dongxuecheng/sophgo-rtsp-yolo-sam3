from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

from .app import build_application
from .config import ConfigError, load_config, validate_runtime_paths


logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BM1684X 多路 RTSP 火焰/烟雾检测服务"
    )
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="只检查配置和本地模型文件，不初始化硬件",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(Path(args.config))
        _configure_logging(config.runtime.log_level)
        validate_runtime_paths(config)
        if args.check_config:
            logger.info(
                "配置检查通过 cameras=%d yolo26=%s sam3=%s",
                len(config.cameras),
                config.yolo26.enabled,
                config.sam3.enabled,
            )
            return 0
    except (ConfigError, ValueError, OSError) as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logger.error("配置检查失败: %s", exc)
        return 2

    stop_event = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del frame
        logger.info("收到退出信号 signal=%d", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    application = None
    try:
        application = build_application(config)
        application.run(stop_event)
        return 0
    except Exception:
        logger.exception("服务异常退出")
        return 1
    finally:
        if application is not None:
            application.close()


if __name__ == "__main__":
    sys.exit(main())

