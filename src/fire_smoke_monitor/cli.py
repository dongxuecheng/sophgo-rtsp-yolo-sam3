from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

import yaml

from .app import build_application
from .config import ConfigError, load_config, validate_runtime_paths
from .logging_setup import LogManager, RedactingFormatter


logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BM1684X 多路 RTSP 上传服务（YOLO 命中 / 定时直传）")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument(
        "--check-config", action="store_true",
        help="只检查配置、密钥和所需模型文件，不初始化硬件、不发请求、不创建日志文件",
    )
    parser.add_argument("--no-console", action="store_true", help="仅记录文件日志（供 systemd 使用）")
    return parser


def _configure_console(level: str) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(RedactingFormatter())
        root.addHandler(handler)
    root.setLevel(level)


def _run_application(config) -> int:
    stop_event = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del frame
        logger.info("收到退出信号 signal=%d", signum)
        stop_event.set()

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, request_stop)
    application = None
    exit_code = 0
    try:
        application = build_application(config)
        application.run(stop_event)
    except Exception:
        logger.exception("服务异常退出")
        exit_code = 1
    finally:
        try:
            if application is not None:
                application.close()
        except Exception:
            logger.exception("关闭服务失败")
            exit_code = 1
        finally:
            logger.info("服务已停止")
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _configure_console("INFO")
    try:
        config = load_config(Path(args.config))
        _configure_console(config.runtime.log_level)
        validate_runtime_paths(config)
        if args.check_config:
            logger.info("配置检查通过 cameras=%d mode=%s yolo26=%s local_save=disabled",
                        len(config.cameras), config.mode, config.yolo26 is not None)
            return 0
    except yaml.YAMLError:
        logger.error("配置检查失败：YAML 语法错误，请检查缩进和引号")
        return 2
    except (ConfigError, ValueError, OSError) as exc:
        logger.error("配置检查失败: %s", exc)
        return 2

    try:
        with LogManager(
            config.runtime.log_dir, config.runtime.log_retention_days,
            config.runtime.log_level, secrets=(config.pipeline.api_key(),),
            console=not args.no_console,
        ):
            logger.info("日志已启用 directory=%s retention_days=%d native_capture=enabled",
                        config.runtime.log_dir, config.runtime.log_retention_days)
            return _run_application(config)
    except (OSError, ValueError) as exc:
        logger.error("日志初始化/关闭失败 error=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
