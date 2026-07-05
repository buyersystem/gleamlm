"""统一日志模块，替代裸 print()"""
from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """创建带统一格式的 logger"""
    logger = logging.getLogger(name or __name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
