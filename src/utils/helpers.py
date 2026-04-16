"""
工具模块
提供日志配置、进度条等通用工具
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import LOGGING_CONFIG


def setup_logging(level: Optional[str] = None):
    """配置日志系统"""
    log_level = level or LOGGING_CONFIG["level"]
    log_format = LOGGING_CONFIG["format"]
    log_file = LOGGING_CONFIG["file"]

    # 创建根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # 清除已有handler
    root_logger.handlers.clear()

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # 文件handler（带轮转）
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOGGING_CONFIG["max_bytes"],
        backupCount=LOGGING_CONFIG["backup_count"],
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    return root_logger


class ProgressTracker:
    """简单的进度追踪器"""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description

    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        percent = (self.current / self.total * 100) if self.total > 0 else 100
        bar_length = 40
        filled = int(bar_length * self.current / self.total) if self.total > 0 else bar_length
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r{self.description}: |{bar}| {percent:.1f}% ({self.current}/{self.total})", end="")
        if self.current >= self.total:
            print()  # 换行

    def finish(self):
        """完成进度"""
        if self.current < self.total:
            self.current = self.total
            self.update(0)
