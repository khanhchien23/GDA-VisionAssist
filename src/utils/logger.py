"""
GDA-VisionAssist - Logging Utilities
Cấu hình logging cho hệ thống GDA.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


# Default format
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", 
                   log_file: Optional[str] = None,
                   log_format: str = LOG_FORMAT):
    """
    Cấu hình logging cho toàn bộ ứng dụng.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Đường dẫn file log (optional)
        log_format: Format string
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = []
    
    # Console handler (UTF-8)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format, LOG_DATE_FORMAT))
    handlers.append(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format, LOG_DATE_FORMAT))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True
    )


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Lấy logger cho một module.
    
    Args:
        name: Tên module (thường dùng __name__)
        level: Log level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Thêm handler nếu chưa có
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)
    
    return logger


class PerformanceLogger:
    """Logger chuyên cho đo performance."""
    
    def __init__(self, name: str = "perf"):
        self.logger = get_logger(f"gda.{name}")
        self.timings = {}
    
    def log_timing(self, component: str, elapsed_ms: float):
        """Log thời gian xử lý."""
        if component not in self.timings:
            self.timings[component] = []
        self.timings[component].append(elapsed_ms)
        self.logger.debug(f"{component}: {elapsed_ms:.1f}ms")
    
    def get_summary(self) -> dict:
        """Lấy tổng kết performance."""
        import numpy as np
        
        summary = {}
        for component, times in self.timings.items():
            arr = np.array(times)
            summary[component] = {
                "count": len(arr),
                "mean_ms": float(arr.mean()),
                "std_ms": float(arr.std()),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
            }
        return summary
    
    def print_summary(self):
        """In tổng kết ra console."""
        summary = self.get_summary()
        print("\n📊 Performance Summary:")
        print(f"{'Component':<30} {'Mean (ms)':<12} {'Count':<8}")
        print("-" * 50)
        for comp, stats in summary.items():
            print(f"{comp:<30} {stats['mean_ms']:<12.1f} {stats['count']:<8}")
