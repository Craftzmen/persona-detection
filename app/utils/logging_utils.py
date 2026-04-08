"""Logging helpers used across the project."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from app.config import (
    LOG_FILE_BACKUP_COUNT,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_NAME,
    LOG_LEVEL,
    LOGS_DIR,
    ensure_directories,
)


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Configure and return a module logger.

    The first call configures the root logger with a consistent format. Later
    calls reuse the existing configuration.
    """

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        ensure_directories()
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)

        file_handler = RotatingFileHandler(
            LOGS_DIR / LOG_FILE_NAME,
            maxBytes=max(1024, LOG_FILE_MAX_BYTES),
            backupCount=max(1, LOG_FILE_BACKUP_COUNT),
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        root_logger.setLevel(level)
        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)

    return logging.getLogger(name if name else __name__)
