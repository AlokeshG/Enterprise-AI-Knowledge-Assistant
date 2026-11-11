"""
Centralized logging configuration.

Provides a single get_logger() factory so every module logs in a consistent
format, instead of each file configuring its own handlers.
"""
import logging
import sys
from backend.config.settings import get_settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.log_level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)
