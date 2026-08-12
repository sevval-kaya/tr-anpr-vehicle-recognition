"""Shared logging configuration for scripts and modules."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with consistent formatting.

    Safe to call repeatedly (e.g. once per module): root handler setup
    only runs once per process.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
