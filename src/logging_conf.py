"""
Logging setup.

Streamlit reruns the script on every interaction, so configuration must be
idempotent — otherwise handlers stack up and every line gets logged N times.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    from src.config import settings

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(level or settings.log_level)
    root.handlers.clear()
    root.addHandler(handler)

    # These are noisy at DEBUG and drown out our own logs.
    for noisy in ("httpx", "httpcore", "urllib3", "botocore", "boto3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
