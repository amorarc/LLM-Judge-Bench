import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    _RICH = True
except ImportError:
    _RICH = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    if _RICH:
        handler = RichHandler(rich_tracebacks=True, markup=True)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
