"""
Rich-based logging utilities.
==============================
Configures structured logging with Rich formatting for pipeline output.
"""

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with Rich formatting.

    Parameters
    ----------
    level : int
        Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
