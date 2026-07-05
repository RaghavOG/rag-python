"""Structured logging helpers for rag-python."""
from __future__ import annotations

import logging

PACKAGE_LOGGER = "rag_python"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger (default: ``rag_python``)."""
    if name:
        return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")
    return logging.getLogger(PACKAGE_LOGGER)


def configure_logging(level: int = logging.INFO) -> None:
    """Enable default console logging for the package (optional convenience)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
