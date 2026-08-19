"""Logging estándar, sin dependencias de observabilidad externas."""

from __future__ import annotations

import logging

LOGGER_NAME = "madrid_ingestion"


def configure_logging(level: str = "INFO") -> None:
    """Configura el logger del paquete respetando handlers ya instalados."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Devuelve el logger raíz del paquete o uno de sus descendientes."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
