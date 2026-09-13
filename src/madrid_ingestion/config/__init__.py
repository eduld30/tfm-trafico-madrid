"""Carga y modelos de configuración declarativa."""

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ingestion.config.models import EnvironmentConfig, SourceConfig

__all__ = ["ConfigLoader", "EnvironmentConfig", "SourceConfig"]
