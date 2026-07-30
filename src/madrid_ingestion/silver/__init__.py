"""Procesamiento y transformaciones Silver."""

from madrid_ingestion.silver.processor import process_silver
from madrid_ingestion.silver.transformations import apply_transformations

__all__ = ["apply_transformations", "process_silver"]
