"""Motor configurable de ingesta Medallion para datos de Madrid."""

from madrid_ingestion.core.context import RunResult
from madrid_ingestion.runner import run_dataset

__all__ = ["RunResult", "run_dataset"]
__version__ = "0.2.0"
