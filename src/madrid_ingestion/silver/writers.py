"""Preparación y despacho de estrategias de escritura Silver."""

from __future__ import annotations

from typing import Any

from madrid_ingestion.config.models import SilverConfig
from madrid_ingestion.core.context import RunContext
from madrid_ingestion.writers.delta_writer import write_silver_delta


def write_silver(
    spark: Any, df: Any, context: RunContext, config: SilverConfig
) -> None:
    """Ejecuta la estrategia Silver configurada."""
    write_silver_delta(
        spark=spark,
        df=df,
        table_name=context.table,
        target_path=context.target_path,
        strategy=config.write_strategy,
        business_keys=config.business_keys,
        partition_by=config.partition_by,
        overwrite_schema=config.overwrite_schema,
    )
