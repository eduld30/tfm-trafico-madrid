"""Preparación y despacho de estrategias de escritura Silver."""

from __future__ import annotations

from typing import Any

from madrid_ingestion.config.models import SilverConfig
from madrid_ingestion.core.context import RunContext
from madrid_ingestion.writers.delta_writer import write_silver_delta


def apply_incremental_filter(
    df: Any, config: SilverConfig, context: RunContext
) -> Any:
    """Restringe los datos antes de un append para evitar reprocesos."""
    if config.write_strategy != "append":
        return df
    assert config.incremental is not None
    from pyspark.sql import functions as F

    if config.incremental.type == "ingestion_run_id":
        return df.filter(F.col("_ingestion_run_id") == context.run_id)
    assert config.incremental.condition is not None
    return df.filter(F.expr(config.incremental.condition))


def write_silver(
    spark: Any, df: Any, context: RunContext, config: SilverConfig
) -> None:
    """Ejecuta la estrategia Silver configurada."""
    filtered_df = apply_incremental_filter(df, config, context)
    write_silver_delta(
        spark=spark,
        df=filtered_df,
        table_name=context.table,
        target_path=context.target_path,
        strategy=config.write_strategy,
        business_keys=config.business_keys,
        partition_by=config.partition_by,
        overwrite_schema=config.overwrite_schema,
    )
