"""Orquestación de una única ejecución Silver."""

from __future__ import annotations

from typing import Any

from madrid_ingestion.config.models import SilverConfig
from madrid_ingestion.core.context import RunContext
from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.silver.transformations import apply_transformations
from madrid_ingestion.silver.writers import write_silver


def process_silver(spark: Any, context: RunContext, config: SilverConfig) -> None:
    """Lee Bronze, transforma y consolida una tabla Silver."""
    if context.source_table is None:
        raise IngestionError("Silver requiere una tabla Bronze de origen.")
    bronze_df = spark.table(context.source_table)
    transformed_df = apply_transformations(
        bronze_df, config.transformations, context
    )
    if "_silver_processed_timestamp" in transformed_df.columns:
        raise IngestionError(
            "La columna de origen colisiona con _silver_processed_timestamp."
        )
    from pyspark.sql import functions as F

    result_df = transformed_df.withColumn(
        "_silver_processed_timestamp", F.current_timestamp()
    )
    write_silver(spark, result_df, context, config)
