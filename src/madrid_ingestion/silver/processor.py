"""Orquestación de una única ejecución Silver."""

from __future__ import annotations

from typing import Any

from madrid_ingestion.config.models import SilverConfig
from madrid_ingestion.core.context import RunContext
from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.silver.transformations import apply_transformations
from madrid_ingestion.silver.writers import write_silver


def _read_incremental_bronze(
    spark: Any, bronze_df: Any, context: RunContext, strategy: str
) -> Any | None:
    """Selecciona las filas Bronze posteriores al último dato procesado."""
    if spark.catalog.tableExists(context.table):
        silver_df = spark.table(context.table)
        watermark = silver_df.selectExpr(
            "MAX(_ingestion_timestamp) AS watermark"
        ).first()["watermark"]
        if watermark is not None:
            bronze_df = bronze_df.filter(
                bronze_df["_ingestion_timestamp"] > watermark
            )

    if bronze_df.isEmpty():
        return None

    if strategy == "overwrite":
        latest_file_date = bronze_df.selectExpr(
            "MAX(_file_date) AS latest_file_date"
        ).first()["latest_file_date"]
        if latest_file_date is None:
            raise IngestionError(
                "No se pudo determinar _file_date para el overwrite Silver."
            )
        bronze_df = bronze_df.filter(bronze_df["_file_date"] == latest_file_date)

    return bronze_df


def process_silver(spark: Any, context: RunContext, config: SilverConfig) -> None:
    """Lee Bronze, transforma y consolida una tabla Silver."""
    if context.source_table is None:
        raise IngestionError("Silver requiere una tabla Bronze de origen.")
    bronze_df = spark.table(context.source_table)
    incremental_df = _read_incremental_bronze(
        spark, bronze_df, context, config.write_strategy
    )
    if incremental_df is None:
        return
    transformed_df = apply_transformations(
        incremental_df, config.transformations, context
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
