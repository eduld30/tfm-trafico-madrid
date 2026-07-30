"""Metadatos técnicos añadidos a cada registro Bronze."""

from __future__ import annotations

from typing import Any

from madrid_ingestion.core.exceptions import IngestionError

BRONZE_METADATA_COLUMNS = frozenset(
    {
        "_ingestion_timestamp",
        "_ingestion_run_id",
        "_source_file",
        "_source_file_modification_time",
    }
)


def add_bronze_metadata(df: Any, run_id: str) -> Any:
    """Añade metadatos de ingestión sin sobrescribir columnas de origen."""
    collisions = BRONZE_METADATA_COLUMNS.intersection(df.columns)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise IngestionError(f"Las columnas de origen colisionan con metadatos Bronze: {names}.")

    from pyspark.sql import functions as F

    return (
        df.withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_ingestion_run_id", F.lit(run_id))
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn(
            "_source_file_modification_time",
            F.col("_metadata.file_modification_time"),
        )
    )
