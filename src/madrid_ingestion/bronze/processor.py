"""Orquestación de una única ejecución Bronze."""

from __future__ import annotations

from typing import Any

from madrid_ingestion.bronze.autoloader import write_available_now
from madrid_ingestion.bronze.metadata import add_bronze_metadata
from madrid_ingestion.bronze.transformations import apply_bronze_transformations
from madrid_ingestion.config.models import BronzeConfig
from madrid_ingestion.core.columns import validate_technical_column_names
from madrid_ingestion.core.context import RunContext
from madrid_ingestion.observability.logger import get_logger
from madrid_ingestion.readers import get_reader
from madrid_ingestion.writers.table_manager import TableManager


def _autoloader_options(config: BronzeConfig) -> dict[str, str]:
    options = {
        "cloudFiles.schemaEvolutionMode": config.autoloader_options.schema_evolution_mode,
        "cloudFiles.rescuedDataColumn": config.autoloader_options.rescued_data_column,
    }
    options.update(config.autoloader_options.extra_options)
    return options


def process_bronze(spark: Any, context: RunContext, config: BronzeConfig) -> None:
    """Ingiere los ficheros nuevos de landing y registra la tabla externa."""
    if context.schema_path is None or context.checkpoint_path is None:
        raise ValueError("Bronze requiere schema_path y checkpoint_path.")
    table_manager = TableManager(spark)
    table_exists = table_manager.table_exists(context.table)
    if table_exists:
        table_manager.validate_table_location(context.table, context.target_path)
    reader = get_reader(config.format)
    stream_df = reader.read_stream(
        spark=spark,
        source_path=context.source_path,
        schema_path=context.schema_path,
        options=config.reader_options,
        autoloader_options=_autoloader_options(config),
    )
    source_columns = stream_df.columns
    transformed_df = apply_bronze_transformations(stream_df, config.transformations)
    validate_technical_column_names(transformed_df, "Bronze")
    if source_columns != transformed_df.columns:
        mapping = dict(zip(source_columns, transformed_df.columns, strict=True))
        changed = {source: target for source, target in mapping.items() if source != target}
        get_logger("bronze").info(
            "Columnas Bronze normalizadas run_id=%s dataset=%s mapping=%s",
            context.run_id,
            context.dataset,
            changed,
        )
    enriched_df = add_bronze_metadata(transformed_df, context.run_id)
    write_available_now(
        enriched_df,
        target_path=context.target_path,
        checkpoint_path=context.checkpoint_path,
    )
    if not table_exists:
        table_manager.ensure_external_table(context.table, context.target_path)
