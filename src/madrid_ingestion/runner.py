"""API pública y orquestación de una ejecución dataset/capa."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ingestion.config.models import DatasetConfig, EnvironmentConfig
from madrid_ingestion.config.validators import validate_layer_enabled
from madrid_ingestion.core.context import RunContext, RunResult
from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.core.naming import build_table_name
from madrid_ingestion.core.paths import (
    build_adls_uri,
    build_checkpoint_path,
    build_schema_path,
    build_table_path,
)
from madrid_ingestion.observability.logger import configure_logging, get_logger

Layer = Literal["bronze", "silver"]


def build_run_context(
    environment: EnvironmentConfig,
    dataset_config: DatasetConfig,
    layer: Layer,
    source: str,
    run_id: str,
) -> RunContext:
    """Resuelve todos los nombres y rutas de una ejecución."""
    account = environment.storage.account_name
    catalog = getattr(environment.catalogs, layer)
    table = build_table_name(catalog, source, dataset_config.name)

    if layer == "bronze":
        layer_config = dataset_config.bronze
        source_path = build_adls_uri(
            account, environment.storage.containers.landing, layer_config.source_path
        )
        target_path = build_table_path(
            account, environment.storage.containers.bronze, layer_config.target_path
        )
        checkpoint_path = build_checkpoint_path(
            target_path, environment.paths.checkpoint_dir
        )
        schema_path = build_schema_path(
            checkpoint_path, environment.paths.schema_location_mode
        )
        source_table = None
    else:
        layer_config = dataset_config.silver
        explicit_source = layer_config.source_table
        source_source = explicit_source.source if explicit_source else source
        source_dataset = (
            explicit_source.dataset if explicit_source else dataset_config.name
        )
        source_table = build_table_name(
            environment.catalogs.bronze, source_source, source_dataset
        )
        source_path = source_table
        target_path = build_table_path(
            account, environment.storage.containers.silver, layer_config.target_path
        )
        checkpoint_path = None
        schema_path = None

    return RunContext(
        environment=environment.environment,
        layer=layer,
        source=source,
        dataset=dataset_config.name,
        run_id=run_id,
        catalog=catalog,
        schema=source,
        table=table,
        source_table=source_table,
        source_path=source_path,
        target_path=target_path,
        checkpoint_path=checkpoint_path,
        schema_path=schema_path,
    )


def _active_spark_session(spark: Any | None) -> Any:
    if spark is not None:
        return spark
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise IngestionError(
            "PySpark no está disponible. Ejecute el motor en Databricks o inyecte "
            "una SparkSession."
        ) from exc
    session = SparkSession.getActiveSession()
    if session is None:
        raise IngestionError("No active SparkSession was found.")
    return session


def run_dataset(
    environment: str,
    layer: Layer,
    source: str,
    dataset: str,
    *,
    spark: Any | None = None,
    run_id: str | None = None,
    config_root: str | Path | None = None,
    log_level: str | None = None,
) -> RunResult:
    """Ejecuta una capa de un dataset y devuelve un resumen inmutable."""
    if layer not in {"bronze", "silver"}:
        raise IngestionError(f"Capa no soportada: {layer!r}.")
    loader = ConfigLoader(config_root)
    environment_config = loader.load_environment(environment)
    source_config, dataset_config = loader.load_dataset(source, dataset)
    validate_layer_enabled(source_config, dataset_config, layer)
    effective_run_id = run_id or str(uuid4())
    configure_logging(log_level or environment_config.runtime.log_level)
    logger = get_logger("runner")
    context = build_run_context(
        environment_config,
        dataset_config,
        layer,
        source_config.source,
        effective_run_id,
    )
    log_context = (
        f"run_id={context.run_id} env={context.environment} layer={context.layer} "
        f"source={context.source} dataset={context.dataset} target={context.table}"
    )
    logger.info("Inicio de ejecución %s", log_context)
    spark_session = _active_spark_session(spark)
    try:
        if layer == "bronze":
            from madrid_ingestion.bronze.processor import process_bronze

            process_bronze(spark_session, context, dataset_config.bronze)
        else:
            from madrid_ingestion.silver.processor import process_silver

            process_silver(spark_session, context, dataset_config.silver)
    except Exception:
        logger.exception("Error de ejecución %s", log_context)
        raise
    logger.info("Fin de ejecución %s", log_context)
    return RunResult(
        run_id=context.run_id,
        environment=context.environment,
        layer=context.layer,
        source=context.source,
        dataset=context.dataset,
        target_table=context.table,
        target_path=context.target_path,
        status="SUCCEEDED",
    )
