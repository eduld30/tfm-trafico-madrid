"""Primitivas batch para tablas Delta externas."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from madrid_ingestion.core.exceptions import WriteStrategyError
from madrid_ingestion.writers.table_manager import TableManager


def _validate_columns(df: Any, columns: Sequence[str], purpose: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise WriteStrategyError(
            f"Faltan columnas para {purpose}: {', '.join(missing)}."
        )


def write_initial_delta(
    df: Any,
    target_path: str,
    mode: str,
    partition_by: Sequence[str],
    overwrite_schema: bool = False,
) -> None:
    """Escribe un DataFrame batch en una ubicación Delta."""
    writer = df.write.format("delta").mode(mode)
    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")
    if partition_by:
        _validate_columns(df, partition_by, "particionado")
        writer = writer.partitionBy(*partition_by)
    writer.save(target_path)


def merge_delta(
    spark: Any,
    df: Any,
    target_path: str,
    business_keys: Sequence[str],
) -> None:
    """Actualiza e inserta por claves de negocio mediante Delta Lake."""
    _validate_columns(df, business_keys, "merge")
    try:
        from delta.tables import DeltaTable
    except ImportError as exc:
        raise WriteStrategyError(
            "La estrategia merge requiere las APIs Delta del Databricks Runtime."
        ) from exc

    def escaped(alias: str, column: str) -> str:
        return f"{alias}.`{column.replace('`', '``')}`"

    condition = " AND ".join(
        f"{escaped('t', key)} <=> {escaped('s', key)}" for key in business_keys
    )
    (
        DeltaTable.forPath(spark, target_path)
        .alias("t")
        .merge(df.alias("s"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def write_silver_delta(
    spark: Any,
    df: Any,
    table_name: str,
    target_path: str,
    strategy: str,
    business_keys: Sequence[str] = (),
    partition_by: Sequence[str] = (),
    overwrite_schema: bool = False,
) -> None:
    """Selecciona la estrategia y garantiza el registro de la tabla."""
    manager = TableManager(spark)
    exists = manager.table_exists(table_name)
    if exists:
        manager.validate_table_location(table_name, target_path)
    if strategy == "overwrite":
        write_initial_delta(
            df, target_path, "overwrite", partition_by, overwrite_schema
        )
    elif strategy == "append":
        write_initial_delta(df, target_path, "append", partition_by)
    elif strategy == "merge":
        if not business_keys:
            raise WriteStrategyError("La estrategia merge requiere business_keys.")
        if exists:
            merge_delta(spark, df, target_path, business_keys)
        else:
            write_initial_delta(df, target_path, "overwrite", partition_by)
    else:
        raise WriteStrategyError(f"Estrategia de escritura desconocida: {strategy!r}.")
    if not exists:
        manager.ensure_external_table(table_name, target_path)
