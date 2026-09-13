"""Primitivas batch para tablas Delta externas."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
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


def _sql_literal(value: Any) -> str:
    """Representa un valor de partición como literal SQL seguro."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | Decimal):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WriteStrategyError(
                "replace_partitions no admite valores de partición no finitos."
            )
        return repr(value)
    if isinstance(value, datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, date):
        return f"DATE '{value.isoformat()}'"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _build_replace_where_predicate(
    partition_by: Sequence[str], partition_rows: Sequence[Any]
) -> str:
    """Construye el predicado que limita el overwrite a las particiones recibidas."""
    if not partition_rows:
        raise WriteStrategyError(
            "replace_partitions no recibió ninguna partición para escribir."
        )

    clauses = []
    for row in partition_rows:
        predicates = []
        for column in partition_by:
            identifier = f"`{column.replace('`', '``')}`"
            value = row[column]
            if value is None:
                predicates.append(f"{identifier} IS NULL")
            else:
                predicates.append(f"{identifier} = {_sql_literal(value)}")
        clauses.append(" AND ".join(predicates))
    return " OR ".join(f"({clause})" for clause in clauses)


def replace_delta_partitions(
    df: Any,
    target_path: str,
    partition_by: Sequence[str],
    overwrite_schema: bool = False,
) -> None:
    """Reemplaza atómicamente solo las particiones presentes en el DataFrame."""
    if not partition_by:
        raise WriteStrategyError(
            "La estrategia replace_partitions requiere columnas de partición."
        )
    _validate_columns(df, partition_by, "reemplazo de particiones")
    partition_rows = df.select(*partition_by).distinct().collect()
    predicate = _build_replace_where_predicate(partition_by, partition_rows)
    writer = (
        df.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", predicate)
    )
    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")
    writer.partitionBy(*partition_by).save(target_path)


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
    elif strategy == "replace_partitions":
        replace_delta_partitions(
            df,
            target_path,
            partition_by,
            overwrite_schema=overwrite_schema,
        )
    else:
        raise WriteStrategyError(f"Estrategia de escritura desconocida: {strategy!r}.")
    if not exists:
        manager.ensure_external_table(table_name, target_path)
