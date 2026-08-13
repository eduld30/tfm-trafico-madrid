"""Registro explícito de transformaciones declarativas Spark."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from madrid_ingestion.config.models import TransformationConfig
from madrid_ingestion.core.columns import (
    normalize_dataframe_columns,
    rename_dataframe_columns,
)
from madrid_ingestion.core.context import RunContext
from madrid_ingestion.core.exceptions import (
    IngestionError,
    UnsupportedTransformationError,
)
from madrid_ingestion.core.naming import build_table_name


def _require_columns(df: Any, columns: Sequence[str], transformation: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise IngestionError(
            f"{transformation}: no existen las columnas {', '.join(missing)}."
        )


def normalize_column_names(
    df: Any, config: TransformationConfig, context: RunContext
) -> Any:
    del config, context
    return normalize_dataframe_columns(df)


def rename_columns(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, dict)
    return rename_dataframe_columns(df, config.columns)


def select_columns(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, list)
    _require_columns(df, config.columns, config.type)
    return df.select(*config.columns)


def drop_columns(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, list)
    _require_columns(df, config.columns, config.type)
    return df.drop(*config.columns)


def cast_columns(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, dict)
    _require_columns(df, list(config.columns), config.type)
    from pyspark.sql import functions as F

    result = df
    for column, data_type in config.columns.items():
        result = result.withColumn(column, F.col(column).cast(data_type))
    return result


def trim_columns(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, list)
    _require_columns(df, config.columns, config.type)
    from pyspark.sql import functions as F

    result = df
    for column in config.columns:
        result = result.withColumn(column, F.trim(F.col(column)))
    return result


def upper_columns(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, list)
    _require_columns(df, config.columns, config.type)
    from pyspark.sql import functions as F

    result = df
    for column in config.columns:
        result = result.withColumn(column, F.upper(F.col(column)))
    return result


def strip_accents(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, list)
    _require_columns(df, config.columns, config.type)
    from pyspark.sql import functions as F

    accented = "áéíóúÁÉÍÓÚñÑüÜ"
    plain = "aeiouAEIOUnNuU"
    result = df
    for column in config.columns:
        result = result.withColumn(column, F.translate(F.col(column), accented, plain))
    return result


def empty_to_null(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, list)
    _require_columns(df, config.columns, config.type)
    from pyspark.sql import functions as F

    result = df
    for column in config.columns:
        result = result.withColumn(
            column,
            F.when(F.trim(F.col(column)) == "", F.lit(None)).otherwise(F.col(column)),
        )
    return result


def replace_values(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert config.column is not None and config.values is not None
    _require_columns(df, [config.column], config.type)
    return df.replace(to_replace=config.values, subset=[config.column])


def regex_replace(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert isinstance(config.columns, list)
    assert config.pattern is not None and config.replacement is not None
    _require_columns(df, config.columns, config.type)
    from pyspark.sql import functions as F

    result = df
    for column in config.columns:
        result = result.withColumn(
            column,
            F.regexp_replace(F.col(column), config.pattern, config.replacement),
        )
    return result


def filter_rows(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert config.condition is not None
    from pyspark.sql import functions as F

    return df.filter(F.expr(config.condition))


def add_literal(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert config.column is not None
    if config.column in df.columns:
        raise IngestionError(f"add_literal no puede sobrescribir {config.column!r}.")
    from pyspark.sql import functions as F

    return df.withColumn(config.column, F.lit(config.value))


def parse_timestamp(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert config.target_column is not None and config.format is not None
    source_columns = config.source_columns or [config.source_column]
    sources = [column for column in source_columns if column is not None]
    _require_columns(df, sources, config.type)
    from pyspark.sql import functions as F

    source_value = (
        F.col(sources[0])
        if len(sources) == 1
        else F.concat_ws(" ", *(F.col(column) for column in sources))
    )
    return df.withColumn(
        config.target_column, F.to_timestamp(source_value, config.format)
    )


def parse_date(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert config.source_column is not None
    assert config.target_column is not None and config.format is not None
    _require_columns(df, [config.source_column], config.type)
    from pyspark.sql import functions as F

    return df.withColumn(
        config.target_column,
        F.to_date(F.col(config.source_column), config.format),
    )


def hourly_wide_to_long(
    df: Any, config: TransformationConfig, context: RunContext
) -> Any:
    """Convierte pares Hxx/Vxx diarios en observaciones horarias."""
    del context
    assert config.year_column is not None
    assert config.month_column is not None
    assert config.day_column is not None
    assert config.value_prefix is not None
    assert config.validity_prefix is not None
    assert config.value_column is not None
    assert config.validity_column is not None
    assert config.timestamp_column is not None
    hours = config.hours or 24
    value_columns = [
        f"{config.value_prefix}{hour:02d}" for hour in range(1, hours + 1)
    ]
    validity_columns = [
        f"{config.validity_prefix}{hour:02d}" for hour in range(1, hours + 1)
    ]
    _require_columns(
        df,
        [
            config.year_column,
            config.month_column,
            config.day_column,
            *value_columns,
            *validity_columns,
        ],
        config.type,
    )
    from pyspark.sql import functions as F

    wide_columns = set(value_columns) | set(validity_columns)
    base_columns = [column for column in df.columns if column not in wide_columns]
    record_alias = "__hourly_record"
    hour_column = "__hour_number"
    reserved = {
        record_alias,
        hour_column,
        config.value_column,
        config.validity_column,
        config.timestamp_column,
    }
    collisions = sorted(reserved.intersection(base_columns))
    if collisions:
        raise IngestionError(
            "hourly_wide_to_long no puede sobrescribir columnas: "
            + ", ".join(collisions)
        )
    records = [
        F.struct(
            F.lit(hour).alias(hour_column),
            F.col(value_columns[hour - 1]).cast("double").alias(config.value_column),
            F.col(validity_columns[hour - 1]).alias(config.validity_column),
        )
        for hour in range(1, hours + 1)
    ]
    result = (
        df.select(
            *base_columns,
            F.explode(F.array(*records)).alias(record_alias),
        )
        .select(
            *base_columns,
            F.col(f"{record_alias}.{hour_column}").alias(hour_column),
            F.col(f"{record_alias}.{config.value_column}").alias(config.value_column),
            F.col(f"{record_alias}.{config.validity_column}").alias(
                config.validity_column
            ),
        )
        .withColumn(
            config.timestamp_column,
            F.make_timestamp(
                F.col(config.year_column).cast("int"),
                F.col(config.month_column).cast("int"),
                F.col(config.day_column).cast("int"),
                F.col(hour_column) + F.lit(config.hour_offset),
                F.lit(0),
                F.lit(0),
            ),
        )
        .drop(hour_column)
    )
    return result


def deduplicate(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    del context
    assert config.keys is not None and config.order_by is not None
    order_columns = [item.column for item in config.order_by]
    _require_columns(df, [*config.keys, *order_columns], config.type)
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    ordering = [
        F.col(item.column).desc() if item.direction == "desc" else F.col(item.column).asc()
        for item in config.order_by
    ]
    window = Window.partitionBy(*config.keys).orderBy(*ordering)
    marker = "__madrid_ingestion_row_number"
    if marker in df.columns:
        raise IngestionError(f"deduplicate: columna reservada ya existente: {marker}.")
    return (
        df.withColumn(marker, F.row_number().over(window))
        .filter(F.col(marker) == 1)
        .drop(marker)
    )


def lookup_join(df: Any, config: TransformationConfig, context: RunContext) -> Any:
    assert config.lookup_table is not None
    assert config.conditions is not None
    assert isinstance(config.select, dict)
    _require_columns(df, list(config.conditions), config.type)
    lookup_table = build_table_name(
        f"{context.environment}_{config.lookup_table.layer}",
        config.lookup_table.source,
        config.lookup_table.dataset,
    )
    lookup_df = df.sparkSession.table(lookup_table)
    lookup_columns = [*config.conditions.values(), *config.select.values()]
    _require_columns(lookup_df, lookup_columns, f"{config.type} lookup")
    collisions = sorted(set(config.select).intersection(df.columns))
    if collisions:
        raise IngestionError(
            "lookup_join no puede sobrescribir columnas: " + ", ".join(collisions)
        )
    from pyspark.sql import functions as F

    source_alias = df.alias("source")
    lookup_alias = lookup_df.alias("lookup")
    predicates = [
        F.col(f"source.`{source}`") == F.col(f"lookup.`{lookup}`")
        for source, lookup in config.conditions.items()
    ]
    condition = predicates[0]
    for predicate in predicates[1:]:
        condition = condition & predicate
    joined = source_alias.join(
        lookup_alias, condition, config.join_type or "left"
    )
    additions = [
        F.col(f"lookup.`{lookup_column}`").alias(target_column)
        for target_column, lookup_column in config.select.items()
    ]
    return joined.select("source.*", *additions)


TransformationFunction = Callable[[Any, TransformationConfig, RunContext], Any]

TRANSFORMATIONS: dict[str, TransformationFunction] = {
    "normalize_column_names": normalize_column_names,
    "rename": rename_columns,
    "select": select_columns,
    "drop": drop_columns,
    "cast": cast_columns,
    "trim": trim_columns,
    "upper": upper_columns,
    "strip_accents": strip_accents,
    "empty_to_null": empty_to_null,
    "replace_values": replace_values,
    "regex_replace": regex_replace,
    "filter": filter_rows,
    "add_literal": add_literal,
    "parse_timestamp": parse_timestamp,
    "parse_date": parse_date,
    "hourly_wide_to_long": hourly_wide_to_long,
    "deduplicate": deduplicate,
    "lookup_join": lookup_join,
}


def apply_transformations(
    df: Any,
    transformations: list[TransformationConfig],
    context: RunContext,
) -> Any:
    """Aplica en orden las transformaciones configuradas."""
    result = df
    for config in transformations:
        try:
            function = TRANSFORMATIONS[config.type]
        except KeyError as exc:
            raise UnsupportedTransformationError(
                f"Transformación no soportada: {config.type!r}."
            ) from exc
        result = function(result, config, context)
    return result
