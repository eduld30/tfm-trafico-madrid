from collections.abc import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from madrid_ml.contract import (
    AIR_MAGNITUDES,
    EXPECTED_DISTRICTS,
    EXPECTED_ROWS,
    FEATURE_BASE_COLUMNS,
    GRID_END,
    GRID_START,
    INPUT_SUFFIXES,
    LABEL_BASE_COLUMNS,
    TRAFFIC_COLUMNS,
    WEATHER_MAGNITUDES,
    Magnitude,
    magnitude_columns,
)


class SnapshotContractError(RuntimeError):
    """Raised when a Silver input cannot satisfy the approved ML snapshot contract."""


def _require_columns(df: DataFrame, columns: Sequence[str], logical_table: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise SnapshotContractError(
            f"{logical_table}: missing required columns {', '.join(missing)}"
        )


def build_district_hour_grid(spark: SparkSession, districts_df: DataFrame) -> DataFrame:
    """Build the fixed v1 district-hour grid from the Silver district dimension."""
    logical_table = INPUT_SUFFIXES["districts"]
    _require_columns(districts_df, ("cod_dis",), logical_table)

    districts = districts_df.select(F.col("cod_dis").cast("int").alias("cod_distrito"))
    if districts.where(F.col("cod_distrito").isNull()).first() is not None:
        raise SnapshotContractError(
            f"{logical_table}: column cod_dis contains null or non-integer values"
        )

    duplicate_codes = [
        row.cod_distrito
        for row in (
            districts.groupBy("cod_distrito")
            .count()
            .where(F.col("count") > 1)
            .orderBy("cod_distrito")
            .limit(10)
            .collect()
        )
    ]
    if duplicate_codes:
        raise SnapshotContractError(
            f"{logical_table}: column cod_dis contains duplicate values; "
            f"examples={duplicate_codes}"
        )

    district_count = districts.count()
    if district_count != EXPECTED_DISTRICTS:
        raise SnapshotContractError(
            f"{logical_table}: expected {EXPECTED_DISTRICTS} districts in cod_dis, "
            f"found {district_count}"
        )

    hours = spark.range(1).select(
        F.explode(
            F.sequence(
                F.to_timestamp(F.lit(GRID_START)),
                F.to_timestamp(F.lit(GRID_END)),
                F.expr("INTERVAL 1 HOUR"),
            )
        ).alias("feature_hour")
    )
    grid = districts.crossJoin(hours).select("cod_distrito", "feature_hour")

    row_count = grid.count()
    if row_count != EXPECTED_ROWS:
        raise SnapshotContractError(
            "district_hour_grid: expected "
            f"{EXPECTED_ROWS} rows from cod_distrito and feature_hour, found {row_count}"
        )
    return grid


def build_accident_labels(accidents_df: DataFrame, grid_df: DataFrame) -> DataFrame:
    """Build one-hour-ahead accident labels on an existing district-hour grid."""
    accidents_table = INPUT_SUFFIXES["accidents"]
    _require_columns(
        accidents_df,
        ("num_expediente", "fecha_hora", "cod_distrito"),
        accidents_table,
    )
    _require_columns(grid_df, ("cod_distrito", "feature_hour"), "district_hour_grid")

    if accidents_df.where(F.col("num_expediente").isNull()).first() is not None:
        raise SnapshotContractError(
            f"{accidents_table}: required column num_expediente contains null values"
        )

    invalid_accident_ids = (
        accidents_df.where(
            F.col("fecha_hora").isNull() | F.col("cod_distrito").isNull()
        )
        .select("num_expediente")
        .distinct()
    )
    valid_accidents = accidents_df.join(
        invalid_accident_ids,
        on="num_expediente",
        how="left_anti",
    )

    grid_districts = grid_df.select("cod_distrito").distinct()
    unknown_districts = [
        row.cod_distrito
        for row in (
            valid_accidents.select("cod_distrito")
            .distinct()
            .join(grid_districts, on="cod_distrito", how="left_anti")
            .orderBy("cod_distrito")
            .limit(10)
            .collect()
        )
    ]
    if unknown_districts:
        raise SnapshotContractError(
            f"{accidents_table}: column cod_distrito contains values absent from "
            f"district_hour_grid; examples={unknown_districts}"
        )

    event_shape = valid_accidents.groupBy("num_expediente").agg(
        F.countDistinct("cod_distrito").alias("district_count"),
        F.countDistinct(F.date_trunc("hour", F.col("fecha_hora"))).alias(
            "hour_count"
        ),
    )
    conflicting_accidents = [
        row.num_expediente
        for row in (
            event_shape.where(
                (F.col("district_count") != 1) | (F.col("hour_count") != 1)
            )
            .select("num_expediente")
            .orderBy("num_expediente")
            .limit(10)
            .collect()
        )
    ]
    if conflicting_accidents:
        raise SnapshotContractError(
            f"{accidents_table}: columns cod_distrito and fecha_hora conflict within "
            f"num_expediente; examples={conflicting_accidents}"
        )

    accidents_by_event = valid_accidents.select(
        "num_expediente",
        F.col("cod_distrito").cast("int").alias("cod_distrito"),
        F.date_trunc("hour", F.col("fecha_hora")).alias("prediction_hour"),
    ).distinct()
    counts = accidents_by_event.groupBy("cod_distrito", "prediction_hour").agg(
        F.count(F.lit(1)).cast("long").alias("n_accidentes_next_hour")
    )

    grid_with_prediction = grid_df.select(
        F.col("cod_distrito").cast("int").alias("cod_distrito"),
        "feature_hour",
        (F.col("feature_hour") + F.expr("INTERVAL 1 HOUR")).alias(
            "prediction_hour"
        ),
    )
    return (
        grid_with_prediction.join(
            counts,
            on=["cod_distrito", "prediction_hour"],
            how="left",
        )
        .withColumn(
            "n_accidentes_next_hour",
            F.coalesce(F.col("n_accidentes_next_hour"), F.lit(0)).cast("long"),
        )
        .withColumn(
            "target_accident_next_hour",
            (F.col("n_accidentes_next_hour") > 0).cast("int"),
        )
        .select(
            "cod_distrito",
            "feature_hour",
            "prediction_hour",
            "n_accidentes_next_hour",
            "target_accident_next_hour",
        )
    )


def _prepare_feature_hours(df: DataFrame) -> DataFrame:
    return (
        df.where(F.col("fecha_hora").isNotNull() & F.col("distrito_cod").isNotNull())
        .withColumn("feature_hour", F.date_trunc("hour", F.col("fecha_hora")))
        .where(
            (F.col("feature_hour") >= F.to_timestamp(F.lit(GRID_START)))
            & (F.col("feature_hour") <= F.to_timestamp(F.lit(GRID_END)))
        )
        .withColumn("cod_distrito", F.col("distrito_cod").cast("int"))
    )


def aggregate_traffic(df: DataFrame) -> DataFrame:
    """Aggregate historical traffic measurements by district and source hour."""
    logical_table = INPUT_SUFFIXES["traffic"]
    _require_columns(
        df,
        ("fecha_hora", "distrito_cod", "intensidad", "ocupacion", "carga", "vmed"),
        logical_table,
    )

    hourly = _prepare_feature_hours(df)
    return hourly.groupBy("cod_distrito", "feature_hour").agg(
        F.avg("intensidad").cast("double").alias("trafico_intensidad_media"),
        F.avg("ocupacion").cast("double").alias("trafico_ocupacion_media"),
        F.avg("carga").cast("double").alias("trafico_carga_media"),
        F.avg("vmed").cast("double").alias("trafico_vmed_media"),
        F.count(F.lit(1)).cast("long").alias("trafico_puntos_n"),
    )


def aggregate_magnitudes(
    df: DataFrame,
    specs: tuple[Magnitude, ...],
) -> DataFrame:
    """Aggregate the approved weather or air magnitudes by district and source hour."""
    if specs == WEATHER_MAGNITUDES:
        logical_table = INPUT_SUFFIXES["weather"]
    elif specs == AIR_MAGNITUDES:
        logical_table = INPUT_SUFFIXES["air"]
    else:
        raise SnapshotContractError(
            "aggregate_magnitudes: specs must be WEATHER_MAGNITUDES or AIR_MAGNITUDES"
        )

    _require_columns(
        df,
        ("fecha_hora", "distrito_cod", "magnitud", "valor"),
        logical_table,
    )
    selected_codes = tuple(spec.code for spec in specs)
    hourly = _prepare_feature_hours(df).where(F.col("magnitud").isin(*selected_codes))

    observed_codes = {
        row.magnitud
        for row in hourly.select("magnitud").distinct().limit(len(specs)).collect()
    }
    missing_codes = [code for code in selected_codes if code not in observed_codes]
    if missing_codes:
        raise SnapshotContractError(
            f"{logical_table}: column magnitud is missing required codes {missing_codes}"
        )

    aggregations = [
        expression
        for spec in specs
        for expression in (
            F.avg(F.when(F.col("magnitud") == spec.code, F.col("valor"))).alias(
                spec.mean_column
            ),
            F.count(F.when(F.col("magnitud") == spec.code, F.col("valor")))
            .cast("long")
            .alias(spec.count_column),
        )
    ]
    return hourly.groupBy("cod_distrito", "feature_hour").agg(*aggregations)


def build_feature_snapshot(
    labels: DataFrame,
    traffic: DataFrame,
    weather: DataFrame,
    air: DataFrame,
) -> DataFrame:
    """Join hourly aggregates to labels while preserving every district-hour key."""
    keys = ("cod_distrito", "feature_hour")
    _require_columns(
        labels,
        tuple(name for name, _, _ in LABEL_BASE_COLUMNS),
        "accident_labels_hourly",
    )
    _require_columns(
        traffic,
        keys + tuple(name for name, _, _ in TRAFFIC_COLUMNS),
        "traffic_hourly_aggregate",
    )
    _require_columns(
        weather,
        keys + tuple(name for name, _, _ in magnitude_columns(WEATHER_MAGNITUDES)),
        "weather_hourly_aggregate",
    )
    _require_columns(
        air,
        keys + tuple(name for name, _, _ in magnitude_columns(AIR_MAGNITUDES)),
        "air_hourly_aggregate",
    )

    result = (
        labels.join(traffic, on=list(keys), how="left")
        .join(weather, on=list(keys), how="left")
        .join(air, on=list(keys), how="left")
        .withColumn("hora_dia", F.hour("feature_hour").cast("integer"))
        .withColumn("dia_semana", F.dayofweek("feature_hour").cast("integer"))
        .withColumn("mes", F.month("feature_hour").cast("integer"))
    )
    count_columns = (
        "trafico_puntos_n",
        *(spec.count_column for spec in WEATHER_MAGNITUDES),
        *(spec.count_column for spec in AIR_MAGNITUDES),
    )
    for column in count_columns:
        result = result.withColumn(
            column,
            F.coalesce(F.col(column), F.lit(0).cast("long")),
        )

    return result.select(*(name for name, _, _ in FEATURE_BASE_COLUMNS))
