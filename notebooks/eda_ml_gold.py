# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# ///
# MAGIC %md
# MAGIC # EDA y gate de calidad del snapshot Gold de ML
# MAGIC
# MAGIC Este notebook responde únicamente a decisiones que condicionan el primer
# MAGIC modelo: prevalencia y estabilidad del target, cobertura por dominio,
# MAGIC valores no finitos, rangos observados, estabilidad temporal y viabilidad
# MAGIC del split temporal propuesto.
# MAGIC
# MAGIC No revalida Silver, no selecciona features por correlación, no fija
# MAGIC imputaciones ni umbrales físicos sin evidencia y no analiza NRT. La paridad
# MAGIC histórico/NRT pertenece al futuro contrato de scoring.

# COMMAND ----------

import json
import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    LongType,
    NumericType,
    StringType,
    StructField,
    StructType,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parámetros y contrato reproducible
# MAGIC
# MAGIC `expected_snapshot_id` impide analizar silenciosamente un overwrite
# MAGIC posterior. Para analizar otro snapshot hay que cambiarlo explícitamente.

# COMMAND ----------

dbutils.widgets.text("gold_catalog", "dev_gold", "Catálogo Gold")  # noqa: F821
dbutils.widgets.text(  # noqa: F821
    "expected_snapshot_id",
    "ea719086-1a93-401c-969b-4e92586e13fd",
    "Snapshot esperado",
)

GOLD_CATALOG = dbutils.widgets.get("gold_catalog").strip()  # noqa: F821
EXPECTED_SNAPSHOT_ID = dbutils.widgets.get("expected_snapshot_id").strip()  # noqa: F821

_IDENTIFIER_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
if not _IDENTIFIER_PATTERN.fullmatch(GOLD_CATALOG):
    raise ValueError(f"gold_catalog inválido: {GOLD_CATALOG!r}; se esperaba lowercase snake_case")
if not EXPECTED_SNAPSHOT_ID:
    raise ValueError("expected_snapshot_id es obligatorio")

ML_SCHEMA = "ml"
LABELS_TABLE = f"{GOLD_CATALOG}.{ML_SCHEMA}.accident_labels_hourly"
FEATURES_TABLE = f"{GOLD_CATALOG}.{ML_SCHEMA}.features_training_snapshot"
EXPECTED_FEATURE_SCHEMA_VERSION = "1"
EXPECTED_TIME_CONTRACT = "source_wall_clock_as_stored_in_silver"
EXPECTED_DISTRICTS = 21
LINEAGE_COLUMNS = (
    "snapshot_id",
    "input_versions_json",
    "code_commit",
    "feature_schema_version",
    "time_contract",
)
MODEL_EXCLUDED_NUMERIC_COLUMNS = {
    "cod_distrito",
    "n_accidentes_next_hour",
    "target_accident_next_hour",
}
CANDIDATE_SPLITS = ("train_2019_2023", "validation_2024", "test_2025")

spark.conf.set("spark.sql.session.timeZone", "Etc/UTC")  # noqa: F821


# COMMAND ----------

# MAGIC %md
# MAGIC ## Lectura fijada a versiones Delta

# COMMAND ----------


def quoted_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) != 3 or any(not _IDENTIFIER_PATTERN.fullmatch(part) for part in parts):
        raise ValueError(f"Nombre de tabla inválido: {name!r}")
    return ".".join(f"`{part}`" for part in parts)


def latest_delta_version(table_name: str) -> int:
    row = (
        spark.sql(  # noqa: F821
            f"DESCRIBE HISTORY {quoted_name(table_name)} LIMIT 1"
        )
        .select("version")
        .first()
    )
    if row is None or row["version"] is None:
        raise RuntimeError(f"No se pudo fijar la versión Delta de {table_name}")
    return int(row["version"])


def read_delta_version(table_name: str, version: int) -> DataFrame:
    return (
        spark.read.format("delta")  # noqa: F821
        .option("versionAsOf", version)
        .table(table_name)
    )


def single_lineage(df: DataFrame, table_name: str) -> dict[str, str]:
    rows = df.select(*LINEAGE_COLUMNS).distinct().limit(2).collect()
    if len(rows) != 1:
        raise RuntimeError(
            f"{table_name}: se esperaba un lineage único y se encontraron {len(rows)}"
        )
    lineage = {column: rows[0][column] for column in LINEAGE_COLUMNS}
    null_columns = [column for column, value in lineage.items() if value is None]
    if null_columns:
        raise RuntimeError(f"{table_name}: lineage nulo en {null_columns}")
    return {column: str(value) for column, value in lineage.items()}


for required_table in (LABELS_TABLE, FEATURES_TABLE):
    if not spark.catalog.tableExists(required_table):  # noqa: F821
        raise RuntimeError(f"No existe la tabla Gold requerida: {required_table}")

labels_version = latest_delta_version(LABELS_TABLE)
features_version = latest_delta_version(FEATURES_TABLE)
labels = read_delta_version(LABELS_TABLE, labels_version)
features = read_delta_version(FEATURES_TABLE, features_version)
labels_lineage = single_lineage(labels, LABELS_TABLE)
features_lineage = single_lineage(features, FEATURES_TABLE)

if labels_lineage != features_lineage:
    raise RuntimeError(
        "Las tablas Gold no pertenecen al mismo snapshot: "
        f"labels={labels_lineage}, features={features_lineage}"
    )
if labels_lineage["snapshot_id"] != EXPECTED_SNAPSHOT_ID:
    raise RuntimeError(
        "El snapshot publicado no coincide con el solicitado: "
        f"expected={EXPECTED_SNAPSHOT_ID}, "
        f"observed={labels_lineage['snapshot_id']}"
    )
if labels_lineage["feature_schema_version"] != EXPECTED_FEATURE_SCHEMA_VERSION:
    raise RuntimeError(
        f"Versión de features no soportada por este EDA: {labels_lineage['feature_schema_version']}"
    )
if labels_lineage["time_contract"] != EXPECTED_TIME_CONTRACT:
    raise RuntimeError(f"Contrato temporal no soportado: {labels_lineage['time_contract']}")

print(
    json.dumps(
        {
            "labels_table": LABELS_TABLE,
            "labels_delta_version": labels_version,
            "features_table": FEATURES_TABLE,
            "features_delta_version": features_version,
            "lineage": labels_lineage,
        },
        sort_keys=True,
    )
)


def positive_count_expression():
    return F.sum(
        F.when(F.col("target_accident_next_hour") == 1, F.lit(1)).otherwise(F.lit(0))
    ).cast("long")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate estructural
# MAGIC
# MAGIC El builder Gold ya demostró unicidad y reconciliación exacta de claves al
# MAGIC publicar este snapshot. Repetir esos dos shuffles aquí no añade evidencia.
# MAGIC Este gate comprueba que seguimos leyendo ese mismo snapshot y que sus
# MAGIC invariantes básicos no han cambiado.

# COMMAND ----------

labels_summary_row = labels.agg(
    F.count(F.lit(1)).cast("long").alias("row_count"),
    F.countDistinct("cod_distrito").cast("long").alias("district_count"),
    F.min("feature_hour").alias("feature_hour_min"),
    F.max("feature_hour").alias("feature_hour_max"),
    positive_count_expression().alias("positive_count"),
    F.sum(
        F.when(
            F.col("target_accident_next_hour").isNull()
            | ~F.col("target_accident_next_hour").isin(0, 1),
            F.lit(1),
        ).otherwise(F.lit(0))
    )
    .cast("long")
    .alias("invalid_target_rows"),
    F.sum(
        F.when(
            F.col("prediction_hour").isNull()
            | F.col("feature_hour").isNull()
            | (F.col("prediction_hour") != F.col("feature_hour") + F.expr("INTERVAL 1 HOUR")),
            F.lit(1),
        ).otherwise(F.lit(0))
    )
    .cast("long")
    .alias("prediction_hour_mismatches"),
).first()
features_summary_row = features.agg(
    F.count(F.lit(1)).cast("long").alias("row_count"),
    F.countDistinct("cod_distrito").cast("long").alias("district_count"),
    F.min("feature_hour").alias("feature_hour_min"),
    F.max("feature_hour").alias("feature_hour_max"),
).first()
if labels_summary_row is None or features_summary_row is None:
    raise RuntimeError("El resumen Gold no devolvió ninguna fila")

labels_row_count = int(labels_summary_row["row_count"])
features_row_count = int(features_summary_row["row_count"])
positive_count = int(labels_summary_row["positive_count"])
structural_blockers: list[str] = []
if labels_row_count != features_row_count:
    structural_blockers.append(
        f"row_count distinto: labels={labels_row_count}, features={features_row_count}"
    )
if int(labels_summary_row["district_count"]) != EXPECTED_DISTRICTS:
    structural_blockers.append(f"labels no contiene {EXPECTED_DISTRICTS} distritos")
if int(features_summary_row["district_count"]) != EXPECTED_DISTRICTS:
    structural_blockers.append(f"features no contiene {EXPECTED_DISTRICTS} distritos")
if int(labels_summary_row["invalid_target_rows"]) != 0:
    structural_blockers.append("target_accident_next_hour contiene valores fuera de {0, 1}")
if int(labels_summary_row["prediction_hour_mismatches"]) != 0:
    structural_blockers.append("prediction_hour no equivale a feature_hour + 1 hora")
if (
    labels_summary_row["feature_hour_min"] != features_summary_row["feature_hour_min"]
    or labels_summary_row["feature_hour_max"] != features_summary_row["feature_hour_max"]
):
    structural_blockers.append("labels y features no comparten el mismo periodo")

snapshot_summary = spark.createDataFrame(  # noqa: F821
    [
        (
            EXPECTED_SNAPSHOT_ID,
            labels_version,
            features_version,
            labels_row_count,
            features_row_count,
            int(labels_summary_row["district_count"]),
            labels_summary_row["feature_hour_min"],
            labels_summary_row["feature_hour_max"],
            positive_count,
            positive_count / labels_row_count,
        )
    ],
    "snapshot_id string, labels_delta_version long, features_delta_version long, "
    "labels_rows long, features_rows long, districts long, "
    "feature_hour_min timestamp, feature_hour_max timestamp, positive_count long, "
    "positive_rate double",
)
display(snapshot_summary)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prevalencia accionable
# MAGIC
# MAGIC Una sola salida segmentada permite decidir desbalance, estabilidad del
# MAGIC target y segmentos de evaluación. No se calculan correlaciones ni tests
# MAGIC estadísticos masivos antes de fijar el protocolo del modelo.

# COMMAND ----------

labels_segmented = (
    labels.withColumn("anio", F.year("feature_hour"))
    .withColumn("mes_calendario", F.month("feature_hour"))
    .withColumn("hora", F.hour("feature_hour"))
    .withColumn("dia_semana_calendario", F.dayofweek("feature_hour"))
)


def prevalence_by(column: str, dimension: str) -> DataFrame:
    return (
        labels_segmented.groupBy(F.col(column).cast("long").alias("segment_order"))
        .agg(
            F.count(F.lit(1)).cast("long").alias("rows"),
            positive_count_expression().alias("positives"),
        )
        .withColumn("positive_rate", F.col("positives") / F.col("rows"))
        .withColumn("dimension", F.lit(dimension))
        .withColumn("segment_value", F.col("segment_order").cast("string"))
        .select(
            "dimension",
            "segment_value",
            "segment_order",
            "rows",
            "positives",
            "positive_rate",
        )
    )


prevalence_segments = prevalence_by("anio", "year")
for column, dimension in (
    ("mes_calendario", "month"),
    ("cod_distrito", "district"),
    ("hora", "hour"),
    ("dia_semana_calendario", "day_of_week"),
):
    prevalence_segments = prevalence_segments.unionByName(prevalence_by(column, dimension))

display(prevalence_segments.orderBy("dimension", "segment_order"))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calidad de las features
# MAGIC
# MAGIC Para cada feature numérica se muestran nulos, `NaN`, infinitos, negativos
# MAGIC y rango observado. Los negativos de medidas físicas no se invalidan sin
# MAGIC contrato de unidad/rango; los negativos en columnas de conteo sí rompen el
# MAGIC contrato. Los nulos y no finitos alimentan la futura política de
# MAGIC preprocessing, no una imputación inventada en este notebook.

# COMMAND ----------

numeric_fields = [
    field
    for field in features.schema.fields
    if isinstance(field.dataType, NumericType) and field.name not in MODEL_EXCLUDED_NUMERIC_COLUMNS
]
model_feature_columns = [field.name for field in numeric_fields]
value_feature_columns = [
    field.name for field in numeric_fields if isinstance(field.dataType, (DoubleType, FloatType))
]


def finite_value(column_name: str):
    value = F.col(column_name).cast("double")
    return F.when(
        value.isNotNull()
        & ~F.isnan(value)
        & (value != F.lit(float("inf")))
        & (value != F.lit(float("-inf"))),
        value,
    )


count_feature_columns = [
    field.name
    for field in numeric_fields
    if field.name.endswith("_n") or field.name == "trafico_puntos_n"
]

quality_expressions = []
for index, field in enumerate(numeric_fields):
    column = F.col(field.name)
    quality_expressions.extend(
        (
            F.sum(F.when(column.isNull(), 1).otherwise(0)).cast("long").alias(f"q_{index}_null"),
            F.sum(F.when(column < 0, 1).otherwise(0)).cast("long").alias(f"q_{index}_negative"),
            F.min(
                finite_value(field.name)
                if field.name in value_feature_columns
                else column.cast("double")
            ).alias(f"q_{index}_min"),
            F.max(
                finite_value(field.name)
                if field.name in value_feature_columns
                else column.cast("double")
            ).alias(f"q_{index}_max"),
        )
    )
    if field.name in value_feature_columns:
        quality_expressions.extend(
            (
                F.sum(F.when(F.isnan(column), 1).otherwise(0)).cast("long").alias(f"q_{index}_nan"),
                F.sum(
                    F.when(
                        (column == F.lit(float("inf"))) | (column == F.lit(float("-inf"))),
                        1,
                    ).otherwise(0)
                )
                .cast("long")
                .alias(f"q_{index}_infinite"),
            )
        )

quality_row = features.agg(*quality_expressions).first()
if quality_row is None:
    raise RuntimeError("El perfil de features no devolvió ninguna fila")

quality_records: list[dict[str, int | float | str | None]] = []
for index, field in enumerate(numeric_fields):
    null_count = int(quality_row[f"q_{index}_null"])
    nan_count = int(quality_row[f"q_{index}_nan"]) if field.name in value_feature_columns else 0
    infinite_count = (
        int(quality_row[f"q_{index}_infinite"]) if field.name in value_feature_columns else 0
    )
    usable_count = features_row_count - null_count - nan_count - infinite_count
    minimum = quality_row[f"q_{index}_min"]
    maximum = quality_row[f"q_{index}_max"]
    quality_records.append(
        {
            "feature": field.name,
            "data_type": field.dataType.simpleString(),
            "null_count": null_count,
            "null_rate": null_count / features_row_count,
            "nan_count": nan_count,
            "infinite_count": infinite_count,
            "usable_count": usable_count,
            "usable_rate": usable_count / features_row_count,
            "negative_count": int(quality_row[f"q_{index}_negative"]),
            "min_observed": None if minimum is None else float(minimum),
            "max_observed": None if maximum is None else float(maximum),
        }
    )

quality_schema = StructType(
    (
        StructField("feature", StringType(), False),
        StructField("data_type", StringType(), False),
        StructField("null_count", LongType(), False),
        StructField("null_rate", DoubleType(), False),
        StructField("nan_count", LongType(), False),
        StructField("infinite_count", LongType(), False),
        StructField("usable_count", LongType(), False),
        StructField("usable_rate", DoubleType(), False),
        StructField("negative_count", LongType(), False),
        StructField("min_observed", DoubleType(), True),
        StructField("max_observed", DoubleType(), True),
    )
)
feature_quality = spark.createDataFrame(quality_records, quality_schema)  # noqa: F821
display(feature_quality.orderBy("usable_rate", "feature"))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cobertura por dominio y viabilidad del split temporal
# MAGIC
# MAGIC El split 2019–2023 / 2024 / 2025 procede del roadmap. 2026 se mantiene
# MAGIC fuera del protocolo candidato hasta decidir si será test futuro o periodo
# MAGIC operativo. La cobertura se define por presencia de observaciones, no por
# MAGIC medias imputadas.

# COMMAND ----------

weather_count_columns = [column for column in count_feature_columns if column.startswith("meteo_")]
air_count_columns = [column for column in count_feature_columns if column.startswith("calair_")]


def any_observation(columns: list[str]):
    condition = F.lit(False)
    for column in columns:
        condition = condition | (F.col(column) > 0)
    return condition


traffic_available = F.col("trafico_puntos_n") > 0
weather_available = any_observation(weather_count_columns)
air_available = any_observation(air_count_columns)
features_with_period = features.withColumn("anio", F.year("feature_hour")).withColumn(
    "candidate_split",
    F.when(F.col("anio").between(2019, 2023), F.lit("train_2019_2023"))
    .when(F.col("anio") == 2024, F.lit("validation_2024"))
    .when(F.col("anio") == 2025, F.lit("test_2025"))
    .otherwise(F.lit("outside_candidate_split")),
)

coverage_aggregations = (
    F.count(F.lit(1)).cast("long").alias("rows"),
    positive_count_expression().alias("positives"),
    F.avg(F.when(traffic_available, 1.0).otherwise(0.0)).alias("traffic_coverage_rate"),
    F.avg(F.when(weather_available, 1.0).otherwise(0.0)).alias("weather_coverage_rate"),
    F.avg(F.when(air_available, 1.0).otherwise(0.0)).alias("air_coverage_rate"),
)

split_summary = (
    features_with_period.groupBy("candidate_split")
    .agg(*coverage_aggregations)
    .withColumn("positive_rate", F.col("positives") / F.col("rows"))
)
year_district_coverage = (
    features_with_period.groupBy("anio", "cod_distrito")
    .agg(*coverage_aggregations)
    .withColumn("positive_rate", F.col("positives") / F.col("rows"))
)

split_summary_rows = split_summary.collect()
split_summary_output = spark.createDataFrame(  # noqa: F821
    split_summary_rows,
    split_summary.schema,
)
display(split_summary_output.orderBy("candidate_split"))  # noqa: F821
display(year_district_coverage.orderBy("anio", "cod_distrito"))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estabilidad temporal de escala
# MAGIC
# MAGIC Media y desviación estándar anual permiten detectar cambios de régimen o
# MAGIC unidad antes de entrenar. No se calculan cientos de tests ni se descartan
# MAGIC variables automáticamente: cualquier cambio necesita interpretación del
# MAGIC dominio y contraste con cobertura.

# COMMAND ----------

yearly_scale_expressions = []
for index, column in enumerate(value_feature_columns):
    yearly_scale_expressions.extend(
        (
            F.avg(finite_value(column)).alias(f"scale_{index}_mean"),
            F.stddev(finite_value(column)).alias(f"scale_{index}_stddev"),
        )
    )

yearly_scale_wide = features_with_period.groupBy("anio").agg(*yearly_scale_expressions)
stack_arguments = []
for index, column in enumerate(value_feature_columns):
    stack_arguments.extend(
        (
            f"'{column}'",
            f"scale_{index}_mean",
            f"scale_{index}_stddev",
        )
    )
yearly_feature_scale = yearly_scale_wide.selectExpr(
    "anio",
    f"stack({len(value_feature_columns)}, {', '.join(stack_arguments)}) "
    "as (feature, mean_value, stddev_value)",
)
display(yearly_feature_scale.orderBy("feature", "anio"))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate y decisiones para el siguiente paso

# COMMAND ----------

quality_by_name = {str(record["feature"]): record for record in quality_records}
for column in count_feature_columns:
    record = quality_by_name[column]
    if record["null_count"] != 0:
        structural_blockers.append(f"{column} contiene conteos nulos")
    if record["negative_count"] != 0:
        structural_blockers.append(f"{column} contiene conteos negativos")

calendar_ranges = {
    "hora_dia": (0.0, 23.0),
    "dia_semana": (1.0, 7.0),
    "mes": (1.0, 12.0),
}
for column, (expected_min, expected_max) in calendar_ranges.items():
    record = quality_by_name[column]
    if record["null_count"] != 0:
        structural_blockers.append(f"{column} contiene valores nulos")
    if record["min_observed"] is None or record["max_observed"] is None:
        structural_blockers.append(f"{column} no tiene rango observable")
    elif record["min_observed"] < expected_min or record["max_observed"] > expected_max:
        structural_blockers.append(
            f"{column} fuera de rango: min={record['min_observed']}, max={record['max_observed']}"
        )

split_rows = {row["candidate_split"]: row.asDict(recursive=True) for row in split_summary_rows}
for split in CANDIDATE_SPLITS:
    row = split_rows.get(split)
    if row is None or int(row["rows"]) == 0:
        structural_blockers.append(f"{split} no contiene filas")
    elif int(row["positives"]) == 0:
        structural_blockers.append(f"{split} no contiene positivos")

preprocessing_columns = [
    {
        "feature": record["feature"],
        "usable_rate": record["usable_rate"],
        "null_count": record["null_count"],
        "nan_count": record["nan_count"],
        "infinite_count": record["infinite_count"],
    }
    for record in quality_records
    if record["usable_rate"] < 1.0
]
non_informative_features = [
    {
        "feature": record["feature"],
        "reason": ("no_finite_values" if record["usable_count"] == 0 else "constant_finite_value"),
    }
    for record in quality_records
    if record["usable_count"] == 0
    or (record["min_observed"] is not None and record["min_observed"] == record["max_observed"])
]
range_review_columns = [
    {
        "feature": record["feature"],
        "negative_count": record["negative_count"],
        "min_observed": record["min_observed"],
        "max_observed": record["max_observed"],
    }
    for record in quality_records
    if record["feature"] in value_feature_columns
    and (record["nan_count"] > 0 or record["infinite_count"] > 0 or record["negative_count"] > 0)
]

eda_result = {
    "status": "READY_FOR_BASELINE_SPEC" if not structural_blockers else "BLOCKED",
    "snapshot_id": EXPECTED_SNAPSHOT_ID,
    "gold_delta_versions": {
        LABELS_TABLE: labels_version,
        FEATURES_TABLE: features_version,
    },
    "row_count": labels_row_count,
    "district_count": int(labels_summary_row["district_count"]),
    "positive_count": positive_count,
    "positive_rate": positive_count / labels_row_count,
    "model_feature_columns": model_feature_columns,
    "structural_blockers": structural_blockers,
    "preprocessing_decisions_required": preprocessing_columns,
    "non_informative_features": non_informative_features,
    "physical_range_review_required": range_review_columns,
    "candidate_split_summary": split_rows,
    "explicit_exclusions": [
        "historical_vs_nrt_parity",
        "publication_latency",
        "feature_selection",
        "imputation_policy",
        "model_thresholds",
    ],
}
print(json.dumps(eda_result, sort_keys=True, default=str))
if structural_blockers:
    raise RuntimeError(
        "El snapshot no supera el gate estructural: " + "; ".join(structural_blockers)
    )

dbutils.notebook.exit(  # noqa: F821
    json.dumps(eda_result, sort_keys=True, default=str)
)
