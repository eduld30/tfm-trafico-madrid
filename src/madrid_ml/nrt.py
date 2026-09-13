"""Near-real-time district risk scoring from the latest Silver observations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from madrid_ml.contract import (
    AIR_MAGNITUDES,
    EXPECTED_DISTRICTS,
    FEATURE_SCHEMA_VERSION,
    WEATHER_MAGNITUDES,
    Magnitude,
)
from madrid_ml.preprocessing import (
    SESSION_TIMEZONE,
    transform_inference_features,
)

SERVING_SCHEMA = "serving"
PREDICTION_TABLE = "riesgo_accidente_distrito"
CURRENT_PREDICTION_VIEW = "v_riesgo_actual_distrito"
DEFAULT_REGISTERED_MODEL = "accident_risk_model"
DEFAULT_MODEL_ALIAS = "Champion"
NRT_TIME_CONTRACT = "rolling_60_minutes_source_wall_clock_as_stored_in_silver"
MADRID_TIMEZONE = "Europe/Madrid"

_IDENTIFIER_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
_MODEL_NAME_PATTERN = re.compile(
    r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$"
)
_ALIAS_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_MLFLOW_DFS_TMP_PATTERN = re.compile(
    r"^/Volumes/[a-z_][a-z0-9_]*/[a-z_][a-z0-9_]*/"
    r"[a-z_][a-z0-9_]*(?:/[A-Za-z0-9._-]+)*$"
)


class NrtScoringError(RuntimeError):
    """Raised when an NRT feature snapshot or prediction violates its contract."""


@dataclass(frozen=True)
class NrtScoringConfig:
    """Runtime identity for one NRT scoring execution."""

    silver_catalog: str
    gold_catalog: str
    registered_model_name: str
    mlflow_dfs_tmp: str
    model_alias: str = DEFAULT_MODEL_ALIAS
    run_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("silver_catalog", "gold_catalog"):
            value = getattr(self, field_name)
            if not _IDENTIFIER_PATTERN.fullmatch(value):
                raise ValueError(f"{field_name} is not a valid identifier: {value!r}")
        if not _MODEL_NAME_PATTERN.fullmatch(self.registered_model_name):
            raise ValueError(
                "registered_model_name must use catalog.schema.model syntax"
            )
        if not _ALIAS_PATTERN.fullmatch(self.model_alias):
            raise ValueError(f"model_alias is invalid: {self.model_alias!r}")
        if not _MLFLOW_DFS_TMP_PATTERN.fullmatch(self.mlflow_dfs_tmp):
            raise ValueError(
                "mlflow_dfs_tmp must be a Unity Catalog volume path under "
                "/Volumes/<catalog>/<schema>/<volume>"
            )


@dataclass(frozen=True)
class LoadedRiskModel:
    """Resolved immutable model version and its fitted preprocessing pipeline."""

    model_name: str
    model_version: str
    model_type: str
    parent_run_id: str
    preprocessor: PipelineModel
    classifier: object


@dataclass(frozen=True)
class NrtScoringResult:
    """Compact evidence returned after publishing one complete district batch."""

    run_id: str
    cutoff: str
    prediction_table: str
    current_view: str
    model_name: str
    model_version: str
    row_count: int


def floor_to_ten_minutes(value: datetime) -> datetime:
    """Return a naive Madrid wall-clock cutoff aligned to ten minutes."""
    if value.tzinfo is not None:
        value = value.astimezone(ZoneInfo(MADRID_TIMEZONE)).replace(tzinfo=None)
    return value.replace(
        minute=(value.minute // 10) * 10,
        second=0,
        microsecond=0,
    )


def current_madrid_cutoff() -> datetime:
    """Return the current Madrid civil time aligned to the scoring cadence."""
    return floor_to_ten_minutes(datetime.now(ZoneInfo(MADRID_TIMEZONE)))


def _wall_clock_timestamp(value: datetime):
    """Build a Spark timestamp literal without host-timezone conversion."""
    return F.lit(value.isoformat(sep=" ")).cast("timestamp")


def _require_columns(df: DataFrame, columns: tuple[str, ...], logical_table: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise NrtScoringError(
            f"{logical_table}: missing required columns {', '.join(missing)}"
        )


def _read_required_table(spark: SparkSession, table_name: str) -> DataFrame:
    if not spark.catalog.tableExists(table_name):
        raise NrtScoringError(f"required Silver table does not exist: {table_name}")
    return spark.table(table_name)


def _traffic_features(
    traffic: DataFrame,
    *,
    window_start: datetime,
    cutoff: datetime,
) -> DataFrame:
    logical_table = "trafico.trafico_nrt"
    _require_columns(
        traffic,
        ("fecha_hora", "distrito_cod", "intensidad", "ocupacion", "carga"),
        logical_table,
    )
    speed_column = "vmed" if "vmed" in traffic.columns else "velocidad"
    _require_columns(traffic, (speed_column,), logical_table)
    filtered = traffic.where(
        (F.col("fecha_hora") >= _wall_clock_timestamp(window_start))
        & (F.col("fecha_hora") < _wall_clock_timestamp(cutoff))
        & F.col("distrito_cod").isNotNull()
    )
    aggregated = filtered.groupBy(
        F.col("distrito_cod").cast("int").alias("cod_distrito")
    ).agg(
        F.avg("intensidad").cast("double").alias("trafico_intensidad_media"),
        F.avg("ocupacion").cast("double").alias("trafico_ocupacion_media"),
        F.avg("carga").cast("double").alias("trafico_carga_media"),
        F.avg(speed_column).cast("double").alias("trafico_vmed_media"),
        F.count(F.lit(1)).cast("long").alias("trafico_puntos_n"),
        F.max("fecha_hora").alias("__source_max_timestamp"),
    )
    return aggregated.withColumn(
        "traffic_data_max_timestamp",
        F.max("__source_max_timestamp").over(Window.partitionBy()),
    ).drop("__source_max_timestamp")


def _latest_magnitude_features(
    frame: DataFrame,
    specs: tuple[Magnitude, ...],
    *,
    cutoff: datetime,
    logical_table: str,
    timestamp_output: str,
) -> DataFrame:
    _require_columns(
        frame,
        ("fecha_hora", "distrito_cod", "estacion", "magnitud", "valor"),
        logical_table,
    )
    identity = ("punto_muestreo",) if "punto_muestreo" in frame.columns else (
        "estacion",
        "magnitud",
    )
    ordering = [F.col("fecha_hora").desc()]
    if "_silver_processed_timestamp" in frame.columns:
        ordering.append(F.col("_silver_processed_timestamp").desc())
    latest = (
        frame.where(
            (F.col("fecha_hora") <= _wall_clock_timestamp(cutoff))
            & F.col("distrito_cod").isNotNull()
            & F.col("magnitud").isin(*(spec.code for spec in specs))
        )
        .withColumn(
            "__row_number",
            F.row_number().over(Window.partitionBy(*identity).orderBy(*ordering)),
        )
        .where(F.col("__row_number") == 1)
        .drop("__row_number")
    )
    aggregations = [
        expression
        for spec in specs
        for expression in (
            F.avg(F.when(F.col("magnitud") == spec.code, F.col("valor")))
            .cast("double")
            .alias(spec.mean_column),
            F.count(F.when(F.col("magnitud") == spec.code, F.col("valor")))
            .cast("long")
            .alias(spec.count_column),
        )
    ]
    aggregated = latest.groupBy(
        F.col("distrito_cod").cast("int").alias("cod_distrito")
    ).agg(
        *aggregations,
        F.max("fecha_hora").alias("__source_max_timestamp"),
    )
    return aggregated.withColumn(
        timestamp_output,
        F.max("__source_max_timestamp").over(Window.partitionBy()),
    ).drop("__source_max_timestamp")


def build_nrt_feature_snapshot(
    spark: SparkSession,
    *,
    silver_catalog: str,
    cutoff: datetime,
) -> DataFrame:
    """Build one unlabeled rolling-hour feature row for every Madrid district."""
    if cutoff != floor_to_ten_minutes(cutoff):
        raise NrtScoringError("cutoff must be aligned to a ten-minute boundary")
    window_start = cutoff - timedelta(hours=1)
    table_names = {
        "districts": f"{silver_catalog}.trafico.dim_distritos",
        "traffic": f"{silver_catalog}.trafico.trafico_nrt",
        "weather": f"{silver_catalog}.meteo.meteo_nrt",
        "air": f"{silver_catalog}.calair.calair_nrt",
    }
    inputs = {
        key: _read_required_table(spark, table_name)
        for key, table_name in table_names.items()
    }
    _require_columns(inputs["districts"], ("cod_dis", "nombre"), table_names["districts"])
    districts = inputs["districts"].select(
        F.col("cod_dis").cast("int").alias("cod_distrito"),
        F.col("nombre").cast("string").alias("distrito_nombre"),
    )
    traffic = _traffic_features(
        inputs["traffic"], window_start=window_start, cutoff=cutoff
    )
    weather = _latest_magnitude_features(
        inputs["weather"],
        WEATHER_MAGNITUDES,
        cutoff=cutoff,
        logical_table=table_names["weather"],
        timestamp_output="weather_data_max_timestamp",
    )
    air = _latest_magnitude_features(
        inputs["air"],
        AIR_MAGNITUDES,
        cutoff=cutoff,
        logical_table=table_names["air"],
        timestamp_output="air_data_max_timestamp",
    )
    result = (
        districts.join(traffic, "cod_distrito", "left")
        .join(weather, "cod_distrito", "left")
        .join(air, "cod_distrito", "left")
        .withColumn("feature_hour", _wall_clock_timestamp(window_start))
        .withColumn("prediction_hour", _wall_clock_timestamp(cutoff))
        .withColumn("hora_dia", F.hour("feature_hour").cast("int"))
        .withColumn("dia_semana", F.dayofweek("feature_hour").cast("int"))
        .withColumn("mes", F.month("feature_hour").cast("int"))
    )
    count_columns = (
        "trafico_puntos_n",
        *(spec.count_column for spec in WEATHER_MAGNITUDES),
        *(spec.count_column for spec in AIR_MAGNITUDES),
    )
    for column in count_columns:
        result = result.withColumn(
            column, F.coalesce(F.col(column), F.lit(0).cast("long"))
        )
    return result


def validate_nrt_feature_snapshot(df: DataFrame) -> int:
    """Require one unique, non-null row for each of Madrid's 21 districts."""
    summary = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("cod_distrito").alias("districts"),
        F.sum(
            F.when(
                F.col("cod_distrito").isNull()
                | F.col("feature_hour").isNull()
                | F.col("prediction_hour").isNull(),
                1,
            ).otherwise(0)
        ).alias("invalid_rows"),
    ).first()
    if (
        summary is None
        or summary.rows != EXPECTED_DISTRICTS
        or summary.districts != EXPECTED_DISTRICTS
        or summary.invalid_rows
    ):
        raise NrtScoringError(
            "NRT feature snapshot must contain one valid row per district; "
            f"observed={None if summary is None else summary.asDict()}"
        )
    return int(summary.rows)


def load_risk_model(config: NrtScoringConfig) -> LoadedRiskModel:
    """Resolve the configured UC alias and load its classifier plus preprocessor."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(registry_uri="databricks-uc")
    version = client.get_model_version_by_alias(
        config.registered_model_name,
        config.model_alias,
    )
    if not version.run_id:
        raise NrtScoringError("registered model version has no source MLflow run")
    run = client.get_run(version.run_id)
    model_type = run.data.tags.get("madrid_ml.comparator") or "unknown"
    parent_run_id = run.data.tags.get("mlflow.parentRunId")
    if not parent_run_id:
        raise NrtScoringError(
            "registered model must come from a child run with a parent training run"
        )
    parent_run = client.get_run(parent_run_id)
    observed_schema = parent_run.data.tags.get("madrid_ml.feature_schema_version")
    if observed_schema != FEATURE_SCHEMA_VERSION:
        raise NrtScoringError(
            "registered model feature schema differs from the scoring contract; "
            f"expected={FEATURE_SCHEMA_VERSION!r}, observed={observed_schema!r}"
        )
    preprocessor = mlflow.spark.load_model(
        f"runs:/{parent_run_id}/preprocessor",
        dfs_tmpdir=config.mlflow_dfs_tmp,
    )
    model_uri = (
        f"models:/{config.registered_model_name}@{config.model_alias}"
    )
    classifier = mlflow.spark.load_model(
        model_uri,
        dfs_tmpdir=config.mlflow_dfs_tmp,
    )
    return LoadedRiskModel(
        model_name=config.registered_model_name,
        model_version=str(version.version),
        model_type=model_type,
        parent_run_id=parent_run_id,
        preprocessor=preprocessor,
        classifier=classifier,
    )


def score_nrt_features(
    features: DataFrame,
    model: LoadedRiskModel,
    *,
    run_id: str,
) -> DataFrame:
    """Apply the exact training preprocessor and resolved classifier."""
    transformed = transform_inference_features(model.preprocessor, features)
    try:
        scored = model.classifier.transform(transformed).withColumn(
            "risk_probability",
            vector_to_array(F.col("probability"))[1].cast("double"),
        )
    except Exception as exc:
        raise NrtScoringError(
            "registered model must expose a Spark probability vector"
        ) from exc
    return scored.select(
        "cod_distrito",
        "distrito_nombre",
        F.col("feature_hour").alias("feature_window_start"),
        F.col("prediction_hour").alias("feature_window_end"),
        F.col("prediction_hour").alias("prediction_window_start"),
        (F.col("prediction_hour") + F.expr("INTERVAL 1 HOUR")).alias(
            "prediction_window_end"
        ),
        "risk_probability",
        F.lit(model.model_name).alias("model_name"),
        F.lit(model.model_version).alias("model_version"),
        F.lit(FEATURE_SCHEMA_VERSION).alias("feature_schema_version"),
        F.lit(run_id).alias("prediction_run_id"),
        F.from_utc_timestamp(
            F.current_timestamp(), MADRID_TIMEZONE
        ).alias("predicted_at"),
        "traffic_data_max_timestamp",
        "weather_data_max_timestamp",
        "air_data_max_timestamp",
        F.lit(NRT_TIME_CONTRACT).alias("time_contract"),
    )


def validate_predictions(df: DataFrame) -> int:
    """Reject partial district batches and invalid probability values."""
    summary = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("cod_distrito").alias("districts"),
        F.sum(
            F.when(
                F.col("risk_probability").isNull()
                | F.isnan("risk_probability")
                | (F.col("risk_probability") < 0.0)
                | (F.col("risk_probability") > 1.0),
                1,
            ).otherwise(0)
        ).alias("invalid_scores"),
    ).first()
    if (
        summary is None
        or summary.rows != EXPECTED_DISTRICTS
        or summary.districts != EXPECTED_DISTRICTS
        or summary.invalid_scores
    ):
        raise NrtScoringError(
            "prediction batch must contain 21 unique valid probabilities; "
            f"observed={None if summary is None else summary.asDict()}"
        )
    return int(summary.rows)


def _quote_qualified_name(name: str) -> str:
    parts = name.split(".")
    if not parts or any(not _IDENTIFIER_PATTERN.fullmatch(part) for part in parts):
        raise NrtScoringError(f"invalid Unity Catalog object name: {name!r}")
    return ".".join(f"`{part}`" for part in parts)


def publish_predictions(
    spark: SparkSession,
    predictions: DataFrame,
    *,
    gold_catalog: str,
) -> tuple[str, str]:
    """Upsert one prediction window and expose the latest complete batch as a view."""
    schema_name = f"{gold_catalog}.{SERVING_SCHEMA}"
    table_name = f"{schema_name}.{PREDICTION_TABLE}"
    view_name = f"{schema_name}.{CURRENT_PREDICTION_VIEW}"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {_quote_qualified_name(schema_name)}")
    if spark.catalog.tableExists(table_name):
        try:
            from delta.tables import DeltaTable
        except ImportError as exc:
            raise NrtScoringError(
                "publishing NRT predictions requires the Delta Lake runtime APIs"
            ) from exc
        (
            DeltaTable.forName(spark, table_name)
            .alias("t")
            .merge(
                predictions.alias("s"),
                "t.cod_distrito <=> s.cod_distrito AND "
                "t.prediction_window_start <=> s.prediction_window_start",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        # This branch is reached only when the table does not exist.  Databricks
        # serverless does not support the Spark ``errorifexists`` alias here;
        # overwrite is safe because there is no existing target yet.
        predictions.write.format("delta").mode("overwrite").saveAsTable(table_name)
    quoted_table = _quote_qualified_name(table_name)
    spark.sql(
        f"CREATE OR REPLACE VIEW {_quote_qualified_name(view_name)} AS "
        f"SELECT p.* FROM {quoted_table} p "
        f"WHERE p.prediction_window_start = ("
        f"SELECT MAX(prediction_window_start) FROM {quoted_table})"
    )
    return table_name, view_name


def run_nrt_scoring(
    spark: SparkSession,
    config: NrtScoringConfig,
    *,
    cutoff: datetime | None = None,
) -> NrtScoringResult:
    """Build, score, validate, and publish one rolling-hour prediction batch."""
    spark.conf.set("spark.sql.session.timeZone", SESSION_TIMEZONE)
    aligned_cutoff = current_madrid_cutoff() if cutoff is None else floor_to_ten_minutes(cutoff)
    run_id = config.run_id.strip() or str(uuid4())
    features = build_nrt_feature_snapshot(
        spark,
        silver_catalog=config.silver_catalog,
        cutoff=aligned_cutoff,
    )
    validate_nrt_feature_snapshot(features)
    model = load_risk_model(config)
    predictions = score_nrt_features(features, model, run_id=run_id)
    row_count = validate_predictions(predictions)
    table_name, view_name = publish_predictions(
        spark,
        predictions,
        gold_catalog=config.gold_catalog,
    )
    return NrtScoringResult(
        run_id=run_id,
        cutoff=aligned_cutoff.isoformat(sep=" "),
        prediction_table=table_name,
        current_view=view_name,
        model_name=model.model_name,
        model_version=model.model_version,
        row_count=row_count,
    )
