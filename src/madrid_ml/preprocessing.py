"""Versioned Spark preprocessing for the Gold ML training snapshot."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Imputer,
    ImputerModel,
    OneHotEncoder,
    OneHotEncoderModel,
    StandardScaler,
    StandardScalerModel,
    StringIndexer,
    StringIndexerModel,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from madrid_ml.contract import (
    AIR_MAGNITUDES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_TABLE,
    FEATURE_TABLE_COLUMNS,
    LABEL_TABLE,
    LABEL_TABLE_COLUMNS,
    ML_SCHEMA,
    TIME_CONTRACT,
    TRAFFIC_COLUMNS,
    WEATHER_MAGNITUDES,
)
from madrid_ml.snapshot import validate_matching_lineage
from madrid_ml.transformations import SnapshotContractError

PREPROCESSING_CONTRACT_VERSION = "1"
SESSION_TIMEZONE = "Etc/UTC"
EXPECTED_VECTOR_SIZE = 111

_TRAFFIC_CONTINUOUS_FEATURES = tuple(
    name for name, _, _ in TRAFFIC_COLUMNS if name.endswith("_media")
)
_TRAFFIC_COUNT_FEATURES = tuple(
    name for name, _, _ in TRAFFIC_COLUMNS if name.endswith("_n")
)
CONTINUOUS_FEATURES = (
    _TRAFFIC_CONTINUOUS_FEATURES
    + tuple(magnitude.mean_column for magnitude in WEATHER_MAGNITUDES)
    + tuple(magnitude.mean_column for magnitude in AIR_MAGNITUDES)
)
COUNT_FEATURES = (
    _TRAFFIC_COUNT_FEATURES
    + tuple(magnitude.count_column for magnitude in WEATHER_MAGNITUDES)
    + tuple(magnitude.count_column for magnitude in AIR_MAGNITUDES)
)
NUMERIC_FEATURES = CONTINUOUS_FEATURES + COUNT_FEATURES
AVAILABLE_FEATURES = tuple(f"{name}_available" for name in CONTINUOUS_FEATURES)


class PreprocessingContractError(RuntimeError):
    """Raised when Gold data cannot satisfy the preprocessing contract."""


@dataclass(frozen=True)
class PhysicalRange:
    """Hard physical range for one continuous input."""

    minimum: float | None = None
    maximum: float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True


_NON_NEGATIVE_RANGE = PhysicalRange(minimum=0.0)
PHYSICAL_RANGES = {
    "trafico_intensidad_media": _NON_NEGATIVE_RANGE,
    "trafico_ocupacion_media": _NON_NEGATIVE_RANGE,
    "trafico_carga_media": _NON_NEGATIVE_RANGE,
    "trafico_vmed_media": _NON_NEGATIVE_RANGE,
    "meteo_velocidad_viento_media": _NON_NEGATIVE_RANGE,
    "meteo_temperatura_media": PhysicalRange(),
    "meteo_humedad_relativa_media": PhysicalRange(minimum=0.0, maximum=100.0),
    "meteo_presion_media": PhysicalRange(minimum=0.0, minimum_inclusive=False),
    "meteo_radiacion_solar_media": _NON_NEGATIVE_RANGE,
    "meteo_precipitacion_media": _NON_NEGATIVE_RANGE,
    "calair_so2_media": _NON_NEGATIVE_RANGE,
    "calair_co_media": _NON_NEGATIVE_RANGE,
    "calair_no_media": _NON_NEGATIVE_RANGE,
    "calair_no2_media": _NON_NEGATIVE_RANGE,
    "calair_pm25_media": _NON_NEGATIVE_RANGE,
    "calair_pm10_media": _NON_NEGATIVE_RANGE,
    "calair_nox_media": _NON_NEGATIVE_RANGE,
    "calair_o3_media": _NON_NEGATIVE_RANGE,
}
MAGNITUDE_COUNT_BY_MEAN = {
    magnitude.mean_column: magnitude.count_column
    for magnitude in (*WEATHER_MAGNITUDES, *AIR_MAGNITUDES)
}


def _finite(column_name: str) -> Column:
    value = F.col(column_name).cast("double")
    return (
        value.isNotNull()
        & ~F.isnan(value)
        & (value != F.lit(float("inf")))
        & (value != F.lit(float("-inf")))
    )


def _valid_value(column_name: str) -> Column:
    rule = PHYSICAL_RANGES[column_name]
    value = F.col(column_name).cast("double")
    valid = _finite(column_name)
    if rule.minimum is not None:
        valid = valid & (
            value >= F.lit(rule.minimum)
            if rule.minimum_inclusive
            else value > F.lit(rule.minimum)
        )
    if rule.maximum is not None:
        valid = valid & (
            value <= F.lit(rule.maximum)
            if rule.maximum_inclusive
            else value < F.lit(rule.maximum)
        )
    return valid


def sanitize_features(df: DataFrame) -> DataFrame:
    """Replace unusable continuous inputs with null and add availability flags."""
    sanitized = df
    for name in CONTINUOUS_FEATURES:
        valid = _valid_value(name)
        sanitized = sanitized.withColumn(
            f"{name}_available", valid.cast("double")
        ).withColumn(
            name,
            F.when(valid, F.col(name).cast("double")).otherwise(
                F.lit(None).cast("double")
            ),
        )
    return sanitized


def _count_matching(condition: Column, alias: str) -> Column:
    return F.coalesce(
        F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0))),
        F.lit(0),
    ).cast("long").alias(alias)


def _validate_feature_domains(df: DataFrame, *, require_target: bool) -> None:
    """Fail once with every invalid categorical, count, and coherence gate."""
    required = set(NUMERIC_FEATURES + CATEGORY_FEATURES)
    if require_target:
        required.add("target_accident_next_hour")
    missing = sorted(required.difference(df.columns))
    if missing:
        raise PreprocessingContractError(
            "Gold feature domain validation is missing columns: " + ", ".join(missing)
        )

    counters: list[Column] = []

    category_domains = {
        "cod_distrito": (1, 21),
        "hora_dia": (0, 23),
        "dia_semana": (1, 7),
        "mes": (1, 12),
    }
    if require_target:
        category_domains["target_accident_next_hour"] = (0, 1)
    for name, (minimum, maximum) in category_domains.items():
        invalid = F.col(name).isNull() | ~F.col(name).between(minimum, maximum)
        counters.append(_count_matching(invalid, f"invalid__{name}"))

    for name in COUNT_FEATURES:
        invalid = F.col(name).isNull() | (F.col(name) < F.lit(0))
        counters.append(_count_matching(invalid, f"invalid__{name}"))

    for mean_name, count_name in MAGNITUDE_COUNT_BY_MEAN.items():
        inconsistent = (F.col(count_name) == F.lit(0)) & _finite(mean_name)
        counters.append(
            _count_matching(inconsistent, f"inconsistent__{mean_name}")
        )

    for mean_name in _TRAFFIC_CONTINUOUS_FEATURES:
        inconsistent = (F.col("trafico_puntos_n") == F.lit(0)) & _finite(
            mean_name
        )
        counters.append(
            _count_matching(inconsistent, f"inconsistent__{mean_name}")
        )

    observed = df.agg(*counters).first().asDict()
    failures = [
        f"{name}={count}"
        for name, count in observed.items()
        if count
    ]
    if failures:
        raise PreprocessingContractError(
            "Gold feature domain validation failed: " + ", ".join(failures)
        )


def validate_feature_domains(df: DataFrame) -> None:
    """Validate the complete training feature and target contract."""
    _validate_feature_domains(df, require_target=True)


def validate_inference_feature_domains(df: DataFrame) -> None:
    """Validate model inputs without requiring an unavailable future target."""
    _validate_feature_domains(df, require_target=False)


def profile_feature_quality(
    df: DataFrame,
) -> dict[str, dict[str, int]]:
    """Count raw missing/non-finite and finite out-of-range values per feature."""
    counters: list[Column] = []
    for name in CONTINUOUS_FEATURES:
        finite = _finite(name)
        counters.extend(
            (
                _count_matching(~finite, f"missing__{name}"),
                _count_matching(
                    finite & ~_valid_value(name),
                    f"invalid_range__{name}",
                ),
            )
        )

    observed = df.agg(*counters).first().asDict()
    return {
        name: {
            "missing_input_count": int(observed[f"missing__{name}"]),
            "invalid_range_count": int(observed[f"invalid_range__{name}"]),
        }
        for name in CONTINUOUS_FEATURES
    }


_IDENTIFIER_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
_SQL_TO_SPARK_TYPE = {
    "DOUBLE": "double",
    "INT": "int",
    "BIGINT": "bigint",
    "STRING": "string",
    "TIMESTAMP": "timestamp",
}
SPLIT_PERIODS = {
    "train": ("2019-01-01 00:00:00", "2024-01-01 00:00:00"),
    "validation": ("2024-01-01 00:00:00", "2025-01-01 00:00:00"),
    "evaluation": ("2025-01-01 00:00:00", "2026-01-01 00:00:00"),
    "excluded": ("2026-01-01 00:00:00", None),
}
EXPECTED_SPLIT_ROWS = {
    "train": 920_304,
    "validation": 184_464,
    "evaluation": 183_960,
    "excluded": 91_203,
    "total": 1_379_931,
}


@dataclass(frozen=True)
class SnapshotProvenance:
    """Version and lineage identity of the two Gold preprocessing inputs."""

    labels_table: str
    features_table: str
    labels_delta_version: int
    features_delta_version: int
    snapshot_id: str
    input_versions_json: str
    code_commit: str
    feature_schema_version: str
    time_contract: str


@dataclass(frozen=True)
class GoldTrainingSnapshot:
    """Versioned Gold feature input and its validated provenance."""

    features: DataFrame
    provenance: SnapshotProvenance


@dataclass(frozen=True)
class TemporalSplits:
    """Retrospective train, validation, evaluation, and excluded partitions."""

    train: DataFrame
    validation: DataFrame
    evaluation: DataFrame
    excluded: DataFrame
    row_counts: dict[str, int]


def _validate_snapshot_schema(
    df: DataFrame,
    expected_columns: Sequence[tuple[str, str, bool]],
    logical_table: str,
) -> None:
    expected = tuple(
        (name, _SQL_TO_SPARK_TYPE[sql_type], nullable)
        for name, sql_type, nullable in expected_columns
    )
    actual = tuple(
        (field.name, field.dataType.simpleString(), field.nullable)
        for field in df.schema.fields
    )
    if actual != expected:
        raise PreprocessingContractError(
            f"{logical_table}: schema differs from the approved contract; "
            f"expected={expected}, actual={actual}"
        )


def _build_provenance(
    labels_df: DataFrame,
    features_df: DataFrame,
    *,
    labels_table: str,
    features_table: str,
    labels_delta_version: int,
    features_delta_version: int,
    expected_snapshot_id: str,
) -> SnapshotProvenance:
    try:
        labels_lineage, _ = validate_matching_lineage(labels_df, features_df)
    except SnapshotContractError as exc:
        raise PreprocessingContractError(
            f"Gold snapshot lineage validation failed: {exc}"
        ) from exc

    observed_snapshot_id = labels_lineage["snapshot_id"]
    if observed_snapshot_id != expected_snapshot_id:
        raise PreprocessingContractError(
            "expected_snapshot_id differs from Gold lineage: "
            f"expected={expected_snapshot_id!r}, observed={observed_snapshot_id!r}"
        )
    if labels_lineage["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise PreprocessingContractError(
            "feature_schema_version differs from the preprocessing contract: "
            f"expected={FEATURE_SCHEMA_VERSION!r}, "
            f"observed={labels_lineage['feature_schema_version']!r}"
        )
    if labels_lineage["time_contract"] != TIME_CONTRACT:
        raise PreprocessingContractError(
            "time_contract differs from the preprocessing contract: "
            f"expected={TIME_CONTRACT!r}, observed={labels_lineage['time_contract']!r}"
        )

    return SnapshotProvenance(
        labels_table=labels_table,
        features_table=features_table,
        labels_delta_version=labels_delta_version,
        features_delta_version=features_delta_version,
        snapshot_id=observed_snapshot_id,
        input_versions_json=labels_lineage["input_versions_json"],
        code_commit=labels_lineage["code_commit"],
        feature_schema_version=labels_lineage["feature_schema_version"],
        time_contract=labels_lineage["time_contract"],
    )


def _validate_load_inputs(
    gold_catalog: str,
    labels_delta_version: int,
    features_delta_version: int,
    expected_snapshot_id: str,
) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(gold_catalog):
        raise PreprocessingContractError(
            f"gold_catalog is not a valid identifier: {gold_catalog!r}"
        )
    if labels_delta_version < 0:
        raise PreprocessingContractError(
            "labels_delta_version must be greater than or equal to zero"
        )
    if features_delta_version < 0:
        raise PreprocessingContractError(
            "features_delta_version must be greater than or equal to zero"
        )
    if not expected_snapshot_id.strip():
        raise PreprocessingContractError("expected_snapshot_id must not be empty")


def load_gold_training_snapshot(
    spark: SparkSession,
    *,
    gold_catalog: str,
    labels_delta_version: int,
    features_delta_version: int,
    expected_snapshot_id: str,
) -> GoldTrainingSnapshot:
    """Read exact Gold Delta versions and validate schema plus shared lineage."""
    _validate_load_inputs(
        gold_catalog,
        labels_delta_version,
        features_delta_version,
        expected_snapshot_id,
    )
    labels_table = f"{gold_catalog}.{ML_SCHEMA}.{LABEL_TABLE}"
    features_table = f"{gold_catalog}.{ML_SCHEMA}.{FEATURE_TABLE}"
    missing = [
        table_name
        for table_name in (labels_table, features_table)
        if not spark.catalog.tableExists(table_name)
    ]
    if missing:
        raise PreprocessingContractError(
            "required Gold tables do not exist: " + ", ".join(missing)
        )

    labels_df = (
        spark.read.format("delta")
        .option("versionAsOf", labels_delta_version)
        .table(labels_table)
    )
    features_df = (
        spark.read.format("delta")
        .option("versionAsOf", features_delta_version)
        .table(features_table)
    )
    _validate_snapshot_schema(labels_df, LABEL_TABLE_COLUMNS, labels_table)
    _validate_snapshot_schema(features_df, FEATURE_TABLE_COLUMNS, features_table)
    provenance = _build_provenance(
        labels_df,
        features_df,
        labels_table=labels_table,
        features_table=features_table,
        labels_delta_version=labels_delta_version,
        features_delta_version=features_delta_version,
        expected_snapshot_id=expected_snapshot_id,
    )
    return GoldTrainingSnapshot(features=features_df, provenance=provenance)


def _split_by_feature_hour(df: DataFrame) -> TemporalSplits:
    spark = df.sparkSession
    spark.conf.set("spark.sql.session.timeZone", SESSION_TIMEZONE)
    observed_timezone = spark.conf.get("spark.sql.session.timeZone")
    if observed_timezone != SESSION_TIMEZONE:
        raise PreprocessingContractError(
            "spark.sql.session.timeZone differs from the preprocessing contract: "
            f"expected={SESSION_TIMEZONE!r}, observed={observed_timezone!r}"
        )

    predicates = {
        name: (
            (F.col("feature_hour") >= F.to_timestamp(F.lit(start)))
            & (
                F.col("feature_hour") < F.to_timestamp(F.lit(end))
                if end is not None
                else F.lit(True)
            )
        )
        for name, (start, end) in SPLIT_PERIODS.items()
    }
    count_columns = [
        _count_matching(predicate, name)
        for name, predicate in predicates.items()
    ]
    count_columns.append(_count_matching(F.lit(True), "total"))
    row_counts = {
        name: int(value)
        for name, value in df.agg(*count_columns).first().asDict().items()
    }
    return TemporalSplits(
        train=df.where(predicates["train"]),
        validation=df.where(predicates["validation"]),
        evaluation=df.where(predicates["evaluation"]),
        excluded=df.where(predicates["excluded"]),
        row_counts=row_counts,
    )


def validate_split_counts(row_counts: dict[str, int]) -> None:
    """Require exact v1 split counts and complete input reconciliation."""
    failures = [
        f"{name}: expected={expected}, observed={row_counts.get(name)!r}"
        for name, expected in EXPECTED_SPLIT_ROWS.items()
        if row_counts.get(name) != expected
    ]
    included = sum(
        row_counts.get(name, 0)
        for name in ("train", "validation", "evaluation", "excluded")
    )
    if included != row_counts.get("total"):
        failures.append(
            f"reconciliation: partitions={included}, total={row_counts.get('total')!r}"
        )
    unexpected = sorted(set(row_counts).difference(EXPECTED_SPLIT_ROWS))
    if unexpected:
        failures.append(f"unexpected split counters={unexpected}")
    if failures:
        raise PreprocessingContractError(
            "Gold temporal split validation failed: " + "; ".join(failures)
        )


def split_training_snapshot(snapshot: GoldTrainingSnapshot) -> TemporalSplits:
    """Validate and split the approved Gold snapshot without random sampling."""
    validate_feature_domains(snapshot.features)
    splits = _split_by_feature_hour(snapshot.features)
    validate_split_counts(splits.row_counts)
    return splits


IMPUTER_RELATIVE_ERROR = 0.001
IMPUTED_FEATURES = tuple(f"{name}__imputed" for name in CONTINUOUS_FEATURES)
CATEGORY_FEATURES = ("cod_distrito", "hora_dia", "dia_semana", "mes")
CATEGORY_STRINGS = tuple(f"{name}__category" for name in CATEGORY_FEATURES)
CATEGORY_INDICES = tuple(f"{name}__index" for name in CATEGORY_FEATURES)
CATEGORY_VECTORS = tuple(f"{name}__one_hot" for name in CATEGORY_FEATURES)
NUMERIC_RAW_VECTOR = "__numeric_raw"
NUMERIC_SCALED_VECTOR = "__numeric_scaled"
MODEL_FEATURES_COLUMN = "features"
CATEGORY_DOMAINS = {
    "cod_distrito": tuple(f"{value:02d}" for value in range(1, 22)),
    "hora_dia": tuple(f"{value:02d}" for value in range(24)),
    "dia_semana": tuple(str(value) for value in range(1, 8)),
    "mes": tuple(f"{value:02d}" for value in range(1, 13)),
}
CATEGORY_REFERENCES = {
    name: labels[-1] for name, labels in CATEGORY_DOMAINS.items()
}


@dataclass(frozen=True)
class PreprocessingManifest:
    """Complete deterministic and learned preprocessing contract state."""

    preprocessing_contract_version: str
    provenance: SnapshotProvenance
    spark_version: str
    session_timezone: str
    split_periods: dict[str, tuple[str, str | None]]
    split_rows: dict[str, int]
    physical_ranges: dict[str, dict[str, float | bool | None]]
    imputer_relative_error: float
    numeric_features: tuple[str, ...]
    available_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    imputation_medians: dict[str, float]
    scaler_mean: tuple[float, ...]
    scaler_std: tuple[float, ...]
    category_labels: dict[str, tuple[str, ...]]
    category_references: dict[str, str]
    vector_size: int


@dataclass(frozen=True)
class FittedPreprocessor:
    """PipelineModel plus the package-side deterministic preprocessing contract."""

    pipeline_model: PipelineModel
    manifest: PreprocessingManifest

    def transform(self, df: DataFrame, *, split_name: str) -> DataFrame:
        """Transform one included split and enforce conservation plus vector gates."""
        if split_name not in {"train", "validation", "evaluation"}:
            raise PreprocessingContractError(
                f"invalid split_name {split_name!r}; expected train, validation, or evaluation"
            )
        prepared = _prepare_for_pipeline(df)
        transformed = self.pipeline_model.transform(prepared)
        _validate_transformed_split(df, transformed, split_name)
        return transformed


def transform_inference_features(
    pipeline_model: PipelineModel,
    df: DataFrame,
) -> DataFrame:
    """Apply the fitted training pipeline to an unlabeled inference snapshot."""
    validate_inference_feature_domains(df)
    transformed = pipeline_model.transform(_prepare_for_pipeline(df))
    _validate_transformed_features(
        df,
        transformed,
        identity=("cod_distrito", "feature_hour"),
        label="inference",
    )
    return transformed


@dataclass(frozen=True)
class PreparedTrainingData:
    """Three transformed model splits and their fitted preprocessing artifact."""

    train: DataFrame
    validation: DataFrame
    evaluation: DataFrame
    fitted_preprocessor: FittedPreprocessor
    quality_by_split: dict[str, dict[str, dict[str, int]]]


def _prepare_for_pipeline(df: DataFrame) -> DataFrame:
    prepared = sanitize_features(df)
    for name in CATEGORY_FEATURES:
        width = 1 if name == "dia_semana" else 2
        prepared = prepared.withColumn(
            f"{name}__category",
            F.format_string(f"%0{width}d", F.col(name)),
        )
    return prepared


def _pipeline() -> Pipeline:
    return Pipeline(
        stages=[
            Imputer(
                strategy="median",
                relativeError=IMPUTER_RELATIVE_ERROR,
                inputCols=list(CONTINUOUS_FEATURES),
                outputCols=list(IMPUTED_FEATURES),
            ),
            VectorAssembler(
                inputCols=list(IMPUTED_FEATURES + COUNT_FEATURES),
                outputCol=NUMERIC_RAW_VECTOR,
                handleInvalid="error",
            ),
            StandardScaler(
                inputCol=NUMERIC_RAW_VECTOR,
                outputCol=NUMERIC_SCALED_VECTOR,
                withMean=True,
                withStd=True,
            ),
            StringIndexer(
                inputCols=list(CATEGORY_STRINGS),
                outputCols=list(CATEGORY_INDICES),
                stringOrderType="alphabetAsc",
                handleInvalid="error",
            ),
            OneHotEncoder(
                inputCols=list(CATEGORY_INDICES),
                outputCols=list(CATEGORY_VECTORS),
                dropLast=True,
                handleInvalid="error",
            ),
            VectorAssembler(
                inputCols=[
                    NUMERIC_SCALED_VECTOR,
                    *AVAILABLE_FEATURES,
                    *CATEGORY_VECTORS,
                ],
                outputCol=MODEL_FEATURES_COLUMN,
                handleInvalid="error",
            ),
        ]
    )


def _require_valid_train_values(prepared_train: DataFrame) -> None:
    observed = (
        prepared_train.agg(
            *[
                F.count(F.col(name)).cast("long").alias(name)
                for name in CONTINUOUS_FEATURES
            ]
        )
        .first()
        .asDict()
    )
    missing = [name for name, count in observed.items() if count == 0]
    if missing:
        raise PreprocessingContractError(
            "train has no valid values for continuous features: "
            + ", ".join(missing)
        )


def _extract_manifest(
    model: PipelineModel,
    provenance: SnapshotProvenance,
    *,
    split_rows: dict[str, int],
    spark_version: str,
) -> PreprocessingManifest:
    stages = model.stages
    imputer_model = stages[0]
    scaler_model = stages[2]
    indexer_model = stages[3]
    encoder_model = stages[4]
    if not isinstance(imputer_model, ImputerModel):
        raise PreprocessingContractError("pipeline stage 0 is not an ImputerModel")
    if not isinstance(scaler_model, StandardScalerModel):
        raise PreprocessingContractError(
            "pipeline stage 2 is not a StandardScalerModel"
        )
    if not isinstance(indexer_model, StringIndexerModel):
        raise PreprocessingContractError(
            "pipeline stage 3 is not a StringIndexerModel"
        )
    if not isinstance(encoder_model, OneHotEncoderModel):
        raise PreprocessingContractError(
            "pipeline stage 4 is not a OneHotEncoderModel"
        )

    surrogate = imputer_model.surrogateDF.first().asDict()
    medians = {
        name: float(surrogate[name])
        for name in CONTINUOUS_FEATURES
        if surrogate.get(name) is not None
    }
    if tuple(medians) != CONTINUOUS_FEATURES:
        missing = sorted(set(CONTINUOUS_FEATURES).difference(medians))
        raise PreprocessingContractError(
            "ImputerModel is missing train medians for: " + ", ".join(missing)
        )

    scaler_mean = tuple(float(value) for value in scaler_model.mean)
    scaler_std = tuple(float(value) for value in scaler_model.std)
    if len(scaler_mean) != len(NUMERIC_FEATURES) or len(scaler_std) != len(
        NUMERIC_FEATURES
    ):
        raise PreprocessingContractError(
            "StandardScalerModel state does not contain 33 numeric components"
        )
    zero_std_features = [
        NUMERIC_FEATURES[index]
        for index, value in enumerate(scaler_std)
        if value <= 0.0
    ]
    if zero_std_features:
        raise PreprocessingContractError(
            "train standard deviation is not positive for: "
            + ", ".join(zero_std_features)
        )

    category_labels = {
        name: tuple(labels)
        for name, labels in zip(
            CATEGORY_FEATURES,
            indexer_model.labelsArray,
            strict=True,
        )
    }
    category_failures = [
        name
        for name in CATEGORY_FEATURES
        if category_labels[name] != CATEGORY_DOMAINS[name]
    ]
    if category_failures:
        details = ", ".join(
            f"{name}: expected={CATEGORY_DOMAINS[name]}, observed={category_labels[name]}"
            for name in category_failures
        )
        raise PreprocessingContractError(
            "train category domains differ from the contract: " + details
        )
    category_sizes = tuple(int(size) for size in encoder_model.categorySizes)
    if category_sizes != (21, 24, 7, 12):
        raise PreprocessingContractError(
            "OneHotEncoderModel category sizes differ from (21, 24, 7, 12): "
            f"observed={category_sizes}"
        )

    return PreprocessingManifest(
        preprocessing_contract_version=PREPROCESSING_CONTRACT_VERSION,
        provenance=provenance,
        spark_version=spark_version,
        session_timezone=SESSION_TIMEZONE,
        split_periods=dict(SPLIT_PERIODS),
        split_rows=dict(split_rows),
        physical_ranges={
            name: asdict(rule) for name, rule in PHYSICAL_RANGES.items()
        },
        imputer_relative_error=IMPUTER_RELATIVE_ERROR,
        numeric_features=NUMERIC_FEATURES,
        available_features=AVAILABLE_FEATURES,
        categorical_features=CATEGORY_FEATURES,
        imputation_medians=medians,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        category_labels=category_labels,
        category_references=dict(CATEGORY_REFERENCES),
        vector_size=EXPECTED_VECTOR_SIZE,
    )


def fit_preprocessor(
    train: DataFrame,
    provenance: SnapshotProvenance,
    *,
    split_rows: dict[str, int],
    spark_version: str,
) -> FittedPreprocessor:
    """Fit every learned preprocessing stage exclusively on train."""
    validate_feature_domains(train)
    prepared_train = _prepare_for_pipeline(train)
    _require_valid_train_values(prepared_train)
    try:
        model = _pipeline().fit(prepared_train)
    except Exception as exc:
        raise PreprocessingContractError(
            "Spark failed while fitting the train-only preprocessing pipeline"
        ) from exc
    manifest = _extract_manifest(
        model,
        provenance,
        split_rows=split_rows,
        spark_version=spark_version,
    )
    return FittedPreprocessor(pipeline_model=model, manifest=manifest)


def _validate_transformed_split(
    source: DataFrame,
    transformed: DataFrame,
    split_name: str,
) -> None:
    _validate_transformed_features(
        source,
        transformed,
        identity=("cod_distrito", "feature_hour", "target_accident_next_hour"),
        label=split_name,
    )


def _validate_transformed_features(
    source: DataFrame,
    transformed: DataFrame,
    *,
    identity: tuple[str, ...],
    label: str,
) -> None:
    source_count = source.count()
    transformed_count = transformed.count()
    if transformed_count != source_count:
        raise PreprocessingContractError(
            f"{label}: row count changed during transform; "
            f"before={source_count}, after={transformed_count}"
        )

    missing_identity = (
        source.select(*identity)
        .exceptAll(transformed.select(*identity))
        .limit(1)
        .count()
    )
    extra_identity = (
        transformed.select(*identity)
        .exceptAll(source.select(*identity))
        .limit(1)
        .count()
    )
    if missing_identity or extra_identity:
        raise PreprocessingContractError(
            f"{label}: identity columns changed during transform"
        )

    values = vector_to_array(F.col(MODEL_FEATURES_COLUMN))
    invalid_value = F.exists(
        values,
        lambda value: (
            value.isNull()
            | F.isnan(value)
            | (value == F.lit(float("inf")))
            | (value == F.lit(float("-inf")))
        ),
    )
    summary = transformed.agg(
        F.min(F.size(values)).alias("minimum_size"),
        F.max(F.size(values)).alias("maximum_size"),
        _count_matching(invalid_value, "invalid_vectors"),
    ).first()
    if (
        summary.minimum_size != EXPECTED_VECTOR_SIZE
        or summary.maximum_size != EXPECTED_VECTOR_SIZE
    ):
        raise PreprocessingContractError(
            f"{label}: feature vector size differs from {EXPECTED_VECTOR_SIZE}; "
            f"minimum={summary.minimum_size}, maximum={summary.maximum_size}"
        )
    if summary.invalid_vectors:
        raise PreprocessingContractError(
            f"{label}: found {summary.invalid_vectors} non-finite feature vectors"
        )


def prepare_training_data(
    spark: SparkSession,
    *,
    gold_catalog: str,
    labels_delta_version: int,
    features_delta_version: int,
    expected_snapshot_id: str,
) -> PreparedTrainingData:
    """Load, split, fit on train, and transform the three included periods."""
    snapshot = load_gold_training_snapshot(
        spark,
        gold_catalog=gold_catalog,
        labels_delta_version=labels_delta_version,
        features_delta_version=features_delta_version,
        expected_snapshot_id=expected_snapshot_id,
    )
    splits = split_training_snapshot(snapshot)
    quality_by_split = {
        "train": profile_feature_quality(splits.train),
        "validation": profile_feature_quality(splits.validation),
        "evaluation": profile_feature_quality(splits.evaluation),
    }
    fitted = fit_preprocessor(
        splits.train,
        snapshot.provenance,
        split_rows=splits.row_counts,
        spark_version=spark.version,
    )
    return PreparedTrainingData(
        train=fitted.transform(splits.train, split_name="train"),
        validation=fitted.transform(
            splits.validation,
            split_name="validation",
        ),
        evaluation=fitted.transform(
            splits.evaluation,
            split_name="evaluation",
        ),
        fitted_preprocessor=fitted,
        quality_by_split=quality_by_split,
    )


def manifest_to_dict(
    manifest: PreprocessingManifest,
    *,
    preprocessing_code_commit: str,
    wheel_sha256: str,
    package_version: str,
    runtime_environment_version: str,
    python_version: str,
) -> dict[str, object]:
    """Serialize contract state plus the exact package and runtime identity."""
    payload = asdict(manifest)
    payload["artifact_identity"] = {
        "preprocessing_code_commit": preprocessing_code_commit,
        "wheel_sha256": wheel_sha256,
        "package_version": package_version,
        "runtime_environment_version": runtime_environment_version,
        "python_version": python_version,
    }
    return payload
