from datetime import datetime, timedelta

import pytest
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from madrid_ml.contract import FEATURE_TABLE_COLUMNS, LABEL_TABLE_COLUMNS
from madrid_ml.preprocessing import (
    CONTINUOUS_FEATURES,
    COUNT_FEATURES,
    PHYSICAL_RANGES,
    PreprocessingContractError,
    sanitize_features,
)

NON_NEGATIVE_FEATURES = (
    "trafico_intensidad_media",
    "trafico_ocupacion_media",
    "trafico_carga_media",
    "trafico_vmed_media",
    "meteo_velocidad_viento_media",
    "meteo_radiacion_solar_media",
    "meteo_precipitacion_media",
    "calair_so2_media",
    "calair_co_media",
    "calair_no_media",
    "calair_no2_media",
    "calair_pm25_media",
    "calair_pm10_media",
    "calair_nox_media",
    "calair_o3_media",
)
TRAFFIC_MEANS = (
    "trafico_intensidad_media",
    "trafico_ocupacion_media",
    "trafico_carga_media",
    "trafico_vmed_media",
)

MAGNITUDE_COUNT_BY_MEAN = {
    "meteo_velocidad_viento_media": "meteo_velocidad_viento_n",
    "meteo_temperatura_media": "meteo_temperatura_n",
    "meteo_humedad_relativa_media": "meteo_humedad_relativa_n",
    "meteo_presion_media": "meteo_presion_n",
    "meteo_radiacion_solar_media": "meteo_radiacion_solar_n",
    "meteo_precipitacion_media": "meteo_precipitacion_n",
    "calair_so2_media": "calair_so2_n",
    "calair_co_media": "calair_co_n",
    "calair_no_media": "calair_no_n",
    "calair_no2_media": "calair_no2_n",
    "calair_pm25_media": "calair_pm25_n",
    "calair_pm10_media": "calair_pm10_n",
    "calair_nox_media": "calair_nox_n",
    "calair_o3_media": "calair_o3_n",
}


def validate_feature_domains(df):
    from madrid_ml.preprocessing import validate_feature_domains as validate

    return validate(df)


def profile_feature_quality(df):
    from madrid_ml.preprocessing import profile_feature_quality as profile

    return profile(df)


def make_row(index: int = 0, **overrides):
    feature_hour = datetime(
        2020,
        index % 12 + 1,
        index % 27 + 1,
        index % 24,
    )
    row = {
        "cod_distrito": index % 21 + 1,
        "feature_hour": feature_hour,
        "prediction_hour": feature_hour + timedelta(hours=1),
        "n_accidentes_next_hour": 0,
        "target_accident_next_hour": index % 2,
        "hora_dia": index % 24,
        "dia_semana": index % 7 + 1,
        "mes": index % 12 + 1,
        "snapshot_id": "snapshot-test",
        "input_versions_json": "{}",
        "code_commit": "a" * 40,
        "feature_schema_version": "1",
        "time_contract": "source_wall_clock_as_stored_in_silver",
    }
    for name in CONTINUOUS_FEATURES:
        row[name] = float(index + 1)
    for name in COUNT_FEATURES:
        row[name] = index % 3 + 1
    row.update(overrides)
    return row


def test_sanitize_marks_non_finite_and_impossible_values_unavailable(spark):
    frame = spark.createDataFrame(
        [
            make_row(
                trafico_vmed_media=-1.0,
                trafico_ocupacion_media=float("nan"),
                meteo_radiacion_solar_media=float("inf"),
                calair_no2_media=-0.1,
            )
        ]
    )

    row = sanitize_features(frame).first()

    assert row.trafico_vmed_media is None
    assert row.trafico_vmed_media_available == 0.0
    assert row.trafico_ocupacion_media is None
    assert row.trafico_ocupacion_media_available == 0.0
    assert row.meteo_radiacion_solar_media is None
    assert row.meteo_radiacion_solar_media_available == 0.0
    assert row.calair_no2_media is None
    assert row.calair_no2_media_available == 0.0


def test_sanitize_keeps_valid_zero_and_negative_temperature(spark):
    frame = spark.createDataFrame(
        [
            make_row(
                trafico_intensidad_media=0.0,
                meteo_temperatura_media=-55.0,
            )
        ]
    )

    row = sanitize_features(frame).first()

    assert row.trafico_intensidad_media == 0.0
    assert row.trafico_intensidad_media_available == 1.0
    assert row.meteo_temperatura_media == -55.0
    assert row.meteo_temperatura_media_available == 1.0


@pytest.mark.parametrize("column_name", NON_NEGATIVE_FEATURES)
def test_sanitize_enforces_each_non_negative_physical_range(spark, column_name):
    frame = spark.createDataFrame(
        [make_row(index=0, **{column_name: -0.1}), make_row(index=1, **{column_name: 0.0})]
    )

    invalid, boundary = sanitize_features(frame).select(
        column_name, f"{column_name}_available"
    ).collect()

    assert invalid[column_name] is None
    assert invalid[f"{column_name}_available"] == 0.0
    assert boundary[column_name] == 0.0
    assert boundary[f"{column_name}_available"] == 1.0


@pytest.mark.parametrize(
    ("invalid_value", "valid_value"),
    [(-0.1, 0.0), (100.1, 100.0)],
)
def test_sanitize_enforces_humidity_bounds(spark, invalid_value, valid_value):
    column_name = "meteo_humedad_relativa_media"
    frame = spark.createDataFrame(
        [
            make_row(index=0, **{column_name: invalid_value}),
            make_row(index=1, **{column_name: valid_value}),
        ]
    )

    invalid, boundary = sanitize_features(frame).select(
        column_name, f"{column_name}_available"
    ).collect()

    assert invalid[column_name] is None
    assert invalid[f"{column_name}_available"] == 0.0
    assert boundary[column_name] == valid_value
    assert boundary[f"{column_name}_available"] == 1.0


def test_sanitize_enforces_strictly_positive_pressure(spark):
    column_name = "meteo_presion_media"
    frame = spark.createDataFrame(
        [make_row(index=0, **{column_name: 0.0}), make_row(index=1, **{column_name: 1.0})]
    )

    invalid, boundary = sanitize_features(frame).select(
        column_name, f"{column_name}_available"
    ).collect()

    assert invalid[column_name] is None
    assert invalid[f"{column_name}_available"] == 0.0
    assert boundary[column_name] == 1.0
    assert boundary[f"{column_name}_available"] == 1.0


def test_physical_ranges_cover_every_continuous_feature():
    assert tuple(PHYSICAL_RANGES) == CONTINUOUS_FEATURES


def test_validate_domains_rejects_negative_count(spark):
    frame = spark.createDataFrame([make_row(trafico_puntos_n=-1)])

    with pytest.raises(PreprocessingContractError, match="trafico_puntos_n"):
        validate_feature_domains(frame)


def test_validate_domains_rejects_null_count(spark):
    frame = spark.createDataFrame(
        [make_row(index=0), make_row(index=1, trafico_puntos_n=None)]
    )

    with pytest.raises(PreprocessingContractError, match="trafico_puntos_n"):
        validate_feature_domains(frame)


@pytest.mark.parametrize(
    ("mean_column", "count_column"),
    tuple(MAGNITUDE_COUNT_BY_MEAN.items()),
)
def test_validate_domains_rejects_zero_count_with_finite_magnitude_mean(
    spark, mean_column, count_column
):
    frame = spark.createDataFrame(
        [make_row(**{count_column: 0, mean_column: 12.0})]
    )

    with pytest.raises(PreprocessingContractError, match=mean_column):
        validate_feature_domains(frame)


@pytest.mark.parametrize("mean_column", TRAFFIC_MEANS)
def test_validate_domains_rejects_zero_traffic_count_with_finite_mean(
    spark, mean_column
):
    frame = spark.createDataFrame(
        [make_row(trafico_puntos_n=0, **{mean_column: -0.1})]
    )

    with pytest.raises(PreprocessingContractError, match=mean_column):
        validate_feature_domains(frame)


def test_validate_domains_allows_positive_count_with_unusable_mean(spark):
    frame = spark.createDataFrame(
        [make_row(meteo_temperatura_n=1, meteo_temperatura_media=float("nan"))]
    )

    validate_feature_domains(frame)


def test_validate_domains_rejects_invalid_category_and_target(spark):
    frame = spark.createDataFrame(
        [make_row(cod_distrito=22, target_accident_next_hour=2)]
    )

    with pytest.raises(
        PreprocessingContractError,
        match="cod_distrito.*target_accident_next_hour|target_accident_next_hour.*cod_distrito",
    ):
        validate_feature_domains(frame)


def test_validate_domains_rejects_null_category_and_target(spark):
    frame = spark.createDataFrame(
        [
            make_row(index=0),
            make_row(index=1, cod_distrito=None, target_accident_next_hour=None),
        ]
    )

    with pytest.raises(
        PreprocessingContractError,
        match="cod_distrito.*target_accident_next_hour|target_accident_next_hour.*cod_distrito",
    ):
        validate_feature_domains(frame)


def test_profile_feature_quality_distinguishes_missing_and_invalid(spark):
    frame = spark.createDataFrame(
        [
            make_row(index=0, trafico_vmed_media=float("nan")),
            make_row(index=1, trafico_vmed_media=float("inf")),
            make_row(index=2, trafico_vmed_media=-1.0),
            make_row(index=3, trafico_vmed_media=0.0),
        ]
    )

    profile = profile_feature_quality(frame)

    assert profile["trafico_vmed_media"] == {
        "missing_input_count": 2,
        "invalid_range_count": 1,
    }


_SPARK_TYPES = {
    "DOUBLE": DoubleType,
    "INT": IntegerType,
    "BIGINT": LongType,
    "STRING": StringType,
    "TIMESTAMP": TimestampType,
}


def contract_schema(columns):
    return StructType(
        [
            StructField(name, _SPARK_TYPES[sql_type](), nullable)
            for name, sql_type, nullable in columns
        ]
    )


def lineage_frame(spark, **overrides):
    values = {
        "snapshot_id": "snapshot-test",
        "input_versions_json": '{"silver.table":7}',
        "code_commit": "a" * 40,
        "feature_schema_version": "1",
        "time_contract": "source_wall_clock_as_stored_in_silver",
    }
    values.update(overrides)
    return spark.createDataFrame([values])


def test_split_uses_utc_half_open_boundaries(spark):
    frame = spark.createDataFrame(
        [
            make_row(feature_hour=datetime(2023, 12, 31, 23), hora_dia=23, mes=12),
            make_row(feature_hour=datetime(2024, 1, 1), hora_dia=0, mes=1),
            make_row(feature_hour=datetime(2025, 1, 1), hora_dia=0, mes=1),
            make_row(feature_hour=datetime(2026, 1, 1), hora_dia=0, mes=1),
        ]
    )
    from madrid_ml.preprocessing import _split_by_feature_hour

    splits = _split_by_feature_hour(frame)

    assert splits.train.count() == 1
    assert splits.validation.count() == 1
    assert splits.evaluation.count() == 1
    assert splits.excluded.count() == 1
    assert splits.row_counts == {
        "train": 1,
        "validation": 1,
        "evaluation": 1,
        "excluded": 1,
        "total": 4,
    }
    assert spark.conf.get("spark.sql.session.timeZone") == "Etc/UTC"


def test_validate_split_counts_requires_production_reconciliation():
    from madrid_ml.preprocessing import validate_split_counts

    validate_split_counts(
        {
            "train": 920_304,
            "validation": 184_464,
            "evaluation": 183_960,
            "excluded": 91_203,
            "total": 1_379_931,
        }
    )

    with pytest.raises(PreprocessingContractError, match="validation"):
        validate_split_counts(
            {
                "train": 920_304,
                "validation": 184_463,
                "evaluation": 183_960,
                "excluded": 91_203,
                "total": 1_379_930,
            }
        )


def test_validate_snapshot_schema_accepts_exact_contract(spark):
    from madrid_ml.preprocessing import _validate_snapshot_schema

    labels = spark.createDataFrame([], contract_schema(LABEL_TABLE_COLUMNS))
    features = spark.createDataFrame([], contract_schema(FEATURE_TABLE_COLUMNS))

    _validate_snapshot_schema(labels, LABEL_TABLE_COLUMNS, "labels")
    _validate_snapshot_schema(features, FEATURE_TABLE_COLUMNS, "features")


@pytest.mark.parametrize("change", ["missing", "extra", "wrong_type", "wrong_nullable"])
def test_validate_snapshot_schema_rejects_incompatible_contract(spark, change):
    from madrid_ml.preprocessing import _validate_snapshot_schema

    fields = list(contract_schema(FEATURE_TABLE_COLUMNS).fields)
    if change == "missing":
        fields.pop()
    elif change == "extra":
        fields.append(StructField("unexpected", StringType(), True))
    elif change == "wrong_type":
        fields[0] = StructField(fields[0].name, StringType(), fields[0].nullable)
    else:
        fields[0] = StructField(
            fields[0].name,
            fields[0].dataType,
            not fields[0].nullable,
        )
    features = spark.createDataFrame([], StructType(fields))

    with pytest.raises(PreprocessingContractError, match="features.*schema"):
        _validate_snapshot_schema(features, FEATURE_TABLE_COLUMNS, "features")


def test_build_provenance_preserves_gold_versions_and_silver_lineage(spark):
    from madrid_ml.preprocessing import _build_provenance

    provenance = _build_provenance(
        lineage_frame(spark),
        lineage_frame(spark),
        labels_table="dev_gold.ml.accident_labels_hourly",
        features_table="dev_gold.ml.features_training_snapshot",
        labels_delta_version=1,
        features_delta_version=2,
        expected_snapshot_id="snapshot-test",
    )

    assert provenance.labels_delta_version == 1
    assert provenance.features_delta_version == 2
    assert provenance.snapshot_id == "snapshot-test"
    assert provenance.input_versions_json == '{"silver.table":7}'
    assert provenance.code_commit == "a" * 40


@pytest.mark.parametrize(
    ("column_name", "changed_value"),
    [
        ("snapshot_id", "other-snapshot"),
        ("feature_schema_version", "2"),
        ("time_contract", "other-time-contract"),
    ],
)
def test_build_provenance_rejects_mismatched_lineage(
    spark, column_name, changed_value
):
    from madrid_ml.preprocessing import _build_provenance

    with pytest.raises(PreprocessingContractError, match=column_name):
        _build_provenance(
            lineage_frame(spark),
            lineage_frame(spark, **{column_name: changed_value}),
            labels_table="dev_gold.ml.accident_labels_hourly",
            features_table="dev_gold.ml.features_training_snapshot",
            labels_delta_version=1,
            features_delta_version=1,
            expected_snapshot_id="snapshot-test",
        )


def test_build_provenance_rejects_unexpected_snapshot(spark):
    from madrid_ml.preprocessing import _build_provenance

    with pytest.raises(PreprocessingContractError, match="expected_snapshot_id"):
        _build_provenance(
            lineage_frame(spark),
            lineage_frame(spark),
            labels_table="dev_gold.ml.accident_labels_hourly",
            features_table="dev_gold.ml.features_training_snapshot",
            labels_delta_version=1,
            features_delta_version=1,
            expected_snapshot_id="other-snapshot",
        )


@pytest.mark.parametrize(
    ("gold_catalog", "labels_version", "features_version", "snapshot_id", "error"),
    [
        ("bad-catalog", 1, 1, "snapshot-test", "gold_catalog"),
        ("dev_gold", -1, 1, "snapshot-test", "labels_delta_version"),
        ("dev_gold", 1, -1, "snapshot-test", "features_delta_version"),
        ("dev_gold", 1, 1, "", "expected_snapshot_id"),
    ],
)
def test_load_gold_rejects_invalid_inputs_before_catalog_access(
    spark,
    gold_catalog,
    labels_version,
    features_version,
    snapshot_id,
    error,
):
    from madrid_ml.preprocessing import load_gold_training_snapshot

    with pytest.raises(PreprocessingContractError, match=error):
        load_gold_training_snapshot(
            spark,
            gold_catalog=gold_catalog,
            labels_delta_version=labels_version,
            features_delta_version=features_version,
            expected_snapshot_id=snapshot_id,
        )


def make_complete_training_frame(spark):
    return spark.createDataFrame([make_row(index=index) for index in range(504)])


def make_test_provenance():
    from madrid_ml.preprocessing import SnapshotProvenance

    return SnapshotProvenance(
        labels_table="dev_gold.ml.accident_labels_hourly",
        features_table="dev_gold.ml.features_training_snapshot",
        labels_delta_version=1,
        features_delta_version=1,
        snapshot_id="snapshot-test",
        input_versions_json='{"silver.table":7}',
        code_commit="a" * 40,
        feature_schema_version="1",
        time_contract="source_wall_clock_as_stored_in_silver",
    )


def fit_test_preprocessor(spark, train):
    from madrid_ml.preprocessing import fit_preprocessor

    train_rows = train.count()
    return fit_preprocessor(
        train,
        make_test_provenance(),
        split_rows={
            "train": train_rows,
            "validation": 0,
            "evaluation": 0,
            "excluded": 0,
            "total": train_rows,
        },
        spark_version=spark.version,
    )


@pytest.fixture(scope="module")
def complete_training_frame(spark):
    return make_complete_training_frame(spark)


@pytest.fixture(scope="module")
def complete_fitted_preprocessor(spark, complete_training_frame):
    return fit_test_preprocessor(spark, complete_training_frame)


def test_fit_preprocessor_learns_median_only_from_train(spark):
    train = make_complete_training_frame(spark).withColumn(
        "trafico_intensidad_media",
        F.when(F.col("cod_distrito") == 1, F.lit(None))
        .when(F.col("cod_distrito") <= 17, F.lit(10.0))
        .otherwise(F.lit(20.0)),
    )
    validation = spark.createDataFrame(
        [make_row(index=600, trafico_intensidad_media=1_000_000.0)]
    )

    fitted = fit_test_preprocessor(spark, train)
    transformed = fitted.transform(validation, split_name="validation")

    assert fitted.manifest.imputation_medians["trafico_intensidad_media"] == 10.0
    assert transformed.count() == 1


def test_fit_preprocessor_rejects_missing_train_category(spark):
    train = make_complete_training_frame(spark).where(F.col("cod_distrito") != 21)

    with pytest.raises(PreprocessingContractError, match="cod_distrito"):
        fit_test_preprocessor(spark, train)


def test_fit_preprocessor_rejects_feature_without_valid_train_values(spark):
    train = make_complete_training_frame(spark).withColumn(
        "calair_so2_media", F.lit(None).cast("double")
    )

    with pytest.raises(PreprocessingContractError, match="calair_so2_media"):
        fit_test_preprocessor(spark, train)


def test_fit_preprocessor_rejects_zero_standard_deviation(spark):
    train = make_complete_training_frame(spark).withColumn(
        "meteo_temperatura_media", F.lit(7.0)
    )

    with pytest.raises(PreprocessingContractError, match="meteo_temperatura_media"):
        fit_test_preprocessor(spark, train)


def test_pipeline_model_round_trip_preserves_vector(
    spark, tmp_path, complete_training_frame, complete_fitted_preprocessor
):
    from madrid_ml.preprocessing import FittedPreprocessor

    path = str(tmp_path / "pipeline")
    complete_fitted_preprocessor.pipeline_model.write().overwrite().save(path)
    loaded = PipelineModel.load(path)
    sample = complete_training_frame.limit(5)

    expected = complete_fitted_preprocessor.transform(
        sample, split_name="train"
    ).select("features").collect()
    actual = FittedPreprocessor(
        loaded, complete_fitted_preprocessor.manifest
    ).transform(sample, split_name="train").select("features").collect()

    assert actual == expected


def test_preprocessing_vector_is_finite_and_has_111_components(
    complete_training_frame, complete_fitted_preprocessor
):
    transformed = complete_fitted_preprocessor.transform(
        complete_training_frame, split_name="train"
    )

    sizes = transformed.select(
        F.size(vector_to_array("features")).alias("size")
    ).distinct().collect()
    assert [row.size for row in sizes] == [111]
    assert transformed.count() == complete_training_frame.count()
    assert (
        transformed.select(
            "cod_distrito", "feature_hour", "target_accident_next_hour"
        )
        .exceptAll(
            complete_training_frame.select(
                "cod_distrito", "feature_hour", "target_accident_next_hour"
            )
        )
        .count()
        == 0
    )


def test_preprocessing_vector_remains_finite_after_all_sanitation_cases(
    spark, complete_fitted_preprocessor
):
    sample = spark.createDataFrame(
        [
            make_row(
                trafico_vmed_media=-1.0,
                trafico_ocupacion_media=float("nan"),
                meteo_humedad_relativa_media=101.0,
                meteo_radiacion_solar_media=float("inf"),
                calair_no2_media=-0.1,
            )
        ]
    )

    vector = complete_fitted_preprocessor.transform(
        sample, split_name="evaluation"
    ).select(vector_to_array("features").alias("features")).first().features

    assert len(vector) == 111
    assert all(
        value is not None and value == value and abs(value) != float("inf")
        for value in vector
    )


def test_manifest_preserves_exact_numeric_and_category_contract(
    complete_fitted_preprocessor,
):
    manifest = complete_fitted_preprocessor.manifest

    assert manifest.numeric_features == CONTINUOUS_FEATURES + COUNT_FEATURES
    assert manifest.available_features == tuple(
        f"{name}_available" for name in CONTINUOUS_FEATURES
    )
    assert manifest.category_labels == {
        "cod_distrito": tuple(f"{value:02d}" for value in range(1, 22)),
        "hora_dia": tuple(f"{value:02d}" for value in range(24)),
        "dia_semana": tuple(str(value) for value in range(1, 8)),
        "mes": tuple(f"{value:02d}" for value in range(1, 13)),
    }
    assert manifest.category_references == {
        "cod_distrito": "21",
        "hora_dia": "23",
        "dia_semana": "7",
        "mes": "12",
    }
    assert manifest.vector_size == 111


class DropAllRowsModel:
    def transform(self, df):
        return df.limit(0)


def test_transform_rejects_row_loss(
    complete_training_frame, complete_fitted_preprocessor
):
    from madrid_ml.preprocessing import FittedPreprocessor

    broken = FittedPreprocessor(
        DropAllRowsModel(), complete_fitted_preprocessor.manifest
    )

    with pytest.raises(PreprocessingContractError, match="row"):
        broken.transform(complete_training_frame, split_name="train")


def test_transform_rejects_unknown_split(
    complete_training_frame, complete_fitted_preprocessor
):
    with pytest.raises(PreprocessingContractError, match="split_name"):
        complete_fitted_preprocessor.transform(
            complete_training_frame, split_name="excluded"
        )


def test_artifact_manifest_contains_complete_identity(complete_fitted_preprocessor):
    from madrid_ml.preprocessing import manifest_to_dict

    payload = manifest_to_dict(
        complete_fitted_preprocessor.manifest,
        preprocessing_code_commit="b" * 40,
        wheel_sha256="c" * 64,
        package_version="0.2.0",
        runtime_environment_version="4",
        python_version="3.10.14",
    )

    assert payload["provenance"]["code_commit"] == "a" * 40
    assert payload["artifact_identity"] == {
        "preprocessing_code_commit": "b" * 40,
        "wheel_sha256": "c" * 64,
        "package_version": "0.2.0",
        "runtime_environment_version": "4",
        "python_version": "3.10.14",
    }
    assert payload["spark_version"] == complete_fitted_preprocessor.manifest.spark_version
    assert payload["session_timezone"] == "Etc/UTC"
    assert payload["split_periods"]["train"] == (
        "2019-01-01 00:00:00",
        "2024-01-01 00:00:00",
    )
    assert payload["split_rows"]["train"] == 504
    assert payload["physical_ranges"]["meteo_presion_media"]["minimum_inclusive"] is False
    assert len(payload["imputation_medians"]) == 18
    assert len(payload["scaler_mean"]) == 33
    assert len(payload["scaler_std"]) == 33


def test_public_api_exports_preprocessing_boundary():
    import madrid_ml
    from madrid_ml.preprocessing import (
        FittedPreprocessor,
        PreparedTrainingData,
        prepare_training_data,
    )

    assert madrid_ml.FittedPreprocessor is FittedPreprocessor
    assert madrid_ml.PreprocessingContractError is PreprocessingContractError
    assert madrid_ml.PreparedTrainingData is PreparedTrainingData
    assert madrid_ml.prepare_training_data is prepare_training_data
