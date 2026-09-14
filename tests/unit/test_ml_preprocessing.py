from datetime import datetime, timedelta

import pytest
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


def test_sanitize_enforces_physical_ranges_and_availability(spark):
    frame = spark.createDataFrame(
        [
            make_row(
                trafico_vmed_media=-1.0,
                trafico_ocupacion_media=float("nan"),
                meteo_humedad_relativa_media=100.1,
                meteo_presion_media=0.0,
                meteo_radiacion_solar_media=float("inf"),
                calair_no2_media=-0.1,
            ),
            make_row(
                index=1,
                trafico_intensidad_media=0.0,
                meteo_temperatura_media=-55.0,
                meteo_humedad_relativa_media=100.0,
                meteo_presion_media=1.0,
            ),
        ]
    )

    invalid, valid = sanitize_features(frame).orderBy("cod_distrito").collect()

    for column_name in (
        "trafico_vmed_media",
        "trafico_ocupacion_media",
        "meteo_humedad_relativa_media",
        "meteo_presion_media",
        "meteo_radiacion_solar_media",
        "calair_no2_media",
    ):
        assert invalid[column_name] is None
        assert invalid[f"{column_name}_available"] == 0.0

    for column_name, expected in (
        ("trafico_intensidad_media", 0.0),
        ("meteo_temperatura_media", -55.0),
        ("meteo_humedad_relativa_media", 100.0),
        ("meteo_presion_media", 1.0),
    ):
        assert valid[column_name] == expected
        assert valid[f"{column_name}_available"] == 1.0


def test_physical_ranges_cover_every_continuous_feature():
    assert tuple(PHYSICAL_RANGES) == CONTINUOUS_FEATURES


def test_validate_domains_reports_categories_counts_and_inconsistent_means(spark):
    frame = spark.createDataFrame(
        [
            make_row(
                index=0,
                cod_distrito=22,
                target_accident_next_hour=2,
                trafico_puntos_n=-1,
            ),
            make_row(
                index=1,
                calair_so2_n=None,
                meteo_temperatura_n=0,
                meteo_temperatura_media=12.0,
            ),
            make_row(
                index=2,
                trafico_puntos_n=0,
                trafico_intensidad_media=12.0,
            ),
        ]
    )

    with pytest.raises(PreprocessingContractError) as exc_info:
        validate_feature_domains(frame)

    message = str(exc_info.value)
    for column_name in (
        "cod_distrito",
        "target_accident_next_hour",
        "trafico_puntos_n",
        "calair_so2_n",
        "meteo_temperatura_media",
        "trafico_intensidad_media",
    ):
        assert column_name in message


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
            make_row(
                feature_hour="2023-12-31 23:00:00",
                prediction_hour="2024-01-01 00:00:00",
                hora_dia=23,
                mes=12,
            ),
            make_row(
                feature_hour="2024-01-01 00:00:00",
                prediction_hour="2024-01-01 01:00:00",
                hora_dia=0,
                mes=1,
            ),
            make_row(
                feature_hour="2025-01-01 00:00:00",
                prediction_hour="2025-01-01 01:00:00",
                hora_dia=0,
                mes=1,
            ),
            make_row(
                feature_hour="2026-01-01 00:00:00",
                prediction_hour="2026-01-01 01:00:00",
                hora_dia=0,
                mes=1,
            ),
        ]
    ).withColumn(
        "feature_hour", F.to_timestamp("feature_hour")
    ).withColumn(
        "prediction_hour", F.to_timestamp("prediction_hour")
    )
    from madrid_ml.preprocessing import _split_by_feature_hour

    splits = _split_by_feature_hour(frame)

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


def test_validate_snapshot_schema_rejects_incompatible_contract(spark):
    from madrid_ml.preprocessing import _validate_snapshot_schema

    fields = list(contract_schema(FEATURE_TABLE_COLUMNS).fields)
    fields[0] = StructField(fields[0].name, StringType(), fields[0].nullable)
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


def test_load_gold_rejects_invalid_inputs_before_catalog_access():
    from madrid_ml.preprocessing import load_gold_training_snapshot

    invalid_inputs = (
        ("bad-catalog", 1, 1, "snapshot-test", "gold_catalog"),
        ("dev_gold", -1, 1, "snapshot-test", "labels_delta_version"),
        ("dev_gold", 1, -1, "snapshot-test", "features_delta_version"),
        ("dev_gold", 1, 1, "", "expected_snapshot_id"),
    )
    for gold_catalog, labels_version, features_version, snapshot_id, error in invalid_inputs:
        with pytest.raises(PreprocessingContractError, match=error):
            load_gold_training_snapshot(
                None,
                gold_catalog=gold_catalog,
                labels_delta_version=labels_version,
                features_delta_version=features_version,
                expected_snapshot_id=snapshot_id,
            )


TEST_TRAIN_ROWS = 24


def make_complete_training_frame(spark):
    # 24 rows cover every value of the largest categorical domain (hour); the
    # other domains are smaller and are covered by the same modulo-based rows.
    return spark.createDataFrame(
        [make_row(index=index) for index in range(TEST_TRAIN_ROWS)]
    )


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

    return fit_preprocessor(
        train,
        make_test_provenance(),
        split_rows={
            "train": TEST_TRAIN_ROWS,
            "validation": 0,
            "evaluation": 0,
            "excluded": 0,
            "total": TEST_TRAIN_ROWS,
        },
        spark_version=spark.version,
    )


@pytest.fixture(scope="module")
def complete_training_frame(spark):
    return make_complete_training_frame(spark)


@pytest.fixture(scope="module")
def complete_fitted_preprocessor(spark, complete_training_frame):
    return fit_test_preprocessor(spark, complete_training_frame)


def test_fit_preprocessor_rejects_feature_without_valid_train_values(spark):
    train = make_complete_training_frame(spark).withColumn(
        "calair_so2_media", F.lit(None).cast("double")
    )

    with pytest.raises(PreprocessingContractError, match="calair_so2_media"):
        fit_test_preprocessor(spark, train)


def test_preprocessing_produces_a_finite_vector_after_sanitation(
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


def test_transform_rejects_unknown_split(complete_fitted_preprocessor):
    with pytest.raises(PreprocessingContractError, match="split_name"):
        complete_fitted_preprocessor.transform(None, split_name="excluded")


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
    assert payload["split_rows"]["train"] == TEST_TRAIN_ROWS
    assert payload["physical_ranges"]["meteo_presion_media"]["minimum_inclusive"] is False
    assert len(payload["imputation_medians"]) == 18
    assert len(payload["scaler_mean"]) == 33
    assert len(payload["scaler_std"]) == 33
