from datetime import datetime, timedelta

import pytest

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
