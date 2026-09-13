import math
from datetime import datetime, timezone

import pytest
from pyspark.sql import functions as F

from madrid_ml.contract import WEATHER_MAGNITUDES
from madrid_ml.nrt import (
    NrtScoringConfig,
    NrtScoringError,
    _latest_magnitude_features,
    _traffic_features,
    floor_to_ten_minutes,
    validate_nrt_feature_snapshot,
)


def test_floor_to_ten_minutes_uses_madrid_wall_clock():
    observed = floor_to_ten_minutes(
        datetime(2024, 1, 2, 9, 27, 45, tzinfo=timezone.utc)
    )

    assert observed == datetime(2024, 1, 2, 10, 20)
    assert observed.tzinfo is None


def test_traffic_features_use_exact_rolling_hour(spark):
    traffic = spark.createDataFrame(
        [
            ("2024-01-02 08:59:59", 1, 99.0, 99.0, 99.0, 99.0),
            ("2024-01-02 09:05:00", 1, 10.0, 1.0, 2.0, 3.0),
            ("2024-01-02 09:55:00", 1, 30.0, 3.0, 4.0, 5.0),
            ("2024-01-02 10:00:00", 1, 99.0, 99.0, 99.0, 99.0),
        ],
        (
            "fecha_hora string, distrito_cod int, intensidad double, "
            "ocupacion double, carga double, velocidad double"
        ),
    ).withColumn("fecha_hora", F.to_timestamp("fecha_hora"))

    row = _traffic_features(
        traffic,
        window_start=datetime(2024, 1, 2, 9),
        cutoff=datetime(2024, 1, 2, 10),
    ).select(
        "*",
        F.date_format(
            "traffic_data_max_timestamp", "yyyy-MM-dd HH:mm:ss"
        ).alias("traffic_data_max_timestamp_text"),
    ).first()

    assert row.trafico_intensidad_media == 20.0
    assert row.trafico_ocupacion_media == 2.0
    assert row.trafico_carga_media == 3.0
    assert row.trafico_vmed_media == 4.0
    assert row.trafico_puntos_n == 2
    assert math.isfinite(row.trafico_intensidad_media)
    assert row.traffic_data_max_timestamp_text == "2024-01-02 09:55:00"


def test_latest_magnitude_value_is_selected_per_station(spark):
    temperature = WEATHER_MAGNITUDES[1]
    frame = spark.createDataFrame(
        [
            ("2024-01-02 08:00:00", 1, 100, temperature.code, 10.0),
            ("2024-01-02 09:00:00", 1, 100, temperature.code, 20.0),
            ("2024-01-02 11:00:00", 1, 100, temperature.code, 99.0),
            ("2024-01-02 09:00:00", 1, 101, temperature.code, 30.0),
        ],
        (
            "fecha_hora string, distrito_cod int, estacion int, "
            "magnitud int, valor double"
        ),
    ).withColumn("fecha_hora", F.to_timestamp("fecha_hora"))

    row = _latest_magnitude_features(
        frame,
        WEATHER_MAGNITUDES,
        cutoff=datetime(2024, 1, 2, 10),
        logical_table="meteo.meteo_nrt",
        timestamp_output="weather_data_max_timestamp",
    ).select(
        "*",
        F.date_format(
            "weather_data_max_timestamp", "yyyy-MM-dd HH:mm:ss"
        ).alias("weather_data_max_timestamp_text"),
    ).first()

    assert getattr(row, temperature.mean_column) == 25.0
    assert getattr(row, temperature.count_column) == 2
    assert row.weather_data_max_timestamp_text == "2024-01-02 09:00:00"


def test_nrt_snapshot_validation_requires_all_districts(spark):
    valid = spark.createDataFrame(
        [
            (district, "2024-01-02 09:00:00", "2024-01-02 10:00:00")
            for district in range(1, 22)
        ],
        "cod_distrito int, feature_hour string, prediction_hour string",
    ).selectExpr(
        "cod_distrito",
        "to_timestamp(feature_hour) feature_hour",
        "to_timestamp(prediction_hour) prediction_hour",
    )

    assert validate_nrt_feature_snapshot(valid) == 21
    with pytest.raises(NrtScoringError, match="one valid row per district"):
        validate_nrt_feature_snapshot(valid.where("cod_distrito < 21"))


def test_nrt_config_requires_three_level_registered_model_name():
    with pytest.raises(ValueError, match="catalog.schema.model"):
        NrtScoringConfig(
            silver_catalog="dev_silver",
            gold_catalog="dev_gold",
            registered_model_name="accident_risk_model",
            mlflow_dfs_tmp="/Volumes/dev_gold/ml/mlflow_tmp/model-scoring",
        )
