import os
import time

import pytest
from pyspark.sql import SparkSession

from madrid_ml.transformations import SnapshotContractError, build_accident_labels


@pytest.fixture(scope="module")
def spark():
    previous_timezone = os.environ.get("TZ")
    os.environ["TZ"] = "Etc/UTC"
    time.tzset()
    session = None
    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("madrid-ml-unit-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "Etc/UTC")
            .getOrCreate()
        )
        yield session
    finally:
        if session is not None:
            session.stop()
        if previous_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_timezone
        time.tzset()


def test_two_people_in_same_accident_count_once_for_next_hour(spark):
    accidents = spark.createDataFrame(
        [
            ("E1", "2024-01-02 11:05:00", 1),
            ("E1", "2024-01-02 11:10:00", 1),
        ],
        "num_expediente string, fecha_hora string, cod_distrito int",
    ).selectExpr("num_expediente", "to_timestamp(fecha_hora) fecha_hora", "cod_distrito")
    grid = spark.createDataFrame(
        [(1, "2024-01-02 10:00:00")],
        "cod_distrito int, feature_hour string",
    ).selectExpr("cod_distrito", "to_timestamp(feature_hour) feature_hour")

    row = build_accident_labels(accidents, grid).first()

    assert row.n_accidentes_next_hour == 1
    assert row.target_accident_next_hour == 1
    assert str(row.prediction_hour) == "2024-01-02 11:00:00"


def test_any_null_required_field_excludes_the_whole_accident(spark):
    accidents = spark.createDataFrame(
        [
            ("E1", "2024-01-02 11:05:00", None),
            ("E1", "2024-01-02 11:05:00", 1),
        ],
        "num_expediente string, fecha_hora string, cod_distrito int",
    ).selectExpr("num_expediente", "to_timestamp(fecha_hora) fecha_hora", "cod_distrito")
    grid = spark.createDataFrame(
        [(1, "2024-01-02 10:00:00")],
        "cod_distrito int, feature_hour string",
    ).selectExpr("cod_distrito", "to_timestamp(feature_hour) feature_hour")

    row = build_accident_labels(accidents, grid).first()

    assert row.n_accidentes_next_hour == 0
    assert row.target_accident_next_hour == 0


def test_unknown_accident_district_fails(spark):
    accidents = spark.createDataFrame(
        [("E1", "2024-01-02 11:05:00", 2)],
        "num_expediente string, fecha_hora string, cod_distrito int",
    ).selectExpr("num_expediente", "to_timestamp(fecha_hora) fecha_hora", "cod_distrito")
    grid = spark.createDataFrame(
        [(1, "2024-01-02 10:00:00")],
        "cod_distrito int, feature_hour string",
    ).selectExpr("cod_distrito", "to_timestamp(feature_hour) feature_hour")

    with pytest.raises(SnapshotContractError, match=r"2"):
        build_accident_labels(accidents, grid)
