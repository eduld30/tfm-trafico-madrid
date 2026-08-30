import pytest

from madrid_ml.contract import AIR_MAGNITUDES, FEATURE_BASE_COLUMNS, WEATHER_MAGNITUDES
from madrid_ml.transformations import (
    SnapshotContractError,
    aggregate_magnitudes,
    aggregate_traffic,
    build_accident_labels,
    build_feature_snapshot,
)


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


def _labels_for_hours(spark, hours):
    return spark.createDataFrame(
        [
            (1, feature_hour, prediction_hour, 0, 0)
            for feature_hour, prediction_hour in hours
        ],
        (
            "cod_distrito int, feature_hour string, prediction_hour string, "
            "n_accidentes_next_hour long, target_accident_next_hour int"
        ),
    ).selectExpr(
        "cod_distrito",
        "to_timestamp(feature_hour) feature_hour",
        "to_timestamp(prediction_hour) prediction_hour",
        "n_accidentes_next_hour",
        "target_accident_next_hour",
    )


def _empty_magnitude_aggregate(spark, specs):
    schema = "cod_distrito int, feature_hour timestamp, " + ", ".join(
        column
        for spec in specs
        for column in (
            f"{spec.mean_column} double",
            f"{spec.count_column} long",
        )
    )
    return spark.createDataFrame([], schema)


def test_traffic_hour_bucket_and_missing_group(spark):
    traffic = spark.createDataFrame(
        [
            ("2024-01-02 10:05:00", 1, 10.0, 1.0, 2.0, 3.0),
            ("2024-01-02 10:55:00", 1, 30.0, 3.0, 4.0, 5.0),
            ("2024-01-02 11:00:00", 1, 100.0, 100.0, 100.0, 100.0),
        ],
        (
            "fecha_hora string, distrito_cod int, intensidad double, ocupacion double, "
            "carga double, vmed double"
        ),
    ).selectExpr(
        "to_timestamp(fecha_hora) fecha_hora",
        "distrito_cod",
        "intensidad",
        "ocupacion",
        "carga",
        "vmed",
    )
    labels = _labels_for_hours(
        spark,
        [
            ("2024-01-02 10:00:00", "2024-01-02 11:00:00"),
            ("2024-01-02 12:00:00", "2024-01-02 13:00:00"),
        ],
    )

    result = build_feature_snapshot(
        labels,
        aggregate_traffic(traffic),
        _empty_magnitude_aggregate(spark, WEATHER_MAGNITUDES),
        _empty_magnitude_aggregate(spark, AIR_MAGNITUDES),
    ).orderBy("feature_hour")
    observed, missing = result.collect()

    assert result.columns == [name for name, _, _ in FEATURE_BASE_COLUMNS]
    assert observed.trafico_intensidad_media == 20.0
    assert observed.trafico_ocupacion_media == 2.0
    assert observed.trafico_carga_media == 3.0
    assert observed.trafico_vmed_media == 4.0
    assert observed.trafico_puntos_n == 2
    assert missing.trafico_intensidad_media is None
    assert missing.trafico_ocupacion_media is None
    assert missing.trafico_carga_media is None
    assert missing.trafico_vmed_media is None
    assert missing.trafico_puntos_n == 0
    for spec in WEATHER_MAGNITUDES + AIR_MAGNITUDES:
        assert getattr(missing, spec.mean_column) is None
        assert getattr(missing, spec.count_column) == 0


def test_traffic_final_bucket_is_inclusive(spark):
    traffic = spark.createDataFrame(
        [
            ("2026-06-30 22:59:59", 1, 1.0, 1.0, 1.0, 1.0),
            ("2026-06-30 23:00:00", 1, 99.0, 99.0, 99.0, 99.0),
        ],
        (
            "fecha_hora string, distrito_cod int, intensidad double, ocupacion double, "
            "carga double, vmed double"
        ),
    ).selectExpr(
        "to_timestamp(fecha_hora) fecha_hora",
        "distrito_cod",
        "intensidad",
        "ocupacion",
        "carga",
        "vmed",
    )

    rows = aggregate_traffic(traffic).collect()

    assert len(rows) == 1
    assert str(rows[0].feature_hour) == "2026-06-30 22:00:00"
    assert rows[0].trafico_intensidad_media == 1.0
    assert rows[0].trafico_puntos_n == 1


@pytest.mark.parametrize("specs", [WEATHER_MAGNITUDES, AIR_MAGNITUDES])
def test_magnitude_aggregate_counts_values_and_ignores_unselected_codes(spark, specs):
    selected_code = specs[0].code
    measurements = [
        ("2024-01-02 10:05:00", 1, selected_code, 10.0),
        ("2024-01-02 10:15:00", 1, selected_code, 20.0),
        ("2024-01-02 10:25:00", 1, selected_code, None),
        *[
            ("2024-01-02 10:35:00", 1, spec.code, 1.0)
            for spec in specs[1:]
        ],
        ("2024-01-02 10:45:00", 1, 999, 999.0),
    ]
    values = spark.createDataFrame(
        measurements,
        "fecha_hora string, distrito_cod int, magnitud int, valor double",
    ).selectExpr(
        "to_timestamp(fecha_hora) fecha_hora",
        "distrito_cod",
        "magnitud",
        "valor",
    )

    result = aggregate_magnitudes(values, specs)
    row = result.first()

    expected_columns = [
        "cod_distrito",
        "feature_hour",
        *[
            column
            for spec in specs
            for column in (spec.mean_column, spec.count_column)
        ],
    ]
    assert result.columns == expected_columns
    assert getattr(row, specs[0].mean_column) == 15.0
    assert getattr(row, specs[0].count_column) == 2
