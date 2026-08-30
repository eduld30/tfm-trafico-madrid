import pytest

from madrid_ml.snapshot import (
    canonical_versions_json,
    count_grid_key_differences,
    validate_expected_versions,
    validate_matching_lineage,
    validate_unique_key,
)
from madrid_ml.transformations import SnapshotContractError


def test_input_versions_json_is_canonical():
    actual = canonical_versions_json({"z.table": 4, "a.table": 1})

    assert actual == '{"a.table":1,"z.table":4}'


def test_output_validation_rejects_duplicate_grid_key(spark):
    duplicated = spark.createDataFrame(
        [(1, "2024-01-01 10:00:00"), (1, "2024-01-01 10:00:00")],
        "cod_distrito int, feature_hour string",
    ).selectExpr("cod_distrito", "to_timestamp(feature_hour) feature_hour")

    with pytest.raises(SnapshotContractError, match="clave duplicada"):
        validate_unique_key(duplicated, ("cod_distrito", "feature_hour"))


def test_grid_key_validation_detects_one_missing_and_one_extra_key(spark):
    grid = spark.createDataFrame(
        [(1, "2024-01-01 10:00:00"), (2, "2024-01-01 10:00:00")],
        "cod_distrito int, feature_hour string",
    ).selectExpr("cod_distrito", "to_timestamp(feature_hour) feature_hour")
    output = spark.createDataFrame(
        [(1, "2024-01-01 10:00:00"), (3, "2024-01-01 10:00:00")],
        "cod_distrito int, feature_hour string",
    ).selectExpr("cod_distrito", "to_timestamp(feature_hour) feature_hour")

    missing, extra = count_grid_key_differences(grid, output)

    assert missing == 1
    assert extra == 1


def test_lineage_comparison_uses_all_five_fields(spark):
    lineage_schema = (
        "snapshot_id string, input_versions_json string, code_commit string, "
        "feature_schema_version string, time_contract string"
    )
    labels = spark.createDataFrame(
        [("snapshot-1", '{"a":1}', "commit-a", "1", "source_wall_clock_as_stored_in_silver")],
        lineage_schema,
    )
    features = spark.createDataFrame(
        [("snapshot-1", '{"a":1}', "commit-b", "1", "source_wall_clock_as_stored_in_silver")],
        lineage_schema,
    )

    with pytest.raises(SnapshotContractError, match="code_commit"):
        validate_matching_lineage(labels, features)


def test_expected_versions_rejects_drift():
    with pytest.raises(SnapshotContractError, match="version"):
        validate_expected_versions(
            {"dev_silver.accidentes.accidentes_historico": 2},
            {"dev_silver.accidentes.accidentes_historico": 1},
        )
