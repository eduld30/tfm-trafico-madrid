import pytest

from madrid_ml.snapshot import (
    _cleanup_temporary_tables,
    _drop_owned_temporary_table,
    _persist_materialized,
    _temporary_storage_owner,
    _temporary_table_name,
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


def test_temporary_storage_identity_is_deterministic_and_role_scoped():
    owner = _temporary_storage_owner("run-a")

    assert len(owner) == 24
    assert int(owner, 16) >= 0
    assert owner == _temporary_storage_owner("run-a")
    assert owner != _temporary_storage_owner("run-b")
    assert (
        _temporary_table_name("dev_gold", owner, "grid")
        == f"dev_gold.ml.__tmp_snapshot_{owner}_grid"
    )

    with pytest.raises(SnapshotContractError, match="run_token"):
        _temporary_storage_owner(" ")


def test_persist_materialized_uses_owned_delta_table_without_extra_scan():
    class Materialized:
        def count(self):
            raise AssertionError("CTAS already materialized the DataFrame")

    class Source:
        def __init__(self):
            self.views = []

        def createOrReplaceTempView(self, name):
            self.views.append(name)

    class Catalog:
        def __init__(self):
            self.dropped_views = []

        def dropTempView(self, name):
            self.dropped_views.append(name)

    class Spark:
        def __init__(self):
            self.catalog = Catalog()
            self.statements = []
            self.materialized = Materialized()

        def sql(self, statement):
            self.statements.append(statement)

        def table(self, _name):
            return self.materialized

    spark = Spark()
    source = Source()
    temporary_tables = []
    owner = "a" * 24
    table_name = f"dev_gold.ml.__tmp_snapshot_{owner}_grid"

    materialized = _persist_materialized(
        spark,
        source,
        table_name,
        owner,
        temporary_tables,
    )

    assert materialized is spark.materialized
    assert temporary_tables == [table_name]
    assert len(source.views) == 1
    assert spark.catalog.dropped_views == source.views
    assert "CREATE TABLE" in spark.statements[0]
    assert "USING DELTA" in spark.statements[0]
    assert "'madrid_ml.snapshot_owner' = 'aaaaaaaaaaaaaaaaaaaaaaaa'" in spark.statements[0]


def test_drop_temporary_table_refuses_foreign_owner():
    class Detail:
        def first(self):
            return {"properties": {"madrid_ml.snapshot_owner": "foreign"}}

    class Catalog:
        @staticmethod
        def tableExists(_name):
            return True

    class Spark:
        catalog = Catalog()

        def __init__(self):
            self.statements = []

        def sql(self, statement):
            self.statements.append(statement)
            return Detail()

    spark = Spark()

    with pytest.raises(SnapshotContractError, match="owned by"):
        _drop_owned_temporary_table(
            spark,
            "dev_gold.ml.__tmp_snapshot_aaaaaaaaaaaaaaaaaaaaaaaa_grid",
            "a" * 24,
        )

    assert all(not statement.startswith("DROP TABLE") for statement in spark.statements)


def test_drop_temporary_table_fails_when_catalog_still_exposes_it():
    owner = "a" * 24
    table_name = f"dev_gold.ml.__tmp_snapshot_{owner}_grid"

    class Detail:
        @staticmethod
        def first():
            return {"properties": {"madrid_ml.snapshot_owner": owner}}

    class Catalog:
        @staticmethod
        def tableExists(_name):
            return True

    class Spark:
        catalog = Catalog()

        @staticmethod
        def sql(statement):
            if statement.startswith("DESCRIBE DETAIL"):
                return Detail()
            return None

    with pytest.raises(SnapshotContractError, match="left temporary table present"):
        _drop_owned_temporary_table(Spark(), table_name, owner)


def test_persist_materialized_tracks_failed_create_for_cleanup():
    class Source:
        @staticmethod
        def createOrReplaceTempView(_name):
            return None

    class Catalog:
        def __init__(self):
            self.dropped_views = []

        def dropTempView(self, name):
            self.dropped_views.append(name)

    class Spark:
        def __init__(self):
            self.catalog = Catalog()

        @staticmethod
        def sql(_statement):
            raise RuntimeError("ctas failed")

    spark = Spark()
    temporary_tables = []
    owner = "a" * 24
    table_name = f"dev_gold.ml.__tmp_snapshot_{owner}_grid"

    with pytest.raises(RuntimeError, match="ctas failed"):
        _persist_materialized(
            spark,
            Source(),
            table_name,
            owner,
            temporary_tables,
        )

    assert temporary_tables == [table_name]
    assert len(spark.catalog.dropped_views) == 1


def test_cleanup_temporary_tables_continues_after_one_drop_fails():
    owner = "a" * 24
    tables = [
        f"dev_gold.ml.__tmp_snapshot_{owner}_{role}"
        for role in ("grid", "labels", "features")
    ]

    class Detail:
        @staticmethod
        def first():
            return {"properties": {"madrid_ml.snapshot_owner": owner}}

    class Catalog:
        def __init__(self, spark):
            self.spark = spark

        def tableExists(self, name):
            return name not in self.spark.dropped_tables

    class Spark:
        def __init__(self):
            self.drop_attempts = []
            self.dropped_tables = set()
            self.catalog = Catalog(self)

        def sql(self, statement):
            if statement.startswith("DESCRIBE DETAIL"):
                return Detail()
            self.drop_attempts.append(statement)
            table_name = next(
                table
                for table in tables
                if table.replace(".", "`.`") in statement
            )
            if table_name == tables[-1]:
                raise RuntimeError("drop failed")
            self.dropped_tables.add(table_name)
            return None

    spark = Spark()

    with pytest.raises(SnapshotContractError, match=tables[-1]):
        _cleanup_temporary_tables(spark, tables, owner)

    assert spark.drop_attempts == [
        f"DROP TABLE `{table.replace('.', '`.`')}`" for table in reversed(tables)
    ]
    assert spark.dropped_tables == set(tables[:-1])
