import pytest

from madrid_ingestion.core.exceptions import WriteStrategyError
from madrid_ingestion.writers import delta_writer


class FakeCatalog:
    def __init__(self, exists):
        self.exists = exists

    def tableExists(self, name):
        return self.exists


class FakeSpark:
    def __init__(self, exists=False):
        self.catalog = FakeCatalog(exists)
        self.statements = []
        self.expected_location = "abfss://silver@account/path/"

    def sql(self, statement):
        self.statements.append(statement)
        return FakeDetail(self.expected_location)


class FakeDetail:
    def __init__(self, location):
        self.location = location

    def select(self, column):
        assert column == "location"
        return self

    def first(self):
        return {"location": self.location}


class FakeWriter:
    def __init__(self):
        self.calls = []

    def format(self, value):
        self.calls.append(("format", value))
        return self

    def mode(self, value):
        self.calls.append(("mode", value))
        return self

    def option(self, key, value):
        self.calls.append(("option", key, value))
        return self

    def partitionBy(self, *columns):
        self.calls.append(("partitionBy", columns))
        return self

    def save(self, path):
        self.calls.append(("save", path))


class FakeDataFrame:
    def __init__(self, columns, partition_rows=None):
        self.columns = columns
        self.write = FakeWriter()
        self.partition_rows = partition_rows or []

    def select(self, *columns):
        self.selected_columns = columns
        return self

    def distinct(self):
        return self

    def collect(self):
        return self.partition_rows


def test_overwrite_selects_batch_writer_and_registers_table():
    spark = FakeSpark()
    df = FakeDataFrame(["id", "fecha"])
    delta_writer.write_silver_delta(
        spark,
        df,
        "dev_silver.trafico.dataset",
        "abfss://silver@account/path/",
        "overwrite",
        partition_by=["fecha"],
        overwrite_schema=True,
    )
    assert ("mode", "overwrite") in df.write.calls
    assert ("partitionBy", ("fecha",)) in df.write.calls
    assert ("option", "overwriteSchema", "true") in df.write.calls
    assert len(spark.statements) == 2


def test_merge_creates_initial_delta_when_table_does_not_exist(monkeypatch):
    spark = FakeSpark(exists=False)
    df = FakeDataFrame(["id"])
    merge_calls = []
    monkeypatch.setattr(
        delta_writer,
        "merge_delta",
        lambda *args, **kwargs: merge_calls.append((args, kwargs)),
    )
    delta_writer.write_silver_delta(
        spark,
        df,
        "dev_silver.trafico.dataset",
        "abfss://silver@account/path/",
        "merge",
        business_keys=["id"],
    )
    assert ("mode", "overwrite") in df.write.calls
    assert merge_calls == []


def test_merge_uses_delta_api_for_existing_table(monkeypatch):
    spark = FakeSpark(exists=True)
    df = FakeDataFrame(["id"])
    calls = []
    monkeypatch.setattr(
        delta_writer,
        "merge_delta",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    delta_writer.write_silver_delta(
        spark,
        df,
        "dev_silver.trafico.dataset",
        "abfss://silver@account/path/",
        "merge",
        business_keys=["id"],
    )
    assert len(calls) == 1
    assert spark.statements == [
        "DESCRIBE DETAIL `dev_silver`.`trafico`.`dataset`"
    ]


def test_replace_partitions_limits_overwrite_to_received_partitions():
    spark = FakeSpark(exists=True)
    df = FakeDataFrame(
        ["id", "anio"],
        partition_rows=[{"anio": 2025}, {"anio": 2026}],
    )

    delta_writer.write_silver_delta(
        spark,
        df,
        "dev_silver.trafico.dataset",
        "abfss://silver@account/path/",
        "replace_partitions",
        partition_by=["anio"],
    )

    assert ("mode", "overwrite") in df.write.calls
    assert (
        "option",
        "replaceWhere",
        "(`anio` = 2025) OR (`anio` = 2026)",
    ) in df.write.calls
    assert ("partitionBy", ("anio",)) in df.write.calls


def test_unknown_strategy_fails():
    with pytest.raises(WriteStrategyError, match="desconocida"):
        delta_writer.write_silver_delta(
            FakeSpark(),
            FakeDataFrame(["id"]),
            "dev_silver.trafico.dataset",
            "abfss://silver@account/path/",
            "unknown",
        )


def test_partition_column_must_exist():
    with pytest.raises(WriteStrategyError, match="particionado"):
        delta_writer.write_initial_delta(
            FakeDataFrame(["id"]),
            "abfss://silver@account/path/",
            "overwrite",
            ["fecha"],
        )
