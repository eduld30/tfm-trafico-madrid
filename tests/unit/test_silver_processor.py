from datetime import date, datetime

import pytest

from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.silver.processor import _read_incremental_bronze


class FakeCondition:
    def __init__(self, column, operator, value):
        self.column = column
        self.operator = operator
        self.value = value


class FakeColumn:
    def __init__(self, name):
        self.name = name

    def __gt__(self, value):
        return FakeCondition(self.name, "gt", value)

    def __eq__(self, value):
        return FakeCondition(self.name, "eq", value)


class FakeAggregation:
    def __init__(self, row):
        self.row = row

    def first(self):
        return self.row


class FakeDataFrame:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, column):
        return FakeColumn(column)

    def selectExpr(self, expression):
        if "_ingestion_timestamp" in expression:
            values = [row["_ingestion_timestamp"] for row in self.rows]
            return FakeAggregation({"watermark": max(values, default=None)})
        values = [row["_file_date"] for row in self.rows]
        return FakeAggregation({"latest_file_date": max(values, default=None)})

    def filter(self, condition):
        if condition.operator == "gt":
            rows = [
                row for row in self.rows if row[condition.column] > condition.value
            ]
        else:
            rows = [
                row for row in self.rows if row[condition.column] == condition.value
            ]
        return FakeDataFrame(rows)

    def isEmpty(self):
        return not self.rows


class FakeCatalog:
    def __init__(self, exists):
        self.exists = exists

    def tableExists(self, table):
        return self.exists


class FakeSpark:
    def __init__(self, silver_rows=None):
        self.silver_rows = silver_rows
        self.catalog = FakeCatalog(silver_rows is not None)

    def table(self, table):
        return FakeDataFrame(self.silver_rows)


class FakeContext:
    table = "dev_silver.trafico.dataset"


def row(day, hour):
    return {
        "_file_date": date.fromisoformat(day),
        "_ingestion_timestamp": datetime.fromisoformat(
            f"2026-08-{day[-2:]}T{hour:02d}:00:00"
        ),
    }


def test_initial_merge_reads_all_bronze_rows():
    bronze = FakeDataFrame([row("2026-08-19", 10), row("2026-08-20", 10)])

    result = _read_incremental_bronze(FakeSpark(), bronze, FakeContext(), "merge")

    assert result.rows == bronze.rows


def test_existing_silver_filters_by_ingestion_watermark():
    silver_rows = [row("2026-08-19", 10)]
    bronze = FakeDataFrame(
        [row("2026-08-19", 10), row("2026-08-20", 11)]
    )

    result = _read_incremental_bronze(
        FakeSpark(silver_rows), bronze, FakeContext(), "merge"
    )

    assert result.rows == [bronze.rows[1]]


def test_overwrite_keeps_only_the_latest_new_file_date():
    bronze = FakeDataFrame([row("2026-08-19", 10), row("2026-08-20", 11)])

    result = _read_incremental_bronze(
        FakeSpark(), bronze, FakeContext(), "overwrite"
    )

    assert result.rows == [bronze.rows[1]]


def test_no_new_rows_skips_processing():
    silver_rows = [row("2026-08-20", 11)]
    bronze = FakeDataFrame([row("2026-08-20", 11)])

    result = _read_incremental_bronze(
        FakeSpark(silver_rows), bronze, FakeContext(), "append"
    )

    assert result is None


def test_overwrite_rejects_rows_without_file_date():
    bronze = FakeDataFrame(
        [{"_file_date": None, "_ingestion_timestamp": datetime(2026, 8, 20)}]
    )

    with pytest.raises(IngestionError, match="_file_date"):
        _read_incremental_bronze(FakeSpark(), bronze, FakeContext(), "overwrite")