import pytest

from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.writers.table_manager import TableManager


class FakeCatalog:
    def tableExists(self, name):
        return name == "dev_bronze.trafico.existe"


class FakeSpark:
    def __init__(self):
        self.catalog = FakeCatalog()
        self.statements = []
        self.detail_location = None

    def sql(self, statement):
        self.statements.append(statement)
        return FakeDetail(self.detail_location)


class FakeDetail:
    def __init__(self, location):
        self.location = location

    def select(self, column):
        assert column == "location"
        return self

    def first(self):
        return None if self.location is None else {"location": self.location}


def test_ensure_external_table_creates_schema_and_table():
    spark = FakeSpark()
    manager = TableManager(spark)
    manager.ensure_external_table(
        "dev_bronze.trafico.trafico_nrt",
        "abfss://bronze@account/path/",
    )
    assert len(spark.statements) == 2
    assert "CREATE SCHEMA IF NOT EXISTS `dev_bronze`.`trafico`" in spark.statements[0]
    assert "USING DELTA LOCATION" in spark.statements[1]
    assert manager.table_exists("dev_bronze.trafico.existe")


def test_existing_table_location_is_validated():
    spark = FakeSpark()
    spark.detail_location = "abfss://bronze@account/correct/"
    manager = TableManager(spark)
    manager.ensure_external_table(
        "dev_bronze.trafico.existe",
        "abfss://bronze@account/correct/",
    )
    assert spark.statements == ["DESCRIBE DETAIL `dev_bronze`.`trafico`.`existe`"]


def test_existing_table_with_wrong_location_fails():
    spark = FakeSpark()
    spark.detail_location = "abfss://bronze@account/other/"
    with pytest.raises(IngestionError, match="está registrada"):
        TableManager(spark).ensure_external_table(
            "dev_bronze.trafico.existe",
            "abfss://bronze@account/expected/",
        )
