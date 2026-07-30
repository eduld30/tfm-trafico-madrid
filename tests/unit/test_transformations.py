import pytest

from madrid_ingestion.config.models import TransformationConfig
from madrid_ingestion.core.columns import (
    normalize_column_name,
    normalize_dataframe_columns,
    rename_dataframe_columns,
    validate_technical_column_names,
)
from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.silver.transformations import (
    TRANSFORMATIONS,
    apply_transformations,
)


class FakeDataFrame:
    def __init__(self, columns):
        self.columns = columns

    def toDF(self, *columns):
        return FakeDataFrame(list(columns))


def test_registry_contains_the_declared_initial_transformations():
    assert set(TRANSFORMATIONS) == {
        "normalize_column_names",
        "rename",
        "select",
        "drop",
        "cast",
        "trim",
        "empty_to_null",
        "replace_values",
        "filter",
        "add_literal",
        "parse_timestamp",
        "deduplicate",
        "lookup_join",
    }


def test_normalize_column_names_does_not_need_spark_actions():
    context = object()
    result = apply_transformations(
        FakeDataFrame(["Tráfico NRT", "ÁREA"]),
        [TransformationConfig(type="normalize_column_names")],
        context,
    )
    assert result.columns == ["trafico_nrt", "area"]


def test_normalization_preserves_technical_prefix():
    assert normalize_column_name("_rescued_data") == "_rescued_data"
    assert normalize_column_name("_Ingestion Timestamp") == "_ingestion_timestamp"


def test_rename_is_simultaneous_and_preserves_values_contract():
    original = FakeDataFrame(["a", "b"])
    result = rename_dataframe_columns(original, {"a": "b", "b": "a"})
    assert result.columns == ["b", "a"]


def test_normalization_detects_collisions():
    with pytest.raises(IngestionError, match="duplicadas"):
        normalize_dataframe_columns(FakeDataFrame(["Área", "area"]))


def test_validation_rejects_remaining_unsafe_names():
    with pytest.raises(IngestionError, match="sin normalizar"):
        validate_technical_column_names(FakeDataFrame(["Nombre Distrito"]), "Bronze")
