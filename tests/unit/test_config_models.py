import pytest
from pydantic import ValidationError

from madrid_ingestion.config.models import (
    BronzeConfig,
    BronzeTransformationConfig,
    DatasetConfig,
    SilverConfig,
    SourceConfig,
    TransformationConfig,
)


def bronze(name="dataset"):
    return BronzeConfig(
        format="csv",
        source_path=f"source/{name}",
        target_path=f"source/{name}",
    )


def silver(name="dataset", **overrides):
    values = {
        "target_path": f"source/{name}",
        "write_strategy": "overwrite",
    }
    values.update(overrides)
    return SilverConfig(**values)


def test_merge_requires_business_keys():
    with pytest.raises(ValidationError, match="business_keys"):
        silver(write_strategy="merge")


def test_append_uses_the_common_incremental_processing():
    config = silver(write_strategy="append")
    assert config.write_strategy == "append"


def test_unknown_transformation_fails_early():
    with pytest.raises(ValidationError, match="no soportada"):
        TransformationConfig(type="python_eval")


def test_parse_timestamp_accepts_multiple_source_columns():
    config = TransformationConfig(
        type="parse_timestamp",
        source_columns=["fecha", "hora"],
        target_column="fecha_hora",
        format="dd/MM/yyyy HH:mm:ss",
    )
    assert config.source_columns == ["fecha", "hora"]


def test_regex_replace_requires_a_list_of_columns():
    with pytest.raises(ValidationError, match="columns debe ser una lista"):
        TransformationConfig(
            type="regex_replace",
            columns={"coordenada": "double"},
            pattern=",",
            replacement=".",
        )


def test_hourly_wide_to_long_validates_hour_range():
    with pytest.raises(ValidationError, match="entre 1 y 24"):
        TransformationConfig(
            type="hourly_wide_to_long",
            year_column="ano",
            month_column="mes",
            day_column="dia",
            value_prefix="h",
            validity_prefix="v",
            value_column="valor",
            validity_column="validez",
            timestamp_column="fecha_hora",
            hours=25,
        )


def test_source_rejects_duplicate_datasets():
    dataset = DatasetConfig(
        name="dataset",
        description="test",
        bronze=bronze(),
        silver=silver(),
    )
    with pytest.raises(ValidationError, match="duplicados"):
        SourceConfig(source="source", datasets=[dataset, dataset])


def test_source_enforces_conventional_paths():
    dataset = DatasetConfig(
        name="dataset",
        description="test",
        bronze=BronzeConfig(
            format="csv",
            source_path="source/other",
            target_path="source/dataset",
        ),
        silver=silver(),
    )
    with pytest.raises(ValidationError, match="source_path"):
        SourceConfig(source="source", datasets=[dataset])


def test_landing_path_must_be_stable_root():
    with pytest.raises(ValidationError, match="raíz estable"):
        BronzeConfig(
            format="csv",
            source_path="source/dataset/2026/07/26/17",
            target_path="source/dataset",
        )


def test_bronze_allows_only_technical_column_transformations():
    config = BronzeConfig(
        format="csv",
        source_path="source/dataset",
        target_path="source/dataset",
        transformations=[
            {"type": "normalize_column_names"},
            {"type": "rename", "columns": {"old_name": "new_name"}},
        ],
    )
    assert [item.type for item in config.transformations] == [
        "normalize_column_names",
        "rename",
    ]


def test_bronze_rejects_value_transformations():
    with pytest.raises(ValidationError, match="Input should be"):
        BronzeConfig(
            format="csv",
            source_path="source/dataset",
            target_path="source/dataset",
            transformations=[{"type": "cast", "columns": {"id": "long"}}],
        )


def test_bronze_rename_requires_columns():
    with pytest.raises(ValidationError, match="mapa columns"):
        BronzeTransformationConfig(type="rename")
