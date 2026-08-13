from pathlib import Path

import pytest

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ingestion.core.exceptions import ConfigurationError, DatasetNotFoundError

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "conf"


def test_all_repository_configuration_is_valid():
    environment, sources = ConfigLoader(CONFIG_ROOT).validate_all("dev")
    assert environment.catalogs.bronze == "dev_bronze"
    assert len(sources) == 5
    assert sum(len(source.datasets) for source in sources) == 14
    assert all(
        dataset.bronze.transformations[0].type == "normalize_column_names"
        for source in sources
        for dataset in source.datasets
    )


def test_load_known_dataset():
    source, dataset = ConfigLoader(CONFIG_ROOT).load_dataset(
        "trafico", "trafico_nrt"
    )
    assert source.source == "trafico"
    assert dataset.bronze.format == "xml"
    assert dataset.bronze.reader_options["row_tag"] == "pms"
    assert dataset.bronze.reader_options["record_path"] == "pm"
    assert dataset.silver.write_strategy == "merge"
    assert dataset.silver.business_keys == ["idelem", "fecha_hora"]


def test_trafico_nrt_filters_coordinates_outside_madrid():
    _, dataset = ConfigLoader(CONFIG_ROOT).load_dataset("trafico", "trafico_nrt")
    filters = [
        transformation
        for transformation in dataset.silver.transformations
        if transformation.type == "filter"
    ]
    assert len(filters) == 1
    condition = filters[0].condition
    assert "st_x" in condition
    assert "st_y" in condition

    cast_step = next(
        transformation
        for transformation in dataset.silver.transformations
        if transformation.type == "cast"
    )
    filter_index = dataset.silver.transformations.index(filters[0])
    cast_index = dataset.silver.transformations.index(cast_step)
    assert cast_index < filter_index, (
        "el filtro de coordenadas debe ir después del cast a double"
    )


@pytest.mark.parametrize(
    ("dataset_name", "source_key"),
    [("trafico_nrt", "idelem"), ("trafico_historico", "id")],
)
def test_enriches_with_dim_trafico_after_cast(dataset_name, source_key):
    _, dataset = ConfigLoader(CONFIG_ROOT).load_dataset("trafico", dataset_name)
    joins = [
        transformation
        for transformation in dataset.silver.transformations
        if transformation.type == "lookup_join"
    ]
    assert len(joins) == 1
    join = joins[0]
    assert join.lookup_table.layer == "silver"
    assert join.lookup_table.source == "trafico"
    assert join.lookup_table.dataset == "dim_trafico"
    assert join.join_type == "left"
    assert join.conditions == {source_key: "id"}
    assert join.select == {
        "distrito": "distrito",
        "latitud": "latitud",
        "longitud": "longitud",
    }

    cast_step = next(
        transformation
        for transformation in dataset.silver.transformations
        if transformation.type == "cast"
    )
    join_index = dataset.silver.transformations.index(join)
    cast_index = dataset.silver.transformations.index(cast_step)
    assert cast_index < join_index, (
        "el lookup_join debe ir después del cast para que la clave "
        "de unión ya sea long, como id en dim_trafico"
    )


def test_missing_dataset_has_domain_error():
    with pytest.raises(DatasetNotFoundError, match="no_existe"):
        ConfigLoader(CONFIG_ROOT).load_dataset("trafico", "no_existe")


def test_invalid_yaml_is_wrapped(tmp_path):
    environments = tmp_path / "environments"
    environments.mkdir()
    (environments / "dev.yaml").write_text("environment: [", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="YAML inválido"):
        ConfigLoader(tmp_path).load_environment("dev")
