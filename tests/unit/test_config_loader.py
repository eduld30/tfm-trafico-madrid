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
    assert dataset.bronze.reader_options["row_tag"] == "pm"


def test_missing_dataset_has_domain_error():
    with pytest.raises(DatasetNotFoundError, match="no_existe"):
        ConfigLoader(CONFIG_ROOT).load_dataset("trafico", "no_existe")


def test_invalid_yaml_is_wrapped(tmp_path):
    environments = tmp_path / "environments"
    environments.mkdir()
    (environments / "dev.yaml").write_text("environment: [", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="YAML inválido"):
        ConfigLoader(tmp_path).load_environment("dev")
