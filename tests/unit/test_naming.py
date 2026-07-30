import pytest

from madrid_ingestion.core.exceptions import ConfigurationError
from madrid_ingestion.core.naming import (
    build_catalog_name,
    build_table_name,
    normalize_identifier,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Tráfico NRT", "trafico_nrt"),
        ("Calidad-del  Aire", "calidad_del_aire"),
        ("ÁREA / Centro", "area_centro"),
        ("2019 accidentes", "_2019_accidentes"),
    ],
)
def test_normalize_identifier(raw, expected):
    assert normalize_identifier(raw) == expected


def test_normalize_identifier_rejects_empty_value():
    with pytest.raises(ConfigurationError):
        normalize_identifier("---")


def test_build_names():
    assert build_catalog_name("dev", "bronze") == "dev_bronze"
    assert (
        build_table_name("dev_bronze", "tráfico", "Tráfico NRT")
        == "dev_bronze.trafico.trafico_nrt"
    )
