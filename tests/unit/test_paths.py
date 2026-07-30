import pytest

from madrid_ingestion.core.exceptions import ConfigurationError
from madrid_ingestion.core.paths import (
    build_adls_uri,
    build_checkpoint_path,
    build_schema_path,
    looks_like_dated_landing_path,
)


def test_build_adls_and_checkpoint_paths():
    table_path = build_adls_uri("storage", "bronze", "/trafico/trafico_nrt/")
    assert (
        table_path
        == "abfss://bronze@storage.dfs.core.windows.net/trafico/trafico_nrt/"
    )
    checkpoint = build_checkpoint_path(table_path, "_checkpoint")
    assert checkpoint == f"{table_path}_checkpoint/"
    assert build_schema_path(checkpoint, "same_as_checkpoint") == checkpoint


def test_checkpoint_must_start_with_underscore():
    with pytest.raises(ConfigurationError, match="comenzar por '_'"):
        build_checkpoint_path("abfss://bronze@x/path/", "checkpoint")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("trafico/nrt", False),
        ("trafico/nrt/2026/07/26", True),
        ("trafico/nrt/2026/07/26/17", True),
    ],
)
def test_dated_landing_detection(path, expected):
    assert looks_like_dated_landing_path(path) is expected
