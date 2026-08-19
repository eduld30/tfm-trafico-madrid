"""Lector CSV basado en Auto Loader."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CsvAutoLoaderReader:
    """Aísla la configuración de Auto Loader para CSV."""

    def read_stream(
        self,
        spark: Any,
        source_path: str,
        schema_path: str,
        options: Mapping[str, str],
        autoloader_options: Mapping[str, str],
    ) -> Any:
        reader = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("cloudFiles.schemaLocation", schema_path)
        )
        for key, value in autoloader_options.items():
            reader = reader.option(key, value)
        for key, value in options.items():
            reader = reader.option(key, value)
        return reader.load(source_path)
