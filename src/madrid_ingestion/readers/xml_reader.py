"""Adaptador XML basado en Auto Loader."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class XmlAutoLoaderReader:
    """Traduce el contrato YAML a opciones del formato XML."""

    def read_stream(
        self,
        spark: Any,
        source_path: str,
        schema_path: str,
        options: Mapping[str, str],
        autoloader_options: Mapping[str, str],
    ) -> Any:
        translated = dict(options)
        if "row_tag" in translated:
            translated["rowTag"] = translated.pop("row_tag")
        reader = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "xml")
            .option("cloudFiles.schemaLocation", schema_path)
        )
        for key, value in autoloader_options.items():
            reader = reader.option(key, value)
        for key, value in translated.items():
            reader = reader.option(key, value)
        return reader.load(source_path)
