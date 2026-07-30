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
        record_path = translated.pop("record_path", None)
        parent_columns_option = translated.pop("parent_columns", "")
        parent_columns = [
            column.strip()
            for column in parent_columns_option.split(",")
            if column.strip()
        ]
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
        df = reader.load(source_path)
        if record_path is None:
            return df

        missing = sorted(
            {record_path, *parent_columns}.difference(df.columns)
        )
        if missing:
            raise ValueError(
                "La estructura XML configurada no existe: " + ", ".join(missing)
            )
        from pyspark.sql import functions as F

        record_alias = "__xml_record"
        if record_alias in df.columns:
            raise ValueError(f"El XML contiene la columna reservada {record_alias}.")
        exploded = df.select(
            *(F.col(column) for column in parent_columns),
            F.explode_outer(F.col(record_path)).alias(record_alias),
        )
        return exploded.select(*parent_columns, f"{record_alias}.*")
