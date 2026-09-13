"""Transformaciones técnicas reutilizables sobre nombres de columnas."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.core.naming import normalize_identifier


def normalize_column_name(value: str) -> str:
    """Normaliza una columna preservando el prefijo de metadatos ``_``."""
    stripped = value.strip()
    is_technical = stripped.startswith("_")
    separated_camel_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stripped)
    normalized = normalize_identifier(separated_camel_case)
    if is_technical:
        return f"_{normalized.lstrip('_')}"
    return normalized


def _ensure_unique(columns: list[str], operation: str) -> None:
    duplicates = sorted(name for name in set(columns) if columns.count(name) > 1)
    if duplicates:
        raise IngestionError(
            f"{operation} produce columnas duplicadas: {', '.join(duplicates)}."
        )


def normalize_dataframe_columns(df: Any) -> Any:
    """Normaliza todos los nombres sin modificar tipos ni valores."""
    normalized = [normalize_column_name(column) for column in df.columns]
    _ensure_unique(normalized, "normalize_column_names")
    return df.toDF(*normalized)


def rename_dataframe_columns(df: Any, columns: Mapping[str, str]) -> Any:
    """Renombra simultáneamente las columnas declaradas."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise IngestionError(f"rename: no existen las columnas {', '.join(missing)}.")
    renamed = [columns.get(column, column) for column in df.columns]
    _ensure_unique(renamed, "rename")
    return df.toDF(*renamed)


def validate_technical_column_names(df: Any, layer: str) -> None:
    """Falla pronto si quedan columnas fuera de la convención técnica."""
    invalid = sorted(
        column
        for column in df.columns
        if normalize_column_name(column) != column
    )
    if invalid:
        raise IngestionError(
            f"{layer}: quedan nombres de columna sin normalizar: "
            f"{', '.join(invalid)}. Configure normalize_column_names o rename."
        )
