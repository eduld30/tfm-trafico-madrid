"""Normalización centralizada de identificadores técnicos."""

from __future__ import annotations

import re
import unicodedata

from madrid_ingestion.core.exceptions import ConfigurationError

_INVALID_CHARS = re.compile(r"[^a-z0-9_]+")
_MULTIPLE_UNDERSCORES = re.compile(r"_+")


def normalize_identifier(value: str) -> str:
    """Convierte un texto en un identificador técnico snake_case."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    normalized = _INVALID_CHARS.sub("_", ascii_value.strip().lower().replace("-", "_"))
    normalized = _MULTIPLE_UNDERSCORES.sub("_", normalized).strip("_")
    if not normalized:
        raise ConfigurationError(f"El identificador {value!r} queda vacío al normalizarse.")
    if normalized[0].isdigit():
        normalized = f"_{normalized}"
    return normalized


def build_catalog_name(environment: str, layer: str) -> str:
    """Construye el catálogo convencional ``<entorno>_<capa>``."""
    return f"{normalize_identifier(environment)}_{normalize_identifier(layer)}"


def build_table_name(catalog: str, source: str, dataset: str) -> str:
    """Construye un nombre completo de tabla de Unity Catalog."""
    return ".".join(
        (
            normalize_identifier(catalog),
            normalize_identifier(source),
            normalize_identifier(dataset),
        )
    )
