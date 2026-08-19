"""Transformaciones técnicas permitidas antes de escribir Bronze."""

from __future__ import annotations

from typing import Any

from madrid_ingestion.config.models import BronzeTransformationConfig
from madrid_ingestion.core.columns import (
    normalize_dataframe_columns,
    rename_dataframe_columns,
)
from madrid_ingestion.core.exceptions import UnsupportedTransformationError


def apply_bronze_transformations(
    df: Any, transformations: list[BronzeTransformationConfig]
) -> Any:
    """Aplica únicamente normalización o renombrado de columnas."""
    result = df
    for transformation in transformations:
        if transformation.type == "normalize_column_names":
            result = normalize_dataframe_columns(result)
        elif transformation.type == "rename":
            assert transformation.columns is not None
            result = rename_dataframe_columns(result, transformation.columns)
        else:
            # Defensa adicional aunque Pydantic ya restringe el tipo.
            raise UnsupportedTransformationError(
                f"Transformación Bronze no soportada: {transformation.type!r}."
            )
    return result
