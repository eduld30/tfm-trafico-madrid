"""Contrato común de lectores streaming."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class BaseReader(Protocol):
    """Interfaz mínima de un adaptador de Auto Loader."""

    def read_stream(
        self,
        spark: Any,
        source_path: str,
        schema_path: str,
        options: Mapping[str, str],
        autoloader_options: Mapping[str, str],
    ) -> Any:
        """Construye un DataFrame streaming sin disparar acciones."""
        ...
