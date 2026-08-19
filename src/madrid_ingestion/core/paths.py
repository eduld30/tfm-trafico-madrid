"""Resolución centralizada de rutas ADLS."""

from __future__ import annotations

import re

from madrid_ingestion.core.exceptions import ConfigurationError

_DATE_SEGMENT = re.compile(r"^\d{4}/\d{2}/\d{2}(?:/\d{2})?/?$")


def normalize_relative_path(relative_path: str) -> str:
    """Valida y normaliza una ruta relativa de configuración."""
    path = relative_path.strip().replace("\\", "/").strip("/")
    if not path or path.startswith(("abfss:", "https:", "/")):
        raise ConfigurationError(f"Se esperaba una ruta ADLS relativa: {relative_path!r}.")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ConfigurationError(f"La ruta contiene segmentos no permitidos: {relative_path!r}.")
    return "/".join(parts)


def looks_like_dated_landing_path(relative_path: str) -> bool:
    """Indica si una ruta termina en la jerarquía temporal YYYY/MM/DD[/HH]."""
    parts = normalize_relative_path(relative_path).split("/")
    suffixes = ("/".join(parts[-4:]), "/".join(parts[-3:]))
    return any(_DATE_SEGMENT.fullmatch(suffix) for suffix in suffixes)


def build_adls_uri(account: str, container: str, relative_path: str) -> str:
    """Construye una URI ABFSS con barra final."""
    clean_account = account.strip()
    clean_container = container.strip()
    if not clean_account or not clean_container:
        raise ConfigurationError("La cuenta y el contenedor ADLS son obligatorios.")
    path = normalize_relative_path(relative_path)
    return f"abfss://{clean_container}@{clean_account}.dfs.core.windows.net/{path}/"


def build_table_path(account: str, container: str, target_path: str) -> str:
    """Resuelve el LOCATION de una tabla Delta externa."""
    return build_adls_uri(account, container, target_path)


def build_checkpoint_path(target_path: str, checkpoint_dir: str) -> str:
    """Añade el directorio de checkpoint a la ruta física de Bronze."""
    if not checkpoint_dir.startswith("_"):
        raise ConfigurationError("El directorio de checkpoint debe comenzar por '_'.")
    return f"{target_path.rstrip('/')}/{checkpoint_dir.strip('/')}/"


def build_schema_path(checkpoint_path: str, mode: str) -> str:
    """Resuelve la ubicación del esquema de Auto Loader."""
    if mode != "same_as_checkpoint":
        raise ConfigurationError(
            f"schema_location_mode no soportado: {mode!r}; use 'same_as_checkpoint'."
        )
    return checkpoint_path
