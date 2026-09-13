"""Validaciones que combinan entorno, fuente y selección de ejecución."""

from __future__ import annotations

from madrid_ingestion.config.models import DatasetConfig, EnvironmentConfig, SourceConfig
from madrid_ingestion.core.exceptions import ConfigurationError


def validate_environment_file_name(config: EnvironmentConfig, requested: str) -> None:
    """Comprueba que el contenido del YAML corresponde al nombre solicitado."""
    if config.environment != requested:
        raise ConfigurationError(
            f"El entorno solicitado {requested!r} no coincide con "
            f"environment={config.environment!r}."
        )


def validate_source_file_name(config: SourceConfig, requested: str) -> None:
    """Comprueba que el contenido del YAML corresponde al nombre solicitado."""
    if config.source != requested:
        raise ConfigurationError(
            f"La fuente solicitada {requested!r} no coincide con source={config.source!r}."
        )


def validate_layer_enabled(
    source_config: SourceConfig, dataset_config: DatasetConfig, layer: str
) -> None:
    """Falla pronto si la fuente, dataset o capa están deshabilitados."""
    if not source_config.enabled:
        raise ConfigurationError(f"La fuente {source_config.source!r} está deshabilitada.")
    if not dataset_config.enabled:
        raise ConfigurationError(f"El dataset {dataset_config.name!r} está deshabilitado.")
    layer_config = getattr(dataset_config, layer, None)
    if layer_config is None:
        raise ConfigurationError(f"Capa no soportada: {layer!r}.")
    if not layer_config.enabled:
        raise ConfigurationError(
            f"La capa {layer!r} del dataset {dataset_config.name!r} está deshabilitada."
        )
