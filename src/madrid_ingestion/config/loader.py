"""Carga segura de YAML y conversión a modelos Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from madrid_ingestion.config.models import DatasetConfig, EnvironmentConfig, SourceConfig
from madrid_ingestion.config.validators import (
    validate_environment_file_name,
    validate_source_file_name,
)
from madrid_ingestion.core.exceptions import (
    ConfigurationError,
    DatasetNotFoundError,
)
from madrid_ingestion.core.naming import normalize_identifier

ModelT = TypeVar("ModelT", bound=BaseModel)


def default_config_root() -> Path:
    """Localiza ``conf`` desde un checkout o desde el directorio actual."""
    cwd_candidate = Path.cwd() / "conf"
    if cwd_candidate.is_dir():
        return cwd_candidate
    package_candidate = Path(__file__).resolve().parents[3] / "conf"
    return package_candidate


class ConfigLoader:
    """Repositorio de configuración basado en una raíz de directorios."""

    def __init__(self, config_root: str | Path | None = None) -> None:
        self.config_root = Path(config_root) if config_root else default_config_root()

    def _read_model(self, path: Path, model: type[ModelT]) -> ModelT:
        if not path.is_file():
            raise ConfigurationError(f"No existe el fichero de configuración: {path}.")
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"YAML inválido en {path}: {exc}") from exc
        except OSError as exc:
            raise ConfigurationError(f"No se pudo leer {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(f"El documento YAML {path} debe ser un mapa.")
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            raise ConfigurationError(f"Configuración inválida en {path}:\n{exc}") from exc

    def load_environment(self, environment: str) -> EnvironmentConfig:
        """Carga y valida un entorno por nombre."""
        name = normalize_identifier(environment)
        config = self._read_model(
            self.config_root / "environments" / f"{name}.yaml", EnvironmentConfig
        )
        validate_environment_file_name(config, name)
        return config

    def load_source(self, source: str) -> SourceConfig:
        """Carga y valida una fuente por nombre."""
        name = normalize_identifier(source)
        config = self._read_model(
            self.config_root / "sources" / f"{name}.yaml", SourceConfig
        )
        validate_source_file_name(config, name)
        return config

    def load_dataset(self, source: str, dataset: str) -> tuple[SourceConfig, DatasetConfig]:
        """Devuelve la fuente y el dataset solicitado."""
        source_config = self.load_source(source)
        name = normalize_identifier(dataset)
        for dataset_config in source_config.datasets:
            if dataset_config.name == name:
                return source_config, dataset_config
        raise DatasetNotFoundError(
            f"No existe el dataset {name!r} en la fuente {source_config.source!r}."
        )

    def source_names(self) -> list[str]:
        """Enumera las fuentes disponibles en orden estable."""
        sources_dir = self.config_root / "sources"
        if not sources_dir.is_dir():
            raise ConfigurationError(f"No existe el directorio de fuentes: {sources_dir}.")
        return sorted(path.stem for path in sources_dir.glob("*.yaml"))

    def validate_all(self, environment: str) -> tuple[EnvironmentConfig, list[SourceConfig]]:
        """Carga todo el contrato para detectar errores antes de ejecutar Spark."""
        environment_config = self.load_environment(environment)
        source_configs = [self.load_source(name) for name in self.source_names()]
        return environment_config, source_configs
