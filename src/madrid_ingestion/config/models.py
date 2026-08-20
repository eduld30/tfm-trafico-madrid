"""Modelos Pydantic que representan el contrato YAML."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from madrid_ingestion.core.naming import normalize_identifier
from madrid_ingestion.core.paths import looks_like_dated_landing_path, normalize_relative_path

SUPPORTED_TRANSFORMATIONS = frozenset(
    {
        "normalize_column_names",
        "rename",
        "select",
        "drop",
        "cast",
        "trim",
        "upper",
        "strip_accents",
        "empty_to_null",
        "replace_values",
        "regex_replace",
        "filter",
        "add_literal",
        "parse_timestamp",
        "parse_date",
        "hourly_wide_to_long",
        "deduplicate",
        "lookup_join",
    }
)


class StrictModel(BaseModel):
    """Base común que rechaza claves YAML desconocidas."""

    model_config = ConfigDict(extra="forbid")


class StorageContainers(StrictModel):
    landing: str
    bronze: str
    silver: str
    gold: str


class StorageConfig(StrictModel):
    account_name: str = Field(min_length=1)
    containers: StorageContainers


class CatalogsConfig(StrictModel):
    bronze: str
    silver: str
    gold: str


class PathsConfig(StrictModel):
    checkpoint_dir: str = "_checkpoint"
    schema_location_mode: Literal["same_as_checkpoint"] = "same_as_checkpoint"

    @field_validator("checkpoint_dir")
    @classmethod
    def validate_checkpoint_dir(cls, value: str) -> str:
        if "/" in value or "\\" in value or not value.startswith("_"):
            raise ValueError(
                "checkpoint_dir debe ser un único directorio cuyo nombre comience por '_'."
            )
        return value


class RuntimeConfig(StrictModel):
    default_trigger: Literal["available_now"] = "available_now"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    diagnostic_counts: bool = False


class EnvironmentConfig(StrictModel):
    environment: str
    storage: StorageConfig
    catalogs: CatalogsConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_names(self) -> EnvironmentConfig:
        normalized_environment = normalize_identifier(self.environment)
        if self.environment != normalized_environment:
            raise ValueError("environment debe estar normalizado en minúsculas y snake_case.")
        expected = {
            "bronze": f"{normalized_environment}_bronze",
            "silver": f"{normalized_environment}_silver",
            "gold": f"{normalized_environment}_gold",
        }
        actual = self.catalogs.model_dump()
        if actual != expected:
            raise ValueError(f"Los catálogos deben seguir la convención {expected!r}.")
        return self


class AutoLoaderOptions(StrictModel):
    schema_evolution_mode: str = "addNewColumns"
    rescued_data_column: str = "_rescued_data"
    extra_options: dict[str, str] = Field(default_factory=dict)


class BronzeTransformationConfig(StrictModel):
    """Transformación técnica de esquema permitida en Bronze."""

    type: Literal["normalize_column_names", "rename"]
    columns: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> BronzeTransformationConfig:
        if self.type == "rename" and not self.columns:
            raise ValueError("Bronze rename requiere un mapa columns no vacío.")
        if self.type == "normalize_column_names" and self.columns is not None:
            raise ValueError("Bronze normalize_column_names no admite columns.")
        return self


class BronzeConfig(StrictModel):
    enabled: bool = True
    format: Literal["csv", "xml"]
    source_path: str
    target_path: str
    reader_options: dict[str, str] = Field(default_factory=dict)
    autoloader_options: AutoLoaderOptions = Field(default_factory=AutoLoaderOptions)
    transformations: list[BronzeTransformationConfig] = Field(default_factory=list)

    @field_validator("source_path", "target_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @field_validator("source_path")
    @classmethod
    def validate_stable_landing_root(cls, value: str) -> str:
        if looks_like_dated_landing_path(value):
            raise ValueError(
                "source_path debe apuntar a la raíz estable, no a una carpeta YYYY/MM/DD[/HH]."
            )
        return value


class LookupTableConfig(StrictModel):
    layer: Literal["bronze", "silver"]
    source: str
    dataset: str


class OrderByConfig(StrictModel):
    column: str
    direction: Literal["asc", "desc"] = "asc"


class TransformationConfig(StrictModel):
    """Unión declarativa compacta de los parámetros de transformación."""

    type: str
    columns: dict[str, str] | list[str] | None = None
    column: str | None = None
    values: dict[Any, Any] | None = None
    condition: str | None = None
    value: Any = None
    source_column: str | None = None
    source_columns: list[str] | None = None
    target_column: str | None = None
    format: str | None = None
    pattern: str | None = None
    replacement: str | None = None
    keys: list[str] | None = None
    order_by: list[OrderByConfig] | None = None
    lookup_table: LookupTableConfig | None = None
    join_type: Literal["left", "inner"] | None = None
    conditions: dict[str, str] | None = None
    select: dict[str, str] | list[str] | None = None
    year_column: str | None = None
    month_column: str | None = None
    day_column: str | None = None
    value_prefix: str | None = None
    validity_prefix: str | None = None
    value_column: str | None = None
    validity_column: str | None = None
    timestamp_column: str | None = None
    hours: int | None = None
    hour_offset: int = -1

    @model_validator(mode="after")
    def validate_parameters(self) -> TransformationConfig:
        if self.type not in SUPPORTED_TRANSFORMATIONS:
            raise ValueError(f"Transformación no soportada: {self.type!r}.")

        required: dict[str, tuple[str, ...]] = {
            "rename": ("columns",),
            "select": ("columns",),
            "drop": ("columns",),
            "cast": ("columns",),
            "trim": ("columns",),
            "upper": ("columns",),
            "strip_accents": ("columns",),
            "empty_to_null": ("columns",),
            "replace_values": ("column", "values"),
            "regex_replace": ("columns", "pattern", "replacement"),
            "filter": ("condition",),
            "add_literal": ("column",),
            "parse_timestamp": ("target_column", "format"),
            "parse_date": ("source_column", "target_column", "format"),
            "hourly_wide_to_long": (
                "year_column",
                "month_column",
                "day_column",
                "value_prefix",
                "validity_prefix",
                "value_column",
                "validity_column",
                "timestamp_column",
            ),
            "deduplicate": ("keys", "order_by"),
            "lookup_join": ("lookup_table", "conditions", "select"),
        }
        missing = [
            name for name in required.get(self.type, ()) if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"{self.type} requiere los parámetros: {', '.join(missing)}.")

        list_column_transforms = {
            "select",
            "drop",
            "trim",
            "upper",
            "strip_accents",
            "empty_to_null",
            "regex_replace",
        }
        mapping_column_transforms = {"rename", "cast"}
        if self.type in list_column_transforms and not isinstance(self.columns, list):
            raise ValueError(f"{self.type}.columns debe ser una lista.")
        if self.type in mapping_column_transforms and not isinstance(self.columns, dict):
            raise ValueError(f"{self.type}.columns debe ser un mapa.")
        if self.type == "parse_timestamp":
            sources = bool(self.source_column) + bool(self.source_columns)
            if sources != 1:
                raise ValueError(
                    "parse_timestamp requiere exactamente source_column o source_columns."
                )
        if self.type == "hourly_wide_to_long":
            effective_hours = self.hours or 24
            if not 1 <= effective_hours <= 24:
                raise ValueError("hourly_wide_to_long.hours debe estar entre 1 y 24.")
            output_columns = {
                self.value_column,
                self.validity_column,
                self.timestamp_column,
            }
            if len(output_columns) != 3:
                raise ValueError(
                    "hourly_wide_to_long requiere tres columnas de salida distintas."
                )
        if self.type == "lookup_join" and not isinstance(self.select, dict):
            raise ValueError("lookup_join.select debe ser un mapa destino: columna_lookup.")
        if self.type == "deduplicate" and (not self.keys or not self.order_by):
            raise ValueError("deduplicate requiere keys y order_by no vacíos.")
        if self.type == "lookup_join" and (
            not self.conditions or not self.select
        ):
            raise ValueError("lookup_join requiere conditions y select no vacíos.")
        return self


class SilverSourceConfig(StrictModel):
    source: str
    dataset: str


class SilverConfig(StrictModel):
    enabled: bool = True
    target_path: str
    write_strategy: Literal["overwrite", "merge", "append"]
    business_keys: list[str] = Field(default_factory=list)
    partition_by: list[str] = Field(default_factory=list)
    transformations: list[TransformationConfig] = Field(default_factory=list)
    overwrite_schema: bool = False
    source_table: SilverSourceConfig | None = None

    @field_validator("target_path")
    @classmethod
    def validate_target_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @model_validator(mode="after")
    def validate_strategy(self) -> SilverConfig:
        if self.write_strategy == "merge" and not self.business_keys:
            raise ValueError("write_strategy='merge' requiere business_keys.")
        if len(self.business_keys) != len(set(self.business_keys)):
            raise ValueError("business_keys contiene valores duplicados.")
        return self


class DatasetConfig(StrictModel):
    name: str
    description: str
    enabled: bool = True
    bronze: BronzeConfig
    silver: SilverConfig

    @model_validator(mode="after")
    def validate_identifier_and_paths(self) -> DatasetConfig:
        if self.name != normalize_identifier(self.name):
            raise ValueError("El nombre del dataset debe estar normalizado.")
        return self


class SourceConfig(StrictModel):
    source: str
    enabled: bool = True
    datasets: list[DatasetConfig]

    @model_validator(mode="after")
    def validate_source(self) -> SourceConfig:
        if self.source != normalize_identifier(self.source):
            raise ValueError("El nombre de la fuente debe estar normalizado.")
        names = [dataset.name for dataset in self.datasets]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"Datasets duplicados: {', '.join(duplicates)}.")
        for dataset in self.datasets:
            expected = f"{self.source}/{dataset.name}"
            if dataset.bronze.source_path != expected:
                raise ValueError(
                    f"{dataset.name}.bronze.source_path debe ser {expected!r}."
                )
            if dataset.bronze.target_path != expected:
                raise ValueError(
                    f"{dataset.name}.bronze.target_path debe ser {expected!r}."
                )
            if dataset.silver.target_path != expected:
                raise ValueError(
                    f"{dataset.name}.silver.target_path debe ser {expected!r}."
                )
        return self
