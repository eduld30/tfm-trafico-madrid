"""Small Unity Catalog model promotion boundary for approved MLflow runs."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MODEL_NAME_PATTERN = re.compile(
    r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$"
)
_ALIAS_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


class ModelPromotionError(RuntimeError):
    """Raised when a run is not an eligible trained madrid_ml model."""


@dataclass(frozen=True)
class ModelPromotionResult:
    """Registered model version and alias produced by one promotion."""

    model_run_id: str
    parent_run_id: str
    model_type: str
    registered_model_name: str
    model_version: str
    model_alias: str


def promote_model_run(
    *,
    model_run_id: str,
    registered_model_name: str,
    model_alias: str = "Champion",
) -> ModelPromotionResult:
    """Register one trained child-run classifier and atomically move its alias."""
    if not model_run_id.strip():
        raise ValueError("model_run_id must not be empty")
    if not _MODEL_NAME_PATTERN.fullmatch(registered_model_name):
        raise ValueError("registered_model_name must use catalog.schema.model syntax")
    if not _ALIAS_PATTERN.fullmatch(model_alias):
        raise ValueError(f"model_alias is invalid: {model_alias!r}")

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(registry_uri="databricks-uc")
    source_run = client.get_run(model_run_id)
    tags = source_run.data.tags
    model_type = tags.get("madrid_ml.comparator") or "unknown"
    parent_run_id = tags.get("mlflow.parentRunId")
    if not parent_run_id:
        raise ModelPromotionError(
            "model_run_id must identify a child run with a parent training run"
        )

    version = mlflow.register_model(
        f"runs:/{model_run_id}/classifier",
        registered_model_name,
        await_registration_for=300,
    )
    client.set_registered_model_alias(
        registered_model_name,
        model_alias,
        version.version,
    )
    client.set_model_version_tag(
        registered_model_name,
        version.version,
        "madrid_ml_parent_run_id",
        parent_run_id,
    )
    client.set_model_version_tag(
        registered_model_name,
        version.version,
        "madrid_ml_model_type",
        model_type,
    )
    return ModelPromotionResult(
        model_run_id=model_run_id,
        parent_run_id=parent_run_id,
        model_type=model_type,
        registered_model_name=registered_model_name,
        model_version=str(version.version),
        model_alias=model_alias,
    )
