# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# ///
import hashlib
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path

import mlflow
from pyspark.ml.functions import vector_to_array

from madrid_ml.contract import FEATURE_TABLE_COLUMNS
from madrid_ml.preprocessing import (
    FittedPreprocessor,
    manifest_to_dict,
    prepare_training_data,
)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UC_VOLUME_PATH_PATTERN = re.compile(
    r"^/Volumes/[a-z_][a-z0-9_]*/[a-z_][a-z0-9_]*/"
    r"[a-z_][a-z0-9_]*(?:/[A-Za-z0-9._-]+)*$"
)
_FEATURE_COLUMNS = tuple(name for name, _, _ in FEATURE_TABLE_COLUMNS)


def required_widget(name: str) -> str:
    value = dbutils.widgets.get(name).strip()  # noqa: F821
    if not value:
        raise ValueError(f"{name} is required")
    return value


def non_negative_version(name: str) -> int:
    raw_value = required_widget(name)
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_vectors(frame):
    rows = (
        frame.select(
            "cod_distrito",
            "feature_hour",
            vector_to_array("features").alias("features"),
        )
        .orderBy("cod_distrito", "feature_hour")
        .collect()
    )
    return [
        (row.cod_distrito, row.feature_hour, tuple(row.features))
        for row in rows
    ]


dbutils.widgets.text("gold_catalog", "dev_gold", "Gold catalog")  # noqa: F821
dbutils.widgets.text("labels_delta_version", "", "Labels Delta version")  # noqa: F821
dbutils.widgets.text(  # noqa: F821
    "features_delta_version", "", "Features Delta version"
)
dbutils.widgets.text("expected_snapshot_id", "", "Expected snapshot ID")  # noqa: F821
dbutils.widgets.text(  # noqa: F821
    "preprocessing_code_commit", "", "Preprocessing source commit"
)
dbutils.widgets.text("wheel_path", "", "Exact workspace wheel path")  # noqa: F821
dbutils.widgets.text("wheel_sha256", "", "Exact wheel SHA-256")  # noqa: F821
dbutils.widgets.text(  # noqa: F821
    "runtime_environment_version", "", "Databricks environment version"
)
dbutils.widgets.text("mlflow_experiment", "", "MLflow experiment path")  # noqa: F821
dbutils.widgets.text("mlflow_dfs_tmp", "", "MLflow Spark UC volume temp path")  # noqa: F821

gold_catalog = required_widget("gold_catalog")
labels_delta_version = non_negative_version("labels_delta_version")
features_delta_version = non_negative_version("features_delta_version")
expected_snapshot_id = required_widget("expected_snapshot_id")
preprocessing_code_commit = required_widget("preprocessing_code_commit")
wheel_path = required_widget("wheel_path")
wheel_sha256 = required_widget("wheel_sha256")
runtime_environment_version = required_widget("runtime_environment_version")
mlflow_experiment = required_widget("mlflow_experiment")
mlflow_dfs_tmp = required_widget("mlflow_dfs_tmp")

if not _IDENTIFIER_PATTERN.fullmatch(gold_catalog):
    raise ValueError(f"invalid gold_catalog: {gold_catalog!r}")
if not _COMMIT_PATTERN.fullmatch(preprocessing_code_commit):
    raise ValueError("preprocessing_code_commit must contain 40 lowercase hex characters")
if not _SHA256_PATTERN.fullmatch(wheel_sha256):
    raise ValueError("wheel_sha256 must contain 64 lowercase hex characters")
if not runtime_environment_version.isdigit():
    raise ValueError("runtime_environment_version must be numeric")
if not mlflow_experiment.startswith("/Users/"):
    raise ValueError("mlflow_experiment must be an absolute /Users/... workspace path")
if not _UC_VOLUME_PATH_PATTERN.fullmatch(mlflow_dfs_tmp):
    raise ValueError(
        "mlflow_dfs_tmp must be a path below /Volumes/<catalog>/<schema>/<volume>"
    )

wheel_file = Path(wheel_path)
if not wheel_path.startswith("/Workspace/") or wheel_file.suffix != ".whl":
    raise ValueError("wheel_path must be an absolute /Workspace/.../*.whl path")
if not wheel_file.is_file():
    raise ValueError(f"wheel_path is not a readable file: {wheel_path}")
observed_wheel_sha256 = sha256_file(wheel_file)
if observed_wheel_sha256 != wheel_sha256:
    raise ValueError(
        "wheel SHA-256 differs from the authorized artifact: "
        f"expected={wheel_sha256}, observed={observed_wheel_sha256}"
    )

package_version = version("madrid-ingestion")
python_version = platform.python_version()

prepared = prepare_training_data(
    spark=spark,  # noqa: F821
    gold_catalog=gold_catalog,
    labels_delta_version=labels_delta_version,
    features_delta_version=features_delta_version,
    expected_snapshot_id=expected_snapshot_id,
)
manifest = prepared.fitted_preprocessor.manifest
artifact_manifest = manifest_to_dict(
    manifest,
    preprocessing_code_commit=preprocessing_code_commit,
    wheel_sha256=wheel_sha256,
    package_version=package_version,
    runtime_environment_version=runtime_environment_version,
    python_version=python_version,
)

mlflow.set_experiment(mlflow_experiment)
with mlflow.start_run(run_name=f"preprocessing-{expected_snapshot_id}") as active_run:
    mlflow.set_tags(
        {
            "madrid_ml.snapshot_id": expected_snapshot_id,
            "madrid_ml.preprocessing_contract_version": (
                manifest.preprocessing_contract_version
            ),
            "madrid_ml.feature_schema_version": manifest.provenance.feature_schema_version,
            "madrid_ml.preprocessing_code_commit": preprocessing_code_commit,
            "madrid_ml.snapshot_code_commit": manifest.provenance.code_commit,
            "madrid_ml.wheel_sha256": wheel_sha256,
            "madrid_ml.runtime_environment_version": runtime_environment_version,
            "madrid_ml.spark_version": spark.version,  # noqa: F821
        }
    )
    mlflow.log_dict(artifact_manifest, "preprocessing_manifest.json")
    mlflow.log_dict(prepared.quality_by_split, "quality_by_split.json")
    mlflow.log_artifact(wheel_path, artifact_path="package")
    mlflow.spark.log_model(
        prepared.fitted_preprocessor.pipeline_model,
        artifact_path="preprocessor",
        dfs_tmpdir=mlflow_dfs_tmp,
    )
    mlflow_run_id = active_run.info.run_id
    loaded_model = mlflow.spark.load_model(
        f"runs:/{mlflow_run_id}/preprocessor",
        dfs_tmpdir=mlflow_dfs_tmp,
    )
    reloaded = FittedPreprocessor(
        pipeline_model=loaded_model,
        manifest=manifest,
    )

    for split_name, transformed in (
        ("validation", prepared.validation),
        ("evaluation", prepared.evaluation),
    ):
        raw_sample = (
            transformed.select(*_FEATURE_COLUMNS)
            .orderBy("cod_distrito", "feature_hour")
            .limit(32)
        )
        expected = prepared.fitted_preprocessor.transform(
            raw_sample, split_name=split_name
        )
        actual = reloaded.transform(raw_sample, split_name=split_name)
        if bounded_vectors(actual) != bounded_vectors(expected):
            raise RuntimeError(
                f"reloaded preprocessing vectors differ for {split_name}"
            )

result = {
    "status": "PREPROCESSING_READY",
    "snapshot_id": manifest.provenance.snapshot_id,
    "labels_delta_version": manifest.provenance.labels_delta_version,
    "features_delta_version": manifest.provenance.features_delta_version,
    "preprocessing_contract_version": manifest.preprocessing_contract_version,
    "train_rows": manifest.split_rows["train"],
    "validation_rows": manifest.split_rows["validation"],
    "evaluation_rows": manifest.split_rows["evaluation"],
    "excluded_rows": manifest.split_rows["excluded"],
    "vector_size": manifest.vector_size,
    "mlflow_run_id": mlflow_run_id,
    "preprocessing_code_commit": preprocessing_code_commit,
    "wheel_sha256": wheel_sha256,
    "runtime_environment_version": runtime_environment_version,
    "quality_by_split": prepared.quality_by_split,
}
dbutils.notebook.exit(  # noqa: F821
    json.dumps(result, sort_keys=True, separators=(",", ":"))
)
