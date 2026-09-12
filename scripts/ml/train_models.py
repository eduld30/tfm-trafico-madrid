"""Databricks entrypoint for the reproducible ML training run."""

import argparse
import json
import os
from dataclasses import asdict
from importlib.metadata import version

from pyspark.sql import SparkSession

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ml.snapshot import latest_snapshot_reference
from madrid_ml.training import TrainingConfig, run_model_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="dev")
    parser.add_argument("--config-root", default="")
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--gold-catalog", default="")
    parser.add_argument("--labels-delta-version", type=int)
    parser.add_argument("--features-delta-version", type=int)
    parser.add_argument("--expected-snapshot-id", default="")
    parser.add_argument("--mlflow-experiment-id", default="")
    parser.add_argument("--mlflow-experiment-name", default="")
    parser.add_argument("--mlflow-dfs-tmp", default="")
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in ("environment",):
        if not getattr(args, name).strip():
            parser.error(f"--{name.replace('_', '-')} must not be empty")
    versions = (args.labels_delta_version, args.features_delta_version)
    if any(version is not None for version in versions) and not all(
        version is not None for version in versions
    ):
        parser.error("labels/features delta versions must be provided together")
    return args


def resolve_experiment_id(name: str) -> str:
    """Create/find the configured experiment so callers need no numeric ID."""
    import mlflow

    mlflow.set_experiment(name)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment could not be resolved: {name!r}")
    return experiment.experiment_id


def main() -> int:
    args = parse_args()
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession was found")
    config_root = args.config_root.strip()
    if not config_root:
        raise ValueError("--config-root is required when training runs from a bundle")
    environment_config = ConfigLoader(config_root).load_environment(args.environment)
    gold_catalog = args.gold_catalog.strip() or environment_config.catalogs.gold
    reference = latest_snapshot_reference(spark, gold_catalog)
    if args.snapshot_id.strip() and args.snapshot_id.strip() != reference.snapshot_id:
        raise ValueError(
            "snapshot_id no coincide con el snapshot Gold más reciente; "
            "use las versiones explícitas para reproducir uno antiguo"
        )
    labels_version = (
        args.labels_delta_version
        if args.labels_delta_version is not None
        else reference.labels_delta_version
    )
    features_version = (
        args.features_delta_version
        if args.features_delta_version is not None
        else reference.features_delta_version
    )
    snapshot_id = args.expected_snapshot_id.strip() or reference.snapshot_id
    mlflow_dfs_tmp = args.mlflow_dfs_tmp.strip() or (
        f"/Volumes/{gold_catalog}/ml/mlflow_tmp"
    )
    code_commit = args.code_commit.strip() or os.environ.get("GIT_COMMIT", "workspace")
    experiment_name = args.mlflow_experiment_name.strip() or (
        f"/Shared/madrid-ml-{args.environment.strip()}"
    )
    experiment_id = args.mlflow_experiment_id.strip() or resolve_experiment_id(
        experiment_name
    )
    config = TrainingConfig(
        gold_catalog=gold_catalog,
        labels_delta_version=labels_version,
        features_delta_version=features_version,
        expected_snapshot_id=snapshot_id,
        mlflow_experiment_id=experiment_id,
        mlflow_dfs_tmp=mlflow_dfs_tmp,
        code_commit=code_commit,
        package_version=version("madrid-ingestion"),
    )
    if args.smoke:
        from madrid_ml.smoke import run_training_smoke

        result = run_training_smoke(spark, config)
    else:
        result = run_model_training(spark, config)
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    main()
