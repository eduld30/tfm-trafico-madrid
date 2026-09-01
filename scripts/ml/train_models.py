"""Databricks entrypoint for the reproducible ML training run."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version

from pyspark.sql import SparkSession

from madrid_ml.training import TrainingConfig, run_model_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-catalog", required=True)
    parser.add_argument("--labels-delta-version", required=True, type=int)
    parser.add_argument("--features-delta-version", required=True, type=int)
    parser.add_argument("--expected-snapshot-id", required=True)
    parser.add_argument("--mlflow-experiment-id", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    for name in (
        "gold_catalog",
        "expected_snapshot_id",
        "mlflow_experiment_id",
        "code_commit",
    ):
        if not getattr(args, name).strip():
            parser.error(f"--{name.replace('_', '-')} must not be empty")
    return args


def main() -> int:
    args = parse_args()
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession was found")
    config = TrainingConfig(
        gold_catalog=args.gold_catalog,
        labels_delta_version=args.labels_delta_version,
        features_delta_version=args.features_delta_version,
        expected_snapshot_id=args.expected_snapshot_id,
        mlflow_experiment_id=args.mlflow_experiment_id,
        code_commit=args.code_commit,
        package_version=version("madrid-ingestion"),
    )
    result = run_model_training(spark, config)
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
