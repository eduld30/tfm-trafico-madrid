from dataclasses import replace
from datetime import datetime

import pytest

from madrid_ml.training import (
    TrainingConfig,
    build_temporal_folds,
)


def test_temporal_folds_are_expanding_and_non_overlapping(spark):
    rows = [
        (datetime(2021, 12, 31, 23),),
        (datetime(2022, 1, 1),),
        (datetime(2022, 12, 31, 23),),
        (datetime(2023, 1, 1),),
        (datetime(2023, 12, 31, 23),),
    ]
    frame = spark.createDataFrame(rows, "feature_hour timestamp")

    fold_1, fold_2 = build_temporal_folds(frame)

    assert [row.feature_hour.year for row in fold_1.train.collect()] == [2021]
    assert {row.feature_hour.year for row in fold_1.validation.collect()} == {2022}
    assert {row.feature_hour.year for row in fold_2.train.collect()} == {2021, 2022}
    assert {row.feature_hour.year for row in fold_2.validation.collect()} == {2023}


def test_training_config_requires_a_unity_catalog_volume_path():
    valid = TrainingConfig(
        gold_catalog="dev_gold",
        labels_delta_version=1,
        features_delta_version=1,
        expected_snapshot_id="snapshot-id",
        mlflow_experiment_id="123",
        mlflow_dfs_tmp="/Volumes/dev_gold/ml/mlflow_tmp/model-training",
        code_commit="abc123",
        package_version="0.1.0",
    )

    assert valid.mlflow_dfs_tmp == (
        "/Volumes/dev_gold/ml/mlflow_tmp/model-training"
    )

    with pytest.raises(ValueError, match="Unity Catalog volume path"):
        replace(valid, mlflow_dfs_tmp="/tmp/mlflow")
