from dataclasses import replace

import pytest
from pyspark.sql import functions as F

from madrid_ml.training import (
    TrainingConfig,
    build_temporal_folds,
)


def test_temporal_folds_are_expanding_and_non_overlapping(spark):
    rows = [
        ("2021-12-31 23:00:00",),
        ("2022-01-01 00:00:00",),
        ("2022-12-31 23:00:00",),
        ("2023-01-01 00:00:00",),
        ("2023-12-31 23:00:00",),
    ]
    frame = spark.createDataFrame(rows, "feature_hour string").withColumn(
        "feature_hour", F.to_timestamp("feature_hour")
    )

    fold_1, fold_2 = build_temporal_folds(frame)

    def years(dataframe):
        return {
            row.year
            for row in dataframe.select(F.year("feature_hour").alias("year"))
            .distinct()
            .collect()
        }

    assert years(fold_1.train) == {2021}
    assert years(fold_1.validation) == {2022}
    assert years(fold_2.train) == {2021, 2022}
    assert years(fold_2.validation) == {2023}


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
