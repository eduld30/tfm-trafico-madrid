import math
from dataclasses import replace
from datetime import datetime

import pytest
from pyspark.ml.linalg import Vectors

from madrid_ml.models import LIGHTGBM_VERSION, fit_lightgbm, score_lightgbm
from madrid_ml.training import (
    BacktestScore,
    TrainingConfig,
    build_temporal_folds,
    select_config,
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


def test_select_config_uses_mean_ap_and_simple_tie_break():
    scores = [
        BacktestScore("logistic_regression", "l2_0_01", "fold_1", 0.20),
        BacktestScore("logistic_regression", "l2_0_01", "fold_2", 0.22),
        BacktestScore("logistic_regression", "l2_0_1", "fold_1", 0.21),
        BacktestScore("logistic_regression", "l2_0_1", "fold_2", 0.21),
    ]

    assert select_config(scores, "logistic_regression") == "l2_0_1"


def test_native_lightgbm_fits_and_scores_a_spark_frame(spark):
    frame = spark.createDataFrame(
        [
            (Vectors.dense(0.0, 0.0), 0.0),
            (Vectors.dense(0.0, 1.0), 0.0),
            (Vectors.dense(1.0, 0.0), 1.0),
            (Vectors.dense(1.0, 1.0), 1.0),
        ],
        ["features", "target_accident_next_hour"],
    )

    model = fit_lightgbm(
        frame,
        config_name="leaves_31",
        vector_size=2,
        fold_name="test",
    )
    scores = [
        row.score
        for row in score_lightgbm(
            model,
            frame,
            comparator="lightgbm",
            vector_size=2,
        )
        .select("score")
        .collect()
    ]

    assert LIGHTGBM_VERSION == "4.6.0"
    assert len(scores) == 4
    assert all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores)


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
