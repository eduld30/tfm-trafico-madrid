"""Temporal backtesting, final fitting, evaluation, and MLflow orchestration."""

from __future__ import annotations

import json
import platform
import re
from dataclasses import asdict, dataclass

from pyspark.ml import Estimator, Model
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from madrid_ml.evaluation import (
    calibration_bins,
    curve_points,
    global_binary_metrics,
    segmented_binary_metrics,
)
from madrid_ml.models import LOGISTIC_CONFIG, LOGISTIC_CONFIG_NAME, build_logistic_regression
from madrid_ml.preprocessing import (
    GoldTrainingSnapshot,
    PreprocessingManifest,
    fit_preprocessor,
    load_gold_training_snapshot,
    split_training_snapshot,
)
from madrid_ml.tracking import (
    load_logged_preprocessor,
    log_spark_model_run,
    start_training_run,
)

TRAINING_COMPLETE = "TRAINING_COMPLETE"
BACKTEST_PERIODS = {
    "fold_1": {
        "train": ("2019-01-01 00:00:00", "2022-01-01 00:00:00"),
        "validation": ("2022-01-01 00:00:00", "2023-01-01 00:00:00"),
    },
    "fold_2": {
        "train": ("2019-01-01 00:00:00", "2023-01-01 00:00:00"),
        "validation": ("2023-01-01 00:00:00", "2024-01-01 00:00:00"),
    },
}

MLFLOW_DFS_TMP_PATTERN = re.compile(
    r"^/Volumes/[a-z_][a-z0-9_]*/[a-z_][a-z0-9_]*/"
    r"[a-z_][a-z0-9_]*(?:/[A-Za-z0-9._-]+)*$"
)


@dataclass(frozen=True)
class TrainingConfig:
    """Identity required to reproduce one training execution."""

    gold_catalog: str
    labels_delta_version: int
    features_delta_version: int
    expected_snapshot_id: str
    mlflow_experiment_id: str
    mlflow_dfs_tmp: str
    code_commit: str
    package_version: str

    def __post_init__(self) -> None:
        """Reject paths that serverless MLflow cannot use for Spark artifacts."""
        if not MLFLOW_DFS_TMP_PATTERN.fullmatch(self.mlflow_dfs_tmp):
            raise ValueError(
                "mlflow_dfs_tmp must be a Unity Catalog volume path under "
                "/Volumes/<catalog>/<schema>/<volume>"
            )


@dataclass(frozen=True)
class BacktestScore:
    """Average Precision for one model configuration and temporal fold."""

    model_name: str
    config_name: str
    fold_name: str
    average_precision: float


@dataclass(frozen=True)
class TemporalFold:
    """One expanding temporal train and its following validation period."""

    name: str
    train: DataFrame
    validation: DataFrame


@dataclass(frozen=True)
class TrainingResult:
    """Compact completed training result without Spark DataFrames."""

    status: str
    parent_run_id: str
    child_run_ids: dict[str, str]
    selected_configs: dict[str, str]
    validation_metrics: dict[str, dict[str, float | int | None]]
    evaluation_metrics: dict[str, dict[str, float | int | None]]


def _between(frame: DataFrame, start: str, end: str) -> DataFrame:
    feature_hour = F.col("feature_hour")
    return frame.where(
        (feature_hour >= F.to_timestamp(F.lit(start)))
        & (feature_hour < F.to_timestamp(F.lit(end)))
    )


def build_temporal_folds(train: DataFrame) -> tuple[TemporalFold, TemporalFold]:
    """Build the two approved expanding backtest folds."""
    fold_1 = BACKTEST_PERIODS["fold_1"]
    fold_2 = BACKTEST_PERIODS["fold_2"]
    return (
        TemporalFold(
            name="fold_1",
            train=_between(train, *fold_1["train"]),
            validation=_between(train, *fold_1["validation"]),
        ),
        TemporalFold(
            name="fold_2",
            train=_between(train, *fold_2["train"]),
            validation=_between(train, *fold_2["validation"]),
        ),
    )


def _fit_model(
    estimator: Estimator,
    train: DataFrame,
    *,
    model_name: str,
    config_name: str,
    fold_name: str,
) -> Model:
    try:
        return estimator.fit(train)
    except Exception as exc:
        training_error = RuntimeError(
            "model fit failed: "
            f"model_name={model_name}, config_name={config_name}, fold_name={fold_name}"
        )
        raise training_error from exc


def _score_model(model: Model, frame: DataFrame, comparator: str) -> DataFrame:
    return (
        model.transform(frame)
        .withColumn("score", vector_to_array(F.col("probability"))[1])
        .withColumn("comparator", F.lit(comparator))
    )


def _compact_evaluation(scored: DataFrame) -> dict[str, object]:
    return {
        "global": global_binary_metrics(scored),
        "segments": [
            row.asDict(recursive=True)
            for row in segmented_binary_metrics(scored).collect()
        ],
        "curves": [
            row.asDict(recursive=True)
            for row in curve_points(scored).orderBy(F.col("threshold").desc()).collect()
        ],
        "calibration": [
            row.asDict(recursive=True) for row in calibration_bins(scored).collect()
        ],
    }


def _require_average_precision(
    metrics: dict[str, float | int | None],
    *,
    model_name: str,
    config_name: str,
    fold_name: str,
) -> float:
    average_precision = metrics["average_precision"]
    if not isinstance(average_precision, float):
        raise RuntimeError(
            "backtest Average Precision is not evaluable: "
            f"model_name={model_name}, config_name={config_name}, fold_name={fold_name}"
        )
    return average_precision


def _run_backtest(
    snapshot: GoldTrainingSnapshot,
    spark: SparkSession,
) -> list[BacktestScore]:
    scores: list[BacktestScore] = []

    for fold in build_temporal_folds(snapshot.features):
        split_rows = {
            "train": fold.train.count(),
            "validation": fold.validation.count(),
        }
        fitted_preprocessor = fit_preprocessor(
            fold.train,
            snapshot.provenance,
            split_rows=split_rows,
            spark_version=spark.version,
        )
        transformed_train = fitted_preprocessor.transform(
            fold.train,
            split_name="train",
        )
        transformed_validation = fitted_preprocessor.transform(
            fold.validation,
            split_name="validation",
        )
        model = _fit_model(
            build_logistic_regression(),
            transformed_train,
            model_name="logistic_regression",
            config_name=LOGISTIC_CONFIG_NAME,
            fold_name=fold.name,
        )
        metrics = global_binary_metrics(
            _score_model(model, transformed_validation, "logistic_regression")
        )
        scores.append(
            BacktestScore(
                "logistic_regression",
                LOGISTIC_CONFIG_NAME,
                fold.name,
                _require_average_precision(
                    metrics,
                    model_name="logistic_regression",
                    config_name=LOGISTIC_CONFIG_NAME,
                    fold_name=fold.name,
                ),
            )
        )

    return scores


def _parent_tags(
    *,
    config: TrainingConfig,
    snapshot: GoldTrainingSnapshot,
    spark: SparkSession,
    manifest: PreprocessingManifest,
) -> dict[str, str]:
    provenance = snapshot.provenance
    return {
        "madrid_ml.snapshot_id": provenance.snapshot_id,
        "madrid_ml.labels_delta_version": str(provenance.labels_delta_version),
        "madrid_ml.features_delta_version": str(provenance.features_delta_version),
        "madrid_ml.code_commit": config.code_commit,
        "madrid_ml.snapshot_code_commit": provenance.code_commit,
        "madrid_ml.package_version": config.package_version,
        "madrid_ml.spark_version": spark.version,
        "madrid_ml.python_version": platform.python_version(),
        "madrid_ml.feature_schema_version": provenance.feature_schema_version,
        "madrid_ml.preprocessing_contract_version": (
            manifest.preprocessing_contract_version
        ),
        "madrid_ml.backtest_folds": json.dumps(
            BACKTEST_PERIODS,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def run_model_training(
    spark: SparkSession,
    config: TrainingConfig,
) -> TrainingResult:
    """Backtest, fit, evaluate, and register the approved logistic model."""

    snapshot = load_gold_training_snapshot(
        spark,
        gold_catalog=config.gold_catalog,
        labels_delta_version=config.labels_delta_version,
        features_delta_version=config.features_delta_version,
        expected_snapshot_id=config.expected_snapshot_id,
    )
    splits = split_training_snapshot(snapshot)
    backtest_scores = _run_backtest(
        GoldTrainingSnapshot(splits.train, snapshot.provenance),
        spark,
    )
    selected_configs = {"logistic_regression": LOGISTIC_CONFIG_NAME}

    final_preprocessor = fit_preprocessor(
        splits.train,
        snapshot.provenance,
        split_rows=splits.row_counts,
        spark_version=spark.version,
    )
    transformed_train = final_preprocessor.transform(
        splits.train,
        split_name="train",
    )
    transformed_validation = final_preprocessor.transform(
        splits.validation,
        split_name="validation",
    )
    transformed_evaluation = final_preprocessor.transform(
        splits.evaluation,
        split_name="evaluation",
    )

    tags = _parent_tags(
        config=config,
        snapshot=snapshot,
        spark=spark,
        manifest=final_preprocessor.manifest,
    )
    child_run_ids: dict[str, str] = {}
    evaluations: dict[str, dict[str, object]] = {}

    with start_training_run(
        experiment_id=config.mlflow_experiment_id,
        run_name=f"model-training-{config.expected_snapshot_id}",
        tags=tags,
        manifest=asdict(final_preprocessor.manifest),
        preprocessor=final_preprocessor.pipeline_model,
        dfs_tmpdir=config.mlflow_dfs_tmp,
    ) as parent_run_id:
        logistic_model = _fit_model(
            build_logistic_regression(),
            transformed_train,
            model_name="logistic_regression",
            config_name=selected_configs["logistic_regression"],
            fold_name="final_2019_2023",
        )
        evaluations["logistic_regression"] = {
            "validation": _compact_evaluation(
                _score_model(logistic_model, transformed_validation, "logistic_regression")
            ),
            "evaluation": _compact_evaluation(
                _score_model(logistic_model, transformed_evaluation, "logistic_regression")
            ),
        }

        raw_validation_sample = splits.validation.orderBy(
            "cod_distrito", "feature_hour"
        ).limit(32)
        logged_validation_sample = load_logged_preprocessor(
            parent_run_id=parent_run_id,
            raw_validation_sample=raw_validation_sample,
            manifest=final_preprocessor.manifest,
            dfs_tmpdir=config.mlflow_dfs_tmp,
        )

        child_run_ids["logistic_regression"] = log_spark_model_run(
            experiment_id=config.mlflow_experiment_id,
            parent_run_id=parent_run_id,
            comparator="logistic_regression",
            parameters={"config_name": LOGISTIC_CONFIG_NAME, **LOGISTIC_CONFIG},
            evaluations=evaluations["logistic_regression"],
            backtest=[
                asdict(score)
                for score in backtest_scores
                if score.model_name == "logistic_regression"
            ],
            model=logistic_model,
            validation_sample=logged_validation_sample,
            dfs_tmpdir=config.mlflow_dfs_tmp,
        )

    return TrainingResult(
        status=TRAINING_COMPLETE,
        parent_run_id=parent_run_id,
        child_run_ids=child_run_ids,
        selected_configs=selected_configs,
        validation_metrics={
            comparator: evaluation["validation"]["global"]
            for comparator, evaluation in evaluations.items()
        },
        evaluation_metrics={
            comparator: evaluation["evaluation"]["global"]
            for comparator, evaluation in evaluations.items()
        },
    )
