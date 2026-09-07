"""Temporal backtesting, final fitting, evaluation, and MLflow orchestration."""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal

from pyspark.ml import Estimator, Model
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from madrid_ml.baselines import BaselineState, fit_baselines, score_baselines
from madrid_ml.evaluation import (
    calibration_bins,
    curve_points,
    global_binary_metrics,
    segmented_binary_metrics,
)
from madrid_ml.models import (
    LIGHTGBM_CONFIGS,
    LIGHTGBM_VERSION,
    LOGISTIC_CONFIGS,
    build_logistic_regression,
    fit_lightgbm,
    score_lightgbm,
)
from madrid_ml.preprocessing import (
    GoldTrainingSnapshot,
    PreprocessingManifest,
    fit_preprocessor,
    load_gold_training_snapshot,
    split_training_snapshot,
)
from madrid_ml.tracking import (
    load_logged_preprocessor,
    log_baseline_run,
    log_lightgbm_run,
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
CONFIG_SIMPLICITY = {
    "logistic_regression": ("l2_0_1", "l2_0_01"),
    "lightgbm": ("leaves_31", "leaves_63"),
}
BASELINE_COMPARATORS = {
    "global": "global_prevalence",
    "district": "district_frequency",
    "district_hour_day": "district_hour_day_frequency",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Identity required to reproduce one training execution."""

    gold_catalog: str
    labels_delta_version: int
    features_delta_version: int
    expected_snapshot_id: str
    mlflow_experiment_id: str
    code_commit: str
    package_version: str


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


def select_config(scores: Sequence[BacktestScore], model_name: str) -> str:
    """Select the best mean AP, breaking ties toward explicit simplicity."""
    simplicity = CONFIG_SIMPLICITY.get(model_name)
    if simplicity is None:
        raise ValueError(f"unknown model_name {model_name!r}")
    if not scores or any(score.model_name != model_name for score in scores):
        raise ValueError(f"backtest scores do not match model_name {model_name!r}")

    grouped: dict[str, list[BacktestScore]] = {name: [] for name in simplicity}
    for score in scores:
        if score.config_name not in grouped:
            raise ValueError(
                f"unknown configuration {score.config_name!r} for {model_name!r}"
            )
        grouped[score.config_name].append(score)

    expected_folds = {"fold_1", "fold_2"}
    means: dict[str, Decimal] = {}
    for config_name, config_scores in grouped.items():
        observed_folds = [score.fold_name for score in config_scores]
        if len(observed_folds) != 2 or set(observed_folds) != expected_folds:
            raise ValueError(
                f"{model_name}/{config_name} requires exactly fold_1 and fold_2; "
                f"observed={observed_folds}"
            )
        means[config_name] = sum(
            Decimal(str(score.average_precision)) for score in config_scores
        ) / len(config_scores)

    return max(simplicity, key=means.__getitem__)


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


def _baseline_states(state: BaselineState) -> dict[str, dict[str, object]]:
    district_frequencies = [
        row.asDict(recursive=True)
        for row in state.district_frequencies.orderBy("cod_distrito").collect()
    ]
    district_hour_day_frequencies = [
        row.asDict(recursive=True)
        for row in state.district_hour_day_frequencies.orderBy(
            "cod_distrito", "hora_dia", "dia_semana"
        ).collect()
    ]
    global_state: dict[str, object] = {
        "global_prevalence": state.global_prevalence,
    }
    return {
        "global_prevalence": global_state,
        "district_frequency": {
            **global_state,
            "district_frequencies": district_frequencies,
        },
        "district_hour_day_frequency": {
            **global_state,
            "district_frequencies": district_frequencies,
            "district_hour_day_frequencies": district_hour_day_frequencies,
        },
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
) -> tuple[list[BacktestScore], dict[str, list[dict[str, object]]]]:
    scores: list[BacktestScore] = []
    baseline_results: dict[str, list[dict[str, object]]] = {
        comparator: [] for comparator in BASELINE_COMPARATORS.values()
    }

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
        baseline_state = fit_baselines(fold.train)
        for key, comparator in BASELINE_COMPARATORS.items():
            metrics = global_binary_metrics(
                score_baselines(baseline_state, fold.validation)[key]
            )
            baseline_results[comparator].append({"fold_name": fold.name, **metrics})

        for config_name in LOGISTIC_CONFIGS:
            model = _fit_model(
                build_logistic_regression(config_name),
                transformed_train,
                model_name="logistic_regression",
                config_name=config_name,
                fold_name=fold.name,
            )
            metrics = global_binary_metrics(
                _score_model(model, transformed_validation, "logistic_regression")
            )
            scores.append(
                BacktestScore(
                    "logistic_regression",
                    config_name,
                    fold.name,
                    _require_average_precision(
                        metrics,
                        model_name="logistic_regression",
                        config_name=config_name,
                        fold_name=fold.name,
                    ),
                )
            )

        for config_name in LIGHTGBM_CONFIGS:
            model = fit_lightgbm(
                transformed_train,
                config_name=config_name,
                vector_size=fitted_preprocessor.manifest.vector_size,
                fold_name=fold.name,
            )
            metrics = global_binary_metrics(
                score_lightgbm(
                    model,
                    transformed_validation,
                    comparator="lightgbm",
                    vector_size=fitted_preprocessor.manifest.vector_size,
                )
            )
            scores.append(
                BacktestScore(
                    "lightgbm",
                    config_name,
                    fold.name,
                    _require_average_precision(
                        metrics,
                        model_name="lightgbm",
                        config_name=config_name,
                        fold_name=fold.name,
                    ),
                )
            )

    return scores, baseline_results


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
        "madrid_ml.lightgbm_version": LIGHTGBM_VERSION,
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
    """Train, compare, log, reload, and return the five approved comparators."""

    snapshot = load_gold_training_snapshot(
        spark,
        gold_catalog=config.gold_catalog,
        labels_delta_version=config.labels_delta_version,
        features_delta_version=config.features_delta_version,
        expected_snapshot_id=config.expected_snapshot_id,
    )
    splits = split_training_snapshot(snapshot)
    backtest_scores, baseline_backtest = _run_backtest(
        GoldTrainingSnapshot(splits.train, snapshot.provenance),
        spark,
    )
    selected_configs = {
        model_name: select_config(
            [score for score in backtest_scores if score.model_name == model_name],
            model_name,
        )
        for model_name in CONFIG_SIMPLICITY
    }

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
    baseline_state = fit_baselines(splits.train)

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
    ) as parent_run_id:
        logistic_model = _fit_model(
            build_logistic_regression(selected_configs["logistic_regression"]),
            transformed_train,
            model_name="logistic_regression",
            config_name=selected_configs["logistic_regression"],
            fold_name="final_2019_2023",
        )
        lightgbm_model = fit_lightgbm(
            transformed_train,
            config_name=selected_configs["lightgbm"],
            vector_size=final_preprocessor.manifest.vector_size,
            fold_name="final_2019_2023",
        )

        validation_baselines = score_baselines(baseline_state, splits.validation)
        evaluation_baselines = score_baselines(baseline_state, splits.evaluation)
        for key, comparator in BASELINE_COMPARATORS.items():
            evaluations[comparator] = {
                "validation": _compact_evaluation(validation_baselines[key]),
                "evaluation": _compact_evaluation(evaluation_baselines[key]),
                "backtest": baseline_backtest[comparator],
            }

        model_frames = {
            "logistic_regression": (
                _score_model(
                    logistic_model,
                    transformed_validation,
                    "logistic_regression",
                ),
                _score_model(
                    logistic_model,
                    transformed_evaluation,
                    "logistic_regression",
                ),
            ),
            "lightgbm": (
                score_lightgbm(
                    lightgbm_model,
                    transformed_validation,
                    comparator="lightgbm",
                    vector_size=final_preprocessor.manifest.vector_size,
                ),
                score_lightgbm(
                    lightgbm_model,
                    transformed_evaluation,
                    comparator="lightgbm",
                    vector_size=final_preprocessor.manifest.vector_size,
                ),
            ),
        }
        for comparator, (validation_frame, evaluation_frame) in model_frames.items():
            evaluations[comparator] = {
                "validation": _compact_evaluation(validation_frame),
                "evaluation": _compact_evaluation(evaluation_frame),
            }

        baseline_states = _baseline_states(baseline_state)
        for comparator in BASELINE_COMPARATORS.values():
            child_run_ids[comparator] = log_baseline_run(
                parent_run_id=parent_run_id,
                comparator=comparator,
                parameters={"training_period": "2019-2023"},
                evaluations=evaluations[comparator],
                state=baseline_states[comparator],
            )

        raw_validation_sample = splits.validation.orderBy(
            "cod_distrito", "feature_hour"
        ).limit(32)
        logged_validation_sample = load_logged_preprocessor(
            parent_run_id=parent_run_id,
            raw_validation_sample=raw_validation_sample,
            manifest=final_preprocessor.manifest,
        )

        logistic_config = selected_configs["logistic_regression"]
        child_run_ids["logistic_regression"] = log_spark_model_run(
            parent_run_id=parent_run_id,
            comparator="logistic_regression",
            parameters={"config_name": logistic_config, **LOGISTIC_CONFIGS[logistic_config]},
            evaluations=evaluations["logistic_regression"],
            backtest=[
                asdict(score)
                for score in backtest_scores
                if score.model_name == "logistic_regression"
            ],
            model=logistic_model,
            validation_sample=logged_validation_sample,
        )

        lightgbm_config = selected_configs["lightgbm"]
        child_run_ids["lightgbm"] = log_lightgbm_run(
            parent_run_id=parent_run_id,
            comparator="lightgbm",
            parameters={
                "config_name": lightgbm_config,
                **LIGHTGBM_CONFIGS[lightgbm_config],
                "random_state": 42,
                "data_random_seed": 42,
                "objective": "binary",
            },
            evaluations=evaluations["lightgbm"],
            backtest=[
                asdict(score)
                for score in backtest_scores
                if score.model_name == "lightgbm"
            ],
            model=lightgbm_model,
            validation_sample=logged_validation_sample,
            vector_size=final_preprocessor.manifest.vector_size,
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
