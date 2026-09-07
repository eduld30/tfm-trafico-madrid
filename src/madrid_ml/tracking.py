"""Minimal MLflow boundary for model training runs and compact artifacts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from pyspark.ml import Model, PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from madrid_ml.models import collect_lightgbm_features
from madrid_ml.preprocessing import FittedPreprocessor, PreprocessingManifest

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier


@contextmanager
def start_training_run(
    *,
    experiment_id: str,
    run_name: str,
    tags: Mapping[str, str],
    manifest: Mapping[str, object],
    preprocessor: PipelineModel,
    dfs_tmpdir: str,
) -> Iterator[str]:
    """Start the parent run and register the final preprocessing artifact."""
    import mlflow

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=dict(tags),
    ) as active_run:
        parent_run_id = active_run.info.run_id
        mlflow.log_dict(dict(manifest), "preprocessing_manifest.json")
        mlflow.spark.log_model(
            preprocessor,
            artifact_path="preprocessor",
            dfs_tmpdir=dfs_tmpdir,
        )
        yield parent_run_id


def _log_global_metrics(evaluations: Mapping[str, object]) -> None:
    import mlflow

    metrics: dict[str, float] = {}
    for period in ("validation", "evaluation"):
        period_evaluation = evaluations.get(period)
        if not isinstance(period_evaluation, Mapping):
            continue
        global_metrics = period_evaluation.get("global")
        if not isinstance(global_metrics, Mapping):
            continue
        for name, value in global_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"{period}_{name}"] = float(value)
    if metrics:
        mlflow.log_metrics(metrics)


def _log_model_metadata(
    *,
    parameters: Mapping[str, object],
    evaluations: Mapping[str, object],
    backtest: Sequence[Mapping[str, object]],
) -> None:
    import mlflow

    mlflow.log_params(dict(parameters))
    _log_global_metrics(evaluations)
    mlflow.log_dict(dict(evaluations), "evaluation.json")
    mlflow.log_dict({"folds": list(backtest)}, "backtest.json")


def log_baseline_run(
    *,
    parent_run_id: str,
    experiment_id: str,
    comparator: str,
    parameters: Mapping[str, object],
    evaluations: Mapping[str, object],
    state: Mapping[str, object],
) -> str:
    """Register one historical baseline as a child run."""
    import mlflow

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=comparator,
        nested=True,
        tags={"mlflow.parentRunId": parent_run_id, "madrid_ml.comparator": comparator},
    ) as active_run:
        mlflow.log_params(dict(parameters))
        _log_global_metrics(evaluations)
        mlflow.log_dict(dict(evaluations), "evaluation.json")
        mlflow.log_dict(dict(state), "baseline_state.json")
        return active_run.info.run_id


def log_spark_model_run(
    *,
    experiment_id: str,
    parent_run_id: str,
    comparator: str,
    parameters: Mapping[str, object],
    evaluations: Mapping[str, object],
    backtest: Sequence[Mapping[str, object]],
    model: Model,
    validation_sample: DataFrame,
    dfs_tmpdir: str,
) -> str:
    """Register, reload, and verify one fitted Spark classifier."""
    import mlflow

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=comparator,
        nested=True,
        tags={"mlflow.parentRunId": parent_run_id, "madrid_ml.comparator": comparator},
    ) as active_run:
        _log_model_metadata(
            parameters=parameters,
            evaluations=evaluations,
            backtest=backtest,
        )
        mlflow.spark.log_model(
            model,
            artifact_path="classifier",
            dfs_tmpdir=dfs_tmpdir,
        )
        logged_model = mlflow.spark.load_model(
            f"runs:/{active_run.info.run_id}/classifier",
            dfs_tmpdir=dfs_tmpdir,
        )
        _verify_spark_scores(logged_model, validation_sample)
        return active_run.info.run_id


def log_lightgbm_run(
    *,
    experiment_id: str,
    parent_run_id: str,
    comparator: str,
    parameters: Mapping[str, object],
    evaluations: Mapping[str, object],
    backtest: Sequence[Mapping[str, object]],
    model: LGBMClassifier,
    validation_sample: DataFrame,
    vector_size: int,
) -> str:
    """Register, reload, and verify one fitted native LightGBM classifier."""
    import mlflow

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=comparator,
        nested=True,
        tags={"mlflow.parentRunId": parent_run_id, "madrid_ml.comparator": comparator},
    ) as active_run:
        _log_model_metadata(
            parameters=parameters,
            evaluations=evaluations,
            backtest=backtest,
        )
        mlflow.lightgbm.log_model(model, artifact_path="classifier")
        logged_model = mlflow.lightgbm.load_model(
            f"runs:/{active_run.info.run_id}/classifier"
        )
        _verify_lightgbm_scores(
            logged_model,
            validation_sample,
            vector_size=vector_size,
        )
        return active_run.info.run_id


def _verify_spark_scores(model: Model, prepared_sample: DataFrame) -> None:
    score = vector_to_array(F.col("probability"))[1]
    summary = model.transform(prepared_sample).select(score.alias("score")).agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(
            F.when(
                F.col("score").isNull()
                | F.isnan("score")
                | (F.col("score") < 0.0)
                | (F.col("score") > 1.0),
                1,
            ).otherwise(0)
        ).alias("invalid_scores"),
    ).first()
    if summary.rows != 32 or summary.invalid_scores:
        raise RuntimeError(
            "reloaded logistic_regression produced invalid validation scores: "
            f"rows={summary.rows}, invalid_scores={summary.invalid_scores}"
        )


def _verify_lightgbm_scores(
    model: LGBMClassifier,
    prepared_sample: DataFrame,
    *,
    vector_size: int,
) -> None:
    import numpy as np

    features = collect_lightgbm_features(prepared_sample, vector_size=vector_size)
    scores = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    invalid_scores = int((~np.isfinite(scores) | (scores < 0.0) | (scores > 1.0)).sum())
    if scores.shape != (32,) or invalid_scores:
        raise RuntimeError(
            "reloaded lightgbm produced invalid validation scores: "
            f"rows={scores.size}, invalid_scores={invalid_scores}"
        )


def load_logged_preprocessor(
    *,
    parent_run_id: str,
    raw_validation_sample: DataFrame,
    manifest: PreprocessingManifest,
    dfs_tmpdir: str,
) -> DataFrame:
    """Reload the parent preprocessor and transform the validation sample."""
    import mlflow

    loaded_pipeline = mlflow.spark.load_model(
        f"runs:/{parent_run_id}/preprocessor",
        dfs_tmpdir=dfs_tmpdir,
    )
    if not isinstance(loaded_pipeline, PipelineModel):
        raise RuntimeError("reloaded preprocessor is not a Spark PipelineModel")
    return FittedPreprocessor(
        pipeline_model=loaded_pipeline,
        manifest=manifest,
    ).transform(raw_validation_sample, split_name="validation")
