"""Minimal MLflow boundary for model training runs and compact artifacts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from pyspark.ml import Model, PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from madrid_ml.preprocessing import FittedPreprocessor, PreprocessingManifest


@contextmanager
def start_training_run(
    *,
    experiment_id: str,
    run_name: str,
    tags: Mapping[str, str],
    manifest: Mapping[str, object],
    preprocessor: PipelineModel,
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
        mlflow.spark.log_model(preprocessor, artifact_path="preprocessor")
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


def log_baseline_run(
    *,
    parent_run_id: str,
    comparator: str,
    parameters: Mapping[str, object],
    evaluations: Mapping[str, object],
    state: Mapping[str, object],
) -> str:
    """Register one historical baseline as a child run."""
    import mlflow

    with mlflow.start_run(
        run_name=comparator,
        nested=True,
        tags={"mlflow.parentRunId": parent_run_id, "madrid_ml.comparator": comparator},
    ) as active_run:
        mlflow.log_params(dict(parameters))
        _log_global_metrics(evaluations)
        mlflow.log_dict(dict(evaluations), "evaluation.json")
        mlflow.log_dict(dict(state), "baseline_state.json")
        return active_run.info.run_id


def log_model_run(
    *,
    parent_run_id: str,
    comparator: str,
    parameters: Mapping[str, object],
    evaluations: Mapping[str, object],
    backtest: Sequence[Mapping[str, object]],
    model: Model,
) -> str:
    """Register one fitted classifier and its compact evaluation artifacts."""
    import mlflow

    with mlflow.start_run(
        run_name=comparator,
        nested=True,
        tags={"mlflow.parentRunId": parent_run_id, "madrid_ml.comparator": comparator},
    ) as active_run:
        mlflow.log_params(dict(parameters))
        _log_global_metrics(evaluations)
        mlflow.log_dict(dict(evaluations), "evaluation.json")
        mlflow.log_dict({"folds": list(backtest)}, "backtest.json")
        mlflow.spark.log_model(model, artifact_path="classifier")
        return active_run.info.run_id


def verify_logged_models(
    *,
    parent_run_id: str,
    model_run_ids: Mapping[str, str],
    raw_validation_sample: DataFrame,
    manifest: PreprocessingManifest,
) -> None:
    """Reload logged Spark artifacts and require 32 bounded model scores."""
    import mlflow

    loaded_pipeline = mlflow.spark.load_model(f"runs:/{parent_run_id}/preprocessor")
    if not isinstance(loaded_pipeline, PipelineModel):
        raise RuntimeError("reloaded preprocessor is not a Spark PipelineModel")
    prepared_sample = FittedPreprocessor(
        pipeline_model=loaded_pipeline,
        manifest=manifest,
    ).transform(raw_validation_sample, split_name="validation")

    for comparator, run_id in model_run_ids.items():
        model = mlflow.spark.load_model(f"runs:/{run_id}/classifier")
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
                f"reloaded {comparator} produced invalid validation scores: "
                f"rows={summary.rows}, invalid_scores={summary.invalid_scores}"
            )
