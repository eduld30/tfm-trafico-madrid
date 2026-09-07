"""Bounded serverless integration smoke; never evaluates the Gold snapshot."""

from dataclasses import asdict
from datetime import datetime, timedelta
from importlib.metadata import version

from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession

from madrid_ml.baselines import fit_baselines, score_baselines
from madrid_ml.evaluation import global_binary_metrics
from madrid_ml.models import build_logistic_regression, fit_lightgbm, score_lightgbm
from madrid_ml.preprocessing import (
    CONTINUOUS_FEATURES,
    COUNT_FEATURES,
    SnapshotProvenance,
    fit_preprocessor,
)
from madrid_ml.tracking import (
    load_logged_preprocessor,
    log_baseline_run,
    log_lightgbm_run,
    log_spark_model_run,
    start_training_run,
)
from madrid_ml.training import (
    BASELINE_COMPARATORS,
    TrainingConfig,
    TrainingResult,
    _baseline_states,
    _compact_evaluation,
    _score_model,
)


def run_training_smoke(spark: SparkSession, config: TrainingConfig) -> TrainingResult:
    """Exercise real fit, distributed scoring, artifacts and normal task return."""
    import mlflow
    import numpy as np

    tied = spark.createDataFrame(
        [(1, 0.9), (0, 0.9), (1, 0.2), (0, 0.1)],
        "target_accident_next_hour int, score double",
    )
    metrics = global_binary_metrics(tied)
    np.testing.assert_allclose(
        [metrics["average_precision"], metrics["roc_auc"], metrics["brier_score"]],
        [7 / 12, 5 / 8, 0.3675],
    )
    negative = spark.createDataFrame(
        [(0, 0.2), (0, 0.4)], "target_accident_next_hour int, score double"
    )
    assert global_binary_metrics(negative)["average_precision"] is None

    # 168 covers all 21 districts, 24 hours, 7 weekdays and 12 months.
    rows = []
    for index in range(168):
        row = {
            "cod_distrito": index % 21 + 1,
            "feature_hour": datetime(2020, 1, 1) + timedelta(hours=index),
            "target_accident_next_hour": index % 2,
            "hora_dia": index % 24,
            "dia_semana": index % 7 + 1,
            "mes": index % 12 + 1,
        }
        row.update({name: float(index % 11 + 1) for name in CONTINUOUS_FEATURES})
        row.update({name: index % 3 + 1 for name in COUNT_FEATURES})
        rows.append(row)
    train = spark.createDataFrame(rows)
    raw_sample = spark.createDataFrame(rows[:32])
    provenance = SnapshotProvenance(
        labels_table="synthetic_smoke",
        features_table="synthetic_smoke",
        labels_delta_version=0,
        features_delta_version=0,
        snapshot_id="synthetic-smoke",
        input_versions_json="{}",
        code_commit=config.code_commit,
        feature_schema_version="1",
        time_contract="synthetic_smoke",
    )
    fitted = fit_preprocessor(
        train,
        provenance,
        split_rows={"train": 168, "validation": 32},
        spark_version=spark.version,
    )
    prepared_train = fitted.transform(train, split_name="train")
    sample = fitted.transform(raw_sample, split_name="validation")
    logistic = build_logistic_regression("l2_0_1").fit(prepared_train)
    lightgbm = fit_lightgbm(
        prepared_train,
        config_name="leaves_31",
        vector_size=111,
        fold_name="smoke",
    )
    children = {}
    with start_training_run(
        experiment_id=config.mlflow_experiment_id,
        run_name="serverless-runtime-smoke",
        tags={
            "madrid_ml.execution_kind": "smoke",
            "madrid_ml.code_commit": config.code_commit,
            "madrid_ml.package_version": version("madrid-ingestion"),
            "madrid_ml.lightgbm_version": version("lightgbm"),
            "madrid_ml.mlflow_version": mlflow.__version__,
            "madrid_ml.spark_version": spark.version,
        },
        manifest=asdict(fitted.manifest),
        preprocessor=fitted.pipeline_model,
        dfs_tmpdir=config.mlflow_dfs_tmp,
    ) as parent:
        loaded_sample = load_logged_preprocessor(
            parent_run_id=parent,
            raw_validation_sample=raw_sample,
            manifest=fitted.manifest,
            dfs_tmpdir=config.mlflow_dfs_tmp,
        )

        def vectors(frame):
            return np.array(
                [
                    row.features
                    for row in frame.orderBy("feature_hour")
                    .select(vector_to_array("features").alias("features"))
                    .collect()
                ]
            )

        np.testing.assert_allclose(vectors(sample), vectors(loaded_sample), rtol=1e-10)
        baseline = fit_baselines(train)
        states = _baseline_states(baseline)
        for key, scored in score_baselines(baseline, raw_sample).items():
            comparator = BASELINE_COMPARATORS[key]
            children[comparator] = log_baseline_run(
                experiment_id=config.mlflow_experiment_id,
                parent_run_id=parent,
                comparator=comparator,
                parameters={"execution_kind": "smoke"},
                evaluations={"validation": _compact_evaluation(scored)},
                state=states[comparator],
            )
        children["logistic_regression"] = log_spark_model_run(
            experiment_id=config.mlflow_experiment_id,
            parent_run_id=parent,
            comparator="logistic_regression",
            parameters={"execution_kind": "smoke"},
            evaluations={
                "validation": _compact_evaluation(
                    _score_model(logistic, sample, "logistic_regression")
                )
            },
            backtest=[],
            model=logistic,
            validation_sample=loaded_sample,
            dfs_tmpdir=config.mlflow_dfs_tmp,
        )
        children["lightgbm"] = log_lightgbm_run(
            experiment_id=config.mlflow_experiment_id,
            parent_run_id=parent,
            comparator="lightgbm",
            parameters={"execution_kind": "smoke"},
            evaluations={
                "validation": _compact_evaluation(
                    score_lightgbm(lightgbm, sample, comparator="lightgbm", vector_size=111)
                )
            },
            backtest=[],
            model=lightgbm,
            validation_sample=loaded_sample,
            vector_size=111,
        )
        reloaded_logistic = mlflow.spark.load_model(
            f"runs:/{children['logistic_regression']}/classifier",
            dfs_tmpdir=config.mlflow_dfs_tmp,
        )
        reloaded_lightgbm = mlflow.lightgbm.load_model(f"runs:/{children['lightgbm']}/classifier")

        def scores(frame):
            values = np.array(
                [row.score for row in frame.orderBy("feature_hour").select("score").collect()]
            )
            assert values.shape == (32,) and np.isfinite(values).all()
            return values

        np.testing.assert_allclose(
            scores(_score_model(logistic, sample, "logistic_regression")),
            scores(_score_model(reloaded_logistic, loaded_sample, "logistic_regression")),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            scores(score_lightgbm(lightgbm, sample, comparator="lightgbm", vector_size=111)),
            scores(
                score_lightgbm(
                    reloaded_lightgbm, loaded_sample, comparator="lightgbm", vector_size=111
                )
            ),
            rtol=1e-10,
        )
    assert mlflow.active_run() is None
    client = mlflow.tracking.MlflowClient()
    for run_id in [parent, *children.values()]:
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.info.experiment_id == config.mlflow_experiment_id
        if run_id != parent:
            assert run.data.tags["mlflow.parentRunId"] == parent
    return TrainingResult(
        status="SMOKE_COMPLETE",
        parent_run_id=parent,
        child_run_ids=children,
        selected_configs={},
        validation_metrics={},
        evaluation_metrics={},
    )
