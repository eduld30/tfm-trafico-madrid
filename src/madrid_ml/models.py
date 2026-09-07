"""Explicit Spark ML and native LightGBM model configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import predict_batch_udf, vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from madrid_ml.preprocessing import MODEL_FEATURES_COLUMN

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier
    from numpy.typing import NDArray

LIGHTGBM_VERSION = "4.6.0"
LIGHTGBM_PREDICTION_BATCH_ROWS = 4096
TARGET_COLUMN = "target_accident_next_hour"

LOGISTIC_CONFIGS = {
    "l2_0_01": {"regParam": 0.01, "elasticNetParam": 0.0, "maxIter": 100},
    "l2_0_1": {"regParam": 0.1, "elasticNetParam": 0.0, "maxIter": 100},
}
LIGHTGBM_CONFIGS = {
    "leaves_31": {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100},
    "leaves_63": {"num_leaves": 63, "learning_rate": 0.1, "n_estimators": 100},
}


def build_logistic_regression(config_name: str) -> LogisticRegression:
    """Build one approved Spark logistic regression candidate."""
    parameters = LOGISTIC_CONFIGS[config_name]
    return LogisticRegression(
        featuresCol=MODEL_FEATURES_COLUMN,
        labelCol=TARGET_COLUMN,
        probabilityCol="probability",
        **parameters,
    )


def build_lightgbm(config_name: str) -> LGBMClassifier:
    """Build one approved native LightGBM candidate."""
    from lightgbm import LGBMClassifier

    parameters = LIGHTGBM_CONFIGS[config_name]
    return LGBMClassifier(
        objective="binary",
        random_state=42,
        data_random_seed=42,
        verbosity=-1,
        **parameters,
    )


def _feature_columns(vector_size: int) -> list:
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    features = vector_to_array(F.col(MODEL_FEATURES_COLUMN), dtype="float32")
    return [features[index].alias(f"feature_{index:03d}") for index in range(vector_size)]


def collect_lightgbm_features(frame: DataFrame, *, vector_size: int) -> NDArray:
    """Collect only a bounded model feature matrix into the Python process."""
    import numpy as np

    features = frame.select(*_feature_columns(vector_size)).toPandas()
    return features.to_numpy(dtype=np.float32, copy=False)


def fit_lightgbm(
    frame: DataFrame,
    *,
    config_name: str,
    vector_size: int,
    fold_name: str,
) -> LGBMClassifier:
    """Fit native LightGBM from the prepared Spark training split."""
    import numpy as np

    training = frame.select(
        *_feature_columns(vector_size),
        F.col(TARGET_COLUMN).cast("int").alias(TARGET_COLUMN),
    ).toPandas()
    labels = training.pop(TARGET_COLUMN).to_numpy(dtype=np.int32, copy=False)
    if labels.size == 0:
        raise RuntimeError(f"LightGBM training split is empty: fold_name={fold_name}")

    model = build_lightgbm(config_name)
    try:
        model.fit(training.to_numpy(dtype=np.float32, copy=False), labels)
    except Exception as exc:
        raise RuntimeError(
            "model fit failed: "
            f"model_name=lightgbm, config_name={config_name}, fold_name={fold_name}"
        ) from exc
    return model


def score_lightgbm(
    model: LGBMClassifier,
    frame: DataFrame,
    *,
    comparator: str,
    vector_size: int,
) -> DataFrame:
    """Score a Spark frame in executor-side batches without collecting its rows."""

    def make_predict_fn():
        def predict(features):
            return model.predict_proba(features)[:, 1]

        return predict

    predict = predict_batch_udf(
        make_predict_fn,
        return_type=DoubleType(),
        batch_size=LIGHTGBM_PREDICTION_BATCH_ROWS,
        input_tensor_shapes=[[vector_size]],
    )
    feature_array = vector_to_array(F.col(MODEL_FEATURES_COLUMN), dtype="float32")
    return frame.withColumn("score", predict(feature_array)).withColumn(
        "comparator", F.lit(comparator)
    )
