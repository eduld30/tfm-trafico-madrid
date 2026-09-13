"""Spark ML model definition used by the accident-risk workflow."""

from pyspark.ml.classification import LogisticRegression

from madrid_ml.preprocessing import MODEL_FEATURES_COLUMN

TARGET_COLUMN = "target_accident_next_hour"
LOGISTIC_CONFIG_NAME = "l2_0_1"
LOGISTIC_CONFIG = {"regParam": 0.1, "elasticNetParam": 0.0, "maxIter": 100}


def build_logistic_regression() -> LogisticRegression:
    """Build the single interpretable classifier approved for production scoring."""
    return LogisticRegression(
        featuresCol=MODEL_FEATURES_COLUMN,
        labelCol=TARGET_COLUMN,
        probabilityCol="probability",
        **LOGISTIC_CONFIG,
    )
