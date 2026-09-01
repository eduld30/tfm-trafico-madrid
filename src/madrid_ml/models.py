"""Explicit Spark ML and SynapseML model configurations."""

from pyspark.ml.classification import LogisticRegression

SYNAPSEML_VERSION = "1.1.3"

LOGISTIC_CONFIGS = {
    "l2_0_01": {"regParam": 0.01, "elasticNetParam": 0.0, "maxIter": 100},
    "l2_0_1": {"regParam": 0.1, "elasticNetParam": 0.0, "maxIter": 100},
}
LIGHTGBM_CONFIGS = {
    "leaves_31": {"numLeaves": 31, "learningRate": 0.1, "numIterations": 100},
    "leaves_63": {"numLeaves": 63, "learningRate": 0.1, "numIterations": 100},
}


def build_logistic_regression(config_name: str) -> LogisticRegression:
    """Build one approved Spark logistic regression candidate."""
    parameters = LOGISTIC_CONFIGS[config_name]
    return LogisticRegression(
        featuresCol="features",
        labelCol="target_accident_next_hour",
        probabilityCol="probability",
        **parameters,
    )


def build_lightgbm(config_name: str):
    """Build one approved distributed SynapseML LightGBM candidate."""
    from synapse.ml.lightgbm import LightGBMClassifier

    parameters = LIGHTGBM_CONFIGS[config_name]
    return LightGBMClassifier(
        seed=42,
        dataRandomSeed=42,
        featuresCol="features",
        labelCol="target_accident_next_hour",
        probabilityCol="probability",
        objective="binary",
        executionMode="streaming",
        **parameters,
    )
