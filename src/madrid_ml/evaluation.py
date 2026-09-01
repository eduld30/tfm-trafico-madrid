"""Distributed binary model evaluation with deterministic tied-score handling."""

from __future__ import annotations

from typing import TypeAlias

from pyspark.sql import DataFrame, Row, Window
from pyspark.sql import functions as F

LABEL = "target_accident_next_hour"
SCORE = "score"
SEGMENTS = ("cod_distrito", "hora_dia")
MetricValue: TypeAlias = float | int | None


def _validate_scored(scored: DataFrame) -> Row:
    label = F.col(LABEL)
    score = F.col(SCORE)
    summary = scored.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(label.isNull() | ~label.isin(0, 1), 1).otherwise(0)).alias(
            "invalid_labels"
        ),
        F.sum(
            F.when(
                score.isNull() | F.isnan(score) | (score < 0.0) | (score > 1.0),
                1,
            ).otherwise(0)
        ).alias("invalid_scores"),
        F.sum(label.cast("long")).alias("positives"),
        F.avg(F.pow(score - label.cast("double"), 2)).alias("brier_score"),
    ).first()
    if summary.rows == 0:
        raise ValueError("cannot evaluate an empty scored DataFrame")
    if summary.invalid_labels:
        raise ValueError("target_accident_next_hour must contain only 0 or 1")
    if summary.invalid_scores:
        raise ValueError("score must contain only finite values in [0, 1]")
    return summary


def _grouped_scores(scored: DataFrame, segments: tuple[str, ...] = ()) -> DataFrame:
    return scored.groupBy(*segments, SCORE).agg(
        F.count(F.lit(1)).cast("long").alias("group_rows"),
        F.sum(F.col(LABEL)).cast("long").alias("group_positives"),
    ).withColumn(
        "group_negatives",
        F.col("group_rows") - F.col("group_positives"),
    )


def _ranked_score_groups(
    grouped: DataFrame,
    segments: tuple[str, ...] = (),
) -> DataFrame:
    partition = Window.partitionBy(*segments) if segments else Window.partitionBy()
    descending = partition.orderBy(F.col(SCORE).desc()).rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow,
    )
    ascending_before = partition.orderBy(F.col(SCORE).asc()).rowsBetween(
        Window.unboundedPreceding,
        -1,
    )
    complete = partition.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    return (
        grouped.withColumn("positives", F.sum("group_positives").over(complete))
        .withColumn("negatives", F.sum("group_negatives").over(complete))
        .withColumn("rows", F.sum("group_rows").over(complete))
        .withColumn("true_positives", F.sum("group_positives").over(descending))
        .withColumn("false_positives", F.sum("group_negatives").over(descending))
        .withColumn(
            "lower_negatives",
            F.coalesce(F.sum("group_negatives").over(ascending_before), F.lit(0)),
        )
        .withColumn(
            "average_precision_contribution",
            (F.col("group_positives") / F.col("positives"))
            * (
                F.col("true_positives")
                / (F.col("true_positives") + F.col("false_positives"))
            ),
        )
        .withColumn(
            "roc_auc_contribution",
            F.col("group_positives")
            * (F.col("lower_negatives") + 0.5 * F.col("group_negatives")),
        )
        .withColumn(
            "brier_sum",
            F.col("group_positives") * F.pow(1.0 - F.col(SCORE), 2)
            + F.col("group_negatives") * F.pow(F.col(SCORE), 2),
        )
    )


def global_binary_metrics(scored: DataFrame) -> dict[str, MetricValue]:
    """Return global binary metrics, grouping tied scores before ranking."""
    summary = _validate_scored(scored)
    positives = int(summary.positives)
    rows = int(summary.rows)
    negatives = rows - positives

    ranked = _ranked_score_groups(_grouped_scores(scored))
    ranking = ranked.agg(
        F.sum("average_precision_contribution").alias("average_precision"),
        F.sum("roc_auc_contribution").alias("roc_auc_numerator"),
    ).first()
    evaluable = positives > 0 and negatives > 0
    return {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "prevalence": positives / rows,
        "average_precision": (
            float(ranking.average_precision) if evaluable else None
        ),
        "brier_score": float(summary.brier_score),
        "roc_auc": (
            float(ranking.roc_auc_numerator) / (positives * negatives)
            if evaluable
            else None
        ),
    }


def segmented_binary_metrics(scored: DataFrame) -> DataFrame:
    """Return district/hour metrics while retaining non-evaluable segments."""
    _validate_scored(scored)
    ranked = _ranked_score_groups(_grouped_scores(scored, SEGMENTS), SEGMENTS)
    aggregated = ranked.groupBy(*SEGMENTS).agg(
        F.first("rows").cast("long").alias("rows"),
        F.first("positives").cast("long").alias("positives"),
        F.first("negatives").cast("long").alias("negatives"),
        F.sum("average_precision_contribution").alias("average_precision_raw"),
        F.sum("roc_auc_contribution").alias("roc_auc_numerator"),
        F.sum("brier_sum").alias("brier_sum"),
    )
    evaluable = (F.col("positives") > 0) & (F.col("negatives") > 0)
    return (
        aggregated.withColumn("prevalence", F.col("positives") / F.col("rows"))
        .withColumn(
            "average_precision",
            F.when(evaluable, F.col("average_precision_raw")),
        )
        .withColumn("brier_score", F.col("brier_sum") / F.col("rows"))
        .withColumn(
            "roc_auc",
            F.when(
                evaluable,
                F.col("roc_auc_numerator")
                / (F.col("positives") * F.col("negatives")),
            ),
        )
        .drop("average_precision_raw", "roc_auc_numerator", "brier_sum")
        .orderBy(*SEGMENTS)
    )


def curve_points(scored: DataFrame) -> DataFrame:
    """Return compact PR/ROC points at exact observed score percentiles."""
    summary = _validate_scored(scored)
    thresholds = sorted(
        set(scored.approxQuantile(SCORE, [value / 100 for value in range(101)], 0.0)),
        reverse=True,
    )
    aggregates = []
    for index, threshold in enumerate(thresholds):
        predicted_positive = F.col(SCORE) >= F.lit(threshold)
        aggregates.extend(
            (
                F.sum(
                    F.when(predicted_positive & (F.col(LABEL) == 1), 1).otherwise(0)
                ).alias(f"tp_{index}"),
                F.sum(
                    F.when(predicted_positive & (F.col(LABEL) == 0), 1).otherwise(0)
                ).alias(f"fp_{index}"),
            )
        )
    counts = scored.agg(*aggregates).first()
    positives = int(summary.positives)
    negatives = int(summary.rows) - positives
    rows = []
    for index, threshold in enumerate(thresholds):
        true_positives = int(counts[f"tp_{index}"])
        false_positives = int(counts[f"fp_{index}"])
        predicted = true_positives + false_positives
        rows.append(
            (
                float(threshold),
                true_positives,
                false_positives,
                true_positives / predicted if predicted else None,
                true_positives / positives if positives else None,
                false_positives / negatives if negatives else None,
            )
        )
    return scored.sparkSession.createDataFrame(
        rows,
        "threshold double, true_positives long, false_positives long, "
        "precision double, recall double, false_positive_rate double",
    )


def calibration_bins(scored: DataFrame) -> DataFrame:
    """Return ten equal-width calibration bins for observed scores."""
    _validate_scored(scored)
    calibration_bin = F.least(F.floor(F.col(SCORE) * 10), F.lit(9)).cast("int")
    return (
        scored.withColumn("calibration_bin", calibration_bin)
        .groupBy("calibration_bin")
        .agg(
            F.count(F.lit(1)).cast("long").alias("rows"),
            F.sum(F.col(LABEL)).cast("long").alias("positives"),
            F.avg(F.col(SCORE)).alias("mean_score"),
            F.avg(F.col(LABEL).cast("double")).alias("observed_frequency"),
        )
        .orderBy("calibration_bin")
    )
