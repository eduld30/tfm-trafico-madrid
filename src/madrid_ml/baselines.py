"""Historical risk baselines fitted with Spark aggregations."""

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

LABEL = "target_accident_next_hour"


@dataclass(frozen=True)
class BaselineState:
    """Train-only frequency state for the three historical comparators."""

    global_prevalence: float
    district_frequencies: DataFrame
    district_hour_day_frequencies: DataFrame


def fit_baselines(train: DataFrame) -> BaselineState:
    """Fit global, district, and district/hour/day frequencies on train."""
    prevalence = train.agg(F.avg(LABEL).alias("score")).first().score
    if prevalence is None:
        raise ValueError("cannot fit baselines on empty train data")

    by_district = train.groupBy("cod_distrito").agg(
        F.avg(LABEL).alias("district_score")
    )
    by_group = train.groupBy("cod_distrito", "hora_dia", "dia_semana").agg(
        F.avg(LABEL).alias("group_score")
    )
    return BaselineState(float(prevalence), by_district, by_group)


def score_baselines(
    state: BaselineState,
    frame: DataFrame,
) -> dict[str, DataFrame]:
    """Score all historical comparators with explicit unseen-group fallbacks."""
    global_score = F.lit(state.global_prevalence).cast("double")
    global_scored = (
        frame.withColumn("score", global_score)
        .withColumn("comparator", F.lit("global_prevalence"))
    )

    district_scored = (
        frame.join(state.district_frequencies, "cod_distrito", "left")
        .withColumn("score", F.coalesce(F.col("district_score"), global_score))
        .drop("district_score")
        .withColumn("comparator", F.lit("district_frequency"))
    )

    district_hour_day_scored = (
        frame.join(
            state.district_hour_day_frequencies,
            ["cod_distrito", "hora_dia", "dia_semana"],
            "left",
        )
        .join(state.district_frequencies, "cod_distrito", "left")
        .withColumn(
            "score",
            F.coalesce(F.col("group_score"), F.col("district_score"), global_score),
        )
        .drop("group_score", "district_score")
        .withColumn("comparator", F.lit("district_hour_day_frequency"))
    )

    return {
        "global": global_scored,
        "district": district_scored,
        "district_hour_day": district_hour_day_scored,
    }
