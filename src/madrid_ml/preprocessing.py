"""Versioned Spark preprocessing for the Gold ML training snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from madrid_ml.contract import AIR_MAGNITUDES, TRAFFIC_COLUMNS, WEATHER_MAGNITUDES

PREPROCESSING_CONTRACT_VERSION = "1"
SESSION_TIMEZONE = "Etc/UTC"
EXPECTED_VECTOR_SIZE = 111

_TRAFFIC_CONTINUOUS_FEATURES = tuple(
    name for name, _, _ in TRAFFIC_COLUMNS if name.endswith("_media")
)
_TRAFFIC_COUNT_FEATURES = tuple(
    name for name, _, _ in TRAFFIC_COLUMNS if name.endswith("_n")
)
CONTINUOUS_FEATURES = (
    _TRAFFIC_CONTINUOUS_FEATURES
    + tuple(magnitude.mean_column for magnitude in WEATHER_MAGNITUDES)
    + tuple(magnitude.mean_column for magnitude in AIR_MAGNITUDES)
)
COUNT_FEATURES = (
    _TRAFFIC_COUNT_FEATURES
    + tuple(magnitude.count_column for magnitude in WEATHER_MAGNITUDES)
    + tuple(magnitude.count_column for magnitude in AIR_MAGNITUDES)
)
NUMERIC_FEATURES = CONTINUOUS_FEATURES + COUNT_FEATURES
AVAILABLE_FEATURES = tuple(f"{name}_available" for name in CONTINUOUS_FEATURES)


class PreprocessingContractError(RuntimeError):
    """Raised when Gold data cannot satisfy the preprocessing contract."""


@dataclass(frozen=True)
class PhysicalRange:
    """Hard physical range for one continuous input."""

    minimum: float | None = None
    maximum: float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True


_NON_NEGATIVE_RANGE = PhysicalRange(minimum=0.0)
PHYSICAL_RANGES = {
    "trafico_intensidad_media": _NON_NEGATIVE_RANGE,
    "trafico_ocupacion_media": _NON_NEGATIVE_RANGE,
    "trafico_carga_media": _NON_NEGATIVE_RANGE,
    "trafico_vmed_media": _NON_NEGATIVE_RANGE,
    "meteo_velocidad_viento_media": _NON_NEGATIVE_RANGE,
    "meteo_temperatura_media": PhysicalRange(),
    "meteo_humedad_relativa_media": PhysicalRange(minimum=0.0, maximum=100.0),
    "meteo_presion_media": PhysicalRange(minimum=0.0, minimum_inclusive=False),
    "meteo_radiacion_solar_media": _NON_NEGATIVE_RANGE,
    "meteo_precipitacion_media": _NON_NEGATIVE_RANGE,
    "calair_so2_media": _NON_NEGATIVE_RANGE,
    "calair_co_media": _NON_NEGATIVE_RANGE,
    "calair_no_media": _NON_NEGATIVE_RANGE,
    "calair_no2_media": _NON_NEGATIVE_RANGE,
    "calair_pm25_media": _NON_NEGATIVE_RANGE,
    "calair_pm10_media": _NON_NEGATIVE_RANGE,
    "calair_nox_media": _NON_NEGATIVE_RANGE,
    "calair_o3_media": _NON_NEGATIVE_RANGE,
}
MAGNITUDE_COUNT_BY_MEAN = {
    magnitude.mean_column: magnitude.count_column
    for magnitude in (*WEATHER_MAGNITUDES, *AIR_MAGNITUDES)
}


def _finite(column_name: str) -> Column:
    value = F.col(column_name).cast("double")
    return (
        value.isNotNull()
        & ~F.isnan(value)
        & (value != F.lit(float("inf")))
        & (value != F.lit(float("-inf")))
    )


def _valid_value(column_name: str) -> Column:
    rule = PHYSICAL_RANGES[column_name]
    value = F.col(column_name).cast("double")
    valid = _finite(column_name)
    if rule.minimum is not None:
        valid = valid & (
            value >= F.lit(rule.minimum)
            if rule.minimum_inclusive
            else value > F.lit(rule.minimum)
        )
    if rule.maximum is not None:
        valid = valid & (
            value <= F.lit(rule.maximum)
            if rule.maximum_inclusive
            else value < F.lit(rule.maximum)
        )
    return valid


def sanitize_features(df: DataFrame) -> DataFrame:
    """Replace unusable continuous inputs with null and add availability flags."""
    sanitized = df
    for name in CONTINUOUS_FEATURES:
        valid = _valid_value(name)
        sanitized = sanitized.withColumn(
            f"{name}_available", valid.cast("double")
        ).withColumn(
            name,
            F.when(valid, F.col(name).cast("double")).otherwise(
                F.lit(None).cast("double")
            ),
        )
    return sanitized


def _count_matching(condition: Column, alias: str) -> Column:
    return F.coalesce(
        F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0))),
        F.lit(0),
    ).cast("long").alias(alias)


def validate_feature_domains(df: DataFrame) -> None:
    """Fail once with every invalid categorical, target, count, or coherence gate."""
    counters: list[Column] = []

    category_domains = {
        "cod_distrito": (1, 21),
        "hora_dia": (0, 23),
        "dia_semana": (1, 7),
        "mes": (1, 12),
        "target_accident_next_hour": (0, 1),
    }
    for name, (minimum, maximum) in category_domains.items():
        invalid = F.col(name).isNull() | ~F.col(name).between(minimum, maximum)
        counters.append(_count_matching(invalid, f"invalid__{name}"))

    for name in COUNT_FEATURES:
        invalid = F.col(name).isNull() | (F.col(name) < F.lit(0))
        counters.append(_count_matching(invalid, f"invalid__{name}"))

    for mean_name, count_name in MAGNITUDE_COUNT_BY_MEAN.items():
        inconsistent = (F.col(count_name) == F.lit(0)) & _finite(mean_name)
        counters.append(
            _count_matching(inconsistent, f"inconsistent__{mean_name}")
        )

    for mean_name in _TRAFFIC_CONTINUOUS_FEATURES:
        inconsistent = (F.col("trafico_puntos_n") == F.lit(0)) & _finite(
            mean_name
        )
        counters.append(
            _count_matching(inconsistent, f"inconsistent__{mean_name}")
        )

    observed = df.agg(*counters).first().asDict()
    failures = [
        f"{name}={count}"
        for name, count in observed.items()
        if count
    ]
    if failures:
        raise PreprocessingContractError(
            "Gold feature domain validation failed: " + ", ".join(failures)
        )


def profile_feature_quality(
    df: DataFrame,
) -> dict[str, dict[str, int]]:
    """Count raw missing/non-finite and finite out-of-range values per feature."""
    counters: list[Column] = []
    for name in CONTINUOUS_FEATURES:
        finite = _finite(name)
        counters.extend(
            (
                _count_matching(~finite, f"missing__{name}"),
                _count_matching(
                    finite & ~_valid_value(name),
                    f"invalid_range__{name}",
                ),
            )
        )

    observed = df.agg(*counters).first().asDict()
    return {
        name: {
            "missing_input_count": int(observed[f"missing__{name}"]),
            "invalid_range_count": int(observed[f"invalid_range__{name}"]),
        }
        for name in CONTINUOUS_FEATURES
    }
