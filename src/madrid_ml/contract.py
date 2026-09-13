from dataclasses import dataclass

FEATURE_SCHEMA_VERSION = "1"
TIME_CONTRACT = "source_wall_clock_as_stored_in_silver"
GRID_START = "2019-01-01 00:00:00"
GRID_END = "2026-06-30 22:00:00"
EXPECTED_DISTRICTS = 21
EXPECTED_HOURS = 65_711
EXPECTED_ROWS = 1_379_931
ML_SCHEMA = "ml"
LABEL_TABLE = "accident_labels_hourly"
FEATURE_TABLE = "features_training_snapshot"

INPUT_SUFFIXES = {
    "accidents": "accidentes.accidentes_historico",
    "districts": "trafico.dim_distritos",
    "traffic": "trafico.trafico_historico",
    "weather": "meteo.meteo_historico",
    "air": "calair.calair_historico",
}


@dataclass(frozen=True)
class Magnitude:
    code: int
    mean_column: str
    count_column: str


WEATHER_MAGNITUDES = (
    Magnitude(81, "meteo_velocidad_viento_media", "meteo_velocidad_viento_n"),
    Magnitude(83, "meteo_temperatura_media", "meteo_temperatura_n"),
    Magnitude(86, "meteo_humedad_relativa_media", "meteo_humedad_relativa_n"),
    Magnitude(87, "meteo_presion_media", "meteo_presion_n"),
    Magnitude(88, "meteo_radiacion_solar_media", "meteo_radiacion_solar_n"),
    Magnitude(89, "meteo_precipitacion_media", "meteo_precipitacion_n"),
)

AIR_MAGNITUDES = (
    Magnitude(1, "calair_so2_media", "calair_so2_n"),
    Magnitude(6, "calair_co_media", "calair_co_n"),
    Magnitude(7, "calair_no_media", "calair_no_n"),
    Magnitude(8, "calair_no2_media", "calair_no2_n"),
    Magnitude(9, "calair_pm25_media", "calair_pm25_n"),
    Magnitude(10, "calair_pm10_media", "calair_pm10_n"),
    Magnitude(12, "calair_nox_media", "calair_nox_n"),
    Magnitude(14, "calair_o3_media", "calair_o3_n"),
)

LABEL_BASE_COLUMNS = (
    ("cod_distrito", "INT", False),
    ("feature_hour", "TIMESTAMP", False),
    ("prediction_hour", "TIMESTAMP", False),
    ("n_accidentes_next_hour", "BIGINT", False),
    ("target_accident_next_hour", "INT", False),
)
CALENDAR_COLUMNS = (
    ("hora_dia", "INT", False),
    ("dia_semana", "INT", False),
    ("mes", "INT", False),
)
TRAFFIC_COLUMNS = (
    ("trafico_intensidad_media", "DOUBLE", True),
    ("trafico_ocupacion_media", "DOUBLE", True),
    ("trafico_carga_media", "DOUBLE", True),
    ("trafico_vmed_media", "DOUBLE", True),
    ("trafico_puntos_n", "BIGINT", False),
)
LINEAGE_COLUMNS = (
    ("snapshot_id", "STRING", False),
    ("input_versions_json", "STRING", False),
    ("code_commit", "STRING", False),
    ("feature_schema_version", "STRING", False),
    ("time_contract", "STRING", False),
)


def magnitude_columns(magnitudes: tuple[Magnitude, ...]) -> tuple[tuple[str, str, bool], ...]:
    return tuple(
        column
        for magnitude in magnitudes
        for column in (
            (magnitude.mean_column, "DOUBLE", True),
            (magnitude.count_column, "BIGINT", False),
        )
    )


LABEL_TABLE_COLUMNS = LABEL_BASE_COLUMNS + LINEAGE_COLUMNS
FEATURE_BASE_COLUMNS = (
    LABEL_BASE_COLUMNS
    + CALENDAR_COLUMNS
    + TRAFFIC_COLUMNS
    + magnitude_columns(WEATHER_MAGNITUDES)
    + magnitude_columns(AIR_MAGNITUDES)
)
FEATURE_TABLE_COLUMNS = FEATURE_BASE_COLUMNS + LINEAGE_COLUMNS
