# Databricks notebook source
# MAGIC %md
# MAGIC # Asignación de distrito a estaciones de meteo y calidad del aire
# MAGIC
# MAGIC Job puntual, fuera del pipeline YAML declarativo (bronze/silver por
# MAGIC dataset): asigna distrito a las estaciones de `dim_meteo` y `dim_calair`
# MAGIC por contención geográfica (point-in-polygon) contra los límites reales
# MAGIC de los distritos de Madrid, ya que ninguna de las dos dimensiones trae
# MAGIC distrito de origen.
# MAGIC
# MAGIC Genera dos tablas Silver derivadas:
# MAGIC - `{env}_silver.meteo.dim_meteo_distrito`
# MAGIC - `{env}_silver.calair.dim_calair_distrito`
# MAGIC
# MAGIC ambas con esquema `(codigo_corto, distrito_cod, distrito_nombre)`, listas
# MAGIC para que `meteo_nrt`/`meteo_historico` y `calair_nrt`/`calair_historico`
# MAGIC las usen con un `lookup_join` normal, igual que `trafico_nrt` usa
# MAGIC `dim_trafico`.
# MAGIC
# MAGIC Se relanza solo cuando cambian las estaciones o el callejero de
# MAGIC distritos, no en cada ejecución de ingesta.

# COMMAND ----------

# MAGIC %pip install shapely pydantic

# COMMAND ----------

dbutils.widgets.text("environment", "dev", "Entorno")
dbutils.widgets.text(
    "repo_root",
    "/Workspace/Users/lissilva@ucm.es/tfm-trafico-madrid",
    "Raíz del repo en el workspace",
)
environment = dbutils.widgets.get("environment")
repo_root = dbutils.widgets.get("repo_root")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports y configuración de entorno

# COMMAND ----------

import re
import sys
import xml.etree.ElementTree as ET
from typing import NamedTuple

sys.path.append(f"{repo_root}/src")

from pyspark.sql import DataFrame, SparkSession
from shapely.geometry import Point, Polygon

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ingestion.core.naming import build_table_name
from madrid_ingestion.core.paths import build_adls_uri
from madrid_ingestion.writers.table_manager import TableManager

config_root = f"{repo_root}/conf/"
loader = ConfigLoader(config_root)
env_config = loader.load_environment(environment)
storage_account = env_config.storage.account_name
silver_catalog = env_config.catalogs.silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cargar los polígonos de distrito
# MAGIC
# MAGIC Geometría estática (no cambia salvo reorganización administrativa), se
# MAGIC trata igual que `DIM_MAGNITUDES_METEO`/`DIM_CALAIR_MAGNITUDES`: se sube
# MAGIC manualmente a `landing/geografia/distritos/distritos.kml` en vez de
# MAGIC pasar por un pipeline ADF. Fuente oficial:
# MAGIC https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/distritos.kml
# MAGIC
# MAGIC Los mismos atributos (`COD_DIS`, `NOMBRE`, `DISTRI_MAY`) que ya tenéis en
# MAGIC `trafico.dim_distritos`, así que `distrito_cod` es directamente
# MAGIC comparable con `dim_distritos.cod_dis`.

# COMMAND ----------

KML_LANDING_PATH = "geografia/distritos"
KML_FILENAME = "distritos.kml"
LOCAL_KML_PATH = "/tmp/distritos.kml"

kml_uri = build_adls_uri(
    storage_account, env_config.storage.containers.landing, KML_LANDING_PATH
) + KML_FILENAME
dbutils.fs.cp(kml_uri, f"file:{LOCAL_KML_PATH}")


class Distrito(NamedTuple):
    cod_dis: int
    nombre: str
    distri_may: str
    polygon: Polygon


_KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
_FIELD_PATTERN = r"<td>{name}</td>\s*<td>([^<]*)</td>"


def _extract_field(description: str, name: str) -> str | None:
    match = re.search(_FIELD_PATTERN.format(name=name), description)
    return match.group(1) if match else None


def parse_distritos_kml(path: str) -> list[Distrito]:
    """Lee el KML oficial de distritos y devuelve un polígono por distrito."""
    root = ET.parse(path).getroot()
    distritos = []
    for placemark in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        description = placemark.find("kml:description", _KML_NS)
        cdata = description.text if description is not None else ""
        cod_dis = _extract_field(cdata, "COD_DIS")
        nombre = _extract_field(cdata, "NOMBRE")
        distri_may = _extract_field(cdata, "DISTRI_MAY")
        if cod_dis is None or nombre is None or distri_may is None:
            raise ValueError(f"Placemark sin COD_DIS/NOMBRE/DISTRI_MAY en {path}.")

        coordinates = placemark.find(".//kml:coordinates", _KML_NS)
        if coordinates is None or not coordinates.text:
            raise ValueError(f"Placemark {nombre!r} sin <coordinates>.")
        points = []
        for triplet in coordinates.text.split():
            lon, lat, *_ = triplet.split(",")
            points.append((float(lon), float(lat)))
        distritos.append(Distrito(int(cod_dis), nombre, distri_may, Polygon(points)))
    return distritos


distritos = parse_distritos_kml(LOCAL_KML_PATH)
assert len(distritos) == 21, f"Se esperaban 21 distritos, se han leído {len(distritos)}."
distritos[:3]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Asignar distrito por contención (con fallback al más cercano)
# MAGIC
# MAGIC El número de estaciones es pequeño (decenas, no miles), así que se
# MAGIC resuelve en el driver con `shapely` en vez de una unión espacial
# MAGIC distribuida — no compensa la complejidad de Sedona para este volumen.
# MAGIC
# MAGIC Si una estación cae justo fuera de todos los polígonos (redondeo de
# MAGIC coordenadas cerca de un límite), se asigna al distrito más cercano en
# MAGIC vez de dejarla sin distrito.

# COMMAND ----------


def assign_distrito(distritos: list[Distrito], lon: float, lat: float) -> Distrito:
    point = Point(lon, lat)
    for distrito in distritos:
        if distrito.polygon.contains(point):
            return distrito
    return min(distritos, key=lambda distrito: distrito.polygon.distance(point))


def build_distrito_dimension(
    spark: SparkSession,
    distritos: list[Distrito],
    source_df: DataFrame,
    station_column: str,
    lon_column: str = "longitud",
    lat_column: str = "latitud",
) -> DataFrame:
    """Devuelve (station_column, distrito_cod, distrito_nombre) por estación."""
    stations = (
        source_df.select(station_column, lon_column, lat_column)
        .where(f"{lon_column} IS NOT NULL AND {lat_column} IS NOT NULL")
        .distinct()
        .collect()
    )
    if not stations:
        raise ValueError(f"{station_column}: no hay estaciones con coordenadas.")
    assigned = []
    for row in stations:
        distrito = assign_distrito(distritos, row[lon_column], row[lat_column])
        assigned.append((row[station_column], distrito.cod_dis, distrito.nombre))
    return spark.createDataFrame(
        assigned, schema=[station_column, "distrito_cod", "distrito_nombre"]
    )


def write_distrito_dimension(
    spark: SparkSession,
    df: DataFrame,
    source: str,
    dataset: str,
) -> None:
    target_table = build_table_name(silver_catalog, source, dataset)
    target_path = build_adls_uri(
        storage_account, env_config.storage.containers.silver, f"{source}/{dataset}"
    )
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        target_path
    )
    TableManager(spark).ensure_external_table(target_table, target_path)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Meteo: `dim_meteo` → `dim_meteo_distrito`
# MAGIC
# MAGIC Se cruza por `codigo_corto`, que es el código que usan `meteo_nrt`/
# MAGIC `meteo_historico` en su columna `estacion` (el campo `estacion` de
# MAGIC `dim_meteo` es el nombre descriptivo de la estación, no el código).

# COMMAND ----------

dim_meteo = spark.table(build_table_name(silver_catalog, "meteo", "dim_meteo"))
dim_meteo_distrito = build_distrito_dimension(
    spark, distritos, dim_meteo, station_column="codigo_corto"
)
write_distrito_dimension(spark, dim_meteo_distrito, "meteo", "dim_meteo_distrito")
display(dim_meteo_distrito)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Calidad del aire: `dim_calair` → `dim_calair_distrito`
# MAGIC
# MAGIC Mismo cruce por `codigo_corto` (coincide con `estacion`/`punto_muestreo`
# MAGIC en `calair_nrt`/`calair_historico`).

# COMMAND ----------

dim_calair = spark.table(build_table_name(silver_catalog, "calair", "dim_calair"))
dim_calair_distrito = build_distrito_dimension(
    spark, distritos, dim_calair, station_column="codigo_corto"
)
write_distrito_dimension(spark, dim_calair_distrito, "calair", "dim_calair_distrito")
display(dim_calair_distrito)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Comprobación
# MAGIC
# MAGIC Todas las estaciones deben tener distrito asignado (el fallback al más
# MAGIC cercano evita nulos); esto solo detecta que no haya quedado ninguna
# MAGIC estación fuera del `collect()`, no la precisión de la asignación.

# COMMAND ----------

assert dim_meteo_distrito.filter("distrito_cod IS NULL").count() == 0
assert dim_calair_distrito.filter("distrito_cod IS NULL").count() == 0
print("OK: todas las estaciones de meteo y calidad del aire tienen distrito asignado.")
