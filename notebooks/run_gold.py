# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Ejecutar capa Gold
# MAGIC
# MAGIC Orquesta la construcción de la capa Gold a partir de las tablas Silver del
# MAGIC proyecto.
# MAGIC
# MAGIC ## Granularidades principales
# MAGIC
# MAGIC - Accidente: una fila por `num_expediente`.
# MAGIC - Accidentalidad: `cod_distrito + timestamp_hora`.
# MAGIC - Tráfico de detalle: `id_punto + timestamp_hora`.
# MAGIC - Tráfico agregado: `cod_distrito + timestamp_hora`.
# MAGIC - Meteorología/calidad del aire: `cod_distrito + timestamp_hora + magnitud`.
# MAGIC - Integración: `cod_distrito + timestamp_hora`.

# COMMAND ----------

# MAGIC %pip install pydantic

# COMMAND ----------

# Widgets
try:
    dbutils.widgets.text("environment", "dev", "Entorno")
    dbutils.widgets.text(
        "repo_root",
        "/Workspace/Users/carloaca@ucm.es/tfm-trafico-madrid",
        "Raíz del repositorio",
    )
    dbutils.widgets.text("gold_schema", "movilidad", "Esquema Gold")
    dbutils.widgets.dropdown(
        "write_mode", "overwrite", ["overwrite"], "Modo de escritura"
    )
    ENVIRONMENT = dbutils.widgets.get("environment")
    REPO_ROOT = dbutils.widgets.get("repo_root")
    GOLD_SCHEMA = dbutils.widgets.get("gold_schema")
    WRITE_MODE = dbutils.widgets.get("write_mode")
except Exception:
    ENVIRONMENT = "dev"
    REPO_ROOT = "/Workspace/Users/carloaca@ucm.es/tfm-trafico-madrid"
    GOLD_SCHEMA = "movilidad"
    WRITE_MODE = "overwrite"

# COMMAND ----------

import sys
from functools import reduce
from typing import Iterable

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

sys.path.append(f"{REPO_ROOT}/src")
from madrid_ingestion.config.loader import ConfigLoader

CONFIG_ROOT = f"{REPO_ROOT}/conf/"
ENV_CONFIG = ConfigLoader(CONFIG_ROOT).load_environment(ENVIRONMENT)
SILVER_CATALOG = ENV_CONFIG.catalogs.silver
GOLD_CATALOG = ENV_CONFIG.catalogs.gold
GOLD_DATABASE = f"{GOLD_CATALOG}.{GOLD_SCHEMA}"

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_DATABASE}")

print(f"Silver: {SILVER_CATALOG}")
print(f"Gold: {GOLD_DATABASE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utilidades de orquestación

# COMMAND ----------

results: dict[str, str] = {}
frames: dict[str, DataFrame] = {}


def silver_table(source: str, dataset: str) -> str:
    return f"{SILVER_CATALOG}.{source}.{dataset}"


def gold_table(name: str) -> str:
    return f"{GOLD_DATABASE}.{name}"


def table_exists(name: str) -> bool:
    return bool(spark.catalog.tableExists(name))


def read_silver(
    source: str,
    dataset: str,
    required: bool = False,
) -> DataFrame | None:
    name = silver_table(source, dataset)
    if not table_exists(name):
        message = f"no existe {name}"
        if required:
            raise RuntimeError(message)
        print(f"[SALTADO] {message}")
        return None
    return spark.table(name)


def write_gold(
    df: DataFrame,
    name: str,
    partition_by: list[str] | None = None,
) -> None:
    writer = (
        df.write.format("delta")
        .mode(WRITE_MODE)
        .option("overwriteSchema", "true")
    )
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.saveAsTable(gold_table(name))
    print(f"[OK] escrita {gold_table(name)}")


def mark(name: str, status: str) -> None:
    results[name] = status
    print(f"[{status}] {name}")


def has(df: DataFrame, name: str) -> bool:
    return name in df.columns


def col_or_null(
    df: DataFrame,
    names: str | list[str],
    dtype: str = "string",
    default=None,
):
    candidates = [names] if isinstance(names, str) else names
    existing = [F.col(name).cast(dtype) for name in candidates if has(df, name)]
    return F.coalesce(*existing) if existing else F.lit(default).cast(dtype)


def first_value(df: DataFrame, names: str | list[str], dtype: str = "string"):
    return col_or_null(df, names, dtype)


def normalize_time(
    df: DataFrame,
    timestamp_columns: list[str] | None = None,
) -> DataFrame:
    timestamp_columns = timestamp_columns or ["fecha_hora", "timestamp"]
    available = [name for name in timestamp_columns if has(df, name)]
    if available:
        ts = F.col(available[0]).cast("timestamp")
    elif has(df, "fecha"):
        ts = F.to_timestamp(
            F.concat_ws(" ", F.col("fecha").cast("string"), col_or_null(df, "hora"))
        )
    else:
        ts = F.lit(None).cast("timestamp")
    return (
        df.withColumn("ts", ts)
        .withColumn("timestamp_hora", F.date_trunc("hour", "ts"))
        .withColumn("fecha", F.to_date("ts"))
        .withColumn("hora", F.hour("ts"))
        .withColumn("anio", F.year("ts"))
        .withColumn("mes", F.month("ts"))
    )


def normalize_district(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "cod_distrito",
        col_or_null(df, ["cod_distrito", "distrito_cod", "cod_dis"], "long"),
    ).withColumn(
        "distrito_nombre",
        col_or_null(
            df,
            ["distrito_normalizado", "distrito_nombre", "distrito", "nombre"],
            "string",
        ),
    )


def add_time_labels(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("dia_semana_num", F.dayofweek("fecha"))
        .withColumn("dia_semana_nombre", F.date_format("fecha", "EEEE"))
        .withColumn(
            "franja_horaria",
            F.when(F.col("hora").between(0, 5), "Madrugada")
            .when(F.col("hora").between(6, 9), "Punta mañana")
            .when(F.col("hora").between(10, 15), "Mediodía")
            .when(F.col("hora").between(16, 19), "Punta tarde")
            .otherwise("Noche"),
        )
    )


def union_all(parts: list[DataFrame]) -> DataFrame:
    if not parts:
        raise RuntimeError("No hay DataFrames para unir")
    return reduce(lambda left, right: left.unionByName(right), parts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Lectura de tablas Silver
# MAGIC
# MAGIC Se utilizan los nombres y columnas observados en las tablas compartidas.
# MAGIC Las dimensiones de magnitudes se publican también en Gold para que Power BI
# MAGIC pueda interpretar códigos, unidades y técnicas de medida.

# COMMAND ----------

silver = {
    "dim_distritos": read_silver("trafico", "dim_distritos", required=True),
    "dim_trafico": read_silver("trafico", "dim_trafico", required=True),
    "dim_meteo": read_silver("meteo", "dim_meteo", required=True),
    "dim_meteo_magnitudes": read_silver("meteo", "dim_meteo_magnitudes", required=True),
    "dim_calair": read_silver("calair", "dim_calair", required=True),
    "dim_calair_magnitudes": read_silver("calair", "dim_calair_magnitudes", required=True),
    "accidentes_historico": read_silver("accidentes", "accidentes_historico", required=True),
    "trafico_historico": read_silver("trafico", "trafico_historico"),
    "trafico_nrt": read_silver("trafico", "trafico_nrt"),
    "meteo_historico": read_silver("meteo", "meteo_historico"),
    "meteo_nrt": read_silver("meteo", "meteo_nrt"),
    "calair_historico": read_silver("calair", "calair_historico"),
    "calair_nrt": read_silver("calair", "calair_nrt"),
    "eventos_culturales": read_silver("eventos", "eventos_culturales"),
}

for key, value in silver.items():
    if value is not None:
        print(f"{key}: {value.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Dimensiones de referencia

# COMMAND ----------

try:
    district = silver["dim_distritos"]
    gold_dim_distrito = district.select(
        col_or_null(district, ["cod_dis", "cod_distrito"], "long").alias("cod_distrito"),
        col_or_null(district, ["nombre", "distrito_nombre", "distri_may"], "string").alias("distrito_nombre"),
        col_or_null(district, "cod_dis_tx", "string").alias("distrito_codigo_texto"),
        col_or_null(district, "distri_mt", "string").alias("distrito_nombre_mayusculas"),
        col_or_null(district, "area", "double").alias("area"),
    ).where(F.col("cod_distrito").isNotNull()).dropDuplicates(["cod_distrito"])
    write_gold(gold_dim_distrito, "gold_dim_distrito")
    frames["dim_distrito"] = gold_dim_distrito
    mark("gold_dim_distrito", "OK")
except Exception as exc:
    mark("gold_dim_distrito", f"FALLO: {exc}")
    raise

# COMMAND ----------

try:
    traffic_dim = silver["dim_trafico"]
    gold_dim_trafico = traffic_dim.select(
        col_or_null(traffic_dim, "id", "long").alias("id_punto"),
        col_or_null(traffic_dim, "tipo_elem", "string").alias("tipo_elem"),
        col_or_null(traffic_dim, "distrito", "long").alias("cod_distrito"),
        col_or_null(traffic_dim, "nombre", "string").alias("punto_nombre"),
        col_or_null(traffic_dim, "cod_cent", "string").alias("codigo_centro"),
        col_or_null(traffic_dim, "utm_x", "double").alias("utm_x"),
        col_or_null(traffic_dim, "utm_y", "double").alias("utm_y"),
        col_or_null(traffic_dim, "longitud", "double").alias("longitud"),
        col_or_null(traffic_dim, "latitud", "double").alias("latitud"),
    ).where(F.col("id_punto").isNotNull()).dropDuplicates(["id_punto"])
    write_gold(gold_dim_trafico, "gold_dim_trafico")
    frames["dim_trafico"] = gold_dim_trafico
    mark("gold_dim_trafico", "OK")
except Exception as exc:
    mark("gold_dim_trafico", f"FALLO: {exc}")
    raise

# COMMAND ----------

try:
    meteo_dim = silver["dim_meteo"]
    gold_dim_meteo = meteo_dim.select(
        col_or_null(meteo_dim, "estacion", "string").alias("estacion"),
        col_or_null(meteo_dim, ["codigo", "codigo_corto"], "long").alias("codigo_estacion"),
        col_or_null(meteo_dim, "direccion", "string").alias("direccion"),
        col_or_null(meteo_dim, ["latitud", "latitud_etrs89"], "double").alias("latitud"),
        col_or_null(meteo_dim, ["longitud", "longitud_etrs89"], "double").alias("longitud"),
        col_or_null(meteo_dim, "altitud", "int").alias("altitud"),
        col_or_null(meteo_dim, "distrito_cod", "long").alias("cod_distrito"),
        col_or_null(meteo_dim, "distrito_nombre", "string").alias("distrito_nombre"),
    ).where(F.col("estacion").isNotNull()).dropDuplicates(["estacion"])
    write_gold(gold_dim_meteo, "gold_dim_meteo")
    frames["dim_meteo"] = gold_dim_meteo
    mark("gold_dim_meteo", "OK")
except Exception as exc:
    mark("gold_dim_meteo", f"FALLO: {exc}")
    raise

# COMMAND ----------

try:
    calair_dim = silver["dim_calair"]
    gold_dim_calair = calair_dim.select(
        col_or_null(calair_dim, ["codigo_corto", "codigo"], "long").alias("codigo_estacion"),
        col_or_null(calair_dim, "estacion", "string").alias("estacion"),
        col_or_null(calair_dim, "direccion", "string").alias("direccion"),
        col_or_null(calair_dim, ["latitud", "latitud_etrs89"], "double").alias("latitud"),
        col_or_null(calair_dim, ["longitud", "longitud_etrs89"], "double").alias("longitud"),
        col_or_null(calair_dim, "altitud", "int").alias("altitud"),
        col_or_null(calair_dim, "cod_tipo", "string").alias("cod_tipo"),
        col_or_null(calair_dim, "nom_tipo", "string").alias("nom_tipo"),
        col_or_null(calair_dim, "distrito_cod", "long").alias("cod_distrito"),
        col_or_null(calair_dim, "distrito_nombre", "string").alias("distrito_nombre"),
    ).where(F.col("estacion").isNotNull()).dropDuplicates(["estacion"])
    write_gold(gold_dim_calair, "gold_dim_calair")
    frames["dim_calair"] = gold_dim_calair
    mark("gold_dim_calair", "OK")
except Exception as exc:
    mark("gold_dim_calair", f"FALLO: {exc}")
    raise

# COMMAND ----------

try:
    meteo_mag = silver["dim_meteo_magnitudes"]
    gold_meteo_mag = meteo_mag.select(
        col_or_null(meteo_mag, "codigo", "int").alias("codigo_magnitud"),
        col_or_null(meteo_mag, "parametro", "string").alias("parametro"),
        col_or_null(meteo_mag, "unidad_de_medida", "string").alias("unidad"),
        col_or_null(meteo_mag, "tecnica_de_medida", "string").alias("tecnica_de_medida"),
    ).dropDuplicates(["codigo_magnitud"])
    write_gold(gold_meteo_mag, "gold_dim_meteo_magnitud")
    frames["dim_meteo_mag"] = gold_meteo_mag
    mark("gold_dim_meteo_magnitud", "OK")
except Exception as exc:
    mark("gold_dim_meteo_magnitud", f"FALLO: {exc}")
    raise

# COMMAND ----------

try:
    calair_mag = silver["dim_calair_magnitudes"]
    gold_calair_mag = calair_mag.select(
        col_or_null(calair_mag, "cod", "int").alias("codigo_magnitud"),
        col_or_null(calair_mag, "magnitud", "string").alias("magnitud"),
        col_or_null(calair_mag, "abrevia", "string").alias("abreviatura"),
        col_or_null(calair_mag, "unidad", "string").alias("unidad"),
        col_or_null(calair_mag, "codigo_tecnica", "string").alias("codigo_tecnica"),
        col_or_null(calair_mag, "tecnica_de_medida", "string").alias("tecnica_de_medida"),
    ).dropDuplicates(["codigo_magnitud"])
    write_gold(gold_calair_mag, "gold_dim_calair_magnitud")
    frames["dim_calair_mag"] = gold_calair_mag
    mark("gold_dim_calair_magnitud", "OK")
except Exception as exc:
    mark("gold_dim_calair_magnitud", f"FALLO: {exc}")
    raise

# COMMAND ----------

try:
    dim_hora = spark.range(24).select(F.col("id").cast("int").alias("hora")).withColumn(
        "franja_horaria",
        F.when(F.col("hora").between(0, 5), "Madrugada")
        .when(F.col("hora").between(6, 9), "Punta mañana")
        .when(F.col("hora").between(10, 15), "Mediodía")
        .when(F.col("hora").between(16, 19), "Punta tarde")
        .otherwise("Noche"),
    ).withColumn(
        "orden_franja",
        F.when(F.col("hora").between(0, 5), 1)
        .when(F.col("hora").between(6, 9), 2)
        .when(F.col("hora").between(10, 15), 3)
        .when(F.col("hora").between(16, 19), 4)
        .otherwise(5),
    )
    write_gold(dim_hora, "gold_dim_hora")
    frames["dim_hora"] = dim_hora
    mark("gold_dim_hora", "OK")
except Exception as exc:
    mark("gold_dim_hora", f"FALLO: {exc}")
    raise

# COMMAND ----------

try:
    dates = []
    for df in silver.values():
        if df is not None and has(df, "fecha_hora"):
            dates.append(df.select(F.to_date("fecha_hora").alias("fecha")))
    if not dates:
        raise RuntimeError("No se encontraron fechas para gold_dim_fecha")
    min_date, max_date = union_all(dates).select(F.min("fecha"), F.max("fecha")).first()
    dim_fecha = spark.sql(
        f"SELECT explode(sequence(to_date('{min_date}'), to_date('{max_date}'), interval 1 day)) AS fecha"
    ).select(
        "fecha", F.year("fecha").alias("anio"), F.quarter("fecha").alias("trimestre"),
        F.month("fecha").alias("mes"), F.date_format("fecha", "MMMM").alias("nombre_mes"),
        F.weekofyear("fecha").alias("semana_anio"), F.dayofweek("fecha").alias("dia_semana_num"),
        F.date_format("fecha", "EEEE").alias("dia_semana_nombre"),
        F.dayofweek("fecha").isin([1, 7]).alias("es_fin_de_semana"),
    )
    write_gold(dim_fecha, "gold_dim_fecha")
    frames["dim_fecha"] = dim_fecha
    mark("gold_dim_fecha", "OK")
except Exception as exc:
    mark("gold_dim_fecha", f"FALLO: {exc}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Accidentes

# COMMAND ----------

try:
    acc = normalize_district(normalize_time(silver["accidentes_historico"])).withColumn(
        "distrito_nombre",
        F.coalesce(F.col("distrito_nombre"), F.lit("Desconocido")),
    )
    lesividad = F.lower(F.coalesce(col_or_null(acc, "lesividad"), F.lit("")))
    people = acc.groupBy("num_expediente").agg(
        F.count("*").alias("n_personas_implicadas"),
        F.sum(F.when(col_or_null(acc, "lesividad").isNotNull() & (col_or_null(acc, "lesividad") != "Sin asistencia sanitaria"), 1).otherwise(0)).alias("n_personas_lesionadas"),
        F.sum(F.when(lesividad.contains("ingreso"), 1).otherwise(0)).alias("n_personas_hospitalizadas"),
        F.sum(F.when(lesividad.contains("fallecid"), 1).otherwise(0)).alias("n_personas_fallecidas"),
        F.max(F.when(F.upper(col_or_null(acc, "positiva_alcohol")) == "S", 1).otherwise(0)).alias("hay_alcohol"),
        F.max(F.when(col_or_null(acc, "positiva_droga", "double") == 1, 1).otherwise(0)).alias("hay_drogas"),
    )
    window = Window.partitionBy("num_expediente").orderBy(
        F.col("_silver_processed_timestamp").desc_nulls_last(),
        F.col("ts").asc_nulls_last(),
    )
    base = acc.withColumn("_rn", F.row_number().over(window)).where(F.col("_rn") == 1).drop("_rn")
    gold_accidente = base.join(people, "num_expediente", "left").select(
        "num_expediente", "ts", "timestamp_hora", "fecha", "hora", "anio", "mes", "cod_distrito", "distrito_nombre",
        col_or_null(base, "localizacion").alias("localizacion"), col_or_null(base, "numero").alias("numero"),
        col_or_null(base, "tipo_accidente").alias("tipo_accidente"), col_or_null(base, "estado_meteorologico").alias("estado_meteorologico"),
        col_or_null(base, "tipo_vehiculo").alias("tipo_vehiculo"), col_or_null(base, "tipo_persona").alias("tipo_persona"),
        col_or_null(base, "rango_edad").alias("rango_edad"), col_or_null(base, "sexo").alias("sexo"),
        col_or_null(base, "cod_lesividad", "long").alias("cod_lesividad"), col_or_null(base, "lesividad").alias("lesividad"),
        col_or_null(base, "coordenada_x_utm", "double").alias("coordenada_x_utm"), col_or_null(base, "coordenada_y_utm", "double").alias("coordenada_y_utm"),
        "n_personas_implicadas", "n_personas_lesionadas", "n_personas_hospitalizadas", "n_personas_fallecidas", "hay_alcohol", "hay_drogas",
    ).withColumn(
        "nivel_gravedad",
        F.when(F.col("n_personas_fallecidas") > 0, "Fatal")
        .when(F.col("n_personas_hospitalizadas") > 0, "Grave")
        .when(F.col("n_personas_lesionadas") > 0, "Lesivo")
        .otherwise("Sin asistencia / desconocido"),
    )
    write_gold(gold_accidente, "gold_accidente", ["anio"])
    frames["accidente"] = gold_accidente
    mark("gold_accidente", "OK")
except Exception as exc:
    mark("gold_accidente", f"FALLO: {exc}")
    raise

# COMMAND ----------

try:
    a = frames["accidente"]
    acc_agg = a.groupBy("cod_distrito", "distrito_nombre", "timestamp_hora", "fecha", "hora", "anio", "mes").agg(
        F.countDistinct("num_expediente").alias("n_accidentes"),
        F.sum(F.when(F.col("n_personas_lesionadas") > 0, 1).otherwise(0)).alias("n_accidentes_lesivos"),
        F.sum(F.when(F.col("n_personas_hospitalizadas") > 0, 1).otherwise(0)).alias("n_accidentes_graves"),
        F.sum(F.when(F.col("n_personas_fallecidas") > 0, 1).otherwise(0)).alias("n_accidentes_fatales"),
        F.sum("n_personas_implicadas").alias("n_personas_implicadas"), F.sum("n_personas_lesionadas").alias("n_personas_lesionadas"),
        F.sum("n_personas_hospitalizadas").alias("n_personas_hospitalizadas"), F.sum("n_personas_fallecidas").alias("n_personas_fallecidas"),
        F.sum("hay_alcohol").alias("n_accidentes_alcohol"), F.sum("hay_drogas").alias("n_accidentes_drogas"),
    )
    districts = a.select("cod_distrito", "distrito_nombre").distinct()
    min_ts, max_ts = a.select(F.min("timestamp_hora"), F.max("timestamp_hora")).first()
    hours = spark.sql(f"SELECT explode(sequence(timestamp('{min_ts}'), timestamp('{max_ts}'), interval 1 hour)) AS timestamp_hora")
    calendar = districts.crossJoin(hours).withColumn("fecha", F.to_date("timestamp_hora")).withColumn("hora", F.hour("timestamp_hora")).withColumn("anio", F.year("timestamp_hora")).withColumn("mes", F.month("timestamp_hora"))
    numeric = ["n_accidentes", "n_accidentes_lesivos", "n_accidentes_graves", "n_accidentes_fatales", "n_personas_implicadas", "n_personas_lesionadas", "n_personas_hospitalizadas", "n_personas_fallecidas", "n_accidentes_alcohol", "n_accidentes_drogas"]
    acc_dh = calendar.join(acc_agg, ["cod_distrito", "distrito_nombre", "timestamp_hora", "fecha", "hora", "anio", "mes"], "left").fillna(0, numeric)
    acc_dh = add_time_labels(acc_dh)
    write_gold(acc_dh, "gold_accidentes_distrito_hora", ["anio"])
    frames["acc_dh"] = acc_dh
    mark("gold_accidentes_distrito_hora", "OK")
except Exception as exc:
    mark("gold_accidentes_distrito_hora", f"FALLO: {exc}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tráfico histórico y NRT
# MAGIC
# MAGIC `trafico_historico` utiliza `id`, `latitud` y `longitud`. `trafico_nrt`
# MAGIC utiliza `idelem`, `st_x`, `st_y`, `velocidad`, `intensidad_sat` y
# MAGIC `nivel_servicio`. Se homogeneizan ambos contratos sin perder las métricas
# MAGIC específicas de NRT.

# COMMAND ----------

traffic_point = None
try:
    parts = []
    for key in ["trafico_historico", "trafico_nrt"]:
        df = silver.get(key)
        if df is None:
            continue
        d = normalize_district(normalize_time(df))
        # Si la fuente tiene distrito pero no nombre, se completa con dim_distritos.
        if has(d, "cod_distrito"):
            district_lookup = frames["dim_distrito"].select("cod_distrito", "distrito_nombre").dropDuplicates(["cod_distrito"])
            d = d.drop("distrito_nombre").join(district_lookup, "cod_distrito", "left") if has(d, "distrito_nombre") else d.join(district_lookup, "cod_distrito", "left")
        parts.append(d.select(
            first_value(d, ["id", "idelem"], "long").alias("id_punto"),
            first_value(d, "tipo_elem").alias("tipo_elem"), "cod_distrito", "distrito_nombre", "timestamp_hora", "fecha", "hora", "anio", "mes",
            first_value(d, ["latitud", "st_y"], "double").alias("latitud"), first_value(d, ["longitud", "st_x"], "double").alias("longitud"),
            first_value(d, "intensidad", "double").alias("intensidad"), first_value(d, "intensidad_sat", "double").alias("intensidad_saturada"),
            first_value(d, "ocupacion", "double").alias("ocupacion"), first_value(d, "carga", "double").alias("carga"),
            first_value(d, ["vmed", "velocidad"], "double").alias("velocidad"), first_value(d, "nivel_servicio", "int").alias("nivel_servicio"),
            first_value(d, "subarea").alias("subarea"), first_value(d, "error").alias("error"), first_value(d, "periodo_integracion", "int").alias("periodo_integracion"),
        ))
    if not parts:
        raise RuntimeError("No existe tráfico histórico ni NRT")
    traffic = union_all(parts).withColumn("velocidad_valida", F.when(F.col("velocidad") > 0, F.col("velocidad")))
    traffic_point = traffic.groupBy("id_punto", "tipo_elem", "cod_distrito", "distrito_nombre", "timestamp_hora", "fecha", "hora", "anio", "mes", "latitud", "longitud", "subarea").agg(
        F.avg("intensidad").alias("intensidad_media"), F.max("intensidad").alias("intensidad_maxima"), F.avg("intensidad_saturada").alias("intensidad_saturada_media"),
        F.avg("ocupacion").alias("ocupacion_media"), F.max("ocupacion").alias("ocupacion_maxima"), F.avg("carga").alias("carga_media"),
        F.avg("velocidad_valida").alias("velocidad_media"), F.max("nivel_servicio").alias("nivel_servicio_maximo"), F.count("*").alias("n_observaciones"),
        F.sum(F.when(F.col("error").isNull() | (F.col("error") == "N"), 1).otherwise(0)).alias("n_observaciones_validas"),
        F.first("periodo_integracion", ignorenulls=True).alias("periodo_integracion"),
    ).withColumn("porcentaje_registros_validos", F.col("n_observaciones_validas") / F.col("n_observaciones"))
    write_gold(traffic_point, "gold_trafico_punto_hora", ["anio"])
    frames["traffic_point"] = traffic_point
    traffic_district = traffic_point.groupBy("cod_distrito", "distrito_nombre", "timestamp_hora", "fecha", "hora", "anio", "mes").agg(
        F.countDistinct("id_punto").alias("n_puntos_medida"), F.avg("intensidad_media").alias("intensidad_media"), F.max("intensidad_maxima").alias("intensidad_maxima"),
        F.avg("intensidad_saturada_media").alias("intensidad_saturada_media"), F.avg("ocupacion_media").alias("ocupacion_media"), F.max("ocupacion_maxima").alias("ocupacion_maxima"),
        F.avg("carga_media").alias("carga_media"), F.avg("velocidad_media").alias("velocidad_media"), F.max("nivel_servicio_maximo").alias("nivel_servicio_maximo"),
        F.sum("n_observaciones").alias("n_observaciones_trafico"), F.sum("n_observaciones_validas").alias("n_observaciones_validas"),
    ).withColumn("porcentaje_registros_validos", F.col("n_observaciones_validas") / F.col("n_observaciones_trafico")).withColumn("nivel_congestion", F.lit(None).cast("string"))
    traffic_district = add_time_labels(traffic_district)
    write_gold(traffic_district, "gold_trafico_distrito_hora", ["anio"])
    frames["traffic_district"] = traffic_district
    mark("gold_trafico_punto_hora", "OK")
    mark("gold_trafico_distrito_hora", "OK")
except Exception as exc:
    mark("gold_trafico", f"FALLO: {exc}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Meteorología y calidad del aire
# MAGIC
# MAGIC Se conserva una tabla larga por magnitud y se enriquece mediante las
# MAGIC dimensiones de estación y magnitudes. Esto evita fijar nombres de columnas
# MAGIC dinámicas y permite incorporar nuevas magnitudes sin cambiar el esquema Gold.

# COMMAND ----------


def build_environment(
    source_name: str,
    keys: list[str],
    station_dim: DataFrame,
    magnitude_dim: DataFrame,
    output_name: str,
) -> DataFrame | None:
    parts = []
    for key in keys:
        df = silver.get(key)
        if df is None:
            continue
        d = normalize_district(normalize_time(df))
        station_col = "estacion" if has(d, "estacion") else "punto_muestreo"
        d = d.withColumn("estacion", F.col(station_col).cast("string"))
        d = d.withColumn("codigo_magnitud", col_or_null(d, "magnitud", "int"))
        d = d.withColumn("valor", col_or_null(d, "valor", "double"))
        d = d.withColumn("validez", col_or_null(d, "validez", "string"))
        parts.append(d.select("estacion", "codigo_magnitud", "valor", "validez", "timestamp_hora", "fecha", "hora", "anio", "mes", "cod_distrito", "distrito_nombre"))
    if not parts:
        return None
    data = union_all(parts)
    station = station_dim.select("estacion", "cod_distrito", "distrito_nombre", "latitud", "longitud").dropDuplicates(["estacion"])
    data = data.drop("cod_distrito", "distrito_nombre").join(station, "estacion", "left")
    magnitude = magnitude_dim.select("codigo_magnitud", *[name for name in ["parametro", "magnitud", "abreviatura", "unidad", "tecnica_de_medida"] if name in magnitude_dim.columns]).dropDuplicates(["codigo_magnitud"])
    data = data.join(magnitude, "codigo_magnitud", "left")
    result = data.groupBy("cod_distrito", "distrito_nombre", "estacion", "latitud", "longitud", "timestamp_hora", "fecha", "hora", "anio", "mes", "codigo_magnitud", *[name for name in ["parametro", "magnitud", "abreviatura", "unidad", "tecnica_de_medida"] if name in data.columns]).agg(
        F.avg(F.when(F.upper(F.col("validez")).isin("V", "VALIDO", "VALID"), F.col("valor")).otherwise(F.col("valor"))).alias("valor"),
        F.count("*").alias("n_observaciones"),
    )
    result = add_time_labels(result)
    write_gold(result, output_name, ["anio"])
    return result

meteo = build_environment("meteo", ["meteo_historico", "meteo_nrt"], frames["dim_meteo"], frames["dim_meteo_mag"], "gold_meteo_distrito_hora")
calair = build_environment("calair", ["calair_historico", "calair_nrt"], frames["dim_calair"], frames["dim_calair_mag"], "gold_calair_distrito_hora")
for key, name, value in [("meteo", "gold_meteo_distrito_hora", meteo), ("calair", "gold_calair_distrito_hora", calair)]:
    if value is None:
        mark(name, "SALTADO: sin fuente Silver")
    else:
        frames[key] = value
        mark(name, "OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Eventos

# COMMAND ----------

try:
    events = silver.get("eventos_culturales")
    if events is None:
        mark("gold_eventos_distrito_hora", "SALTADO: sin fuente Silver")
    else:
        events = normalize_district(normalize_time(events, ["fecha_hora", "timestamp"]))
        event_dh = events.where(F.col("timestamp_hora").isNotNull()).groupBy("cod_distrito", "distrito_nombre", "timestamp_hora", "fecha", "hora", "anio", "mes").agg(F.count("*").alias("n_eventos"))
        event_dh = add_time_labels(event_dh)
        write_gold(event_dh, "gold_eventos_distrito_hora", ["anio"])
        frames["events"] = event_dh
        mark("gold_eventos_distrito_hora", "OK")
except Exception as exc:
    mark("gold_eventos_distrito_hora", f"FALLO: {exc}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tabla integrada para analistas y Power BI

# COMMAND ----------

try:
    a = frames["acc_dh"].alias("a")
    t = frames["traffic_district"].alias("t")
    mobility = a.join(t, ["cod_distrito", "timestamp_hora"], "full").select(
        F.coalesce(F.col("a.cod_distrito"), F.col("t.cod_distrito")).alias("cod_distrito"),
        F.coalesce(F.col("a.distrito_nombre"), F.col("t.distrito_nombre")).alias("distrito_nombre"),
        F.coalesce(F.col("a.timestamp_hora"), F.col("t.timestamp_hora")).alias("timestamp_hora"),
        F.coalesce(F.col("a.fecha"), F.col("t.fecha")).alias("fecha"), F.coalesce(F.col("a.hora"), F.col("t.hora")).alias("hora"), F.coalesce(F.col("a.anio"), F.col("t.anio")).alias("anio"), F.coalesce(F.col("a.mes"), F.col("t.mes")).alias("mes"),
        *[F.col(f"a.{x}") for x in ["n_accidentes", "n_accidentes_lesivos", "n_accidentes_graves", "n_accidentes_fatales", "n_personas_implicadas", "n_personas_lesionadas", "n_personas_hospitalizadas", "n_personas_fallecidas", "n_accidentes_alcohol", "n_accidentes_drogas"]],
        *[F.col(f"t.{x}") for x in ["n_puntos_medida", "intensidad_media", "intensidad_maxima", "intensidad_saturada_media", "ocupacion_media", "ocupacion_maxima", "carga_media", "velocidad_media", "nivel_servicio_maximo", "n_observaciones_trafico", "n_observaciones_validas", "porcentaje_registros_validos"]],
        F.when(F.col("a.cod_distrito").isNotNull(), 1).otherwise(0).alias("hay_datos_accidentes"),
        F.when(F.col("t.cod_distrito").isNotNull(), 1).otherwise(0).alias("hay_datos_trafico"),
    )
    for key in ["meteo", "calair"]:
        if key in frames:
            env = frames[key]
            pivot_cols = [x for x in ["magnitud", "abreviatura", "parametro"] if x in env.columns]
            pivot_col = pivot_cols[0] if pivot_cols else "codigo_magnitud"
            env_wide = env.groupBy("cod_distrito", "timestamp_hora").pivot(pivot_col).agg(F.first("valor"))
            mobility = mobility.join(env_wide, ["cod_distrito", "timestamp_hora"], "left")
    if "events" in frames:
        mobility = mobility.join(frames["events"].select("cod_distrito", "timestamp_hora", "n_eventos"), ["cod_distrito", "timestamp_hora"], "left")
    mobility = mobility.withColumn("n_eventos", F.coalesce(F.col("n_eventos"), F.lit(0))) if has(mobility, "n_eventos") else mobility.withColumn("n_eventos", F.lit(0).cast("long"))
    mobility = add_time_labels(mobility)
    write_gold(mobility, "gold_movilidad_distrito_hora", ["anio"])
    frames["mobility"] = mobility
    mark("gold_movilidad_distrito_hora", "OK")
except Exception as exc:
    mark("gold_movilidad_distrito_hora", f"FALLO: {exc}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validaciones de la capa Gold

# COMMAND ----------

try:
    checks = {
        "gold_accidente expediente único": frames["accidente"].groupBy("num_expediente").count().where("count > 1").limit(1).count(),
        "gold_accidentes_distrito_hora única": frames["acc_dh"].groupBy("cod_distrito", "timestamp_hora").count().where("count > 1").limit(1).count(),
        "gold_trafico_punto_hora única": frames["traffic_point"].groupBy("id_punto", "timestamp_hora").count().where("count > 1").limit(1).count(),
        "gold_trafico_distrito_hora única": frames["traffic_district"].groupBy("cod_distrito", "timestamp_hora").count().where("count > 1").limit(1).count(),
        "gold_movilidad_distrito_hora única": frames["mobility"].groupBy("cod_distrito", "timestamp_hora").count().where("count > 1").limit(1).count(),
    }
    detail_count = frames["accidente"].select("num_expediente").distinct().count()
    aggregate_count = frames["acc_dh"].select(F.sum("n_accidentes")).first()[0]
    if detail_count != aggregate_count:
        raise RuntimeError(f"Total accidentes inconsistente: {detail_count} != {aggregate_count}")
    invalid = {name: value for name, value in checks.items() if value != 0}
    if invalid:
        raise RuntimeError(f"Claves Gold duplicadas: {invalid}")
    mark("validaciones_gold", "OK")
except Exception as exc:
    mark("validaciones_gold", f"FALLO: {exc}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Resumen

# COMMAND ----------

for name, status in results.items():
    print(f"{name}: {status}")

failed = {name: status for name, status in results.items() if status.startswith("FALLO")}
if failed:
    raise RuntimeError(f"Fallaron {len(failed)} salidas Gold: {failed}")

print("Ejecución Gold finalizada correctamente")
