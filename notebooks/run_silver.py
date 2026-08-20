# Databricks notebook source
# MAGIC %md
# MAGIC # Ejecutar Silver
# MAGIC
# MAGIC Orquesta las 12 tablas Silver operativas (quedan fuera los catálogos
# MAGIC estáticos de magnitudes, que se cargan de forma puntual) en el orden
# MAGIC correcto:
# MAGIC
# MAGIC 1. Dimensiones base: `dim_trafico`, `dim_distritos`, `dim_meteo`, `dim_calair`.
# MAGIC    Las dimensiones de estaciones reciben su distrito durante su propia
# MAGIC    transformación Silver usando el KML estático incluido en el paquete.
# MAGIC 2. Hechos que dependen de las anteriores vía `lookup_join`:
# MAGIC    `trafico_historico`, `trafico_nrt`, `accidentes_historico`,
# MAGIC    `eventos_culturales`, `meteo_nrt`, `meteo_historico`, `calair_nrt`,
# MAGIC    `calair_historico`.
# MAGIC
# MAGIC Antes de ejecutar cada dataset comprueba que su tabla Bronze exista — si
# MAGIC no, lo salta con un aviso en vez de fallar todo el notebook. Igual con
# MAGIC las dependencias de Silver: si `dim_trafico` no se pudo procesar,
# MAGIC `trafico_nrt`/`trafico_historico` se saltan también.
# MAGIC
# MAGIC ## Antes de la primera ejecución
# MAGIC 1. Bronze debe estar corrido (al menos para las fuentes que quieras
# MAGIC    procesar) — este notebook no ingesta desde landing.

# COMMAND ----------

# MAGIC %pip install shapely pydantic

# COMMAND ----------

dbutils.widgets.text("environment", "dev", "Entorno")
dbutils.widgets.text(
    "repo_root",
    "/Workspace/Users/lissilva@ucm.es/tfm-trafico-madrid",
    "Raíz del repo en el workspace",
)
ENVIRONMENT = dbutils.widgets.get("environment")
REPO_ROOT = dbutils.widgets.get("repo_root")
CONFIG_ROOT = f"{REPO_ROOT}/conf/"

# COMMAND ----------

import sys

sys.path.append(f"{REPO_ROOT}/src")

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ingestion.core.naming import build_table_name
from madrid_ingestion.runner import run_dataset

env_config = ConfigLoader(CONFIG_ROOT).load_environment(ENVIRONMENT)
BRONZE_CATALOG = env_config.catalogs.bronze

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utilidades de orquestación

# COMMAND ----------

# resultado por dataset: "OK", "SALTADO (<motivo>)" o "FALLO: <error>"
results: dict[tuple[str, str], str] = {}


def bronze_exists(source: str, dataset: str) -> bool:
    table = build_table_name(BRONZE_CATALOG, source, dataset)
    return bool(spark.catalog.tableExists(table))


def run_silver(source: str, dataset: str, requires: list[tuple[str, str]] | None = None) -> None:
    key = (source, dataset)
    for dependency in requires or []:
        if results.get(dependency) != "OK":
            results[key] = f"SALTADO (depende de {dependency[0]}.{dependency[1]})"
            print(f"[SALTADO] {source}.{dataset}: {results[key]}")
            return
    if not bronze_exists(source, dataset):
        results[key] = "SALTADO (no existe la tabla Bronze)"
        print(f"[SALTADO] {source}.{dataset}: no existe {BRONZE_CATALOG}.{source}.{dataset}")
        return
    try:
        run_dataset(
            environment=ENVIRONMENT,
            layer="silver",
            source=source,
            dataset=dataset,
            config_root=CONFIG_ROOT,
        )
        results[key] = "OK"
        print(f"[OK] {source}.{dataset}")
    except Exception as exc:  # noqa: BLE001 - se registra y se sigue con el resto
        results[key] = f"FALLO: {exc}"
        print(f"[FALLO] {source}.{dataset}: {exc}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dimensiones base

# COMMAND ----------

run_silver("trafico", "dim_trafico")
run_silver("trafico", "dim_distritos")
run_silver("meteo", "dim_meteo")
run_silver("calair", "dim_calair")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Hechos que dependen de las dimensiones anteriores

# COMMAND ----------

run_silver("trafico", "trafico_historico", requires=[("trafico", "dim_trafico")])
run_silver("trafico", "trafico_nrt", requires=[("trafico", "dim_trafico")])
run_silver("accidentes", "accidentes_historico", requires=[("trafico", "dim_distritos")])
run_silver("eventos", "eventos_culturales", requires=[("trafico", "dim_distritos")])
run_silver("meteo", "meteo_nrt", requires=[("meteo", "dim_meteo")])
run_silver("meteo", "meteo_historico", requires=[("meteo", "dim_meteo")])
run_silver("calair", "calair_nrt", requires=[("calair", "dim_calair")])
run_silver("calair", "calair_historico", requires=[("calair", "dim_calair")])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen

# COMMAND ----------

for (source, dataset), status in results.items():
    print(f"{source}.{dataset}: {status}")

failed = {key: status for key, status in results.items() if status.startswith("FALLO")}
if failed:
    raise RuntimeError(f"{len(failed)} dataset(s) Silver fallaron: {list(failed)}")
