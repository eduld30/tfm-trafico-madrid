# Databricks notebook source
import json
from dataclasses import asdict
from datetime import datetime

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ml.nrt import DEFAULT_REGISTERED_MODEL, NrtScoringConfig, run_nrt_scoring

# COMMAND ----------

dbutils.widgets.text("environment", "", "Entorno")  # noqa: F821
dbutils.widgets.text("config_root", "", "Directorio de configuracion")  # noqa: F821
dbutils.widgets.text("registered_model_name", "", "Modelo registrado")  # noqa: F821
dbutils.widgets.text("model_alias", "Champion", "Alias del modelo")  # noqa: F821
dbutils.widgets.text("mlflow_dfs_tmp", "", "Volumen temporal MLflow")  # noqa: F821
dbutils.widgets.text("prediction_run_id", "", "Identificador de ejecucion")  # noqa: F821
dbutils.widgets.text("cutoff", "", "Cutoff opcional ISO")  # noqa: F821

environment = dbutils.widgets.get("environment").strip()  # noqa: F821
config_root = dbutils.widgets.get("config_root").strip()  # noqa: F821
registered_model_name = dbutils.widgets.get("registered_model_name").strip()  # noqa: F821
model_alias = dbutils.widgets.get("model_alias").strip()  # noqa: F821
mlflow_dfs_tmp = dbutils.widgets.get("mlflow_dfs_tmp").strip()  # noqa: F821
prediction_run_id = dbutils.widgets.get("prediction_run_id").strip()  # noqa: F821
cutoff_value = dbutils.widgets.get("cutoff").strip()  # noqa: F821

if not environment or not config_root or not model_alias:
    raise ValueError("environment, config_root y model_alias son obligatorios")

environment_config = ConfigLoader(config_root).load_environment(environment)
gold_catalog = environment_config.catalogs.gold
registered_model_name = registered_model_name or (
    f"{gold_catalog}.ml.{DEFAULT_REGISTERED_MODEL}"
)
mlflow_dfs_tmp = mlflow_dfs_tmp or (
    f"/Volumes/{gold_catalog}/ml/mlflow_tmp/model-scoring"
)
cutoff = datetime.fromisoformat(cutoff_value) if cutoff_value else None

# COMMAND ----------

result = run_nrt_scoring(
    spark=spark,  # noqa: F821
    config=NrtScoringConfig(
        silver_catalog=environment_config.catalogs.silver,
        gold_catalog=gold_catalog,
        registered_model_name=registered_model_name,
        model_alias=model_alias,
        mlflow_dfs_tmp=mlflow_dfs_tmp,
        run_id=prediction_run_id,
    ),
    cutoff=cutoff,
)
dbutils.notebook.exit(  # noqa: F821
    json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
)
