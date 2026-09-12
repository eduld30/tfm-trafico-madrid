# Databricks notebook source
import json
import os
from dataclasses import asdict

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ml import build_training_snapshot, capture_silver_versions

# COMMAND ----------

dbutils.widgets.text("environment", "", "Entorno")  # noqa: F821
dbutils.widgets.text("config_root", "", "Config root")  # noqa: F821
dbutils.widgets.text("code_commit", "", "Commit opcional")  # noqa: F821
dbutils.widgets.text(
    "expected_versions_json",
    "",
    "Versiones Silver opcionales para reproducibilidad",
)  # noqa: F821

environment = dbutils.widgets.get("environment").strip()  # noqa: F821
config_root = dbutils.widgets.get("config_root").strip()  # noqa: F821
code_commit = dbutils.widgets.get("code_commit").strip()  # noqa: F821
expected_versions_json = dbutils.widgets.get(  # noqa: F821
    "expected_versions_json"
).strip()
if not environment or not config_root:
    raise ValueError("environment y config_root son obligatorios")

environment_config = ConfigLoader(config_root).load_environment(environment)
silver_catalog = environment_config.catalogs.silver
gold_catalog = environment_config.catalogs.gold
code_commit = code_commit or next(
    (
        os.environ.get(name, "").strip()
        for name in ("GIT_COMMIT", "CODE_COMMIT", "DATABRICKS_BUNDLE_GIT_COMMIT")
        if os.environ.get(name, "").strip()
    ),
    "workspace",
)
expected_versions = (
    json.loads(expected_versions_json)
    if expected_versions_json
    else capture_silver_versions(spark, silver_catalog)  # noqa: F821
)

# COMMAND ----------

result = build_training_snapshot(
    spark=spark,  # noqa: F821
    silver_catalog=silver_catalog,
    gold_catalog=gold_catalog,
    code_commit=code_commit,
    expected_versions=expected_versions,
)
dbutils.notebook.exit(  # noqa: F821
    json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
)
