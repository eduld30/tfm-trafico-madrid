# Databricks notebook source
import json
from dataclasses import asdict

from madrid_ml import build_training_snapshot

# COMMAND ----------

dbutils.widgets.text("silver_catalog", "", "Catálogo Silver")  # noqa: F821
dbutils.widgets.text("gold_catalog", "", "Catálogo Gold")  # noqa: F821
dbutils.widgets.text("code_commit", "", "Commit de código")  # noqa: F821
dbutils.widgets.text(  # noqa: F821
    "expected_versions_json",
    "",
    "Versiones Silver autorizadas",
)

silver_catalog = dbutils.widgets.get("silver_catalog").strip()  # noqa: F821
gold_catalog = dbutils.widgets.get("gold_catalog").strip()  # noqa: F821
code_commit = dbutils.widgets.get("code_commit").strip()  # noqa: F821
expected_versions_json = dbutils.widgets.get(  # noqa: F821
    "expected_versions_json"
).strip()
if not all((silver_catalog, gold_catalog, code_commit, expected_versions_json)):
    raise ValueError(
        "silver_catalog, gold_catalog, code_commit y expected_versions_json son obligatorios"
    )
expected_versions = json.loads(expected_versions_json)

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
