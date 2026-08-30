# Databricks notebook source
import json
import re

# COMMAND ----------

dbutils.widgets.text("temporary_table", "", "Temporary Delta table")  # noqa: F821
dbutils.widgets.text("ownership_token", "", "Cleanup ownership token")  # noqa: F821
temporary_table = dbutils.widgets.get("temporary_table").strip()  # noqa: F821
ownership_token = dbutils.widgets.get("ownership_token").strip()  # noqa: F821

_TABLE_PATTERN = re.compile(
    r"^dev_gold\.ml\.__tmp_serverless_delta_[a-z0-9_]+$"
)
_TOKEN_PATTERN = re.compile(r"^omp-delta-smoke-[0-9]{8}T[0-9]{9}Z$")
_OWNER_PROPERTY = "madrid_ml.smoke_owner"

if not _TABLE_PATTERN.fullmatch(temporary_table):
    raise ValueError(
        "temporary_table must match "
        "dev_gold.ml.__tmp_serverless_delta_<lowercase_run_suffix>"
    )
if not _TOKEN_PATTERN.fullmatch(ownership_token):
    raise ValueError("ownership_token has an invalid format")
if spark.catalog.tableExists(temporary_table):  # noqa: F821
    raise RuntimeError(f"refusing to replace pre-existing table {temporary_table}")

quoted_table = f"`{temporary_table.replace('.', '`.`')}`"


def assert_owned_table_if_present() -> bool:
    if not spark.catalog.tableExists(temporary_table):  # noqa: F821
        return False
    detail = spark.sql(f"DESCRIBE DETAIL {quoted_table}").first()  # noqa: F821
    if detail is None:
        raise RuntimeError(f"DESCRIBE DETAIL returned no row for {temporary_table}")
    properties = detail["properties"] or {}
    observed_owner = properties.get(_OWNER_PROPERTY)
    if observed_owner != ownership_token:
        raise RuntimeError(
            f"refusing to drop table owned by {observed_owner!r}: {temporary_table}"
        )
    return True


# COMMAND ----------

result = None
run_error = None
cleanup_error = None
try:
    spark.sql(  # noqa: F821
        f"CREATE TABLE {quoted_table} (id INT, value STRING) USING DELTA "
        f"TBLPROPERTIES ('{_OWNER_PROPERTY}' = '{ownership_token}')"
    )
    source = spark.createDataFrame(  # noqa: F821
        [(1, "alpha"), (2, "beta"), (3, "gamma")],
        "id INT, value STRING",
    )
    source.write.mode("append").insertInto(temporary_table)

    metadata = spark.catalog.getTable(temporary_table)  # noqa: F821
    detail = spark.sql(f"DESCRIBE DETAIL {quoted_table}").first()  # noqa: F821
    observed = spark.table(temporary_table).orderBy("id").collect()  # noqa: F821

    if metadata.isTemporary or metadata.tableType.upper() != "MANAGED":
        raise RuntimeError(
            f"expected a managed table, found {metadata.tableType!r}"
        )
    if detail is None or str(detail["format"]).lower() != "delta":
        actual_format = None if detail is None else detail["format"]
        raise RuntimeError(f"expected Delta format, found {actual_format!r}")
    if (detail["properties"] or {}).get(_OWNER_PROPERTY) != ownership_token:
        raise RuntimeError("temporary table ownership property changed")
    if [(row.id, row.value) for row in observed] != [
        (1, "alpha"),
        (2, "beta"),
        (3, "gamma"),
    ]:
        raise RuntimeError(f"Delta round trip mismatch: {observed!r}")

    result = {
        "temporary_table": temporary_table,
        "ownership_token": ownership_token,
        "table_type": metadata.tableType,
        "format": str(detail["format"]).lower(),
        "row_count": len(observed),
        "id_sum": sum(row.id for row in observed),
    }
except Exception as exc:
    run_error = exc
finally:
    try:
        if assert_owned_table_if_present():
            spark.sql(f"DROP TABLE {quoted_table}")  # noqa: F821
        if spark.catalog.tableExists(temporary_table):  # noqa: F821
            raise RuntimeError(f"cleanup left table present: {temporary_table}")
    except Exception as exc:
        cleanup_error = exc

if run_error is not None:
    if cleanup_error is not None:
        raise RuntimeError(
            f"smoke failed and cleanup also failed: {cleanup_error}"
        ) from run_error
    raise run_error
if cleanup_error is not None:
    raise cleanup_error

result["cleanup_verified"] = True
dbutils.notebook.exit(  # noqa: F821
    json.dumps(result, sort_keys=True, separators=(",", ":"))
)
