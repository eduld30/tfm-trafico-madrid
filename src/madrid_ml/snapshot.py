"""Version-pinned construction and publication of the v1 Gold ML snapshot."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from madrid_ml.contract import (
    AIR_MAGNITUDES,
    EXPECTED_DISTRICTS,
    EXPECTED_ROWS,
    FEATURE_SCHEMA_VERSION,
    FEATURE_TABLE,
    FEATURE_TABLE_COLUMNS,
    GRID_END,
    GRID_START,
    INPUT_SUFFIXES,
    LABEL_TABLE,
    LABEL_TABLE_COLUMNS,
    LINEAGE_COLUMNS,
    ML_SCHEMA,
    TIME_CONTRACT,
    WEATHER_MAGNITUDES,
)
from madrid_ml.transformations import (
    SnapshotContractError,
    aggregate_magnitudes,
    aggregate_traffic,
    build_accident_labels,
    build_district_hour_grid,
    build_feature_snapshot,
)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
_KEY_COLUMNS = ("cod_distrito", "feature_hour")
_LINEAGE_COLUMN_NAMES = tuple(name for name, _, _ in LINEAGE_COLUMNS)
_SQL_TO_SPARK_TYPE = {
    "INT": "int",
    "BIGINT": "bigint",
    "DOUBLE": "double",
    "STRING": "string",
    "TIMESTAMP": "timestamp",
}

_TEMPORARY_OWNER_PROPERTY = "madrid_ml.snapshot_owner"
_TEMPORARY_ROLES = frozenset(("grid", "labels", "features"))


@dataclass(frozen=True)
class SnapshotResult:
    """Evidence returned after both managed Gold snapshots pass validation."""

    snapshot_id: str
    labels_table: str
    features_table: str
    input_versions: dict[str, int]
    code_commit: str
    feature_schema_version: str
    time_contract: str
    labels_row_count: int
    features_row_count: int
    district_count: int
    feature_hour_min: str
    feature_hour_max: str
    positive_count: int
    positive_rate: float
    positive_districts: int
    reconciled_districts: int
    labels_duplicate_keys: int
    features_duplicate_keys: int
    labels_missing_grid_keys: int
    labels_extra_grid_keys: int
    features_missing_grid_keys: int
    features_extra_grid_keys: int
    prediction_hour_mismatches: int
    invalid_target_rows: int
    invalid_count_rows: int
    labels_lineage: dict[str, str]
    features_lineage: dict[str, str]


@dataclass(frozen=True)
class SnapshotReference:
    """Delta versions and lineage of the latest published ML snapshot."""

    snapshot_id: str
    labels_delta_version: int
    features_delta_version: int


def canonical_versions_json(versions: Mapping[str, int]) -> str:
    """Serialize pinned Delta versions in their canonical lineage representation."""
    return json.dumps(dict(versions), sort_keys=True, separators=(",", ":"))


def validate_expected_versions(
    captured: Mapping[str, int],
    expected: Mapping[str, int],
) -> None:
    """Reject missing, extra, or changed Delta input versions."""
    captured_versions = dict(captured)
    expected_versions = dict(expected)
    if captured_versions != expected_versions:
        raise SnapshotContractError(
            "input version drift: "
            f"expected={canonical_versions_json(expected_versions)}, "
            f"captured={canonical_versions_json(captured_versions)}"
        )


def capture_silver_versions(
    spark: SparkSession,
    silver_catalog: str,
) -> dict[str, int]:
    """Capture the current Delta version of every Silver snapshot input."""
    return _capture_table_versions(spark, _input_table_names(silver_catalog))


def latest_snapshot_reference(
    spark: SparkSession,
    gold_catalog: str,
) -> SnapshotReference:
    """Resolve the latest coherent labels/features snapshot from Gold."""
    labels_table, features_table = _gold_table_names(gold_catalog)
    labels_version = int(
        spark.sql(f"DESCRIBE HISTORY {labels_table}").select("version").first()[0]
    )
    features_version = int(
        spark.sql(f"DESCRIBE HISTORY {features_table}").select("version").first()[0]
    )
    labels = (
        spark.read.option("versionAsOf", labels_version)
        .table(labels_table)
        .select("snapshot_id")
        .distinct()
        .collect()
    )
    features = (
        spark.read.option("versionAsOf", features_version)
        .table(features_table)
        .select("snapshot_id")
        .distinct()
        .collect()
    )
    label_ids = {row[0] for row in labels if row[0]}
    feature_ids = {row[0] for row in features if row[0]}
    if len(label_ids) != 1 or label_ids != feature_ids:
        raise SnapshotContractError(
            "latest Gold ML tables do not contain one coherent snapshot_id"
        )
    return SnapshotReference(
        snapshot_id=next(iter(label_ids)),
        labels_delta_version=labels_version,
        features_delta_version=features_version,
    )


def validate_unique_key(df: DataFrame, key_columns: Sequence[str]) -> int:
    """Fail when an output contains more than one row for a logical key."""
    duplicate = (
        df.groupBy(*key_columns)
        .count()
        .where(F.col("count") > 1)
        .select(*key_columns)
        .limit(1)
        .collect()
    )
    if duplicate:
        raise SnapshotContractError(
            f"clave duplicada para {tuple(key_columns)}; example={duplicate[0].asDict()}"
        )
    return 0


def count_grid_key_differences(
    grid_df: DataFrame,
    output_df: DataFrame,
    key_columns: Sequence[str] = _KEY_COLUMNS,
) -> tuple[int, int]:
    """Return missing and extra output key counts relative to the authoritative grid."""
    grid_keys = grid_df.select(*key_columns).distinct()
    output_keys = output_df.select(*key_columns).distinct()
    missing = grid_keys.join(output_keys, on=list(key_columns), how="left_anti").count()
    extra = output_keys.join(grid_keys, on=list(key_columns), how="left_anti").count()
    return missing, extra


def _single_lineage(df: DataFrame, logical_table: str) -> dict[str, str]:
    rows = df.select(*_LINEAGE_COLUMN_NAMES).distinct().limit(2).collect()
    if len(rows) != 1:
        raise SnapshotContractError(
            f"{logical_table}: expected one lineage combination, found {len(rows)}"
        )
    row = rows[0]
    null_columns = [name for name in _LINEAGE_COLUMN_NAMES if row[name] is None]
    if null_columns:
        raise SnapshotContractError(
            f"{logical_table}: null lineage columns {null_columns}"
        )
    return {name: str(row[name]) for name in _LINEAGE_COLUMN_NAMES}


def validate_matching_lineage(
    labels_df: DataFrame,
    features_df: DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """Require exactly one identical five-column lineage tuple in both outputs."""
    labels_lineage = _single_lineage(labels_df, LABEL_TABLE)
    features_lineage = _single_lineage(features_df, FEATURE_TABLE)
    differing = [
        name
        for name in _LINEAGE_COLUMN_NAMES
        if labels_lineage[name] != features_lineage[name]
    ]
    if differing:
        raise SnapshotContractError(
            f"published lineage differs in columns {differing}"
        )
    return labels_lineage, features_lineage


def _validate_identifier(identifier: str, purpose: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise SnapshotContractError(
            f"invalid {purpose} identifier {identifier!r}; expected lowercase snake_case"
        )
    return identifier


def _validated_table_name(catalog: str, suffix: str) -> str:
    parts = (catalog, *suffix.split("."))
    if len(parts) != 3:
        raise SnapshotContractError(f"invalid table suffix {suffix!r}")
    validated = tuple(
        _validate_identifier(part, "table")
        for part in parts
    )
    return ".".join(validated)


def _quoted_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) not in (2, 3):
        raise SnapshotContractError(f"invalid qualified identifier {name!r}")
    return ".".join(
        f"`{_validate_identifier(part, 'SQL')}`"
        for part in parts
    )


def _input_table_names(silver_catalog: str) -> dict[str, str]:
    catalog = _validate_identifier(silver_catalog, "silver catalog")
    return {
        role: _validated_table_name(catalog, suffix)
        for role, suffix in INPUT_SUFFIXES.items()
    }


def _gold_table_names(gold_catalog: str) -> tuple[str, str]:
    catalog = _validate_identifier(gold_catalog, "gold catalog")
    return (
        _validated_table_name(catalog, f"{ML_SCHEMA}.{LABEL_TABLE}"),
        _validated_table_name(catalog, f"{ML_SCHEMA}.{FEATURE_TABLE}"),
    )


def _capture_table_versions(
    spark: SparkSession,
    input_tables: Mapping[str, str],
) -> dict[str, int]:
    missing_tables = [
        table_name
        for table_name in input_tables.values()
        if not spark.catalog.tableExists(table_name)
    ]
    if missing_tables:
        raise SnapshotContractError(
            f"missing required Silver tables {sorted(missing_tables)}"
        )

    versions: dict[str, int] = {}
    for table_name in input_tables.values():
        row = (
            spark.sql(f"DESCRIBE HISTORY {_quoted_name(table_name)} LIMIT 1")
            .select("version")
            .first()
        )
        if row is None or row["version"] is None:
            raise SnapshotContractError(
                f"DESCRIBE HISTORY returned no version for {table_name}"
            )
        versions[table_name] = int(row["version"])
    return versions


def _read_versioned_inputs(
    spark: SparkSession,
    input_tables: Mapping[str, str],
    versions: Mapping[str, int],
) -> dict[str, DataFrame]:
    return {
        role: (
            spark.read.format("delta")
            .option("versionAsOf", versions[table_name])
            .table(table_name)
        )
        for role, table_name in input_tables.items()
    }


def _add_lineage(
    df: DataFrame,
    snapshot_id: str,
    input_versions_json: str,
    code_commit: str,
) -> DataFrame:
    return (
        df.withColumn("snapshot_id", F.lit(snapshot_id))
        .withColumn("input_versions_json", F.lit(input_versions_json))
        .withColumn("code_commit", F.lit(code_commit))
        .withColumn("feature_schema_version", F.lit(FEATURE_SCHEMA_VERSION))
        .withColumn("time_contract", F.lit(TIME_CONTRACT))
    )


def _ddl_columns(columns: Sequence[tuple[str, str, bool]]) -> str:
    return ",\n  ".join(
        f"`{name}` {sql_type}{'' if nullable else ' NOT NULL'}"
        for name, sql_type, nullable in columns
    )


def _expected_schema(
    columns: Sequence[tuple[str, str, bool]],
) -> tuple[tuple[str, str, bool], ...]:
    return tuple(
        (name.lower(), _SQL_TO_SPARK_TYPE[sql_type], nullable)
        for name, sql_type, nullable in columns
    )


def _validate_table_schema(
    spark: SparkSession,
    table_name: str,
    columns: Sequence[tuple[str, str, bool]],
) -> None:
    actual = tuple(
        (field.name.lower(), field.dataType.simpleString(), field.nullable)
        for field in spark.table(table_name).schema.fields
    )
    expected = _expected_schema(columns)
    if actual != expected:
        raise SnapshotContractError(
            f"{table_name}: existing schema differs from v1 contract; "
            f"expected={expected}, actual={actual}"
        )


def _validate_managed_delta_table(
    spark: SparkSession,
    table_name: str,
) -> None:
    metadata = spark.catalog.getTable(table_name)
    if metadata.isTemporary or metadata.tableType.upper() != "MANAGED":
        raise SnapshotContractError(
            f"{table_name}: expected a managed table, found {metadata.tableType!r}"
        )
    try:
        detail = (
            spark.sql(f"DESCRIBE DETAIL {_quoted_name(table_name)}")
            .select("format")
            .first()
        )
    except Exception as exc:
        raise SnapshotContractError(
            f"{table_name}: expected a managed Delta table"
        ) from exc
    if detail is None or str(detail["format"]).lower() != "delta":
        actual_format = None if detail is None else detail["format"]
        raise SnapshotContractError(
            f"{table_name}: expected Delta format, found {actual_format!r}"
        )


def _ensure_managed_tables(
    spark: SparkSession,
    gold_catalog: str,
) -> tuple[str, str]:
    catalog = _validate_identifier(gold_catalog, "gold catalog")
    schema_name = f"{catalog}.{ML_SCHEMA}"
    labels_table = _validated_table_name(catalog, f"{ML_SCHEMA}.{LABEL_TABLE}")
    features_table = _validated_table_name(catalog, f"{ML_SCHEMA}.{FEATURE_TABLE}")

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {_quoted_name(schema_name)}")
    for table_name, columns in (
        (labels_table, LABEL_TABLE_COLUMNS),
        (features_table, FEATURE_TABLE_COLUMNS),
    ):
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {_quoted_name(table_name)} (\n"
            f"  {_ddl_columns(columns)}\n"
            ") USING DELTA"
        )
        _validate_managed_delta_table(spark, table_name)
        _validate_table_schema(spark, table_name, columns)
    return labels_table, features_table


def _select_contract_columns(
    df: DataFrame,
    columns: Sequence[tuple[str, str, bool]],
    logical_table: str,
) -> DataFrame:
    missing = [name for name, _, _ in columns if name not in df.columns]
    if missing:
        raise SnapshotContractError(
            f"{logical_table}: missing output columns {missing}"
        )
    return df.select(
        *(F.col(name).cast(sql_type).alias(name) for name, sql_type, _ in columns)
    )


def _invalid_count_condition(count_columns: Sequence[str]):
    condition = F.lit(False)
    for column in count_columns:
        condition = condition | F.col(column).isNull() | (F.col(column) < 0)
    return condition


def _validate_output(
    df: DataFrame,
    grid_df: DataFrame,
    columns: Sequence[tuple[str, str, bool]],
    logical_table: str,
    row_count: int | None = None,
) -> dict[str, int | dict[str, str]]:
    actual_rows = df.count() if row_count is None else row_count
    if actual_rows != EXPECTED_ROWS:
        raise SnapshotContractError(
            f"{logical_table}: expected {EXPECTED_ROWS} rows, found {actual_rows}"
        )

    duplicate_keys = validate_unique_key(df, _KEY_COLUMNS)
    missing_keys, extra_keys = count_grid_key_differences(grid_df, df)
    if missing_keys or extra_keys:
        raise SnapshotContractError(
            f"{logical_table}: grid key mismatch; missing={missing_keys}, extra={extra_keys}"
        )

    prediction_hour_mismatches = df.where(
        F.col("prediction_hour").isNull()
        | F.col("feature_hour").isNull()
        | (
            F.col("prediction_hour")
            != F.col("feature_hour") + F.expr("INTERVAL 1 HOUR")
        )
    ).count()
    if prediction_hour_mismatches:
        raise SnapshotContractError(
            f"{logical_table}: prediction_hour mismatches={prediction_hour_mismatches}"
        )

    invalid_target_rows = df.where(
        F.col("target_accident_next_hour").isNull()
        | ~F.col("target_accident_next_hour").isin(0, 1)
    ).count()
    if invalid_target_rows:
        raise SnapshotContractError(
            f"{logical_table}: invalid target rows={invalid_target_rows}"
        )

    count_columns = tuple(
        name for name, sql_type, _ in columns if sql_type == "BIGINT"
    )
    invalid_count_rows = df.where(_invalid_count_condition(count_columns)).count()
    if invalid_count_rows:
        raise SnapshotContractError(
            f"{logical_table}: invalid count rows={invalid_count_rows}"
        )

    non_nullable_columns = tuple(
        name for name, _, nullable in columns if not nullable
    )
    null_condition = F.lit(False)
    for column in non_nullable_columns:
        null_condition = null_condition | F.col(column).isNull()
    invalid_null_rows = df.where(null_condition).count()
    if invalid_null_rows:
        raise SnapshotContractError(
            f"{logical_table}: nulls in non-nullable columns; rows={invalid_null_rows}"
        )

    lineage = _single_lineage(df, logical_table)
    return {
        "row_count": actual_rows,
        "duplicate_keys": duplicate_keys,
        "missing_grid_keys": missing_keys,
        "extra_grid_keys": extra_keys,
        "prediction_hour_mismatches": prediction_hour_mismatches,
        "invalid_target_rows": invalid_target_rows,
        "invalid_count_rows": invalid_count_rows,
        "lineage": lineage,
    }


def _temporary_storage_owner(run_token: str) -> str:
    normalized = run_token.strip()
    if not normalized:
        raise SnapshotContractError("run_token must not be blank")
    return sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _temporary_table_name(gold_catalog: str, owner: str, role: str) -> str:
    catalog = _validate_identifier(gold_catalog, "gold catalog")
    if not re.fullmatch(r"[0-9a-f]{24}", owner):
        raise SnapshotContractError(f"invalid temporary storage owner {owner!r}")
    if role not in _TEMPORARY_ROLES:
        raise SnapshotContractError(f"invalid temporary table role {role!r}")
    return f"{catalog}.{ML_SCHEMA}.__tmp_snapshot_{owner}_{role}"


def _persist_materialized(
    spark: SparkSession,
    df: DataFrame,
    table_name: str,
    owner: str,
    temporary_tables: list[str],
) -> DataFrame:
    quoted_table = _quoted_name(table_name)
    view_name = _validate_identifier(
        f"{table_name.rsplit('.', maxsplit=1)[-1]}_view",
        "temporary view",
    )
    df.createOrReplaceTempView(view_name)
    temporary_tables.append(table_name)
    try:
        spark.sql(
            f"CREATE TABLE {quoted_table} USING DELTA "
            f"TBLPROPERTIES ('{_TEMPORARY_OWNER_PROPERTY}' = '{owner}') "
            f"AS SELECT * FROM `{view_name}`"
        )
    finally:
        spark.catalog.dropTempView(view_name)

    return spark.table(table_name)


def _drop_owned_temporary_table(
    spark: SparkSession,
    table_name: str,
    owner: str,
) -> None:
    if not spark.catalog.tableExists(table_name):
        return
    detail = spark.sql(f"DESCRIBE DETAIL {_quoted_name(table_name)}").first()
    if detail is None:
        raise SnapshotContractError(
            f"{table_name}: DESCRIBE DETAIL returned no row during cleanup"
        )
    properties = detail["properties"] or {}
    observed_owner = properties.get(_TEMPORARY_OWNER_PROPERTY)
    if observed_owner != owner:
        raise SnapshotContractError(
            f"refusing to drop temporary table owned by {observed_owner!r}: "
            f"{table_name}"
        )
    spark.sql(f"DROP TABLE {_quoted_name(table_name)}")
    if spark.catalog.tableExists(table_name):
        raise SnapshotContractError(
            f"DROP TABLE left temporary table present: {table_name}"
        )


def _cleanup_temporary_tables(
    spark: SparkSession,
    temporary_tables: Sequence[str],
    owner: str,
) -> None:
    failures: list[tuple[str, Exception]] = []
    for table_name in reversed(temporary_tables):
        try:
            _drop_owned_temporary_table(spark, table_name, owner)
        except Exception as exc:
            failures.append((table_name, exc))
    if failures:
        failed_names = [table_name for table_name, _ in failures]
        raise SnapshotContractError(
            f"failed to clean temporary Delta tables {failed_names}"
        ) from failures[0][1]


def _output_summary(df: DataFrame) -> dict[str, int | str]:
    row = (
        df.agg(
            F.count(F.lit(1)).cast("long").alias("row_count"),
            F.countDistinct("cod_distrito").cast("long").alias("district_count"),
            F.date_format(
                F.min("feature_hour"), "yyyy-MM-dd HH:mm:ss"
            ).alias("feature_hour_min"),
            F.date_format(
                F.max("feature_hour"), "yyyy-MM-dd HH:mm:ss"
            ).alias("feature_hour_max"),
            F.sum(
                F.when(F.col("target_accident_next_hour") == 1, F.lit(1)).otherwise(
                    F.lit(0)
                )
            )
            .cast("long")
            .alias("positive_count"),
            F.countDistinct(
                F.when(
                    F.col("target_accident_next_hour") == 1,
                    F.col("cod_distrito"),
                )
            )
            .cast("long")
            .alias("positive_districts"),
        )
        .first()
    )
    if row is None:
        raise SnapshotContractError("published snapshot summary returned no row")
    return {
        "row_count": int(row["row_count"]),
        "district_count": int(row["district_count"]),
        "feature_hour_min": str(row["feature_hour_min"]),
        "feature_hour_max": str(row["feature_hour_max"]),
        "positive_count": int(row["positive_count"]),
        "positive_districts": int(row["positive_districts"]),
    }


def _validate_summary(
    summary: Mapping[str, int | str],
    logical_table: str,
) -> None:
    expected = {
        "row_count": EXPECTED_ROWS,
        "district_count": EXPECTED_DISTRICTS,
        "feature_hour_min": GRID_START,
        "feature_hour_max": GRID_END,
    }
    observed = {name: summary[name] for name in expected}
    if observed != expected:
        raise SnapshotContractError(
            f"{logical_table}: published summary mismatch; "
            f"expected={expected}, observed={observed}"
        )


def _unique_accidents_for_reconciliation(accidents_df: DataFrame) -> DataFrame:
    invalid_ids = (
        accidents_df.where(
            F.col("fecha_hora").isNull() | F.col("cod_distrito").isNull()
        )
        .select("num_expediente")
        .distinct()
    )
    return (
        accidents_df.join(invalid_ids, on="num_expediente", how="left_anti")
        .select(
            "num_expediente",
            F.col("cod_distrito").cast("int").alias("cod_distrito"),
            F.date_trunc("hour", F.col("fecha_hora")).alias("prediction_hour"),
        )
        .distinct()
    )


def _reconciled_positive_districts(
    labels_df: DataFrame,
    accidents_df: DataFrame,
) -> int:
    first_positive = (
        labels_df.where(F.col("target_accident_next_hour") == 1)
        .withColumn(
            "__row_number",
            F.row_number().over(
                Window.partitionBy("cod_distrito").orderBy("feature_hour")
            ),
        )
        .where(F.col("__row_number") == 1)
        .drop("__row_number")
    )
    actual_counts = _unique_accidents_for_reconciliation(accidents_df).groupBy(
        "cod_distrito", "prediction_hour"
    ).agg(F.count(F.lit(1)).cast("long").alias("actual_accident_count"))
    return (
        first_positive.join(
            actual_counts,
            on=["cod_distrito", "prediction_hour"],
            how="left",
        )
        .where(
            F.col("actual_accident_count") == F.col("n_accidentes_next_hour")
        )
        .select("cod_distrito")
        .distinct()
        .count()
    )


def build_training_snapshot(
    spark: SparkSession,
    silver_catalog: str,
    gold_catalog: str,
    code_commit: str,
    expected_versions: Mapping[str, int],
) -> SnapshotResult:
    """Build, publish, and verify the two managed v1 Gold ML snapshots."""
    observed_timezone = spark.conf.get("spark.sql.session.timeZone")
    if observed_timezone != "Etc/UTC":
        raise SnapshotContractError(
            "spark.sql.session.timeZone must be Etc/UTC; "
            f"observed={observed_timezone!r}"
        )
    normalized_commit = code_commit.strip()
    if not normalized_commit:
        raise SnapshotContractError("code_commit must not be blank")

    input_tables = _input_table_names(silver_catalog)
    _validate_identifier(gold_catalog, "gold catalog")
    captured_versions = _capture_table_versions(spark, input_tables)
    validate_expected_versions(captured_versions, expected_versions)
    inputs = _read_versioned_inputs(
        spark,
        input_tables,
        captured_versions,
    )
    labels_table, features_table = _ensure_managed_tables(spark, gold_catalog)

    snapshot_id = str(uuid4())
    input_versions_json = canonical_versions_json(captured_versions)
    temporary_owner = _temporary_storage_owner(snapshot_id)
    temporary_tables: list[str] = []
    try:
        grid = _persist_materialized(
            spark,
            build_district_hour_grid(spark, inputs["districts"]),
            _temporary_table_name(gold_catalog, temporary_owner, "grid"),
            temporary_owner,
            temporary_tables,
        )

        labels_base = build_accident_labels(inputs["accidents"], grid)
        labels = _persist_materialized(
            spark,
            _select_contract_columns(
                _add_lineage(
                    labels_base,
                    snapshot_id,
                    input_versions_json,
                    normalized_commit,
                ),
                LABEL_TABLE_COLUMNS,
                LABEL_TABLE,
            ),
            _temporary_table_name(gold_catalog, temporary_owner, "labels"),
            temporary_owner,
            temporary_tables,
        )
        labels_rows = labels.count()
        _validate_output(
            labels,
            grid,
            LABEL_TABLE_COLUMNS,
            LABEL_TABLE,
            row_count=labels_rows,
        )
        labels.write.insertInto(labels_table, overwrite=True)

        traffic = aggregate_traffic(inputs["traffic"])
        weather = aggregate_magnitudes(inputs["weather"], WEATHER_MAGNITUDES)
        air = aggregate_magnitudes(inputs["air"], AIR_MAGNITUDES)
        features_base = build_feature_snapshot(
            labels,
            traffic,
            weather,
            air,
        )
        features = _persist_materialized(
            spark,
            _select_contract_columns(
                _add_lineage(
                    features_base,
                    snapshot_id,
                    input_versions_json,
                    normalized_commit,
                ),
                FEATURE_TABLE_COLUMNS,
                FEATURE_TABLE,
            ),
            _temporary_table_name(gold_catalog, temporary_owner, "features"),
            temporary_owner,
            temporary_tables,
        )
        features_rows = features.count()
        _validate_output(
            features,
            grid,
            FEATURE_TABLE_COLUMNS,
            FEATURE_TABLE,
            row_count=features_rows,
        )
        features.write.insertInto(features_table, overwrite=True)

        published_labels = spark.table(labels_table)
        published_features = spark.table(features_table)
        labels_metrics = _validate_output(
            published_labels,
            grid,
            LABEL_TABLE_COLUMNS,
            labels_table,
        )
        features_metrics = _validate_output(
            published_features,
            grid,
            FEATURE_TABLE_COLUMNS,
            features_table,
        )
        labels_lineage, features_lineage = validate_matching_lineage(
            published_labels,
            published_features,
        )

        labels_summary = _output_summary(published_labels)
        features_summary = _output_summary(published_features)
        _validate_summary(labels_summary, labels_table)
        _validate_summary(features_summary, features_table)
        if labels_summary != features_summary:
            raise SnapshotContractError(
                "published labels and features summaries differ; "
                f"labels={labels_summary}, features={features_summary}"
            )

        positive_districts = int(labels_summary["positive_districts"])
        reconciled_districts = _reconciled_positive_districts(
            published_labels,
            inputs["accidents"],
        )
        if reconciled_districts != positive_districts:
            raise SnapshotContractError(
                "positive district reconciliation failed; "
                f"expected={positive_districts}, reconciled={reconciled_districts}"
            )

        labels_row_count = int(labels_summary["row_count"])
        positive_count = int(labels_summary["positive_count"])
        return SnapshotResult(
            snapshot_id=snapshot_id,
            labels_table=labels_table,
            features_table=features_table,
            input_versions=dict(captured_versions),
            code_commit=normalized_commit,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            time_contract=TIME_CONTRACT,
            labels_row_count=labels_row_count,
            features_row_count=int(features_summary["row_count"]),
            district_count=int(labels_summary["district_count"]),
            feature_hour_min=str(labels_summary["feature_hour_min"]),
            feature_hour_max=str(labels_summary["feature_hour_max"]),
            positive_count=positive_count,
            positive_rate=positive_count / labels_row_count,
            positive_districts=positive_districts,
            reconciled_districts=reconciled_districts,
            labels_duplicate_keys=int(labels_metrics["duplicate_keys"]),
            features_duplicate_keys=int(features_metrics["duplicate_keys"]),
            labels_missing_grid_keys=int(labels_metrics["missing_grid_keys"]),
            labels_extra_grid_keys=int(labels_metrics["extra_grid_keys"]),
            features_missing_grid_keys=int(features_metrics["missing_grid_keys"]),
            features_extra_grid_keys=int(features_metrics["extra_grid_keys"]),
            prediction_hour_mismatches=int(
                labels_metrics["prediction_hour_mismatches"]
            ),
            invalid_target_rows=int(labels_metrics["invalid_target_rows"]),
            invalid_count_rows=(
                int(labels_metrics["invalid_count_rows"])
                + int(features_metrics["invalid_count_rows"])
            ),
            labels_lineage=labels_lineage,
            features_lineage=features_lineage,
        )
    finally:
        _cleanup_temporary_tables(spark, temporary_tables, temporary_owner)
