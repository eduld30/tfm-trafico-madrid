# ML Preprocessing Contract v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar un preprocessing Spark ML versionado que sanee el snapshot Gold, ajuste todo estado solo con 2019–2023, transforme 2024 y 2025 sin reajuste y produzca un vector finito de 111 componentes con manifiesto reproducible.

**Architecture:** `madrid_ml.preprocessing` será la única frontera de preprocessing. Leerá labels y features Gold mediante versiones Delta explícitas, validará provenance, dividirá el snapshot con timezone UTC, saneará valores y ajustará un `PipelineModel` Spark sobre train. Un wrapper pequeño aplicará el saneamiento determinista antes del modelo aprendido y generará el manifiesto que el notebook registrará en MLflow.

**Tech Stack:** Python 3.10+, PySpark 3.5, Spark ML, Delta Lake en Databricks, MLflow incluido en el runtime Databricks, pytest y Databricks serverless environment version 4.

## Global Constraints

- Modificar únicamente propiedad ML: `src/madrid_ml/**`, `tests/unit/test_ml_*`, `notebooks/run_ml_preprocessing.py`, documentación ML y artefactos privados bajo `.team-workspace/`.
- No modificar `src/madrid_ingestion/**`, `conf/**`, ADF, Silver, jobs de ingesta ni los nueve jobs existentes.
- No ejecutar ningún job de ingesta.
- No añadir Pandas, scikit-learn, LightGBM ni dependencias nuevas al `pyproject.toml`.
- No crear tablas Gold adicionales, feature store, snapshot preprocesado ni job persistente.
- Mantener el alcance retrospectivo aprobado: evitar únicamente fuga de estado aprendido entre splits; no añadir ni afirmar garantía point-in-time o latencia de publicación.
- Mantener `feature_schema_version="1"`, `preprocessing_contract_version="1"` y `time_contract="source_wall_clock_as_stored_in_silver"`.
- Fijar `spark.sql.session.timeZone="Etc/UTC"` antes de construir límites temporales.
- Train: `[2019-01-01 00:00:00, 2024-01-01 00:00:00)`; validación: 2024; evaluación final: 2025; excluir desde 2026.
- Ajustar medianas, media, desviación estándar e índices categóricos exclusivamente con train.
- Conservar exactamente 920.304 filas de train, 184.464 de validación, 183.960 de evaluación y contabilizar 91.203 exclusiones.
- El smoke remoto requiere autorización explícita separada, una sola tarea serverless, 600 segundos, cero reintentos y limpieza terminal verificada.
- Seguir TDD: observar cada prueba nueva fallar por el comportamiento ausente antes de escribir producción.

---

## File Structure

- Create `src/madrid_ml/preprocessing.py`: contrato, tipos, saneamiento, carga Gold versionada, split, pipeline Spark, wrapper y manifiesto.
- Modify `src/madrid_ml/__init__.py`: exportar únicamente la API pública del preprocessing.
- Create `tests/unit/test_ml_preprocessing.py`: contratos Spark observables y helpers sintéticos.
- Create `notebooks/run_ml_preprocessing.py`: wrapper Databricks fino, logging MLflow y salida compacta.
- Modify `tests/unit/test_ml_notebooks.py`: validar límites de celdas del nuevo notebook SOURCE.
- Modify `README.md`: documentar API, periodos, artefactos y ejecución manual.
- Create private execution evidence under `.team-workspace/` only during the authorized smoke; do not commit it.

---

### Task 1: Deterministic Sanitization and Domain Gates

**Files:**
- Create: `src/madrid_ml/preprocessing.py`
- Create: `tests/unit/test_ml_preprocessing.py`

**Interfaces:**
- Produces: `PreprocessingContractError`, `sanitize_features(df: DataFrame) -> DataFrame`, `profile_feature_quality(df: DataFrame) -> dict[str, dict[str, int]]`, `validate_feature_domains(df: DataFrame) -> None`.
- Consumes: feature names and schema constants from `madrid_ml.contract`.

- [ ] **Step 1: Write the synthetic-row helper and failing finite-value tests**

Create a helper that emits every required Gold column. Keep lineage constant and vary the 33 numerical inputs so later scaling has non-zero variance.

```python
from datetime import datetime, timedelta

import pytest
from pyspark.sql import functions as F

from madrid_ml.preprocessing import (
    CONTINUOUS_FEATURES,
    COUNT_FEATURES,
    PHYSICAL_RANGES,
    sanitize_features,
)


def make_row(index: int = 0, **overrides):
    row = {
        "cod_distrito": index % 21 + 1,
        "feature_hour": (
            feature_hour := datetime(
                2020,
                index % 12 + 1,
                index % 27 + 1,
                index % 24,
            )
        ),
        "prediction_hour": feature_hour + timedelta(hours=1),
        "n_accidentes_next_hour": 0,
        "target_accident_next_hour": index % 2,
        "hora_dia": index % 24,
        "dia_semana": index % 7 + 1,
        "mes": index % 12 + 1,
        "snapshot_id": "snapshot-test",
        "input_versions_json": "{}",
        "code_commit": "a" * 40,
        "feature_schema_version": "1",
        "time_contract": "source_wall_clock_as_stored_in_silver",
    }
    # Fill every mean with float(index + 1) and every count with index % 3 + 1.
    for name in CONTINUOUS_FEATURES:
        row[name] = float(index + 1)
    for name in COUNT_FEATURES:
        row[name] = index % 3 + 1
    row.update(overrides)
    return row
```


Add tests:

```python
def test_sanitize_marks_non_finite_and_impossible_values_unavailable(spark):
    frame = spark.createDataFrame([
        make_row(
            trafico_vmed_media=-1.0,
            trafico_ocupacion_media=float("nan"),
            meteo_radiacion_solar_media=float("inf"),
            calair_no2_media=-0.1,
        )
    ])

    row = sanitize_features(frame).first()

    assert row.trafico_vmed_media is None
    assert row.trafico_vmed_media_available == 0.0
    assert row.trafico_ocupacion_media is None
    assert row.trafico_ocupacion_media_available == 0.0
    assert row.meteo_radiacion_solar_media is None
    assert row.meteo_radiacion_solar_media_available == 0.0
    assert row.calair_no2_media is None
    assert row.calair_no2_media_available == 0.0


def test_sanitize_keeps_valid_zero_and_negative_temperature(spark):
    frame = spark.createDataFrame([
        make_row(

            trafico_intensidad_media=0.0,
            meteo_temperatura_media=-10.0,
        )
    ])

    row = sanitize_features(frame).first()

    assert row.trafico_intensidad_media == 0.0
    assert row.trafico_intensidad_media_available == 1.0
    assert row.meteo_temperatura_media == -10.0
    assert row.meteo_temperatura_media_available == 1.0
```

Add parameterized `test_sanitize_respects_every_physical_range`: each non-negative traffic/weather/air feature rejects `-0.1` and accepts `0.0`; humidity rejects `-0.1` and `100.1` while accepting `0.0` and `100.0`; pressure rejects `0.0` and accepts `1.0`; temperature accepts `-55.0`. This is the executable coverage map for every approved physical rule, not a representative sample.

- [ ] **Step 2: Run the sanitation tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py -k "sanitize"
```

Expected: collection fails because `madrid_ml.preprocessing` does not exist.

- [ ] **Step 3: Implement exact ordered constants and physical ranges**

Define in `preprocessing.py`:

```python
PREPROCESSING_CONTRACT_VERSION = "1"
SESSION_TIMEZONE = "Etc/UTC"
EXPECTED_VECTOR_SIZE = 111

CONTINUOUS_FEATURES = (
    "trafico_intensidad_media",
    "trafico_ocupacion_media",
    "trafico_carga_media",
    "trafico_vmed_media",
    "meteo_velocidad_viento_media",
    "meteo_temperatura_media",
    "meteo_humedad_relativa_media",
    "meteo_presion_media",
    "meteo_radiacion_solar_media",
    "meteo_precipitacion_media",
    "calair_so2_media",
    "calair_co_media",
    "calair_no_media",
    "calair_no2_media",
    "calair_pm25_media",
    "calair_pm10_media",
    "calair_nox_media",
    "calair_o3_media",
)

COUNT_FEATURES = (
    "trafico_puntos_n",
    "meteo_velocidad_viento_n",
    "meteo_temperatura_n",
    "meteo_humedad_relativa_n",
    "meteo_presion_n",
    "meteo_radiacion_solar_n",
    "meteo_precipitacion_n",
    "calair_so2_n",
    "calair_co_n",
    "calair_no_n",
    "calair_no2_n",
    "calair_pm25_n",
    "calair_pm10_n",
    "calair_nox_n",
    "calair_o3_n",
)

NUMERIC_FEATURES = CONTINUOUS_FEATURES + COUNT_FEATURES
AVAILABLE_FEATURES = tuple(f"{name}_available" for name in CONTINUOUS_FEATURES)
```

Represent hard ranges without lambdas:

```python
@dataclass(frozen=True)
class PhysicalRange:
    minimum: float | None = None
    maximum: float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
```

Populate every continuous feature exactly as the approved spec: non-negative traffic except no upper bound, finite temperature, humidity `[0, 100]`, pressure `> 0`, non-negative remaining weather and air.

- [ ] **Step 4: Implement finite and range expressions plus sanitation**

Use Spark expressions only:

```python
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
```

- [ ] **Step 5: Run the finite-value tests and verify GREEN**

Run the command from Step 2.

Expected: all sanitation tests pass.

- [ ] **Step 6: Write failing domain and coherence tests**

Add tests for:

```python
def test_validate_domains_rejects_negative_count(spark):
    frame = spark.createDataFrame([make_row(trafico_puntos_n=-1)])
    with pytest.raises(PreprocessingContractError, match="trafico_puntos_n"):
        validate_feature_domains(frame)


def test_validate_domains_rejects_zero_count_with_finite_mean(spark):
    frame = spark.createDataFrame([
        make_row(meteo_temperatura_n=0, meteo_temperatura_media=12.0)
    ])
    with pytest.raises(PreprocessingContractError, match="meteo_temperatura"):
        validate_feature_domains(frame)


def test_validate_domains_rejects_invalid_category_and_target(spark):
    frame = spark.createDataFrame([make_row(cod_distrito=22, target_accident_next_hour=2)])
    with pytest.raises(PreprocessingContractError, match="cod_distrito|target"):
        validate_feature_domains(frame)
```

Also test that `profile_feature_quality` distinguishes raw missing/non-finite from finite range violations in one aggregate row.

- [ ] **Step 7: Run the new tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py -k "validate_domains or profile_feature_quality"
```

Expected: FAIL because gates and profiling are absent.

- [ ] **Step 8: Implement one aggregated gate and one aggregated profile**

Build all invalid counters in a single `df.agg(...).first()` per function. Do not call `count()` once per feature. Fail with one `PreprocessingContractError` listing every non-zero structural counter.

Cross-coherence pairs are:

```python
MAGNITUDE_COUNT_BY_MEAN = {
    "meteo_velocidad_viento_media": "meteo_velocidad_viento_n",
    "meteo_temperatura_media": "meteo_temperatura_n",
    "meteo_humedad_relativa_media": "meteo_humedad_relativa_n",
    "meteo_presion_media": "meteo_presion_n",
    "meteo_radiacion_solar_media": "meteo_radiacion_solar_n",
    "meteo_precipitacion_media": "meteo_precipitacion_n",
    "calair_so2_media": "calair_so2_n",
    "calair_co_media": "calair_co_n",
    "calair_no_media": "calair_no_n",
    "calair_no2_media": "calair_no2_n",
    "calair_pm25_media": "calair_pm25_n",
    "calair_pm10_media": "calair_pm10_n",
    "calair_nox_media": "calair_nox_n",
    "calair_o3_media": "calair_o3_n",
}
```

For traffic, `trafico_puntos_n == 0` forbids any finite traffic mean.

- [ ] **Step 9: Run Task 1 tests and commit**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py -k "sanitize or validate_domains or profile_feature_quality"
uvx ruff check src/madrid_ml/preprocessing.py tests/unit/test_ml_preprocessing.py
```

Expected: all selected tests pass; Ruff reports `All checks passed!`.

Commit:

```bash
git add src/madrid_ml/preprocessing.py tests/unit/test_ml_preprocessing.py
git commit -m "feat(ml): add preprocessing sanitation contract"
```

---

### Task 2: Versioned Gold Loading and Temporal Splits

**Files:**
- Modify: `src/madrid_ml/preprocessing.py`
- Modify: `tests/unit/test_ml_preprocessing.py`

**Interfaces:**
- Consumes: `PreprocessingContractError`, `validate_feature_domains` from Task 1 and `validate_matching_lineage` from `madrid_ml.snapshot`.
- Produces: `SnapshotProvenance`, `GoldTrainingSnapshot`, `TemporalSplits`, `load_gold_training_snapshot(...)`, `split_training_snapshot(...)`.

- [ ] **Step 1: Write failing boundary and reconciliation tests**

Add exact boundary rows:

```python
def test_split_uses_utc_half_open_boundaries(spark):
    frame = spark.createDataFrame([
        make_row(feature_hour=datetime(2023, 12, 31, 23, 0), hora_dia=23, mes=12),
        make_row(feature_hour=datetime(2024, 1, 1, 0, 0), hora_dia=0, mes=1),
        make_row(feature_hour=datetime(2025, 1, 1, 0, 0), hora_dia=0, mes=1),
        make_row(feature_hour=datetime(2026, 1, 1, 0, 0), hora_dia=0, mes=1),
    ])

    splits = _split_by_feature_hour(frame)

    assert splits.train.count() == 1
    assert splits.validation.count() == 1
    assert splits.evaluation.count() == 1
    assert splits.excluded.count() == 1
    assert spark.conf.get("spark.sql.session.timeZone") == "Etc/UTC"
```

Add a pure count gate:

```python
def test_validate_split_counts_requires_production_reconciliation():
    validate_split_counts({
        "train": 920_304,
        "validation": 184_464,
        "evaluation": 183_960,
        "excluded": 91_203,
        "total": 1_379_931,
    })

    with pytest.raises(PreprocessingContractError, match="validation"):
        validate_split_counts({
            "train": 920_304,
            "validation": 184_463,
            "evaluation": 183_960,
            "excluded": 91_203,
            "total": 1_379_930,
        })
```

- [ ] **Step 2: Run boundary tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py -k "split_uses_utc or validate_split_counts"
```

Expected: FAIL because split types and functions are absent.

- [ ] **Step 3: Implement provenance and split dataclasses**

```python
@dataclass(frozen=True)
class SnapshotProvenance:
    labels_table: str
    features_table: str
    labels_delta_version: int
    features_delta_version: int
    snapshot_id: str
    input_versions_json: str
    code_commit: str
    feature_schema_version: str
    time_contract: str


@dataclass(frozen=True)
class GoldTrainingSnapshot:
    features: DataFrame
    provenance: SnapshotProvenance


@dataclass(frozen=True)
class TemporalSplits:
    train: DataFrame
    validation: DataFrame
    evaluation: DataFrame
    excluded: DataFrame
    row_counts: dict[str, int]
```

Use these exact contract mappings for predicates, reconciliation and manifest serialization:

```python
SPLIT_PERIODS = {
    "train": ("2019-01-01 00:00:00", "2024-01-01 00:00:00"),
    "validation": ("2024-01-01 00:00:00", "2025-01-01 00:00:00"),
    "evaluation": ("2025-01-01 00:00:00", "2026-01-01 00:00:00"),
    "excluded": ("2026-01-01 00:00:00", None),
}
EXPECTED_SPLIT_ROWS = {
    "train": 920_304,
    "validation": 184_464,
    "evaluation": 183_960,
    "excluded": 91_203,
    "total": 1_379_931,
}
```

Implement `_split_by_feature_hour` after setting and reading back `SESSION_TIMEZONE`. Use `F.to_timestamp(F.lit(...))` and half-open predicates; do not use `year(feature_hour)` as the authoritative split rule.

Implement `validate_split_counts` with exact constants and the reconciliation equation.

- [ ] **Step 4: Run boundary tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing schema and lineage/provenance tests**

Use small labels/features DataFrames with the five lineage columns. Verify:

- matching lineage and `expected_snapshot_id` produce a `SnapshotProvenance`;
- mismatched snapshot or feature schema raises `PreprocessingContractError`;
- explicit Gold versions are preserved separately from `input_versions_json`.
- `_validate_snapshot_schema(...)` accepts exact `LABEL_TABLE_COLUMNS`/`FEATURE_TABLE_COLUMNS` schemas and rejects a missing, extra or wrong-typed column before lineage validation.

Target pure helpers `_validate_snapshot_schema(...)` and `_build_provenance(...)`; do not mock `SparkSession.read` or assert builder call order.

- [ ] **Step 6: Run schema and provenance tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py -k "snapshot_schema or provenance or lineage"
```

Expected: FAIL because `_validate_snapshot_schema` and `_build_provenance` are absent.

- [ ] **Step 7: Implement versioned loading and provenance validation**

Signature:

```text
def load_gold_training_snapshot(
    spark: SparkSession,
    *,
    gold_catalog: str,
    labels_delta_version: int,
    features_delta_version: int,
    expected_snapshot_id: str,
) -> GoldTrainingSnapshot:
```

Behavior:

1. Validate `gold_catalog` with `^[a-z_][a-z0-9_]*$`.
2. Reject negative versions and empty snapshot ID.
3. Resolve `f"{gold_catalog}.ml.accident_labels_hourly"` and `f"{gold_catalog}.ml.features_training_snapshot"`.
4. Require both tables to exist.
5. Read both with `spark.read.format("delta").option("versionAsOf", version).table(name)`.
6. Validate their exact column names, Spark types and nullability against `LABEL_TABLE_COLUMNS` and `FEATURE_TABLE_COLUMNS`.
7. Call `validate_matching_lineage`; wrap `SnapshotContractError` as `PreprocessingContractError` with `raise ... from exc`.
8. Require the observed snapshot, feature schema and time contract to equal the explicit contract.
9. Return the full features DataFrame and explicit Gold provenance. Labels are used only for schema and lineage validation.

- [ ] **Step 8: Implement production splitting and exact count gate**

`split_training_snapshot(snapshot)` must:

1. call `validate_feature_domains` on the raw versioned features;
2. split into four DataFrames;
3. compute train, validation, evaluation, excluded and total counts in one aggregate over the input;
4. validate all exact counts and their reconciliation;
5. return `TemporalSplits`.

Do not cache here. The measured v1 snapshot is small enough to start with direct scans; add bounded persistence only if the smoke evidence shows material repeated I/O, and always unpersist it before exit.

- [ ] **Step 9: Run Task 2 tests and commit**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py -k "split or snapshot_schema or provenance or lineage or load_gold"
uvx ruff check src/madrid_ml/preprocessing.py tests/unit/test_ml_preprocessing.py
```

Expected: all selected tests pass.

Commit:

```bash
git add src/madrid_ml/preprocessing.py tests/unit/test_ml_preprocessing.py
git commit -m "feat(ml): load versioned Gold preprocessing input"
```

---

### Task 3: Fitted Spark Pipeline, Transform Wrapper, and Manifest

**Files:**
- Modify: `src/madrid_ml/preprocessing.py`
- Modify: `tests/unit/test_ml_preprocessing.py`
- Modify: `src/madrid_ml/__init__.py`

**Interfaces:**
- Consumes: `TemporalSplits`, ordered features and sanitation from Tasks 1–2.
- Produces: `PreprocessingManifest`, `FittedPreprocessor`, `PreparedTrainingData`, internal `fit_preprocessor(train, provenance, *, split_rows, spark_version)`, public `prepare_training_data(...)`, and `manifest_to_dict(...)`.

- [ ] **Step 1: Add a complete train fixture plus failing fit and persistence tests**

Build 504 rows so every district/hour/day/month category appears and each numerical feature varies:

```python
def make_complete_training_frame(spark):
    rows = [make_row(index=index) for index in range(504)]
    return spark.createDataFrame(rows)


def fit_test_preprocessor(spark, train):
    train_rows = train.count()
    return fit_preprocessor(
        train,
        make_test_provenance(),
        split_rows={
            "train": train_rows,
            "validation": 0,
            "evaluation": 0,
            "excluded": 0,
            "total": train_rows,
        },
        spark_version=spark.version,
    )
```

Test that validation cannot alter train state:

```python
def test_fit_preprocessor_learns_median_only_from_train(spark):
    train = make_complete_training_frame(spark)
    train = train.withColumn(
        "trafico_intensidad_media",
        F.when(F.col("cod_distrito") == 1, F.lit(None)).otherwise(F.lit(10.0)),
    )
    validation = spark.createDataFrame([
        make_row(index=600, trafico_intensidad_media=1_000_000.0)
    ])

    fitted = fit_test_preprocessor(spark, train)
    transformed = fitted.transform(validation, split_name="validation")

    assert fitted.manifest.imputation_medians["trafico_intensidad_media"] == 10.0
    assert transformed.count() == 1
```

Add three train-state gates before implementation:

```python
def test_fit_preprocessor_rejects_missing_train_category(spark):
    train = make_complete_training_frame(spark).where(F.col("cod_distrito") != 21)
    with pytest.raises(PreprocessingContractError, match="cod_distrito"):
        fit_test_preprocessor(spark, train)


def test_fit_preprocessor_rejects_feature_without_valid_train_values(spark):
    train = make_complete_training_frame(spark).withColumn(
        "calair_so2_media", F.lit(None).cast("double")
    )
    with pytest.raises(PreprocessingContractError, match="calair_so2_media"):
        fit_test_preprocessor(spark, train)


def test_fit_preprocessor_rejects_zero_standard_deviation(spark):
    train = make_complete_training_frame(spark).withColumn(
        "meteo_temperatura_media", F.lit(7.0)
    )
    with pytest.raises(PreprocessingContractError, match="meteo_temperatura_media"):
        fit_test_preprocessor(spark, train)
```

Add the artifact round-trip contract before implementation:

```python
def test_pipeline_model_round_trip_preserves_vector(spark, tmp_path):
    train = make_complete_training_frame(spark)
    fitted = fit_test_preprocessor(spark, train)
    path = str(tmp_path / "pipeline")
    fitted.pipeline_model.write().overwrite().save(path)
    loaded = PipelineModel.load(path)

    expected = fitted.transform(train.limit(5), split_name="train").select("features").collect()
    actual = FittedPreprocessor(loaded, fitted.manifest).transform(
        train.limit(5), split_name="train"
    ).select("features").collect()
    assert actual == expected
```

- [ ] **Step 2: Run the fit and persistence tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py -k "fit_preprocessor or round_trip"
```

Expected: FAIL because fitted types and functions are absent.

- [ ] **Step 3: Define fitted types and exact pipeline columns**

```python
@dataclass(frozen=True)
class PreprocessingManifest:
    preprocessing_contract_version: str
    provenance: SnapshotProvenance
    spark_version: str
    session_timezone: str
    split_periods: dict[str, tuple[str, str | None]]
    split_rows: dict[str, int]
    physical_ranges: dict[str, dict[str, float | bool | None]]
    imputer_relative_error: float
    numeric_features: tuple[str, ...]
    available_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    imputation_medians: dict[str, float]
    scaler_mean: tuple[float, ...]
    scaler_std: tuple[float, ...]
    category_labels: dict[str, tuple[str, ...]]
    category_references: dict[str, str]
    vector_size: int


@dataclass(frozen=True)
class FittedPreprocessor:
    pipeline_model: PipelineModel
    manifest: PreprocessingManifest

    def transform(self, df: DataFrame, *, split_name: str) -> DataFrame:
        if split_name not in {"train", "validation", "evaluation"}:
            raise PreprocessingContractError(f"invalid split_name {split_name!r}")
        prepared = _prepare_for_pipeline(df)
        return self.pipeline_model.transform(prepared)
```

Use exact internal names:

```python
IMPUTED_FEATURES = tuple(f"{name}__imputed" for name in CONTINUOUS_FEATURES)
CATEGORY_FEATURES = ("cod_distrito", "hora_dia", "dia_semana", "mes")
CATEGORY_STRINGS = tuple(f"{name}__category" for name in CATEGORY_FEATURES)
CATEGORY_INDICES = tuple(f"{name}__index" for name in CATEGORY_FEATURES)
CATEGORY_VECTORS = tuple(f"{name}__one_hot" for name in CATEGORY_FEATURES)
NUMERIC_RAW_VECTOR = "__numeric_raw"
NUMERIC_SCALED_VECTOR = "__numeric_scaled"
MODEL_FEATURES_COLUMN = "features"
```

- [ ] **Step 4: Implement deterministic category preparation and pipeline construction**

`_prepare_for_pipeline` must call `sanitize_features`, then format categories:

- district, hour and month: `format_string("%02d", column)`;
- day of week: `format_string("%01d", column)`.

Build stages in this exact order:

```python
stages = [
    Imputer(
        strategy="median",
        relativeError=0.001,
        inputCols=list(CONTINUOUS_FEATURES),
        outputCols=list(IMPUTED_FEATURES),
    ),
    VectorAssembler(
        inputCols=list(IMPUTED_FEATURES + COUNT_FEATURES),
        outputCol=NUMERIC_RAW_VECTOR,
        handleInvalid="error",
    ),
    StandardScaler(
        inputCol=NUMERIC_RAW_VECTOR,
        outputCol=NUMERIC_SCALED_VECTOR,
        withMean=True,
        withStd=True,
    ),
    StringIndexer(
        inputCols=list(CATEGORY_STRINGS),
        outputCols=list(CATEGORY_INDICES),
        stringOrderType="alphabetAsc",
        handleInvalid="error",
    ),
    OneHotEncoder(
        inputCols=list(CATEGORY_INDICES),
        outputCols=list(CATEGORY_VECTORS),
        dropLast=True,
        handleInvalid="error",
    ),
    VectorAssembler(
        inputCols=[NUMERIC_SCALED_VECTOR, *AVAILABLE_FEATURES, *CATEGORY_VECTORS],
        outputCol=MODEL_FEATURES_COLUMN,
        handleInvalid="error",
    ),
]
```

Fit only the prepared train DataFrame.

- [ ] **Step 5: Extract and validate learned state**

After `Pipeline.fit(prepared_train)`:

1. Read imputation medians from `ImputerModel.surrogateDF`.
2. Read `mean` and `std` from `StandardScalerModel` and require 33 entries with every `std > 0`.
3. Read `labelsArray` from `StringIndexerModel`; require exact ordered domains.
4. Read `categorySizes` from `OneHotEncoderModel`; require `(21, 24, 7, 12)`.
5. Build the manifest with the complete `SnapshotProvenance`, exact split periods/counts, Spark/session identity, serialized physical rules and reference categories `("21", "23", "7", "12")`.
6. Fail through `PreprocessingContractError`; do not silently drop constant columns or omit required manifest state.

- [ ] **Step 6: Run the fit and persistence tests and verify GREEN**

Run the command from Step 2.

Expected: both tests pass.

- [ ] **Step 7: Write failing vector, row-conservation and artifact-manifest tests**

Add:

```python
def test_preprocessing_vector_is_finite_and_has_111_components(spark):
    train = make_complete_training_frame(spark)
    fitted = fit_test_preprocessor(spark, train)
    transformed = fitted.transform(train, split_name="train")

    sizes = transformed.select(
        F.size(vector_to_array("features")).alias("size")
    ).distinct().collect()
    assert [row.size for row in sizes] == [111]
    assert transformed.count() == train.count()
    assert transformed.select("cod_distrito", "feature_hour", "target_accident_next_hour").exceptAll(
        train.select("cod_distrito", "feature_hour", "target_accident_next_hour")
    ).count() == 0
```

Add one row containing every sanitation case and assert `vector_to_array("features")` has no null, `NaN` or infinity.

Add a test that checks `manifest.numeric_features == CONTINUOUS_FEATURES + COUNT_FEATURES`; this prevents the schema order from reverting to the interleaved physical Gold order.

Add a wrapper-boundary test that cannot pass until `transform` enforces conservation:

```python
class DropAllRowsModel:
    def transform(self, df):
        return df.limit(0)


def test_transform_rejects_row_loss(spark):
    train = make_complete_training_frame(spark)
    fitted = fit_test_preprocessor(spark, train)
    broken = FittedPreprocessor(DropAllRowsModel(), fitted.manifest)

    with pytest.raises(PreprocessingContractError, match="row"):
        broken.transform(train, split_name="train")
```

Add a focused `test_artifact_manifest_contains_complete_identity` that calls the not-yet-created `manifest_to_dict` and requires the nested Gold lineage commit, distinct preprocessing commit, package version, wheel hash, Spark/Python/runtime identities, exact periods/counts, physical rules and all learned-state fields.

- [ ] **Step 8: Run the new contract tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py -k "vector or row_conservation or rejects_row_loss or manifest_numeric or artifact_manifest"
```

Expected: `test_transform_rejects_row_loss` fails because the wrapper does not yet enforce conservation, and the artifact-manifest test fails because `manifest_to_dict` is absent.

- [ ] **Step 9: Implement transform validation and complete manifest serialization**

Validate transformed output in fail-fast order: row count, exact `(cod_distrito, feature_hour, target)` multiset equality, then one aggregate for vector-size and non-finite counters. This ensures a row-loss defect reports the conservation breach even if the broken output also lacks a valid vector.

Define:

```python
@dataclass(frozen=True)
class PreparedTrainingData:
    train: DataFrame
    validation: DataFrame
    evaluation: DataFrame
    fitted_preprocessor: FittedPreprocessor
    quality_by_split: dict[str, dict[str, dict[str, int]]]
```

The public entrypoint remains exactly the approved five-input boundary:

```text
def prepare_training_data(
    spark: SparkSession,
    *,
    gold_catalog: str,
    labels_delta_version: int,
    features_delta_version: int,
    expected_snapshot_id: str,
) -> PreparedTrainingData:
```

`prepare_training_data(...)` composes versioned load, split, quality profile, train-only fit and three transforms. It passes the already validated `TemporalSplits.row_counts` and `spark.version` into `fit_preprocessor`. It must not expose or transform the excluded 2026 DataFrame.

Serialize the complete retained artifact identity without confusing the Gold-builder commit with the preprocessing commit:

```python
def manifest_to_dict(
    manifest: PreprocessingManifest,
    *,
    preprocessing_code_commit: str,
    wheel_sha256: str,
    package_version: str,
    runtime_environment_version: str,
    python_version: str,
) -> dict[str, object]:
    payload = asdict(manifest)
    payload["artifact_identity"] = {
        "preprocessing_code_commit": preprocessing_code_commit,
        "wheel_sha256": wheel_sha256,
        "package_version": package_version,
        "runtime_environment_version": runtime_environment_version,
        "python_version": python_version,
    }
    return payload
```

Make the focused artifact-manifest test from Step 7 pass with this serialization boundary.

- [ ] **Step 10: Run the serialization round-trip regression**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py::test_pipeline_model_round_trip_preserves_vector
```

Expected: PASS; the loaded `PipelineModel` produces the same vectors through the package wrapper.

- [ ] **Step 11: Run full preprocessing tests and exports**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py
uvx ruff check src/madrid_ml/preprocessing.py tests/unit/test_ml_preprocessing.py
```

Expected: all preprocessing tests pass.

Export only:

```python
from madrid_ml.preprocessing import (
    FittedPreprocessor,
    PreprocessingContractError,
    PreparedTrainingData,
    prepare_training_data,
)
```

Update `src/madrid_ml/__init__.py::__all__` with those four names; keep snapshot exports.

- [ ] **Step 12: Commit**

```bash
git add src/madrid_ml/preprocessing.py src/madrid_ml/__init__.py tests/unit/test_ml_preprocessing.py
git commit -m "feat(ml): fit versioned Spark preprocessing pipeline"
```

---

### Task 4: Databricks Notebook and MLflow Artifact Boundary

**Files:**
- Create: `notebooks/run_ml_preprocessing.py`
- Modify: `tests/unit/test_ml_notebooks.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `prepare_training_data`, `manifest_to_dict`, exact wheel identity, preprocessing source commit, runtime identity and explicit Gold provenance.
- Produces: one notebook result JSON and one MLflow run containing the `PipelineModel`, complete manifest, quality profile and exact wheel.

- [ ] **Step 1: Write the failing notebook SOURCE-boundary test**

Refactor the existing helper in `tests/unit/test_ml_notebooks.py` to accept a path, then test both notebooks:

```python
@pytest.mark.parametrize(
    "notebook_name",
    ["eda_ml_gold.py", "run_ml_preprocessing.py"],
)
def test_ml_markdown_cells_do_not_swallow_python(notebook_name: str) -> None:
    source = (NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8")
    for cell_number, cell in enumerate(source.split(COMMAND_SEPARATOR), start=1):
        lines = [line for line in cell.splitlines() if line.strip()]
        if not lines or lines[0] != "# MAGIC %md":
            continue
        non_magic_lines = [line for line in lines if not line.startswith("# MAGIC")]
        assert non_magic_lines == [], (
            f"Databricks interpretará Python como Markdown en la celda {cell_number}: "
            f"{non_magic_lines[0]}"
        )
```

- [ ] **Step 2: Run the notebook test and verify RED**

Run:

```bash
pytest -q tests/unit/test_ml_notebooks.py
```

Expected: FAIL because `notebooks/run_ml_preprocessing.py` does not exist.

- [ ] **Step 3: Create a thin single-code-cell notebook**

Widgets, all mandatory except catalog:

```text
gold_catalog=dev_gold
labels_delta_version
features_delta_version
expected_snapshot_id
preprocessing_code_commit
wheel_path
wheel_sha256
runtime_environment_version
mlflow_experiment
```

Validate identifiers, non-negative integer versions, 40-character preprocessing commit, 64-character SHA-256, environment version and non-empty paths before Spark actions. Require `wheel_path` to be a readable `/Workspace/.../*.whl` file, compute its SHA-256 inside the notebook and fail if it differs from `wheel_sha256`. Read the installed package version with `importlib.metadata.version("madrid-ingestion")`, Python with `platform.python_version()` and Spark with `spark.version`.

Call:

```python
prepared = prepare_training_data(
    spark=spark,
    gold_catalog=gold_catalog,
    labels_delta_version=labels_delta_version,
    features_delta_version=features_delta_version,
    expected_snapshot_id=expected_snapshot_id,
)
```

- [ ] **Step 4: Log the exact artifact set to MLflow**

Use one run:

```python
artifact_manifest = manifest_to_dict(
    prepared.fitted_preprocessor.manifest,
    preprocessing_code_commit=preprocessing_code_commit,
    wheel_sha256=wheel_sha256,
    package_version=package_version,
    runtime_environment_version=runtime_environment_version,
    python_version=python_version,
)

mlflow.set_experiment(mlflow_experiment)
with mlflow.start_run(run_name=f"preprocessing-{expected_snapshot_id}") as run:
    mlflow.set_tags({
        "madrid_ml.snapshot_id": expected_snapshot_id,
        "madrid_ml.preprocessing_contract_version": "1",
        "madrid_ml.feature_schema_version": "1",
        "madrid_ml.preprocessing_code_commit": preprocessing_code_commit,
        "madrid_ml.snapshot_code_commit": (
            prepared.fitted_preprocessor.manifest.provenance.code_commit
        ),
        "madrid_ml.wheel_sha256": wheel_sha256,
        "madrid_ml.runtime_environment_version": runtime_environment_version,
        "madrid_ml.spark_version": spark.version,
    })
    mlflow.log_dict(artifact_manifest, "preprocessing_manifest.json")
    mlflow.log_dict(prepared.quality_by_split, "quality_by_split.json")
    mlflow.log_artifact(wheel_path, artifact_path="package")
    mlflow.spark.log_model(
        prepared.fitted_preprocessor.pipeline_model,
        artifact_path="preprocessor",
    )
    mlflow_run_id = run.info.run_id
```

Do not log sample rows, credentials, source files or full DataFrames.

- [ ] **Step 5: Reload the Spark model before closing the MLflow run**

Still inside the `with mlflow.start_run(...)` block, load:

```text
    loaded_model = mlflow.spark.load_model(f"runs:/{mlflow_run_id}/preprocessor")
    reloaded = FittedPreprocessor(
        pipeline_model=loaded_model,
        manifest=prepared.fitted_preprocessor.manifest,
    )
```

Transform bounded validation and evaluation samples with both wrappers, convert only those bounded vectors to arrays and require exact equality. This proves artifact loading; it does not refit. Any mismatch raises inside the active MLflow context so the run cannot be marked successful.

- [ ] **Step 6: Return a compact notebook result**

`dbutils.notebook.exit` must include:

```text
status=PREPROCESSING_READY
snapshot_id
labels_delta_version
features_delta_version
preprocessing_contract_version
train_rows=920304
validation_rows=184464
evaluation_rows=183960
excluded_rows=91203
vector_size=111
mlflow_run_id
preprocessing_code_commit
wheel_sha256
quality_by_split
```

Do not include all rows, vectors or medians in the Jobs API result; those belong in the MLflow artifacts.

- [ ] **Step 7: Run notebook test, compile and lint**

Run:

```bash
pytest -q tests/unit/test_ml_notebooks.py
python -m py_compile notebooks/run_ml_preprocessing.py
uvx ruff check notebooks/run_ml_preprocessing.py tests/unit/test_ml_notebooks.py
```

Expected: notebook tests pass, compilation succeeds and Ruff is clean.

- [ ] **Step 8: Document the public boundary**

Update `README.md` with:

- Gold input tables and explicit version/snapshot parameters;
- train/validation/evaluation intervals;
- treatment of missing/invalid values;
- vector size 111;
- `prepare_training_data` example;
- MLflow artifact contents;
- explicit exclusions: no scoring, LightGBM, new Gold table or ingestion changes;
- statement that remote execution requires a separate bounded authorization.

- [ ] **Step 9: Run focused local verification and commit**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py tests/unit/test_ml_notebooks.py
python -m py_compile src/madrid_ml/preprocessing.py notebooks/run_ml_preprocessing.py
uvx ruff check \
  src/madrid_ml/preprocessing.py \
  src/madrid_ml/__init__.py \
  notebooks/run_ml_preprocessing.py \
  tests/unit/test_ml_preprocessing.py \
  tests/unit/test_ml_notebooks.py
git -c core.whitespace=cr-at-eol diff --check
```

Expected: all focused tests pass and all static checks are clean.

Commit:

```bash
git add \
  src/madrid_ml/preprocessing.py \
  src/madrid_ml/__init__.py \
  notebooks/run_ml_preprocessing.py \
  tests/unit/test_ml_preprocessing.py \
  tests/unit/test_ml_notebooks.py \
  README.md
git commit -m "feat(ml): add preprocessing artifact notebook"
```

---

### Task 5: Authorized Serverless Smoke and Evidence

**Files:**
- Create privately during execution: `.team-workspace/run_ml_preprocessing_e2e.sh`
- Create privately during execution: `.team-workspace/ml_preprocessing_e2e_evidence.json`
- Modify after successful smoke: `README.md`

**Interfaces:**
- Consumes: committed notebook and wheel, Gold versions 1/1, snapshot `ea719086-1a93-401c-969b-4e92586e13fd`.
- Produces: terminal Jobs run, persistent MLflow run, private evidence JSON and concise README evidence.

- [ ] **Step 1: Obtain explicit compute and MLflow-write authorization**

Present before submit:

- reads: two Gold Delta tables, versions 1 and 1;
- writes: one MLflow run only; zero Gold writes;
- one serverless notebook task;
- environment version 4;
- timeout 600 seconds;
- retries 0, queue disabled, auto-optimization disabled;
- cleanup order and retained MLflow artifact.

Do not submit without this authorization.

- [ ] **Step 2: Build a private fail-safe controller**

Base it on `.team-workspace/run_ml_eda_e2e.sh`, but use prefix `omp-ml-preprocessing-`. The controller must:

1. require a clean Git tree;
2. build exactly one wheel and record SHA-256;
3. verify both Gold tables plus snapshot metadata without treating Unity Catalog's asynchronous `delta.lastUpdateVersion` property as Delta-history truth;
4. reject existing active prefix runs or saved prefix jobs;
5. create a unique workspace path under `/Workspace/Users/$ME/.tmp/madrid-ml-preprocessing/$TOKEN`;
6. import notebook and wheel;
7. derive MLflow experiment `/Users/$ME/madrid-ml/preprocessing`;
8. submit one idempotent one-shot run;
9. watchdog only its owned run;
10. cancel at 600 seconds and wait for a terminal state;
11. recover output before deleting the notebook;
12. delete the workspace path;
13. verify zero active prefix runs and zero saved jobs;
14. preserve the successful MLflow run;
15. write private evidence.

- [ ] **Step 3: Validate the one-shot payload before submit**

Require with `jq`:

```text
run_name == idempotency_token
timeout_seconds == 600
queue.enabled == false
one task: fit_ml_preprocessing
environment_key == default
task timeout == 600
max_retries == 0
retry_on_timeout == false
disable_auto_optimization == true
no existing_cluster_id
no new_cluster
no job_clusters
environment_version == 4
dependencies == [the same /Workspace wheel path passed to the notebook]
```

Base parameters:

```text
gold_catalog=dev_gold
labels_delta_version=1
features_delta_version=1
expected_snapshot_id=ea719086-1a93-401c-969b-4e92586e13fd
preprocessing_code_commit=$CODE_COMMIT
wheel_path=$WHEEL_WORKSPACE_PATH
wheel_sha256=$WHEEL_SHA256
runtime_environment_version=4
mlflow_experiment=/Users/$ME/madrid-ml/preprocessing
```

The controller resolves `CODE_COMMIT`, `WHEEL_WORKSPACE_PATH`, `WHEEL_SHA256` and `ME` before creating and validating the payload. `WHEEL_WORKSPACE_PATH` is the imported `/Workspace/.../*.whl` path used by both the task dependency and `mlflow.log_artifact`; the committed notebook contains no user name or commit value.

- [ ] **Step 4: Execute and watch the real boundary**

Run the private controller through the supervised process tool. Record parent `run_id` immediately. Poll every 15 seconds and accept only:

```text
life_cycle_state=TERMINATED
result_state=SUCCESS
```

Any other terminal state fails. No retry or timeout increase is automatic.

- [ ] **Step 5: Validate the compact result and MLflow artifact**

Require:

```text
status=PREPROCESSING_READY
snapshot_id=ea719086-1a93-401c-969b-4e92586e13fd
Gold versions 1/1
train_rows=920304
validation_rows=184464
evaluation_rows=183960
excluded_rows=91203
vector_size=111
non-empty mlflow_run_id
matching preprocessing commit, runtime environment version and wheel SHA-256
```

Use MLflow artifact APIs to verify the run contains:

```text
preprocessing_manifest.json
quality_by_split.json
package/$WHEEL_NAME
preprocessor/MLmodel
```

Inspect the compact manifest JSON and require the Gold lineage commit, preprocessing commit, package/wheel identity, Spark/Python/runtime identity, exact periods/counts, physical rules, learned medians/scaler/categories and vector size. Do not download or print full DataFrames.

- [ ] **Step 6: Verify cleanup and immutable inputs**

After terminal state:

```text
active prefix runs = 0
saved prefix jobs = 0
workspace temporary path absent
Gold table metadata unchanged before/after
notebook result reports labels_delta_version=1 and features_delta_version=1
```

Do not infer Delta versions from the potentially stale Unity Catalog `delta.lastUpdateVersion` property. The versioned reads and returned provenance are authoritative for this smoke. The MLflow run remains because it is the approved result, not an ephemeral execution artifact.

- [ ] **Step 7: Record evidence and update README**

Private evidence must contain run/task IDs, MLflow run ID, snapshot-builder and preprocessing commits, wheel/notebook hashes, runtime identities, payload safety settings, duration, result JSON and cleanup state.

Add one concise verified-run section to `README.md` with run ID, MLflow run ID, snapshot, versions, duration, vector size and cleanup result. Do not commit the private controller or evidence.

- [ ] **Step 8: Run final verification and commit documentation**

Run:

```bash
pytest -q tests/unit/test_ml_preprocessing.py tests/unit/test_ml_notebooks.py
python -m py_compile src/madrid_ml/preprocessing.py notebooks/run_ml_preprocessing.py
uvx ruff check \
  src/madrid_ml/preprocessing.py \
  src/madrid_ml/__init__.py \
  notebooks/run_ml_preprocessing.py \
  tests/unit/test_ml_preprocessing.py \
  tests/unit/test_ml_notebooks.py
git -c core.whitespace=cr-at-eol diff --check
```

Expected: all focused tests and static checks pass.

Commit only the evidence summary in README:

```bash
git add README.md
git commit -m "docs(ml): record verified preprocessing smoke"
```

---

## Plan Acceptance Checklist

- Every approved physical range maps to a Spark expression and a focused test.
- The exact 33 numerical inputs and 111-vector ordering are constants, tested and present in the manifest.
- Gold versions and snapshot are explicit inputs; Silver lineage is validated separately.
- UTC boundaries and four exact row counts are tested.
- Only train fits estimators; validation/evaluation call a transform-only wrapper.
- No row, key or label is lost inside an included split.
- The sanitizer is versioned package code; learned state is the Spark `PipelineModel`; both are retained with the manifest.
- The notebook is thin and contains no dataset-specific alternatives beyond the approved contract.
- No ingestion module, source configuration, ADF resource or ingestion job changes.
- The only remote write is the explicitly authorized MLflow run.
- The remote smoke is bounded, observable and cleaned before completion is claimed.
