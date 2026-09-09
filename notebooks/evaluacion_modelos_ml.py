# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# ///
# MAGIC %md
# MAGIC # Evaluación de modelos de riesgo de accidente en Madrid
# MAGIC ### Arquitectura end-to-end en Azure · TFM de Big Data & Data Engineering
# MAGIC **Informe exploratorio reproducible basado exclusivamente en artefactos MLflow.**
# MAGIC La unidad de predicción es el **distrito–hora**: presencia de al menos un accidente
# MAGIC en la hora siguiente. No se estima el riesgo individual de un conductor ni el
# MAGIC número de accidentes. Este informe no entrena modelos, no genera predicciones
# MAGIC nuevas y no consulta tablas Gold o Silver.

# COMMAND ----------

import base64
import io
import json
import math
import tempfile
from html import escape
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Trazabilidad y alcance
# MAGIC Se fija un run padre de entrenamiento completo; no se selecciona el «último run».
# MAGIC Se exige que el padre y sus cinco hijos hayan finalizado y pertenezcan al mismo
# MAGIC experimento. Los JSON se descargan a un directorio temporal y se eliminan al
# MAGIC terminar la lectura. Los modelos serializados no se descargan.

# COMMAND ----------

dbutils.widgets.text(  # noqa: F821
    "parent_run_id", "d7e8a05135dd4880b9ec15b4cac658ee", "Run padre de entrenamiento"
)
PARENT_RUN_ID = dbutils.widgets.get("parent_run_id").strip()  # noqa: F821
NAMES = {
    "global_prevalence": "Prevalencia global",
    "district_frequency": "Frecuencia por distrito",
    "district_hour_day_frequency": "Distrito, hora y día semanal",
    "logistic_regression": "Regresión logística",
    "lightgbm": "LightGBM",
}
PERIODS = {"validation": "Validación · 2024", "evaluation": "Evaluación · 2025"}
COLORS = dict(zip(NAMES, ["#8292a2", "#b79b59", "#8663a6", "#2878b5", "#008875"], strict=True))
BASELINE = "district_hour_day_frequency"
client = mlflow.MlflowClient(tracking_uri="databricks")
parent = client.get_run(PARENT_RUN_ID)
if parent.info.status != "FINISHED" or parent.data.tags.get("madrid_ml.execution_kind") == "smoke":
    raise ValueError("El run padre debe ser un entrenamiento completo finalizado, no un smoke.")
children = client.search_runs(
    [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{PARENT_RUN_ID}'"
)
runs = {}
for child in children:
    comparator = child.data.tags.get("madrid_ml.comparator")
    if comparator not in NAMES or comparator in runs or child.info.status != "FINISHED":
        raise ValueError("Hijos incompletos, duplicados o comparadores no reconocidos.")
    runs[comparator] = child
if set(runs) != set(NAMES):
    raise ValueError("Se requieren exactamente los cinco comparadores del entrenamiento.")

with tempfile.TemporaryDirectory(prefix="madrid-model-report-") as temporary:

    def load_json(run_id, artifact):
        location = client.download_artifacts(run_id, artifact, temporary)
        return json.loads(Path(location).read_text(encoding="utf-8"))

    manifest = load_json(PARENT_RUN_ID, "preprocessing_manifest.json")
    evaluations = {
        name: load_json(run.info.run_id, "evaluation.json") for name, run in runs.items()
    }
    backtests = {
        name: load_json(runs[name].info.run_id, "backtest.json")["folds"]
        for name in ("logistic_regression", "lightgbm")
    }

# Reconciliación de agregados: evita mezclar periodos, muestras o artefactos parciales.
for period in PERIODS:
    reference = evaluations["global_prevalence"][period]["global"]
    for name in NAMES:
        part = evaluations[name][period]
        metrics = part["global"]
        for field in ("rows", "positives", "negatives"):
            if metrics[field] != reference[field]:
                raise ValueError(f"Muestras distintas: {name}/{period}/{field}")
        if metrics["rows"] != manifest["split_rows"][period]:
            raise ValueError("Las filas evaluadas no coinciden con el manifiesto.")
        for collection in ("segments", "calibration"):
            for field in ("rows", "positives"):
                if sum(row[field] for row in part[collection]) != metrics[field]:
                    raise ValueError(f"Agregados no reconciliados: {name}/{period}/{collection}")
        for point in part["curves"]:
            tp, fp = point["true_positives"], point["false_positives"]
            if not (0 <= tp <= metrics["positives"] and 0 <= fp <= metrics["negatives"]):
                raise ValueError("Conteos de curva fuera de rango.")
            if not math.isclose(point["recall"], tp / metrics["positives"], abs_tol=1e-10):
                raise ValueError("Recall incompatible con los conteos guardados.")
            if tp + fp and not math.isclose(point["precision"], tp / (tp + fp), abs_tol=1e-10):
                raise ValueError("Precisión incompatible con los conteos guardados.")

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#f6f8fb",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#25364a",
        "text.color": "#25364a",
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.dpi": 120,
    }
)
STYLE = """<style>
.report{font:16px/1.65 system-ui,sans-serif;color:#25364a;max-width:1150px;margin:14px auto}
.report h2{font-size:26px;color:#10283f;border-bottom:3px solid #008875;padding-bottom:10px}
.report h3{color:#008875}
.report .note{background:#edf6f4;border-left:4px solid #008875;padding:14px 20px}
.report .warning{background:#fff7e7;border-left:4px solid #c68a21;padding:14px 20px}
.report table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}
.report th{background:#10283f;color:white;text-align:left;padding:10px}
.report td{padding:9px;border-bottom:1px solid #dde4eb}
.report tr:nth-child(even){background:#f4f7fa}
.report img{width:100%;height:auto}
.report .meta{color:#596b7e;font-size:13px;overflow-wrap:anywhere}
</style>"""


def show(title, body):
    displayHTML(STYLE + f'<section class="report"><h2>{escape(title)}</h2>{body}</section>')  # noqa: F821


def table(frame):
    return frame.to_html(
        index=False, border=0, na_rep="No evaluable", float_format=lambda x: f"{x:.4f}"
    )


def chart(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="Gráfico de evaluación de modelos" src="data:image/png;base64,{encoded}">'


show(
    "Lectura del experimento",
    f"""
<p>Entrenamiento final: <b>2019–2023</b>; comparación en <b>2024</b> y evaluación temporal en
<b>2025</b>. Vector de entrada: <b>{manifest["vector_size"]} atributos</b>.</p>
<div class="note">Las métricas de 2025 se muestran para describir generalización temporal.
No se utilizan aquí para elegir hiperparámetros ni umbrales. Los dos folds internos
validan en 2022 y 2023 usando únicamente años anteriores.</div>
<p class="meta">Run padre: {escape(PARENT_RUN_ID)}<br>Snapshot:
{escape(manifest["provenance"]["snapshot_id"])}<br>Commit de entrenamiento:
{escape(parent.data.tags.get("madrid_ml.code_commit", "No registrado"))}</p>
""",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Comparación global: ¿aportan señal los modelos?
# MAGIC **AP** resume la curva precisión–recall respetando empates; mayor es mejor.
# MAGIC **ROC AUC** mide discriminación; 0,5 corresponde a ausencia de discriminación.
# MAGIC **Brier** mide el error cuadrático de las probabilidades; menor es mejor.
# MAGIC La prevalencia es una referencia esencial para AP en un problema desbalanceado.

# COMMAND ----------

comparison_rows = []
for period, period_label in PERIODS.items():
    reference = evaluations[BASELINE][period]["global"]
    constant = evaluations["global_prevalence"][period]["global"]
    for name, label in NAMES.items():
        m = evaluations[name][period]["global"]
        comparison_rows.append(
            {
                "Periodo": period_label,
                "Modelo": label,
                "Filas": m["rows"],
                "Positivos": m["positives"],
                "Prevalencia": m["prevalence"],
                "AP": m["average_precision"],
                "ROC AUC": m["roc_auc"],
                "Brier": m["brier_score"],
                "Δ AP vs. base temporal": m["average_precision"] - reference["average_precision"],
                "Mejora relativa AP (%)": 100
                * (m["average_precision"] / reference["average_precision"] - 1),
                "Reducción Brier vs. constante (%)": 100
                * (1 - m["brier_score"] / constant["brier_score"]),
            }
        )
comparison = pd.DataFrame(comparison_rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
for ax, (period, label) in zip(axes, PERIODS.items(), strict=True):
    values = [evaluations[name][period]["global"]["average_precision"] for name in NAMES]
    ax.barh(list(NAMES.values()), values, color=list(COLORS.values()))
    ax.set(title=label, xlabel="Average Precision (mayor es mejor)")
    ax.invert_yaxis()
    for index, value in enumerate(values):
        ax.text(value + 0.002, index, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.2)
fig.tight_layout()
insights = []
for period, label in PERIODS.items():
    m = evaluations["lightgbm"][period]["global"]
    b = evaluations[BASELINE][period]["global"]
    lr = evaluations["logistic_regression"][period]["global"]
    insights.append(
        f"<li><b>{label}:</b> LightGBM obtiene AP={m['average_precision']:.4f}, "
        f"{m['average_precision'] / m['prevalence']:.2f} veces la prevalencia; "
        f"su mejora relativa frente a la base temporal es "
        f"{100 * (m['average_precision'] / b['average_precision'] - 1):.1f} %. "
        f"La diferencia absoluta frente a logística es "
        f"{m['average_precision'] - lr['average_precision']:.4f}.</li>"
    )
show(
    "1 · Señal predictiva y valor incremental",
    chart(fig)
    + table(comparison)
    + "<ul>"
    + "".join(insights)
    + "</ul><div class='note'>La comparación relevante no es solo "
    "contra una probabilidad constante: la base distrito–hora–día ya captura patrones "
    "históricos fuertes. Las mejoras del modelo deben interpretarse sobre esa referencia. "
    "La diferencia pequeña entre los dos modelos mantiene a logística como alternativa "
    "interpretable. No se ha calculado significación estadística.</div>",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Precisión–recall y volumen de alertas
# MAGIC Se representan los puntos guardados (hasta 101 umbrales observados por modelo).
# MAGIC Las líneas unen puntos para facilitar la lectura: no reconstruyen la curva completa.
# MAGIC **No se recalcula AP integrando esta representación reducida.** Una alerta representa
# MAGIC una celda distrito–hora clasificada como positiva, no un accidente evitado.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (period, label) in zip(axes, PERIODS.items(), strict=True):
    for name, model_label in NAMES.items():
        points = pd.DataFrame(evaluations[name][period]["curves"]).sort_values(
            ["recall", "precision"], ascending=[True, False]
        )
        ax.plot(
            points.recall,
            points.precision,
            label=model_label,
            color=COLORS[name],
            marker="o" if len(points) == 1 else None,
        )
    ax.set(
        title=label,
        xlabel="Recall · proporción de positivos detectados",
        ylabel="Precisión · positivos entre las alertas",
        xlim=(0, 1),
        ylim=(0, 1),
    )
axes[0].legend(fontsize=8, loc="upper right")
fig.tight_layout()
alert_rows = []
# Escenarios ilustrativos; no límites operativos aprobados ni optimización en 2025.
for name in (BASELINE, "logistic_regression", "lightgbm"):
    points = pd.DataFrame(evaluations[name]["validation"]["curves"])
    total = evaluations[name]["validation"]["global"]["rows"]
    points["alert_fraction"] = (points.true_positives + points.false_positives) / total
    for budget in (0.05, 0.10, 0.20):
        feasible = points[points.alert_fraction <= budget].sort_values("alert_fraction")
        if feasible.empty:
            continue
        point = feasible.iloc[-1]
        alert_rows.append(
            {
                "Modelo": NAMES[name],
                "Escenario: máximo de celdas (%)": 100 * budget,
                "Celdas alertadas (%)": 100 * point.alert_fraction,
                "Umbral guardado": point.threshold,
                "Precisión": point.precision,
                "Recall": point.recall,
                "Verdaderos positivos": int(point.true_positives),
                "Falsos positivos": int(point.false_positives),
            }
        )
alerts = pd.DataFrame(alert_rows)
show(
    "2 · De una métrica agregada a un posible uso",
    chart(fig)
    + table(alerts)
    + "<div class='warning'>Escenarios descriptivos de 2024: se elige el punto guardado con "
    "más alertas sin superar el presupuesto ilustrativo. Los empates pueden producir "
    "saltos y algunos presupuestos no serán alcanzables. No se interpolan umbrales ni "
    "se transfieren a 2025, cuyos puntos guardados pueden tener otros umbrales.</div>"
    "<p>El criterio práctico depende del coste de una falsa alerta y de omitir una celda "
    "positiva. Un AP mayor no garantiza utilidad a cualquier volumen de alertas. "
    "Antes de una decisión operativa haría falta definir esos costes y validar un umbral "
    "fijado exclusivamente con datos anteriores.</p>",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Calibración: ¿son interpretables las probabilidades?
# MAGIC Los artefactos agrupan probabilidades en intervalos de anchura 0,1.
# MAGIC La diagonal indica concordancia entre probabilidad media y frecuencia observada.
# MAGIC El tamaño de los marcadores representa el número de observaciones del intervalo.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
calibration_rows = []
for ax, (period, label) in zip(axes, PERIODS.items(), strict=True):
    ax.plot([0, 1], [0, 1], "--", color="#8292a2", label="Concordancia ideal")
    for name in ("logistic_regression", "lightgbm"):
        bins = pd.DataFrame(evaluations[name][period]["calibration"]).sort_values("calibration_bin")
        sizes = 20 + 280 * bins.rows / bins.rows.max()
        ax.plot(bins.mean_score, bins.observed_frequency, color=COLORS[name], alpha=0.6)
        ax.scatter(
            bins.mean_score, bins.observed_frequency, s=sizes, color=COLORS[name], label=NAMES[name]
        )
        for row in bins.to_dict("records"):
            calibration_rows.append(
                {
                    "Periodo": label,
                    "Modelo": NAMES[name],
                    "Bin": row["calibration_bin"],
                    "Filas": row["rows"],
                    "Positivos": row["positives"],
                    "Probabilidad media": row["mean_score"],
                    "Frecuencia observada": row["observed_frequency"],
                    "Diferencia": row["mean_score"] - row["observed_frequency"],
                }
            )
    ax.set(
        title=label,
        xlabel="Probabilidad predicha media",
        ylabel="Frecuencia observada",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)
fig.tight_layout()
calibration = pd.DataFrame(calibration_rows)
show(
    "3 · Probabilidades, soporte y prudencia",
    chart(fig)
    + table(calibration)
    + "<div class='note'>Los intervalos con muchas filas son más informativos que las colas "
    "con pocos casos. Una frecuencia cero en cuatro observaciones no demuestra que una "
    "probabilidad de 0,4 sea incorrecta. La tabla permite ver ese soporte directamente.</div>"
    "<p>Un Brier menor indica mejor error probabilístico global, pero no demuestra por sí "
    "solo buena calibración. Estos bins pueden ocultar errores dentro de cada intervalo; "
    "no se han estimado bandas de incertidumbre ni se ha ajustado un calibrador.</p>",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distrito y hora: localizar heterogeneidad sin promediar métricas incorrectamente
# MAGIC Cada fila guardada representa una combinación distrito–hora del día.
# MAGIC La diferencia de AP se calcula frente a la base temporal en **la misma celda**.
# MAGIC No se promedian AP o AUC para inventar una métrica global por distrito u hora.
# MAGIC Sí se pueden agregar conteos y Brier ponderándolo por el número de filas.

# COMMAND ----------

segment_frames = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
for period in PERIODS:
    model = pd.DataFrame(evaluations["lightgbm"][period]["segments"])
    baseline = pd.DataFrame(evaluations[BASELINE][period]["segments"])
    merged = model.merge(
        baseline, on=["cod_distrito", "hora_dia"], suffixes=("", "_base"), validate="one_to_one"
    )
    if len(merged) != len(model) or len(merged) != len(baseline):
        raise ValueError("Celdas no comparables entre modelo y baseline.")
    if not (merged.rows.eq(merged.rows_base) & merged.positives.eq(merged.positives_base)).all():
        raise ValueError("Muestras de segmentos distintas.")
    merged["delta_ap"] = merged.average_precision - merged.average_precision_base
    segment_frames[period] = merged
limit = max(frame.delta_ap.abs().max() for frame in segment_frames.values())
for ax, (period, label) in zip(axes, PERIODS.items(), strict=True):
    pivot = segment_frames[period].pivot(
        index="cod_distrito", columns="hora_dia", values="delta_ap"
    )
    image = ax.imshow(pivot, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set(
        title=label,
        xlabel="Hora del día",
        xticks=range(0, 24, 3),
        xticklabels=range(0, 24, 3),
        yticks=range(len(pivot)),
        yticklabels=pivot.index,
    )
axes[0].set_ylabel("Código de distrito")
fig.colorbar(image, ax=axes, label="Δ AP: LightGBM − base temporal", shrink=0.7)
segment_tables = []
for period, label in PERIODS.items():
    frame = segment_frames[period]
    evaluable = frame.delta_ap.notna().sum()
    improved = frame.delta_ap.gt(0).sum()
    segment_tables.append(
        f"<p><b>{label}:</b> mejora de AP en {improved} de {evaluable} celdas evaluables "
        f"({100 * improved / evaluable:.1f} %). No es una tasa de significación estadística.</p>"
    )
    weak = frame.nsmallest(10, "delta_ap")[
        [
            "cod_distrito",
            "hora_dia",
            "rows",
            "positives",
            "prevalence",
            "average_precision",
            "average_precision_base",
            "delta_ap",
        ]
    ]
    weak = weak.rename(
        columns={
            "cod_distrito": "Distrito",
            "hora_dia": "Hora",
            "rows": "Filas",
            "positives": "Positivos",
            "prevalence": "Prevalencia",
            "average_precision": "AP LightGBM",
            "average_precision_base": "AP base",
            "delta_ap": "Δ AP",
        }
    )
    segment_tables.append(f"<h3>{label}: diez celdas con menor diferencia de AP</h3>" + table(weak))
    for group, group_label in [("cod_distrito", "Distrito"), ("hora_dia", "Hora")]:
        weighted = frame.assign(
            brier_total=frame.brier_score * frame.rows,
            baseline_total=frame.brier_score_base * frame.rows,
        )
        aggregate = weighted.groupby(group)[
            ["rows", "positives", "brier_total", "baseline_total"]
        ].sum()
        aggregate["Prevalencia"] = aggregate.positives / aggregate.rows
        aggregate["Brier LightGBM"] = aggregate.brier_total / aggregate.rows
        aggregate["Brier base"] = aggregate.baseline_total / aggregate.rows
        aggregate = (
            aggregate.drop(columns=["brier_total", "baseline_total"])
            .reset_index()
            .rename(columns={group: group_label, "rows": "Filas", "positives": "Positivos"})
        )
        segment_tables.append(
            f"<details><summary>{label}: detalle por {group_label.lower()}</summary>"
            + table(aggregate)
            + "</details>"
        )
show(
    "4 · Dónde se concentra la mejora y dónde no",
    chart(fig)
    + "".join(segment_tables)
    + "<div class='warning'>Este ranking es exploratorio, no una prueba de inferioridad "
    "estadística. El número de positivos puede ser pequeño; además se comparan muchas "
    "celdas. AP depende de la prevalencia y no debe compararse entre distritos sin ese "
    "contexto. Los códigos se conservan tal como fueron registrados: no se inventan nombres "
    "geográficos mediante una tabla externa.</div>",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Backtesting temporal y conclusiones
# MAGIC Fold 1: entrenamiento 2019–2021 y validación 2022.
# MAGIC Fold 2: entrenamiento 2019–2022 y validación 2023.
# MAGIC Las configuraciones seleccionadas se reajustan posteriormente con 2019–2023.
# MAGIC Los puntos de 2024/2025 proceden de ese ajuste final, no de modelos ajustados cada año.

# COMMAND ----------

fold_rows = []
for name in NAMES:
    folds = backtests[name] if name in backtests else evaluations[name]["backtest"]
    for fold in folds:
        fold_rows.append(
            {
                "Modelo": NAMES[name],
                "Configuración": fold.get("config_name", "Histórica"),
                "Fold": fold["fold_name"],
                "AP": fold["average_precision"],
            }
        )
fold_table = pd.DataFrame(fold_rows)
fig, ax = plt.subplots(figsize=(10, 4.5))
temporal_series = {}
for name in NAMES:
    if name in backtests:
        selected = runs[name].data.params["config_name"]
        entries = [row for row in backtests[name] if row["config_name"] == selected]
    else:
        entries = evaluations[name]["backtest"]
    fold_values = {row["fold_name"]: row["average_precision"] for row in entries}
    values = [
        fold_values["fold_1"],
        fold_values["fold_2"],
        evaluations[name]["validation"]["global"]["average_precision"],
        evaluations[name]["evaluation"]["global"]["average_precision"],
    ]
    temporal_series[name] = values
    ax.plot(
        ["2022 · fold 1", "2023 · fold 2", "2024 · validación", "2025 · evaluación"],
        values,
        marker="o",
        color=COLORS[name],
        label=NAMES[name],
    )
ax.axvline(1.5, color="#8292a2", linestyle="--")
ax.set(ylabel="Average Precision", title="Evolución temporal · configuraciones seleccionadas")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
gains = [
    100 * (model / baseline - 1)
    for model, baseline in zip(temporal_series["lightgbm"], temporal_series[BASELINE], strict=True)
]
temporal_insight = (
    "La mejora relativa de AP de LightGBM frente a la base temporal es "
    + ", ".join(
        f"{year}: {gain:+.1f} %" for year, gain in zip([2022, 2023, 2024, 2025], gains, strict=True)
    )
    + ". Son comparaciones descriptivas; no intervalos de confianza."
)
show(
    "5 · Consistencia temporal y conclusión del experimento",
    table(fold_table)
    + chart(fig)
    + f"<div class='note'><b>Resultado temporal:</b> {temporal_insight} "
    "La mayor complejidad de un modelo no implica una mejora proporcional.</div>"
    "<h3>Qué no podemos concluir</h3><ul>"
    "<li>No se demuestra causalidad: el modelo no estima el efecto de una intervención.</li>"
    "<li>No se demuestra utilidad operacional sin costes, capacidad de respuesta "
    "y umbral fijado.</li>"
    "<li>Los dos folds no constituyen una muestra suficiente para afirmar "
    "significación estadística.</li>"
    "<li>La estabilidad retrospectiva no verifica la disponibilidad de variables en tiempo real "
    "ni la equivalencia del futuro pipeline NRT.</li>"
    "<li>El informe no añade validación de fuga temporal: hereda el contrato del snapshot "
    "y del entrenamiento; cualquier interpretación depende de su corrección.</li></ul>"
    "<h3>Siguiente decisión, sin más entrenamiento</h3><p>Documentar un escenario de uso "
    "concreto y revisar en 2024 los compromisos de alertas, la calibración y los segmentos "
    "débiles. Mantener 2025 como evaluación descriptiva, sin ajustar decisiones para mejorar "
    "retrospectivamente sus métricas.</p>",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Referencias y cierre
# MAGIC - [MLflow: artefactos y cliente de seguimiento](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
# MAGIC - [Average Precision: definición y diferencia frente a integración trapezoidal](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
# MAGIC - [Calibración de probabilidades](https://scikit-learn.org/stable/modules/calibration.html)
# MAGIC - [Tareas notebook en Azure Databricks](https://learn.microsoft.com/en-us/azure/databricks/jobs/tasks/notebook)
# MAGIC El informe conserva resultados de lectura; no crea nuevos runs MLflow
# MAGIC ni modifica los modelos.

# COMMAND ----------

report_result = {
    "status": "MODEL_EVALUATION_REPORT_COMPLETE",
    "parent_run_id": PARENT_RUN_ID,
    "child_run_ids": {name: run.info.run_id for name, run in runs.items()},
    "sections": 5,
    "source": "mlflow_json_artifacts_only",
    "training_executed": False,
    "tables_read": [],
    "models_loaded": [],
    "matplotlib_version": matplotlib.__version__,
}
dbutils.notebook.exit(json.dumps(report_result, sort_keys=True))  # noqa: F821
