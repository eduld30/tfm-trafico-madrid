# Machine learning

## Flujo operativo

```text
Silver histórico → snapshot managed → preprocessing + entrenamiento → MLflow
                                                                  ↓
Silver NRT → scoring ← modelo UC con alias Champion
```

## Snapshot reproducible

`ml_training_snapshot` captura las versiones Delta de accidentes, distritos,
tráfico histórico, meteorología histórica y calidad del aire histórica. Si se
proporciona `expected_versions_json`, comprueba que coincidan antes de publicar:

```text
<gold_catalog>.ml.accident_labels_hourly
<gold_catalog>.ml.features_training_snapshot
```

Ambas son tablas managed y comparten `snapshot_id`, versiones de entrada,
commit, versión del schema de features y contrato temporal. Cada nueva
ejecución sustituye el snapshot anterior.

## Entrenamiento y registro

`ml_model_training` selecciona el snapshot publicado más reciente. El parámetro
opcional `snapshot_id` sirve como comprobación: no permite apuntar
silenciosamente a otro snapshot.

El flujo ajusta el preprocessing únicamente con train, realiza backtesting
temporal, entrena una regresión logística de Spark ML y evalúa periodos
posteriores. MLflow guarda el manifiesto, el preprocessor, métricas y el
clasificador. No publica automáticamente una versión en Model Registry.

`ml_model_promotion` recibe el `model_run_id` aprobado, registra
`runs:/<run_id>/classifier` como
`<gold_catalog>.ml.accident_risk_model` y mueve el alias `Champion`.

## Jobs y parámetros

- `ml_training_snapshot`: `environment`, `code_commit` y versiones Silver opcionales.
- `ml_model_training`: `environment`, comprobación opcional de `snapshot_id` y experimento MLflow.
- `ml_model_promotion`: `model_run_id`, entorno, modelo y alias.
- `gold_nrt_predictions`: usa el modelo y el alias del entorno configurado.

El entrenamiento utiliza `/Shared/madrid-ml-dev` o `/Shared/madrid-ml-pro`. El
staging temporal de modelos Spark usa
`/Volumes/<gold_catalog>/ml/mlflow_tmp`.

## Requisitos operativos

Antes del primer scoring deben existir:

- las cuatro tablas Silver NRT y de distritos;
- el volumen UC temporal;
- un run de entrenamiento completo;
- el modelo registrado con alias `Champion`.

El snapshot, el entrenamiento y la promoción no se ejecutan desde ADF ni tienen
calendario automático.
