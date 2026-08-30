# Evidencia de ejecución efímera ML

**Fecha UTC:** 2026-08-30  
**Resultado:** `FAILED`; no se materializó ningún snapshot Gold.  
**Decisión:** no repetir el run sin un diseño revisado y una autorización nueva.

## Frontera autorizada

La autorización cubrió un único `jobs submit` ML idempotente, staging temporal del notebook y del wheel, dos posibles escrituras Gold y la eliminación de cualquier salida creada en caso de fallo. No autorizó `bundle deploy`, `bundle run`, `jobs create`, `jobs run-now` ni la ejecución de jobs de ingesta.

El preflight confirmó que estaban ausentes:

- `dev_gold.ml.accident_labels_hourly`
- `dev_gold.ml.features_training_snapshot`

## Artefactos e inputs congelados

- Usuario: `pedpenaf@ucm.es`
- Commit: `e0fbe7475f70bf229da6f34a7c00bc717cb8d464`
- SHA-256 del wheel: `813e6d7dcfb4686eef2ac709726171a1619712e6c6a82b2efa3b8ab7b1852abd`
- Token y nombre del run: `omp-ml-snapshot-20260830T205309850Z`
- Staging temporal: `/Workspace/Users/pedpenaf@ucm.es/.tmp/madrid-ml-snapshot/omp-ml-snapshot-20260830T205309850Z/`
- Accidentes: `dev_silver.accidentes.accidentes_historico` versión `1`
- Calidad del aire: `dev_silver.calair.calair_historico` versión `3`
- Meteorología: `dev_silver.meteo.meteo_historico` versión `2`
- Distritos: `dev_silver.trafico.dim_distritos` versión `2`
- Tráfico histórico: `dev_silver.trafico.trafico_historico` versión `4`

El payload revisado contenía una única tarea `build_ml_snapshot`, environment serverless versión `4`, `queue.enabled=false`, timeout de 900 segundos, `max_retries=0`, `retry_on_timeout=false` y `disable_auto_optimization=true`. No declaraba clúster ni referencia a un job de ingesta.

## Resultado terminal

- Run efímero: `1012014570864612`
- Task run: `907597204167883`
- Inicio: `2026-08-30T20:53:17.111Z`
- Fin: `2026-08-30T20:54:24.464Z`
- Duración: `67.353 s`
- Intento: `0`
- Estado del run: `INTERNAL_ERROR / FAILED`
- Estado de la tarea: `TERMINATED / FAILED`
- Cancelación o timeout: `false`
- Error: `[NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE is not supported on serverless compute. SQLSTATE: 0A000`

El fallo ocurrió cuando `build_training_snapshot` intentó persistir un agregado mediante `DataFrame.persist`. Serverless rechazó la operación antes de producir el resultado del notebook. Por ello no existen `snapshot_id`, conteos, prevalencia ni validaciones de salida que registrar; atribuirles valores sería incorrecto.

Eliminar la persistencia sin reemplazarla no es una corrección segura: provocaría recomputaciones del scan y de los shuffles del histórico de tráfico. La ruta queda bloqueada hasta elegir y revisar una solución compatible con serverless o un compute clásico acotado.

## Limpieza y postcondiciones verificadas

Tras el estado terminal, el controlador eliminó cualquier tabla Gold creada y todo el staging temporal. Las comprobaciones finales devolvieron:

- `databricks tables list dev_gold ml -o json`: lista vacía.
- `databricks workspace list /Workspace/Users/pedpenaf@ucm.es/.tmp/madrid-ml-snapshot -o json`: lista vacía.
- `databricks jobs list-runs --run-type SUBMIT_RUN --active-only -o json`: lista vacía.
- `databricks jobs list --name omp-ml-snapshot-20260830T205309850Z -o json`: lista vacía; no quedó un job guardado.
- `git status --short --branch`: checkout limpio antes de registrar esta evidencia.

No se desplegó el Asset Bundle y no se creó, modificó ni ejecutó ningún job de ingesta. No se inició un segundo run ML.
