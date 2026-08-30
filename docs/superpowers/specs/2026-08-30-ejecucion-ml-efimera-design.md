# Ejecución efímera del snapshot ML

## Objetivo

Construir una sola vez `dev_gold.ml.accident_labels_hourly` y `dev_gold.ml.features_training_snapshot` sin crear, desplegar ni ejecutar los nueve jobs de ingesta del repositorio.

## Decisión

Usar `databricks jobs submit` para crear un run efímero. Este endpoint ejecuta el workload sin guardar una definición de job. Los runs enviados así no reintentan al fallar y no usan la auto-optimización serverless.

El Asset Bundle existente queda fuera de esta ejecución porque su primer deploy crearía las diez definiciones de job. La CLI no ofrece selección de un resource key durante `bundle deploy`.

## Artefactos

Antes del run se crea una ruta única bajo el usuario actual:

```text
/Workspace/Users/<usuario>/.tmp/madrid-ml-snapshot/<token>/
```

La ruta contiene únicamente:

- `run_ml_snapshot`, importado como notebook Python en formato `SOURCE`;
- el wheel construido desde el commit Git limpio, importado como fichero `RAW`.

Ambos artefactos se eliminan de forma incondicional al cerrar la ejecución. Las dos tablas Gold son los únicos entregables persistentes.

## Payload del run

El payload contiene una sola `notebook_task` serverless y un entorno versión 4 cuya única dependencia es el wheel temporal. Incluye:

- `timeout_seconds: 900` en run y tarea;
- `max_retries: 0`;
- `retry_on_timeout: false`;
- un token de idempotencia único, reutilizado en `run_name` para poder reconciliar una respuesta ambigua;
- `silver_catalog=dev_silver`;
- `gold_catalog=dev_gold`;
- el commit Git del wheel;
- las cinco versiones Silver autorizadas.

No contiene schedules, triggers, tareas de ingesta, ADF, clúster persistente ni una segunda ejecución.

## Flujo operativo

1. Exigir un checkout limpio y construir el wheel.
2. Obtener el usuario, commit y token de idempotencia; usar el token también como `run_name`.
3. Crear la ruta temporal e importar notebook y wheel.
4. Construir y revisar el payload completo antes de enviarlo.
5. Ejecutar exactamente un `jobs submit --no-wait`.
6. Si la respuesta es ambigua, reconciliar el run por token y parámetros; no reenviar.
7. Esperar un estado terminal. Cancelar el único run si supera 900 segundos o entra en un estado inesperado.
8. Recuperar el resultado de la tarea mediante `jobs get-run-output`.
9. Validar el resultado y ejecutar siempre la limpieza.

## Verificación de éxito

El resultado debe demostrar:

- 1.379.931 filas en cada tabla;
- 21 distritos;
- periodo entre `2019-01-01 00:00:00` y `2026-06-30 22:00:00`;
- cero claves duplicadas, faltantes o extra;
- cero horas de predicción, targets o conteos inválidos;
- lineage idéntico en las dos tablas;
- versiones Silver y commit iguales a los autorizados;
- `reconciled_districts = positive_districts`;
- estado terminal del run.

No se ejecutan consultas adicionales sobre las tablas: el builder vuelve a leer y validar ambas salidas antes de devolver el resultado.

## Limpieza y consumo

La limpieza se ejecuta tanto en éxito como en fallo:

1. cancelar el run si todavía no es terminal;
2. esperar el estado terminal;
3. comprobar que el run ya no está activo;
4. eliminar notebook, wheel y directorio temporal del workspace;
5. eliminar payloads temporales locales;
6. repetir la comprobación de que no queda ejecución activa asociada al token.

Un run terminal y sus metadatos históricos no mantienen compute. No se crea ningún job guardado.

## Exclusiones

- No desplegar el Asset Bundle.
- No crear ni ejecutar jobs de ingesta.
- No modificar ADF, Silver, `conf/**` ni `src/madrid_ingestion/**`.
- No crear schedules, triggers o recurrencia.
- No lanzar un segundo run.
- No entrenar modelos ni ejecutar scoring.
