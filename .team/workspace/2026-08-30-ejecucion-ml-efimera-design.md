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

El preflight exige que ninguna de las dos tablas Gold exista. En éxito son los únicos entregables persistentes. Si el run falla después de crear una o ambas tablas, la limpieza las elimina mediante la API de Unity Catalog y verifica que vuelven a estar ausentes; no se deja una pareja con lineage divergente.

## Payload del run

El payload contiene una sola `notebook_task` serverless y un entorno versión 4 cuya única dependencia es el wheel temporal. Incluye:

- `timeout_seconds: 900` en run y tarea;
- `max_retries: 0`;
- `retry_on_timeout: false`;
- un token de idempotencia único, reutilizado en `run_name`; ante una respuesta ambigua se repite de forma acotada el mismo submit con el payload y token idénticos, que la API garantiza que corresponden a un único run;
- `silver_catalog=dev_silver`;
- `gold_catalog=dev_gold`;
- el commit Git del wheel;
- las cinco versiones Silver autorizadas.

No contiene schedules, triggers, tareas de ingesta, ADF, clúster persistente ni una segunda ejecución.

## Flujo operativo

1. Corregir el README para retirar la ruta operativa mediante el bundle y exigir un checkout limpio.
2. Construir un wheel único y obtener usuario, commit y token de idempotencia.
3. Consultar por API de Unity Catalog que las dos tablas Gold no existen; cualquier tabla existente bloquea esta ejecución y exige una decisión nueva.
4. Inicializar todas las variables del controlador e instalar un `trap` para `EXIT`, `INT` y `TERM` antes de la primera mutación remota.
5. Crear la ruta temporal e importar notebook y wheel.
6. Construir y revisar el payload completo antes de enviarlo.
7. Ejecutar el submit. Si la respuesta es ambigua, repetir bajo un deadline local el mismo payload con el mismo token hasta recuperar el único `run_id`; `list-runs --run-type SUBMIT_RUN` aporta evidencia secundaria.
8. Un watchdog consulta solo ese run y lo cancela al alcanzar 900 segundos o ante una configuración inesperada.
9. Recuperar el resultado mediante `jobs get-run-output`, validarlo y dejar que el `trap` ejecute siempre la limpieza.

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

## Fallo parcial de publicación

El builder publica labels antes que features. Por ello esta ejecución efímera solo se autoriza cuando ambas salidas están ausentes. En cualquier resultado distinto de `TERMINATED/SUCCESS`, el controlador espera que el run sea terminal y elimina mediante `databricks tables delete` cualquiera de las dos tablas que haya sido creada. Después consulta ambas por la API de Unity Catalog y acepta como ausencia tanto `RESOURCE_DOES_NOT_EXIST` como los errores explícitos `Schema '<schema>' does not exist` y `Table '<full_name>' does not exist`.

No se intenta restaurar una versión anterior: encontrar una tabla preexistente detiene el preflight antes del submit. Esta restricción evita borrar o reemplazar una salida cuyo propietario o estado previo no estén dentro de la autorización.

## Limpieza y consumo

La ejecución usa un único controlador finito. Antes de `workspace mkdirs`, el controlador inicializa `run_id`, token, rutas y estado de las tablas, e instala un `trap` para `EXIT`, `INT` y `TERM`. El trap desactiva sus propias señales para no ejecutarse dos veces y:

1. resuelve el único `run_id` con el token idempotente si la respuesta inicial fue ambigua;
2. cancela el run si todavía no es terminal y espera el estado terminal;
3. recupera el output si está disponible;
4. en fallo, elimina las dos tablas creadas y verifica que ambas quedan ausentes;
5. elimina notebook, wheel y directorio temporal del workspace;
6. elimina payloads y builds temporales locales;
7. ejecuta `list-runs --run-type SUBMIT_RUN --active-only` y exige cero coincidencias para `run_name`.

Un run terminal y sus metadatos históricos no mantienen compute. No se crea ningún job guardado.

## Exclusiones

- No desplegar el Asset Bundle.
- No crear ni ejecutar jobs de ingesta.
- No modificar ADF, Silver, `conf/**` ni `src/madrid_ingestion/**`.
- No crear schedules, triggers o recurrencia.
- No lanzar un segundo run; las repeticiones de control usan el mismo payload y token idempotente y resuelven el mismo run.
- No entrenar modelos ni ejecutar scoring.
