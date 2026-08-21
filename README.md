# Madrid Ingestion

Motor configurable de ingesta Bronze/Silver para el TFM de monitorización y
análisis del tráfico de Madrid. Se ejecuta sobre Azure Databricks, escribe tablas
Delta externas en ADLS y las registra en Unity Catalog.

La lógica reside en el paquete Python. Los YAML describen fuentes, datasets,
lectura, transformaciones y estrategia de escritura sin incluir código
ejecutable ni secretos.

## Arquitectura implementada

```text
landing (CSV/XML)
  -> Bronze: Auto Loader + metadatos técnicos + Delta externo
  -> Silver: lectura incremental + transformaciones + overwrite/merge
  -> Unity Catalog: <entorno>_<capa>.<fuente>.<dataset>
```

Bronze usa Auto Loader con `trigger(availableNow=True)`. Cada dataset mantiene
un único checkpoint y schema location bajo su propia ubicación Delta:

```text
abfss://bronze@<cuenta>.dfs.core.windows.net/<fuente>/<dataset>/
  _delta_log/
  _checkpoint/
```

Silver lee exclusivamente la tabla Bronze correspondiente. Ambas capas se
registran como tablas externas; eliminar una tabla de Unity Catalog no elimina
sus datos en ADLS.

## Datasets configurados

Las cinco fuentes y sus catorce datasets tienen Bronze y Silver habilitados.
Las ejecuciones agrupadas respetan el orden de los datasets en cada YAML, donde
las dimensiones aparecen antes que los datasets que las consultan.

| Fuente | Dataset | Formato Bronze | Escritura Silver |
|---|---|---|---|
| `trafico` | `dim_trafico` | CSV | `overwrite` |
| `trafico` | `dim_distritos` | CSV | `overwrite` |
| `trafico` | `trafico_historico` | CSV | `merge` |
| `trafico` | `trafico_nrt` | XML | `merge` |
| `accidentes` | `accidentes_historico` | CSV | `replace_partitions` por año |
| `meteo` | `dim_meteo` | CSV | `overwrite` |
| `meteo` | `dim_meteo_magnitudes` | CSV | `overwrite` |
| `meteo` | `meteo_nrt` | CSV | `merge` |
| `meteo` | `meteo_historico` | CSV | `merge` |
| `eventos` | `eventos_culturales` | CSV | `merge` |
| `calair` | `dim_calair` | CSV | `overwrite` |
| `calair` | `dim_calair_magnitudes` | CSV | `overwrite` |
| `calair` | `calair_nrt` | CSV | `merge` |
| `calair` | `calair_historico` | CSV | `merge` |

Las claves de negocio, casts, formatos de fecha y enriquecimientos concretos
están declarados en `conf/sources/*.yaml`.

## Incrementalidad

Auto Loader identifica en Bronze los ficheros nuevos mediante el checkpoint del
dataset. Cada registro Bronze incorpora:

- `_ingestion_timestamp`
- `_ingestion_run_id`
- `_source_file`
- `_source_file_modification_time`
- `_file_date`, extraído de un segmento `YYYY/MM/DD` de la ruta del fichero

Silver aplica el mismo criterio incremental en todos los modos: procesa las
filas Bronze cuyo `_ingestion_timestamp` sea posterior al máximo almacenado en
Silver. Si no hay filas nuevas, finaliza sin escribir.

- `merge`: actualiza o inserta el lote nuevo mediante `business_keys`.
- `append`: añade el lote nuevo.
- `overwrite`: dentro del lote nuevo selecciona el mayor `_file_date` y usa esa
  versión completa para sustituir Silver.
- `replace_partitions`: conserva la fotografía con mayor `_file_date` de cada
  partición recibida y reemplaza esas particiones mediante `replaceWhere`. Se
  utiliza en accidentes para renovar el año en curso sin borrar años anteriores
  ni deduplicar personas implicadas.

Silver añade además `_silver_processed_timestamp`. El diseño presupone una sola
versión completa por dataset y fecha para las cargas `overwrite`.

## Configuración

```text
conf/
  environments/   # storage, contenedores, catálogos y runtime
  sources/        # datasets y comportamiento Bronze/Silver
  schemas/        # reservado para esquemas explícitos confirmados
```

Los entornos disponibles son `dev` y `pro`, con catálogos que siguen la
convención `<entorno>_bronze`, `<entorno>_silver` y `<entorno>_gold`. La
autenticación contra ADLS se delega al entorno Databricks; los YAML no contienen
claves, tokens ni SAS.

`source_path` siempre apunta a la raíz estable del dataset en landing. Los
ficheros pueden estar organizados debajo mediante rutas `YYYY/MM/DD[/HH]`, pero
la ruta configurada no cambia entre ejecuciones.

Bronze admite exclusivamente `normalize_column_names` y `rename`. Silver
dispone de:

```text
normalize_column_names, rename, select, drop, cast, trim, upper,
strip_accents, empty_to_null, replace_values, regex_replace, filter,
add_literal, parse_timestamp, parse_date, extract_year, hourly_wide_to_long,
deduplicate, lookup_join, assign_district
```

Las transformaciones se ejecutan en el orden declarado. `lookup_join` resuelve
tablas Bronze o Silver mediante catálogo, fuente y dataset; el mapa `select`
define `columna_destino: columna_lookup`.

Al adoptar `replace_partitions`, una tabla ya creada sin particionar no puede
convertirse en particionada mediante una escritura incremental. La primera
ejecución de `accidentes.accidentes_historico` requiere eliminar explícitamente
el registro Silver anterior y su ruta física para reconstruirla particionada por
`anio_accidente`. No es necesario reiniciar Bronze ni su checkpoint.

## Instalación

El paquete requiere Python 3.10 o superior. En Databricks, PySpark y las APIs
Delta ya forman parte del runtime:

```bash
pip install -e .
```

Para desarrollo local y pruebas unitarias no se necesita Spark:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

El extra `databricks` instala `pyspark` y `delta-spark` únicamente para entornos
locales que necesiten esas APIs.

## CLI

Validar toda la configuración o listar una fuente no requiere Spark:

```bash
python -m madrid_ingestion validate-config --env dev
python -m madrid_ingestion list-datasets --source trafico
```

Ejecutar un dataset y una capa:

```bash
python -m madrid_ingestion run --env dev --layer bronze \
  --source trafico --dataset trafico_nrt
python -m madrid_ingestion run --env dev --layer silver \
  --source trafico --dataset trafico_nrt
```

Ejecutar secuencialmente una fuente completa o todos los datasets habilitados:

```bash
python -m madrid_ingestion run --env dev --layer bronze --source trafico
python -m madrid_ingestion run --env dev --layer bronze --all
```

`run` también acepta `--run-id`, `--config-root` y `--log-level`. Sin
`--dataset`, se ejecutan los datasets habilitados de la fuente. `--all` no se
puede combinar con `--source` ni `--dataset`.

## Databricks Asset Bundle

El repositorio incluye un bundle para desplegar nueve jobs de ejecución manual:

```text
ingesta_dimensiones
ingesta_trafico_nrt
ingesta_meteo_nrt
ingesta_calair_nrt
ingesta_eventos
ingesta_trafico_historico
ingesta_accidentes_historico
ingesta_meteo_historico
ingesta_calair_historico
```

Cada dataset se ejecuta mediante dos tareas `python_wheel_task` con dependencia
explícita `Bronze -> Silver`. Los jobs no tienen calendario, `run_as` ni clúster
asignado: utilizan serverless compute con el entorno `default`. En el job de
dimensiones, las seis ramas se ejecutan en paralelo y cada tarea Silver depende
únicamente de la carga Bronze de su propio dataset.

El bundle construye el wheel del paquete, sincroniza `conf/` y despliega los
targets lógicos `dev` y `pro`. Para preparar el entorno local y desplegar en
desarrollo:

```bash
python -m pip install -e ".[dev]"
databricks auth login --host https://<workspace>.azuredatabricks.net
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

Si se utiliza un perfil distinto del predeterminado, añadir
`--profile <perfil>` a los comandos del bundle. Tras el despliegue, los jobs se
pueden lanzar desde `Workflows > Jobs & Pipelines` en la UI de Databricks.
Ejecutar primero `ingesta_dimensiones`; los restantes jobs consultan esas
dimensiones desde sus transformaciones Silver.

La versión del artefacto se hace dinámica en cada despliegue para que serverless
no reutilice un wheel anterior con el mismo número de versión del proyecto.
La integración con ADF y los calendarios se mantienen fuera de esta primera
iteración.

## API Python

```python
from madrid_ingestion.runner import run_dataset

result = run_dataset(
    environment="dev",
    layer="bronze",
    source="trafico",
    dataset="trafico_nrt",
)
```

Puede inyectarse una `SparkSession` mediante `spark`. En Databricks, si no se
indica, se utiliza la sesión activa. La ejecución devuelve un `RunResult`, no un
DataFrame.

## XML

El tráfico NRT usa Auto Loader con `cloudFiles.format=xml`. El lector traduce
`row_tag` a `rowTag` y permite declarar `record_path` y `parent_columns` para
expandir los registros XML antes de escribir Bronze. El runtime debe incluir
soporte XML, nativo en versiones recientes de Databricks Runtime.

## Enriquecimiento geográfico de estaciones

El paquete incluye como recurso estático el KML oficial con los 21 distritos de
Madrid. La transformación Silver `assign_district` utiliza Shapely en el driver
para añadir `distrito_cod` y `distrito_nombre` directamente a `dim_meteo` y
`dim_calair`. Es una operación deliberadamente acotada a dimensiones de pocas
decenas de estaciones; no requiere procesamiento espacial distribuido.

Una estación fuera de los polígonos o situada de forma ambigua sobre una
frontera provoca un error claro; no se aproxima silenciosamente al distrito más
cercano. Los hechos meteorológicos y de calidad del aire consultan después sus
respectivas dimensiones de estaciones mediante `lookup_join`.

## Validación local

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

Las pruebas actuales son unitarias y utilizan dobles de Spark/Delta, por lo que
no requieren Azure, Java, PySpark ni una sesión Spark local. Las ejecuciones de
datos se realizan directamente en Databricks.
