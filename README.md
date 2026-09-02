# Madrid Ingestion

Motor configurable de ingesta Bronze/Silver para el TFM de monitorización y
análisis del tráfico de Madrid. Se ejecuta sobre Azure Databricks, escribe tablas
Delta externas en ADLS y las registra en Unity Catalog.

La lógica reside en el paquete Python. Los YAML describen fuentes, datasets,
lectura, transformaciones y estrategia de escritura sin incluir código
ejecutable ni secretos.

## Arquitectura implementada

```text
fuentes HTTP / ficheros estáticos
  -> Azure Data Factory: descarga y organización temporal en landing
  -> Bronze: Auto Loader + metadatos técnicos + Delta externo
  -> Silver: lectura incremental + transformaciones + escritura idempotente
  -> ML manual: snapshots Delta administrados en <gold_catalog>.ml
  -> Unity Catalog
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
define `columna_destino: columna_lookup`. Los casts usan `try_cast`, por lo que
un valor incompatible se convierte en nulo sin abortar el lote completo.

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

Para desarrollo local y las pruebas de ingesta:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Las pruebas ML usan Spark local y requieren Java y el extra `databricks`:

```bash
python -m pip install -e ".[dev,databricks]"
```

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

El repositorio incluye un bundle para desplegar diez jobs sin calendario propio:

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
ml_training_snapshot
```

Cada dataset se ejecuta mediante dos tareas `python_wheel_task` con dependencia
explícita `Bronze -> Silver`. Los jobs no declaran `run_as` ni clúster: utilizan
serverless compute con el entorno `default`. En el job de dimensiones, las seis
ramas se ejecutan en paralelo y cada tarea Silver depende únicamente de la
carga Bronze de su propio dataset.

`ml_training_snapshot` es la definición versionada de un `notebook_task` manual
e independiente de ADF. No está desplegada en el workspace `dev`: el primer
deploy del bundle raíz crearía también los nueve jobs de ingesta. Por tanto,
esa definición no se utiliza para materializar el snapshot ML mientras esos
recursos permanezcan fuera de la frontera de propiedad del equipo ML.

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
`--profile <perfil>` a los comandos del bundle. Tras el despliegue, todos los
jobs se pueden lanzar desde `Workflows > Jobs & Pipelines`; los nueve jobs de
ingesta también pueden invocarse desde los pipelines ADF.
Las dimensiones deben cargarse primero, ya que los jobs restantes las consultan
desde sus transformaciones Silver.

La versión del artefacto se hace dinámica en cada despliegue para que serverless
no reutilice un wheel anterior con el mismo número de versión del proyecto. Los
calendarios de ingesta se mantienen en ADF; el job ML no tiene schedule ni
refresco automático. El despliegue del bundle y ADF está automatizado mediante
GitHub Actions.

## Orquestación con Azure Data Factory

La carpeta `adf/` contiene trece pipelines: cuatro cargas individuales de
dimensiones, un pipeline agrupador y ocho cargas de hechos. Estas últimas
descargan el fichero público en la raíz estable del dataset bajo `landing` y,
cuando la copia finaliza correctamente, invocan el job correspondiente de
Databricks. El histórico de tráfico añade un paso de descompresión desde
`_staging`.

Las recurrencias configuradas usan la zona horaria `Romance Standard Time`:

| Carga | Recurrencia |
|---|---|
| Tráfico NRT | Cada 10 minutos |
| Meteorología NRT | Cada 10 minutos |
| Calidad del aire NRT | Cada 20 minutos |
| Eventos culturales | Diaria a las 07:00 |
| Tráfico histórico | Día 1 de cada mes a las 07:00 |
| Accidentes | Día 1 de cada mes a las 07:30 |
| Meteorología histórica | Día 1 de cada mes a las 08:30 |
| Calidad del aire histórica | Día 1 de cada mes a las 09:30 |
| Dimensiones | Ejecución manual, sin trigger |

Los ocho triggers están versionados con `runtimeState: Stopped`; deben activarse
explícitamente en la factoría cuando proceda. `pl_ingest_dimensiones` descarga
en paralelo las cuatro dimensiones obtenidas de fuentes externas y después
lanza el job agrupado de seis dimensiones Bronze/Silver.

Los identificadores de los nueve jobs son parámetros globales de ADF y se
sobrescriben mediante `adf/params/<entorno>.json`. El linked service de
Databricks utiliza la identidad administrada de la factoría, por lo que esta
debe tener permisos para ejecutar los jobs del workspace.

## CI/CD

El workflow `.github/workflows/ci-cd.yml` aplica un flujo de promoción entre
los dos entornos:

- Una pull request hacia `develop` valida ADF y el Asset Bundle para `dev`.
- Un push o merge en `develop` despliega ADF y Databricks en `dev`.
- Una pull request hacia `main` valida ambos componentes para `pro`.
- Un push o merge en `main` despliega ADF y Databricks en `pro`.
- `workflow_dispatch` usa el entorno asociado a la rama desde la que se lance.

En todos los casos también se ejecutan lint, tests y build del paquete. Tras las
validaciones, se despliega primero el Asset Bundle; ADF se despliega únicamente
cuando los jobs de Databricks ya están disponibles.

La autenticación usa GitHub OIDC y una identidad federada de Azure, sin client
secret ni token personal de Databricks. El workflow espera estas
variables de repositorio en GitHub:

```text
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID
DEV_RESOURCE_GROUP
DEV_FACTORY_NAME
DEV_DATABRICKS_HOST
PRO_RESOURCE_GROUP
PRO_FACTORY_NAME
PRO_DATABRICKS_HOST
```

Si ambos entornos usan el mismo workspace, `DEV_DATABRICKS_HOST` y
`PRO_DATABRICKS_HOST` tendrán el mismo valor. La separación de datos sigue
estando garantizada por los catálogos y rutas declarados en cada entorno.

Los parámetros ADF de `dev` incluyen los nueve identificadores de jobs. En
`adf/params/pro.json` esos valores siguen marcados como `<<pendiente>>`; la
validación de CI detiene deliberadamente la promoción a `pro` hasta sustituirlos
por los identificadores creados para ese entorno.

La identidad debe tener tres credenciales federadas para el repositorio
`eduld30/tfm-trafico-madrid`: una de tipo **Pull request**, otra de tipo
**Branch** para `develop` y otra de tipo **Branch** para `main`. En el formulario
de Azure, el ID de la organización es `95371349` y el ID del repositorio es
`1304265160`. Conviene usar el formulario de GitHub Actions para que Azure genere
el sujeto OIDC, incluido su formato inmutable, en lugar de escribirlo
manualmente. No se asocia un GitHub Environment a los jobs porque eso haría que
el sujeto estuviera ligado al entorno en lugar de a la rama o pull request.

Para ADF, la identidad necesita `Data Factory Contributor` sobre los grupos de
recursos de las dos factorías, ya que el workflow crea el ARM deployment en
esos ámbitos. Para Databricks, la misma identidad debe estar añadida al
workspace como service principal y disponer de acceso al workspace, a los
catálogos Bronze y Silver de ambos entornos, y a las ubicaciones externas
utilizadas por el motor.

El export de ADF se genera con la utilidad oficial fijada en `package-lock.json`.
Antes y después del despliegue se usa la versión fijada por commit del script
oficial de Microsoft para detener los triggers modificados, limpiar recursos
eliminados y recuperar su estado. Si falla el ARM deployment, el workflow
reactiva los triggers que estaban en ejecución antes de comenzar.

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

## Snapshot ML desde Silver

El módulo `madrid_ml` crea dos tablas Delta administradas en
`<gold_catalog>.ml`:

| Tabla | Grano |
|---|---|
| `accident_labels_hourly` | Una fila por distrito y `feature_hour`, con el número y el indicador de accidentes en la hora siguiente |
| `features_training_snapshot` | Una fila por distrito y `feature_hour`, con la etiqueta, calendario y agregados históricos de tráfico, meteorología y calidad del aire |

La API fija primero las versiones Delta de las cinco entradas Silver y después
las lee con `versionAsOf`:

```python
from madrid_ml import build_training_snapshot

result = build_training_snapshot(
    spark=spark,
    silver_catalog="<catalogo_silver>",
    gold_catalog="<catalogo_gold>",
    code_commit="<commit_git>",
    expected_versions=versiones_silver_autorizadas,
)
```

`expected_versions` debe contener el nombre completo y la versión Delta
autorizada de cada tabla Silver de accidentes, distritos, tráfico histórico,
meteorología histórica y calidad del aire histórica.

El resource key `ml_training_snapshot` permanece versionado, pero no es la ruta
operativa autorizada para crear estas tablas en `dev`. No se debe ejecutar
`bundle deploy` ni `bundle run` para el snapshot ML: el bundle raíz no permite
seleccionar un único recurso y su primer deploy crearía también los nueve jobs
de ingesta.

El primer snapshot se ejecuta mediante un único `databricks jobs submit`
efímero, supervisado por el procedimiento de Task 6. Ese procedimiento:

- construye un wheel inequívoco desde un checkout limpio;
- importa temporalmente el wheel y `notebooks/run_ml_snapshot.py`;
- exige que las dos salidas Gold estén ausentes;
- limita el run serverless a 900 segundos, sin cola, reintentos ni
  auto-optimización;
- valida el resultado completo y elimina todo el staging;
- elimina las salidas creadas si el run falla.

El primer intento del 30 de agosto de 2026 no materializó el snapshot porque
serverless no admite `DataFrame.persist()`. Tras sustituirlo por tablas Delta
administradas temporales con ownership y cleanup en `finally`, el run efímero
`1074840131667081` ejecutó end-to-end el commit
`f3e535819b80065e2179cc5d020240ffb03412da` con estado
`TERMINATED/SUCCESS`. Publicó 1.379.931 filas para cada una de las dos tablas,
21 distritos, cero duplicados o diferencias respecto al grid y lineage común
con `snapshot_id=ea719086-1a93-401c-969b-4e92586e13fd`. El cleanup eliminó el
staging y las tablas temporales; al finalizar no quedaron runs activos ni jobs
guardados.

No existe integración ADF, schedule ni refresco automático para este snapshot.
Cada combinación de versiones Silver requiere autorización explícita. Tras una
ejecución correcta, las dos tablas Gold se conservan para su uso posterior; no
queda ningún job guardado.

El contrato temporal interpreta `fecha_hora` como hora civil ya almacenada en
Silver y exige una sesión Spark con timezone `Etc/UTC`; no aplica otra
conversión UTC o DST. Cada fila publicada conserva `snapshot_id`,
`input_versions_json`, `code_commit`, `feature_schema_version` y
`time_contract`. El alcance termina en el snapshot de features: no incluye
entrenamiento ni scoring.

## Preprocessing ML v1

`madrid_ml.prepare_training_data` lee las dos tablas Gold mediante versiones
Delta explícitas y exige que compartan `snapshot_id`, lineage, schema de
features y contrato temporal:

```python
from madrid_ml import prepare_training_data

prepared = prepare_training_data(
    spark=spark,
    gold_catalog="dev_gold",
    labels_delta_version=1,
    features_delta_version=1,
    expected_snapshot_id="ea719086-1a93-401c-969b-4e92586e13fd",
)
```

El protocolo usa `spark.sql.session.timeZone=Etc/UTC` y conserva cuatro
intervalos semiabiertos:

| Tramo | Intervalo de `feature_hour` | Filas autorizadas |
|---|---|---:|
| Train | `[2019-01-01, 2024-01-01)` | 920.304 |
| Validación | `[2024-01-01, 2025-01-01)` | 184.464 |
| Evaluación final | `[2025-01-01, 2026-01-01)` | 183.960 |
| Excluido | desde `2026-01-01` | 91.203 |

Solo train ajusta estado. Las 18 medias convierten `null`, `NaN`, infinitos y
valores físicamente inválidos a ausente, crean un indicador de disponibilidad
y se imputan con la mediana de train. Los 15 conteos se conservan como
features numéricas. El pipeline estandariza esas 33 componentes y añade 18
indicadores más 60 componentes one-hot de distrito, hora, día de semana y mes.
El vector final contiene exactamente 111 valores finitos.

El contrato reproducible no es únicamente el `PipelineModel`: el notebook
`notebooks/run_ml_preprocessing.py` registra en un único run MLflow:

- `preprocessor/`: estado aprendido por Spark ML;
- `preprocessing_manifest.json`: versiones Gold, lineage, periodos, reglas
  físicas, orden de inputs, medianas, escalado, categorías, runtime y paquete;
- `quality_by_split.json`: ausentes e invalideces por feature y split;
- `package/`: el wheel exacto, verificado mediante SHA-256 antes de ejecutar.

El preprocessing no materializa otra tabla Gold, no modifica Silver ni
ingesta, no entrena LightGBM y no realiza scoring. Es un baseline
retrospectivo: evita fuga del estado aprendido entre splits, pero no afirma
disponibilidad point-in-time ni latencia de publicación.

La ejecución remota requiere una autorización separada para una única tarea
serverless acotada y para escribir un run MLflow. Hasta obtenerla no se debe
enviar el notebook ni crear recursos persistentes.

## Entrenamiento y comparación de modelos ML

El flujo de `src/madrid_ml/training.py` compara cinco alternativas sobre el
mismo snapshot Gold versionado: prevalencia global, frecuencia por distrito,
frecuencia por distrito/hora/día de semana, regresión logística Spark ML y
LightGBM SynapseML. Los dos folds internos usan train 2019–2021 con validación
2022 y train 2019–2022 con validación 2023. Después se reajusta con 2019–2023,
se compara en 2024 y se registra una evaluación transparente de 2025.

El entrypoint es `scripts/ml/train_models.py`. El bundle independiente vive en
`bundles/ml` y despliega únicamente `madrid-ml-model-training`; no incluye los
jobs de ingesta. LightGBM requiere SynapseML 1.1.3 y un job cluster clásico con
dos workers fijos. El Job tiene un límite duro de 60 minutos, cero reintentos,
cola desactivada y terminación automática del job cluster al finalizar. El tipo
de nodo se proporciona mediante `ML_NODE_TYPE_ID`.

Desde `bundles/ml`:

```bash
export DATABRICKS_AUTH_STORAGE=plaintext
databricks bundle validate -t dev --profile tfm-dev --var "ml_node_type_id=$ML_NODE_TYPE_ID"
databricks bundle deploy -t dev --profile tfm-dev --var "ml_node_type_id=$ML_NODE_TYPE_ID"
databricks bundle run -t dev --profile tfm-dev --var "ml_node_type_id=$ML_NODE_TYPE_ID" ml_model_training --params "$ML_JOB_PARAMS"
```

La ejecución solo lee Gold mediante versiones Delta explícitas y escribe runs
y artefactos compactos en MLflow. No escribe Gold o Silver y no registra
modelos en Model Registry.

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

Las pruebas de ingesta siguen usando dobles de Spark y Delta. Las pruebas de
`madrid_ml` ejecutan Spark local con Java, PySpark y Delta instalados mediante
el extra `databricks`; no acceden a Azure ni a Databricks.
