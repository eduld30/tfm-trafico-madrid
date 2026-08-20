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
| `accidentes` | `accidentes_historico` | CSV | `overwrite` |
| `meteo` | `dim_meteo` | CSV | `overwrite` |
| `meteo` | `dim_meteo_magnitudes` | CSV | `overwrite` |
| `meteo` | `meteo_nrt` | CSV | `merge` |
| `meteo` | `meteo_historico` | CSV | `merge` |
| `eventos` | `eventos_culturales` | CSV | `overwrite` |
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
add_literal, parse_timestamp, parse_date, hourly_wide_to_long,
deduplicate, lookup_join
```

Las transformaciones se ejecutan en el orden declarado. `lookup_join` resuelve
tablas Bronze o Silver mediante catálogo, fuente y dataset; el mapa `select`
define `columna_destino: columna_lookup`.

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

## Validación local

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

Las pruebas actuales son unitarias y utilizan dobles de Spark/Delta, por lo que
no requieren Azure, Java, PySpark ni una sesión Spark local. Las ejecuciones de
datos se realizan directamente en Databricks.
