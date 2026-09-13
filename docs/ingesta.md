# Ingesta Bronze/Silver y ADF

## Datasets

| Fuente | Dimensiones | Hechos |
|---|---|---|
| `trafico` | `dim_trafico`, `dim_distritos` | `trafico_historico`, `trafico_nrt` |
| `accidentes` | — | `accidentes_historico` |
| `meteo` | `dim_meteo`, `dim_meteo_magnitudes` | `meteo_historico`, `meteo_nrt` |
| `eventos` | — | `eventos_culturales` |
| `calair` | `dim_calair`, `dim_calair_magnitudes` | `calair_historico`, `calair_nrt` |

Las dimensiones se procesan antes que los hechos que las consultan.

## Configuración declarativa

Cada fuente se define en `conf/sources/*.yaml`. Un dataset declara formato,
ruta de landing, opciones de lectura, transformaciones Silver, claves de negocio
y estrategia de escritura. `conf/environments/*.yaml` resuelve almacenamiento,
catálogos y opciones de runtime. No se guardan secretos en estos ficheros.

Las rutas relativas se resuelven usando el entorno:

```text
landing/<source>/<dataset>/
bronze/<source>/<dataset>/
silver/<source>/<dataset>/
```

`source_path` siempre identifica la raíz estable. Los subdirectorios
`YYYY/MM/DD[/HH]` organizan ficheros, pero no cambian la configuración.

## Bronze

Auto Loader lee la raíz estable con `availableNow=True`: procesa lo disponible y
finaliza. Cada dataset mantiene un checkpoint exclusivo y su schema location en
`<ruta_bronze>/_checkpoint/`. El prefijo `_` evita que Delta trate ese directorio
como datos.

El checkpoint no debe eliminarse de forma aislada mientras se conserva la tabla,
porque Auto Loader podría volver a descubrir ficheros ya procesados. En las
tablas externas, `DROP TABLE` elimina únicamente el metadato de Unity Catalog.
Una reconstrucción completa requiere eliminar también la ruta física del dataset,
incluido su checkpoint.

Bronze conserva el contenido original, permite una evolución de esquema
configurable y añade:

- `_ingestion_timestamp` y `_ingestion_run_id`;
- `_source_file` y `_source_file_modification_time`;
- `_file_date`, obtenida de la fecha de la ruta.

CSV y XML se encapsulan detrás de lectores configurables. El tráfico NRT usa el lector XML y su `row_tag` declarado en YAML.

## Silver

Silver lee exclusivamente la tabla Bronze. Solo procesa filas posteriores al
mayor `_ingestion_timestamp` ya publicado y termina sin escribir cuando no hay
datos nuevos.

Silver conserva `_ingestion_timestamp`, `_ingestion_run_id` y `_source_file`
para mantener el linaje de Bronze, y añade `_silver_processed_timestamp`.

| Estrategia | Uso principal | Condición |
|---|---|---|
| `merge` | Históricos y NRT con reprocesos | Claves de negocio |
| `overwrite` | Dimensiones o fotografía completa | Última `_file_date` recibida |
| `append` | Datos inmutables | Mecanismo incremental explícito |
| `replace_partitions` | Fotografías por partición | `partition_by` y `_file_date` |

`accidentes_historico` utiliza `replace_partitions` porque cada fichero anual es
una fotografía actualizada y no existe una clave estable por persona implicada.
Solo se sustituye la partición anual recibida, sin reescribir el resto del
histórico.

Las transformaciones cubren normalización y selección de columnas, casts,
fechas, limpieza de texto, filtros, deduplicación y joins acotados. Las
dimensiones de estaciones pueden usar `assign_district`, que asigna uno de los
21 distritos mediante el KML versionado. La asignación debe ser unívoca; si una
estación no pertenece exactamente a un distrito, el proceso falla y no aproxima
automáticamente al distrito más cercano. El detalle exacto está en los YAML y en
`src/madrid_ingestion/silver/transformations.py`.

## Ejecución

```bash
python -m madrid_ingestion validate-config --env dev
python -m madrid_ingestion list-datasets --source trafico
python -m madrid_ingestion run --env dev --layer bronze --source trafico --dataset trafico_nrt
python -m madrid_ingestion run --env dev --layer silver --source trafico --dataset trafico_nrt
```

Sin `--dataset` se ejecuta secuencialmente una fuente. `--all` ejecuta todos los
datasets habilitados de la capa, también de forma secuencial.

## ADF

Los pipelines de `adf/` descargan las fuentes y, tras una copia correcta,
ejecutan el job de Databricks correspondiente.

| Carga | Frecuencia configurada |
|---|---|
| Tráfico NRT | 10 minutos |
| Meteorología y calidad del aire NRT | 20 minutos |
| Eventos | Diaria, 07:00 |
| Históricos | Día 1 mensual, entre 07:00 y 09:30 |
| Dimensiones | Manual |

Los triggers se versionan detenidos y se activan explícitamente. Los parámetros
por entorno están en `adf/params/`; `pro.json` necesita los IDs definitivos de
los jobs antes del despliegue productivo.

El job `gold_nrt_predictions` no tiene calendario propio. Está previsto que ADF
lo invoque después de que finalice correctamente la ingesta Bronze/Silver de
tráfico NRT, reutilizando la última meteorología y calidad del aire disponibles.
