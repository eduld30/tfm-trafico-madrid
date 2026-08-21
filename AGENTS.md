# Instrucciones de desarrollo — Motor de ingesta Medallion en Azure Databricks

## 1. Propósito de este documento

Este fichero define el contexto funcional, las decisiones de arquitectura, las convenciones y los criterios de aceptación para desarrollar un paquete Python de ingesta de datos en Azure Databricks.

El proyecto forma parte de un TFM del Máster en Big Data & Data Engineering titulado:

> **Arquitectura end-to-end en Azure para la monitorización y el análisis predictivo de accidentes de tráfico en Madrid**

El producto final del TFM incluirá cuadros de mando en Power BI con:

- Índices de riesgo de accidente por distrito de Madrid.
- Datos históricos de accidentes.
- Intensidad y estado del tráfico.
- Meteorología.
- Calidad del aire.
- Eventos culturales.
- Análisis descriptivos y predictivos.

El alcance actual se limita al **motor de ingesta y transformación Bronze/Silver**. La orquestación productiva, la capa Gold, Power BI, el modelo predictivo y la observabilidad avanzada se desarrollarán posteriormente.

---

## 2. Objetivo del motor

Construir un paquete Python reutilizable y configurable que:

1. Lea ficheros raw depositados en el contenedor `landing` de ADLS.
2. Procese los ficheros mediante Azure Databricks.
3. Implemente un patrón Medallion:
   - `landing`: ficheros raw.
   - `bronze`: ingestión técnica con Auto Loader.
   - `silver`: limpieza, tipado, normalización y consolidación.
4. Escriba tablas Delta externas en ADLS.
5. Registre las tablas en Unity Catalog.
6. Permita configurar fuentes y datasets mediante ficheros YAML.
7. Evite lógica específica de datasets codificada dentro del núcleo del motor.
8. Ofrezca una API Python y una CLI sencilla.
9. Mantenga una arquitectura preparada para añadir posteriormente:
   - Databricks Asset Bundles.
   - Databricks Jobs.
   - Reglas de calidad.
   - Auditoría.
   - Métricas.
   - Capa Gold.
   - Nuevas fuentes.

---

## 3. Principios de diseño

### 3.1 Simplicidad adecuada al TFM

Diseñar una solución limpia, mantenible y extensible, pero evitar componentes propios de plataformas productivas de gran escala que no aporten valor directo al TFM.

No implementar inicialmente:

- Un framework complejo de plugins.
- Un motor de workflows propio.
- Paralelismo interno.
- Sistemas de alertas.
- Reintentos sofisticados.
- Telemetría distribuida.
- Catálogo de reglas de calidad avanzado.
- Interfaces web.
- APIs HTTP.
- Persistencia de estados operativos en bases de datos externas.

### 3.2 Configuración declarativa

Los ficheros YAML deben declarar **qué debe hacer el motor**.

El código Python debe implementar **cómo se ejecuta**.

No permitir código Python arbitrario dentro de los YAML.

### 3.3 Separación de responsabilidades

Separar claramente:

- Carga y validación de configuración.
- Resolución de nombres y rutas.
- Lectura.
- Transformaciones.
- Escritura.
- Registro de tablas.
- Logging.
- Orquestación de una ejecución.

### 3.4 Idempotencia

Priorizar ejecuciones repetibles:

- Bronze debe confiar en el seguimiento de ficheros de Auto Loader.
- Silver debe usar `merge` u `overwrite` cuando sea posible.
- `append` debe utilizarse solo cuando la configuración garantice que no se reprocesan los mismos registros.

### 3.5 Código independiente de notebooks

La lógica de negocio debe residir en el paquete Python.

Los notebooks, si se crean más adelante, serán wrappers delgados que invoquen el paquete.

### 3.6 Spark nativo

Priorizar:

- DataFrame API.
- Funciones de `pyspark.sql.functions`.
- Operaciones Delta.
- Transformaciones lazy.

Evitar:

- `collect()` para procesar datos.
- Conversión a Pandas.
- UDFs Python cuando exista una función Spark equivalente.
- Acciones innecesarias como `count()` únicamente para logging.

### 3.7 Pruebas orientadas a comportamiento

Las pruebas automatizadas deben comprobar principalmente la funcionalidad del
código y los invariantes que el motor necesita para operar correctamente.

No crear pruebas que congelen decisiones de configuración susceptibles de
cambiar durante la evolución del proyecto, por ejemplo:

- El valor concreto de `overwrite_schema` de un dataset.
- La presencia o ausencia de calendarios en los jobs.
- El uso de serverless, un clúster existente u otro tipo de compute.
- El número exacto de fuentes, datasets o jobs configurados.
- Opciones operativas concretas de un entorno o workflow.

Las configuraciones deben validarse mediante sus modelos, los validadores del
motor y, cuando corresponda, las herramientas nativas como
`databricks bundle validate`. Reservar los tests sobre configuración para
contratos importantes, casi estáticos y necesarios para la corrección funcional.

---

## 4. Fuentes de datos

| Fuente | Dataset conceptual | Tipo | Formato | Frecuencia | Identificador geográfico |
|---|---|---:|---|---|---|
| Tráfico | Histórico de tráfico desde 2013 | Batch | CSV `;` | Mensual | `id` → `dim_trafico` → distrito |
| Tráfico | Tráfico NRT | NRT | XML | Cada 10 minutos | `idelem` → `dim_trafico` → distrito |
| Tráfico | Dimensión de puntos de medida | Batch | CSV `;` | Puntual | `id`, distrito, coordenadas |
| Tráfico | Dimensión de distritos | Batch | CSV `;` | Puntual | `COD_DIS`, `NOMBRE` |
| Accidentes | Histórico de accidentes desde 2019 | Batch | CSV `;` | Mensual | Distrito en texto |
| Meteorología | Meteo NRT | NRT | CSV `;` | Cada 10 minutos | `ESTACION` → `dim_meteo` → distrito |
| Meteorología | Meteo histórico (granularidad horaria) | Batch | CSV `;` | Mensual | `ESTACION` → `dim_meteo` → distrito |
| Meteorología | Dimensión de estaciones | Batch | CSV `;` | Puntual | `ESTACION`, coordenadas |
| Meteorología | Dimensión de magnitudes | Estático | CSV generado | Único | No aplica |
| Eventos | Próximos 100 días | NRT | CSV `;` | Diario | Distrito en texto |
| Calidad del aire | Calidad del aire NRT | NRT | CSV `;` | Cada 20 minutos | `ESTACION` → `dim_calair` → distrito |
| Calidad del aire | Histórico de calidad del aire (granularidad horaria) | Batch | CSV `;` | Mensual | `ESTACION` → `dim_calair` → distrito |
| Calidad del aire | Dimensión de estaciones | Batch | CSV `;` | Puntual | `CODIGO_CORTO`, coordenadas |
| Calidad del aire | Dimensión de magnitudes | Estático | CSV generado | Único | No aplica |

### 4.1 Nombres lógicos recomendados

```text
trafico.historico_trafico
trafico.trafico_nrt
trafico.dim_trafico

trafico.dim_distritos

accidentes.historico_accidentes

meteorologia.meteo_nrt
meteorologia.meteo_historico
meteorologia.dim_meteo
meteorologia.dim_magnitudes_meteo

eventos.eventos_culturales

calidad_aire.calidad_aire_nrt
calidad_aire.calidad_aire_historico
calidad_aire.dim_calair
calidad_aire.dim_magnitudes_calair
```

Los identificadores técnicos deben utilizar minúsculas, snake case y no contener espacios, tildes ni caracteres especiales.

`dim_distritos` pertenece al esquema `trafico`; no se creará un esquema independiente `geografia`.

Aunque los históricos meteorológicos y de calidad del aire tengan granularidad horaria, sus nombres técnicos no deben incluir el sufijo `_horario`.

---

## 5. Arquitectura de almacenamiento

### 5.1 Contenedores ADLS

```text
landing
bronze
silver
gold
```

La capa `gold` queda fuera del alcance actual.

### 5.2 Convención de rutas de landing

La configuración `source_path` debe apuntar al directorio base estable del dataset:

```text
abfss://landing@<storage-account>.dfs.core.windows.net/<source>/<dataset>/
```

Ejemplo:

```text
abfss://landing@<storage-account>.dfs.core.windows.net/trafico/trafico_nrt/
```

Los ficheros pueden organizarse en subdirectorios temporales:

```text
trafico/trafico_nrt/2026/07/26/17/trafico_nrt202607261715.xml
```

Auto Loader debe iniciarse sobre la ruta base del dataset, no sobre la carpeta de una fecha concreta:

```yaml
bronze:
  source_path: "trafico/trafico_nrt"
```

Con este enfoque, Auto Loader descubre los ficheros de los subdirectorios de forma recursiva y mantiene un único estado incremental para el dataset.

Usar componentes temporales con longitud fija y ceros a la izquierda:

```text
YYYY/MM/DD/HH
```

No es necesario modificar `source_path` en cada ejecución ni configurar una ruta diaria. Los directorios `2026/07/26/17` organizan los ficheros, pero no generan automáticamente columnas de año, mes, día y hora. Si esas columnas son necesarias, deben derivarse en Silver a partir del timestamp de negocio o de `_source_file`.

### 5.3 Ruta de tabla y checkpoint de Auto Loader

Para simplificar la gestión del TFM, Bronze utilizará directamente el directorio del dataset como `LOCATION` de la tabla Delta. El checkpoint de Structured Streaming residirá dentro de esa misma ruta en un subdirectorio cuyo nombre comienza por `_`.

Patrón:

```text
abfss://bronze@<storage-account>.dfs.core.windows.net/<source>/<dataset>/
├── _delta_log/
├── _checkpoint/
├── part-00000-....snappy.parquet
└── ...
```

Ejemplo:

```text
abfss://bronze@<storage-account>.dfs.core.windows.net/trafico/trafico_nrt/
├── _delta_log/
├── _checkpoint/
├── part-00000-....snappy.parquet
└── ...
```

Las rutas serán:

```text
Delta table location:
abfss://bronze@<storage-account>.dfs.core.windows.net/trafico/trafico_nrt/

Auto Loader checkpoint:
abfss://bronze@<storage-account>.dfs.core.windows.net/trafico/trafico_nrt/_checkpoint/
```

Por defecto, `cloudFiles.schemaLocation` utilizará la misma ruta que `checkpointLocation`. Databricks permite compartir ambas ubicaciones:

```text
Auto Loader schema location:
abfss://bronze@<storage-account>.dfs.core.windows.net/trafico/trafico_nrt/_checkpoint/
```

Esta convención es válida para una tabla Delta:

- Delta solo considera como datos los ficheros registrados en `_delta_log`.
- `VACUUM` ignora los directorios cuyo nombre comienza por `_` o `.`.
- El checkpoint queda asociado físicamente al dataset.
- La eliminación explícita del directorio completo elimina conjuntamente datos, log Delta y estado de Auto Loader.

El nombre debe comenzar siempre por `_`. No usar un directorio como `checkpoint/` sin prefijo, porque podría ser tratado como contenido no gestionado y eliminado por operaciones de mantenimiento.

Cada consulta de Structured Streaming debe tener un checkpoint exclusivo. No compartir `_checkpoint` entre dos fuentes, dos tablas o dos consultas diferentes.

La tabla Silver utilizará directamente:

```text
abfss://silver@<storage-account>.dfs.core.windows.net/<source>/<dataset>/
```

Silver no necesita checkpoint de Auto Loader porque lee desde tablas Bronze.

#### Consideración sobre Unity Catalog

La documentación específica de Auto Loader con tablas externas de Unity Catalog recomienda una ruta de checkpoint separada de los datos. Sin embargo, la documentación de Delta Lake admite explícitamente el patrón `<table-name>/_checkpoints`, y `VACUUM` preserva directorios con prefijo `_`.

Para este TFM se adopta el checkpoint anidado por simplicidad y facilidad de limpieza. La función de resolución de rutas debe quedar centralizada para poder trasladar los checkpoints a una raíz independiente en una futura productivización sin cambiar los procesadores.

#### Eliminación de tablas externas

`DROP TABLE` sobre una tabla externa elimina únicamente el metadato de Unity Catalog; no borra los ficheros de ADLS.

Cuando se quiera reconstruir completamente una tabla se necesitarán dos acciones explícitas:

1. Eliminar el registro de la tabla.
2. Eliminar recursivamente su ruta física en ADLS.

Al borrar la ruta completa también se elimina `_checkpoint`. La siguiente ejecución se comportará como una ingesta nueva y podrá volver a procesar todos los ficheros existentes en `landing`.

### 5.4 Tablas externas

Todas las tablas Bronze y Silver serán Delta externas.

Patrón:

```text
<catalog>.<schema>.<table>
```

Donde:

- Catálogo: `{env}_{layer}`.
- Schema: nombre de la fuente.
- Tabla: nombre del dataset.

Ejemplos:

```text
dev_bronze.trafico.trafico_nrt
dev_silver.trafico.trafico_nrt
```

---

## 6. Unidad de ejecución

La unidad mínima será **un dataset y una capa**.

API Python conceptual:

```python
run_dataset(
    environment="dev",
    layer="bronze",
    source="trafico",
    dataset="trafico_nrt",
)
```

CLI conceptual:

```bash
python -m madrid_ingestion run \
  --env dev \
  --layer bronze \
  --source trafico \
  --dataset trafico_nrt
```

### 6.1 Ejecuciones agrupadas

La implementación puede incluir selectores opcionales:

```bash
# Todos los datasets habilitados de una fuente y capa
python -m madrid_ingestion run \
  --env dev \
  --layer bronze \
  --source trafico

# Todos los datasets habilitados de una capa
python -m madrid_ingestion run \
  --env dev \
  --layer bronze \
  --all
```

Estas ejecuciones deben resolverse como una secuencia de ejecuciones individuales. No implementar paralelismo dentro del paquete.

---

## 7. Flujo general

```text
CLI o API Python
      ↓
Carga de configuración de entorno
      ↓
Carga de configuración de fuente y dataset
      ↓
Validación
      ↓
Construcción del contexto de ejecución
      ↓
Selección del procesador Bronze o Silver
      ↓
Lectura
      ↓
Transformación
      ↓
Escritura Delta externa
      ↓
Registro o validación de tabla
      ↓
Logging de finalización
```

---

## 8. Estructura del repositorio

Usar layout `src`.

```text
madrid-ingestion/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── conf/
│   ├── environments/
│   │   ├── dev.yaml
│   │   └── pro.yaml
│   ├── sources/
│   │   ├── trafico.yaml
│   │   ├── accidentes.yaml
│   │   ├── meteorologia.yaml
│   │   ├── eventos.yaml
│   │   └── calidad_aire.yaml
│   └── schemas/
│       ├── trafico/
│       ├── accidentes/
│       ├── meteorologia/
│       ├── eventos/
│       └── calidad_aire/
├── src/
│   └── madrid_ingestion/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── runner.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── models.py
│       │   └── validators.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── context.py
│       │   ├── exceptions.py
│       │   ├── naming.py
│       │   └── paths.py
│       ├── bronze/
│       │   ├── __init__.py
│       │   ├── processor.py
│       │   ├── autoloader.py
│       │   └── metadata.py
│       ├── silver/
│       │   ├── __init__.py
│       │   ├── processor.py
│       │   ├── transformations.py
│       │   └── writers.py
│       ├── readers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── csv_reader.py
│       │   └── xml_reader.py
│       ├── geospatial/
│       │   ├── __init__.py
│       │   └── districts.py
│       ├── resources/
│       │   ├── __init__.py
│       │   └── distritos.kml
│       ├── writers/
│       │   ├── __init__.py
│       │   ├── delta_writer.py
│       │   └── table_manager.py
│       └── observability/
│           ├── __init__.py
│           └── logger.py
└── tests/
    ├── unit/
    └── integration/
```

No crear módulos vacíos sin una responsabilidad clara.

---

## 9. Dependencias

Mantener el número de dependencias reducido.

Dependencias funcionales previstas:

```text
pyspark
APIs Delta disponibles en Databricks
PyYAML
pydantic
shapely
```

Dependencias de desarrollo opcionales:

```text
pytest
ruff
mypy
```

No añadir frameworks adicionales salvo necesidad demostrable. La versión de Python debe alinearse con el Databricks Runtime objetivo.

---

## 10. Configuración

### 10.1 Configuración de entorno

Ejemplo `conf/environments/dev.yaml`:

```yaml
environment: dev

storage:
  account_name: "<storage-account>"
  containers:
    landing: landing
    bronze: bronze
    silver: silver
    gold: gold

catalogs:
  bronze: dev_bronze
  silver: dev_silver
  gold: dev_gold

paths:
  checkpoint_dir: "_checkpoint"
  schema_location_mode: "same_as_checkpoint"

runtime:
  default_trigger: available_now
  log_level: INFO
```

No almacenar secretos, claves ni tokens en YAML. La autenticación contra ADLS será responsabilidad del entorno Databricks.

### 10.2 Configuración por fuente

Ejemplo resumido `conf/sources/trafico.yaml`:

```yaml
source: trafico
enabled: true

datasets:
  - name: historico_trafico
    description: "Histórico mensual de tráfico"
    enabled: true

    bronze:
      enabled: true
      format: csv
      source_path: "trafico/historico_trafico"
      target_path: "trafico/historico_trafico"

      reader_options:
        delimiter: ";"
        header: "true"
        encoding: "UTF-8"

      autoloader_options:
        schema_evolution_mode: addNewColumns
        rescued_data_column: _rescued_data

    silver:
      enabled: true
      target_path: "trafico/historico_trafico"
      write_strategy: merge

      business_keys:
        - id
        - fecha_hora

      transformations:
        - type: normalize_column_names

        - type: cast
          columns:
            id: long
            intensidad: double
            ocupacion: double
            fecha_hora: timestamp

        - type: deduplicate
          keys:
            - id
            - fecha_hora
          order_by:
            - column: _ingestion_timestamp
              direction: desc
```

### 10.3 Contrato mínimo de dataset

```yaml
name: dataset_name
description: "Descripción"
enabled: true

bronze:
  enabled: true
  format: csv
  source_path: "source/dataset"
  target_path: "source/dataset"
  reader_options: {}
  autoloader_options: {}

silver:
  enabled: true
  target_path: "source/dataset"
  write_strategy: overwrite
  business_keys: []
  partition_by: []
  transformations: []
```

`source_path` y `target_path` son rutas relativas:

- `source_path` identifica la raíz estable del dataset en `landing`.
- `target_path` identifica directamente el `LOCATION` de la tabla Delta.
- En Bronze, el checkpoint se deriva añadiendo `/_checkpoint`.
- Por defecto, `schema_path` es igual a `checkpoint_path`.

Por ejemplo, con `target_path: "trafico/trafico_nrt"`:

```text
target_path = .../trafico/trafico_nrt/
checkpoint_path = .../trafico/trafico_nrt/_checkpoint/
schema_path = .../trafico/trafico_nrt/_checkpoint/
```

### 10.4 Validación

La carga de configuración debe fallar pronto y con mensajes claros cuando:

- Falta un campo obligatorio.
- El entorno, fuente o dataset no existe.
- La capa está deshabilitada.
- `write_strategy` no es válida.
- Se usa `merge` sin claves de negocio.
- Se usa `append` sin mecanismo incremental explícito.
- Se define una transformación desconocida.
- Una ruta contradice las convenciones del entorno.
- El directorio de checkpoint no comienza por `_`.
- La ruta de landing apunta a una carpeta de fecha/hora en vez de a la raíz estable del dataset.
- Dos datasets de una fuente tienen el mismo nombre.

Usar modelos Pydantic para representar y validar la configuración.

---

## 11. Contexto de ejecución

Crear un objeto inmutable o tratado como inmutable, por ejemplo:

```python
@dataclass(frozen=True)
class RunContext:
    environment: str
    layer: str
    source: str
    dataset: str
    run_id: str
    catalog: str
    schema: str
    table: str
    source_path: str
    target_path: str
    checkpoint_path: str | None
    schema_path: str | None
```

El `run_id` debe generarse al iniciar una ejecución, salvo que sea recibido explícitamente. Usar UUID. No introducir lógica de negocio dentro del contexto.

---

## 12. Resolución de nombres y rutas

Centralizar la generación de nombres:

```python
build_catalog_name(environment: str, layer: str) -> str
build_table_name(catalog: str, source: str, dataset: str) -> str
build_adls_uri(account: str, container: str, relative_path: str) -> str
build_table_path(...)
build_checkpoint_path(...)
build_schema_path(...)
```

No concatenar rutas manualmente en múltiples módulos.

Sanear nombres técnicos:

- Minúsculas.
- Snake case.
- Sin espacios.
- Sin tildes.
- Sin caracteres especiales.
- No empezar por números.

---

## 13. Capa Bronze

### 13.1 Responsabilidad

Bronze debe:

1. Descubrir nuevos ficheros con Auto Loader.
2. Leer el formato configurado.
3. Mantener la información con mínima transformación.
4. Añadir metadatos técnicos.
5. Escribir Delta en la ruta externa Bronze.
6. Registrar la tabla en el catálogo Bronze.
7. Usar checkpoint y schema location independientes por dataset.

Bronze no debe:

- Aplicar reglas de negocio.
- Enriquecer con distritos.
- Corregir valores funcionales.
- Deduplicar por claves de negocio.
- Realizar agregaciones.
- Homogeneizar magnitudes.

### 13.2 Auto Loader

Usar el patrón de streaming incremental de Auto Loader, orientado a jobs programados que procesan los ficheros disponibles y finalizan.

Patrón conceptual:

```python
stream_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", configured_format)
    .option("cloudFiles.schemaLocation", schema_path)
    .load(source_path)
)
```

Escritura conceptual:

```python
query = (
    transformed_df.writeStream
    .format("delta")
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .start(target_path)
)

query.awaitTermination()
```

Encapsular este comportamiento. No dispersar llamadas directas a Auto Loader por el código.

`source_path` debe ser la raíz estable de landing, por ejemplo:

```text
.../landing/trafico/trafico_nrt/
```

Auto Loader debe descubrir debajo de ella rutas como:

```text
2026/07/26/17/trafico_nrt202607261715.xml
```

No construir la ruta de entrada con la fecha actual y no crear un checkpoint por día u hora. El checkpoint es único por dataset y consulta.

No eliminar únicamente `_checkpoint` mientras se conserva la tabla salvo que se pretenda reiniciar deliberadamente el estado de ingesta. Un checkpoint nuevo puede provocar que Auto Loader vuelva a descubrir ficheros ya procesados.

### 13.3 CSV

El lector CSV debe aceptar opciones configurables:

```yaml
reader_options:
  delimiter: ";"
  header: "true"
  encoding: "UTF-8"
  quote: '"'
  escape: '"'
  mode: PERMISSIVE
```

No asumir que todos los CSV comparten exactamente las mismas opciones.

### 13.4 XML

El tráfico NRT utiliza XML.

Crear un adaptador específico para XML detrás de una interfaz común.

La implementación concreta debe:

- Aislar las opciones del formato.
- Permitir configurar el elemento raíz o `rowTag` equivalente.
- No introducir lógica de tráfico en el lector genérico.
- Mantener el mismo contrato de salida DataFrame.
- Documentar cualquier dependencia adicional exigida por el runtime.

Ejemplo:

```yaml
bronze:
  format: xml
  reader_options:
    row_tag: "pm"
```

### 13.5 Metadatos técnicos Bronze

Añadir como mínimo:

```text
_ingestion_timestamp
_ingestion_run_id
_source_file
_source_file_modification_time
```

Cuando sea posible, conservar:

```text
_rescued_data
```

No sobrescribir columnas de origen con el mismo nombre. Si existe colisión, fallar con un error claro.

### 13.6 Evolución de esquema

Permitir configurar el modo de evolución.

Valor inicial recomendado:

```yaml
schema_evolution_mode: addNewColumns
```

No ocultar cambios de esquema silenciosamente.

### 13.7 Registro de tabla

Garantizar una tabla externa sobre la ruta Delta:

```sql
CREATE TABLE IF NOT EXISTS dev_bronze.trafico.trafico_nrt
USING DELTA
LOCATION 'abfss://bronze@.../trafico/trafico_nrt/'
```

Encapsular la operación en `TableManager`.

---

## 14. Capa Silver

### 14.1 Responsabilidad

Silver debe:

1. Leer desde la tabla Bronze.
2. Aplicar transformaciones declarativas.
3. Normalizar nombres.
4. Convertir tipos.
5. Tratar vacíos y nulos cuando se configure.
6. Deduplicar.
7. Enriquecer mediante dimensiones cuando se configure.
8. Escribir mediante `merge`, `overwrite` o `append`.
9. Registrar la tabla externa en el catálogo Silver.

Silver no debe leer directamente desde `landing`.

### 14.2 Origen y destino

Origen:

```text
<env>_bronze.<source>.<dataset>
```

Destino:

```text
<env>_silver.<source>.<dataset>
```

Permitir una tabla Bronze distinta solo cuando se declare explícitamente.

### 14.3 Registro de transformaciones

Interfaz conceptual:

```python
def apply_transformations(
    df: DataFrame,
    transformations: list[TransformationConfig],
    context: RunContext,
) -> DataFrame:
    ...
```

Cada transformación debe:

- Recibir un DataFrame.
- Devolver un DataFrame.
- Ser determinista.
- Validar sus parámetros.
- No ejecutar escrituras.
- No acceder a variables globales.

### 14.4 Transformaciones iniciales

#### `normalize_column_names`

```yaml
- type: normalize_column_names
```

#### `rename`

```yaml
- type: rename
  columns:
    IDELEM: idelem
    FECHA: fecha
```

#### `select`

```yaml
- type: select
  columns:
    - idelem
    - fecha_hora
    - intensidad
```

#### `drop`

```yaml
- type: drop
  columns:
    - columna_auxiliar
```

#### `cast`

```yaml
- type: cast
  columns:
    idelem: long
    intensidad: double
    fecha_hora: timestamp
```

#### `trim`

```yaml
- type: trim
  columns:
    - distrito
    - nombre
```

#### `empty_to_null`

```yaml
- type: empty_to_null
  columns:
    - distrito
    - descripcion
```

#### `replace_values`

```yaml
- type: replace_values
  column: distrito
  values:
    "CENTRO ": "CENTRO"
```

#### `filter`

Restringir a expresiones SQL explícitas:

```yaml
- type: filter
  condition: "fecha_hora IS NOT NULL"
```

No evaluar expresiones Python.

#### `add_literal`

```yaml
- type: add_literal
  column: fuente
  value: "ayuntamiento_madrid"
```

#### `parse_timestamp`

```yaml
- type: parse_timestamp
  source_column: fecha_hora_raw
  target_column: fecha_hora
  format: "yyyy-MM-dd HH:mm:ss"
```

#### `deduplicate`

```yaml
- type: deduplicate
  keys:
    - idelem
    - fecha_hora
  order_by:
    - column: _ingestion_timestamp
      direction: desc
```

Implementar mediante una ventana Spark.

#### `lookup_join`

Necesaria para mapear estaciones o puntos de medida a distritos.

```yaml
- type: lookup_join
  lookup_table:
    layer: silver
    source: trafico
    dataset: dim_trafico
  join_type: left
  conditions:
    idelem: id
  select:
    distrito: distrito
    longitud: longitud
    latitud: latitud
```

El motor resolverá el nombre completo de la tabla. No implementar un lenguaje genérico de joins arbitrarios en la primera versión.

#### `assign_district`

Transformación acotada a las pequeñas dimensiones de estaciones meteorológicas
y de calidad del aire. Utiliza el KML estático versionado dentro del paquete y
añade el código y el nombre del distrito directamente a la dimensión Silver.

```yaml
- type: assign_district
  station_key: codigo_corto
  longitude_column: longitud
  latitude_column: latitud
  district_code_column: distrito_cod
  district_name_column: distrito_nombre
```

El KML del proyecto contiene exactamente 21 contornos cerrados, uno por
distrito. La implementación debe validar ese contrato y fallar si una estación
no pertenece de forma unívoca a un distrito. No aproximar automáticamente al
distrito más cercano. El uso de `collect()` queda permitido únicamente para
estas dimensiones acotadas de pocas decenas de estaciones.

### 14.5 Metadatos Silver

Conservar metadatos técnicos útiles:

```text
_ingestion_timestamp
_ingestion_run_id
_source_file
```

Añadir:

```text
_silver_processed_timestamp
```

---

## 15. Estrategias de escritura Silver

### 15.1 `overwrite`

Adecuada para dimensiones, ficheros estáticos y fotografías completas.

```yaml
silver:
  write_strategy: overwrite
```

Permitir configurar `overwriteSchema` cuando sea necesario.

### 15.2 `merge`

Adecuada para históricos, NRT, correcciones y reprocesos.

```yaml
silver:
  write_strategy: merge
  business_keys:
    - idelem
    - fecha_hora
```

Comportamiento conceptual:

```python
target.alias("t").merge(
    source.alias("s"),
    "t.idelem = s.idelem AND t.fecha_hora = s.fecha_hora",
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
```

Si la tabla no existe:

1. Escribir el DataFrame inicial.
2. Registrar la tabla.
3. No ejecutar un merge contra un destino inexistente.

### 15.3 `append`

Solo usar cuando exista un mecanismo explícito que evite reprocesar el mismo lote:

- Filtrado por `_ingestion_run_id`.
- Watermark.
- Predicado incremental.
- Dataset inmutable con ejecución controlada.

Ejemplo:

```yaml
silver:
  write_strategy: append
  incremental:
    type: ingestion_run_id
```

Si `append` no declara estrategia incremental, la validación debe fallar. No hacer `append` de toda la tabla Bronze en cada ejecución.

---

## 16. Estrategias recomendadas por dataset

Las claves definitivas deben confirmarse tras inspeccionar los esquemas reales.

| Dataset | Estrategia Silver recomendada | Motivo |
|---|---|---|
| `trafico.historico_trafico` | `merge` | Histórico mensual susceptible de reproceso |
| `trafico.trafico_nrt` | `merge` | Datos frecuentes con posible repetición |
| `trafico.dim_trafico` | `overwrite` | Dimensión puntual y pequeña |
| `trafico.dim_distritos` | `overwrite` | Dimensión estática |
| `accidentes.historico_accidentes` | `merge` | Histórico mensual |
| `meteorologia.meteo_nrt` | `merge` | Datos frecuentes |
| `meteorologia.meteo_historico` | `merge` | Histórico mensual |
| `meteorologia.dim_meteo` | `overwrite` | Dimensión puntual |
| `meteorologia.dim_magnitudes_meteo` | `overwrite` | Catálogo estático |
| `eventos.eventos_culturales` | `overwrite` | Ventana móvil de próximos 100 días |
| `calidad_aire.calidad_aire_nrt` | `merge` | Datos frecuentes |
| `calidad_aire.calidad_aire_historico` | `merge` | Histórico mensual |
| `calidad_aire.dim_calair` | `overwrite` | Dimensión puntual |
| `calidad_aire.dim_magnitudes_calair` | `overwrite` | Catálogo estático |

No inventar claves de negocio.

---

## 17. Logging y observabilidad básica

Usar el módulo estándar `logging`.

```python
logger = logging.getLogger("madrid_ingestion")
```

Los mensajes deben incluir, cuando aplique:

- `run_id`.
- Entorno.
- Capa.
- Fuente.
- Dataset.
- Tabla origen.
- Tabla destino.
- Estrategia de escritura.
- Inicio, fin y errores.

No ejecutar `count()` por defecto para imprimir filas. Permitir conteos solo con una opción explícita de diagnóstico.

No implementar inicialmente Application Insights, Log Analytics, OpenTelemetry, tablas de auditoría, Prometheus ni alertas.

---

## 18. Gestión de errores

Crear una jerarquía pequeña:

```python
class IngestionError(Exception):
    """Base exception for the ingestion package."""


class ConfigurationError(IngestionError):
    pass


class DatasetNotFoundError(ConfigurationError):
    pass


class UnsupportedFormatError(IngestionError):
    pass


class UnsupportedTransformationError(IngestionError):
    pass


class WriteStrategyError(IngestionError):
    pass
```

Reglas:

- No capturar excepciones para ocultarlas.
- Añadir contexto al log.
- Encadenar excepciones con `raise ... from exc`.
- No usar `except Exception: pass`.
- No devolver `None` para indicar errores.
- La CLI debe terminar con código distinto de cero cuando falle.

---

## 19. API pública

Mantener una API mínima:

```python
from madrid_ingestion.runner import run_dataset

result = run_dataset(
    environment="dev",
    layer="bronze",
    source="trafico",
    dataset="trafico_nrt",
)
```

Resultado conceptual:

```python
@dataclass(frozen=True)
class RunResult:
    run_id: str
    environment: str
    layer: str
    source: str
    dataset: str
    target_table: str
    target_path: str
    status: str
```

No devolver DataFrames como resultado principal de una ejecución completa.

---

## 20. CLI

Usar inicialmente `argparse`.

Comandos previstos:

```bash
python -m madrid_ingestion run --env dev --layer bronze --source trafico --dataset trafico_nrt
python -m madrid_ingestion run --env dev --layer silver --source trafico --dataset trafico_nrt
python -m madrid_ingestion validate-config --env dev
python -m madrid_ingestion list-datasets --source trafico
```

Opciones recomendadas de `run`:

```text
--env
--layer
--source
--dataset
--run-id
--config-root
--log-level
```

La CLI debe ser un wrapper fino. No duplicar lógica del runner.

---

## 21. Inyección de SparkSession

No depender de una variable global `spark` creada por un notebook.

```python
def run_dataset(
    ...,
    spark: SparkSession | None = None,
) -> RunResult:
    spark_session = spark or SparkSession.getActiveSession()

    if spark_session is None:
        raise IngestionError("No active SparkSession was found.")
```

---

## 22. Registro de lectores

Crear una resolución simple por formato:

```python
READERS = {
    "csv": CsvAutoLoaderReader,
    "xml": XmlAutoLoaderReader,
}
```

No crear un sistema dinámico de plugins.

Interfaz común:

```python
class BaseReader(Protocol):
    def read_stream(
        self,
        spark: SparkSession,
        source_path: str,
        schema_path: str,
        options: Mapping[str, str],
    ) -> DataFrame:
        ...
```

---

## 23. Registro de transformaciones

Usar un registro explícito:

```python
TRANSFORMATIONS = {
    "normalize_column_names": normalize_column_names,
    "rename": rename_columns,
    "select": select_columns,
    "drop": drop_columns,
    "cast": cast_columns,
    "trim": trim_columns,
    "empty_to_null": empty_to_null,
    "replace_values": replace_values,
    "filter": filter_rows,
    "add_literal": add_literal,
    "parse_timestamp": parse_timestamp,
    "deduplicate": deduplicate,
    "lookup_join": lookup_join,
    "assign_district": assign_district,
}
```

No resolver funciones mediante `eval`, `exec` ni importaciones arbitrarias declaradas en YAML.

---

## 24. Particionado

No particionar todas las tablas por defecto.

```yaml
partition_by:
  - fecha
```

Aplicar particionado solo cuando la columna, cardinalidad, volumen y patrón de consulta lo justifiquen.

Evitar particionar por identificadores únicos, timestamps completos o columnas de alta cardinalidad.

La primera versión puede dejar `partition_by: []` en la mayoría de datasets.

---

## 25. Calidad de datos

La calidad avanzada queda fuera del alcance inicial.

Preparar la arquitectura para añadir posteriormente:

- `not_null`.
- Rangos.
- Valores permitidos.
- Unicidad.
- Integridad referencial.
- Cuarentena.
- Métricas de calidad.

En esta fase:

- Mantener `_rescued_data`.
- Registrar errores de configuración.
- No descartar registros silenciosamente.
- No implementar un DSL de calidad.
- No introducir Great Expectations salvo decisión posterior.

---

## 26. Pruebas

Las pruebas se ampliarán incrementalmente.

La arquitectura debe permitir probar:

### Configuración

- Carga de YAML válido.
- Error por campos obligatorios.
- Error por `merge` sin claves.
- Error por `append` sin incrementalidad.
- Error por transformación no soportada.

### Naming y paths

- Catálogo `{env}_{layer}`.
- Tabla `<catalog>.<source>.<dataset>`.
- URI ADLS.
- Normalización de nombres.

### Transformaciones

- Renombrado.
- Cast.
- Trim.
- Deduplicación.
- Conversión de vacíos a nulos.

### Writers

- Selección correcta de estrategia.
- Creación inicial antes de `merge`.
- Error por estrategia desconocida.

No es necesario crear infraestructura completa de integración con Azure en la primera iteración.

---

## 27. Estilo de código

Aplicar:

- Type hints.
- Docstrings en API pública.
- Funciones pequeñas.
- Clases solo cuando encapsulen estado o comportamiento real.
- Nombres descriptivos.
- Imports absolutos.
- Constantes centralizadas.
- Código compatible con linting.

Evitar:

- Clases genéricas sin responsabilidad concreta.
- Herencia profunda.
- Singletons.
- Estado global mutable.
- Funciones de cientos de líneas.
- Parámetros booleanos ambiguos.
- Configuraciones duplicadas.
- Dependencias circulares.

---

## 28. Seguridad

No almacenar:

- Claves de storage.
- Client secrets.
- Tokens.
- SAS.
- Credenciales.
- URLs con secretos.

El paquete recibe nombres de recursos y rutas, no secretos. La autenticación se delega al entorno Databricks.

---

## 29. Databricks Asset Bundles y Jobs

Quedan fuera de la primera versión.

La arquitectura debe permitir añadir después tareas como:

```text
bronze_trafico_nrt
silver_trafico_nrt
bronze_meteo_nrt
silver_meteo_nrt
```

Cada tarea futura invocará la misma API o CLI. No acoplar el motor a widgets de notebook.

---

## 30. Secuencia de implementación

### Iteración 1 — Esqueleto

Crear:

- `pyproject.toml`.
- Layout `src`.
- Modelos Pydantic.
- Cargador YAML.
- Naming.
- Paths.
- Logger.
- Excepciones.
- CLI mínima.
- Runner con validación.

Criterio:

```bash
python -m madrid_ingestion validate-config --env dev
```

### Iteración 2 — Bronze CSV

Implementar:

- Auto Loader CSV.
- Metadatos Bronze.
- Escritura Delta.
- Checkpoint.
- Schema location.
- Registro de tabla externa.

### Iteración 3 — Bronze XML

Implementar:

- Adaptador XML.
- Opciones propias.
- Prueba con tráfico NRT.
- Documentación de dependencias del runtime.

### Iteración 4 — Silver overwrite

Implementar:

- Lectura de Bronze.
- Transformaciones básicas.
- Escritura `overwrite`.
- Registro de tabla.

### Iteración 5 — Silver merge

Implementar:

- Claves de negocio.
- Deduplicación previa.
- Creación inicial.
- Delta merge.

### Iteración 6 — Silver append controlado

Implementar:

- Configuración incremental obligatoria.
- Filtro por lote o watermark.
- Validaciones de idempotencia.

### Iteración 7 — Enriquecimiento geográfico

Implementar:

- `lookup_join`.
- Puntos de tráfico a distrito.
- Estaciones meteo a distrito.
- Estaciones de calidad del aire a distrito.
- Normalización del distrito textual de accidentes y eventos.

### Iteración 8 — Endurecimiento básico

Añadir:

- Pruebas unitarias esenciales.
- Documentación.
- Ejemplos YAML.
- Linting.
- Manejo de errores consistente.

---

## 31. Criterios de aceptación

La primera versión será válida cuando:

1. El paquete se pueda instalar.
2. Los YAML se validen antes de ejecutar Spark.
3. Un CSV pueda ingerirse desde `landing` a Bronze con Auto Loader.
4. El XML de tráfico pueda ingerirse con un lector específico.
5. Bronze añada metadatos técnicos.
6. Bronze escriba Delta externo.
7. La tabla Bronze se registre con el patrón definido.
8. Silver lea exclusivamente desde Bronze.
9. Silver permita transformaciones declarativas básicas.
10. Silver permita `overwrite`.
11. Silver permita `merge` con claves configuradas.
12. `append` falle sin configuración incremental.
13. La ejecución se pueda invocar por API y CLI.
14. El logger muestre inicio, fin y errores con contexto.
15. No exista lógica `if dataset == ...` en el núcleo.
16. Las particularidades residan en YAML o transformaciones explícitas.
17. No se almacenen secretos.
18. El código sea comprensible y defendible en el TFM.

---

## 32. Restricciones para Codex

Al modificar el repositorio:

1. Leer primero este documento y los YAML existentes.
2. No cambiar convenciones sin justificarlo.
3. No añadir dependencias automáticamente.
4. No implementar componentes fuera del alcance.
5. No duplicar lógica entre Bronze y Silver.
6. No introducir valores específicos de `dev` en el código.
7. No hardcodear storage accounts.
8. No hardcodear rutas completas.
9. No inventar esquemas sin muestras reales.
10. No inventar claves de negocio.
11. No usar `eval` ni `exec`.
12. No ocultar excepciones.
13. No usar Pandas para el procesamiento principal.
14. No ejecutar acciones Spark innecesarias.
15. Mantener la API pública pequeña.
16. Crear código ejecutable, no solo pseudocódigo, cuando se solicite implementación.
17. Añadir o actualizar pruebas al modificar lógica cubierta.
18. Mantener ejemplos YAML sincronizados con los modelos Pydantic.
19. Explicar cualquier decisión que aumente significativamente la complejidad.
20. Priorizar una solución defendible académicamente sobre una plataforma sobredimensionada.

---

## 33. Formato esperado de las respuestas de Codex

Cuando se solicite una implementación:

1. Indicar brevemente qué se modificará.
2. Enumerar los ficheros afectados.
3. Implementar el cambio completo.
4. Explicar las decisiones relevantes.
5. Indicar supuestos.
6. Señalar puntos pendientes que dependan de datos reales.
7. Proporcionar comandos de validación o ejecución.
8. No afirmar que una prueba ha pasado si no se ha ejecutado.

Cuando falten detalles menores:

- Usar placeholders claros.
- Documentar el supuesto.
- No bloquear innecesariamente el avance.

Cuando falten detalles que afecten a claves, formatos o semántica:

- No inventarlos.
- Dejar el punto configurable.
- Marcarlo como pendiente de confirmar con una muestra real.

---

## 34. Ejemplo completo de configuración

```yaml
source: trafico
enabled: true

datasets:
  - name: trafico_nrt
    description: "Mediciones de tráfico cercanas a tiempo real"
    enabled: true

    bronze:
      enabled: true
      format: xml
      source_path: "trafico/trafico_nrt"
      target_path: "trafico/trafico_nrt"

      reader_options:
        row_tag: "pm"

      autoloader_options:
        schema_evolution_mode: addNewColumns
        rescued_data_column: _rescued_data

    silver:
      enabled: true
      target_path: "trafico/trafico_nrt"
      write_strategy: merge

      business_keys:
        - idelem
        - fecha_hora

      partition_by: []

      transformations:
        - type: normalize_column_names

        - type: rename
          columns:
            fecha: fecha_raw
            hora: hora_raw

        - type: cast
          columns:
            idelem: long
            intensidad: double
            ocupacion: double
            carga: double
            nivel_servicio: string

        - type: parse_timestamp
          source_columns:
            - fecha_raw
            - hora_raw
          target_column: fecha_hora
          format: "dd/MM/yyyy HH:mm:ss"

        - type: deduplicate
          keys:
            - idelem
            - fecha_hora
          order_by:
            - column: _ingestion_timestamp
              direction: desc

        - type: lookup_join
          lookup_table:
            layer: silver
            source: trafico
            dataset: dim_trafico
          join_type: left
          conditions:
            idelem: id
          select:
            distrito: distrito
            latitud: latitud
            longitud: longitud
```

El ejemplo es orientativo. Los nombres de columnas y formatos deben verificarse con ficheros reales.

---

## 35. Ejemplo de configuración de `trafico.dim_distritos`

```yaml
source: trafico
enabled: true

datasets:
  - name: dim_distritos
    description: "Catálogo de distritos de Madrid"
    enabled: true

    bronze:
      enabled: true
      format: csv
      source_path: "trafico/dim_distritos"
      target_path: "trafico/dim_distritos"

      reader_options:
        delimiter: ";"
        header: "true"
        encoding: "UTF-8"

      autoloader_options:
        schema_evolution_mode: addNewColumns
        rescued_data_column: _rescued_data

    silver:
      enabled: true
      target_path: "trafico/dim_distritos"
      write_strategy: overwrite

      transformations:
        - type: normalize_column_names

        - type: rename
          columns:
            cod_dis: distrito_id
            nombre: distrito_nombre

        - type: cast
          columns:
            distrito_id: integer

        - type: trim
          columns:
            - distrito_nombre

        - type: empty_to_null
          columns:
            - distrito_nombre
```

---

## 36. Resultado arquitectónico esperado

```text
                       ┌─────────────────────────┐
                       │   YAML de configuración │
                       └────────────┬────────────┘
                                    │
                                    ▼
┌─────────────┐          ┌───────────────────────┐
│ CLI / API   │─────────▶│ Runner por dataset    │
└─────────────┘          └───────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
       ┌─────────────────────┐             ┌─────────────────────┐
       │ Processor Bronze    │             │ Processor Silver    │
       │ Auto Loader         │             │ Transformaciones    │
       └──────────┬──────────┘             └──────────┬──────────┘
                  │                                   │
                  ▼                                   ▼
       ┌─────────────────────┐             ┌─────────────────────┐
       │ Delta externo       │             │ Delta externo       │
       │ dev_bronze.*        │             │ dev_silver.*        │
       └─────────────────────┘             └─────────────────────┘
```

La solución debe poder explicarse así:

> Un motor Python carga la definición declarativa de un dataset, resuelve su entorno, ejecuta el procesador de la capa y escribe una tabla Delta externa registrada en Unity Catalog.

---

## 37. Decisión final de alcance

Implementar primero:

```text
Configuración
+ Runner
+ Bronze con Auto Loader
+ Silver con transformaciones básicas
+ Delta externo
+ Merge / Overwrite / Append controlado
+ Logging básico
```

Dejar para fases posteriores:

```text
Asset Bundles
+ Jobs
+ Calidad avanzada
+ Auditoría
+ Métricas
+ Gold
+ Power BI
+ Machine Learning
```

Toda nueva funcionalidad debe evaluarse contra esta pregunta:

> ¿Es necesaria para validar académicamente el motor de ingesta o puede incorporarse en una fase posterior de productivización?
