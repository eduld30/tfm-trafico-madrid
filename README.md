# Madrid Ingestion

Motor configurable de ingesta Bronze/Silver para el TFM de monitorización del
tráfico de Madrid. El paquete está diseñado para ejecutarse en Azure Databricks,
escribe tablas Delta externas en ADLS y las registra en Unity Catalog.

## Instalación

La versión objetivo es Python 3.10 o superior. En Databricks, `pyspark` y Delta
ya están incluidos en el runtime:

```bash
pip install -e .
pip install -e ".[dev]"
```

## Configuración

- `conf/environments`: cuentas, contenedores, catálogos y opciones de runtime.
- `conf/sources`: definición declarativa de fuentes y datasets.
- `conf/schemas`: reservado para esquemas explícitos cuando se validen muestras
  reales; Auto Loader puede inferirlos en la primera versión.

Los valores `replace-with-...` de los entornos deben sustituirse por nombres de
cuentas de almacenamiento, nunca por secretos. La autenticación se delega al
entorno de Databricks.

Los Silver históricos y NRT que requieren `merge` están deshabilitados en los
ejemplos hasta confirmar sus claves y formatos con datos reales. El motor ya
implementa `merge`; basta con declarar `business_keys` y habilitar la capa.

Para un `append` con `incremental.type: ingestion_run_id`, Silver debe recibir
el mismo `--run-id` que identificó la ingestión Bronze que se desea publicar.

### Nombres de columnas Bronze

Bronze puede aplicar exclusivamente transformaciones técnicas de nombres antes
de escribir Delta. No modifican tipos ni valores:

```yaml
bronze:
  transformations:
    - type: normalize_column_names
    - type: rename
      columns:
        nombre_normalizado_del_origen: nombre_tecnico
```

Las transformaciones se ejecutan en el orden declarado. `rename` debe referirse
a los nombres existentes en ese punto de la secuencia. Cualquier limpieza,
cast, filtro, deduplicación o enriquecimiento continúa siendo responsabilidad
de Silver. `_rescued_data` y los metadatos con prefijo `_` conservan ese
prefijo durante la normalización.

## CLI

```bash
python -m madrid_ingestion validate-config --env dev
python -m madrid_ingestion list-datasets --source trafico
python -m madrid_ingestion run --env dev --layer bronze \
  --source trafico --dataset trafico_nrt
```

También puede ejecutarse una fuente completa o todos los datasets habilitados:

```bash
python -m madrid_ingestion run --env dev --layer bronze --source trafico
python -m madrid_ingestion run --env dev --layer bronze --all
```

## API

```python
from madrid_ingestion.runner import run_dataset

result = run_dataset(
    environment="dev",
    layer="bronze",
    source="trafico",
    dataset="trafico_nrt",
)
```

Puede inyectarse una `SparkSession` mediante el argumento `spark`. Si no se
indica, se usa la sesión activa.

## XML

El lector XML usa Auto Loader con `cloudFiles.format=xml` y traduce la opción
de configuración `row_tag` a `rowTag`. El Databricks Runtime seleccionado debe
incluir soporte del formato XML (nativo en runtimes modernos o mediante el
conector XML correspondiente en runtimes anteriores).

## Validación local

```bash
pytest
ruff check .
```

Las pruebas unitarias no necesitan acceso a Azure. Las pruebas Spark/Delta de
integración deberán ejecutarse posteriormente en un entorno Databricks.
