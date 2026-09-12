# Desarrollo

## Organización del código

- `src/madrid_ingestion/`: configuración, lectores, Bronze, Silver, rutas y escritores.
- `src/madrid_ml/`: contratos, snapshots, preprocessing, entrenamiento, registro y scoring.
- `notebooks/`: wrappers finos para los jobs Databricks.
- `tests/unit/`: pruebas de comportamiento del paquete.

La lógica reutilizable debe permanecer en `src/`. Los notebooks no deben contener rutas personales, runs por defecto ni lógica que solo pueda ejecutarse en una sesión concreta.

El motor también puede invocarse desde Python:

```python
from madrid_ingestion.runner import run_dataset

result = run_dataset(
    environment="dev",
    layer="bronze",
    source="trafico",
    dataset="trafico_nrt",
)
```

En Databricks usa la sesión Spark activa; en pruebas se puede inyectar una
sesión o un doble. La respuesta es un `RunResult`, no un DataFrame.

## Cambios habituales

Para añadir un dataset:

1. Declararlo en `conf/sources/` y mantener rutas relativas estables.
2. Validar claves, estrategia de escritura y transformaciones.
3. Añadir el job o pipeline ADF cuando requiera orquestación propia.
4. Cubrir el comportamiento nuevo con pruebas.

Para añadir una transformación, implementarla en el registro Bronze o Silver
correspondiente. Debe recibir y devolver un DataFrame, ser determinista y no
escribir datos por sí misma.

Los identificadores técnicos usan minúsculas y snake case. No se introduce
código ejecutable ni secretos en YAML.

## Validación

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,databricks]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

Las pruebas de ingesta usan principalmente dobles de Spark y Delta. Las pruebas
ML que ejecutan Spark local requieren Java y el extra `databricks`; ninguna
prueba debe acceder a Azure.

Los objetos `spark` y `dbutils` de los notebooks son proporcionados por Databricks. Las supresiones `# noqa: F821` solo deben usarse en esas referencias inyectadas; el paquete Python debe mantener dependencias y argumentos explícitos.
