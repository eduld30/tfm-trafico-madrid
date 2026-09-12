# Plataforma de datos de tráfico de Madrid

Motor de ingesta y transformación para el TFM **Arquitectura end-to-end en Azure para la monitorización y el análisis predictivo de accidentes de tráfico en Madrid**.

El repositorio contiene la ingesta, transformación, capa analítica y flujo de
machine learning del proyecto. Recibe datos públicos de tráfico, accidentes,
meteorología, calidad del aire y eventos, y prepara información por distrito de
Madrid para análisis histórico y estimación de riesgo a corto plazo.

## Arquitectura

```text
Fuentes públicas → ADF → ADLS landing
                           ↓
                  Bronze externa → Silver externa
                                         ├─→ Gold movilidad managed → Power BI
                                         └─→ snapshot ML → MLflow → modelo UC
                                                                        ↓
                                  Silver NRT → scoring → Gold serving managed
```

- `landing`: ficheros descargados por ADF.
- `bronze`: ingestión técnica incremental con Auto Loader.
- `silver`: datos normalizados y enriquecidos.
- `gold`: tablas managed para análisis histórico y serving NRT.
- `ml`: proceso transversal que usa Silver y Gold; no es otra capa Medallion.

Las tablas Bronze y Silver son Delta externas en ADLS y se apoyan en external
locations de Unity Catalog. Las tablas Gold y los snapshots ML son managed. Los
catálogos se separan por entorno: `dev_bronze`, `dev_silver`, `dev_gold` y sus
equivalentes `pro_*`.

## Qué contiene el repositorio

```text
conf/                 Configuración de entornos y datasets
src/madrid_ingestion/ Motor Bronze/Silver
src/madrid_ml/        Snapshot, preprocessing, entrenamiento y scoring
notebooks/            Wrappers de ejecución para Databricks
resources/jobs/       Jobs del Databricks Asset Bundle
adf/                  Pipelines, triggers y parámetros de ADF
tests/                Pruebas unitarias
docs/                 Documentación técnica por área
```

## Datos y configuración

Las fuentes y datasets están declarados en `conf/sources/`. La configuración de entorno está en `conf/environments/`. Los YAML describen qué procesar; la lógica de ejecución permanece en el paquete Python.

Los datasets cubren tráfico, accidentes, meteorología, eventos y calidad del aire. Sus rutas de entrada apuntan a la raíz estable del dataset en `landing`; las carpetas temporales de los ficheros no se incorporan a la configuración.

## Inicio rápido local

Requisitos: Python 3.10 o superior; Java solo es necesario para las pruebas que
levantan Spark local. Para instalar el proyecto con dependencias de desarrollo:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,databricks]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

La validación de configuración no necesita Spark. La ejecución de una capa sí
requiere una sesión Spark compatible con Databricks:

```bash
python -m madrid_ingestion validate-config --env dev
python -m madrid_ingestion run --env dev --layer bronze \
  --source trafico --dataset trafico_nrt
```

## Databricks y ADF

El Asset Bundle raíz despliega los jobs de ingesta, Gold y ML en los targets `dev` y `pro`:

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev ml_model_training
```

Los detalles de permisos, volúmenes, CI/CD y promoción a producción están en [`docs/despliegue.md`](docs/despliegue.md). La integración y las recurrencias de ADF están descritas en [`docs/ingesta.md`](docs/ingesta.md).

Antes de activar producción deben configurarse los identificadores de jobs en `adf/params/pro.json`, crear el volumen MLflow de `pro` y conceder los permisos de Unity Catalog necesarios.

## Estado del proyecto

El alcance implementado incluye:

- ingesta Bronze/Silver configurable;
- Gold analítica mensual;
- snapshot Gold para entrenamiento;
- preprocessing y entrenamiento de regresión logística con MLflow;
- promoción del modelo a Unity Catalog;
- scoring NRT por distrito.

La ejecución del snapshot, la promoción del modelo y la activación inicial de
los jobs Gold son operaciones explícitas. Los calendarios de ingesta se
mantienen en ADF y los de Gold se despliegan pausados.

## Documentación

- [Arquitectura](docs/arquitectura.md): componentes, capas y separación de entornos.
- [Ingesta Bronze/Silver y ADF](docs/ingesta.md): configuración, Auto Loader y transformaciones.
- [Capa Gold](docs/gold.md): tablas analíticas, serving y calendarios.
- [Machine learning](docs/ml.md): snapshots, entrenamiento, MLflow y scoring NRT.
- [Despliegue](docs/despliegue.md): Asset Bundles, ADF, permisos y producción.
- [Desarrollo](docs/desarrollo.md): estructura del código, pruebas y extensiones.
