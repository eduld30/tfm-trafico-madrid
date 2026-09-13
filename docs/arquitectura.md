# Arquitectura

## Componentes y responsabilidades

```text
Fuentes públicas → Azure Data Factory → ADLS landing
                                             ↓
                              Databricks Bronze → Silver
                                                    ├─→ Gold movilidad → Power BI
                                                    └─→ MLflow / modelo UC
                                                               ↓
                              Silver NRT → scoring → Gold serving
```

- ADF descarga los ficheros, organiza las rutas de landing e inicia los jobs de
  ingesta.
- Databricks contiene el motor Python, Auto Loader, las transformaciones, Gold
  y ML.
- ADLS almacena landing y las tablas Delta externas Bronze/Silver.
- Unity Catalog gobierna catálogos, schemas, tablas, vistas, modelos, volúmenes
  y external locations.
- Power BI consume las salidas Gold; no accede a Bronze ni a landing.

## Persistencia

| Área | Responsabilidad | Tipo |
|---|---|---|
| Landing | Ficheros raw y staging | ADLS |
| Bronze | Ingesta técnica y trazabilidad | Delta externa |
| Silver | Limpieza, tipado, deduplicación y joins | Delta externa |
| Gold movilidad | Modelo analítico histórico | Delta managed |
| Gold serving | Predicciones NRT y vista actual | Delta managed / vista |
| ML | Snapshots, artefactos y modelo registrado | Delta managed / MLflow / UC Model |

Los catálogos disponen de ubicaciones en ADLS. Bronze y Silver las utilizan
para tablas externas. El catálogo Gold tiene como almacenamiento administrado
el contenedor `gold`: sus tablas y los snapshots ML son managed por Unity
Catalog, aunque los datos se almacenan físicamente en ese contenedor de ADLS.

## Nombres y entornos

Los entornos `dev` y `pro` se definen en `conf/environments/` y separan cuenta
de almacenamiento y catálogos:

```text
<env>_bronze.<source>.<dataset>
<env>_silver.<source>.<dataset>
<env>_gold.movilidad.<table>
<env>_gold.ml.<object>
<env>_gold.serving.<object>
```

El Asset Bundle inyecta el entorno con `${var.environment}`. MLflow también se
separa: `/Shared/madrid-ml-dev` y `/Shared/madrid-ml-pro`.

## Principios de diseño

- YAML declara qué procesar; Python implementa cómo hacerlo.
- La unidad mínima de ejecución es un dataset y una capa.
- Silver nunca lee directamente de landing.
- La lógica reutilizable vive en `src/`; los notebooks son wrappers.
- Se priorizan operaciones Spark nativas e idempotentes.
