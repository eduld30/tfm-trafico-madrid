# Despliegue

## Databricks Asset Bundle

Desde la raíz del repositorio:

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle validate -t pro
databricks bundle deploy -t pro
```

Si se usa un perfil no predeterminado, se añade `--profile <perfil>`. El bundle
construye un wheel con versión dinámica, sincroniza `conf/` y despliega los jobs
de `resources/jobs/` sobre compute serverless con entorno 4.

| Grupo | Recursos |
|---|---|
| Ingesta | dimensiones y ocho jobs de hechos NRT/históricos |
| Gold | `gold_analytics_monthly`, `gold_nrt_predictions` |
| ML | `ml_training_snapshot`, `ml_model_training`, `ml_model_promotion` |

Los jobs de ingesta se invocan desde ADF y los jobs ML son manuales.
`gold_analytics_monthly` conserva su calendario inicialmente pausado;
`gold_nrt_predictions` no tiene calendario propio y ADF lo invoca después de
la ingesta de tráfico NRT.

## Producción

Antes de activar el flujo en `pro` hay que comprobar:

- `conf/environments/pro.yaml` y sus catálogos;
- los IDs de jobs en `adf/params/pro.json`;
- los permisos de la identidad administrada de ADF;
- los permisos de la identidad de despliegue en Databricks y Unity Catalog;
- las external locations de ADLS utilizadas por Bronze y Silver;
- `/Volumes/pro_gold/ml/mlflow_tmp`;
- el modelo registrado y su alias `Champion` antes de activar NRT.

`adf/params/pro.json` contiene marcadores `<<pendiente>>` mientras no se hayan
asignado los IDs de jobs productivos. La validación de CI falla deliberadamente
si queda alguno.

## CI/CD

`.github/workflows/ci-cd.yml` aplica esta correspondencia:

| Rama | Target |
|---|---|
| `develop` | `dev` |
| `main` | `pro` |

Las pull requests validan Python, ADF y el bundle. Los pushes y ejecuciones
manuales despliegan primero Databricks y después ADF. La autenticación usa
GitHub OIDC: la identidad federada necesita acceso al workspace, catálogos,
external locations y factoría de cada entorno. Hosts, grupos de recursos e IDs
de Azure se proporcionan mediante variables del repositorio.
