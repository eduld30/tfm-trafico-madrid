# Contrato de preprocessing ML v1

## 1. Objetivo

Definir un preprocessing único y reproducible para el primer baseline retrospectivo de regresión logística sobre `dev_gold.ml.features_training_snapshot`.

La v1 evita fuga del estado aprendido entre los splits: ningún estimador usa filas posteriores a 2023 durante `fit`. No demuestra disponibilidad point-in-time ni latencia de publicación porque Gold v1 no contiene `feature_cutoff_ts` o `published_at`.

El contrato resuelve únicamente:

- disponibilidad de observaciones;
- nulos, `NaN` e infinitos;
- valores físicamente inválidos;
- imputación;
- codificación categórica;
- escalado necesario para la regresión logística;
- conservación del split temporal aprobado.

No selecciona features, no ajusta pesos de clase, no calibra probabilidades y no define umbrales de decisión.

## 2. Evidencia de partida

El EDA del snapshot `ea719086-1a93-401c-969b-4e92586e13fd` se ejecutó sobre las versiones Delta 1 de:

- `dev_gold.ml.accident_labels_hourly`;
- `dev_gold.ml.features_training_snapshot`.

El snapshot contiene 1.379.931 filas, 21 distritos y 136.293 positivos. El gate estructural terminó como `READY_FOR_BASELINE_SPEC`.

De las 36 features numéricas distintas del distrito y del target, 18 medias contienen valores ausentes o no finitos. La disponibilidad observada varía desde aproximadamente 23 % para algunas magnitudes de calidad del aire hasta más de 99 % para intensidad y carga de tráfico. También se observaron valores físicamente imposibles: velocidad de tráfico negativa, humedad fuera de `[0, 100]`, radiación negativa y concentraciones negativas.

Esta evidencia justifica un tratamiento explícito de ausencia y rangos. No justifica eliminar features mediante un umbral de cobertura ni crear interpolaciones o ventanas nuevas.

## 3. Frontera y versiones

### 3.1 Entrada

El entrypoint público recibe:

```text
spark
gold_catalog
labels_delta_version
features_delta_version
expected_snapshot_id
```

Resuelve `accident_labels_hourly` y `features_training_snapshot`, exige las dos versiones
Gold autorizadas y lee cada tabla mediante `versionAsOf`. Para el snapshot aprobado en
esta especificación ambas versiones valen `1`.

Antes del `fit`, valida que labels y features comparten el `expected_snapshot_id` y un
único `input_versions_json`, `code_commit`, `feature_schema_version` y `time_contract`.
`input_versions_json` conserva las versiones Silver; no sustituye a las versiones Gold
recibidas explícitamente.

El DataFrame versionado de features debe cumplir:

```text
feature_schema_version = "1"
time_contract = "source_wall_clock_as_stored_in_silver"
```

La clave lógica permanece:

```text
(cod_distrito, feature_hour)
```

El preprocessing no modifica Gold ni materializa otra tabla.

### 3.2 Versión del contrato

```text
preprocessing_contract_version = "1"
```

Cambiar una feature de entrada, una regla física, el orden del vector o un periodo requiere incrementar esta versión.

## 4. Split temporal

Los periodos se expresan como intervalos semiabiertos:

| Tramo | Intervalo de `feature_hour` | Uso |
|---|---|---|
| Train | `[2019-01-01 00:00:00, 2024-01-01 00:00:00)` | Ajustar preprocessing y modelo |
| Validación | `[2024-01-01 00:00:00, 2025-01-01 00:00:00)` | Decisiones de modelo |
| Evaluación final | `[2025-01-01 00:00:00, 2026-01-01 00:00:00)` | Evaluación final |
| Fuera del protocolo | Desde `2026-01-01 00:00:00` | Excluido |

Antes de construir los límites, el entrypoint fija
`spark.sql.session.timeZone="Etc/UTC"` y comprueba que la sesión conserva ese valor.
El timezone se registra en el manifiesto.

Para el snapshot aprobado, los recuentos exactos son:

| Tramo | Filas |
|---|---:|
| Train | 920.304 |
| Validación | 184.464 |
| Evaluación final | 183.960 |
| Excluidas desde 2026 | 91.203 |

Los tres splits incluidos suman 1.288.728 filas. La reconciliación obligatoria es
`1.288.728 + 91.203 = 1.379.931`. Cada `transform` conserva exactamente las filas,
labels y claves `(cod_distrito, feature_hour)` de su split; la conservación no se
compara contra las filas 2026 excluidas.

Medianas, media, desviación estándar y codificadores se ajustan exclusivamente con train. Validación y evaluación solo ejecutan `transform`.

No se realiza split aleatorio, sobremuestreo ni balanceo dentro del preprocessing.

## 5. Columnas de entrada

### 5.1 Target y columnas excluidas

El label es:

```text
target_accident_next_hour
```

Debe ser entero y pertenecer a `{0, 1}`.

Quedan fuera de `X`:

- `n_accidentes_next_hour`;
- `feature_hour` y `prediction_hour`, salvo para dividir los periodos;
- `snapshot_id`;
- `input_versions_json`;
- `code_commit`;
- `feature_schema_version`;
- `time_contract`.

### 5.2 Categóricas

Se tratan como categorías, no como magnitudes ordinales:

| Columna | Dominio exacto |
|---|---|
| `cod_distrito` | enteros `1..21` |
| `hora_dia` | enteros `0..23` |
| `dia_semana` | enteros `1..7` |
| `mes` | enteros `1..12` |

Un valor nulo o fuera del dominio bloquea la ejecución. Train debe contener todas las categorías esperadas.

### 5.3 Medias continuas

Tráfico:

- `trafico_intensidad_media`;
- `trafico_ocupacion_media`;
- `trafico_carga_media`;
- `trafico_vmed_media`.

Meteorología:

- `meteo_velocidad_viento_media`;
- `meteo_temperatura_media`;
- `meteo_humedad_relativa_media`;
- `meteo_presion_media`;
- `meteo_radiacion_solar_media`;
- `meteo_precipitacion_media`.

Calidad del aire:

- `calair_so2_media`;
- `calair_co_media`;
- `calair_no_media`;
- `calair_no2_media`;
- `calair_pm25_media`;
- `calair_pm10_media`;
- `calair_nox_media`;
- `calair_o3_media`.

### 5.4 Conteos de cobertura

- `trafico_puntos_n`;
- `meteo_velocidad_viento_n`;
- `meteo_temperatura_n`;
- `meteo_humedad_relativa_n`;
- `meteo_presion_n`;
- `meteo_radiacion_solar_n`;
- `meteo_precipitacion_n`;
- `calair_so2_n`;
- `calair_co_n`;
- `calair_no_n`;
- `calair_no2_n`;
- `calair_pm25_n`;
- `calair_pm10_n`;
- `calair_nox_n`;
- `calair_o3_n`.

Deben ser enteros, no nulos y mayores o iguales que cero. Se conservan como features numéricas.

## 6. Saneamiento y disponibilidad

### 6.1 Regla común

Para cada una de las 18 medias `x`:

1. `null`, `NaN`, `+Inf` y `-Inf` se consideran ausentes.
2. Un valor finito fuera del rango físico de la sección 6.2 se considera inválido.
3. Ausentes e inválidos se convierten al mismo valor interno ausente antes de imputar.
4. Se crea siempre `x_available`:
   - `1.0` si el valor original es finito y físicamente válido;
   - `0.0` en otro caso.
5. Se registran por split, fuera del vector del modelo:
   - `x_missing_input_count`;
   - `x_invalid_range_count`.

No se eliminan filas. Un cero físicamente válido permanece cero y produce `x_available=1.0`.

### 6.2 Rangos físicos duros

| Feature | Condición válida |
|---|---|
| `trafico_intensidad_media` | finito y `>= 0` |
| `trafico_ocupacion_media` | finito y `>= 0` |
| `trafico_carga_media` | finito y `>= 0` |
| `trafico_vmed_media` | finito y `>= 0` |
| `meteo_velocidad_viento_media` | finito y `>= 0` m/s |
| `meteo_temperatura_media` | cualquier valor finito |
| `meteo_humedad_relativa_media` | finito y `0 <= x <= 100` % |
| `meteo_presion_media` | finito y `> 0` mb |
| `meteo_radiacion_solar_media` | finito y `>= 0` W/m² |
| `meteo_precipitacion_media` | finito y `>= 0` l/m² |
| Cada `calair_*_media` | finito y `>= 0` en su unidad declarada |

No se fijan máximos para tráfico ni límites plausibles para temperatura y presión: el repositorio no contiene una fuente que los justifique. Valores como temperatura `-55` o presión `1` se conservan y se reportan. No se aplica clipping por percentiles.

### 6.3 Coherencia con los conteos

Bloquean la ejecución:

- una media meteorológica o de aire finita cuando su conteo asociado es `0`;
- cualquier media de tráfico finita cuando `trafico_puntos_n` es `0`.

Se permite un conteo positivo con media ausente o inválida. Representa observaciones recibidas sin un valor utilizable y queda expresado por `x_available=0.0`.

## 7. Imputación y transformación

### 7.1 Imputación

Cada media saneada se imputa con su mediana válida de train mediante Spark ML `Imputer(strategy="median", relativeError=0.001)`.

La ejecución falla si una media no tiene ningún valor válido en train. No se imputa mediante:

- cero;
- media global;
- media por distrito;
- interpolación temporal;
- forward/backward fill;
- datos de validación o evaluación.

### 7.2 Escalado numérico

Las 18 medias imputadas y los 15 conteos se ensamblan en un vector numérico de 33 componentes.

La tupla exacta, indexada desde cero, es:

```text
(
  trafico_intensidad_media,
  trafico_ocupacion_media,
  trafico_carga_media,
  trafico_vmed_media,
  meteo_velocidad_viento_media,
  meteo_temperatura_media,
  meteo_humedad_relativa_media,
  meteo_presion_media,
  meteo_radiacion_solar_media,
  meteo_precipitacion_media,
  calair_so2_media,
  calair_co_media,
  calair_no_media,
  calair_no2_media,
  calair_pm25_media,
  calair_pm10_media,
  calair_nox_media,
  calair_o3_media,
  trafico_puntos_n,
  meteo_velocidad_viento_n,
  meteo_temperatura_n,
  meteo_humedad_relativa_n,
  meteo_presion_n,
  meteo_radiacion_solar_n,
  meteo_precipitacion_n,
  calair_so2_n,
  calair_co_n,
  calair_no_n,
  calair_no2_n,
  calair_pm25_n,
  calair_pm10_n,
  calair_nox_n,
  calair_o3_n,
)
```

No se permite intercalar cada media con su conteo ni ordenar columnas por nombre.

Se aplica Spark ML `StandardScaler(withMean=true, withStd=true)` ajustado en train.

No se usa `RobustScaler`: en features con más del 75 % de valores imputados a la misma
mediana, como `calair_co_media` y `calair_so2_media`, el IQR puede ser cero aunque la
feature conserve observaciones útiles. Eso introduciría una rama de fallback o un gate
que bloquearía el snapshot observado. Tras retirar los valores físicamente inválidos,
el escalado estándar es el mecanismo más pequeño y estable para el baseline.

Una feature con desviación estándar nula en train bloquea la ejecución para evitar una
dimensión numérica degenerada.

### 7.3 Codificación categórica

Las categorías se convierten a una representación canónica ordenable, se indexan de forma ascendente y se codifican con Spark ML `OneHotEncoder(dropLast=true, handleInvalid="error")`.

La representación canónica usa decimal con ceros a la izquierda: dos dígitos para
distrito, hora y mes, y un dígito para día de semana. `StringIndexer` se configura con
`stringOrderType="alphabetAsc"` y `handleInvalid="error"`.

El contrato exige los tamaños:

- distrito: 20 componentes;
- hora: 23;
- día de semana: 6;
- mes: 11.

La categoría de referencia es la última del dominio ordenado: distrito 21, hora 23, día 7 y mes 12.

### 7.4 Vector final

Orden fijo:

1. vector numérico estandarizado: 33 componentes;
2. indicadores `x_available` en el orden de la sección 5.3: 18;
3. one-hot de distrito, hora, día y mes: 60.

Dimensión total:

```text
33 + 18 + 60 = 111
```

El vector debe ser finito. Su orden y dimensión forman parte de `preprocessing_contract_version="1"`.

## 8. Arquitectura mínima

Se usará Spark ML y transformaciones DataFrame. No se recopila el dataset en el driver ni se convierte a Pandas.

Flujo:

```text
Gold leída con versionAsOf
  -> validar versiones, snapshot, lineage y timezone
  -> reconciliar y dividir por feature_hour
  -> sanear y crear disponibilidad
  -> fit del preprocessor solo con train
  -> transform de train, validación y evaluación
  -> regresión logística
```

El preprocessing se implementará como una unidad con estas responsabilidades:

- constantes ordenadas del contrato;
- saneamiento determinista de un DataFrame;
- validación estructural;
- construcción y ajuste del pipeline Spark ML;
- transformación sin reajuste;
- generación de un manifiesto legible.

No se crea un framework genérico de transformaciones ni una configuración declarativa adicional.

## 9. Artefacto reproducible

El modelo y el preprocessing deben registrar:

- `preprocessing_contract_version`;
- `feature_schema_version`;
- `snapshot_id`;
- versiones Delta Gold;
- commit;
- runtime;
- `spark.sql.session.timeZone`;
- `Imputer.relativeError`;
- filas por split y filas excluidas;
- hash o versión del paquete ML;
- periodos exactos;
- lista y orden de inputs;
- reglas físicas;
- medianas de train;
- media y desviación estándar de train;
- categorías y referencias one-hot;
- dimensión final;
- métricas de ausentes e inválidos por split.

Todo el estado aprendido reside en el `PipelineModel`. El saneamiento determinista y
los indicadores se aplican mediante el wrapper público del paquete
`madrid_ml.preprocessing` antes de invocar ese modelo. El artefacto de MLflow fija el
wheel o versión del paquete, el `PipelineModel` y `preprocessing_manifest.json`;
`PipelineModel` aislado no constituye el contrato completo sobre Gold raw.

No se crea una tabla Gold preprocesada, un feature store ni un segundo snapshot.

## 10. Errores y gates

Una única excepción pública, `PreprocessingContractError`, indica que el entrenamiento debe detenerse.

Bloquean la ejecución:

1. versión Gold distinta de la autorizada;
2. schema o lineage incompatible;
3. snapshot distinto del esperado o no único;
4. timezone distinto de `Etc/UTC`;
5. recuentos por split o reconciliación con las 91.203 filas excluidas incorrectos;
6. categoría, target o conteo inválido;
7. incoherencia de conteo cero con media finita;
8. ausencia total de valores válidos en train para una media;
9. categoría esperada ausente en train;
10. desviación estándar nula;
11. cambio de filas, claves o labels dentro de un split durante la transformación;
12. vector no finito;
13. dimensión distinta de 111;
14. intento de ajustar estado con validación o evaluación.

Los valores continuos inválidos no bloquean individualmente: se convierten a ausentes y se contabilizan.

## 11. Verificación

Pruebas focalizadas:

1. `NaN`, infinitos y negativos imposibles terminan imputados con disponibilidad `0`.
2. Un cero válido se conserva con disponibilidad `1`.
3. Humedad fuera de `[0, 100]` se invalida.
4. Temperatura negativa permanece válida.
5. La mediana de validación no altera el estado aprendido con train.
6. Conteos y categorías inválidos fallan.
7. Conteo cero con media finita falla.
8. Los timestamps de frontera se asignan al split correcto con `Etc/UTC`.
9. Una versión Gold o un `expected_snapshot_id` distintos fallan antes del `fit`.
10. Los cuatro recuentos del protocolo se reconcilian con 1.379.931.
11. Se conservan filas, claves y labels dentro de cada split incluido.
12. El vector final es finito, ordenado y de dimensión 111.
13. El manifiesto coincide con el `PipelineModel` y el paquete registrados.

Prueba de ejecución:

- un único smoke serverless y acotado contra las versiones Gold autorizadas;
- sin escrituras en Gold;
- timeout, cero reintentos, estado terminal y limpieza verificados;
- resultado compacto recuperable por Jobs API.

## 12. Exclusiones

Fuera de v1:

- selección automática por cobertura;
- eliminación de filas incompletas;
- clipping o winsorization;
- imputación por distrito;
- interpolación temporal;
- ventanas y lags;
- PCA;
- sobremuestreo;
- pesos de clase;
- pipelines distintos por algoritmo;
- LightGBM;
- feature store;
- tabla preprocesada;
- scoring y paridad histórico/NRT;
- garantía point-in-time y latencia de publicación;

Estas decisiones se reconsideran únicamente si la regresión logística y la validación 2024 demuestran una limitación concreta.

## 13. Criterios de aceptación

El contrato está implementado cuando:

1. la misma entrada versionada, snapshot, paquete, runtime y configuración de sesión producen el mismo vector y manifiesto;
2. ningún estado aprendido usa filas posteriores a 2023;
3. la sesión usa `Etc/UTC` y las fronteras se asignan al split correcto;
4. train, validación, evaluación y exclusiones contienen 920.304, 184.464, 183.960 y 91.203 filas;
5. cada split incluido conserva sus filas, claves y labels antes y después de `transform`;
6. las cuatro particiones del protocolo reconcilian las 1.379.931 filas de entrada;
7. todas las ausencias e invalideces quedan observables mediante indicadores y métricas;
8. el vector tiene 111 componentes finitos;
9. las pruebas focalizadas pasan;
10. el smoke serverless sobre las versiones Gold autorizadas termina correctamente y deja cero recursos activos;
11. el artefacto registrado y el paquete fijado transforman validación y evaluación sin recalcular estado.
