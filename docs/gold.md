# Capa Gold

## Gold analítica

`notebooks/run_gold.py` lee Silver y reconstruye mediante `overwrite` el schema
`<gold_catalog>.movilidad`. Las tablas son managed y las de hechos se particionan
por año cuando procede.

| Grupo | Salidas principales |
|---|---|
| Dimensiones | distrito, punto de tráfico, estaciones, magnitudes, fecha y hora |
| Accidentes | detalle y agregado distrito-hora |
| Tráfico | punto-hora y distrito-hora |
| Entorno | meteorología y calidad del aire por distrito-hora y magnitud |
| Eventos | eventos por distrito-hora |
| Integración | `gold_movilidad_distrito_hora` |

La tabla integrada reúne accidentalidad, tráfico, entorno y eventos para
análisis y Power BI. La reconstrucción es completa: no es una actualización
incremental de Gold.

El job `gold_analytics_monthly` está programado para el día 2 de cada mes a las
06:00, después de las cargas históricas, y se despliega inicialmente pausado.

## Gold NRT

`notebooks/run_nrt_predictions.py` lee desde Silver tráfico, meteorología,
calidad del aire y distritos. Construye una fila por cada uno de los 21
distritos, carga el modelo registrado con alias `Champion` y publica:

```text
<gold_catalog>.serving.riesgo_accidente_distrito
<gold_catalog>.serving.v_riesgo_actual_distrito
```

La tabla conserva el histórico mediante `MERGE` con
`cod_distrito + prediction_window_start`; la vista devuelve el lote más
reciente. El cutoff se redondea a diez minutos en hora civil de Madrid y la
predicción cubre los 60 minutos siguientes. `predicted_at` registra también la
hora civil de Madrid y respeta los cambios CET/CEST.

El job `gold_nrt_predictions` no tiene calendario propio. ADF lo invoca tras
completar correctamente la ingesta de tráfico NRT, generando una predicción
cada diez minutos con la última meteorología y calidad del aire disponibles.
Requiere las cuatro tablas Silver, el volumen temporal MLflow y el modelo
promocionado.

## Consumo

Power BI puede usar `movilidad` para el análisis histórico y
`v_riesgo_actual_distrito` para el mapa NRT. La elección entre Import y
DirectQuery depende de la necesidad de actualización del informe; la vista NRT
es la candidata natural a DirectQuery.
