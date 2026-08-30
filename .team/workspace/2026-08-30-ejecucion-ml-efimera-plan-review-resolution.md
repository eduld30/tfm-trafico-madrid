# Resolución de la revisión staff del plan efímero

## Resolution Summary

Total findings: 4
Resolved: 4
Disputed: 0
Deferred: 0

## Resolutions

### Restaura la pareja Gold si falla la segunda publicación

**Severity:** High
**Resolution:** FIXED
**What changed:** Task 6 ahora exige que ambas tablas Gold estén ausentes antes del submit. Si el run no termina con éxito, el controlador elimina por la API de Unity Catalog cualquiera de las dos salidas creadas y comprueba que ambas vuelven a responder `RESOURCE_DOES_NOT_EXIST`. Si encuentra una tabla preexistente, detiene el preflight y exige una decisión nueva; no intenta restaurar ni borrar estado ajeno.
**Verification:** Se contrastó el orden real de publicación en `src/madrid_ml/snapshot.py`: labels se escribe antes que features. También se verificó que la CLI expone `databricks tables get` y `databricks tables delete` por nombre completo.
**Blast radius:** Especificación efímera, autorización informada, preflight, cleanup, evidencia final y self-review de Task 6.

### Reintenta el submit ambiguo con el mismo token

**Severity:** High
**Resolution:** FIXED
**What changed:** El plan repite durante un máximo de 60 segundos el mismo `jobs submit` con payload e `idempotency_token` idénticos hasta recuperar el único `run_id`. `list-runs --run-type SUBMIT_RUN` queda como evidencia secundaria, no como sustituto de la garantía idempotente.
**Verification:** La documentación oficial de `POST /api/2.2/jobs/runs/submit` establece que un token permite repetir la petición y garantiza exactamente un run.
**Blast radius:** Especificación efímera, resolución del run, cleanup y descripción de idempotencia.

### Instala la limpieza antes de la primera mutación remota

**Severity:** High
**Resolution:** FIXED
**What changed:** Task 6 exige un único controlador con `set -Eeuo pipefail`, variables inicializadas y `trap cleanup EXIT INT TERM` instalado antes de `workspace mkdirs`. El cleanup resuelve el run, cancela y espera estado terminal, elimina salidas creadas en fallo, borra staging local/remoto y exige cero submit runs activos con el token. El watchdog consulta cada 15 segundos y cancela al alcanzar 900 segundos.
**Verification:** La secuencia se contrastó con los puntos de mutación: imports de workspace, submit, dos writes Delta y borrado por Tables API.
**Blast radius:** Preflight, staging, monitorización, cleanup y criterios de cierre.

### Sustituye la ruta operativa del bundle en el README

**Severity:** High
**Resolution:** FIXED
**What changed:** Task 6 modifica `README.md` y hace commit antes de cualquier mutación remota. Debe retirar `bundle run` como ruta ML, advertir que el bundle raíz crearía los nueve jobs de ingesta y documentar el controlador efímero como única ruta autorizada para el smoke.
**Verification:** Se comprobó que `README.md` todavía presenta `ml_training_snapshot` y `databricks bundle run`, mientras el workspace no tiene jobs y el primer bundle plan proponía crear diez.
**Blast radius:** Lista de ficheros de Task 6, preflight Git, `CODE_COMMIT` y documentación operativa.

## Deferred Items

None.

## Disputed Findings

None.
