#!/usr/bin/env bash
set -Eeuo pipefail

export DATABRICKS_AUTH_STORAGE="${DATABRICKS_AUTH_STORAGE:-plaintext}"
export DATABRICKS_CONFIG_PROFILE="${DATABRICKS_CONFIG_PROFILE:-tfm-dev}"

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOCAL_NOTEBOOK="$ROOT_DIR/notebooks/smoke_serverless_delta.py"
STAMP=$(date -u +%Y%m%dT%H%M%S%3NZ)
TOKEN="omp-delta-smoke-$STAMP"
TABLE_SUFFIX=$(printf '%s' "$STAMP" | tr '[:upper:]' '[:lower:]')
TEMPORARY_TABLE="dev_gold.ml.__tmp_serverless_delta_$TABLE_SUFFIX"
OWNER_PROPERTY="madrid_ml.smoke_owner"
PAYLOAD=""
RUN_ID=""
TASK_RUN_ID=""
SUBMIT_STARTED_AT_MS=""
WORKSPACE_ROOT=""
NOTEBOOK_PATH=""
TABLE_PREFLIGHT_ABSENT=false
WORKSPACE_PREFLIGHT_ABSENT=false
RUN_JSON=""

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

table_state() {
  local output
  if output=$(databricks tables get "$TEMPORARY_TABLE" -o json 2>&1); then
    printf '%s' "$output"
    return 0
  fi
  if [[ "$output" == *"RESOURCE_DOES_NOT_EXIST"* \
    || "$output" == *"does not exist"* \
    || "$output" == *"doesn't exist"* ]]; then
    return 1
  fi
  printf 'Unexpected table lookup failure: %s\n' "$output" >&2
  return 2
}

workspace_state() {
  local path=$1 output
  if output=$(databricks workspace get-status "$path" -o json 2>&1); then
    return 0
  fi
  if [[ "$output" == *"RESOURCE_DOES_NOT_EXIST"* \
    || "$output" == *"does not exist"* \
    || "$output" == *"doesn't exist"* ]]; then
    return 1
  fi
  printf 'Unexpected workspace lookup failure for %s: %s\n' "$path" "$output" >&2
  return 2
}

run_json_matches_contract() {
  local run_json=$1
  jq -e \
    --arg token "$TOKEN" \
    --arg notebook_path "$NOTEBOOK_PATH" \
    --arg temporary_table "$TEMPORARY_TABLE" \
    '(.run_type == "SUBMIT_RUN")
     and (.run_name == $token)
     and (.tasks | length == 1)
     and (.tasks[0].task_key == "serverless_delta_round_trip")
     and (.tasks[0].environment_key == "default")
     and (.tasks[0].attempt_number == 0)
     and (.tasks[0].notebook_task.notebook_path == $notebook_path)
     and (.tasks[0].notebook_task.base_parameters == {
       temporary_table: $temporary_table,
       ownership_token: $token
     })' \
    <<<"$run_json" >/dev/null
}

load_owned_run() {
  RUN_JSON=$(databricks jobs get-run "$RUN_ID" -o json) || return 1
  run_json_matches_contract "$RUN_JSON"
}

run_is_terminal() {
  local life_cycle
  load_owned_run || return 1
  life_cycle=$(jq -r '.state.life_cycle_state // .status.state // empty' <<<"$RUN_JSON")
  [[ "$life_cycle" == "TERMINATED" \
    || "$life_cycle" == "SKIPPED" \
    || "$life_cycle" == "INTERNAL_ERROR" ]]
}

wait_for_terminal_after_cancel() {
  local deadline
  deadline=$(( $(date +%s) + 120 ))
  while ! run_is_terminal; do
    if (( $(date +%s) >= deadline )); then
      return 1
    fi
    sleep 5
  done
}

reconcile_run_id() {
  local runs_json count candidate
  runs_json=$(databricks jobs list-runs --run-type SUBMIT_RUN -o json) || return 2
  count=$(jq --arg token "$TOKEN" '[.[] | select(.run_name == $token)] | length' <<<"$runs_json") || return 2
  if [[ "$count" == 0 ]]; then
    return 1
  fi
  if [[ "$count" != 1 ]]; then
    printf 'Expected at most one run for token %s, found %s\n' "$TOKEN" "$count" >&2
    return 2
  fi
  candidate=$(jq -er --arg token "$TOKEN" '.[] | select(.run_name == $token) | .run_id' <<<"$runs_json") || return 2
  RUN_ID=$candidate
  if ! load_owned_run; then
    printf 'Refusing to manage reconciled run with unexpected identity: %s\n' "$RUN_ID" >&2
    RUN_ID=""
    return 2
  fi
}

cleanup_table() {
  local table_json state owner
  set +e
  table_json=$(table_state)
  state=$?
  set -e
  case "$state" in
    0)
      owner=$(jq -r --arg property "$OWNER_PROPERTY" '.properties[$property] // empty' <<<"$table_json") || return 1
      if [[ "$owner" != "$TOKEN" ]]; then
        printf 'Refusing to delete table owned by %q: %s\n' "$owner" "$TEMPORARY_TABLE" >&2
        return 1
      fi
      log "CLEANUP deleting temporary_table=$TEMPORARY_TABLE owner=$owner"
      databricks tables delete "$TEMPORARY_TABLE" >/dev/null || return 1
      ;;
    1) ;;
    2) return 1 ;;
  esac

  set +e
  table_state >/dev/null
  state=$?
  set -e
  [[ "$state" == 1 ]]
}

cleanup_workspace() {
  local state

  set +e
  workspace_state "$NOTEBOOK_PATH"
  state=$?
  set -e
  case "$state" in
    0)
      log "CLEANUP deleting notebook_path=$NOTEBOOK_PATH"
      databricks workspace delete "$NOTEBOOK_PATH" >/dev/null || return 1
      ;;
    1) ;;
    2) return 1 ;;
  esac

  set +e
  workspace_state "$WORKSPACE_ROOT"
  state=$?
  set -e
  case "$state" in
    0)
      log "CLEANUP deleting empty workspace_root=$WORKSPACE_ROOT"
      databricks workspace delete "$WORKSPACE_ROOT" >/dev/null || return 1
      ;;
    1) ;;
    2) return 1 ;;
  esac

  set +e
  workspace_state "$WORKSPACE_ROOT"
  state=$?
  set -e
  [[ "$state" == 1 ]]
}

cleanup() {
  local failed=0 reconcile_state active_json active_count saved_json saved_count

  if [[ -n "$SUBMIT_STARTED_AT_MS" && -z "$RUN_ID" ]]; then
    set +e
    reconcile_run_id
    reconcile_state=$?
    set -e
    case "$reconcile_state" in
      0) log "CLEANUP reconciled run_id=$RUN_ID" ;;
      1) ;;
      2) failed=1 ;;
    esac
  fi

  if [[ -n "$RUN_ID" ]]; then
    if ! load_owned_run; then
      failed=1
    elif ! run_is_terminal; then
      log "CLEANUP cancelling run_id=$RUN_ID"
      databricks jobs cancel-run "$RUN_ID" --no-wait >/dev/null || failed=1
      wait_for_terminal_after_cancel || failed=1
    fi
  fi

  if [[ "$TABLE_PREFLIGHT_ABSENT" == true ]]; then
    cleanup_table || failed=1
  fi
  if [[ "$WORKSPACE_PREFLIGHT_ABSENT" == true && -n "$WORKSPACE_ROOT" ]]; then
    cleanup_workspace || failed=1
  fi

  if [[ -n "$SUBMIT_STARTED_AT_MS" ]]; then
    active_json=$(
      databricks jobs list-runs \
        --run-type SUBMIT_RUN \
        --active-only \
        -o json
    ) || failed=1
    if [[ -n "${active_json:-}" ]]; then
      active_count=$(jq --arg token "$TOKEN" '[.[] | select(.run_name == $token)] | length' <<<"$active_json") || failed=1
      [[ "$active_count" == 0 ]] || failed=1
    fi
  fi

  saved_json=$(databricks jobs list --name "$TOKEN" -o json) || failed=1
  if [[ -n "${saved_json:-}" ]]; then
    saved_count=$(jq 'length' <<<"$saved_json") || failed=1
    [[ "$saved_count" == 0 ]] || failed=1
  fi

  if [[ -n "$PAYLOAD" ]]; then
    rm -f "$PAYLOAD"
  fi
  if ((failed != 0)); then
    log "CLEANUP_FAILED token=$TOKEN"
    return 1
  fi
  log "CLEANUP_OK table_absent=true workspace_absent=true active_runs=0 saved_jobs=0"
}

finish() {
  local status=$?
  trap - EXIT INT TERM
  cleanup || status=1
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
PAYLOAD=$(mktemp)

resolve_run_id() {
  local deadline response
  deadline=$(( $(date +%s) + 30 ))
  while (( $(date +%s) < deadline )); do
    if response=$(databricks jobs submit --json @"$PAYLOAD" --no-wait -o json 2>&1); then
      :
    fi
    RUN_ID=$(jq -r '.run_id // empty' <<<"${response:-}" 2>/dev/null || true)
    if [[ -n "$RUN_ID" ]]; then
      return 0
    fi
    sleep 3
  done
  printf 'Unable to resolve idempotent submit response: %s\n' "${response:-empty}" >&2
  return 1
}

command -v databricks >/dev/null
command -v jq >/dev/null
[[ -f "$LOCAL_NOTEBOOK" ]]
[[ "${#TOKEN}" -le 64 ]]

set +e
table_state >/dev/null
initial_table_state=$?
set -e
if [[ "$initial_table_state" != 1 ]]; then
  if [[ "$initial_table_state" == 0 ]]; then
    printf 'Temporary table already exists: %s\n' "$TEMPORARY_TABLE" >&2
  fi
  exit 1
fi
TABLE_PREFLIGHT_ABSENT=true

ME=$(databricks current-user me -o json | jq -er '.userName')
WORKSPACE_ROOT="/Workspace/Users/$ME/.tmp/madrid-ml-delta-smoke/$TOKEN"
NOTEBOOK_PATH="$WORKSPACE_ROOT/smoke_serverless_delta"

set +e
workspace_state "$WORKSPACE_ROOT"
initial_workspace_state=$?
set -e
if [[ "$initial_workspace_state" != 1 ]]; then
  if [[ "$initial_workspace_state" == 0 ]]; then
    printf 'Workspace staging root already exists: %s\n' "$WORKSPACE_ROOT" >&2
  fi
  exit 1
fi
WORKSPACE_PREFLIGHT_ABSENT=true

NOTEBOOK_SHA256=$(sha256sum "$LOCAL_NOTEBOOK" | cut -d' ' -f1)
log "PREFLIGHT_OK user=$ME token=$TOKEN temporary_table=$TEMPORARY_TABLE notebook_sha256=$NOTEBOOK_SHA256"

databricks workspace mkdirs "$WORKSPACE_ROOT"
databricks workspace import "$NOTEBOOK_PATH" \
  --file "$LOCAL_NOTEBOOK" \
  --format SOURCE \
  --language PYTHON

databricks workspace get-status "$NOTEBOOK_PATH" -o json | jq -e '.object_type == "NOTEBOOK"' >/dev/null

jq -n \
  --arg run_name "$TOKEN" \
  --arg notebook_path "$NOTEBOOK_PATH" \
  --arg temporary_table "$TEMPORARY_TABLE" \
  '{
    run_name: $run_name,
    idempotency_token: $run_name,
    timeout_seconds: 300,
    queue: {enabled: false},
    tasks: [{
      task_key: "serverless_delta_round_trip",
      environment_key: "default",
      timeout_seconds: 300,
      max_retries: 0,
      retry_on_timeout: false,
      disable_auto_optimization: true,
      notebook_task: {
        notebook_path: $notebook_path,
        source: "WORKSPACE",
        base_parameters: {
          temporary_table: $temporary_table,
          ownership_token: $run_name
        }
      }
    }],
    environments: [{
      environment_key: "default",
      spec: {environment_version: "4"}
    }]
  }' >"$PAYLOAD"

jq -e \
  --arg token "$TOKEN" \
  --arg notebook_path "$NOTEBOOK_PATH" \
  --arg temporary_table "$TEMPORARY_TABLE" \
  '(.run_name == $token)
   and (.idempotency_token == $token)
   and (.timeout_seconds == 300)
   and (.queue.enabled == false)
   and (.tasks | length == 1)
   and (.tasks[0].task_key == "serverless_delta_round_trip")
   and (.tasks[0].environment_key == "default")
   and (.tasks[0].timeout_seconds == 300)
   and (.tasks[0].max_retries == 0)
   and (.tasks[0].retry_on_timeout == false)
   and (.tasks[0].disable_auto_optimization == true)
   and (.tasks[0].notebook_task.notebook_path == $notebook_path)
   and (.tasks[0].notebook_task.base_parameters == {
     temporary_table: $temporary_table,
     ownership_token: $token
   })
   and (.environments == [{environment_key: "default", spec: {environment_version: "4"}}])
   and (has("job_clusters") | not)
   and (.tasks[0] | has("new_cluster") | not)
   and (.tasks[0] | has("existing_cluster_id") | not)' \
  "$PAYLOAD" >/dev/null
log "PAYLOAD_OK one_task=true timeout_seconds=300 retries=0 queue=false"

SUBMIT_STARTED_AT_MS=$(date +%s%3N)
resolve_run_id
log "SUBMIT_OK run_id=$RUN_ID"

load_owned_run

while true; do
  load_owned_run
  life_cycle=$(jq -r '.state.life_cycle_state // .status.state // empty' <<<"$RUN_JSON")
  result_state=$(jq -r '.state.result_state // empty' <<<"$RUN_JSON")
  elapsed_ms=$(( $(date +%s%3N) - SUBMIT_STARTED_AT_MS ))
  log "WATCHDOG run_id=$RUN_ID lifecycle=$life_cycle result=${result_state:-none} elapsed_ms=$elapsed_ms"

  if [[ "$life_cycle" == "TERMINATED" \
    || "$life_cycle" == "SKIPPED" \
    || "$life_cycle" == "INTERNAL_ERROR" ]]; then
    break
  fi
  if ((elapsed_ms >= 300000)); then
    databricks jobs cancel-run "$RUN_ID" --no-wait >/dev/null
    wait_for_terminal_after_cancel
    printf 'Micro-smoke exceeded 300 seconds and was cancelled\n' >&2
    exit 1
  fi
  sleep 10
done

if [[ "$life_cycle" != "TERMINATED" || "$result_state" != "SUCCESS" ]]; then
  TASK_RUN_ID=$(jq -r '.tasks[0].run_id // empty' <<<"$RUN_JSON")
  if [[ -n "$TASK_RUN_ID" ]]; then
    databricks jobs get-run-output "$TASK_RUN_ID" -o json >&2 || true
  fi
  printf 'Micro-smoke failed: lifecycle=%s result=%s\n' "$life_cycle" "$result_state" >&2
  exit 1
fi

TASK_RUN_ID=$(jq -er '.tasks[0].run_id' <<<"$RUN_JSON")
task_output=$(databricks jobs get-run-output "$TASK_RUN_ID" -o json)
result_json=$(jq -er '.notebook_output.result' <<<"$task_output")
jq -e \
  --arg temporary_table "$TEMPORARY_TABLE" \
  --arg token "$TOKEN" \
  '(.temporary_table == $temporary_table)
   and (.ownership_token == $token)
   and (.table_type == "MANAGED")
   and (.format == "delta")
   and (.row_count == 3)
   and (.id_sum == 6)
   and (.cleanup_verified == true)' \
  <<<"$result_json" >/dev/null

log "SMOKE_OK run_id=$RUN_ID task_run_id=$TASK_RUN_ID result=$result_json"
