#!/bin/zsh
# Manually load the configured auto-start model on a running cluster.
set -euo pipefail
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

EXO_DIR="${EXO_DIR:-$HOME/exo}"
AUTO_MODEL_ENV="${EXO_DIR}/auto-model.env"
[[ -f "$AUTO_MODEL_ENV" ]] && source "$AUTO_MODEL_ENV"
EXO_AUTO_LOAD_MODEL="${EXO_AUTO_LOAD_MODEL:-1}"
EXO_AUTO_MODEL_ID="${EXO_AUTO_MODEL_ID:-pipenetwork/GLM-5.2-MLX-8bit}"
EXO_AUTO_MODEL_SHARDING="${EXO_AUTO_MODEL_SHARDING:-Tensor}"
EXO_AUTO_MODEL_META="${EXO_AUTO_MODEL_META:-MlxJaccl}"
EXO_AUTO_MODEL_MIN_NODES="${EXO_AUTO_MODEL_MIN_NODES:-2}"

log() { print -r -- "[$(date "+%Y-%m-%d %H:%M:%S")] $*" >&2 }

wait_for_api() {
  local url="$1" tries="${2:-30}" i
  for i in $(seq 1 "$tries"); do
    curl -sf --max-time 3 "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

model_already_loaded() {
  local model_id="$1"
  curl -sf http://127.0.0.1:52415/state | EXO_CHECK_MODEL_ID="$model_id" python3 -c '
import json, os, sys
model_id = os.environ["EXO_CHECK_MODEL_ID"]
state = json.load(sys.stdin)
for inst in state.get("instances", {}).values():
    for body in inst.values():
        sa = body.get("shardAssignments") or body.get("shard_assignments") or {}
        mid = sa.get("modelId") or sa.get("model_id") or ""
        if mid == model_id:
            sys.exit(0)
sys.exit(1)
' 2>/dev/null
}

if [[ "$EXO_AUTO_LOAD_MODEL" != "1" ]]; then
  log "Auto model load disabled in $AUTO_MODEL_ENV"
  exit 0
fi

model_id="$EXO_AUTO_MODEL_ID"
wait_for_api "http://127.0.0.1:52415/state" 30 || { log "API not up"; exit 1 }

if model_already_loaded "$model_id"; then
  log "Already loaded: $model_id"
  exit 0
fi

log "Placing $model_id..."
curl -sf -X POST http://127.0.0.1:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\":\"${model_id}\",\"sharding\":\"${EXO_AUTO_MODEL_SHARDING}\",\"instance_meta\":\"${EXO_AUTO_MODEL_META}\",\"min_nodes\":${EXO_AUTO_MODEL_MIN_NODES}}"
log "place_instance sent for $model_id"
