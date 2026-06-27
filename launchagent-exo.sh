#!/bin/zsh
set -euo pipefail

EXO_DIR=/Users/jeweled/exo
LOG_DIR="$HOME/.exo/exo_log"
mkdir -p "$LOG_DIR"

log() {
  print -r -- "[$(date "+%Y-%m-%d %H:%M:%S")] $*"
}

if pgrep -f "/Applications/EXO.app" >/dev/null 2>&1; then
  log "Stopping EXO.app (bundled stock MLX)"
  pkill -f "/Applications/EXO.app" || true
  sleep 2
fi

for pid in ${(f)"$(lsof -tiTCP:52415 -sTCP:LISTEN 2>/dev/null || true)"}; do
  cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
  if [[ "$cmd" == *"$EXO_DIR/.venv/bin/exo"* ]] || [[ "$cmd" == *"/Applications/EXO.app"* ]]; then
    log "Stopping prior exo listener pid=$pid"
    kill "$pid" 2>/dev/null || true
  else
    log "Port 52415 held by unexpected pid=$pid: $cmd"
    exit 1
  fi
done
sleep 1

# Let Wi-Fi / Thunderbolt / Netbird come up after login or reboot.
sleep 30

log "Starting exo via start-exo.sh"
cd "$EXO_DIR"
exec ./start-exo.sh
