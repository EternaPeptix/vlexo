#!/bin/zsh
# 512S2: Exo is started by the 512S1 coordinator (boot-exo-cluster.sh) over TB SSH.
set -euo pipefail
LOG_DIR="$HOME/.exo/exo_log"
mkdir -p "$LOG_DIR"
print -r -- "[$(date "+%Y-%m-%d %H:%M:%S")] $(hostname -s): worker node — exo is started by 512S1 boot-exo-cluster.sh"
exit 0
