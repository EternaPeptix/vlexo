#!/bin/zsh
# Manual cluster start — same path as reboot (512S1 coordinator only).
set -euo pipefail
if [[ "$(hostname -s)" != "512S1" ]]; then
  echo "Run restart-cluster-tb.sh on 512S1 only." >&2
  exit 1
fi
exec /Users/jeweled/exo/boot-exo-cluster.sh
