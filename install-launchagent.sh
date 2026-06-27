#!/bin/zsh
# Install Exo auto-start for TB/RDMA cluster (run on each Studio; S1 uses system LaunchDaemon).
set -euo pipefail

EXO_DIR="${EXO_DIR:-$HOME/exo}"
USER_ID=$(id -u)
GUI_DOMAIN="gui/$USER_ID"
LABEL=com.jeweled.exo
HOST="$(hostname -s)"
DAEMON_PLIST="/Library/LaunchDaemons/${LABEL}.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/.exo/exo_log"

install_coordinator() {
  launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
  if [[ -f "$EXO_DIR/com.jeweled.exo.daemon.plist" ]]; then
    echo "Installing system LaunchDaemon (boots without GUI login)..."
    sudo cp "$EXO_DIR/com.jeweled.exo.daemon.plist" "$DAEMON_PLIST"
    sudo chown root:wheel "$DAEMON_PLIST"
    sudo chmod 644 "$DAEMON_PLIST"
    plutil -lint "$DAEMON_PLIST"
    sudo launchctl bootout system/"$LABEL" 2>/dev/null || true
    sudo launchctl bootstrap system "$DAEMON_PLIST"
    sudo launchctl enable system/"$LABEL"
    sudo launchctl kickstart -k system/"$LABEL" || true
    echo "512S1 LaunchDaemon installed."
    echo "Logs: ~/.exo/exo_log/launchdaemon.{stdout,stderr}.log"
    echo "Check: sudo launchctl print system/$LABEL"
    return 0
  fi
  local plist="$HOME/Library/LaunchAgents/com.jeweled.exo.plist"
  cp "$EXO_DIR/com.jeweled.exo.plist" "$plist"
  chmod 644 "$plist"
  plutil -lint "$plist"
  launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
  launchctl bootstrap "$GUI_DOMAIN" "$plist"
  launchctl enable "$GUI_DOMAIN/$LABEL"
  launchctl kickstart -k "$GUI_DOMAIN/$LABEL" || true
  echo "512S1 LaunchAgent installed (requires GUI login at boot)."
  echo "Logs: ~/.exo/exo_log/launchagent.{stdout,stderr}.log"
}

disable_worker() {
  launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
  if [[ -f "$HOME/Library/LaunchAgents/com.jeweled.exo.plist" ]]; then
    cp "$EXO_DIR/com.jeweled.exo.worker.plist" "$HOME/Library/LaunchAgents/com.jeweled.exo.plist"
    chmod 644 "$HOME/Library/LaunchAgents/com.jeweled.exo.plist"
  fi
  echo "512S2: auto-start disabled (exo is started by 512S1 over Thunderbolt)."
}

for f in boot-exo-cluster.sh launchagent-exo-worker.sh com.jeweled.exo.plist com.jeweled.exo.worker.plist; do
  if [[ ! -f "$EXO_DIR/$f" ]]; then
    echo "Missing $EXO_DIR/$f" >&2
    exit 1
  fi
done
chmod +x "$EXO_DIR/boot-exo-cluster.sh" "$EXO_DIR/launchagent-exo-worker.sh"

case "$HOST" in
  512S1)
    install_coordinator
    ;;
  512S2)
    disable_worker
    ;;
  *)
    echo "Unknown host $HOST — install coordinator only on 512S1, disable on 512S2." >&2
    exit 1
    ;;
esac
