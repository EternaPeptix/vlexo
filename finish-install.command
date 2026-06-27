#!/bin/zsh
cd "$HOME/exo"
./install-launchagent.sh
osascript -e 'display dialog "Exo TB/RDMA auto-start installed on 512S1. Cluster will bootstrap on next login/reboot." buttons {"OK"} default button 1 with title "Exo LaunchAgent"'
read -k '?Press any key to close...'
