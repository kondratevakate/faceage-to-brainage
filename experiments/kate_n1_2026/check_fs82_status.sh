#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/Lenovo/Documents/Codex/2026-06-11/prior-conversation-with-codex-conversation-role"
DATA="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years"
SD="$DATA/reprocessed_2026/fs82"
LOGD="$DATA/reprocessed_2026/logs_fs82"

echo "== processes =="
ps -ef | grep -E 'run_fs82_local|docker pull|recon-all|freesurfer' | grep -v grep || true

echo
echo "== memory =="
free -h
swapon --show || true

echo
echo "== disk =="
df -h / /mnt/d

echo
echo "== docker =="
sudo -n docker ps --format 'name={{.Names}} image={{.Image}} status={{.Status}}' 2>/dev/null || true
sudo -n docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | grep freesurfer || true

echo
echo "== driver tail =="
tail -40 "$ROOT/work/fs82_local_driver.log" 2>/dev/null || true

echo
echo "== subjects =="
for sid in 2018 2022 kate_fs82_base 2018.long.kate_fs82_base 2022.long.kate_fs82_base 2024_cross_probe; do
  if [ -f "$SD/$sid/scripts/recon-all.done" ] && [ ! -f "$SD/$sid/scripts/recon-all.error" ]; then
    echo "$sid done"
  elif [ -f "$SD/$sid/scripts/recon-all.error" ]; then
    echo "$sid ERROR"
  elif [ -d "$SD/$sid" ]; then
    echo "$sid running-or-incomplete"
  else
    echo "$sid not-started"
  fi
done

echo
echo "== current FS8 logs =="
ls -lh "$LOGD" 2>/dev/null || true
