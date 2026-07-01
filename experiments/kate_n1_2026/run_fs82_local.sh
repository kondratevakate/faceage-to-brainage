#!/usr/bin/env bash
set -euo pipefail

DATA="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years"
LIC="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/Epilepsy segmentation Kulakov/NEUROML2020/seminar2/license.txt"
IMG="${FS82_IMAGE:-freesurfer/freesurfer:8.2.0}"

SD="/data/reprocessed_2026/fs82"
HOST_SD="$DATA/reprocessed_2026/fs82"
LOGD="$DATA/reprocessed_2026/logs_fs82"
BASE_ID="${FS82_BASE_ID:-kate_fs82_base}"
THREADS="${FS82_THREADS:-2}"
RUN_2024_PROBE="${FS82_RUN_2024_PROBE:-1}"

mkdir -p "$HOST_SD" "$LOGD"

start_docker() {
  if sudo -n docker info >/dev/null 2>&1; then
    return 0
  fi

  if ! pgrep -x dockerd >/dev/null 2>&1; then
    echo "[docker] starting dockerd"
    sudo -n nohup dockerd > /tmp/dockerd-fs82.log 2>&1 &
  fi

  for _ in $(seq 1 60); do
    if sudo -n docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  echo "[docker] failed to start; see /tmp/dockerd-fs82.log" >&2
  return 1
}

start_monitor() {
  (
    while true; do
      echo "===== $(date -Is) ====="
      free -h || true
      df -h / /mnt/d || true
      sudo -n docker ps --format 'container={{.Names}} image={{.Image}} status={{.Status}}' || true
      sudo -n docker stats --no-stream --format 'container={{.Name}} mem={{.MemUsage}} cpu={{.CPUPerc}}' || true
      sleep 60
    done
  ) >> "$LOGD/fs82_resource_monitor.log" 2>&1 &
  MONITOR_PID=$!
  trap 'kill "$MONITOR_PID" >/dev/null 2>&1 || true' EXIT
}

fs() {
  sudo -n docker run --rm --user root \
    -v "$DATA:/data" \
    -v "$LIC:/fs_license/license.txt:ro" \
    -e FS_LICENSE=/fs_license/license.txt \
    -e SUBJECTS_DIR="$SD" \
    "$IMG" "$@"
}

done_marker() {
  [ -f "$HOST_SD/$1/scripts/recon-all.done" ] && [ ! -f "$HOST_SD/$1/scripts/recon-all.error" ]
}

long_done_marker() {
  [ -f "$HOST_SD/$1.long.$BASE_ID/scripts/recon-all.done" ] && [ ! -f "$HOST_SD/$1.long.$BASE_ID/scripts/recon-all.error" ]
}

run_cross_if_needed() {
  local sid="$1"
  local input="$2"
  if done_marker "$sid"; then
    echo "[skip] cross $sid already done"
    return 0
  fi

  echo "[start] cross $sid input=$input threads=$THREADS"
  fs recon-all -all -s "$sid" -i "$input" -threads "$THREADS" \
    > "$LOGD/fs82_cross_${sid}.log" 2>&1
  done_marker "$sid"
  echo "[done] cross $sid"
}

backup_incomplete_base() {
  local base_dir="$HOST_SD/$BASE_ID"
  if [ ! -d "$base_dir" ]; then
    return 0
  fi
  if done_marker "$BASE_ID"; then
    echo "[skip] base already done"
    return 0
  fi

  local backup="$HOST_SD/${BASE_ID}_broken_$(date +%Y%m%d_%H%M%S)"
  echo "[backup] moving incomplete $BASE_ID to $backup"
  mv "$base_dir" "$backup"
}

run_base_if_needed() {
  if done_marker "$BASE_ID"; then
    echo "[skip] base already done"
    return 0
  fi

  echo "[start] base $BASE_ID from 2018 2022 threads=$THREADS"
  fs recon-all -base "$BASE_ID" -tp 2018 -tp 2022 -all -threads "$THREADS" \
    > "$LOGD/fs82_base_2018_2022.log" 2>&1
  done_marker "$BASE_ID"
  echo "[done] base $BASE_ID"
}

run_long_one() {
  local sid="$1"
  if long_done_marker "$sid"; then
    echo "[skip] long $sid already done"
    return 0
  fi

  echo "[start] long $sid base=$BASE_ID threads=$THREADS"
  fs recon-all -long "$sid" "$BASE_ID" -all -threads "$THREADS" \
    > "$LOGD/fs82_long_${sid}.log" 2>&1
  long_done_marker "$sid"
  echo "[done] long $sid"
}

echo "== FreeSurfer 8.2 local WSL run =="
echo "DATA=$DATA"
echo "SUBJECTS_DIR=$HOST_SD"
echo "LOGD=$LOGD"
echo "IMG=$IMG THREADS=$THREADS RUN_2024_PROBE=$RUN_2024_PROBE"
echo "Started: $(date -Is)"

if [ ! -f "$LIC" ]; then
  echo "[fatal] missing FreeSurfer license: $LIC" >&2
  exit 2
fi

start_docker
start_monitor

echo "[docker] pulling/checking $IMG"
sudo -n docker pull "$IMG"
sudo -n docker image inspect "$IMG" > "$LOGD/fs82_docker_image_inspect.json"
fs recon-all -version > "$LOGD/fs82_recon_all_version.txt" 2>&1 || true

run_cross_if_needed 2018 /data/images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz
run_cross_if_needed 2022 /data/images/2022/nifti/4_t1_se_sag.nii.gz

backup_incomplete_base
run_base_if_needed

run_long_one 2018
run_long_one 2022

if [ "$RUN_2024_PROBE" = "1" ]; then
  run_cross_if_needed 2024_cross_probe /data/images/2024/nifti/901_3di_mc_hr.nii.gz
else
  echo "[skip] 2024 cross probe disabled"
fi

echo "ALL DONE: $(date -Is)"
