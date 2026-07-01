#!/usr/bin/env bash
set -euo pipefail

DATA="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years"
LIC="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/Epilepsy segmentation Kulakov/NEUROML2020/seminar2/license.txt"
IMG="freesurfer/freesurfer:7.4.1"

SD="/data/reprocessed_2026/fs_long"
HOST_SD="$DATA/reprocessed_2026/fs_long"
LOGD="$DATA/reprocessed_2026/logs"
BASE_ID="${FS_BASE_ID:-kate_base}"
THREADS="${FS_THREADS:-4}"
PAR="${FS_PARALLEL:-1}"

mkdir -p "$HOST_SD" "$LOGD"

fs() {
  docker run --rm --user root \
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

  echo "[start] cross $sid"
  fs recon-all -all -s "$sid" -i "$input" -threads "$THREADS" \
    > "$LOGD/fs_cross_${sid}.log" 2>&1
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

  echo "[start] base $BASE_ID from 2018 2022"
  fs recon-all -base "$BASE_ID" -tp 2018 -tp 2022 -all -threads "$THREADS" \
    > "$LOGD/fs_base_2018_2022.log" 2>&1
  done_marker "$BASE_ID"
  echo "[done] base $BASE_ID"
}

run_long_one() {
  local sid="$1"
  if long_done_marker "$sid"; then
    echo "[skip] long $sid already done"
    return 0
  fi

  echo "[start] long $sid"
  fs recon-all -long "$sid" "$BASE_ID" -all -threads "$THREADS" \
    > "$LOGD/fs_long_${sid}.log" 2>&1
  long_done_marker "$sid"
  echo "[done] long $sid"
}

throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do
    sleep 10
  done
}

echo "== FreeSurfer longitudinal 2018+2022 =="
echo "DATA=$DATA"
echo "THREADS=$THREADS PAR=$PAR"

run_cross_if_needed 2018 /data/images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz
run_cross_if_needed 2022 /data/images/2022/nifti/4_t1_se_sag.nii.gz

backup_incomplete_base
run_base_if_needed

for sid in 2018 2022; do
  throttle
  run_long_one "$sid" &
done
wait

echo "ALL DONE"
