#!/usr/bin/env bash
set -euo pipefail

DATA="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years"
LIC="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/Epilepsy segmentation Kulakov/NEUROML2020/seminar2/license.txt"
IMG="${RECONANY_IMAGE:-freesurfer/freesurfer:8.2.0}"
THREADS="${RECONANY_THREADS:-2}"

SD="/data/reprocessed_2026/reconany"
HOST_SD="$DATA/reprocessed_2026/reconany"
LOGD="$DATA/reprocessed_2026/logs_reconany"

mkdir -p "$HOST_SD" "$LOGD"

start_docker() {
  if sudo -n docker info >/dev/null 2>&1; then
    return 0
  fi
  if ! pgrep -x dockerd >/dev/null 2>&1; then
    sudo -n nohup dockerd > /tmp/dockerd-reconany.log 2>&1 &
  fi
  for _ in $(seq 1 60); do
    sudo -n docker info >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "[fatal] docker daemon unavailable" >&2
  return 1
}

fs() {
  sudo -n docker run --rm --user root \
    -v "$DATA:/data" \
    -v "$LIC:/fs_license/license.txt:ro" \
    -e FS_LICENSE=/fs_license/license.txt \
    -e SUBJECTS_DIR="$SD" \
    "$IMG" "$@"
}

run_one() {
  local sid="$1"
  local input="$2"

  if [ -f "$HOST_SD/$sid/scripts/recon-all.done" ] && [ ! -f "$HOST_SD/$sid/scripts/recon-all.error" ]; then
    echo "[skip] $sid already done"
    return 0
  fi

  echo "[start] recon-any $sid input=$input threads=$THREADS"
  fs run_recon-any "$input" "$sid" "$THREADS" both "$SD" > "$LOGD/reconany_${sid}.log" 2>&1
  echo "[done] recon-any $sid"
}

echo "== ReconAny local WSL run =="
echo "IMG=$IMG THREADS=$THREADS"
echo "SUBJECTS_DIR=$HOST_SD"
echo "LOGD=$LOGD"

start_docker
sudo -n docker pull "$IMG"

if ! fs bash -lc 'command -v run_recon-any >/dev/null 2>&1'; then
  echo "[fatal] run_recon-any is not available in image $IMG."
  echo "Use a FreeSurfer dev image/build that includes ReconAny, then set RECONANY_IMAGE."
  exit 3
fi

run_one 2018_reconany /data/images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz
run_one 2022_reconany /data/images/2022/nifti/4_t1_se_sag.nii.gz
run_one 2024_3di_reconany /data/images/2024/nifti/901_3di_mc_hr.nii.gz

echo "ALL DONE: $(date -Is)"
