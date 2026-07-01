#!/usr/bin/env bash
set -euo pipefail

DATA="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years"
LIC="/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/Epilepsy segmentation Kulakov/NEUROML2020/seminar2/license.txt"
IMG="deepmi/fastsurfer:cpu-v2.5.3"

OUT="/data/reprocessed_2026/symmetry/fastsurfer_long_v2"
HOST_OUT="$DATA/reprocessed_2026/symmetry/fastsurfer_long_v2"
LOGD="$DATA/reprocessed_2026/symmetry/logs"
LOG="$LOGD/fastsurfer_long_symmetry_v2.log"
TID="sym_fast_base"
THREADS="${FS_THREADS:-4}"
HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

mkdir -p "$HOST_OUT" "$LOGD"

done_marker() {
  [ -f "$HOST_OUT/$1/scripts/recon-surf.done" ] || \
  grep -q "finished without error" "$HOST_OUT/$1/scripts/recon-surf.log" 2>/dev/null
}

if done_marker "sym_rotpos" && done_marker "sym_rotneg"; then
  echo "[skip] FastSurfer Long symmetry outputs already look complete"
  exit 0
fi

echo "== FastSurfer Long symmetry pair =="
echo "DATA=$DATA"
echo "OUT=$HOST_OUT"
echo "IMG=$IMG"
echo "THREADS=$THREADS"
echo "HOST_UID=$HOST_UID HOST_GID=$HOST_GID"
echo "Started $(date -Is)"

docker image inspect "$IMG" --format 'image={{.RepoTags}} digest={{index .RepoDigests 0}} id={{.Id}} created={{.Created}}' || true

STAGE_ARGS=()
if [ -f "$HOST_OUT/$TID/base-tps.fastsurfer" ] && \
   [ -d "$HOST_OUT/$TID/long-inputs/sym_rotpos" ] && \
   [ -d "$HOST_OUT/$TID/long-inputs/sym_rotneg" ]; then
  echo "[resume] prepare stage exists; continuing with template/long stages"
  STAGE_ARGS=(--stage template_seg --stage template_surf --stage long_seg --stage long_surf)
fi

docker run --rm --user "$HOST_UID:$HOST_GID" \
  -v "$DATA:/data" \
  -v "$LIC:/fs_license/license.txt:ro" \
  --entrypoint /fastsurfer/long_fastsurfer.sh \
  "$IMG" \
  --fs_license /fs_license/license.txt \
  --tid "$TID" \
  --t1s \
    /data/reprocessed_2026/symmetry/nifti/2018_sym_rotpos.nii.gz \
    /data/reprocessed_2026/symmetry/nifti/2018_sym_rotneg.nii.gz \
  --tpids sym_rotpos sym_rotneg \
  --sd "$OUT" \
  "${STAGE_ARGS[@]}" \
  --device cpu \
  --vox_size 1 \
  --3T \
  --threads "$THREADS" \
  --parallel_seg 1 \
  --parallel_surf 1 \
  > "$LOG" 2>&1

echo "Ended $(date -Is)"
echo "ALL DONE"
