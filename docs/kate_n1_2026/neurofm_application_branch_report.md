# NeuroFM Application Branch Status

Date: 2026-07-06

## Scope

This branch uses the user-requested repository `https://github.com/rockNroll87q/NeuroFM`, not the earlier `peirong26/BrainFM` feature-only branch. The local clone is `D:\projects\02_academia\_external\NeuroFM` at commit `d4e3c463910d939a681d24ebdeb26d44dea6878f` (`v1.0.3`).

Scientific guard: NeuroFM brain-health outputs are model outputs for a foundation-model application branch. They do not prove segmentation quality, morphometric validity, or individual clinical brain health.

## Facts

- NeuroFM code was installed in an isolated WSL environment: `/home/kate/.venvs/neurofm_py311`.
- TensorFlow 2.13.0 and NeuroFM 1.0.3 are installed there.
- The upstream docs/model card require skull-stripped T1w NIfTI input; NeuroFM internally conforms resolution/orientation.
- The model card states that the model is not suitable for skull-on inputs or populations substantially outside the 40-90 year range.
- Kate 2018 primary T1 was skull-stripped with HD-BET 2.0.1 CPU, `--disable_tta`, producing:
  - `/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/foundation_models/neurofm_hdbet_inputs/kate_2018_t1_hdbet.nii.gz`
  - status: `data/kate_n1_2026/neurofm_kate_hdbet_preprocessing_status.csv`
- NeuroFM inference did not produce a brain-age estimate because the official `NeuroAI-UofG/NeuroFM` Hugging Face weights are gated. The unauthenticated `neurofm-s.h5` download returned 401.
- No fake checkpoint or substitute weight was created.
- SIMON FastSurfer-mask preprocessing was prepared as a future labeled sanity branch:
  - source: 94 visible `*_orig.mgz` internal source images under `D:\data\fastserfer_simon`
  - mask source: matching `*_aparcDKT+aseg.mgz` label maps where available
  - result: 87/94 NeuroFM-ready masked NIfTI inputs
  - status: `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_input_status.csv`
  - resolved inputs: `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_inputs_resolved.csv`

## Blockers

1. Official weights require accepting Hugging Face model conditions/authentication.
2. The previous BrainChop-mask input branch is unavailable in the current data root because the referenced `tissue_fast` NIfTI files are missing.
3. The SIMON FastSurfer-mask branch is partial: 7/94 rows lack matching `aparcDKT+aseg.mgz` labels.
4. Even after weights are available, Kate/SIMON ages below 40 would be outside NeuroFM's documented age range, so results must be treated as application/QC output until calibrated against labeled data.

## Next Command

After accepting access and placing official `neurofm-s.h5` outside git:

```bash
cd /mnt/d/projects/02_academia/faceage-to-brainage
export NEUROFM_WEIGHTS=/mnt/d/projects/02_academia/_external/NeuroFM/weights_cache/neurofm-s.h5
export PREPARE_MASKED_INPUTS=0
export SOURCE_MANIFEST=/mnt/d/projects/02_academia/faceage-to-brainage/experiments/kate_n1_2026/neurofm_kate_hdbet_inputs.csv
export PREDICTIONS_CSV=/mnt/d/projects/02_academia/faceage-to-brainage/data/kate_n1_2026/neurofm_kate_hdbet_predictions.csv
export SUMMARY_CSV=/mnt/d/projects/02_academia/faceage-to-brainage/data/kate_n1_2026/neurofm_kate_hdbet_summary.csv
export METADATA_JSON=/mnt/d/projects/02_academia/faceage-to-brainage/data/kate_n1_2026/neurofm_kate_hdbet_metadata.json
bash experiments/kate_n1_2026/run_neurofm_local.sh
```

Do not commit raw MRI, HD-BET NIfTI outputs, NeuroFM weights, logs, or caches.

For the prepared SIMON branch after weights are available:

```bash
cd /mnt/d/projects/02_academia/faceage-to-brainage
export NEUROFM_WEIGHTS=/mnt/d/projects/02_academia/_external/NeuroFM/weights_cache/neurofm-s.h5
export PREPARE_MASKED_INPUTS=0
export SOURCE_MANIFEST=/mnt/d/projects/02_academia/faceage-to-brainage/data/kate_n1_2026/neurofm_simon_fastsurfer_mask_inputs_resolved.csv
export PREDICTIONS_CSV=/mnt/d/projects/02_academia/faceage-to-brainage/data/kate_n1_2026/neurofm_simon_fastsurfer_mask_predictions.csv
export SUMMARY_CSV=/mnt/d/projects/02_academia/faceage-to-brainage/data/kate_n1_2026/neurofm_simon_fastsurfer_mask_summary.csv
export METADATA_JSON=/mnt/d/projects/02_academia/faceage-to-brainage/data/kate_n1_2026/neurofm_simon_fastsurfer_mask_metadata.json
bash experiments/kate_n1_2026/run_neurofm_local.sh
```
