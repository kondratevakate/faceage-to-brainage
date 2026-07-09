# NeuroFM Application Branch Status

Date: 2026-07-09

## Scope

This branch uses the user-requested repository `https://github.com/rockNroll87q/NeuroFM`, not the earlier `peirong26/BrainFM` feature-only branch. The local clone is `D:\projects\02_academia\_external\NeuroFM` at commit `d4e3c463910d939a681d24ebdeb26d44dea6878f` (`v1.0.3`).

Scientific guard: NeuroFM brain-health outputs are model outputs for a foundation-model application/QC branch. They do not prove segmentation quality, morphometric validity, individual clinical brain health, or Kate's biological brain age.

## Weights

Official `neurofm-s.h5` was downloaded from `https://huggingface.co/NeuroAI-UofG/NeuroFM` after gated access acceptance.

- local path: `D:\projects\02_academia\_external\NeuroFM\.cache\neurofm-s.h5`
- WSL path: `/mnt/d/projects/02_academia/_external/NeuroFM/.cache/neurofm-s.h5`
- size: `2055816` bytes
- SHA256: `8015a0552214b87e43b5462b6c183f8d0da2d957d7ae11ed09a2e3355f5e991f`
- git policy: weights remain outside git

## Kate HD-BET Run

Inputs were skull-stripped with HD-BET 2.0.1 CPU, `--disable_tta`, and run through NeuroFM-S on CPU with `--output-mode summary` and `outputs=brain_health`.

Artifacts:

- manifest: `experiments/kate_n1_2026/neurofm_kate_hdbet_inputs.csv`
- preprocessing status: `data/kate_n1_2026/neurofm_kate_hdbet_preprocessing_status.csv`
- predictions: `data/kate_n1_2026/neurofm_kate_hdbet_predictions.csv`
- summary: `data/kate_n1_2026/neurofm_kate_hdbet_summary.csv`
- metadata: `data/kate_n1_2026/neurofm_kate_hdbet_metadata.json`
- external NeuroFM summary: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\foundation_models\neurofm_kate_hdbet\results\results_summary.csv`

Results:

| scan_id | predicted_brain_age_years | predicted_sex_binary |
|---|---:|---:|
| `kate_2018_t1_hdbet` | 52.5202 | 0.0 |
| `kate_2022_t1_hdbet` | 64.5316 | 0.0 |
| `kate_2024_3di_hdbet` | 51.4625 | 1.0 |
| `kate_2024_t1_ffe_401_hdbet` | 45.9739 | 0.0 |
| `kate_2024_t1_ffe_601_hdbet` | 52.3006 | 0.0 |

Summary: 5/5 completed; mean predicted brain age `53.3578`, min `45.9739`, max `64.5316`. No chronological-age field is available in this manifest, so no accuracy metric is computed for Kate. The predicted-sex inconsistency across Kate protocols, especially the 2024 3DI branch differing from the other four inputs, is a QC/protocol-sensitivity signal.

## SIMON FastSurfer-Mask Run

SIMON was run as a derivative sanity/domain-risk branch, not as raw validation. Inputs are existing `*_orig.mgz` internal source images multiplied by matching `*_aparcDKT+aseg.mgz > 0` masks where available.

Artifacts:

- preprocessing status: `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_input_status.csv`
- resolved inputs: `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_inputs_resolved.csv`
- predictions: `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_predictions.csv`
- summary: `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_summary.csv`
- metadata: `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_metadata.json`
- external NeuroFM summary: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\foundation_models\neurofm_simon_fastsurfer_mask\results\results_summary.csv`

Input preparation: 87/94 visible SIMON rows were usable. Seven rows lacked matching `aparcDKT+aseg.mgz` label maps.

Results against chronological ages:

- completed: 87/87
- chronological age range: 29.69-46.41 years
- predicted brain age range: 68.8255-87.1781 years
- mean predicted brain age: 80.9927 years
- MAE: 37.1518 years
- bias, predicted minus chronological: +37.1518 years
- RMSE: 37.4354 years
- Pearson r: 0.057847
- predicted sex counts: 83 rows `0.0`, 4 rows `1.0`

Interpretation: this is a strong negative sanity check for using NeuroFM-S as a brain-age estimator on the SIMON FastSurfer-mask derivative branch. It does not support reporting Kate's brain age from NeuroFM.

## Critical Limitations

1. NeuroFM documentation/model card is oriented to skull-stripped T1w input and a 40-90 year population. Much of SIMON, and likely Kate depending on date, is outside or near the lower edge of that range.
2. Kate inputs are heterogeneous across acquisition/protocol, and NeuroFM logged internal resampling warnings for the Kate HD-BET inputs.
3. SIMON is not raw scanner-native validation here; it is a FastSurfer internal-source plus label-mask derivative branch.
4. NeuroFM `brain_health` outputs are model estimates. They are not segmentation outputs, morphometric validation, or clinical evidence.

## Decision

Keep NeuroFM as a foundation-model application/QC branch only. Do not use the Kate NeuroFM numbers as a biological age claim. For a brain-age claim, require an adult-calibrated model with documented preprocessing and a labeled sanity branch that passes before applying it to Kate.
