# NeuroFM SIMON HD-BET Application Branch

Date: 2026-07-10

## Scope

This run tested the user-requested NeuroFM repository (`https://github.com/rockNroll87q/NeuroFM.git`, commit `d4e3c463910d939a681d24ebdeb26d44dea6878f`) on all 94 visible SIMON FastSurfer `orig.mgz` source-derivative inputs after independent HD-BET skull stripping.

This is an application/QC branch and labeled sanity check. It is not a validation claim, not a clinical brain-age estimate, and not evidence that NeuroFM outputs validate segmentation or morphometry.

## Preprocessing

Input manifest: `experiments/kate_n1_2026/midi_brainage_simon_all_orig_inputs.csv`

Preprocessing command path: `experiments/kate_n1_2026/prepare_neurofm_simon_hdbet_inputs.py`

Output manifest: `data/kate_n1_2026/neurofm_simon_hdbet_inputs_resolved.csv`

Status CSV: `data/kate_n1_2026/neurofm_simon_hdbet_preprocessing_status.csv`

External heavy outputs were written under:

`D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\foundation_models\neurofm_simon_hdbet`

HD-BET completed 94/94 image and mask pairs. The branch uses `orig.mgz -> NIfTI -> HD-BET skull-strip`, with CPU and `--disable_tta`. Sixteen HD-BET outputs still had non-1 mm voxel sizes, so NeuroFM internally resampled them.

## NeuroFM Result

Main outputs:

- `data/kate_n1_2026/neurofm_simon_hdbet_predictions.csv`
- `data/kate_n1_2026/neurofm_simon_hdbet_summary.csv`
- `data/kate_n1_2026/neurofm_simon_hdbet_metadata.json`
- `data/kate_n1_2026/neurofm_simon_hdbet_qc_summary.csv`

Summary for 94/94 SIMON rows:

| Branch | n | Mean predicted age | MAE | Bias | RMSE | Pearson r |
|---|---:|---:|---:|---:|---:|---:|
| NeuroFM HD-BET all-94 | 94 | 58.44 | 14.95 | +14.95 | 15.72 | 0.314 |
| NeuroFM raw-orig all-94 | 94 | 77.56 | 34.06 | +34.06 | 34.94 | 0.143 |
| NeuroFM FastSurfer-mask | 87 | 80.99 | 37.15 | +37.15 | 37.44 | 0.058 |
| MIDIBrainAge all-orig | 94 | 34.92 | 9.11 | -8.58 | 9.96 | -0.035 |

HD-BET skull stripping materially reduced the NeuroFM age offset compared with the raw-orig and FastSurfer-mask NeuroFM branches, but the result remains too biased for a Kate brain-age claim.

## Resampling And Brain-Size QC

From `data/kate_n1_2026/neurofm_simon_hdbet_qc_summary.csv`:

| Group | n | MAE | Bias | Pearson r | Mean predicted age |
|---|---:|---:|---:|---:|---:|
| all_hdbet | 94 | 14.95 | +14.95 | 0.314 | 58.44 |
| hdbet_1mm_inputs | 78 | 14.06 | +14.06 | 0.467 | 57.90 |
| hdbet_non_1mm_inputs | 16 | 19.28 | +19.28 | 0.298 | 61.09 |

The non-1 mm subgroup was worse and older on average. This supports treating internal resampling as a robustness concern, not as a solved preprocessing detail.

For all HD-BET rows, the correlation between NeuroFM predicted age and NeuroFM predicted brain volume was weak (`r=0.119`), and the correlation with HD-BET mask fraction was also weak (`r=0.145`). This does not prove independence from brain size; it only says that this coarse proxy did not explain most prediction variance in this branch.

## Perturbation Stability QC

Perturbation outputs:

- `data/kate_n1_2026/simon_stability_perturbation_inputs.csv`
- `data/kate_n1_2026/neurofm_simon_stability_perturbation_predictions.csv`
- `data/kate_n1_2026/neurofm_simon_stability_perturbation_deltas.csv`
- `data/kate_n1_2026/neurofm_simon_stability_perturbation_delta_summary.csv`

This perturbation run used 12 SIMON FastSurfer-mask skull-stripped inputs and synthetic perturbations. It is a robustness probe only; the underlying branch is already invalid for age claims.

| Perturbation family | n | Mean abs delta | Median abs delta | P90 abs delta | Max abs delta | Sex flips |
|---|---:|---:|---:|---:|---:|---:|
| brain_size | 24 | 1.85 | 1.87 | 3.75 | 5.19 | 3 |
| resample_roundtrip | 24 | 1.84 | 1.89 | 2.37 | 2.59 | 0 |
| rotation | 24 | 1.44 | 1.43 | 2.21 | 3.70 | 2 |

Small perturbations can move NeuroFM-S brain-age outputs by approximately 1-2 years on average, with larger outliers and occasional sex-class flips. This is additional QC evidence against treating a single NeuroFM output as a stable individual age estimate.

## SynthStrip Attempt

I also tested a separate SynthStrip route because a normal skull-strip should not depend only on existing FastSurfer labels. Docker/Singularity were unavailable locally. `nipreps-synthstrip` was installed in an isolated venv and the official FreeSurfer `synthstrip.1.pt` weight was downloaded outside git, but the one-scan CPU smoke did not complete within 15 minutes and was killed. No SynthStrip output was used for the NeuroFM result.

## Plot

Updated comparison plot:

- `docs/kate_n1_2026/figures/simon_age_predictions_by_model.svg`
- `docs/kate_n1_2026/figures/simon_age_predictions_by_model.png`
- `data/kate_n1_2026/simon_age_predictions_by_model_long.csv`

The plot now includes `NeuroFM HD-BET all-94` as the corrected skull-stripped comparator.

## Interpretation

HD-BET all-94 is the best NeuroFM SIMON branch run so far, but it still fails the sanity threshold for a calibrated brain-age claim on SIMON. It should remain in the foundation-model/QC evidence ledger. Kate's NeuroFM output should not be reported as biological brain age unless an independent adult-calibrated model and a preprocessing-matched labeled sanity set support it.
