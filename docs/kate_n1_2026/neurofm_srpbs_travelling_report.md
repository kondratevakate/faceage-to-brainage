# NeuroFM SRPBS Travelling Branch

Date: 2026-07-09

## Scope

This run used the user-requested NeuroFM repository `https://github.com/rockNroll87q/NeuroFM` at commit `d4e3c463910d939a681d24ebdeb26d44dea6878f`, variant `neurofm-s`.

The input branch is derivative, not raw scanner-native T1w: existing SRPBS travelling `*_orig.mgz` FastSurfer internal source images were multiplied by matching `*_aparcDKT+aseg.mgz > 0` masks, then passed to NeuroFM as skull-stripped NIfTI inputs.

Scientific guard: this is a labelled sanity/domain-risk branch. It is not a validation claim for NeuroFM, Kate brain age, segmentation quality, or morphometry.

## Inputs

- source directory: `D:\data\fastserfer_travelling`
- source manifest: `experiments/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_orig_inputs.csv`
- prepared input status: `data/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_mask_input_status.csv`
- resolved NeuroFM inputs: `data/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_mask_inputs_resolved.csv`
- prepared external NIfTI inputs: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\foundation_models\neurofm_srpbs_travelling_fastsurfer_mask\masked_inputs`

Preparation result: 143/143 rows were usable.

## Weights

- official file: `neurofm-s.h5`
- path: `D:\projects\02_academia\_external\NeuroFM\.cache\neurofm-s.h5`
- SHA256: `8015a0552214b87e43b5462b6c183f8d0da2d957d7ae11ed09a2e3355f5e991f`
- kept outside git

## Results

Artifacts:

- predictions: `data/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_mask_predictions.csv`
- overall summary: `data/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_mask_summary.csv`
- subject summary: `data/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_mask_subject_summary.csv`
- site summary: `data/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_mask_site_summary.csv`
- duplicate input groups: `data/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_mask_duplicate_inputs.csv`
- metadata: `data/kate_n1_2026/neurofm_srpbs_travelling_fastsurfer_mask_metadata.json`

Overall labelled age sanity:

- completed: 143/143
- chronological age range: 24-32 years
- predicted brain age range: 69.3585-89.4440 years
- mean predicted brain age: 81.0564 years
- MAE: 54.0354 years
- bias, predicted minus chronological: +54.0354 years
- RMSE: 54.1854 years
- Pearson r: 0.233402

siteATTd1-only subset, for comparison with the prior MIDIBrainAge SRPBS gate:

- completed: 9/9
- predicted brain age range: 77.4175-87.1339 years
- mean predicted brain age: 83.5747 years
- MAE: 56.5747 years
- Pearson r: 0.431728

Travelling-site spread:

- mean within-subject predicted-age SD: 3.0401 years
- mean within-subject predicted-age range: 9.7526 years
- maximum within-subject predicted-age range: 12.7935 years

Data caveat:

- 53/143 rows fall into duplicate image+label hash groups.
- These duplicates mainly involve `siteATTd1`-`siteATTd5` and `siteATV`, so repeated outputs across those labels should not be interpreted as independent site robustness.

## Interpretation

This branch fails as a brain-age sanity check. The predictions are systematically old for young adult SRPBS travelling participants, with +54 year mean bias. This is consistent with the known domain risk: NeuroFM documents a 40-90 year target range, while SRPBS travelling participants here are 24-32 years old, and the inputs are FastSurfer-derived masked images rather than raw/documented NeuroFM-preprocessed T1w images.

Decision: keep this result as negative-control/domain-preprocessing evidence. Do not use NeuroFM-S derivative outputs to report Kate's brain age.
