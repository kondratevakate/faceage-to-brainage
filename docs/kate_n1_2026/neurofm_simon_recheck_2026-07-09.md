# NeuroFM SIMON Recheck

Date: 2026-07-09

## Scope

Repository: `D:\projects\02_academia\_external\NeuroFM`, remote `https://github.com/rockNroll87q/NeuroFM.git`, commit `d4e3c463910d939a681d24ebdeb26d44dea6878f` (`v1.0.3`).

Weights: official `neurofm-s.h5` from `https://huggingface.co/NeuroAI-UofG/NeuroFM`, kept outside git at `D:\projects\02_academia\_external\NeuroFM\.cache\neurofm-s.h5`, SHA256 `8015a0552214b87e43b5462b6c183f8d0da2d957d7ae11ed09a2e3355f5e991f`.

Scientific guard: NeuroFM outputs here are application/QC/domain-risk outputs. They do not prove segmentation quality, morphometric validity, clinical brain health, or Kate's biological brain age.

## Masked Branch Recheck

The existing SIMON FastSurfer-mask branch was rerun from the resolved 87-row manifest without regenerating inputs.

Artifacts:

- `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_inputs_resolved.csv`
- `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_predictions.csv`
- `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_summary.csv`
- `data/kate_n1_2026/neurofm_simon_fastsurfer_mask_metadata.json`

Recheck result: predictions, compact summary, and external NeuroFM `results_summary.csv` were bitwise identical to the previous committed run.

Hashes after rerun:

- predictions SHA256 `35ad35cf283a3207abd507696f9a26bddc8c38cae81824e19fd870744c495277`
- summary SHA256 `951d685fb4fe8aa725a05153ff54a17b46681ecd37bbc73b5c0fd264c3217130`
- external `results_summary.csv` SHA256 `b8683f8bde1f5cb79a2dffe416837e954c077474890b42238cb8abdc121c41e7`

Summary remained unchanged: 87/87 predictions; chronological age range 29.69-46.41 years; mean predicted brain age 80.9927 years; MAE 37.1518 years; bias +37.1518 years; RMSE 37.4354 years; Pearson r 0.057847.

## Raw-Orig All-94 Branch

The source folder `D:\data\fastserfer_simon` contains 94 `*_orig.mgz` files but only 87 matching `*_aparcDKT+aseg.mgz` label maps. The seven missing label-map scans are:

- `ses-004_run-1_T1w`
- `ses-005_run-1_T1w`
- `ses-005_run-2_T1w`
- `ses-005_run-3_T1w`
- `ses-008_run-1_T1w`
- `ses-029_acq-T1Cube_run-1_T1w`
- `ses-029_acq-dirWMCube_run-1_T1w`

To cover all visible SIMON orig scans, a separate non-skull-stripped diagnostic branch converted 94 FastSurfer `_orig.mgz` files to NIfTI and ran NeuroFM-S with `--output-mode summary`.

Artifacts:

- `experiments/kate_n1_2026/prepare_neurofm_orig_nifti_inputs.py`
- `experiments/kate_n1_2026/run_neurofm_simon_raw_orig.sh`
- `data/kate_n1_2026/neurofm_simon_raw_orig_input_status.csv`
- `data/kate_n1_2026/neurofm_simon_raw_orig_inputs_resolved.csv`
- `data/kate_n1_2026/neurofm_simon_raw_orig_predictions.csv`
- `data/kate_n1_2026/neurofm_simon_raw_orig_summary.csv`
- `data/kate_n1_2026/neurofm_simon_raw_orig_metadata.json`

Raw-orig result: 94/94 inputs prepared and 94/94 NeuroFM predictions completed.

Summary:

- chronological age range: 29.69-46.41 years
- predicted brain age range: 50.1406-91.7619 years
- mean predicted brain age: 77.5551 years
- MAE: 34.0564 years
- bias, predicted minus chronological: +34.0564 years
- RMSE: 34.9448 years
- Pearson r: 0.143206
- predicted sex counts: 38 rows `0.0`, 56 rows `1.0`

Input QC: 16/94 raw-orig NIfTI inputs had voxel sizes other than 1x1x1 mm and triggered or required NeuroFM internal resampling.

## Interpretation

The all-orig raw branch covers 94/94 visible SIMON orig images, but it is non-skull-stripped and therefore does not match NeuroFM's recommended skull-stripped T1w input expectation. It is useful as a stress/QC branch only.

Both SIMON branches remain negative sanity checks for using NeuroFM-S as a calibrated brain-age estimator here. The masked branch is reproducible, and the raw all-orig branch does not rescue the age signal. NeuroFM should remain an application/QC branch unless a documented preprocessing path and independent age-head models pass labeled sanity checks.
