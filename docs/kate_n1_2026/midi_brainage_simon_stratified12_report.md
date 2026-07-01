# MIDIBrainAge SIMON stratified-12 gate

Date: 2026-06-27

## Scope

This run tested the MIDIBrainAge T1 ensemble on 12 stratified SIMON sessions from the visible local FastSurfer derivative branch:

- input manifest: `experiments/kate_n1_2026/midi_brainage_simon_stratified12_inputs.csv`
- predictions: `data/kate_n1_2026/midi_brainage_simon_stratified12_predictions.csv`
- summary: `data/kate_n1_2026/midi_brainage_simon_stratified12_summary.csv`

The input files are existing FastSurfer `orig.mgz` derivatives under `D:\data\fastserfer_simon`. They are not raw SIMON BIDS/NIfTI inputs. Therefore this is a labeled branch sanity gate, not a validation of MIDIBrainAge on its intended raw/documented T1 preprocessing path and not a validation of Kate brain age.

## Command

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
bash experiments/kate_n1_2026/run_midi_brainage_simon_stratified12.sh
```

The wrapper calls `run_midi_brainage_batch.py` with `--return-metrics --resume` and then summarizes the results with `summarize_midi_brainage_results.py`. Per-case MIDIBrainAge work directories were cleaned after each prediction.

Runtime should not be used as a clean performance benchmark for this run. The first part of the run overlapped with local BrainChop CPU jobs, so elapsed seconds are useful only as rough local execution provenance.

## Results

All 12 cases completed successfully. Chronological ages ranged from 29.69 to 46.41 years. Predictions ranged from 27.06 to 39.04 years.

Summary metrics:

- MAE: 8.99 years
- Median absolute error: 7.70 years
- Mean prediction minus chronological age: -7.66 years
- RMSE: 10.37 years
- Pearson r: -0.43
- Predicted-vs-chronological slope: -0.35

Per-case results:

| Scan | Age | Predicted | Delta |
|---|---:|---:|---:|
| ses-001_run-1_T1w | 29.69 | 37.60 | +7.90 |
| ses-008_run-1_T1w | 38.81 | 38.92 | +0.12 |
| ses-014_run-1_T1w | 42.35 | 37.60 | -4.80 |
| ses-021_run-1_T1w | 42.70 | 37.54 | -5.16 |
| ses-027_run-1_T1w | 43.27 | 37.90 | -5.40 |
| ses-034_acq-08iso_run-1_T1w | 43.83 | 30.98 | -12.82 |
| ses-040_run-1_T1w | 44.81 | 39.04 | -5.76 |
| ses-047_run-1_T1w | 45.17 | 32.70 | -12.50 |
| ses-053_run-1_T1w | 45.44 | 37.96 | -7.44 |
| ses-060_run-1_T1w | 46.41 | 27.06 | -19.34 |
| ses-066_run-1_T1w | 46.41 | 30.90 | -15.50 |
| ses-073_run-1_T1w | 46.41 | 35.22 | -11.18 |

## Interpretation

This FastSurfer `orig.mgz` SIMON branch fails the labeled sanity gate for Kate-age reporting. The negative correlation and negative slope are inconsistent with a usable age-calibration branch over the sampled 29.69-46.41 year SIMON range. Predictions also appear compressed and low for older sessions.

This does not prove that MIDIBrainAge is invalid in general. It shows that this local derivative branch is not a safe basis for interpreting the Kate MIDIBrainAge smoke value. A scientifically defensible Kate estimate still requires a model/branch that passes labeled SIMON/SRPBS gates and at least one independent comparator.

## Next

Do not scale SIMON session-first or all-orig FastSurfer-derivative runs for age-claim purposes unless the goal is to characterize this failure mode. The next higher-value steps are:

1. Run MIDIBrainAge on raw/documented T1 inputs where available, preserving the official preprocessing path.
2. Add BrainAgeNeXt as an independent preprocessed T1 comparator.
3. Add SynthBA as a raw/heterogeneous MRI robustness comparator.
4. Keep SRPBS FastSurfer-orig as a separate site/test-retest robustness branch, not as proof of SIMON/Kate age accuracy.
