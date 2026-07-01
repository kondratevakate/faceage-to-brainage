# MIDIBrainAge SIMON all-orig branch characterization

Date: 2026-07-01

## Scope

This run processed all 94 locally visible SIMON FastSurfer `orig.mgz` inputs:

- input manifest: `experiments/kate_n1_2026/midi_brainage_simon_all_orig_inputs.csv`
- predictions: `data/kate_n1_2026/midi_brainage_simon_all_orig_predictions.csv`
- overall summary: `data/kate_n1_2026/midi_brainage_simon_all_orig_summary.csv`
- session summary: `data/kate_n1_2026/midi_brainage_simon_all_orig_session_summary.csv`

The branch uses existing FastSurfer internal source images, not raw SIMON BIDS/NIfTI. Multi-run and acquisition variants are repeated measurements for branch/QC characterization, not independent subjects.

## Command

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
bash experiments/kate_n1_2026/run_midi_brainage_simon_all_orig.sh
```

The wrapper seeded the 12 completed stratified predictions and resumed the remaining all-orig manifest rows. Temporary MIDIBrainAge work directories were cleaned by the batch runner after each case.

## Results

All 94 cases completed successfully.

Overall metrics:

- chronological age range: 29.69-46.41 years
- predicted age range: 23.54-45.16 years
- MAE: 9.11 years
- median absolute error: 8.34 years
- mean prediction minus chronological age: -8.58 years
- RMSE: 9.96 years
- Pearson r: -0.03
- predicted-vs-chronological slope: -0.04

## Interpretation

This all-orig run confirms the stratified-12 result: the MIDIBrainAge-on-SIMON FastSurfer `orig.mgz` derivative branch is not a usable age-calibration branch for Kate reporting. Predictions are compressed and generally low for older SIMON sessions. The nearly flat correlation and slope are incompatible with an age-valid branch over this local SIMON range.

This does not prove MIDIBrainAge is invalid in general. It shows that this local derivative input branch is unsuitable for Kate-age interpretation. Results should be retained as preprocessing/domain-branch failure evidence and as repeated/acquisition QC evidence.

## Next

Do not promote Kate MIDIBrainAge smoke output as biological age from this branch. Next steps should prioritize:

1. The user-requested NeuroFM/BrainFM branch from `https://github.com/rockNroll87q/NeuroFM`, treating it as feature/QC unless an official age head is available.
2. Independent age-head comparators, especially BrainAgeNeXt and SynthBA.
3. Raw/documented T1 branches where source data are available.
