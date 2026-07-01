# MIDIBrainAge SRPBS Traveling Subjects scale gate

Date: 2026-06-24

## Scope

The requested "other dataset" was identified as SRPBS Traveling Subjects. I found `D:\data\SRPBS_TS.tar.gz`, which contains BIDS-like raw/source T1w files under `SRPBS_TS/sourcedata/sub-*/ses-site*/anat/*_T1w.nii.gz`. I did not find a literal `*.tar.zip` under `D:\data` or `D:\projects\02_academia`. The file `D:\data\my_data.zip` is gzip/tar content with Kate 2024 data and was not used for this SRPBS run.

The archive also contains `SRPBS_TS/sourcedata/participants.tsv`; extracted age labels were saved as `data/kate_n1_2026/srpbs_travelling_participants.tsv`.

This run used the already available derivative branch `D:\data\fastserfer_travelling\*_orig.mgz`:

- 143 FastSurfer `orig.mgz` inputs.
- 9 traveling subjects.
- 16 sites, with one missing subject-site point (`sub-06/siteATTd3`).

## Method

MIDIBrainAge T1 ensemble was run in the isolated WSL venv at `/home/kate/.venvs/midi_brainage_py311`. For T1, MIDIBrainAge requires `--skull_strip`; its preprocessing calls HD-BET and then registers to MNI152 before inference. The batch runner was fixed so the child process PATH uses the venv `bin` path without resolving the Python symlink to the base UV interpreter; otherwise `hd-bet` was not visible inside `os.system`.

No large volume outputs were written to git. Per-case temporary files were cleaned after each prediction.

## Results

Scale gate: one site (`siteATTd1`) across all 9 subjects.

Output files:

- `data/kate_n1_2026/midi_brainage_srpbs_travelling_fastsurfer_orig_siteATTd1_predictions.csv`
- `data/kate_n1_2026/midi_brainage_srpbs_travelling_fastsurfer_orig_siteATTd1_summary.csv`
- `data/kate_n1_2026/srpbs_travelling_participants.tsv`
- `experiments/kate_n1_2026/midi_brainage_srpbs_travelling_fastsurfer_orig_siteATTd1_inputs.csv`

All 9 cases completed successfully. Chronological ages ranged from 24 to 32 years. Predictions ranged from 24.96 to 33.14 years, with mean 28.53 and SD 3.03 years. Against the extracted participant ages, this single-site gate had MAE 1.74 years, mean prediction minus chronological age +1.53 years, RMSE 2.15 years, and Pearson r 0.85. Mean runtime was 584.53 seconds per case on CPU. A full 143-case sequential FastSurfer-orig run is estimated at about 23.23 CPU hours.

Per-subject predictions:

| Subject | Site | Chronological age | Predicted age | Delta |
|---|---:|---:|---:|---:|
| sub-01 | siteATTd1 | 25 | 29.14 | +4.14 |
| sub-02 | siteATTd1 | 27 | 28.52 | +1.52 |
| sub-03 | siteATTd1 | 26 | 25.08 | -0.92 |
| sub-04 | siteATTd1 | 26 | 26.42 | +0.42 |
| sub-05 | siteATTd1 | 32 | 33.14 | +1.14 |
| sub-06 | siteATTd1 | 24 | 24.96 | +0.96 |
| sub-07 | siteATTd1 | 25 | 26.46 | +1.46 |
| sub-08 | siteATTd1 | 28 | 31.96 | +3.96 |
| sub-09 | siteATTd1 | 30 | 31.12 | +1.12 |

## Interpretation

This is an application/robustness branch with a small labeled sanity check, not a validation claim. The siteATTd1 result supports that MIDIBrainAge is at least producing plausible adult-year values on this SRPBS branch, unlike the previously rejected BrainIAC adult-age interpretation. However, n=9, one site, and a narrow 24-32 year age range are not enough to validate accuracy or generalize to Kate. The main SRPBS value remains QC: whether predictions are stable across sites for the same traveling subject once the full site matrix is run.

The current branch uses FastSurfer `orig.mgz`, which is a conformed derivative, not raw BIDS input. A raw T1w branch from `SRPBS_TS.tar.gz` is available in principle, but requires extracting only the `anat/*_T1w.nii.gz` files and then running the same MIDIBrainAge preprocessing path.

## Next

Run the full 143-case FastSurfer-orig SRPBS matrix only as a long CPU batch or on faster hardware. The main analysis should be within-subject across-site prediction spread, while age-error summaries should stay secondary because the age range is narrow.

For the no-preprocessing branch, extract only raw/source T1w files from `SRPBS_TS.tar.gz`, build a raw manifest, and run a small raw-vs-FastSurfer paired gate before scaling.
