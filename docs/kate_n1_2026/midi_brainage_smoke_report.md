# MIDIBrainAge smoke report

Date: 2026-06-23

Scope: technical smoke test of an explicit brain-age head after the model search. This is not a validation claim and not a final Kate age estimate.

## What Ran

Model: `MIDIBrainAge_T1_ensemble`

Input: Kate 2018 raw T1-like GE FSPGR, `/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz`

Environment: isolated WSL venv at `/home/kate/.venvs/midi_brainage_py311`

Working copy: `/home/kate/midi_brainage_work`, not the vendor repo on D.

Preprocessing path: HD-BET CPU skull stripping, ANTs N4 plus affine registration to the bundled MNI152 T1 brain template, MIDI RAS reorientation, 1.4 mm spacing, crop/pad to 130 cubed, z-score normalization, clamp -1..5.

## Result

`midi_smoke_kate2018_patched_v2/brain_age_output.csv`:

```text
ID,Predicted_age (years)
kate_2018_t1,29.8
```

The tracked summary row is in `data/kate_n1_2026/midi_brainage_smoke_results.csv`.

## SIMON-3 Sanity Check

I also ran the same patched MIDIBrainAge T1 path on the first three local SIMON FastSurfer `orig.mgz` derivatives. These are not raw BIDS/NIfTI inputs, so this is a technical sanity check, not the final SIMON validation branch.

Tracked results: `data/kate_n1_2026/midi_brainage_simon3_smoke_results.csv`.

| ID | Chronological age | Predicted age | Delta |
| --- | ---: | ---: | ---: |
| SIMON_ses001_run1 | 29.7 | 37.24 | 7.54 |
| SIMON_ses002_run1 | 30.4 | 33.20 | 2.80 |
| SIMON_ses003_run1 | 32.2 | 35.22 | 3.02 |

Summary: MAE 4.45 years, mean bias +4.45 years, prediction range 33.20 to 37.24 years. The n=3 correlation is negative (-0.27), but this is not interpretable because the subset is tiny and age range is only 2.5 years.

This is materially better than the failed BrainIAC adult-age behavior because the units are plausible adult years, but it still does not validate MIDIBrainAge for Kate. It only justifies scaling to a broader SIMON subset/full run if disk and CPU budget allow.

## Compatibility Notes

The upstream MIDIBrainAge repo expects an older HD-BET command line and older MONAI behavior. I did not edit the source vendor repo on D. I made a temporary `/home` working copy and applied two compatibility fixes there:

- `hd-bet -mode fast` was replaced with HD-BET 2.x `--disable_tta`;
- MONAI `Spacing` output was kept as a channel-first array and converted back to `np.ndarray` before returning to `run_inference.py`.

The first smoke attempt failed after registration because of the MONAI tensor shape/API mismatch. The patched second run completed.

## Interpretation

This result only shows that one explicit age-regression model can execute locally on one Kate scan. It does not establish biological age, clinical age, or model validity for Kate.

Before reporting any Kate age estimate, run SIMON calibration and at least one additional model branch. Minimum checks:

- plausible adult-year units on SIMON;
- positive relationship with SIMON chronological age on a sufficiently broad subset/full run;
- no pathological bias over SIMON age range 29.7 to 46.4 years;
- consistent preprocessing provenance per scan;
- compare against BrainAgeNeXt and/or SynthBA before any model averaging.
