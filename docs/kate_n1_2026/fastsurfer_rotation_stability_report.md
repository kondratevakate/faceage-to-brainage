# FastSurfer Rotation Stability Report

Date: 2026-06-16

Source artifacts:

- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\summary.md`
- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\fastsurfer`
- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\logs\fs_sym_rotpos.log`
- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\logs\fs_sym_rotneg.log`
- Related longitudinal report: `outputs/fastsurfer_long_symmetry_report.md`

## Question

Is ordinary FastSurfer more stable than SynthSeg under controlled rotations of
the same scan?

## Method

The same 2018 anatomical scan was synthetically rotated in opposite directions
and processed through FastSurfer. The resulting subcortical volumes were
compared against the equivalent SynthSeg rotation floor.

This is a repeatability/stability benchmark, not a ground-truth accuracy test.

## SynthSeg vs FastSurfer Rotation Floor

Same +/-3 degree pair, 2018 scan.

| Region | SynthSeg floor % | FastSurfer floor % |
|---|---:|---:|
| L Hippocampus | 2.37 | 0.58 |
| R Hippocampus | 0.87 | 2.01 |
| L Amygdala | 2.57 | 2.40 |
| R Amygdala | 0.22 | 2.14 |
| L Thalamus | 1.17 | 0.89 |
| R Thalamus | 1.31 | 1.25 |
| L Caudate | 0.90 | 2.03 |
| R Caudate | 1.87 | 1.62 |
| L Putamen | 1.54 | 1.69 |
| R Putamen | 0.39 | 1.34 |
| L Pallidum | 1.64 | 1.11 |
| R Pallidum | 2.18 | 0.70 |

Median SynthSeg floor: 1.43%.

Median FastSurfer floor: 1.48%.

Interpretation: the two DL segmenters are essentially tied in median rotation
stability for this controlled pair. The important result is not "FastSurfer is
globally better" or "SynthSeg is globally better"; it is that both have a
similar non-equivariant rotation floor, with different structure-level error
locations.

The earlier summary also recorded cross-method error correlation around
`r = -0.068`, meaning the error patterns are effectively uncorrelated in this
small test. Same floor magnitude, different failure locations.

## Relationship to FastSurfer Long

FastSurfer Long was then tested on the same rotation pair:

- subcortical median CV improved from 0.24% to 0.12%;
- DKT cortical parcel volume median CV improved from 0.36% to 0.10%;
- HypVINN median CV improved from 1.53% to 0.80%.

This supports the conclusion that longitudinal/template-based processing can
reduce rotation sensitivity in a controlled repeatability test. It still does
not prove biological accuracy without visual QC or manual reference labels.

## Important Failure Outside This Rotation Pair

The ordinary FastSurfer run on the 2024 3DI scan failed quality control: the
earlier summary records a warning that total segmentation volume was too small
and BrainSeg was about 167 mL. Those labels should not be used as valid volume
data.

This is a method-by-contrast failure. FastSurfer can be stable on the 2018
T1-like rotation pair while still failing on an out-of-distribution 2024 3DI
contrast.

## Conclusion

For the 2018 rotation-pair test, FastSurfer and SynthSeg have similar median
orientation floors, about 1.5%. FastSurfer Long reduces this floor substantially
for the controlled rotation pair. However, 2024 3DI remains a separate failure
case and requires FS8, ReconAny/recon-all-clinical, or harmonized preprocessing
before it can be interpreted.
