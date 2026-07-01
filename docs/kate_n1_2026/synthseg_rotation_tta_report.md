# SynthSeg Rotation and TTA Stability Report

Date: 2026-06-16

Source artifacts:

- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\summary.md`
- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\seg`
- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\vol`
- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\seg_sweep`
- `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\vol_sweep`

## Question

How much of the observed scan-to-scan variation can be explained by algorithmic
orientation instability of SynthSeg, rather than by scanner/protocol differences?

## Method

The same 2018 scan was synthetically rotated. The controlled +/-3 degree pair
estimates the method-variance floor. A 9-angle sweep from -12 to +12 degrees in
3 degree steps estimates test-time augmentation behavior.

Pairwise floor is interpreted as a consistency/stability metric, not accuracy.

## Reuter Symmetry Floor vs Cross-Scanner Spread

Same 2018 scan, +/-3 degree trilinear pair. Within-pair spread is the method
floor; cross-scanner spread is from 2018/2022/2024 native scans.

| Region | SynthSeg floor % | Cross-scanner % | Ratio |
|---|---:|---:|---:|
| L Hippocampus | 2.4 | 7.3 | 3.1x |
| R Hippocampus | 0.9 | 15.7 | 18.2x |
| L Amygdala | 2.6 | 6.5 | 2.5x |
| R Amygdala | 0.2 | 28.2 | 130.5x |
| L Thalamus | 1.2 | 6.8 | 5.8x |
| R Thalamus | 1.3 | 4.5 | 3.4x |
| L Caudate | 0.9 | 29.7 | 33.1x |
| R Caudate | 1.9 | 11.1 | 5.9x |
| L Putamen | 1.5 | 18.2 | 11.8x |
| R Putamen | 0.4 | 17.7 | 45.2x |
| L Pallidum | 1.6 | 45.3 | 27.7x |
| R Pallidum | 2.2 | 27.8 | 12.7x |

Median floor: 1.4%.

Median cross-scanner spread: 16.7%.

Interpretation: cross-scanner/protocol variation is about 12x the SynthSeg
rotation floor. The scanner/protocol effect is therefore not just processing
noise.

## Decomposition of Rotation Floor

Same 2018 scan, +3 degree rotation.

| Component | Value | Meaning |
|---|---:|---|
| Interpolation only | 0.05% | Rotate the label map, no model rerun. |
| Model instability | 1.36% | Added by rerunning the segmentation model. |
| Total floor | 1.37% | Observed floor for this perturbation. |

Interpretation: about 97% of the rotation floor is model instability and about
3% is interpolation physics. SynthSeg is not rotation-equivariant in this test.

## 9-Angle TTA Sweep

Angles: -12, -9, -6, -3, 0, +3, +6, +9, +12 degrees.

| Region | CV % | TTA mean mL | Max excursion at +/-12 degrees |
|---|---:|---:|---:|
| L Hippocampus | 0.72 | 3.789 | 1.36% |
| R Hippocampus | 1.27 | 3.757 | 2.34% |
| L Amygdala | 1.81 | 1.662 | 5.49% |
| R Amygdala | 1.83 | 1.839 | 4.80% |
| L Thalamus | 1.21 | 6.751 | 1.98% |
| R Thalamus | 1.29 | 6.821 | 2.52% |
| L Caudate | 0.69 | 3.854 | 2.02% |
| R Caudate | 0.88 | 3.863 | 1.45% |
| L Putamen | 0.71 | 5.317 | 0.21% |
| R Putamen | 0.28 | 5.384 | 0.24% |
| L Pallidum | 1.36 | 1.469 | 3.19% |
| R Pallidum | 1.38 | 1.650 | 2.43% |

Median 9-angle TTA CV: 1.24%.

Amygdala is the most sensitive structure in this sweep, with about 5.5% maximum
excursion at the outer angles.

## Conclusion

SynthSeg is robust enough that its orientation floor is much smaller than the
observed cross-scanner/protocol variation, but it is not invariant to rotation.
TTA reduces and characterizes orientation bias, but it does not remove the core
scanner/protocol problem. This branch should be reported as a method-stability
floor, not as an accuracy benchmark.
