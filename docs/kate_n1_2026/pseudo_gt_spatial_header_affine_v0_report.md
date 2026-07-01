# Pseudo-GT Spatial Header-Affine v0 Report

Date: 2026-06-22

## Scope

This is the first spatial pseudo-GT pilot. It resamples 2024 label maps into a
single 2024 T1 FFE target grid using NIfTI header affines and nearest-neighbor
interpolation, then builds a trusted hard-vote pseudo-GT from FFE sources.

This is not the final spatial validation stage. It does not perform nonlinear
registration, and it does not compute HD95, ASSD, or surface Dice. Its purpose is
to provide a fast reproducible Dice/Jaccard QC pass before investing in a full
subject-template registration workflow.

## Inputs

Manifest:

```text
experiments/kate_n1_2026/pseudo_gt_spatial_inputs.csv
```

Trusted pseudo-GT sources:

- SynthSeg 2024 T1 FFE axial;
- SynthSeg 2024 T1 FFE sagittal;
- TIGERBx 2024 T1 FFE 401;
- TIGERBx 2024 T1 FFE 601.

Scored but excluded from trusted pseudo-GT:

- SynthSeg 2024 3DI;
- TIGERBx 2024 3DI.

Target grid:

```text
synthseg_2024_t1ffe_ax
```

## Outputs

Runtime outputs:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\pseudo_gt\spatial_header_affine_v0
```

Tracked summaries:

```text
data/kate_n1_2026/pseudo_gt_spatial_header_affine_metrics.csv
data/kate_n1_2026/pseudo_gt_spatial_header_affine_source_summary.csv
data/kate_n1_2026/pseudo_gt_spatial_header_affine_metadata.json
```

Large NIfTI outputs are not tracked in git:

```text
pseudo_gt_trusted_hard_vote.nii.gz
pseudo_gt_trusted_vote_fraction.nii.gz
pseudo_gt_trusted_vote_count.nii.gz
```

## Source Summary

Against the trusted hard-vote header-affine pseudo-GT:

| Source | Median Dice | p10 Dice | Median abs volume error | Interpretation |
|---|---:|---:|---:|---|
| SynthSeg 2024 3DI | 0.816 | 0.637 | 15.52% | Moderate spatial agreement |
| TIGERBx 2024 3DI | 0.764 | 0.560 | 16.94% | Moderate spatial agreement |
| SynthSeg 2024 T1 FFE axial | 0.901 | 0.824 | 9.54% | High spatial agreement |
| SynthSeg 2024 T1 FFE sagittal | 0.901 | 0.809 | 11.02% | High spatial agreement |
| TIGERBx 2024 T1 FFE 401 | 0.805 | 0.714 | 6.75% | Moderate spatial agreement |
| TIGERBx 2024 T1 FFE 601 | 0.792 | 0.545 | 7.95% | Moderate spatial agreement |

Leave-one-source-out among trusted FFE sources reduces Dice, as expected:

| Source | Median Dice | p10 Dice | Median abs volume error |
|---|---:|---:|---:|
| SynthSeg 2024 T1 FFE axial | 0.836 | 0.697 | 9.17% |
| SynthSeg 2024 T1 FFE sagittal | 0.840 | 0.720 | 11.70% |
| TIGERBx 2024 T1 FFE 401 | 0.754 | 0.633 | 6.50% |
| TIGERBx 2024 T1 FFE 601 | 0.750 | 0.508 | 9.11% |

## Interpretation

The spatial pilot is consistent with the volume-level result, but less severe:
2024 3DI is not a complete spatial collapse under header-affine Dice, but it is
weaker than the FFE sources and has low-dice structures such as CSF, inferior
lateral ventricles, cortex, white matter, accumbens, and caudate.

The strongest spatial agreement comes from SynthSeg FFE sources when compared
to the hard-vote consensus. This is partly expected because SynthSeg FFE sources
are included in the consensus; leave-one-source-out Dice is the less biased
number for trusted sources.

TIGERBx FFE sources have lower Dice but relatively good median volume error.
This suggests a boundary/ontology/registration difference rather than a simple
global volume failure.

## Decision

- Keep this as `spatial_header_affine_v0`, a QC pilot only.
- Do not use this as final segmentation accuracy.
- Next spatial stage should use explicit registration to an unbiased
  subject-template space, then compute Dice, HD95/ASSD, surface Dice, and
  uncertainty maps.
