# FastSurfer Long 2018 Rotation-Pair Label-Ensemble Report

Date: 2026-06-27

## Purpose

This report records the first comparator run through the shared TTA
label-ensemble evaluator. It uses the existing FastSurfer Long v2 outputs from
the 2018 synthetic rotation pair and scores DKT+ASEG label agreement in a common
FastSurfer conformed space.

## Inputs

Manifest:

```text
experiments/kate_n1_2026/fastsurfer_long_2018_rotation_pair_label_ensemble_inputs.csv
```

Included label maps:

```text
reprocessed_2026/symmetry/fastsurfer_long_v2/sym_rotneg/mri/aparc.DKTatlas+aseg.deep.mgz
reprocessed_2026/symmetry/fastsurfer_long_v2/sym_rotpos/mri/aparc.DKTatlas+aseg.deep.mgz
```

Both label maps were verified before evaluation:

```text
shape: 256 x 256 x 256
affine: identical within 1e-4 tolerance
labels: 96 including background
```

## Runtime Outputs

Tracked summaries:

- `data/kate_n1_2026/fastsurfer_long_2018_rotation_pair_label_summary.csv`
- `data/kate_n1_2026/fastsurfer_long_2018_rotation_pair_global_summary.csv`

Runtime metadata outside git:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\tta_label_ensembles\fastsurfer_long_2018_rotation_pair\tta_label_ensemble_metadata.json
```

Voxel uncertainty maps were not written for this first comparator pass. The
current purpose is small tracked summary evidence; NIfTI maps can be generated
later after deciding whether the two-member pair is worth visual QC overlays.

## Global Result

The ensemble used two FastSurfer Long v2 DKT+ASEG label maps.

```text
foreground union voxels: 1,087,481
hard-vote foreground voxels: 1,070,360
mean foreground vote fraction: 0.992128
mean foreground entropy bits: 0.015744
```

Compared with the populated SynthSeg 9-angle sweep, this FastSurfer Long
rotation-pair run shows much lower voxel-level disagreement. The comparison is
not one-to-one because SynthSeg uses nine angles over a wider range and
FastSurfer Long uses only a +/-3 degree pair. Still, it is useful as an
independent comparator showing that the shared evaluator can score both SynthSeg
NIfTI labels and FastSurfer/FastSurfer Long MGZ labels.

## Highest Volume-CV Labels

Top labels by volume CV:

| Label | CV % | Mean volume ml | Mean vote fraction | Mean entropy bits |
|---:|---:|---:|---:|---:|
| 63 | 1.924 | 0.441 | 1.000 | 0.000 |
| 1026 | 1.274 | 3.442 | 1.000 | 0.000 |
| 1021 | 1.073 | 1.911 | 1.000 | 0.000 |
| 2021 | 1.018 | 2.015 | 1.000 | 0.000 |
| 2026 | 0.765 | 2.033 | 1.000 | 0.000 |

The pairwise result is consistent with the earlier FastSurfer Long volume
stability report: the longitudinal stream is highly repeatable under this small
rotation perturbation.

## Scientific Interpretation

This is a repeatability and uncertainty result, not an accuracy result.
FastSurfer Long can be stable and still share systematic anatomical bias.

Use this result as:

- a comparator branch for the TTA label-ensemble evaluator;
- evidence that the evaluator can handle FastSurfer MGZ label maps;
- a basis for deciding whether FastSurfer Long labels should be exported to
  common-space NIfTI uncertainty maps.

Do not use it as:

- manual ground truth;
- evidence that FastSurfer Long handles the 2024 heterogeneous scans;
- direct proof that FastSurfer Long is spatially more accurate than SynthSeg.

## Next Step

The next benchmark step should move from Kate-only rotation pairs to external
test-retest data. For uncertainty validation, the project needs to test whether
TTA disagreement predicts repeatability error across SIMON or SBPR/SRPBS.
