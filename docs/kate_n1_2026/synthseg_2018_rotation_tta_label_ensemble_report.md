# SynthSeg 2018 Rotation TTA Label-Ensemble Report

Date: 2026-06-27

## Purpose

This is the first populated run of the general TTA label-ensemble evaluator on
real label maps. It converts the existing SynthSeg 2018 9-angle rotation sweep
from a volume-only stability result into a voxel-wise uncertainty artifact with
hard vote, vote fraction, and entropy maps.

## Inputs

Manifest:

```text
experiments/kate_n1_2026/synthseg_2018_rotation_tta_label_ensemble_inputs.csv
```

Included augmentations:

```text
-12, -9, -6, -3, 0, +3, +6, +9, +12 degrees
```

All nine label maps were verified to share the same grid before evaluation:

```text
shape: 240 x 240 x 150
affine: identical within 1e-4 tolerance
labels: 33 including background
```

## Runtime Outputs

Tracked summaries:

- `data/kate_n1_2026/synthseg_2018_rotation_tta_label_summary.csv`
- `data/kate_n1_2026/synthseg_2018_rotation_tta_global_summary.csv`

Runtime NIfTI outputs outside git:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\tta_label_ensembles\synthseg_2018_rotation_sweep
```

Files:

- `hard_vote.nii.gz`
- `vote_fraction.nii.gz`
- `entropy_bits.nii.gz`
- `tta_label_ensemble_metadata.json`

## Global Result

The ensemble used 9 SynthSeg label maps. The foreground union contained
`1,521,297` voxels, while the hard-vote foreground contained `1,255,086` voxels.
Mean foreground vote fraction was `0.694545`, and mean foreground entropy was
`0.906881` bits.

This means that the 9-angle rotation sweep has substantial voxel-boundary
disagreement even when per-structure volume CV remains modest. Volume stability
and spatial uncertainty are related but not equivalent.

## Highest Volume-CV Labels

Top labels by volume CV:

| Label | Likely structure | CV % | Mean volume ml | Mean vote fraction | Mean entropy bits |
|---:|---|---:|---:|---:|---:|
| 44 | right inferior lateral ventricle | 3.692 | 0.425 | 0.473 | 1.529 |
| 5 | left inferior lateral ventricle | 3.298 | 0.347 | 0.402 | 1.810 |
| 14 | third ventricle | 3.166 | 0.664 | 0.707 | 0.841 |
| 54 | right amygdala | 2.201 | 1.842 | 0.691 | 0.929 |
| 58 | right accumbens area | 1.847 | 0.786 | 0.551 | 1.379 |
| 18 | left amygdala | 1.667 | 1.655 | 0.648 | 1.057 |

The ranking is consistent with the prior volume-only report: small structures,
ventricular boundaries, and amygdala/accumbens regions are most sensitive to the
rotation sweep.

## Scientific Interpretation

This result improves the TTA evidence layer because it adds voxel-wise
uncertainty maps to the prior per-structure volume CV. It still does not prove
anatomical accuracy: all nine augmentations come from the same model family and
can share systematic label bias.

Use this result as:

- a real populated example of the TTA label-ensemble schema;
- a SynthSeg orientation-sensitivity uncertainty map for the high-quality 2018
  scan;
- a baseline for applying the same evaluator to other methods and test-retest
  datasets.

Do not use it as:

- manual ground truth;
- proof that low-entropy regions are anatomically correct;
- evidence that 2024 heterogeneous scans are safe without separate visual QC and
  registered pseudo-GT comparison.

## Next Step

Populate the same TTA evaluator for a comparator method. The most practical next
candidate is a FastSurfer/FastSurfer Long label subset if common-space label maps
can be exported, or BrainChop `tissue_fast` if tissue-level QC is the target.
