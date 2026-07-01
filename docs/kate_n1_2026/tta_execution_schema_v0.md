# TTA Execution Schema v0

Date: 2026-06-27

## Purpose

This document defines the reproducible test-time-augmentation label-ensemble
schema for the Kate n=1 application study and later SBPR/SIMON research runs.
It converts method-specific TTA outputs into a common measurement layer:
inverse-resampled label maps, per-label volume stability, hard-vote consensus,
vote fraction, and entropy.

## Scientific boundary

TTA agreement is a stability metric, not anatomical ground truth. A method can be
stable and wrong if all augmentations preserve the same systematic error. A TTA
branch can be promoted only when it is interpreted together with visual QC,
registered pseudo-GT disagreement, and test-retest or travelling-head evidence.

## Required pipeline contract

Each method-specific wrapper must produce a manifest compatible with:

```text
experiments/kate_n1_2026/tta_label_ensemble_inputs.schema.csv
```

Required columns:

- `augmentation_id`: unique row id, for example `yaw_m03` or `native`.
- `method`: segmenter family and version, for example `SynthSeg_2.x`.
- `scan_id`: stable input scan id.
- `label_path`: label map path, absolute or relative to the data root.
- `include_in_vote`: `1` for label maps used in the ensemble.

Optional but recommended columns:

- `transform_id`: unique transform provenance id.
- `transform_family`: rotation, reflection, resampling, contrast, or native.
- `angle_deg`: signed angle for rotation sweeps when applicable.
- `notes`: short runtime or QC note.

All label maps must already be inverse-resampled into the same target grid. The
evaluator checks that shape and affine match before calculating metrics.

## Evaluator

Primary evaluator:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
/home/kate/.venvs/tigerbx/bin/python \
  experiments/kate_n1_2026/evaluate_tta_label_ensemble.py \
  --manifest experiments/kate_n1_2026/my_method_tta_manifest.csv \
  --output-dir /mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/tta_label_ensembles/my_method_scan \
  --summary-csv data/kate_n1_2026/my_method_tta_label_summary.csv \
  --global-summary-csv data/kate_n1_2026/my_method_tta_global_summary.csv \
  --write-hard-vote \
  --write-vote-fraction \
  --write-entropy
```

Tracked outputs should remain small CSV summaries and protocol documents. NIfTI
hard-vote, vote-fraction, and entropy maps are runtime artifacts and should stay
outside git.

## Metrics

Per label:

- mean, standard deviation, and coefficient of variation of voxel counts;
- mean and standard deviation of volume in ml;
- hard-vote consensus voxel count;
- number of voxels where the label appears in any augmentation;
- mean consensus vote fraction;
- mean consensus entropy in bits.

Global:

- number of included augmentations;
- common grid shape;
- number of labels;
- foreground union voxels;
- hard-vote foreground voxels;
- mean foreground vote fraction;
- mean foreground entropy.

## Recommended first transforms

The first application set should match the existing SynthSeg rotation evidence:
native plus yaw rotations from `-12` to `+12` degrees in 3 degree steps. For
generalization, add pitch and roll only after the yaw-only branch has passed
visual QC and registered-space comparison. Reflection is a stress test and must
not be mixed into the main anatomical vote unless label laterality has been
explicitly corrected.

## Promotion gates

A TTA label ensemble can inform `your-brain-mri-visualization` only when:

1. input orientation and inverse-resampling provenance are recorded;
2. hard-vote, vote-fraction, and entropy maps are generated in a documented
   common space;
3. visual QC overlays show no gross alignment or label-collapse failure;
4. spatial comparison against the current registered pseudo-GT reports
   Dice/Jaccard/HD95/ASSD or an explicit reason it cannot;
5. test-retest or travelling-head data show that low TTA uncertainty corresponds
   to reproducible measurements.

## Current interpretation

This schema turns the existing TTA/uncertainty ledger into executable
infrastructure. It does not change the current application conclusion: 2024 3DI
anatomical labels remain non-promoted; FFE-derived registered consensus and
SynthSeg FFE remain the strongest current 2024 visualization candidates; and
BrainChop `tissue_fast` remains a tissue-QC branch rather than an anatomical
label source.
