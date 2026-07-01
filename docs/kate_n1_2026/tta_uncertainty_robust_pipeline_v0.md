# TTA and Uncertainty Robust Pipeline Evidence v0

Date: 2026-06-27

This is a compact evidence ledger for the segmentation robustness study. It
combines already computed Kate n=1 TTA, rotation, registered pseudo-GT, and
BrainChop smoke-test evidence. It is not a final benchmark.

Tracked CSV:

- `data/kate_n1_2026/tta_uncertainty_method_evidence.csv`

Execution schema added after this ledger:

- `experiments/kate_n1_2026/tta_label_ensemble_inputs.schema.csv`
- `experiments/kate_n1_2026/evaluate_tta_label_ensemble.py`
- `docs/kate_n1_2026/tta_execution_schema_v0.md`

First populated real label-ensemble run added after the schema:

- `experiments/kate_n1_2026/synthseg_2018_rotation_tta_label_ensemble_inputs.csv`
- `data/kate_n1_2026/synthseg_2018_rotation_tta_label_summary.csv`
- `data/kate_n1_2026/synthseg_2018_rotation_tta_global_summary.csv`
- `docs/kate_n1_2026/synthseg_2018_rotation_tta_label_ensemble_report.md`

First comparator label-ensemble run:

- `experiments/kate_n1_2026/fastsurfer_long_2018_rotation_pair_label_ensemble_inputs.csv`
- `data/kate_n1_2026/fastsurfer_long_2018_rotation_pair_label_summary.csv`
- `data/kate_n1_2026/fastsurfer_long_2018_rotation_pair_global_summary.csv`
- `docs/kate_n1_2026/fastsurfer_long_2018_rotation_pair_label_ensemble_report.md`

## Decision Counts

- `candidate_or_comparator`: 2
- `do_not_interpret_cross_protocol_change_without_uncertainty`: 1
- `exclude_from_visualization`: 2
- `not_promoted_runtime_timeout`: 4
- `primary_2024_candidate`: 2
- `quick_tissue_qc_candidate`: 3
- `use_as_method_floor_when_input_qc_passes`: 7
- `use_as_primary_tta_volume_floor`: 4

## Current Pipeline Position

Primary 2024 anatomical segmentation candidate:

- FFE-derived registered consensus, with SynthSeg FFE sources as the strongest
  current single-method spatial candidates.

Do not promote:

- 2024 3DI anatomical segmentations from SynthSeg, TIGERBx, FastSurfer, or
  BrainChop anatomical models.
- BrainChop `tissue_fast` as anatomical segmentation. It is tissue-level QC
  only.

Use as uncertainty signals:

- SynthSeg 9-angle rotation TTA CV and TTA mean for per-structure volume
  stability.
- Registered source-vs-consensus disagreement for 2024 FFE/3DI spatial
  uncertainty.
- BrainChop `tissue_fast` as a fast tissue-level contrast/QC branch.
- FastSurfer and FastSurfer Long rotation CV as method-floor evidence when
  input QC passes.

## Scientific Boundary

TTA reduces or characterizes orientation sensitivity. It does not prove
anatomical accuracy and does not solve scanner/protocol harmonization. The
current evidence supports a robustness-aware prediction pipeline, not a claim of
manual ground truth.

## Next Experiments

1. Apply the same TTA schema to at least SynthSeg and one fast/light comparator
   on a test-retest dataset before claiming a general pipeline.
2. Add nonlinear/unbiased subject-template registration before final spatial
   accuracy claims.
