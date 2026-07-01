# SIAM Benchmark Candidate Note

Date: 2026-06-27

## Identification

The model the user referred to as "siam segment it all" is SIAM: Segment It All
Model.

Primary sources checked:

- arXiv: `https://arxiv.org/abs/2605.02737`
- upstream Python repository: `https://github.com/romainVala/SIAM`
- native C++/ONNX port: `https://github.com/NeuroJSON/siamize`
- Neurostars `siamize` announcement:
  `https://neurostars.org/t/announcing-siamize-v0-2-native-c-port-of-siam-for-fast-full-head-segmentation-with-gpu/36249`

## Why It Belongs In The Benchmark

SIAM is relevant because it targets head and brain tissue segmentation across
heterogeneous 3D human head volumes, including multiple MRI contrasts and CT.
That makes it directly relevant to the current failure mode: Kate 2024 3DI and
FFE inputs are acquisition/contrast stress tests where ordinary FreeSurfer and
FastSurfer-style anatomical streams can fail or become unstable.

It should be benchmarked as a contrast-robust whole-head/tissue comparator, not
as a direct FreeSurfer DKT/ASEG replacement.

## Candidate Execution Routes

1. Upstream Python SIAM via `siam-pred`.
2. `siamize`, the C++/ONNX port, via local build or Docker image
   `openjdata/siamize:v2026.6`.

For this project, `siamize` is the more practical first route because it exposes
native CPU/GPU backends and is designed as a deployment port. The upstream SIAM
Python route remains the reference implementation.

## TTA And Uncertainty Role

SIAM can contribute uncertainty in three ways:

- external TTA: rotations/resampling followed by inverse-resampling into the
  existing `evaluate_tta_label_ensemble.py` schema;
- fold/ensemble disagreement if the chosen SIAM/siamize route exposes fold-wise
  predictions;
- TPM/probability-map uncertainty if available from `siamize`.

The first benchmark should not average SIAM with SynthSeg/TIGERBx until the
label ontology is mapped. It should first measure runtime feasibility, visual QC,
and tissue-level agreement/disagreement with the current 2024 FFE-derived
registered consensus.

## Critical Interpretation

SIAM is new and promising, but the evidence level for this project remains
exploratory until locally reproduced. A recent paper, open repository, or
claimed contrast robustness is not enough to promote outputs to visualization.

Promotion gates:

1. smoke run completes on at least one 2024 FFE candidate;
2. output labels and voxel space are documented;
3. label ontology is mapped to a tissue/common-label subset;
4. visual QC overlay passes;
5. registered-space metrics are computed against the current FFE-derived
   pseudo-GT;
6. test-retest or travelling-head stability is measured.

## Current Status

Recorded as `candidate_recorded_not_run`. Do not run locally on the laptop until
RAM/GPU feasibility is checked. Upstream SIAM notes high RAM/GPU needs for large
fields of view; the first serious run should prefer a GPU/cloud path or the
lighter `siamize` port with explicit resource guards.
