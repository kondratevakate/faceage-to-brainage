# SIMON FastSurfer `_orig.mgz` Input Benchmark Note

Date: 2026-06-27

## Decision

We can run segmenters on the local SIMON `_orig.mgz` files and use them as the
available input dataset for a standardized-input benchmark.

The branch must be named explicitly:

```text
SIMON FastSurfer orig internal-source input benchmark
```

It should not be called a segmentation-derivative benchmark. It also should not
be overclaimed as untouched scanner-native DICOM/NIfTI unless the original input
provenance and voxel geometry are verified.

## Why `_orig.mgz` Is Useful

The files are already local and cover many SIMON sessions. They provide a
practical common input for fast segmenter smoke tests, repeatability metrics,
and method-failure triage. This is especially useful for methods that can run on
MGZ/NIfTI volumes without requiring the original DICOM layout.

## Scientific Classification

`_orig.mgz` is best treated as a FreeSurfer/FastSurfer internal source image.
It is not a segmentation output and not a morphometric derivative. In standard
FreeSurfer processing, input images are first converted into `mri/orig/NNN.mgz`;
then `rawavg.mgz` is conformed into `orig.mgz`. FastSurfer also checks and
conforms input images for network compatibility when needed.

So for this project the precise wording is:

```text
available internal-source `_orig.mgz` inputs
```

not:

```text
segmentation derivatives
```

and not automatically:

```text
untouched scanner-native raw inputs
```

The remaining risk is input-conditioning bias: if a method is evaluated on
`_orig.mgz`, part of the result may reflect FreeSurfer/FastSurfer conversion,
orientation, conforming, or averaging decisions rather than pure native scanner
robustness.

## Supported Claims

This branch can support these claims:

- the method can run on the available standardized SIMON inputs;
- the method has a measured test-retest or consecutive-session stability on
  internal-source `_orig.mgz` inputs;
- the method fails or succeeds under a reproducible source-image input setting.

This branch cannot support these claims by itself:

- the method is robust to untouched scanner-native DICOM/NIfTI input;
- the method is anatomically accurate without visual QC or a consensus/reference
  comparison;
- the method improves over native-input processing, unless compared against a
  raw/native-input branch.

## Generated Artifacts

Builder:

```text
experiments/kate_n1_2026/build_simon_orig_segmentation_inputs.py
```

Tracked outputs:

```text
data/kate_n1_2026/simon_fastsurfer_orig_segmentation_inputs.csv
data/kate_n1_2026/simon_fastsurfer_orig_segmentation_run1_pairs.csv
data/kate_n1_2026/simon_orig_input_benchmark_status.csv
```

## Current Manifest Counts

The local root contains 94 `_orig.mgz` inputs represented in the manifest:

- 73 unique SIMON sessions;
- 88 files explicitly marked as `T1w`;
- 6 files with unknown modality from the filename;
- 69 run-1 inputs;
- 56 consecutive run-1 pairs for first-pass repeatability.

## Next Action

Run a small run-1 subset first. The priority order is:

1. SynthSeg, because it is already central to the current robust-pipeline
   evidence and can provide comparable label maps.
2. BrainChop tissue/anatomical variants, only where runtime is acceptable and
   the ontology is mapped.
3. SIAM/siamize, as a whole-head/tissue robustness comparator rather than a
   FreeSurfer DKT/ASEG replacement.

Promote the branch only after visual QC and registered spatial comparisons are
available.
