# SIMON Segmentation Test-Retest Manifest Report

Date: 2026-06-27

## Purpose

This report records the first external-dataset scaffold for the segmentation
benchmark completion criteria. It does not yet run new SOTA segmenters. It
identifies locally available SIMON FreeSurfer8 derivative label maps that can be
used for derivative-level repeatability analysis while raw-input processing is
still unresolved.

## Generated Artifacts

Builder:

```text
experiments/kate_n1_2026/build_simon_segmentation_test_retest_manifest.py
```

Tracked outputs:

```text
data/kate_n1_2026/simon_freesurfer8_segmentation_sessions.csv
data/kate_n1_2026/simon_freesurfer8_segmentation_pairs.csv
data/kate_n1_2026/simon_segmentation_test_retest_status.csv
```

Local derivative root:

```text
D:\data\freesurfer8_simon
```

## Result

The manifest builder found:

```text
SIMON phenotype sessions: 73
sessions with FreeSurfer8 DKT+ASEG label map: 69
consecutive session pairs: 72
usable consecutive derivative pairs: 66
```

The label target for the first scaffold is:

```text
mri/aparc.DKTatlas+aseg.mgz
```

## Interpretation

This is useful progress toward the test-retest criterion because it gives a
concrete external dataset manifest and consecutive-pair structure for spatial
repeatability analysis.

It is not sufficient for the final goal because it is derivative-level evidence:
the manifest starts from existing FreeSurfer8 outputs, not from raw MRI inputs
processed uniformly through SynthSeg, FastSurfer, BrainChop, SIAM, TIGERBx, or
other candidate methods.

Use this branch as:

- a repeatability scaffold;
- a way to debug pairwise spatial metrics and aggregation;
- secondary evidence for FreeSurfer8 derivative consistency.

Do not use this branch as:

- proof of raw-input SOTA robustness;
- proof that TTA uncertainty is calibrated;
- a substitute for multi-method processing of raw or harmonized inputs.

## Next Step

The next benchmark step is to connect this manifest to method outputs:

1. locate or generate raw/harmonized SIMON inputs for at least SynthSeg and one
   comparator;
2. populate TTA label ensembles on a small SIMON subset;
3. compare TTA uncertainty with pairwise repeatability errors from the SIMON
   consecutive-pair scaffold.
