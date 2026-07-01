# Pseudo-GT Spatial Registered v1 Report

Date: 2026-06-22

## Scope

This is the first registered spatial pseudo-GT evaluation. It uses 2024 T1 FFE
401 as the fixed subject space, registers moving 2024 images with SimpleITK
affine mutual-information registration, resamples label maps with nearest
neighbor interpolation, builds trusted hard-vote pseudo-GT from FFE sources, and
scores each segmentation against that pseudo-GT.

This is stronger than the header-affine pilot because moving scans are
explicitly registered. It is still not anatomical ground truth and it is not a
nonlinear unbiased template. The appropriate interpretation is: registered
subject-space segmentation agreement against an operational FFE-derived
pseudo-GT.

## Inputs

Manifest:

```text
experiments/kate_n1_2026/pseudo_gt_registered_inputs.csv
```

Fixed registration image:

```text
images/2024/nifti/401_t1w_ffe.nii.gz
```

Fixed label grid:

```text
reprocessed_2026/seg/seg_2024_phi_t1ffe_ax.nii.gz
```

Trusted pseudo-GT sources:

- SynthSeg 2024 T1 FFE axial;
- SynthSeg 2024 T1 FFE sagittal;
- TIGERBx 2024 T1 FFE 401;
- TIGERBx 2024 T1 FFE 601.

Scored but excluded from trusted pseudo-GT:

- SynthSeg 2024 3DI;
- TIGERBx 2024 3DI.

## Outputs

Runtime outputs:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\pseudo_gt\spatial_registered_v1
```

Tracked summaries:

```text
data/kate_n1_2026/pseudo_gt_spatial_registered_metrics.csv
data/kate_n1_2026/pseudo_gt_spatial_registered_source_summary.csv
data/kate_n1_2026/pseudo_gt_spatial_registered_sources.csv
data/kate_n1_2026/pseudo_gt_spatial_registered_metadata.json
```

Large outputs are not tracked in git:

```text
registered_labels/*.nii.gz
transforms/*.tfm
pseudo_gt_registered_trusted_hard_vote.nii.gz
pseudo_gt_registered_trusted_vote_fraction.nii.gz
pseudo_gt_registered_trusted_vote_count.nii.gz
```

## Metrics

Against the trusted registered hard-vote pseudo-GT:

| Source | Median Dice | p10 Dice | Median HD95 mm | p90 HD95 mm | Median ASSD mm | Median abs volume error | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| SynthSeg 2024 3DI | 0.831 | 0.648 | 1.73 | 22.00 | 0.73 | 16.33% | Moderate spatial agreement |
| TIGERBx 2024 3DI | 0.775 | 0.575 | 2.12 | 25.08 | 0.84 | 17.76% | Moderate spatial agreement |
| SynthSeg 2024 T1 FFE axial | 0.903 | 0.863 | 1.41 | 2.00 | 0.41 | 8.88% | High spatial agreement |
| SynthSeg 2024 T1 FFE sagittal | 0.903 | 0.821 | 1.41 | 2.24 | 0.45 | 10.37% | High spatial agreement |
| TIGERBx 2024 T1 FFE 401 | 0.806 | 0.696 | 2.00 | 2.45 | 0.75 | 6.47% | Moderate spatial agreement |
| TIGERBx 2024 T1 FFE 601 | 0.789 | 0.562 | 2.24 | 2.83 | 0.82 | 7.29% | Moderate spatial agreement |

Leave-one-source-out among trusted sources:

| Source | Median Dice | p10 Dice | Median HD95 mm | Median ASSD mm | Median abs volume error |
|---|---:|---:|---:|---:|---:|
| SynthSeg 2024 T1 FFE axial | 0.836 | 0.696 | 1.73 | 0.69 | 9.77% |
| SynthSeg 2024 T1 FFE sagittal | 0.837 | 0.731 | 1.83 | 0.73 | 11.48% |
| TIGERBx 2024 T1 FFE 401 | 0.756 | 0.632 | 2.24 | 0.92 | 5.72% |
| TIGERBx 2024 T1 FFE 601 | 0.750 | 0.504 | 2.45 | 1.01 | 8.45% |

## Interpretation

Registration does not rescue 2024 3DI to the level of the FFE-derived consensus.
Both 3DI sources remain weaker than FFE sources by Dice and by p90 surface
distance. SynthSeg 2024 3DI is stronger than TIGERBx 2024 3DI in spatial Dice,
but both remain only moderate.

The high p90 HD95 for 3DI (`22-25 mm`) indicates localized structures or
boundaries with large disagreement even when median Dice is moderate. This is
consistent with the earlier volume-level finding that 3DI is not safe to promote
without visual QC.

SynthSeg FFE sources are highest by Dice, partly because SynthSeg FFE labels
participate in the hard-vote pseudo-GT. The leave-one-source-out numbers are the
less biased check. TIGERBx FFE sources have lower Dice but good median volume
error, suggesting boundary/ontology differences rather than a pure volume
collapse.

## Decision

- Use `spatial_registered_v1` as the current strongest spatial pseudo-GT
  accuracy stage.
- Keep 2024 3DI excluded from trusted pseudo-GT construction.
- Do not promote 2024 3DI segmentation volumes or labels to
  `your-brain-mri-visualization`.
- Promote only small QC summaries after visual overlay review.
- A future `spatial_template_v2` should use an unbiased subject template and
  deformable registration before making stronger spatial accuracy claims.
