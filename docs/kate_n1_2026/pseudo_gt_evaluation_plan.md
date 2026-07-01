# Pseudo-GT Evaluation Plan

Date: 2026-06-22

## Claim Boundary

This project does not have true ground-truth segmentation. Without expert manual
labels, physical phantom data, or histology, the defensible target is a
subject-specific pseudo-ground-truth reference with explicit uncertainty.

The operational model is MRI multi-view semantic fusion:

- scan, contrast, acquisition, and method outputs are noisy views of the same
  anatomy;
- registration into a subject/template space is the MRI analogue of camera
  pose/SLAM;
- the fused segmentation is a probabilistic semantic map;
- disagreement between views is uncertainty, not a nuisance to hide.

## Pseudo-GT Variants To Evaluate

| Variant | Current status | Use | Main limitation |
|---|---|---|---|
| Trusted source median | Implemented for volume summaries | Robust central reference from QC-passing sources. | More acquisitions from one method can dominate. |
| Trusted method-balanced median | Implemented for volume summaries | First collapse within method, then combine methods. | Needs at least two method families. |
| Leave-one-source-out median | Implemented for volume summaries | Scores a source against trusted sources excluding itself. | Does not remove same-method family bias. |
| Leave-one-method-out median | Implemented for volume summaries | Reduces circular validation when scoring a method. | Weak when only two method families are available. |
| STAPLE-style probabilistic fusion | Planned for registered label maps | Estimates latent consensus and source reliability. | Requires common voxel space and careful ontology mapping. |
| Majority / weighted vote label fusion | Planned for registered label maps | Simple spatial pseudo-GT baseline. | Hard labels discard uncertainty unless probabilities are retained. |
| TTA / rotation / resampling consensus | Partly available from SynthSeg TTA | Estimates method stability floor. | Same-model consensus preserves systematic bias. |
| Left-right reflection check | QC prior only | Flags gross asymmetry/orientation failures. | Brain anatomy is not perfectly symmetric. |

## Evaluation Conditions

A source can enter a trusted pseudo-GT only if:

- input QC is acceptable for the target method;
- runtime completed normally;
- visual overlay QC does not show gross brainmask or label collapse;
- label ontology overlaps with the reference ontology;
- the source is not already documented as failed or suspicious for that
  structure/acquisition;
- the pseudo-GT has at least two sources, and preferably at least two method
  families.

A source can be scored against a pseudo-GT even if it is excluded from the
trusted reference. This is how the 2024 3DI outputs are handled: they are
evaluated against the 2024 FFE consensus, not used to define it.

## Method Inclusion Notes

FastSurfer is already present in the application study, but it is not part of
the current 2024 registered pseudo-GT reference. The valid FastSurfer evidence
is mainly the 2018 rotation and FastSurfer Long rotation-pair branch. The
available 2024 3DI FastSurfer run is a documented collapse/failure case, not a
trusted 2024 FFE-compatible label source.

BrainChop 0.2.5 has been added as a reproducible candidate branch. As of
2026-06-26, `tissue_fast` completes on 2024 3DI/FFE 401/FFE 601 and can be used
as a quick tissue-QC or uncertainty signal. `mindgrab` times out locally at 5
minutes per scan, `subcortical-mini` times out locally at 10 minutes on 2024 T1
FFE 401, and the prior `subcortical` smoke-run exceeded a 15 minute CPU window.
BrainChop must remain out of anatomical pseudo-GT construction and visualization
promotion until an anatomical label model completes, has an ontology mapping,
and passes visual plus registered spatial QC.

## Metrics

Volume-level metrics available now:

- signed relative volume error;
- absolute relative volume error;
- median/p90/max error across structures;
- operational volume accuracy: `max(0, 100 - abs_relative_error_pct)`.

Spatial metrics require registered label maps:

- Dice and Jaccard overlap;
- HD95, ASSD, and/or surface Dice;
- voxel-wise entropy/disagreement maps;
- label confusion matrices;
- lesion or boundary-specific checks if relevant.

Operational interpretation thresholds for median absolute volume error:

| Median error | Interpretation |
|---:|---|
| <5% | Close volume match |
| 5-10% | Moderate disagreement; visual QC required |
| 10-20% | Large disagreement |
| >=20% | Severe disagreement / likely unusable for direct longitudinal claim |

These thresholds are QC gates, not proof of biological accuracy.

## Current Calculations

The first implemented calculation is volume-level and registration-free. It
uses overlapping ASEG/subcortical labels from SynthSeg and TIGERBx summaries.

Important design choice:

- 2024 T1 FFE sources are allowed into the trusted 2024 pseudo-GT;
- 2024 3DI sources are scored against that pseudo-GT but excluded from trusted
  reference construction because they are already known stress/failure-prone
  outputs.

Outputs are written under:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\pseudo_gt\volume_v0
```

The second implemented calculation is a fast spatial pilot. It resamples 2024
label maps to one T1 FFE grid using only NIfTI header affines and builds a
trusted hard-vote pseudo-GT from FFE sources:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\pseudo_gt\spatial_header_affine_v0
```

This pilot reports Dice/Jaccard, but it is not the final spatial accuracy stage.
The third implemented calculation explicitly registers 2024 images to the T1
FFE 401 subject space before label resampling and scoring:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\pseudo_gt\spatial_registered_v1
```

This stage reports Dice, Jaccard, HD95, ASSD, and volume error. It is the current
strongest spatial pseudo-GT accuracy stage. It remains affine-only and
fixed-template rather than nonlinear/unbiased.
