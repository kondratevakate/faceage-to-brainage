# Pseudo-GT Volume v0 Report

Date: 2026-06-22

## Scope

This is the first computed pseudo-ground-truth evaluation for the Kate n=1
application branch. It is volume-level only. It does not compute Dice, HD95,
ASSD, or surface Dice, because those require label maps registered to a common
subject/template space.

The purpose is to test whether segmentation-derived volumes agree with
operational pseudo-GT references built from multiple noisy views.

## Inputs

Manifest:

```text
experiments/kate_n1_2026/pseudo_gt_volume_inputs.csv
```

Current sources:

- SynthSeg 2018 primary T1;
- TIGERBx 2018 primary T1;
- SynthSeg 2022 primary T1;
- TIGERBx 2022 primary T1;
- SynthSeg 2024 3DI;
- TIGERBx 2024 3DI;
- SynthSeg 2024 T1 FFE axial and sagittal;
- TIGERBx 2024 T1 FFE 401 and 601.

The 2024 FFE sources enter the trusted 2024 pseudo-GT. The 2024 3DI sources are
scored against the pseudo-GT but excluded from trusted reference construction,
because 2024 3DI is already a documented stress/failure-prone input.

## Pseudo-GT Variants Computed

| Variant | Meaning |
|---|---|
| `all_source_median` | Median across all sources in the session group, including suspicious ones. |
| `trusted_source_median` | Median across sources marked `include_in_reference=1`. |
| `trusted_method_balanced_median` | Median of per-method trusted medians, reducing domination by one method family. |
| `trusted_leave_one_source_out` | Trusted median excluding the exact source being scored. |
| `trusted_leave_one_method_out` | Trusted median excluding all sources from the method being scored. |

## Outputs

Runtime outputs:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\pseudo_gt\volume_v0
```

Tracked small summaries:

```text
data/kate_n1_2026/pseudo_gt_volume_long.csv
data/kate_n1_2026/pseudo_gt_volume_references.csv
data/kate_n1_2026/pseudo_gt_volume_accuracy.csv
data/kate_n1_2026/pseudo_gt_volume_source_summary.csv
data/kate_n1_2026/pseudo_gt_volume_metadata.json
```

Run metadata:

- sources: 10;
- normalized volume rows: 315;
- reference rows: 276;
- source-vs-reference accuracy rows: 1281.

## 2024 T1-Like Result

Median absolute volume error across overlapping structures:

| Reference variant | Source | Median error | p90 error | Interpretation |
|---|---|---:|---:|---|
| trusted source median | SynthSeg 2024 3DI | 10.52% | 32.25% | Large disagreement |
| trusted source median | TIGERBx 2024 3DI | 13.86% | 47.27% | Large disagreement |
| trusted source median | SynthSeg 2024 FFE axial | 6.30% | 16.83% | Moderate disagreement |
| trusted source median | SynthSeg 2024 FFE sagittal | 5.78% | 16.26% | Moderate disagreement |
| trusted source median | TIGERBx 2024 FFE 401 | 4.56% | 12.43% | Close volume match |
| trusted source median | TIGERBx 2024 FFE 601 | 5.83% | 46.61% | Moderate disagreement |
| trusted method-balanced median | SynthSeg 2024 3DI | 11.44% | 31.74% | Large disagreement |
| trusted method-balanced median | TIGERBx 2024 3DI | 11.18% | 38.06% | Large disagreement |
| trusted method-balanced median | TIGERBx 2024 FFE 401 | 4.91% | 18.20% | Close volume match |
| trusted leave-one-method-out | SynthSeg 2024 3DI | 20.38% | 42.55% | Severe disagreement |
| trusted leave-one-method-out | TIGERBx 2024 3DI | 24.76% | 65.14% | Severe disagreement |

## Interpretation

The result supports the current QC stance: 2024 3DI should not be accepted as a
recovered segmentation target by volume agreement alone. Both SynthSeg 2024 3DI
and TIGERBx 2024 3DI show large or severe disagreement against FFE-derived
pseudo-GT variants.

The strongest current 2024 volume candidate is TIGERBx 2024 FFE 401 under the
trusted source and method-balanced references. This is still not enough for
visualization promotion; visual overlay QC is required.

Leave-one-method-out is intentionally stricter than the method-balanced median.
It asks whether a method agrees with the other method family, not with a
consensus partly built from itself. For 2024 3DI, this stricter test produces
severe disagreement for both SynthSeg and TIGERBx.

Maximum relative errors are sometimes extreme because very small structures
produce unstable percent errors. The primary summary statistic for this stage is
the median error, with p90 used to flag structures needing visual inspection.

## Decision

- Keep `volume_v0` as the first computed pseudo-GT accuracy stage.
- Use trusted FFE-derived pseudo-GT for 2024 volume triage.
- Do not promote 2024 3DI volumes to `your-brain-mri-visualization`.
- Prepare the next stage: registered label maps in a subject-template space,
  then spatial pseudo-GT with Dice/HD95/surface metrics.
