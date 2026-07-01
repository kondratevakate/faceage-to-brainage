# BrainFM feature QC summary

Date: 2026-06-22

## Scope

This note summarizes BrainFM feature-only outputs for Kate n=1. It uses cosine distances between pooled BrainFM feature summaries from 11 local MRI inputs. The goal is QC and protocol-sensitivity review, not segmentation or morphometry validation.

Input feature matrix:

`D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\foundation_models\brainfm\brainfm_feature_summaries.csv`

Tracked summaries:

- `data/kate_n1_2026/brainfm_feature_scan_summary.csv`
- `data/kate_n1_2026/brainfm_feature_pairwise_distances.csv`
- `data/kate_n1_2026/brainfm_feature_contrast_summary.csv`

## Main QC Signals

Across all 55 scan pairs, full-vector cosine distance ranged from `0.012553747` to `0.188394872`, with median `0.074258036`.

The closest pairs were dominated by FLAIR/T2-type contrasts:

- `kate_2022_flair` vs `kate_2022_t2_tra`: `0.012553747`
- `kate_2024_flair` vs `kate_2024_t2_501`: `0.018433838`
- `kate_2022_t2_tra` vs `kate_2024_t2_501`: `0.023172567`

The farthest pairs were dominated by the 2022 sagittal T1 SE scan:

- `kate_2022_t1` vs `kate_2022_flair`: `0.188394872`
- `kate_2022_t1` vs `kate_2022_t2_tra`: `0.187545834`
- `kate_2018_t1` vs `kate_2022_t1`: `0.185205257`

This is consistent with the existing concern that the 2022 T1 SE thick-slice protocol behaves differently from higher-resolution 3D-like inputs. It is a feature-space QC signal, not proof that a given segmentation is wrong or right.

## 2024 Candidate Checks

Same-session alternative 2024 checks were relatively low in this feature space:

- `kate_2024_t2_501` vs `kate_2024_t2_801`: `0.033447688`
- `kate_2024_t1_ffe_401` vs `kate_2024_t1_ffe_601`: `0.043883070`
- `kate_2024_3di` vs 2024 FFE alternatives: median `0.057413098`

The 2024 3DI-vs-FFE feature distances are lower than the primary/probe trio median (`0.175863450`) and much lower than the 2022 T1-vs-secondary median (`0.187545834`). This supports using BrainFM features as a QC lens for 2024 candidate comparison. It does not rescue the 2024 3DI segmentation failure modes already observed in FS/FastSurfer-style pipelines.

## Guardrails

BrainFM embeddings summarize model feature activations after preprocessing. They do not directly measure anatomical boundary accuracy, label correctness, surface topology, cortical thickness validity, or longitudinal biological change. A low feature distance can reflect shared contrast/preprocessing behavior rather than correct anatomy. A high feature distance can reflect acquisition differences rather than pathology or segmentation failure.

The next use of these outputs should be limited to ranking QC priorities and protocol-sensitivity contrasts before visual overlays and cross-method checks.
