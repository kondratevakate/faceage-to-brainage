# TIGERBx bmadq Application Report

Date: 2026-06-21

## Scope

This is an application first pass for the Kate n=1 MRI series, not a validation
claim. The goal was to test whether TIGERBx can produce usable brain extraction,
ASEG-like labels, and deep-gray labels on the difficult 2018/2022/2024 inputs
after the unchanged FS8 2024 3DI surface run was stopped.

## Run

Environment:

- WSL Ubuntu venv: `/home/kate/.venvs/tigerbx`
- TIGERBx install: `tigerbx[cpu]` from `htylab/tigerbx` tag `v0.2.3`
- CLI flags: `bmadq`
- GPU: no
- HLC/cortical thickness: not run

Output root:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\asian_mri_tools\tigerbx
```

Derived summaries:

```text
summary\tigerbx_label_volumes.csv
summary\tigerbx_scan_summary.csv
summary\tigerbx_pairwise_relative_differences.csv
summary\tigerbx_run_metadata.json
```

## Scan Summary

| Scan | Type | Labels | Total ml | QC |
|---|---:|---:|---:|---:|
| kate_2018_t1 | aseg | 43 | 1074.113 | 100 |
| kate_2018_t1 | dgm | 12 | 43.095 | 100 |
| kate_2022_t1 | aseg | 40 | 1102.116 | 100 |
| kate_2022_t1 | dgm | 12 | 38.079 | 100 |
| kate_2024_3di | aseg | 42 | 756.821 | 96 |
| kate_2024_3di | dgm | 12 | 35.415 | 96 |
| kate_2024_t1_ffe_401 | aseg | 42 | 1101.333 | 100 |
| kate_2024_t1_ffe_401 | dgm | 12 | 39.461 | 100 |
| kate_2024_t1_ffe_601 | aseg | 43 | 1095.913 | 97 |
| kate_2024_t1_ffe_601 | dgm | 12 | 40.894 | 97 |

## 2024 Consistency Check

| Output | Pair | Median label diff | Total diff |
|---|---|---:|---:|
| aseg | 3DI vs FFE 401 | 17.38% | 37.08% |
| aseg | 3DI vs FFE 601 | 17.93% | 36.60% |
| aseg | FFE 401 vs FFE 601 | 4.46% | 0.49% |
| dgm | 3DI vs FFE 401 | 12.57% | 10.81% |
| dgm | 3DI vs FFE 601 | 16.55% | 14.36% |
| dgm | FFE 401 vs FFE 601 | 10.52% | 3.57% |

## Interpretation

TIGERBx completed successfully and produced all expected first-pass files for
five inputs. The high tBET QC scores are useful runtime evidence, but they are
not sufficient output QC. The critical finding is that 2024 3DI has a much lower
ASEG total volume than both 2024 FFE alternatives despite QC=96. This means the
3DI result should be treated as suspicious until visual overlay QC proves
otherwise.

The two 2024 FFE inputs are much closer to each other in total ASEG volume
(0.49% difference), but their regional deep-gray differences remain non-trivial.
They are candidate rescue inputs, not accepted longitudinal measurements yet.

## Current Decision

- Keep TIGERBx `bmadq` as a completed first-pass application branch.
- Do not promote 2024 3DI TIGERBx volumes to visualization.
- Run visual QC overlays for brain mask, ASEG, and DGM before promoting any
  TIGERBx-derived 2024 measurements.
- Run OpenMAP-T1 or ReconAny/recon-all-clinical next if the goal is to test
  whether 2024 can be rescued by another contrast-tolerant method.
