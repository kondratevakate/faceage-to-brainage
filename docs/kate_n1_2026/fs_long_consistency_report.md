# FreeSurfer Longitudinal Consistency Report

Date: 2026-06-12
Dataset: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years`
SUBJECTS_DIR: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\fs_long`

## Status

The required 2018+2022 FreeSurfer 7.4.1 longitudinal chain completed successfully.

| Subject | done | error | aseg.stats | aparc stats |
|---|---:|---:|---:|---:|
| 2018 | yes | no | yes | yes |
| 2022 | yes | no | yes | yes |
| kate_base | yes | no | yes | yes |
| 2018.long.kate_base | yes | no | yes | yes |
| 2022.long.kate_base | yes | no | yes | yes |

`2024_FAILED_talairach` remains excluded and was not used for the base.

## Runtime

| Stage | Started UTC | Ended UTC | Runtime |
|---|---|---|---:|
| kate_base | 2026-06-11 14:26:50 | 2026-06-11 19:43:27 | 5.277 h |
| 2018.long.kate_base | 2026-06-11 19:43:30 | 2026-06-11 21:42:12 | 1.978 h |
| 2022.long.kate_base | 2026-06-11 21:42:15 | 2026-06-11 23:41:22 | 1.985 h |

All three logs end with `finished without error`.

## Environment Manifest

Docker image:

`freesurfer/freesurfer@sha256:10b6468cbd9fcd2db3708f4651d59ad75d4da849a2c5d8bb6dba217f08b8c46b`

Input and key-output SHA256:

| Item | SHA256 |
|---|---|
| input_2018_nii | `23C62DBEB732B38D8443A0AF270A63F164198E959185AB28FC8D807EF0AD69FC` |
| input_2022_nii | `89AD926292F8314BE65C3EA726E82980168FA689D383CE0B1F7DC7DC954540D0` |
| fs_license | `97BD4C29FBE416BFE5E78F290280061130995D80DD6EDCE670C2F82407AAB536` |
| aseg_2018_cross | `751285C332C418059DE611BEAE85A525B49882BFEEA441F676C6D02C17E497B9` |
| aseg_2022_cross | `724607E57D08B73D1041DBBE2D33C907A7ADA3714E7120CF3CBB3BF7205143A9` |
| aseg_kate_base | `D0715D94B76AF4561B755F33289888E39DD38730F189212AF8FC77E0257AB099` |
| aseg_2018_long | `C9C557C6990F5422447C27BC0AEAAB28BA5A41F81E715EE392D1E555291AFE54` |
| aseg_2022_long | `56B7656BDBFB17B33B2C273A1626D792F5DAD153E97D34023F350BB671B75A42` |

## Table 10: Subcortical CV

CV is across 2018 and 2022. Lower longitudinal CV is better.

| Region | cross 2018 | cross 2022 | long 2018 | long 2022 | cross CV% | long CV% | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| L Hippocampus | 3.85 | 4.05 | 3.90 | 4.46 | 2.51 | 6.76 | 2.69x |
| R Hippocampus | 3.74 | 3.76 | 3.71 | 4.17 | 0.22 | 5.72 | 26.02x |
| L Amygdala | 1.56 | 1.52 | 1.47 | 1.61 | 1.18 | 4.43 | 3.74x |
| R Amygdala | 1.70 | 1.60 | 1.68 | 1.75 | 3.16 | 1.85 | 0.59x |
| L Thalamus | 6.28 | 11.51 | 6.78 | 8.83 | 29.43 | 13.12 | 0.45x |
| R Thalamus | 5.68 | 7.72 | 5.94 | 7.33 | 15.20 | 10.44 | 0.69x |
| L Caudate | 3.28 | 2.32 | 3.57 | 2.58 | 17.17 | 15.99 | 0.93x |
| R Caudate | 3.40 | 2.10 | 3.62 | 2.45 | 23.61 | 19.33 | 0.82x |
| L Putamen | 4.71 | 4.95 | 5.06 | 6.24 | 2.50 | 10.48 | 4.20x |
| R Putamen | 4.69 | 4.83 | 4.70 | 5.89 | 1.50 | 11.22 | 7.47x |
| L Pallidum | 1.52 | 2.09 | 1.58 | 1.34 | 15.90 | 8.20 | 0.52x |
| R Pallidum | 1.65 | 2.40 | 1.77 | 2.34 | 18.39 | 13.66 | 0.74x |

Median cross-sectional CV: 9.18%
Median longitudinal CV: 10.46%

Interpretation: the longitudinal stream improved 7/12 subcortical regions, especially thalamus and pallidum, but did not reduce the median subcortical CV because hippocampus and putamen worsened. This is likely dominated by the 2022 5 mm scan; the template cannot recover missing through-plane resolution.

## Cortical Thickness Summary

| Metric | cross 2018 | cross 2022 | long 2018 | long 2022 | cross CV% | long CV% | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| lh mean thickness mm | 2.520 | 2.041 | 2.622 | 2.922 | 10.49 | 5.41 | 0.52x |
| rh mean thickness mm | 2.519 | 2.135 | 2.626 | 2.940 | 8.24 | 5.65 | 0.68x |
| BrainSegVol mL | 1075.4 | 1164.5 | 1074.5 | 968.6 | 3.98 | 5.18 | 1.30x |
| eTIV mL | 1418.5 | 1429.6 | 1448.9 | 1448.9 | 0.39 | 0.00 | 0.00x |
| CortexVol mL | 476.5 | 309.9 | 486.1 | 468.2 | 21.19 | 1.88 | 0.09x |

Regional aparc thickness:

| Metric | Value |
|---|---:|
| regions | 68 |
| median cross regional thickness CV | 10.50% |
| median longitudinal regional thickness CV | 5.83% |
| improved regions | 43/68 |

Interpretation: the longitudinal stream clearly stabilizes cortical thickness and cortical gray-matter volume, even though the subcortical median does not improve.

## How To Compare Two Computers

Do not compare whole FreeSurfer subject folders byte-for-byte as the primary check: logs, timestamps, and some generated files can differ. Compare in layers:

1. Same inputs: hash the NIfTI files and license.
2. Same container: compare the Docker image digest.
3. Same commands: compare `recon-all` command lines, timepoints, base ID, and thread count.
4. Completion: require `scripts/recon-all.done` and no `scripts/recon-all.error`.
5. Numeric outputs: compare `stats/aseg.stats`, `lh.aparc.stats`, and `rh.aparc.stats`.
6. Visual QC: inspect `brainmask.mgz`, `aseg.mgz`, white/pial surfaces for 2018 and especially the 2022 5 mm scan.

For a true machine A vs machine B comparison, keep two separate folders, for example:

```text
fs_long_compA/
fs_long_compB/
```

Then compare the same subjects:

```text
2018
2022
kate_base
2018.long.kate_base
2022.long.kate_base
```

The current disk only contains one active `fs_long` folder, so this report validates the completed handoff and current outputs, but it is not a two-independent-rerun reproducibility comparison.

## Next Calculations

1. Add Table 10 and the cortical thickness summary to `reprocessed_2026/summary.md`.
2. Do visual QC for `2018.long.kate_base` and `2022.long.kate_base`, focusing on hippocampus, putamen, thalamus, and cortical surfaces.
3. For the talk/manuscript: report that FreeSurfer longitudinal improves cortical measures strongly, improves some deep-gray structures, but does not universally reduce subcortical CV under the 2018 1 mm vs 2022 5 mm mismatch.
4. Do not use `2024_FAILED_talairach` in this FS7.4 longitudinal base.
5. Optional next experiment: FastSurfer longitudinal and/or symmetry-pair longitudinal runs, if the goal is to compare method-variance floors across methods.
