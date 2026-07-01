# FastSurfer Long Symmetry Consistency Report

Date: 2026-06-14

Dataset: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years`

Cross-sectional FastSurfer: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\fastsurfer`

FastSurfer Long v2: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\fastsurfer_long_v2`

## Status

| stream | subject | aseg+DKT | HypVINN | surf done | surf error |
| --- | --- | --- | --- | --- | --- |
| cross | sym_rotpos | yes | yes | no | no |
| cross | sym_rotneg | yes | yes | no | no |
| long | sym_rotpos | yes | yes | yes | no |
| long | sym_rotneg | yes | yes | yes | no |
| long-base | sym_fast_base | yes | n/a | yes | no |

The wrapper log appears complete.

## Method

This compares the same 2018 T1 after opposite synthetic rotations (`sym_rotpos` vs `sym_rotneg`). Pairwise CV is:

`100 * abs(rotpos - rotneg) / (rotpos + rotneg)`

Lower CV means better repeatability under this rotation perturbation. It is an internal consistency / method-floor check, not proof of biological accuracy.

## Summary

| Metric family | n | cross median CV% | long median CV% | improved |
|---|---:|---:|---:|---:|
| Global aseg measures | 8 | 0.01 | 0.01 | 3/8 |
| Subcortical volumes | 12 | 0.24 | 0.12 | 10/12 |
| Cortical DKT parcel volumes | 62 | 0.36 | 0.10 | 53/62 |
| HypVINN volumes | 24 | 1.53 | 0.80 | 18/24 |

Long-only surface symmetry, because the available cross-sectional FastSurfer folders are segmentation-only:

| Surface metric | n | long median CV% | long max CV% |
|---|---:|---:|---:|
| ThickAvg | 62 | 0.22 | 2.22 |
| GrayVol | 62 | 0.27 | 1.70 |
| SurfArea | 62 | 0.20 | 1.64 |

## Subcortical Volumes

| Region | cross pos | cross neg | long pos | long neg | cross CV% | long CV% | delta | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Left-Hippocampus | 3916.8 | 3931.4 | 3860.8 | 3871.6 | 0.19 | 0.14 | -0.05 | 0.74x |
| Right-Hippocampus | 3987.5 | 4028.7 | 3972.7 | 3977.3 | 0.51 | 0.06 | -0.45 | 0.11x |
| Left-Amygdala | 1432.6 | 1433.7 | 1474.6 | 1470.4 | 0.04 | 0.14 | 0.11 | 3.77x |
| Right-Amygdala | 1562.9 | 1543.5 | 1592.8 | 1603.8 | 0.62 | 0.34 | -0.28 | 0.55x |
| Left-Thalamus | 6732.4 | 6792.3 | 6815.2 | 6825.3 | 0.44 | 0.07 | -0.37 | 0.17x |
| Right-Thalamus | 6272.7 | 6246.0 | 6332.7 | 6308.7 | 0.21 | 0.19 | -0.02 | 0.89x |
| Left-Caudate | 3354.8 | 3369.9 | 3406.0 | 3404.8 | 0.22 | 0.02 | -0.21 | 0.08x |
| Right-Caudate | 3391.7 | 3409.0 | 3456.6 | 3465.1 | 0.25 | 0.12 | -0.13 | 0.48x |
| Left-Putamen | 4815.6 | 4848.8 | 4897.5 | 4904.9 | 0.34 | 0.08 | -0.27 | 0.22x |
| Right-Putamen | 4831.6 | 4820.7 | 4872.7 | 4861.8 | 0.11 | 0.11 | -0.00 | 1.00x |
| Left-Pallidum | 1723.0 | 1717.6 | 1702.0 | 1716.9 | 0.16 | 0.44 | 0.28 | 2.82x |
| Right-Pallidum | 1690.4 | 1701.0 | 1646.3 | 1648.0 | 0.31 | 0.05 | -0.26 | 0.16x |

## Global Measures

| Measure | cross pos | cross neg | long pos | long neg | cross CV% | long CV% | delta | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BrainSegVol | 1050363.9 | 1050197.1 | 1053054.9 | 1053227.2 | 0.01 | 0.01 | 0.00 | 1.03x |
| BrainSegVolNotVent | 1034947.5 | 1034799.3 | 1037627.7 | 1037822.0 | 0.01 | 0.01 | 0.00 | 1.31x |
| SupraTentorialVol | 931348.6 | 931212.8 | 934052.5 | 934178.0 | 0.01 | 0.01 | -0.00 | 0.92x |
| SupraTentorialVolNotVent | 915932.1 | 915815.0 | 918625.3 | 918772.8 | 0.01 | 0.01 | 0.00 | 1.26x |
| SubCortGrayVol | 52751.6 | 52757.9 | 53031.1 | 53088.3 | 0.01 | 0.05 | 0.05 | 9.07x |
| lhCerebralWhiteMatterVol | 211347.4 | 211337.2 | 208489.9 | 208470.2 | 0.00 | 0.00 | 0.00 | 1.95x |
| rhCerebralWhiteMatterVol | 212739.9 | 212970.3 | 210267.6 | 210387.4 | 0.05 | 0.03 | -0.03 | 0.53x |
| CerebralWhiteMatterVol | 424087.3 | 424307.5 | 418757.5 | 418857.6 | 0.03 | 0.01 | -0.01 | 0.46x |

## Largest CV Improvements

| Region | cross pos | cross neg | long pos | long neg | cross CV% | long CV% | delta | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctx-rh-isthmuscingulate | 2369.9 | 2284.1 | 2318.8 | 2321.1 | 1.84 | 0.05 | -1.80 | 0.03x |
| ctx-rh-cuneus | 3500.8 | 3385.4 | 3421.1 | 3426.4 | 1.68 | 0.08 | -1.60 | 0.05x |
| ctx-rh-lingual | 5762.6 | 5542.0 | 5809.7 | 5761.9 | 1.95 | 0.41 | -1.54 | 0.21x |
| ctx-rh-entorhinal | 1448.2 | 1493.8 | 1503.2 | 1496.2 | 1.55 | 0.23 | -1.32 | 0.15x |
| ctx-rh-parahippocampal | 1819.9 | 1856.9 | 1864.5 | 1862.5 | 1.01 | 0.05 | -0.95 | 0.05x |
| ctx-rh-rostralanteriorcingulate | 1881.8 | 1934.9 | 1954.2 | 1935.8 | 1.39 | 0.47 | -0.92 | 0.34x |
| ctx-rh-pericalcarine | 1829.2 | 1780.4 | 1869.8 | 1852.7 | 1.35 | 0.46 | -0.89 | 0.34x |
| ctx-rh-parsorbitalis | 2458.3 | 2416.4 | 2492.3 | 2490.1 | 0.86 | 0.04 | -0.82 | 0.05x |

## Largest CV Worsenings

| Region | cross pos | cross neg | long pos | long neg | cross CV% | long CV% | delta | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctx-lh-rostralanteriorcingulate | 3345.5 | 3348.7 | 3384.2 | 3420.3 | 0.05 | 0.53 | 0.48 | 10.89x |
| ctx-lh-parsorbitalis | 1892.2 | 1892.7 | 1912.0 | 1895.7 | 0.01 | 0.43 | 0.42 | 32.01x |
| Left-Pallidum | 1723.0 | 1717.6 | 1702.0 | 1716.9 | 0.16 | 0.44 | 0.28 | 2.82x |
| ctx-rh-medialorbitofrontal | 4315.7 | 4350.8 | 4341.9 | 4391.9 | 0.41 | 0.57 | 0.17 | 1.41x |
| ctx-lh-pericalcarine | 1659.4 | 1665.5 | 1754.8 | 1742.9 | 0.18 | 0.34 | 0.16 | 1.86x |
| ctx-lh-caudalanteriorcingulate | 2983.1 | 2986.3 | 2974.2 | 2961.9 | 0.05 | 0.21 | 0.15 | 3.82x |
| ctx-rh-precuneus | 8743.8 | 8737.0 | 8919.8 | 8952.4 | 0.04 | 0.18 | 0.14 | 4.66x |
| Left-Amygdala | 1432.6 | 1433.7 | 1474.6 | 1470.4 | 0.04 | 0.14 | 0.11 | 3.77x |

## Files

- CSV table: `C:\Users\Lenovo\Documents\Codex\2026-06-11\prior-conversation-with-codex-conversation-role\outputs\fastsurfer_long_symmetry_volume_cv.csv`
- Main FastSurfer Long log: `D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\symmetry\logs\fastsurfer_long_symmetry_v2.log`

## HypVINN Volumes

| Region | cross pos | cross neg | long pos | long neg | cross CV% | long CV% | delta | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ant-Commisure | 267.7 | 271.8 | 277.4 | 279.8 | 0.76 | 0.43 | -0.33 | 0.57x |
| Epiphysis | 410.1 | 408.7 | 372.5 | 378.2 | 0.17 | 0.76 | 0.59 | 4.44x |
| Hypophysis | 931.9 | 918.4 | 938.2 | 925.9 | 0.73 | 0.66 | -0.07 | 0.90x |
| Infundibulum | 59.6 | 66.0 | 69.0 | 71.1 | 5.10 | 1.50 | -3.60 | 0.29x |
| L-Ant-Hypothalamus | 112.0 | 98.1 | 109.6 | 106.9 | 6.62 | 1.25 | -5.37 | 0.19x |
| L-C.mammilare | 105.6 | 110.1 | 109.7 | 111.4 | 2.09 | 0.77 | -1.32 | 0.37x |
| L-Chiasma-Opticum | 92.4 | 93.4 | 93.6 | 91.1 | 0.54 | 1.35 | 0.82 | 2.51x |
| L-Fornix | 118.2 | 131.3 | 119.6 | 120.5 | 5.25 | 0.37 | -4.88 | 0.07x |
| L-Lat-Hypothalamus | 42.3 | 43.1 | 46.3 | 45.6 | 0.94 | 0.76 | -0.18 | 0.81x |
| L-Med-Hypothalamus | 39.1 | 38.4 | 37.3 | 38.0 | 0.90 | 0.93 | 0.03 | 1.03x |
| L-N.opticus | 387.5 | 385.0 | 353.8 | 359.7 | 0.32 | 0.83 | 0.50 | 2.56x |
| L-Optic-tract | 318.8 | 328.6 | 314.9 | 323.6 | 1.51 | 1.36 | -0.15 | 0.90x |
| L-Post-Hypothalamus | 156.7 | 161.6 | 157.0 | 150.8 | 1.54 | 2.01 | 0.47 | 1.31x |
| R-Ant-Hypothalamus | 86.7 | 79.1 | 95.7 | 98.7 | 4.58 | 1.54 | -3.04 | 0.34x |
| R-C.mammilare | 112.5 | 113.5 | 97.0 | 96.5 | 0.44 | 0.26 | -0.18 | 0.58x |
| R-Chiasma-Opticum | 108.9 | 109.6 | 118.6 | 120.2 | 0.32 | 0.67 | 0.35 | 2.09x |
| R-Fornix | 125.9 | 111.4 | 125.5 | 129.0 | 6.11 | 1.38 | -4.74 | 0.23x |
| R-Lat-Hypothalamus | 53.8 | 56.2 | 52.5 | 50.8 | 2.18 | 1.65 | -0.54 | 0.75x |
| R-Med-Hypothalamus | 41.1 | 44.6 | 44.1 | 45.5 | 4.08 | 1.56 | -2.52 | 0.38x |
| R-N.opticus | 433.4 | 456.0 | 413.1 | 422.4 | 2.54 | 1.11 | -1.43 | 0.44x |
| R-Optic-tract | 259.9 | 254.6 | 246.5 | 248.2 | 1.03 | 0.34 | -0.69 | 0.33x |
| R-Post-Hypothalamus | 164.0 | 157.7 | 154.8 | 155.7 | 1.96 | 0.29 | -1.67 | 0.15x |
| Third-Ventricle | 856.0 | 843.7 | 857.6 | 852.3 | 0.72 | 0.31 | -0.41 | 0.43x |
| Tuberal-Region | 39.3 | 41.1 | 42.3 | 42.6 | 2.24 | 0.35 | -1.89 | 0.16x |

## Long-Only Surface Thickness

| Region | cross pos | cross neg | long pos | long neg | cross CV% | long CV% | delta | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rh-pericalcarine |  |  | 1.7 | 1.6 |  | 2.22 |  |  |
| lh-entorhinal |  |  | 2.8 | 2.7 |  | 1.32 |  |  |
| lh-pericalcarine |  |  | 1.7 | 1.7 |  | 0.89 |  |  |
| rh-rostralmiddlefrontal |  |  | 2.3 | 2.3 |  | 0.80 |  |  |
| lh-inferiorparietal |  |  | 2.4 | 2.5 |  | 0.63 |  |  |
| rh-paracentral |  |  | 2.6 | 2.6 |  | 0.55 |  |  |
| lh-paracentral |  |  | 2.6 | 2.6 |  | 0.53 |  |  |
| lh-precuneus |  |  | 2.4 | 2.4 |  | 0.51 |  |  |
| rh-posteriorcingulate |  |  | 2.5 | 2.5 |  | 0.50 |  |  |
| rh-transversetemporal |  |  | 2.7 | 2.7 |  | 0.50 |  |  |
| lh-transversetemporal |  |  | 2.5 | 2.5 |  | 0.48 |  |  |
| lh-supramarginal |  |  | 2.5 | 2.5 |  | 0.48 |  |  |
