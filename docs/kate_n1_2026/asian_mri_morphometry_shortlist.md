# Asian MRI Morphometry / Segmentation Tools

Date: 2026-06-18, updated 2026-06-21

## Decision

The next locally reproducible method to try is **TIGERBx**. It is open-source,
has a direct Python/CLI interface, uses downloadable ONNX models, and produces
outputs that can be compared to the existing FreeSurfer/FastSurfer/SynthSeg
branches: brain mask, ASEG-like labels, deep gray matter labels, HLC labels,
cortical thickness, and tissue probability maps.

OpenMAP-T1 is the next candidate after TIGERBx, but it requires a manual
download/application step for the OpenMAP-T1 v3 pretrained model. It should be
tracked now and run later when the model folder is present.

## Runnable Now or Soon

| Method | Status | Why it matters | Current blocker |
|---|---|---|---|
| TIGERBx | First-pass `bmadq` run completed | Taiwan-affiliated open research toolkit; closest local target for a fast FreeSurfer-like stress test. | Needs visual QC before any visualization promotion; HLC/thickness not run yet. |
| OpenMAP-T1 | Pipeline prepared, not runnable yet | Japan/Hosei + OishiLab 280-region T1 parcellation; fast whole-brain parcellation. | Manual v3 model download and separate Python/PyTorch environment. |

## Review Only / Not Locally Reproducible

| Method | Region | Why not in local pipeline now |
|---|---|---|
| Neurophet AQUA | Korea | Commercial/FDA-approved tool; papers compare it with FreeSurfer, but no open local executable. |
| VUNO Med-DeepBrain | Korea | Commercial clinical software for volumetry, WMH, cortical thickness; no reproducible local pipeline without license. |
| Inbrain / MIDAS IT | Korea | Commercial; useful comparator in papers, not locally runnable here. |
| AccuBrain | Hong Kong/CUHK ecosystem | Commercial/cloud brain volumetry; useful in literature review, not local reproducible code. |

## Scientific Caveats

TIGERBx and OpenMAP-T1 are not one-to-one FreeSurfer replacements. FreeSurfer
reconstructs surfaces and has mature longitudinal modeling; TIGERBx/OpenMAP-T1
are deep-learning segmentation/parcellation pipelines. The fair comparison is:

1. Can they produce usable labels on the difficult 2022 and 2024 scans?
2. Are their label volumes more stable across scans than SynthSeg/FastSurfer/FS?
3. Do they fail in different places than FS8 topology correction?
4. Do their QC scores or extracted masks flag the 2024 failure mode earlier?

For this n=1 project, faster output is not evidence of better anatomy. Treat
TIGERBx/OpenMAP-T1 as independent stress tests and compare against visual QC and
the already computed cross-scan stability metrics.

## Local Commands

TIGERBx:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
bash experiments/kate_n1_2026/run_tigerbx_local.sh
python experiments/kate_n1_2026/summarize_tigerbx_bmad.py \
  --volume-csv /mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/asian_mri_tools/tigerbx/summary/tigerbx_label_volumes.csv \
  --qc-dir /mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/asian_mri_tools/tigerbx/bx \
  --scan-summary-csv /mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/asian_mri_tools/tigerbx/summary/tigerbx_scan_summary.csv \
  --pairwise-csv /mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/asian_mri_tools/tigerbx/summary/tigerbx_pairwise_relative_differences.csv
```

OpenMAP-T1:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
bash experiments/kate_n1_2026/run_openmap_t1_local.sh
```

Both launchers intentionally refuse to run while FS8 is active and while free
disk is below their default guard threshold. The completed TIGERBx run used
`ALLOW_LOW_DISK=1` after FS8 2024 was stopped and current resource state was
checked.
