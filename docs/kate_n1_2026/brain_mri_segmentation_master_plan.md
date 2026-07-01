# Brain MRI Segmentation: Master Method Matrix and Execution Plan

Date: 2026-06-16

This file is the project index for the local brain MRI segmentation comparison.
It does not copy or modify heavy imaging outputs. It records what is already
computed, what is missing, where runs failed, and what should be run next.

## Roots and Existing Artifacts

Data root:

`D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years`

Reprocessed root:

`D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026`

Current Codex output reports:

- `outputs/fs_long_consistency_report.md`
- `outputs/fastsurfer_long_symmetry_report.md`
- `outputs/fastsurfer_long_symmetry_volume_cv.csv`

Work scripts already present:

- `work/run_fs_long_2018_2022.sh`
- `work/run_fastsurfer_long_symmetry.sh`
- `work/compare_fastsurfer_long_symmetry.py`

Git status:

No Git repository was detected in the current Codex workspace or under the main
YandexDisk data roots. Current results are local/YandexDisk artifacts, not a
GitHub project. If a repository is created later, commit reports, scripts, logs
manifests, and small CSV summaries only. Do not commit `.nii`, `.nii.gz`, `.mgz`,
or full FreeSurfer/FastSurfer subject folders.

## Interpretation Rule

"Improved" means one of the following:

- lower pairwise CV under a controlled rotation perturbation;
- lower longitudinal scan-to-scan CV;
- successful completion where another stream failed.

It does not prove biological accuracy. There is no manual ground-truth
segmentation in this project yet. Visual QC remains mandatory for any claim that
a method is usable for anatomy, not only numerically stable.

## Pseudo-Ground-Truth as MRI Multi-View Semantic Fusion

This project should not claim a true ground-truth segmentation without expert
manual labels, physical phantom data, or histology. The correct target is a
subject-specific **probabilistic consensus reference** with explicit
uncertainty.

The working analogy is borrowed from 3D perception:

| 3D perception concept | MRI application concept |
|---|---|
| Multiple RGB-D frames or camera views | Multiple MRI scans, contrasts, acquisitions, and segmentation methods |
| Camera pose / SLAM | Registration into a common subject/template space |
| Semantic fusion map | Consensus probabilistic segmentation map |
| View disagreement / occlusion uncertainty | Voxel-wise label disagreement, method uncertainty, acquisition/contrast uncertainty |

In this framing, each scan/method pair is one noisy view of the same anatomy.
Registration estimates how those views align. Consensus segmentation fuses their
labels into a probabilistic semantic map, not a hard anatomical truth.

Required safeguards:

- compare every method against a **leave-one-method-out** consensus that excludes
  the method being evaluated;
- keep hard labels and probability/entropy maps separately;
- report Dice/Jaccard, relative volume error, HD95/ASSD or surface Dice, and
  scan-to-scan CV where labels overlap;
- do not let a high runtime/QC score replace visual overlay QC;
- treat TTA/rotation/resampling consensus as a method-stability estimate, not as
  anatomical ground truth;
- use left-right reflection only as a QC prior for gross failures, because real
  brain anatomy can be asymmetric.

Near-term implementation target:

1. Build `pseudo_gt_v0` for overlapping subcortical/ASEG labels only.
2. Use only QC-passing sources; exclude known failed 2024 3DI outputs.
3. Register accepted label maps to an unbiased subject-template space.
4. Build majority-vote and STAPLE-style probabilistic consensus variants.
5. Score each source with leave-one-method-out comparisons.
6. Promote only consensus-derived small CSV/QC summaries to
   `your-brain-mri-visualization` after visual QC.

## Inputs Used So Far

Main T1-like inputs:

| Year | Scanner/protocol label | File | Notes |
|---|---|---|---|
| 2018 | GE 3T FSPGR/BRAVO, 1 mm-like | `images\2018\nifti\3_fspgr_bravo_10mm_ax.nii.gz` | Main high-quality anatomical scan. |
| 2022 | Siemens 1.5T T1 SE sagittal, thick slices | `images\2022\nifti\4_t1_se_sag.nii.gz` | Low through-plane resolution; major driver of longitudinal instability. |
| 2024 | Philips 1.5T 3D IR / 3DI | `images\2024\nifti\901_3di_mc_hr.nii.gz` | High resolution but different contrast; causes failures in some T1-trained methods. |

Additional 2024 contrast inputs already segmented by SynthSeg include FLAIR,
T1 FFE axial/sagittal, T2 TSE axial/coronal.

## Method Matrix

| Branch | Scope | Status | Main output | Current result | Next action |
|---|---|---|---|---|---|
| SynthSeg native cross-scanner | 2018, 2022, 2024 and 2024 multi-contrast | Done | `reprocessed_2026/summary.md`, `reprocessed_2026/seg`, `reprocessed_2026/vol` | Cross-scanner median spread is much larger than rotation floor; 2024 3DI has QC flag in general gray matter. | Keep as contrast-robust volumetry baseline; add visual QC snapshots. |
| SynthSeg rotation and TTA | 2018 rotation pair and 9-angle sweep | Done | `reprocessed_2026/symmetry/seg*`, `vol*`, `summary.md` | Median +/-3 deg floor about 1.43%; 9-angle TTA CV about 1.24%; scanner/protocol spread about 12x the method floor. | Convert the existing tables into a standalone rotation-stability report. |
| FastSurfer rotation | 2018 +/-3 deg pair | Done as raw/summary result | `reprocessed_2026/symmetry/fastsurfer`, `summary.md` | Median rotation floor about 1.48%, essentially tied with SynthSeg, but different structures fail. | Make a standalone FastSurfer rotation report; do not rerun unless commands need verification. |
| FastSurfer Long rotation | 2018 +/-3 deg pair with longitudinal base | Done | `outputs/fastsurfer_long_symmetry_report.md` | Subcortical median CV improved 0.24% to 0.12%; DKT cortical volume CV improved 0.36% to 0.10%; HypVINN improved 1.53% to 0.80%. | Add visual QC for base and long timepoints; include in final method-stability conclusion. |
| FreeSurfer 7.4.1 longitudinal | 2018 + 2022 | Done | `outputs/fs_long_consistency_report.md` | Cortical thickness/volume stabilized strongly; subcortical result mixed because 2022 is a 5 mm scan. 2024 excluded. | Add visual QC for hippocampus, putamen, thalamus, brainmask, white/pial surfaces. |
| FreeSurfer 8.2.0 cross + long | 2018, 2022; 2024 as rescue cross-run | Running local pull/job | `reprocessed_2026/fs82`, `reprocessed_2026/logs_fs82` | Official stable FreeSurfer is 8.2.0 as of 2026-06-16. Local WSL run started with 8GB RAM + 64GB swap and `threads=2`. | Monitor local run; move to cloud/high-RAM if it thrashes, hits OOM, or becomes impractically slow. |
| ReconAny | 2018, 2022, 2024, especially heterogeneous 2024/2022 | Not run | Planned: `reprocessed_2026/reconany` | Official page describes it as a dev FreeSurfer stream for adult 3D scans with arbitrary orientation/resolution/contrast. | Run after obtaining a dev FreeSurfer build/container; compare whether it rescues 2024 and handles 2022 better. |
| recon-all-clinical | 2022/2024 clinical-style scans | Not run | Planned: `reprocessed_2026/recon_all_clinical` | Available since FS 7.4; targets arbitrary modality/contrast/resolution. Thickness may degrade with thick slices. | Use as practical fallback if ReconAny dev build is not available. Apply script bug fix if using FS 7.4. |
| Harmonized preprocessing | Image-domain and metric-domain harmonization | Not run | Planned: `reprocessed_2026/harmonized` | Needed to separate acquisition effects from segmentation algorithm effects. | Build a locked preprocessing branch; never overwrite raw inputs or existing subject folders. |
| SIAM / Segment It All Model | 2018, 2022, 2024 heterogeneous scans; especially 2024 3DI vs FFE | Recorded, not run | Planned: `reprocessed_2026/siam` or `reprocessed_2026/siamize` | Emerging whole-head/tissue segmentation candidate trained synthetically from few high-quality templates; relevant for contrast-robust tissue/head labels, not a direct DKT/ASEG replacement. | Prefer a guarded `siamize` smoke run on one 2024 FFE candidate after RAM/GPU feasibility check; map label ontology before pseudo-GT or TTA scoring. |

## Current Stability Conclusions

1. Cross-scanner/protocol variation is the dominant problem. The SynthSeg
   rotation floor is about 1.4%, while the cross-scanner spread is about 16.7%
   median in the existing summary, roughly a 12x difference.
2. Orientation sensitivity is real for both SynthSeg and FastSurfer. Their
   median rotation floors are similar, but structure-level error patterns are
   not the same.
3. FastSurfer Long improves rotation repeatability for the controlled 2018
   rotation pair. This is a stability result, not an accuracy result.
4. FreeSurfer 7.4.1 Long improves cortical metrics for 2018 vs 2022, but does
   not universally improve subcortical metrics because the 2022 scan lacks
   through-plane resolution.
5. The 2024 scan is the stress test. FS7 failed Talairach on it, and ordinary
   FastSurfer segmentation collapsed on the 2024 3DI contrast. This is exactly
   why ReconAny/recon-all-clinical and harmonization are needed.

## Failure Notes

| Method | Scan/branch | Stage | Symptom | Interpretation | Action |
|---|---|---|---|---|---|
| FreeSurfer 7.4.1 | 2024 3DI | Talairach | `talairach_afd ... FAILED`; `ERROR: Talairach failed!` | Atlas-based FS7 stream cannot robustly handle this contrast/acquisition as run. | Do not include 2024 in FS7 longitudinal base; test FS8, ReconAny, recon-all-clinical. |
| FastSurfer ordinary | 2024 3DI | Segmentation QC | Warning that total segmentation volume is too small; BrainSeg about 167 mL in earlier summary. | Segmentation collapsed; numeric labels are not usable. | Treat as failure, not as volume data. Test clinical/contrast-robust streams. |
| FreeSurfer 7.4.1 Long | 2022 | Longitudinal comparison | Cortical metrics improve, but hippocampus/putamen and some subcortical CVs worsen. | Longitudinal template cannot recover missing slice-direction information. | Report as mixed; visual QC required before biological interpretation. |
| FastSurfer Long first attempt | Rotation branch | Runtime/setup | Root/UID and `--tpids` naming issues; first output folder not the final result. | Execution issue, not method failure. | Use only `fastsurfer_long_v2` and its report. |
| FreeSurfer 8.2.0 local WSL | Planned | Memory | Current WSL memory is far below practical FS8 recommendation. | Local run likely to fail or thrash. | Run on cloud/high-RAM VM, one subject at a time. |
| ReconAny | Planned | Availability | Official page says dev FreeSurfer version. | Need exact dev build/container before running. | Verify install path/container first, then run. |

## Next Execution Order

### 1. Finish local reporting from existing outputs

No heavy compute needed.

- Create a standalone `fastsurfer_rotation_report.md` from the existing
  FastSurfer rotation tables in `reprocessed_2026/summary.md`.
- Create a standalone `synthseg_rotation_tta_report.md` from the existing
  SynthSeg rotation/TTA tables.
- Create a compact `failure_notes.csv` and `method_status_matrix.csv` for paper
  or slide reuse.

### 2. Visual QC of completed runs

QC targets:

- FS7 Long: `2018.long.kate_base`, `2022.long.kate_base`
- FastSurfer Long v2: `sym_fast_base`, `sym_rotpos`, `sym_rotneg`
- Failed 2024 outputs: FS7 `2024_FAILED_talairach`, FastSurfer `2024_phi_3di`

Structures to inspect first:

- hippocampus;
- putamen;
- thalamus;
- pallidum;
- amygdala;
- brainmask;
- white and pial surfaces;
- cortical ribbon near temporal pole and entorhinal cortex.

### 3. Run FreeSurfer 8.2.0 on high-RAM machine

Local fallback was started on 2026-06-16 after increasing WSL swap to 64 GB:

- `.wslconfig`: `memory=8GB`, `swap=64GB`,
  `swapFile=D:\WSL\swap\swap.vhdx`
- script: `work/run_fs82_local.sh`
- status script: `work/check_fs82_status.sh`
- output root: `reprocessed_2026/fs82`
- logs: `reprocessed_2026/logs_fs82`
- run policy: one subject at a time, `threads=2`

This is a pragmatic fallback, not the preferred performance setup. A high-RAM
cloud VM remains the safer route if the local run starts swapping heavily.

Recommended run policy:

- use cloud/high-RAM VM;
- run one subject at a time first;
- use separate output root, for example `reprocessed_2026/fs82`;
- keep FS7 and FS8 outputs strictly separated;
- hash inputs, license, image/build, and final stats.

Planned subjects:

```text
fs82/2018
fs82/2022
fs82/2024_cross_probe
fs82/kate_fs82_base
fs82/2018.long.kate_fs82_base
fs82/2022.long.kate_fs82_base
```

Initial command shape after FS8.2 is installed and sourced:

```bash
export FS_LICENSE=/path/to/license.txt
export SUBJECTS_DIR=/data/reprocessed_2026/fs82

recon-all -all -s 2018 -i /data/images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz -threads 8
recon-all -all -s 2022 -i /data/images/2022/nifti/4_t1_se_sag.nii.gz -threads 8

# Cross-run only as a rescue/probe first. Do not add to the longitudinal base
# unless it passes completion and visual QC.
recon-all -all -s 2024_cross_probe -i /data/images/2024/nifti/901_3di_mc_hr.nii.gz -threads 8

recon-all -base kate_fs82_base -tp 2018 -tp 2022 -all -threads 8
recon-all -long 2018 kate_fs82_base -all -threads 8
recon-all -long 2022 kate_fs82_base -all -threads 8
```

Comparison against FS7:

- `aseg.stats` subcortical volumes;
- `lh.aparc.stats` and `rh.aparc.stats` thickness and cortical volumes;
- eTIV/sTIV changes;
- completion and error markers;
- visual QC.

### 4. Run ReconAny or recon-all-clinical

Goal:

- test whether clinical/heterogeneous-scan streams rescue 2024;
- test whether 2022 thick-slice scan gets more plausible surfaces/parcels;
- separate "contrast robustness" from standard T1-MPRAGE assumptions.

ReconAny command shape, after a dev FreeSurfer build with `run_recon-any` is
installed and sourced:

```bash
export FS_LICENSE=/path/to/license.txt
export SUBJECTS_DIR=/data/reprocessed_2026/reconany

run_recon-any /data/images/2018/nifti/3_fspgr_bravo_10mm_ax.nii.gz 2018_reconany 8 both "$SUBJECTS_DIR"
run_recon-any /data/images/2022/nifti/4_t1_se_sag.nii.gz 2022_reconany 8 both "$SUBJECTS_DIR"
run_recon-any /data/images/2024/nifti/901_3di_mc_hr.nii.gz 2024_3di_reconany 8 both "$SUBJECTS_DIR"
```

recon-all-clinical fallback command shape:

```bash
export FS_LICENSE=/path/to/license.txt
export SUBJECTS_DIR=/data/reprocessed_2026/recon_all_clinical

recon-all-clinical.sh /data/images/2022/nifti/4_t1_se_sag.nii.gz 2022_clinical 8 "$SUBJECTS_DIR"
recon-all-clinical.sh /data/images/2024/nifti/901_3di_mc_hr.nii.gz 2024_3di_clinical 8 "$SUBJECTS_DIR"
```

QC rule:

If a stream completes but cortical thickness is unstable on thick slices, report
parcellation/volumetry separately from thickness.

### 5. Harmonization Branch

Create a new output root:

`reprocessed_2026/harmonized`

Image-domain preprocessing candidates:

- canonical orientation/conformation, without overwriting raw images;
- N4 bias correction;
- SynthStrip brain extraction;
- SynthSR 1 mm synthetic T1-like reconstruction for thick/clinical scans;
- SynthMorph or similar registration to a common subject/template space;
- intensity normalization or histogram matching, applied in a locked recipe.

Metric-domain harmonization candidates:

- eTIV or sTIV normalization for volumes;
- left/right and structure-to-brain ratios;
- within-subject percent change relative to 2018;
- robust z-scores against the subject's own method distribution;
- ComBat-style scanner harmonization only if a larger reference cohort exists.

Do not mix harmonized and native results in the same subject folders. Every
harmonized output should encode the preprocessing recipe in its path or manifest.

## Deliverables Still Missing

| Deliverable | Status | Source |
|---|---|---|
| Standalone SynthSeg rotation/TTA report | Missing | `reprocessed_2026/summary.md`, symmetry CSVs |
| Standalone FastSurfer rotation report | Missing | `reprocessed_2026/summary.md`, `symmetry/fastsurfer` |
| FS8.2 cross + long report | Missing | Needs cloud/high-RAM run |
| ReconAny/recon-all-clinical report | Missing | Needs dev/clinical stream run |
| Harmonization protocol report | Missing | Needs locked preprocessing plan and run |
| Visual QC image set | Missing | Needs FreeView/fsxvfb or equivalent screenshot workflow |
| Final method-stability conclusion | Partial | Requires standalone reports plus visual QC |
| Final scan-to-scan consistency conclusion | Partial | Requires FS8/ReconAny comparison if included |

## Official References Checked

- FreeSurfer Release Notes: stable 8.2.0 release dated 2026-03-18; FS8
  `recon-all` differs from FS7 and uses newer DL components.
  https://surfer.nmr.mgh.harvard.edu/fswiki/ReleaseNotes
- FreeSurfer 8.2 setup notes: 8.X `recon-all` is faster but can require much
  more memory; official notes mention around 80 GB at some point and recommend
  high-memory conditions.
  https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads/rel8notes
- ReconAny: `run_recon-any` is a dev FreeSurfer recon-all-like stream for adult
  3D volumes of arbitrary orientation, resolution, and contrast.
  https://surfer.nmr.mgh.harvard.edu/fswiki/ReconAny
- recon-all-clinical: available since FreeSurfer 7.4 and intended for arbitrary
  modality/contrast/resolution clinical scans; thickness degrades with larger
  slice spacing.
  https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all-clinical
