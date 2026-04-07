# Brain Age Related Works

This page lists MRI-based brain-age systems that are relevant to the datasets and modalities used in this repository. Modality is explicit for every entry because it is the main practical constraint for `IXI`, `SIMON`, and any future extension of the project.

## Testable now

| Work | Modality | Training data | Preprocessing expectations | Fit with IXI | Fit with SIMON | Repo fit |
| --- | --- | --- | --- | --- | --- | --- |
| SFCN | T1w MRI only | Official repo: UK Biobank 14,503-subject release, with pretrained SFCN weight reported on 12,949 training subjects | Expects skull-stripped, conformed T1 volumes with model input shape `[1, 160, 192, 160]` | Yes for T1, after skull stripping and conforming | Potentially yes for T1, but current wrapper assumes skull-stripped input and should be treated cautiously | Existing wrapper in `src/brain_age.py`; decode logic is provisional |
| Wood et al. 2024 / MIDIconsortium BrainAge | Sequence-specific checkpoints for T1w, T2w, coronal FLAIR, DWI, SWI, sagittal T1 | Official repo states the clinical Wood models were trained on more than 23,000 axial T2 head MRIs from two large UK hospitals and exposes multiple inference branches in one repo | NIfTI input; reorients to LPS; some branches require skull stripping; T1 volumetric model uses HD-BET | Partial fit: T1 and T2 are directly relevant to IXI; FLAIR/DWI/SWI are not in IXI | Partial fit: only the T1 branch is likely relevant here | New wrapper needed; strong sequence-specific clinical baseline family |
| SynthBA | Cross-sequence T1w, T2w, FLAIR MRI with varying resolutions | Synthetic MRIs sampled from healthy-subject segmentations, with priors estimated from real MRIs across multiple sequences | Built-in preprocessing handles skull stripping and alignment unless skipped | Yes for T1 and T2; no PD branch | Yes for T1 in principle | New wrapper needed; strongest open protocol-robust candidate |
| BrainIAC | Multiparametric structural MRI; downstream brain-age task uses T1w | Nature paper: SSL pretraining on 32,015 MRIs, full study corpus 48,965 MRIs, and a brain-age benchmark built from 6,249 T1w scans | Uses its own preprocessing and downstream checkpoints; official repo provides checkpoint download path | Yes for T1; other IXI contrasts are not the published brain-age branch | Yes for T1 in principle | New wrapper needed; best open foundation-model context |

### SFCN

- Primary modality: T1-weighted MRI.
- Why it matters here: it is already wired into this repo and remains a classic open T1 brain-age baseline.
- Practical fit:
  - `IXI`: good fit for T1.
  - `SIMON`: only if the T1 input is exported in the expected form and skull-stripped.
- Caution: the current wrapper in `src/brain_age.py` should be treated as provisional until the age-bin decoding is verified.

### Wood et al. 2024 / MIDIconsortium BrainAge

- Primary modalities exposed in the official repo: volumetric T1, skull-stripped T2, coronal FLAIR, axial DWI, axial SWI, sagittal T1.
- Why it matters here: it is the most convenient open sequence-specific clinical baseline family.
- Practical fit:
  - `IXI`: T1 and T2 branches are relevant.
  - `SIMON`: T1 branch is the main candidate.
- Repo fit: not yet integrated, but the official repo already provides runnable inference code.

### SynthBA

- Primary modalities: T1w, T2w, FLAIR.
- Why it matters here: it directly targets cross-sequence and cross-resolution robustness, which is highly relevant for scanner and protocol variability.
- Practical fit:
  - `IXI`: T1 and T2 fit; PD does not match the published modality list.
  - `SIMON`: T1 should be usable in principle.
- Repo fit: attractive addition if this project wants a stronger protocol-agnostic brain-age branch than plain T1-only baselines.

### BrainIAC

- Primary modality for the published brain-age downstream task: T1-weighted MRI.
- Why it matters here: it is the strongest open foundation-model style context in the current shortlist, with public code and checkpoint instructions.
- Practical fit:
  - `IXI`: T1 branch is relevant.
  - `SIMON`: T1 branch is relevant in principle after conversion into the expected pipeline format.
- Repo fit: more integration work than SFCN or SynthBA, but it is now open enough to be considered genuinely testable.

## Promising but not openly runnable

| Work | Modality | Why it is promising | Why it is not in the testable bucket |
| --- | --- | --- | --- |
| Kim et al. 2025, routine clinical MRI scans | Axial 2D T1w MRI | Very strong clinical result for routine 2D T1, with MAE 2.73 years on clinical scans after bias correction | The paper states that code is available only upon reasonable request, so there is no confirmed public runnable release |
| Barbano et al. 2023, contrastive learning for regression | T1w structural MRI | Strong site-robust result on the OpenBHB multi-site challenge and public training code | No confirmed public pretrained weights were found from the official source |

### Kim et al. 2025, routine clinical MRI scans

- Primary modality: axial 2D T1-weighted MRI.
- Why it matters here: it is one of the clearest clinical brain-age papers aimed at routine, non-research acquisitions.
- Dataset note: the paper reports 8,681 research-grade 3D T1 scans from SMC plus 24 public datasets for model development, with clinical 2D T1 testing from SMC.
- Repo fit: interesting future comparator if the authors release code or checkpoints openly.

### Barbano et al. 2023, contrastive learning for regression

- Primary modality: T1-weighted structural MRI in the OpenBHB setting.
- Why it matters here: it specifically targets site robustness, which is directly relevant to scanner variability questions.
- Dataset note: the paper positions itself around the OpenBHB multi-site age-prediction challenge.
- Repo fit: the public code is useful, but without confirmed official pretrained weights it is not yet a practical drop-in baseline.

## Background note on SynthSeg

SynthSeg is important in this repository as a robust segmentation engine, but it is not listed here as a standalone brain-age model. In this project, it should be framed as a segmentation-to-regression pipeline component rather than a published end-to-end SOTA brain-age model by itself.

## Sources

- SFCN official repo: <https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain>
- MIDIconsortium BrainAge official repo: <https://github.com/MIDIconsortium/BrainAge>
- SynthBA official repo: <https://github.com/LemuelPuglisi/SynthBA>
- BrainIAC paper: <https://www.nature.com/articles/s41593-026-02202-6>
- BrainIAC official repo: <https://github.com/AIM-KannLab/BrainIAC>
- Kim et al. 2025 paper: <https://www.nature.com/articles/s41514-025-00260-x>
- Barbano et al. 2023 official repo: <https://github.com/EIDOSLAB/contrastive-brain-age-prediction>
