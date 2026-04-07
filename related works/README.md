# Related Works

This folder tracks the most relevant open face-age and brain-age models for this repository. The emphasis is practical: which models are open enough to test, which inputs they expect, and how directly they fit the MRI-to-face / MRI-to-brain pipeline used here.

## Current local support

- Current local code support is strongest for FaceAge, SFCN, and the SynthSeg-based volumetric pipeline.
- Treat the current SFCN wrapper in `../src/brain_age.py` as provisional until the age-bin decoding is verified.
- The labels below describe integration readiness for this repository, not paper quality alone.

## Overview

| Work | Task | Input / modality | Training data | Public access | Can test in this repo | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [FaceAge](face_age.md#faceage) | Face age / biological age | RGB face crop or portrait | Official repo describes a curated IMDb-Wiki-derived training set (56,304 age-labelled images) and curated UTK validation/demo data | Paper + official GitHub + weights via official Drive link | Yes, existing wrapper | Best match to the current MRI-rendered face PNG pipeline |
| [MiVOLO](face_age.md#mivolo) | Face age plus gender | RGB face-only or face+body photo | IMDB-cleaned, UTKFace, Lagenda | Paper + official GitHub + HuggingFace/Drive checkpoints | Yes, new wrapper needed | Use face-only checkpoints first; body-aware checkpoints are less relevant here |
| [SFCN](brain_age.md#sfcn) | Brain age | T1w MRI | UK Biobank 14,503-subject release; pretrained SFCN weight reported on 12,949 training subjects | Paper + official GitHub with pretrained weights | Yes, existing wrapper | Existing wrapper already present, but decode logic should be rechecked |
| [MIDIconsortium BrainAge](brain_age.md#wood-et-al-2024--midiconsortium-brainage) | Sequence-specific brain age | T1w, T2w, coronal FLAIR, DWI, SWI, sagittal T1 MRI | Official repo states the Wood clinical models were trained on more than 23,000 axial T2 head MRIs from two large UK hospitals and exposes multiple sequence-specific checkpoints | Papers + official GitHub repo with inference code | Yes, new wrapper needed | Strong clinical baseline family, especially relevant if non-T1 protocols enter the project |
| [SynthBA](brain_age.md#synthba) | Robust brain age across sequences/resolutions | T1w, T2w, FLAIR MRI | Synthetic MRIs generated from healthy-subject segmentations with priors estimated from real multi-sequence MRIs | Paper + official GitHub + packaged checkpoints | Yes, new wrapper needed | Most attractive open option for protocol-robust brain age |
| [BrainIAC](brain_age.md#brainiac) | Foundation-model brain age | Structural brain MRI; downstream brain-age benchmark uses T1w | SSL pretraining on 32,015 MRIs; full study corpus 48,965 MRIs; brain-age downstream benchmark built from 6,249 T1w scans | Paper + official GitHub + official checkpoint link | Yes, new wrapper needed | Most ambitious open foundation-model baseline in the current shortlist |
| [FAHR-Face / FAHR-FaceAge](face_age.md#fahr-face--fahr-faceage) | Foundation face-health / face-age | RGB face photographs | Large-scale face-photo pretraining and face-age fine-tuning described in the FAHR-Face preprint | Paper signal visible, but no confirmed official checkpoint path found during this review | No, no confirmed open weights | Promising follow-on from the FaceAge line |
| [Kim et al. 2025](brain_age.md#kim-et-al-2025-routine-clinical-mri-scans) | Clinical brain age | Axial 2D T1w MRI | 8,681 research-grade 3D T1 scans from SMC plus 24 public datasets; tested on clinical 2D T1 from SMC | Paper only; code available upon reasonable request | No, no confirmed open weights | Strong clinical result, but not openly runnable from public assets |
| [Barbano et al. 2023](brain_age.md#barbano-et-al-2023-contrastive-learning-for-regression) | Multi-site robust brain age | T1w MRI | OpenBHB / multi-site structural MRI challenge setting | Paper + official GitHub code, but no confirmed public pretrained weights | No, no confirmed open weights | Strong robustness result worth watching |

See [research_questions.md](research_questions.md) for a dated memo on chronological-age accuracy, gap interpretation, 2D-to-3D face reconstruction, and the practical value of non-defaced MRI cohorts.

## Face age

See [face_age.md](face_age.md) for details on which face-photo models are most plausible for MRI-rendered faces, and why FaceAge remains the primary baseline for this repository.

## Brain age

See [brain_age.md](brain_age.md) for MRI modality coverage, training-data provenance, preprocessing expectations, and a repo-specific fit assessment for each open brain-age candidate.

## Promising but not openly runnable

- `FAHR-Face / FAHR-FaceAge`: strong next-generation face-health direction, but this review did not confirm an official public checkpoint download path.
- `Kim et al. 2025`: excellent clinical 2D T1 result, but the paper states that code is available only upon reasonable request.
- `Barbano et al. 2023`: public training code and strong OpenBHB result, but no confirmed pretrained weights were found from the official source.
