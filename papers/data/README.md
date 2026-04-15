# Data

All data directories are gitignored. This file describes how to obtain each dataset.

---

## SIMON

Single-subject MRI dataset: 1 healthy male, 73 sessions, 36 scanners.

- **Already local** — FreeSurfer-processed `.mgz` files in `data/simon/`
- Expected filename pattern: `simon_freesurfer<version>_ses-<NNN>_mri_orig.mgz`
- Not defaced → full facial soft-tissue present in each scan

---

## IXI

Cross-sectional healthy brain MRI: ~580 subjects, ages 20–86, 3 sites.

**Download from**: https://brain-development.org/ixi-dataset/

Files needed:
```
data/ixi/
├── IXI.xls           # demographic spreadsheet (age, sex, site)
├── T1/               # IXI-T1.tar extracted → IXI???-<site>-????-T1.nii.gz
├── T2/               # IXI-T2.tar extracted → IXI???-<site>-????-T2.nii.gz  [notebook 04 only]
└── PD/               # IXI-PD.tar extracted → IXI???-<site>-????-PD.nii.gz  [notebook 04 only]
```

Minimum for notebooks 01–03: `IXI.xls` + `T1/`
Full multi-contrast experiment (notebook 04): all three modalities.

IXI is **not defaced** — faces are present in T1 volumes.

---

## FaceAge model weights

The FaceAge repository (`vendor/FaceAge`) is tracked as source code but **weights are not included** (>100 MB).

1. Clone the repo: `git clone https://github.com/AIM-Harvard/FaceAge vendor/FaceAge`
2. Download weights from the Google Drive link in `vendor/FaceAge/README.md`
3. Place in `vendor/FaceAge/models/`:
   - `FaceAge_weights.pt`
   - `age_regressor.pt`

Reference: Bontempi et al., *The Lancet Digital Health*, 2025.
DOI: 10.1016/S2589-7500(25)00045-4

---

## SynthBA model weights

SynthBA is installed via pip — no manual weight download needed:
```bash
pip install synthba
```
Or see `vendor/MODELS.md` for the pinned version used in this paper.

---

## SFCN pretrained weights (optional, notebook 03)

For SFCN-based brain age (Peng et al. 2021, UK Biobank):

1. Clone: `git clone https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain vendor/SFCN`
2. Download pretrained weights and place in `vendor/SFCN/`

SFCN is optional — SynthSeg (FreeSurfer 7.4+) is the primary brain age method.
