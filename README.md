# faceage-to-brainage

**Does the face in your MRI scan know how old your brain is?**

A proof-of-concept pipeline for paired face-age and brain-age estimation from the same non-defaced T1 MRI volume.

**Current status:**
- Face pipeline: FaceAge on 9-view MRI renders + linear calibration ✅
- Face morphometrics: GPA + EDMA on BioFace3D-20 landmarks → Ridge regression ✅  IXI MAE 7.81 yr
- Brain pipeline: SynthBA via SynthStrip ✅
- Gap correlation (face vs brain, same scan): r = 0.31, p < 0.01 ✅
- Scanner reproducibility (SIMON, 36 scanners): SynthBA SD = 1.21 yr ✅
- SFCN, MIDIBrainAge, BrainIAC: in progress 🔄

Paper: [MIDL 2026 Short Paper #99](papers/midl2026/midl-shortpaper.tex) · Pipeline diagram: [pipeline.drawio](papers/midl2026/pipeline.drawio)

---

## Research Question

Standard brain-age models predict age from brain parenchyma: grey and white matter volumes, cortical thickness, and related structural signals. But every non-defaced T1 MRI also contains the full 3D morphology of the face: subcutaneous fat distribution, orbital recession, facial bone structure, and soft-tissue shape.

This project asks: if you extract two age estimates from the same T1 scan — one from the brain and one from the face — do they agree? Do they capture the same underlying aging process, or partially independent biological signals?

Why this is interesting:
- prior work typically paired MRI with photographs taken separately
- this repo extracts both signals from a single file
- the face signal is morphological and MRI-derived, not a standard photo-age setting

---

## Results

| Method | Dataset | N | MAE (yr) | r | Bias (yr) |
|--------|---------|---|----------|---|-----------|
| FaceAge, 9-view renders | IXI | 406 | 11.34 | 0.83 | +10.04 |
| FaceAge + linear calibration | IXI | 80 | **10.24** | 0.83 | −2.12 |
| Photorealistic render (SD) | IXI | — | 19.91 | — | +19.91 |
| Face morphometrics (GPA+EDMA, Ridge) | IXI | 72 | **7.81** | 0.775 | −1.83 |
| SynthBA | IXI | 105 | **6.33** | 0.84 | −4.15 |
| SynthBA scan-rescan | SIMON | 99 | 16.16\* | — | −16.16\* |
| **Gap correlation** (face vs brain) | IXI | 93 | Pearson r = 0.31, Spearman ρ = 0.32, p < 0.01 | | |

\* Large bias on SIMON is subject-specific (model predicts ~27 yr for a 43-yr-old); prediction SD across 36 scanners is only 1.21 yr, showing high reproducibility.

Calibration formula: `true_age = 1.010 × face_pred − 17.049` (fit on 20 subjects, applied to 80).

---

## Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H1 | FaceAge on MRI renders detects a monotonic aging signal | ✅ r = 0.83 on IXI |
| H2 | Brain-age gap and face-age gap are positively correlated | ✅ r = 0.31, p < 0.01 |
| H3 | Face-age variability across scanners ≤ brain-age variability | 🔄 SIMON face-age pending |
| H4 | Multi-contrast renders improve face-age vs T1-only | 🔄 notebook 04 pending |

---

## Pipeline

```text
T1 MRI (.nii.gz)
    │
    ├── FACE BRANCH
    │   marching cubes (t=30) → PyVista 9-view render
    │   → FaceAge (ResNet-50) → average → linear calibration
    │   → face_age, face_age_gap
    │
    └── BRAIN BRANCH
        SynthStrip (skull strip) → SynthBA
        → brain_age, brain_age_gap

Per subject: chron_age, face_age, brain_age, face_age_gap, brain_age_gap
Gap analysis: scripts/gap_correlation.py → papers/tables/gap_correlation.csv
```

See [papers/midl2026/pipeline.drawio](papers/midl2026/pipeline.drawio) for the full diagram (open in VS Code Draw.io extension or app.diagrams.net).

---

## Datasets

### IXI
Cross-sectional healthy brain MRI from three London sites.
- ~580 subjects, ages 20–86, sites: Guys, HH, IOP
- T1, T2, PD modalities; non-defaced
- Used: 406 subjects (face age), 105 (brain age), 93 (gap correlation)
- Download: <https://brain-development.org/ixi-dataset/>

### SIMON
Single-subject MRI reliability dataset.
- 1 healthy male, 73 sessions, 36 scanners, ages 29–46
- 99 T1 scans; non-defaced
- Used for scanner reproducibility (H3)

See [papers/data/README.md](papers/data/README.md) for full download instructions.

---

## Repository Structure

```text
faceage-to-brainage/
├── src/
│   ├── brain_age.py
│   ├── face_age.py
│   ├── render.py
│   ├── utils.py
│   ├── faceage/                    ← FaceAge deep-learning pipeline (smileyenot983)
│   │   └── faceage_mri/FaceAge/   ← MTCNN + ResNet-50 age model, rendering scripts
│   └── face_age_morphometrics/     ← 3D morphometrics pipeline (BobrG)
│       ├── bioface3d/              ← BioFace3D-20 MVCNN landmark detector
│       ├── src/                    ← GPA + EDMA features, Ridge regression
│       └── scripts/                ← extract_features.py, benchmark_regressors.py
│
├── scripts/
│   ├── gap_correlation.py   ← compute & save face/brain gap correlation
│   ├── batch_render.py
│   ├── batch_face_age.py
│   ├── batch_brain_age.py
│   └── batch_sfcn.py
│
├── notebooks/
│   ├── 01_poc_single_scan.ipynb
│   ├── 02_simon_reliability.ipynb
│   ├── 03_ixi_main_experiment.ipynb
│   ├── 04_multicontrast_rgb.ipynb
│   ├── 05_sfcn_colab_bootstrap.ipynb
│   ├── 06_brainage_colab.ipynb
│   ├── 07_synthba_colab.ipynb          ← SynthBA on IXI (main result)
│   ├── 08_synthba_simon_colab.ipynb    ← SynthBA on SIMON (reproducibility)
│   ├── 09_midi_simon_colab.ipynb       ← MIDIBrainAge (in progress)
│   └── 10_brainiac_simon_colab.ipynb   ← BrainIAC (in progress)
│
├── papers/
│   ├── midl2026/         ← LaTeX source + pipeline.drawio
│   ├── data/             ← dataset download instructions
│   ├── tables/           ← CSV results (gitignored, .gitkeep tracked)
│   ├── figures/          ← generated figures (gitignored)
│   ├── notes/            ← reproduction notes
│   └── related_works/    ← literature review
│
├── vendor/
│   ├── MODELS.md         ← pinned model versions & commit hashes
│   └── FaceAge/          ← cloned, weights gitignored
│
├── config/
│   └── brain_age_runtime.example.json
├── tests/
├── environment.yml
└── requirements.txt
```

---

## Setup

### Option A: conda
```bash
conda env create -f environment.yml
conda activate faceage
```

### Option B: pip
```bash
pip install -r requirements.txt
```

### Model weights

See [vendor/MODELS.md](vendor/MODELS.md) for pinned versions and download links.

```bash
# FaceAge
git clone https://github.com/AIM-Harvard/FaceAge vendor/FaceAge
# Download FaceAge_weights.pt and age_regressor.pt → vendor/FaceAge/models/

# SynthBA — installed via pip (see requirements.txt)
# No manual weight download needed

# SFCN (optional)
git clone https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain vendor/SFCN
# Download run_20190719_00_epoch_best_mae.p → vendor/SFCN/brain_age/
```

### Runtime config
Copy the example config and fill in local paths:
```bash
cp config/brain_age_runtime.example.json config/local/brain_age_runtime.json
```

---

## Quickstart

### Reproduce main results (local)

```bash
# 1. Proof of concept — single scan
jupyter notebook notebooks/01_poc_single_scan.ipynb

# 2. Batch render face images from IXI T1 scans
python scripts/batch_render.py papers/data/ixi/T1/ papers/figures/ixi_renders/ --workers 4

# 3. Run FaceAge on rendered PNGs
python scripts/batch_face_age.py papers/figures/ixi_renders/ papers/tables/face_ages.csv \
    --faceage vendor/FaceAge --bypass-mtcnn

# 4. Run SynthBA brain-age
python scripts/batch_brain_age.py papers/data/ixi/T1/ papers/tables/brain_ages.csv

# 5. Compute gap correlation
python scripts/gap_correlation.py
# → saves papers/tables/gap_correlation.csv
```

### Colab notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| [07_synthba_colab.ipynb](notebooks/07_synthba_colab.ipynb) | SynthBA on IXI (main brain-age result) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kondratevakate/faceage-to-brainage/blob/main/notebooks/07_synthba_colab.ipynb) |
| [08_synthba_simon_colab.ipynb](notebooks/08_synthba_simon_colab.ipynb) | SynthBA on SIMON (reproducibility) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kondratevakate/faceage-to-brainage/blob/main/notebooks/08_synthba_simon_colab.ipynb) |
| [05_sfcn_colab_bootstrap.ipynb](notebooks/05_sfcn_colab_bootstrap.ipynb) | SFCN baseline on SIMON | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kondratevakate/faceage-to-brainage/blob/main/notebooks/05_sfcn_colab_bootstrap.ipynb) |

---

## Key External Tools

| Tool | Reference | Use |
|------|-----------|-----|
| FaceAge | Bontempi et al., *Lancet Digital Health* 2025 | Face age from MRI renders |
| SynthBA | Lemaître et al. 2022 | Primary brain-age model |
| SynthStrip | Hoopes et al. 2022 | Skull stripping |
| SFCN | Peng et al., *Med Image Anal* 2021 | Brain-age baseline |
| MIDIBrainAge | MIDI Consortium | Sequence-specific brain age (in progress) |
| BrainIAC | Tak et al. 2026 | Foundation brain-age model (in progress) |
| PyVista | Sullivan & Kaszynski, *JOSS* 2019 | 3D MRI surface rendering |

---

## Brain-Age Model Status

| Model | IXI | SIMON | Notes |
|-------|-----|-------|-------|
| SynthBA | ✅ MAE 6.33 yr | ✅ SD 1.21 yr | Primary result |
| SFCN | 🔄 | 🔄 | Age-bin decoding under validation |
| MIDIBrainAge | 🔄 | 🔄 | Notebook 09 in progress |
| BrainIAC | 🔄 | 🔄 | Notebook 10 in progress |

---

## Contact

**Ekaterina Kondrateva** — kondratevakate@gmail.com
**Ramil Khafizov** · **Gleb Bobrovskikh**

Code: <https://github.com/kondratevakate/faceage-to-brainage>
