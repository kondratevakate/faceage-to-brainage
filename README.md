# faceage-to-brainage

**Does the face in your MRI scan know how old your brain is?**

A proof-of-concept pipeline for paired face-age and brain-age estimation from the same non-defaced T1 MRI volume.

<p align="center">
  <img src="papers/figures/ex_render.png" alt="Multi-view MRI-derived face renders" width="540"/>
</p>
<p align="center"><em>Nine-view frontal renders extracted from a single T1 MRI via marching cubes. These are the inputs to the face branch.</em></p>

Article draft: [papers/midl2026/midl-shortpaper.tex](papers/midl2026/midl-shortpaper.tex) · Pipeline diagram: [pipeline.drawio](papers/midl2026/pipeline.drawio) · Literature: [papers/related_works/literature_review.md](papers/related_works/literature_review.md)

---

## Global Landscape: Age Estimation from Face and Brain

### Face age estimation

The face is one of the most information-dense age signals available non-invasively. Under controlled conditions, deep models trained on large face-photo datasets achieve around **3 years MAE** for chronological age (Zhang et al. 2023; Rothe/DEX ~3.2 yr on MORPH-II). Among the components that carry most of the age signal:

- **Skin tone, texture, and facial contrast** - account for roughly 25-33% of age-perception accuracy; their removal from older faces collapses judgments toward chance
- **Periocular region and sclera** - scleral color (darker, redder, yellower) and orbital changes concentrate multiple aging processes in a small area; highly sensitive to rendering artifacts
- **Facial fat compartments** - MRI evidence shows significant age-related change in cheek fat distribution; muscle volume does not differ significantly across age groups in healthy women

The most important practical distinction: **apparent/perceived age**, **chronological age**, and **biological age** are different targets. FaceAge (Bontempi et al., *Lancet Digital Health* 2025) is a biological-age model - cancer patients look 4.79 years older on average, and the face-age gap predicts survival. It is not a fair chronological-age benchmark. For a direct chronological-age comparison, MiVOLO (face-only checkpoints, MAE ~4.3 yr) is the appropriate open baseline.

### Brain age estimation

Whole-brain structural T1-weighted MRI is one of the strongest non-invasive age signals available. Across large adult lifespan cohorts, realistic expectation is **4-6 years MAE** for healthy adults:

- SFCN (Peng et al. 2021): **2.14 yr MAE** on UK Biobank in-distribution; but 9-10 yr on independent CamCAN (scanner shift). In-distribution performance should not be taken as a universal ceiling.
- SynthBA: best open protocol-agnostic option; handles T1, T2, and FLAIR without retraining
- BrainIAC (Tak et al. 2026, *Nature Neuroscience*): foundation model (ViT-B, SSL on ~49k MRIs); brain-age MAE 6.55 yr at 20% fine-tuning; demonstrates few-shot generalization across 7 simultaneous tasks
- Kim et al. 2025: **2.73 yr MAE** on clinical 2D T1 after bias correction - strongest clinical result, but not openly runnable

Brain vascular markers (WMH, microangiopathic change) are auxiliary biomarkers of aging heterogeneity, not mature standalone age clocks. Healthy aging diverges substantially after ~70 years, particularly in hippocampus, amygdala, and temporal cortex.

### The link - and the gap

| Evidence | Direction | Source |
|----------|-----------|--------|
| Twin who looked older died first in 73% of pairs | Face -> mortality | Christensen et al., *BMJ* 2009 |
| Looking 5 yr younger -> lower COPD, osteoporosis, cognitive decline risk | Face -> health | Rotterdam Study, *BJD* 2023 |
| Brain-PAD at midlife predicts older facial appearance | Brain -> Face | Belsky et al., *Mol Psychiatry* 2019 |
| Multimodal brain-PAD not significantly associated with facial aging | Null | Cole et al., *NeuroImage* 2020 |

The Cole 2020 null result used **subjective facial age ratings** - imprecise and not scalable. This project replaces that with AI-derived face age from **MRI morphology**: bone structure, subcutaneous fat volume, orbital recession. These are fundamentally different measurements. Our pipeline tests whether the morphological face-age signal from T1 MRI correlates with brain parenchymal aging in the same scan.

**Core tension**: the most age-informative facial cues (skin appearance, scleral color, eye-region detail, fat redistribution) are least grounded in structural MRI and most vulnerable to hallucination by generative models. This makes validation against brain age from the same scan essential.

---

## Research Question

Standard brain-age models predict age from brain parenchyma: grey and white matter volumes, cortical thickness, and related structural signals. But every non-defaced T1 MRI also contains the full 3D morphology of the face: subcutaneous fat distribution, orbital recession, facial bone structure, and soft-tissue shape.

This project asks: if you extract two age estimates from the same T1 scan - one from the brain and one from the face - do they agree? Do they capture the same underlying aging process, or partially independent biological signals?

Why this is interesting:
- prior work typically paired MRI with photographs taken separately
- this repo extracts both signals from a single file
- the face signal is morphological and MRI-derived, not a standard photo-age setting

---

## Pipeline

```text
T1 MRI (.nii.gz)
    |
    |-- FACE BRANCH
    |   marching cubes (t=30) -> PyVista 9-view render
    |   -> FaceAge (ResNet-50) -> average -> linear calibration
    |   -> face_age, face_age_gap
    |
    `-- BRAIN BRANCH
        SynthStrip (skull strip) -> SynthBA
        -> brain_age, brain_age_gap

Per subject: chron_age, face_age, brain_age, face_age_gap, brain_age_gap
Gap analysis: scripts/gap_correlation.py -> papers/tables/gap_correlation.csv
```

See [papers/midl2026/pipeline.drawio](papers/midl2026/pipeline.drawio) for the full diagram (open in VS Code Draw.io extension or app.diagrams.net).

---

## Datasets

### IXI
Cross-sectional healthy brain MRI from three London sites.
- ~580 subjects, ages 20-86, sites: Guys, HH, IOP
- T1, T2, PD modalities; non-defaced
- Used: 406 subjects (face age), 105 (brain age), 93 (gap correlation)
- Download: <https://brain-development.org/ixi-dataset/>

### SIMON
Single-subject MRI reliability dataset.
- 1 healthy male, 73 sessions, 36 scanners, ages 29-46
- 99 T1 scans; non-defaced
- Used for scanner reproducibility (H3)

See [papers/data/README.md](papers/data/README.md) for full download instructions.

---

## Repository Structure

```text
faceage-to-brainage/
|-- src/
|   |-- brain_age.py
|   |-- face_age.py
|   |-- render.py
|   |-- utils.py
|   |-- faceage/                    <- FaceAge deep-learning pipeline (smileyenot983)
|   |   `-- faceage_mri/FaceAge/   <- MTCNN + ResNet-50 age model, rendering scripts
|   `-- face_age_morphometrics/    <- 3D morphometrics pipeline (BobrG)
|       |-- bioface3d/             <- BioFace3D-20 MVCNN landmark detector
|       |-- src/                   <- GPA + EDMA features, Ridge regression
|       `-- scripts/               <- extract_features.py, benchmark_regressors.py
|
|-- scripts/
|   |-- gap_correlation.py         <- compute and save face/brain gap correlation
|   |-- batch_render.py
|   |-- batch_face_age.py
|   |-- batch_brain_age.py
|   `-- batch_sfcn.py
|
|-- notebooks/
|   |-- 01_poc_single_scan.ipynb
|   |-- 02_simon_reliability.ipynb
|   |-- 03_ixi_main_experiment.ipynb
|   |-- 04_multicontrast_rgb.ipynb
|   |-- 05_sfcn_colab_bootstrap.ipynb
|   |-- 06_brainage_colab.ipynb
|   |-- 07_synthba_colab.ipynb          <- SynthBA on IXI (main result)
|   |-- 08_synthba_simon_colab.ipynb    <- SynthBA on SIMON (reproducibility)
|   |-- 09_midi_simon_colab.ipynb       <- MIDIBrainAge (in progress)
|   `-- 10_brainiac_simon_colab.ipynb   <- BrainIAC (in progress)
|
|-- papers/
|   |-- midl2026/         <- article draft source + pipeline.drawio
|   |-- data/             <- dataset download instructions
|   |-- tables/           <- CSV results (gitignored, .gitkeep tracked)
|   |-- figures/          <- generated figures (gitignored)
|   |-- notes/            <- implementation reproduction notes
|   `-- related_works/    <- literature review (single source of truth)
|
|-- vendor/
|   |-- MODELS.md         <- pinned model versions and commit hashes
|   `-- FaceAge/          <- cloned, weights gitignored
|
|-- config/
|   `-- brain_age_runtime.example.json
|-- tests/
|-- environment.yml
`-- requirements.txt
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
# Download FaceAge_weights.pt and age_regressor.pt -> vendor/FaceAge/models/

# SynthBA - installed via pip (see requirements.txt)
# No manual weight download needed

# SFCN (optional)
git clone https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain vendor/SFCN
# Download run_20190719_00_epoch_best_mae.p -> vendor/SFCN/brain_age/
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
# 1. Proof of concept - single scan
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
# -> saves papers/tables/gap_correlation.csv
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
| SynthBA | MAE 6.33 yr | SD 1.21 yr | Primary result |
| SFCN | In progress | In progress | Age-bin decoding under validation |
| MIDIBrainAge | In progress | In progress | Notebook 09 in progress |
| BrainIAC | In progress | In progress | Notebook 10 in progress |

---

## Contact

**Ekaterina Kondrateva** - kondratevakate@gmail.com  
**Ramil Khafizov**, **Gleb Bobrovskikh**

Code: <https://github.com/kondratevakate/faceage-to-brainage>
