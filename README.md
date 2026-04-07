# faceage-to-brainage

**Does the face in your MRI scan know how old your brain is?**

A proof-of-concept research project for MIDL 2025 (short paper track).  
Deadline: **15 April 2025**.

---

## Research Question

Standard brain age models predict a subject's age from brain parenchyma — grey/white matter volumes, cortical thickness, white matter hyperintensities. But every non-defaced T1 MRI also contains the full 3D morphology of the subject's face: subcutaneous fat distribution, orbital recession, facial bone structure.

**This project asks**: if you extract *two* age estimates from the *same* T1 scan — one from the brain, one from the face — do they agree? Do they capture the same underlying aging process, or independent biological signals?

This is novel because:
- Prior work paired MRI with *photographs* taken separately (different time, conditions, domain shift).
- We extract both signals from a **single file** — guaranteed paired, no acquisition gap.
- The face signal here is **morphological** (3D geometry), not textural (skin colour, wrinkles) — a different biology than photo-based face age.

---

## Hypotheses

| ID | Hypothesis | Primary test |
|----|-----------|--------------|
| H1 | FaceAge applied to MRI-rendered faces detects a monotonic aging signal (ρ > 0.4 with chronological age) | Spearman ρ, IXI cross-sectional |
| H2 | Brain age gap and face age gap (same T1) are positively correlated at the individual level | Spearman ρ, IXI, n≈580 |
| H3 | Face age variability across scanners (SIMON) is lower than brain age variability | CV comparison, 73 sessions |
| H4 | Multi-contrast RGB render (T1→R, T2→G, PD→B) improves age signal vs T1-only | Δρ, IXI subjects with all 3 modalities |

The **key baseline** is the null result from Cole et al. 2020 (*NeuroImage*), who found no association between brain-PAD and subjective facial age ratings in UK Biobank. We test the same question with AI-derived, morphology-based face age — a cleaner, objective measurement.

---

## Pipeline

```
T1 MRI (.nii.gz / .mgz)
        │
        ├── skull-strip → brain parenchyma
        │       └── SynthSeg / SFCN → brain_age
        │
        └── marching-cubes (skin surface, level≈40)
                └── PyVista frontal render → face.png (RGB)
                        └── FaceAge (Inception-ResNet-v1) → face_age

Per subject: { chron_age, brain_age, face_age }
                → brain_age_gap = brain_age - chron_age
                → face_age_gap  = face_age  - chron_age
                → Spearman ρ(brain_age_gap, face_age_gap)
```

---

## Datasets

### SIMON *(local)*
Single-subject MRI reliability dataset.  
1 healthy male, scanned repeatedly across **73 sessions**, **36 scanners**, **13 MRI models**, ages 29–46.  
FreeSurfer-processed `.mgz` files, **not defaced** (faces present).  
→ Used for **scanner-variance experiment** (H3): how stable are face age and brain age predictions across different hardware?

### IXI *(to download)*
Cross-sectional healthy brain MRI from 3 sites in London.  
~580 subjects, ages 20–86, T1 + T2 + PD modalities.  
Sites: Guys Hospital, Hammersmith Hospital (HH), Institute of Psychiatry (IOP).  
**Not defaced.**  
→ Used for **main cross-sectional experiment** (H1, H2, H4).

Download: https://brain-development.org/ixi-dataset/  
Files needed: `IXI-T1.tar`, `IXI.xls` (demographics). For H4 also: `IXI-T2.tar`, `IXI-PD.tar`.

---

## Repository Structure

```
faceage-to-brainage/
├── src/
│   ├── render.py        # MRI volume → frontal face PNG (marching cubes + PyVista)
│   ├── face_age.py      # FaceAge wrapper with MTCNN-bypass for MRI renders
│   ├── brain_age.py     # SynthSeg (FreeSurfer) + SFCN inference wrappers
│   └── utils.py         # Volume loading, RAS reorientation, metadata helpers
│
├── scripts/
│   ├── batch_render.py       # Render faces for a whole directory of scans
│   ├── batch_brain_age.py    # Run SynthSeg on all scans
│   └── batch_face_age.py     # Run FaceAge on all rendered PNGs
│
├── notebooks/
│   ├── 01_poc_single_scan.ipynb      # PoC: one SIMON scan end-to-end
│   ├── 02_simon_reliability.ipynb    # 73 sessions: face age CV vs brain age CV
│   ├── 03_ixi_main_experiment.ipynb  # IXI: brain gap × face gap correlation
│   └── 04_multicontrast_rgb.ipynb    # T1+T2+PD → RGB render experiment
│
├── notes/
│   └── literature_review.md  # Background reading: FaceAge, SFCN, brain-PAD, shared mechanisms
│
├── data/
│   └── README.md         # How to obtain IXI, SIMON, FaceAge weights, SFCN weights
│
├── vendor/               # External repos cloned here (FaceAge, SFCN) — not committed
├── results/              # Generated figures and tables — not committed
├── requirements.txt
└── environment.yml
```

See `related works/README.md` for the current shortlist of testable baselines and promising works without confirmed open weights.
See `related works/research_questions.md` for a dated memo on chronological-age accuracy, face-age versus brain-age gaps, 2D-to-3D avatar reconstruction, and non-defaced dataset opportunities.

---

## Setup

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate faceage
```

### Option B — pip

```bash
pip install -r requirements.txt
```

### External dependencies (not on PyPI)

```bash
# FaceAge model + weights
git clone https://github.com/AIM-Harvard/FaceAge vendor/FaceAge
# Download FaceAge_weights.pt + age_regressor.pt from the Google Drive link in vendor/FaceAge/README.md
# Place them in vendor/FaceAge/models/

# SFCN pretrained weights (optional, SynthSeg is the default brain age method)
git clone https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain vendor/SFCN
```

### FreeSurfer (for SynthSeg)

SynthSeg is bundled with FreeSurfer 7.4+. Install from https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall  
On Colab: `!bash <(wget -qO- https://surfer.nmr.mgh.harvard.edu/colab/env.sh)`

---

## Quickstart

```bash
# 1. Test on a single SIMON scan
jupyter notebook notebooks/01_poc_single_scan.ipynb

# 2. Batch render all SIMON sessions
python scripts/batch_render.py data/simon/ results/simon_renders/ --workers 4

# 3. Run FaceAge on renders
python scripts/batch_face_age.py results/simon_renders/ results/simon_face_ages.csv \
    --faceage vendor/FaceAge --bypass-mtcnn

# 4. Run SynthSeg brain age
python scripts/batch_brain_age.py data/simon/ results/simon_brain_ages/

# 5. Full IXI analysis
jupyter notebook notebooks/03_ixi_main_experiment.ipynb
```

---

## Key External Tools

| Tool | Paper | Use in this project |
|------|-------|---------------------|
| **FaceAge** | Bontempi et al., *Lancet Digital Health* 2025 | Face age from MRI renders |
| **SynthSeg** | Billot et al., *Nature Methods* 2023 | Brain segmentation + age |
| **SFCN** | Peng et al., *Medical Image Analysis* 2021 | Brain age baseline |
| **BrainIAC** | Tak et al., *Nature Neuroscience* 2026 | Context (same AIM-Harvard group) |

→ See `notes/literature_review.md` for a full annotated review including the key question: *does accelerated facial aging correlate with accelerated brain aging?*

---

Current local code support is strongest for `FaceAge`, `SFCN`, and the SynthSeg-based volumetric pipeline. Treat the current SFCN wrapper in `src/brain_age.py` as provisional until the age-bin decoding is verified.

## Background: What We Know

- **FaceAge gap predicts mortality** in cancer patients (HR 1.15/decade, n=6,196) — Bontempi 2025.
- **Brain-PAD predicts older facial appearance** at midlife — Belsky 2019, n=954.
- **But**: brain-PAD was NOT associated with subjective facial age ratings in UK Biobank — Cole 2020, n=2,205.
- **Gap**: no study has used AI-derived, morphology-based face age from MRI. Prior comparisons used subjective ratings or photos taken separately.

---

## Computing

Experiments run in VS Code locally. GPU-heavy inference (FaceAge, SFCN) can be offloaded to Google Colab Pro via the Remote Tunnels extension.

All `src/` modules are Colab-compatible. PyVista rendering uses offscreen mode (`pv.start_xvfb()` on Colab).

---

## Contact

Ekaterina Kondrateva — kondratevakate@github  
Affiliations: Maastricht University / MAASTRO Clinic  
Collaboration: AIM-Harvard / Mass General Brigham (FaceAge group, Andre Dekker)
