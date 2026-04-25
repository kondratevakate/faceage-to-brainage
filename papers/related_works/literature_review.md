# Literature Review: Face Age & Brain Age — Are They the Same Clock?

> **Companion specs in this folder** (datasets, design, methodology — kept separate so this file stays a *literature* review):
> - [`ai_experiment_planning.md`](ai_experiment_planning.md) — SOTA practices for AI experiment planning: hypothesis-driven frameworks, TRIPOD+AI/CLAIM/PROBAST+AI reporting standards, statistical rigor, tools, top-5 ranked actions for this project
> - [`sota_design.md`](sota_design.md) — SOTA face-age & brain-age experimental design 2024–2026
> - [`data_audit.md`](data_audit.md) — current IXI/SIMON cohort coverage audit
> - [`test_retest_datasets.md`](test_retest_datasets.md) — small/obscure open T1 test–retest cohorts (Cao 2015, Huang 2016, Maclaren ds000239, B-Q MINDED, GSP retest) for OOD validation, deliberately avoiding HCP/IXI/OASIS/ADNI/UKBB
> - [`avatar_3d_datasets.md`](avatar_3d_datasets.md) — public 2D+3D face datasets, "no-hair + closed-eyes" match analysis, internal anatomy modeling
> - [`clinical_facial_datasets.md`](clinical_facial_datasets.md) — clinical face datasets (cosmetology / dental / facial surgery)

## 1. Face Age Estimation

### FaceAge — Bontempi et al., *Lancet Digital Health*, 2025
**PubMed: 40345937 | DOI: 10.1016/S2589-7500(25)00042-1**

AIM-Harvard / Mass General Brigham (Hugo Aerts, Andre Dekker labs).

- **Architecture**: Inception-ResNet-v1, pretrained on FaceNet (face recognition), fine-tuned on 56,304 age-labelled photos (IMDb-Wiki + UTKFace).
- **Training set**: 58,851 healthy individuals (presumed cancer-free), ages 60+.
- **Clinical validation**: 6,196 cancer patients across Netherlands (MAASTRO) and US cohorts.
- **Key findings**:
  - Cancer patients look **4.79 years older** than chronological age (p < 0.0001).
  - Per-decade face age gap → HR 1.15 for overall survival across pan-cancer cohort (n=4,906).
  - FaceAge improves 6-month survival prediction from AUC 0.74 → 0.80 in palliative patients; outperforms physician judgment (80% vs 61% accuracy).
  - FaceAge gap significantly associated with **CDK6 expression** (senescence gene, G1/S checkpoint) — chronological age was not. This positions FaceAge as a biomarker of cellular senescence, not just appearance.
- **Relevance**: Direct baseline for our project. Code + weights open-source on GitHub (AIM-Harvard/FaceAge).

### DEX — Rothe et al., *IJCV*, 2018
- VGG-16 trained on IMDb-Wiki (~500k photos), outputs probability distribution over ages → expected value.
- MAE ~3.2 years on MORPH-II.
- Predecessor to FaceAge; still used as baseline in many papers.

### FAHR-Face — Haugg et al., *arXiv:2506.14909*, 2025
- Foundation model for health recognition from face photos, same AIM-Harvard group.
- Next generation of FaceAge; embeddings potentially more robust to domain shift (relevant if MRI renders don't match FaceAge training distribution).

---

## 2. Brain Age Estimation

### Brain-AGE — Franke et al., *NeuroImage*, 2010
- First formalization of the brain age gap concept.
- Gaussian process regression on morphometric MRI features.
- Established "brain-PAD" (predicted age difference) as a biomarker of brain health.

### SFCN — Peng et al., *NeuroImage*, 2021 | PAC 2019 winner
- Simple Fully Convolutional Network, trained on 14,503 UK Biobank T1 scans.
- **MAE 2.14 years** on UK Biobank held-out set.
- Generalisation problem: MAE 9–10 years on independent CamCAN cohort (different scanner, protocol).
- Pretrained weights available on GitHub (ha-ha-ha-han/UKBiobank_deep_pretrain).
- **Our primary brain age baseline.**

### SynthSeg — Billot et al., *Nature Methods*, 2023
- Contrast- and resolution-agnostic segmentation via synthetic training.
- FreeSurfer 7.4+ built-in (`mri_synthseg`).
- Works on any T1/T2/FLAIR contrast without retraining — ideal for cross-scanner datasets like SIMON and IXI.
- Outputs regional brain volumes → can be used as features for age regression.

### SFCN-reg with improved preprocessing — *Human Brain Mapping*, 2025
- Addresses SFCN's generalisation gap with better data augmentation and site harmonisation.
- Focus on cross-site robustness — most relevant brain age baseline for our SIMON/IXI work.

### BrainIAC — Tak et al., *Nature Neuroscience*, Feb 2026
- Foundation model (ViT-B, SimCLR self-supervised pretraining), same AIM-Harvard lab as FaceAge.
- Trained on ~49,000 brain MRIs, solves 7 tasks simultaneously: brain age, dementia prediction, IDH mutation, glioblastoma survival, stroke time-to-event, MRI sequence classification, segmentation.
- Brain age MAE **6.55 years** at 20% fine-tuning data — worse than SFCN in absolute terms, but demonstrates few-shot generalisation across tasks.
- **Not our baseline** for age prediction accuracy, but important context: same group building unified brain + face models.

---

## 3. The Key Question: Does Looking Older = Older Brain?

### 3.1 Direct evidence: facial aging predicts mortality and disease

**Christensen et al., *BMJ*, 2009 (Danish twins, n=1,826)**
- Among same-sex twins aged 70+, the twin who looked older died first in 73% of cases.
- Perceived age (rated by nurses) independently predicted survival after controlling for actual age (HR ~1.36).
- The "face" component (not hair/clothes) explained 82% of the survival association.

**Rotterdam Study — *British Journal of Dermatology*, 2023 (n=2,679)**
- Looking 5 years younger than chronological age → significantly lower risk of COPD, osteoporosis, cataracts, cognitive decline — even after adjusting for smoking and BMI.
- Effect persisted for both self-reported and AI-assessed facial age.

**Meta-analysis, 15 studies, 2023**
- Risk of all-cause mortality +6–51% (HR 1.06–1.51) in those who "look older."
- Associations with CVD, COPD, osteoporosis replicated across cohorts.

### 3.2 Brain age predicts facial appearance (not the reverse)

**Belsky et al., *Molecular Psychiatry*, 2019 (Dunedin birth cohort, n=954, age 45)**
- Brain-PAD (positive = brain looks older than chronological age) at midlife was associated with:
  - **Older facial appearance** (rated by blinded observers)
  - Accelerated pace of biological aging (18 biomarkers)
  - Lower cognitive performance in both childhood and adulthood
- **Direction of association**: brain-PAD → facial appearance, not the other way around.

### 3.3 The null result: UK Biobank, Cole et al., 2020

**Cole et al., *NeuroImage*, 2020 (n=2,205 multimodal MRI)**
- Multimodal brain age model (6 MRI modalities), MAE 3.55 years.
- Brain-PAD associated with: blood pressure, diabetes, stroke, smoking, alcohol, cognitive measures.
- **Brain-PAD NOT significantly associated with facial aging.**
- Interpretation: brain age gap and facial aging may share common upstream risk factors but are not directly correlated at the individual level.

### 3.4 Shared mechanisms

| Mechanism | Face | Brain |
|-----------|------|-------|
| Telomere shortening | Skin cell senescence, pigmentation loss | Neurodegeneration, oligodendrocyte loss |
| Chronic inflammation (IL-6, TNF-α) | Collagen breakdown, dermal thinning | Microglial activation, white matter lesions |
| Cardiovascular risk | Facial fat redistribution, vascular skin changes | White matter hyperintensities, cortical thinning |
| Cellular senescence (CDK6, p16INK4a) | Fibroblast/keratinocyte senescence | Astrocyte senescence |
| Ectodermal origin | Skin, subcutaneous tissue | CNS neurons, glia |

**Key insight**: Face and brain share ectodermal developmental origin. Systemic aging drivers (CVD, inflammation, oxidative stress) affect both in parallel — which is why both predict mortality — but the correlation at the **individual level** may be weak or site-dependent.

---

## 4. Face as a Diagnostic Window for Brain Disease?

### Parkinson's Disease
- AI analysis of facial micro-expressions and movement from video selfies: high accuracy for PD detection.
- PD "masked face" (hypomimia, reduced spontaneous expression) is a clinical diagnostic sign.
- Several groups have demonstrated >80% accuracy from video selfies vs clinical controls.

### Dementia / Cognitive Decline
- **Facial aging and dementia risk (*Alzheimer's Research & Therapy*, 2024)**: High perceived facial age → HR 1.61 for dementia (95% CI 1.33–1.96).
- Crow's feet wrinkles → OR 2.48 for cognitive impairment in the Chinese Longitudinal Healthy Longevity Survey.
- Proposed mechanism: shared vascular/systemic aging, not a direct causal pathway from face to brain.

### Cancer (General — not brain-specific)
- FaceAge: cancer patients look older; face age gap predicts survival independently of tumour type.
- Mechanism: tumour-driven accelerated systemic senescence manifests in the face.

### Brain Tumours Specifically
- **No evidence** that facial appearance predicts brain tumour presence or type.
- Brain tumour AI (FastGlioma, BrainIAC) operates on MRI, not photographs.
- The question "can a selfie detect a brain tumour?" is currently unanswered — and likely to remain negative because brain tumours don't systematically alter facial morphology.
- Fundable as exploratory research only with strong pilot data.

---

## 5. Research Gap Our Project Addresses

**What exists**:
- Face age from photos → mortality, cancer survival (FaceAge, Christensen twins)
- Brain age from MRI → health outcomes, cognitive decline (SFCN, BrainIAC)
- Indirect link: brain-PAD predicts older facial appearance (Belsky 2019)
- One null result at individual level: brain-PAD not associated with facial aging in UK Biobank (Cole 2020) — using **rated/perceived** age, not AI-derived

**What doesn't exist**:
- Direct extraction of **both** aging signals from the **same T1 MRI scan**
- Testing whether FaceAge applied to MRI-rendered face images captures an aging signal independent of the brain signal from the same scan
- Scanner-reliability analysis comparing MRI-derived face age vs brain age variability (our SIMON experiment)

**Why the gap matters**:  
The Cole 2020 null result used subjective facial age ratings — imprecise, subjective, not scalable. AI-derived face age from **MRI morphology** is a fundamentally different measurement: it captures bone structure, subcutaneous fat volume, orbital recession — not skin texture or pigmentation. Our pipeline tests whether the **morphological** face age signal from T1 MRI correlates with brain parenchymal aging in the same scan. This eliminates domain shift and acquisition confounds that plague studies pairing photos with MRI.

If we find a correlation — it's the first AI-based evidence linking facial morphological aging to brain aging from a single imaging modality.  
If we find no correlation — it suggests morphological face aging and brain aging are truly independent signals in T1, publishable as a null result with methodological novelty.

---

## 6. Funding Landscape

| Funder | Mechanism | Angle |
|--------|-----------|-------|
| NIH/NCI | R01, U01 (AI in oncology PAR-23-070) | Face + brain aging biomarkers in cancer cohorts |
| ERC Starting/Consolidator | ERC-StG / ERC-CoG | AI biomarkers, multimodal aging |
| Alzheimer's Association | AARG, AARG-NTF | Facial aging as dementia screening tool |
| DoD CDMRP | Idea Award | Cross-modal biological age |
| Dutch ZonMw | Open Competition | MAASTRO / Maastricht connection (Andre Dekker) |

**Immediate next step**: MIDL 2025 short paper (April 15 deadline) — establish the method with PoC results, then use as pilot data for grant applications.

---

## 7. Age Biomarker Signals: Expected Error Hierarchy

### Scope and framing

The strongest quantitative age signals come from **global representations** — the whole brain on structural MRI, or the full face in a standardized photograph. Individual sub-features (skin tone, wrinkles, periocular changes, facial fat redistribution, scleral discoloration) are well established as age-related cues but are not supported by equally robust standalone error estimates.

A central methodological distinction: the literature mixes at least three targets — **chronological age**, **apparent/perceived age**, and **biological age**. These should not be conflated. Models such as FaceAge are valuable for prognosis and biological aging but their outputs are not directly comparable to classical chronological age regression MAE.

### 7.1 Whole-brain structural MRI

The most reliable MRI-based age signal comes from the **entire brain** on structural T1-weighted MRI. Across large adult lifespan cohorts, a realistic expectation for healthy adults is roughly **4–6 years MAE**.

- **Feng et al., Neurobiology of Aging, 2020**: structural brain MRI model, ~**4.06 years MAE** on hold-out test set; ~4.21 years on independent lifespan validation cohort.
- **Xifra-Porxas et al., NeuroImage, 2021** [ref 10]: T1-derived MRI features on Cam-CAN cohort, ~**5.33 years MAE**.
- **SFCN (Peng et al., 2021)** [ref 5]: **2.14 years MAE** on UK Biobank held-out; but 9–10 years on independent CamCAN (scanner/protocol shift). UK Biobank in-distribution performance should not be treated as a universal ceiling.

For the current project: if a face-derived pipeline from MRI renders performs substantially worse than this range, that does not make it useless — only that it draws from a much weaker and more indirect source of age information than a full structural brain model.

### 7.2 Brain vascular and microangiopathic markers

Vascular burden — **white matter hyperintensities** and related microangiopathic change — increases with age and becomes more pronounced in late life. However, vascular markers are **not mature standalone age regressors**. No robust standalone MAE is currently supported in the assembled literature.

**Best treated as**: auxiliary MRI biomarker of aging heterogeneity, one component of brain aging diversity rather than a primary age clock.

### 7.3 Adult skeletal morphology

A useful contrast baseline from forensic anthropology. In adult skeletal age-at-death estimation:

- Best expert-integrative approach: ~**6.13 years mean inaccuracy** (Bailey & Vidoli, 2023) [ref 11]
- Classical partial-skeleton methods: ~**7.56–13.75 years**

This places realistic skeletal age estimation in the **6–14 year** range for adults. Methodological implication: a model relying on shape-dominant cues without rich surface/skin information may perform closer to this coarser forensic regime than to modern full-face photo models.

### 7.4 Full-face photographic age prediction

- **Zhang et al., Skin Research and Technology, 2023** [ref 12]: deep model on large clinical/cosmetic database, **MAE 3.011 years**; SVM baseline on the same task, **MAE 10.185 years**.
- **DEX (Rothe et al., 2018)** [ref 8]: VGG-16 on IMDb-Wiki, ~**3.2 years MAE** on MORPH-II.

Under favorable conditions (standardized imaging, large training set, full face), approximately **3 years MAE** is achievable. This does not imply that isolated face sub-features can independently match this performance. Best-performing models leverage the **joint distribution** of many weak and strong cues.

### 7.5 Skin, wrinkles, texture, pigmentation, and facial contrast

Skin is one of the most important age cues in photographs, especially post-midlife, but the literature supports skin mainly as a **major contributing cue**, not a standalone validated chronological age predictor.

- **González-Alvarez & Sos-Peña, Vision Research, 2023** [ref 13]: skin tone and texture account for ~**25–33% of age-perception accuracy**; removing this information from older faces collapses age judgments toward chance.
- **Hsieh et al., Scientific Reports, 2023** [ref 14]: color information systematically biases facial age estimation.
- **Porcheron et al., PLOS ONE, 2013** [ref 15]: facial contrast (lip/skin, eye/skin luminance) declines with age and affects perceived age.

In the MRI-to-face setting: once a generative model invents skin texture, pores, and shading, it may also be inventing a large fraction of the age signal.

### 7.6 Periocular region and sclera

The eye region concentrates multiple aging processes simultaneously: skin, fat, musculature, fascial support.

- **Russell et al., Psychology and Aging, 2014** [ref 16]: scleral appearance becomes systematically darker, redder, and yellower with age; scleral manipulations affect perceived age, health, and attractiveness.

For MRI-derived faces: the periocular region is likely one of the most sensitive parts of any generated image. Artifacts around eyelids, under-eye shadows, orbital fat, or scleral tone may substantially shift age predictions even when craniofacial geometry is unchanged.

### 7.7 Facial soft tissues on MRI: fat versus muscle

When considering face-related MRI signals (not brain MRI), the literature points more to **fat compartments and soft tissue redistribution** than to mimetic muscles.

- **Gosain et al., Plastic and Reconstructive Surgery, 2005** [ref 17]: significant age-related changes in **distribution and volume of the cheek fat pad** on high-resolution MRI; muscle length, thickness, and volume did **not** differ significantly across age groups in healthy women.

Methodological constraint: if extracting age-related information from MRI-derived facial anatomy, the strongest candidate signal lies in **fat redistribution and soft tissue support architecture**, not muscle volume alone.

### 7.8 Teeth

Teeth are real age biomarkers, but most adult dental age estimation relies on **panoramic radiographs** or other dental imaging — not visible teeth in standard facial photographs (Adserias-Garriga, 2023) [ref 18].

**Treat as**: possible supplementary visual cue, not a primary benchmarked predictor in the photo or structural MRI setting.

### 7.9 Expected error hierarchy

| Source | Expected error | Status |
|--------|---------------|--------|
| Full-face photograph (controlled) | ~3 years MAE | Strong quantitative anchor |
| Whole-brain structural MRI | ~4–6 years MAE | Strong quantitative anchor |
| Adult skeletal morphology | ~6–14 years | Strong quantitative anchor (forensic) |
| Brain vascular burden (WMH) | — | Strong cue, no standalone MAE |
| Skin tone, texture, wrinkles | — | Strong cue, no standalone MAE |
| Periocular region / sclera | — | Strong cue, no standalone MAE |
| Facial fat / MRI soft tissue | — | Strong cue, no standalone MAE |
| Teeth (in standard photos) | — | Weak cue, no standalone MAE |

### 7.10 Healthy aging heterogeneity

Even healthy aging does not proceed identically. Longitudinal MRI evidence shows rates of brain atrophy accelerate in some structures after ~**70 years**, especially temporal cortex, hippocampus, and amygdala.

- Up to roughly **65–70 years**: healthy aging is constrained enough that age prediction from global anatomical signals remains relatively stable.
- After ~70 years: divergence in trajectories grows substantially. Some healthy individuals retain structurally "younger" profiles than chronological peers.

The practical implication: age prediction becomes harder not only because models are imperfect, but because the biology itself becomes more heterogeneous in late life.

### 7.11 Working conclusion for the MRI-to-face project

| Pipeline | Expected regime |
|----------|----------------|
| Whole-brain MRI → brain age | ~4–6 years MAE |
| Real full-face photograph → face age | ~3 years MAE (favorable conditions) |
| MRI-derived face render → face age | Unknown; likely worse than both above |

The core tension:

> The most age-informative facial cues (skin appearance, eye-region detail, soft tissue/fat distribution) are also the cues **least grounded in structural MRI** and **most vulnerable to synthetic hallucination** by generative models.

If a generative model invents plausible-looking skin texture, scleral color, or under-eye shadows — it is simultaneously inventing a substantial part of the age signal. This makes validation against brain age from the same scan especially important: it is the only ground truth that is not contaminated by the generative model's own priors.

---

## 8. Open Model Landscape

### Overview

Practical fit assessment for the MRI-to-face / MRI-to-brain pipeline. *Integration readiness* describes what can realistically be tested in this repository, not paper quality alone.

| Model | Task | Input | Training data | Public access | Testable now | Notes |
|-------|------|-------|---------------|---------------|--------------|-------|
| FaceAge | Face age / biological age | RGB face crop | 56,304 age-labelled IMDb-Wiki-derived images; curated UTK validation | Paper + GitHub + weights via Drive | Yes — existing wrapper | Best match to MRI-rendered face PNGs |
| MiVOLO | Face age + gender | RGB face-only or face+body | IMDB-cleaned, UTKFace, Lagenda | Paper + GitHub + HuggingFace checkpoints | Yes — new wrapper needed | Use face-only checkpoints; body-aware models are a poor fit for MRI renders |
| FAHR-Face | Foundation face-health / face-age | RGB face photos | Large-scale pretraining described in preprint | Paper visible; no confirmed official checkpoint path | No | Promising FaceAge follow-on; monitor for release |
| SFCN | Brain age | T1w MRI | UK Biobank 14,503 subjects; weights on 12,949 training subjects | Paper + GitHub + pretrained weights | Yes — existing wrapper | Existing wrapper provisional; age-bin decoding needs verification |
| MIDIconsortium BrainAge | Sequence-specific brain age | T1w, T2w, FLAIR, DWI, SWI | >23,000 axial T2 head MRIs from two UK hospitals | Paper + GitHub inference code | Yes — new wrapper needed | Best open sequence-specific clinical baseline family |
| SynthBA | Robust brain age | T1w, T2w, FLAIR | Synthetic MRIs from healthy-subject segmentations | Paper + GitHub + packaged checkpoints | Yes — new wrapper needed | Strongest open protocol-agnostic candidate |
| BrainIAC | Foundation brain age | Structural MRI (T1w downstream) | SSL on 32,015 MRIs; brain-age benchmark on 6,249 T1w | Paper + GitHub + checkpoint link | Yes — new wrapper needed | Most ambitious open foundation-model baseline |
| Kim et al. 2025 | Clinical brain age | Axial 2D T1w | 8,681 3D T1 + 24 public datasets | Paper only; code on reasonable request | No | MAE 2.73 yr on clinical T1 after bias correction |
| Barbano et al. 2023 | Multi-site robust brain age | T1w MRI | OpenBHB multi-site challenge | Paper + GitHub code; no confirmed pretrained weights | No | Strong site-robustness result; watch for weight release |

### 8.1 Face-age models: practical fit

**FaceAge** (Bontempi et al., *Lancet Digital Health* 2025)
- Input: standard RGB face photography via detected or pre-cropped face
- Training set: IMDb-Wiki-derived curated set (56,304 images), curated UTK validation/demo data
- Already wired into this repo (`src/face_age.py`); expected input closest to rendered face PNGs from T1 MRI
- Caveat: biological-age model, not a pure chronological-age benchmark — larger deviations from chronological age are part of the design

**MiVOLO** (WildChlamydia)
- Input: face-only RGB (also face+body, but body checkpoints are irrelevant for MRI renders)
- Official repo: face-only MAE ~4.22–4.29 yr on IMDB-cleaned and UTKFace checkpoints
- Best open comparator if the goal is pure chronological face-age estimation from 2D renders
- Requires new wrapper; use face-only checkpoints exclusively

**FAHR-Face / FAHR-FaceAge**
- Next-generation follow-on from the same AIM-Harvard group as FaceAge
- No confirmed official checkpoint release as of this review — monitor [AIM-Harvard/FaceAge](https://github.com/AIM-Harvard/FaceAge) repo for updates

### 8.2 Brain-age models: modality coverage and practical fit

| Model | T1w | T2w | FLAIR | Other | IXI fit | SIMON fit |
|-------|-----|-----|-------|-------|---------|-----------|
| SFCN | ✅ | — | — | — | T1 after skull-strip + conform | T1 if exported correctly |
| MIDIBrainAge | ✅ | ✅ | ✅ | DWI, SWI | T1 + T2 branches | T1 branch |
| SynthBA | ✅ | ✅ | ✅ | — | T1 + T2; PD not supported | T1 in principle |
| BrainIAC | ✅ | — | — | — | T1 branch | T1 branch |

**SynthSeg** is used in this repository as a robust segmentation engine (FreeSurfer 7.4+, `mri_synthseg`), not as a standalone brain-age model. Frame as a segmentation-to-regression component.

**SynthBA** preprocessing: built-in skull stripping and alignment unless skipped. Most attractive option for cross-scanner and cross-protocol work (directly relevant to SIMON).

**BrainIAC**: strongest open foundation-model context; more integration work than SynthBA but officially accessible. Brain-age downstream MAE ~6.55 yr at 20% fine-tuning data — demonstrates few-shot generalization rather than raw accuracy.

---

## 9. Research Questions and Forward-Looking Work

### 9.1 Chronological vs biological age: which model for which task?

| Track | Model | Target | MAE |
|-------|-------|--------|-----|
| Brain — chron | SFCN | Chronological age | 2.14 yr (UK Biobank in-distribution) |
| Brain — chron | Kim et al. 2025 | Chronological age | 2.73 yr after bias correction; 2.23 yr ensemble |
| Face — chron | MiVOLO | Apparent/chronological age | 4.22–4.29 yr (IMDB-cleaned, UTKFace) |
| Face — bio | FaceAge | Biological / health-associated age | Not comparable — designed for gap, not chron MAE |

For a fair experiment: run two parallel comparisons on the same non-defaced cohort.
- **Chronological-age track**: SFCN vs MiVOLO
- **Biological-gap track**: FaceAge vs brain-age-gap models assessed for disease sensitivity

### 9.2 Who shows the largest age gap?

| Pattern | Evidence | Interpretation |
|---------|----------|----------------|
| Largest face gap | Cancer patients +4.79 yr (FaceAge) | Face-side gap especially informative under systemic illness |
| Large brain gap — inflammatory | RRMS: mean gap 13.0 ± 14.7 yr; MS: 4.4 ± 6.6 yr | MS is one of the clearest structural aging-like settings |
| Large brain gap — psychiatric | Schizophrenia: 6–8 yr older | Severe psychiatric disease shifts structural brain aging |
| Moderate brain gap — neurodegeneration | AD: mean corrected gap 3.10 yr (Kim 2025) | Neurodegenerative disease shifts brain older, less dramatically |
| Best agreement | Cognitively unimpaired (Kim 2025): mean corrected gap 0.09 yr | Best agreement in healthy, bias-corrected, in-distribution subjects |

The unresolved question for this repository: do `face_age_gap` and `brain_age_gap` co-vary inside the same person when both are derived from the same MRI session? The literature is mixed (Belsky: yes at midlife; Cole 2020: no in UK Biobank). That tension is precisely the gap this project addresses.

### 9.3 Would 3D face reconstruction help?

| Claim | Status |
|-------|--------|
| Direct 2D face age estimation is mature | Strongly supported |
| 2D→3D reconstruction exists | Strongly supported (DECA, EMOCA, 3DDFA_V2) |
| 2D→avatar→age outperforms direct 2D | Not established |

Likely advantages of a 3D avatar branch: better pose control, cleaner geometry/texture separation, multi-view rendering from one reconstructed shape. Likely risks: reconstruction smooths age-relevant high-frequency detail; reconstruction error adds noise before the age model starts; NoW benchmark measures geometry fidelity, not age-signal preservation.

**Key distinction**: MRI-derived face surfaces are closer to *true 3D facial data* (Chen et al. 2015: ±6 yr individual variation in facial age from 3D scans; Matthews et al. 2018: MAE 1.19 yr in children from 3D facial prototypes) than to a selfie-to-avatar pipeline. This makes a direct MRI-surface age model scientifically more attractive.

### 9.4 Non-defaced MRI datasets: policy landscape

| Dataset | Face available | Age/cohort | Modalities | Fit for this project |
|---------|---------------|------------|------------|---------------------|
| IXI | Yes — 581 non-defaced T1w analyzed in defacing literature | Healthy, ~20–86 yr | T1, T2, PD, MRA, DTI | **Best open testbed** — already aligned with this repo |
| OASIS-3 | Raw head images present, but data-use terms explicitly prohibit facial recognition and derivative reconstruction | Older adults, normal aging + AD | Multimodal MRI + PET | Promising but policy-sensitive |
| ADNI | ADNI4 de-faces all applicable images before public release; legacy ADNI status varies by release | Older adults, MCI + AD | Structural MRI, FLAIR, PET, clinical | Verify release-level before any face work |
| ABIDE / ABIDE II | Contributors removed PHI including faces; ears also removed | Pediatric/young-adult autism/control | Structural MRI, rs-fMRI, some DWI | Brain-side benchmarking only |

Practical ranking: IXI → OASIS (with DUA verification) → legacy ADNI (release-level verification) → ABIDE (brain only).

### 9.4b Open test–retest T1 cohorts for OOD validation

For SIMON-like reproducibility evaluation, deliberately picking small/obscure cohorts that are unlikely to be in foundation-model pretraining (i.e. not HCP/IXI/OASIS/ADNI/UKBB). Full table and access notes in [`test_retest_datasets.md`](test_retest_datasets.md).

| Dataset | N | Sessions | Interval | License |
|---|---|---|---|---|
| **Cao 2015 (BNU1)** | 57 | 2 | ~6 weeks | CC-BY-4.0 |
| **Huang 2016 (BNU2)** | 61 | 2 | 103–189 d | CC-BY-4.0 |
| Maclaren ds000239 (OpenNeuro) | 3 | 40 each across 20 sessions | 31 d | CC0 |
| B-Q MINDED / TRCSMM (Zenodo) | ~20 | 2 (Skyra + PrismaFit) | same day | open |
| GSP retest (Harvard Dataverse) | 69 | 2 | ~77 d | DUA |

Top picks for our project: **Cao 2015 + Huang 2016** combined give 118 subjects across short and long retest intervals on CC-BY-4.0 (no DUA friction). Maclaren is the right cohort to report intra-subject prediction SD with ~40 scans per subject. B-Q MINDED isolates pure scanner variance via same-day cross-scanner pairing.

### 9.5 Future experiments

| Experiment | Minimum data | Success criterion |
|-----------|-------------|------------------|
| Direct chron-age benchmark | Non-defaced cohort, T1 MRI, shared train/test across face + brain | Brain vs face MAE and calibration on identical subjects |
| Gap phenotype analysis | Chron age + diagnosis, BMI, smoking, scanner/site | Clear enrichment pattern, not scanner-driven noise |
| 3D avatar ablation | MRI face surface + direct 2D render + one open mesh-based age pipeline | Avatar branch beats or complements direct 2D |
| Multi-view rendering | Reconstructed MRI face mesh + renderer + one face-age model | Multi-view aggregation improves robustness or calibration |
| Dataset expansion | Verified OASIS or legacy ADNI access | Disease contrast or longitudinal signal added without DUA violation |
| Joint modeling | Paired chron/face/brain age + downstream outcomes | Combined model outperforms either branch alone |

---

## References

1. Bontempi et al. (2025). Artificial intelligence–based biological age from facial photographs. *Lancet Digital Health*. PMID: 40345937. DOI: 10.1016/S2589-7500(25)00042-1.
2. Cole et al. (2020). Multimodality neuroimaging brain-age in UK Biobank. *NeuroImage*, 210, 116504.
3. Belsky et al. (2019). Brain aging and accelerated biological aging at midlife. *Molecular Psychiatry*, 27, 768–775.
4. Christensen et al. (2009). Perceived age as clinically useful biomarker of ageing. *BMJ*, 339, b5262.
5. Peng et al. (2021). Accurate brain age prediction with lightweight deep neural networks. *Medical Image Analysis*, 68, 101871.
6. Billot et al. (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. *Nature Methods*, 20, 1737–1745.
7. Tak et al. (2026). BrainIAC: A foundation model for brain imaging. *Nature Neuroscience*. DOI: 10.1038/s41593-026-02202-6.
8. Rothe et al. (2018). Deep expectation of real and apparent age from a single image without facial landmarks. *IJCV*, 126(2–4), 144–157.
9. Franke et al. (2010). Estimating the age of healthy subjects from T1-weighted MRI scans using kernel methods. *NeuroImage*, 50(3), 883–892.
10. Xifra-Porxas et al. (2021). Estimating brain age from structural MRI and MEG data: Insights from dimensionality reduction techniques. *NeuroImage*, 231, 117822. DOI: 10.1016/j.neuroimage.2021.117822.
11. Bailey & Vidoli (2023). Age-at-Death Estimation: Accuracy and Reliability of Common Age-Reporting Strategies in Forensic Anthropology. *Forensic Sciences*, 3(1), 179–191. DOI: 10.3390/forensicsci3010014.
12. Zhang et al. (2023). Exploring artificial intelligence from a clinical perspective: A comparison and application analysis of two facial age predictors trained on a large-scale Chinese cosmetic patient database. *Skin Research and Technology*, 29(7), e13402. DOI: 10.1111/srt.13402.
13. González-Alvarez & Sos-Peña (2023). The role of facial skin tone and texture in the perception of age. *Vision Research*, 213, 108319. DOI: 10.1016/j.visres.2023.108319.
14. Hsieh et al. (2023). Colour information biases facial age estimation and reduces inter-observer variability. *Scientific Reports*, 13, 13224. DOI: 10.1038/s41598-023-39902-z.
15. Porcheron et al. (2013). Aspects of facial contrast decrease with age and are cues for age perception. *PLOS ONE*, 8(3), e57985. DOI: 10.1371/journal.pone.0057985.
16. Russell et al. (2014). Sclera color changes with age and is a cue for perceiving age, health, and beauty. *Psychology and Aging*, 29(3), 626–635. DOI: 10.1037/a0036142.
17. Gosain et al. (2005). A volumetric analysis of soft-tissue changes in the aging midface using high-resolution MRI. *Plastic and Reconstructive Surgery*, 115(4), 1143–1152. DOI: 10.1097/01.PRS.0000156333.57852.2F.
18. Adserias-Garriga (2023). Age-at-Death Estimation by Dental Means as a Part of the Skeletal Analysis. *Forensic Sciences*, 3(2), 357–367. DOI: 10.3390/forensicsci3020027.
