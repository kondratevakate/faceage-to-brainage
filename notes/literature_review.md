# Literature Review: Face Age & Brain Age — Are They the Same Clock?

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
