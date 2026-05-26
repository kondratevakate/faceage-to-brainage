# SOTA Experimental Design for Face-Age and Brain-Age Prediction (2024–2026)

A competitive design brief for the same-scan face-age vs brain-age project.

---

## 1. SOTA face-age papers (2024–2026)

### 1.1 FaceAge — Bontempi et al., *Lancet Digital Health*, 2025
- **Venue/year**: *Lancet Digital Health*, May 2025. DOI: 10.1016/S2589-7500(25)00042-1. PubMed 40345937.
- **Architecture**: Inception-ResNet-v1 (FaceNet backbone), fine-tuned for age regression.
- **Training set**: 58,851 presumed-healthy individuals aged 60+. 56,304 from IMDb-Wiki for training; 2,547 UTKFace for initial validation.
- **Evaluation / OOD**: Clinical utility on 6,196 cancer patients from MAASTRO (NL), Harvard Thoracic, and Harvard Palliative cohorts — these are true OOD sets (different demographics, photo conditions, clinical context).
- **Split protocol**: Subject-disjoint; held-out clinical cohorts are demographically and geographically disjoint from training.
- **Reported metrics**: Designed as a *biological-age* estimator, not a chronological MAE benchmark. Per-decade HR 1.151 for overall survival in pan-cancer cohort (p=0.013, n=4,906). Palliative-cohort 6-month survival AUC improved 0.74 → 0.80 (p<0.0001). Cancer patients appeared 4.79 years older than chronological age.
- **Bias-correction**: Cohort-level calibration to reference non-cancer distribution; no per-subject age-bias regression as in brain age.
- **Open-source**: Yes — `AIM-Harvard/FaceAge` on GitHub, weights via Google Drive.
- **Face preprocessing**: MTCNN face detection → aligned face crop (FaceNet pipeline, 160×160 standard).

### 1.2 FAHR-Face / FAHR-FaceAge — Haugg et al., *arXiv:2506.14909*, June 2025
- **Venue/year**: arXiv preprint 2506.14909, AIM-Harvard group (same lab as FaceAge).
- **Architecture**: Foundation model; age-estimation head (FAHR-FaceAge) and survival head (FAHR-FaceSurvival).
- **Training set**: >40 million facial images for self-supervised pretraining, then two-stage age-balanced fine-tuning on 749,935 public images. Survival head fine-tuned on 34,389 cancer-patient photos.
- **Evaluation / OOD**: Public age datasets + cancer cohorts.
- **Reported MAE**: **5.1 years** on public datasets — claimed lowest among benchmarks across the full human lifespan. Survival: top-quartile risk has >3× mortality of bottom quartile.
- **Bias correction**: Age-balanced fine-tuning (a design-time rather than post-hoc calibration).
- **Open-source**: Paper posted; official checkpoint release not yet confirmed. Watch `AIM-Harvard/FaceAge`.
- **Face preprocessing**: Same AIM pipeline as FaceAge (MTCNN + aligned crop).

### 1.3 MiVOLO v2 — SaluteDevices, 2024
- **Venue/year**: Technical report / GitHub release 2024; v1 in *arXiv:2307.04616*.
- **Architecture**: Multi-input Vision Transformer combining face and body crops.
- **Training set v2**: 807,694 images (390,730 M / 416,964 F) — extended ~40% over v1, with targeted addition of v1's error cases. Sources: IMDb-cleaned, UTKFace, Lagenda.
- **Evaluation / OOD**: IMDB-cleaned, UTKFace, Lagenda benchmarks (still mostly celebrity/web distribution).
- **Reported MAE**: ~**4.22–4.29 years** on IMDB-cleaned and UTKFace (face-only checkpoints). In an independent 2025–2026 VLM benchmark, MiVOLO scored **MAE 5.10** and was the only non-LLM architecture competitive with Gemini 3 Flash (MAE 4.32).
- **Bias correction**: None beyond training-time balancing.
- **Open-source**: Yes — `WildChlamydia/MiVOLO`, HuggingFace checkpoints.
- **Face preprocessing**: YOLO-based detector; body branch expects person crop; face-only models need face crop. Body branch is irrelevant for MRI renders.

### 1.4 Paplhám & Franc — "A Call to Reflect on Evaluation Practices for Age Estimation", *CVPR 2024*
- **The most important methodological paper** in the field for 2024–2026. Shows that almost all reported face-age gains come from evaluation leaks, not model improvements.
- **Finding**: only ~10% of MORPH-II papers use subject-exclusive (identity-disjoint) splitting. Under subject-exclusive splits with matched backbone/data, virtually all "SOTA" losses collapse to near-identical MAE.
- **Implication for our paper**: any face-age MAE we report must be (a) subject-disjoint, (b) under a fixed training setup, (c) benchmarked against their unified pipeline or explicitly framed as biological-age (as FaceAge is).
- **Open-source**: `paplhjak/facial-age-estimation-benchmark` on GitHub.

### 1.5 FaRL-based age estimators (ongoing 2024–2025)
- **Backbone**: FaRL ViT-B/16 pretrained on 50M image–text pairs from LAION-Face. Paplhám CVPR 2024 uses FaRL as a reference backbone.
- **Reported MAE**: competitive with best supervised models when backbone, loss, and split are held constant; differences across age-specific losses are within benchmark noise.

### 1.6 "Out of the box" VLM benchmark — Karaev et al., *arXiv:2602.07815*, 2026
- Compared open-weight VLMs (Gemini, GPT-4o, Claude, LLaVA) against specialised face-age models zero-shot.
- **Best result**: Gemini 3 Flash Preview, MAE 4.32. Best non-LLM: MiVOLO, MAE 5.10. A 15% gap, closing fast.
- Relevance: worth including a zero-shot VLM baseline on MRI renders as a sanity check.

### 1.7 UniFace (Yakhyo 2024+) — production face-analysis library
- Not a new model — a packaged ONNX runtime library bundling detection, landmarks, age, parsing. Useful as an easy additional wrapper for integration tests, but not a novel scientific baseline.

### Face-age takeaways for this project
- **Use FaceAge as primary (biological)** and MiVOLOv2 face-only or a FaRL-backbone model as **chronological** comparator.
- **Do NOT** trust legacy published MAE numbers that predate Paplhám CVPR 2024; require subject-disjoint evaluation.
- **Face preprocessing must match the model's training**: FaceAge/FaceNet = 160×160 aligned crop via MTCNN; MiVOLO = YOLO+face-only; mismatching the pipeline can inflate MAE by several years.

---

## 2. SOTA brain-age papers (2024–2026)

### 2.1 SynthBA — Puglisi et al., IEEE MetroXRAINE 2024 (arXiv:2406.00365)
- **Training**: domain-randomised synthetic MRIs generated from healthy-subject segmentations; random contrast each iteration → contrast-agnostic.
- **Evaluation**: multiple external datasets with heterogeneous protocols; SynthBA-g (general) generalises where baseline models blow up to 7–20 yr MAE on external scanners.
- **Relevance**: **best-positioned open baseline** for cross-scanner work (exactly our SIMON case).
- **Open-source**: Yes, `LemuelPuglisi/SynthBA`, T1/T2/FLAIR supported.
- **Bias correction**: Not inherent — must be applied post-hoc.

### 2.2 Wood et al. — MIDIconsortium BrainAge, *Human Brain Mapping*, 45(4):e26625, 2024
- **Training data**: >23,000 axial T2 head MRI from two UK hospitals + other sequence branches.
- **Held-out test MAE**: 2.85 yr (T2 axial), ensemble **2.44 yr**.
- **External IXI (transfer)**: T2 axial MAE **3.08 yr**, r=0.97 — strong cross-site generalisation.
- **Open-source**: Yes, `MIDIconsortium/BrainAge` on GitHub with pretrained weights.
- **Bias correction**: Standard linear (Cole/Smith style) applied before reporting.

### 2.3 Kim et al., *npj Aging* 2025
- **Training**: 8,681 3D research-grade T1 + 24 public datasets.
- **Target**: 2D axial clinical T1 slices.
- **Held-out**: 175 cognitively-unimpaired subjects; **MAE 3.68 yr raw, 2.73 yr after age-bias correction** (Pearson r=0.918). Ensemble reaches 2.23 yr.
- **Disease gap**: AD > CU by 3.1 yr (p<0.001), PD gap associated with disease progression.
- **Open-source**: Code on reasonable request. Not a drop-in baseline.

### 2.4 BrainIAC — Tak et al., *Nature Neuroscience*, 2026
- **Training**: SimCLR SSL on 32,015 brain MRIs; overall evaluation across 48,519 MRIs, 34 datasets, 10 conditions, 7 downstream tasks.
- **Brain-age downstream**: MAE ~**6.55 yr** at 20% fine-tuning data — less accurate than SFCN on UKB, but demonstrates few-shot generalisation.
- **Open-source**: Yes, `AIM-KannLab/BrainIAC` with checkpoint.
- **Relevance**: same lab as FaceAge — natural pairing for a "same-model-family" face+brain age analysis.

### 2.5 SFCN (Peng et al., *Medical Image Analysis* 2021) — still the UKB MAE leader
- **MAE 2.14 yr** on UK Biobank held-out. Degrades to 9–10 yr on CamCAN (scanner shift).
- **Open-source**: `ha-ha-ha-han/UKBiobank_deep_pretrain`.
- **SFCN-reg (HBM 2025)** variants add augmentation and harmonisation, narrowing the UKB→external gap.

### 2.6 Barbano et al. — Contrastive regression, ISBI 2023 + Anatomical Foundation Models, arXiv:2408.07079 (2024–2025)
- **Dataset**: OpenBHB.
- **Result**: Contrastive site-invariant regression beats pure supervised baselines on site-invariance metric while preserving age MAE. No published pretrained weights yet.

### 2.7 Dular & Špiclin — *Imaging Neuroscience* 2025 ("Deep Models Need a Hand to Generalize")
- Evaluated 4 pretrained brain-age architectures on 4 external datasets. Shows uniform preprocessing + bias correction is required for fair benchmarking; models with better in-distribution MAE do not automatically generalise better.

### 2.8 Brain-age + UK Biobank external pipelines — Yin et al., *NeuroImage* 2025 (UK Biobank study) and *Communications Medicine* 2025
- UKB-trained ViT achieves MAE **2.68 yr** (UKB held-out), **2.99–3.20 yr** on ADNI/PPMI after harmonisation.
- Reinforces that "<3 yr MAE" is the 2025 ceiling for in-distribution healthy adults only.

### Harmonisation standards — current consensus (2024–2025)
- **ComBat** (Fortin, 2017) — voxel-statistics linear harmonisation. On the OpenBHB leaderboard, ComBat removes most site effects but **degrades age MAE**; not recommended on quasi-raw data.
- **NeuroHarmonize** (Pomponio 2019) — non-linear age-trend-aware ComBat; standard for lifespan datasets. Official Python package, still maintained.
- **neuroCombat (R/Python)** — active, recommended for feature-level harmonisation.
- **Contrastive / adversarial site removal** (Barbano 2023, Dufumier 2024) — now the research frontier, but not yet standard practice.
- **Practical rule of thumb** (2024–2025): if you train on a dataset with >5 sites, apply NeuroHarmonize on derived features or adopt site-invariant training (SynthBA, Barbano); always report both harmonised and raw numbers.

### Age-bias correction — still universal
- **Smith et al. (2019)** regression-based correction on a held-out control sample is effectively mandatory when reporting PAD. Beheshti 2019 and de Lange 2020 are equivalent linear variants.
- **Age-level bias correction** (Zhang et al., *Frontiers in Neuroscience* 2023) refines Smith's approach for age-stratified bias.
- Report both **raw and bias-corrected MAE** — Kim 2025 and Wood 2024 both do this. Papers that only report corrected MAE are increasingly flagged as inflating performance.

---

## 3. Gold-standard OOD evaluation protocols

### 3.1 Multi-cohort external validation
Current standard is **≥2 external held-out datasets** with disjoint sites. Wood 2024 uses IXI as external validation; Kim 2025 uses 24 public datasets; SynthBA uses multiple external scanners per contrast. BASE (Mouches 2023) codifies this: multi-site + new-unseen-site + test-retest + longitudinal.

### 3.2 Sample size for a reliable MAE claim
Minimums seen in accepted papers:
- Internal held-out: **≥500** for a ±0.2 yr MAE CI at r≈0.9 correlation.
- External site: **≥100** per scanner for a site-stratified MAE.
- **Repeatability test-retest**: ≥20 subjects scanned twice to estimate ICC with usable 95% CI; ≥40 preferred.

### 3.3 Travelling-subject / SIMON-like cohorts
- **SIMON** (*Single Individual volunteer for Multiple Observations across Networks*) — 1 subject × ~36 scanners × 73–99 sessions, 17 years longitudinal. Foundational for segmentation/morphometry reproducibility; used in 2025 benchmarks such as *arXiv:2504.15931* (Dular et al.) to quantify inter-scan variability with Dice, Surface Dice, HD95, MAPE.
- **SRPBS Traveling Subjects** — 9-site test-retest on healthy volunteers; commonly paired with SIMON.
- **CamCAN 2nd wave** (Rescan, medRxiv 2025) — test-retest across visits.
- **Kennedy Krieger, MASSIVE, HCP test-retest** — useful smaller retest sets.
- **Use pattern in SOTA**: run the model once per scan, compute PAD per scan, then report ICC(2,1) across scanners, coefficient of variation of PAD, and absolute PAD range per subject.

### 3.4 Standard statistics
- **ICC(2,1) absolute agreement** under a 2-way mixed-effects model (McGraw & Wong). Report with 95% CI. Koo-Li 2016 thresholds: <0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good, >0.9 excellent.
- **Bland-Altman** plot + limits of agreement for same-subject repeats.
- **Repeatability coefficient** RC = 1.96·√2·SD(differences) — the value below which 95% of same-subject differences will fall.
- **Coefficient of variation** (CV) for PAD across scanners.
- **Comparison of brain-age-package ICCs (Dular et al. 2023–2024)**: brainageR, pyment, mccqrnn achieved ICC 0.94–0.98 — this is the bar for a publishable "reproducible brain age" claim.

### 3.5 Raw vs bias-corrected reporting
- **Report both**. Papers that only show bias-corrected metrics are increasingly criticised because age-bias correction collapses PAD variance and artificially shrinks MAE even for poor models. Dular & Špiclin 2025 explicitly flags this.
- **PLOS Biology 2025 (Hahn et al.)** showed brain-age models with *worse* in-distribution MAE often have higher clinical sensitivity — so optimising only for corrected MAE may hurt the biomarker.

---

## 4. Precedent for same-scan face-vs-brain age comparison

### 4.1 Is there a direct precedent?
**No.** A systematic search (PubMed, arXiv 2024–2026, MICCAI/MIDL proceedings, bioRxiv/medRxiv) returns **zero papers** that (a) render a facial surface from a subject's MRI, (b) score it with a photograph-trained face-age model, (c) compare that prediction to a brain-age model run on the same MRI volume.

### 4.2 Implicit / partial precedents
- **De-identification / re-identification literature** renders faces from MRI specifically to attack privacy, not to score age. Schwarz et al. *NEJM* 2019; Abramian 2024 (*eClinicalMedicine*); *arXiv:2501.18834* (Pitfalls of defacing, 2025) — 3D rendered faces achieve 50–59% re-identification against public face-recognition models. These prove that **face recognition networks transfer from photo to MRI render** — direct evidence that a face-age network should too.
- **Impact of defacing on brain-age** (Schwarz 2021, Rubbert 2023) — measures MAE drift when the face is removed, but runs only brain-age, not face-age.
- **Belsky et al., *Mol Psychiatry* 2019** — brain-PAD at midlife correlates with blinded human raters' perceived facial age from photos (Dunedin cohort, n=954). This is the closest multimodal precedent, but uses (i) separately collected photos, (ii) human rather than AI face-age.
- **Cole et al., *NeuroImage* 2020** — brain-PAD vs perceived facial age in UK Biobank, null result. Perceived, not AI-derived.
- **Christensen *BMJ* 2009** (Danish twins): perceived facial age predicts mortality; no MRI.
- **FAHR-Face / FaceAge on cancer photos** — photos only; no MRI pairing.
- **BrainIAC** — MRI only; no face pairing (despite being the same lab as FaceAge).
- **ADNI4 / OASIS-3 / OASIS-4 / UK Biobank** — no publicly released paired standardised facial photograph for any of these cohorts as of April 2026. ADNI4 defaces releases; OASIS-3 DUA forbids facial reconstruction; UKB has no selfie enhancement.

### 4.3 Implication for the project
This is a genuinely empty niche. The nearest competitors (Belsky 2019; Cole 2020) used **rated/perceived** facial age and **separately collected** photos — never an AI score on an MRI-derived face. The defacing literature proves the face→photo-model transfer is technically feasible. **The contribution is defensible and un-scooped** as of April 2026.

---

## 5. Concrete recommendations for a competitive paper

### 5.1 Datasets (ranked top 3 beyond IXI + SIMON)
1. **OpenBHB** — 5,330 subjects, 71 sites, ages 6–88. Sets the community harmonisation/OOD standard. Internal/external/privateBHB protocol is already tooled on Codalab/RAMP. Adds statistical power and legitimises the OOD claim.
2. **CamCAN** — 639 subjects, ages 18–88, single-site but classic external validator; if phase 5 (Rescan, 2025) is accessible, adds longitudinal repeatability.
3. **Kennedy Krieger + NKI / OASIS-3** (as DUA permits) — for additional external sites and subject-level test-retest; OASIS-3 specifically gives older-adult coverage that IXI lacks.

Also strongly consider **ADNI (legacy, pre-ADNI4)** if any release still contains raw non-defaced volumes — adds disease contrast (CU vs MCI vs AD), which is what will make the paper clinically interesting.

### 5.2 Preprocessing pipeline likely missing in current notebooks
Given the IXI half-patient dropout problem, the preprocessing chain is probably under-robust. SOTA pipelines share these stages:
1. **N4 bias-field correction** (ANTs) — stable, required before any surface extraction.
2. **Orientation/voxel conform** (FreeSurfer `mri_convert --conform` or SynthBA's internal step). This is where most IXI drops happen when the default assumes LIA orientation.
3. **Skull+scalp mask via SynthStrip** (Hoopes 2022) or SynthSeg's robust mode — *do not* rely on BET.
4. **Face surface extraction** — 3DSlicer MarchingCubes on head mask; or `mri_watershed` output; Pydeface-inverse is *not* reliable.
5. **Render with fixed virtual camera** (pyrender / trimesh) — frontal + ±30° views, identical lighting, identical FOV across scans, to keep the face-age model input distribution stationary.
6. **MTCNN face detection + FaceNet 160×160 aligned crop** to match FaceAge training distribution. Never feed the raw render — FaceAge expects an MTCNN-aligned face.

The most likely cause of IXI dropout: SynthBA's internal skull-strip fails silently on a subset of volumes because of the orientation field or non-uniform intensity. **Fix**: run SynthStrip *before* SynthBA with `--no-strip` flag to SynthBA; log the exact failure stage per subject; target the fraction-lost under 2%.

### 5.3 Expected baselines for the brain side
- **SFCN** (UKB-pretrained) — the in-distribution reference.
- **SynthBA** — the cross-site reference (our primary).
- **BrainIAC** — the foundation-model reference (SSL).
- **Wood 2024 MIDIBrainAge** — the clinical-protocol-agnostic reference with IXI transfer numbers already published.
Running **all four** is the minimum the brain-age reviewer community will ask for.

### 5.4 Expected baselines for the face side
- **FaceAge** (primary — biological-age framing).
- **MiVOLOv2 face-only** (chronological baseline).
- **FAHR-FaceAge** *if* checkpoints release in time.
- Optional: a zero-shot VLM (e.g. Gemini 3 Flash, GPT-5-V) via API as sanity check — documents that our finding is not a specific-model artefact.

### 5.5 Minimum-acceptable statistical reporting
- **Raw MAE and bias-corrected MAE** with **95% bootstrap CIs** (≥10,000 bootstraps over subjects, not scans).
- **Pearson r and Spearman ρ** with 95% CIs.
- **ICC(2,1) absolute agreement** with 95% CI for SIMON repeat scans; report for both face-age and brain-age PAD.
- **Bland-Altman plots + repeatability coefficient** for same-subject scan pairs.
- **Coefficient of variation of PAD** across scanners per subject.
- **Correlation of face-age-gap vs brain-age-gap** — partial correlation controlling for chronological age, sex, site; report with 95% CI.
- **Site-stratified MAE table** (per dataset, per scanner vendor).
- Pre-register split / analysis plan on OSF before unblinding the SIMON correlation. This counters the reviewer concern that this project is a fishing expedition.

### 5.6 Sample size for a publishable correlation claim
- For the cross-subject face-gap vs brain-gap Pearson correlation: detecting r=0.2 at 80% power, α=0.05 two-tailed → **n≈194**. IXI alone (if preprocessing is fixed) should clear this.
- For the within-subject scanner reproducibility (SIMON): 99 scans on 1 subject gives very tight ICC CI for within-scanner, but **n=1 subject limits claims about between-subject reliability**. Pair SIMON with SRPBS Traveling Subjects or another ≥5-subject retest cohort.
- For a disease-contrast claim (if we add ADNI/OASIS): **≥50 cases and ≥50 controls per diagnostic group** per published effect-size norms.

### 5.7 The single strongest differentiating experiment
**"Same-scan scanner reproducibility of face-age vs brain-age PAD"** on SIMON + SRPBS Traveling Subjects, with:
- ICC(2,1) with 95% CI for both modalities on the identical subject × scanner matrix.
- Bland-Altman + repeatability coefficient per scanner vendor.
- Test whether face-age PAD is **more**, **less**, or **equally** scanner-invariant than brain-age PAD from SynthBA/SFCN.

This is publishable even if the correlation between face-gap and brain-gap is weak. A clean finding like "MRI-derived face-age ICC = 0.78 (0.70–0.85) vs brain-age ICC = 0.92 (0.88–0.95) across 36 scanners" is a **novel, quantified, unscooped result** with immediate methodological implications for any downstream use of face-age on neuroimaging data. It is also the most direct answer to the single sceptical question a reviewer will ask: *is the "face-age from MRI" signal stable enough to be a biomarker at all?*

### 5.8 Closing shape of the paper
Targeting MIDL 2026 (short paper) or *NeuroImage*: Methods (pipeline + failure-mode log), Reproducibility (SIMON + SRPBS), Correlation (IXI + OpenBHB), Clinical contrast (ADNI/OASIS-3 if accessible). Ship the pipeline as open code + a Docker image — reproducibility is the one thing the Paplhám CVPR 2024 and Dular 2025 critiques make unavoidable.

---

## References (beyond existing `literature_review.md`)

- Bontempi et al. (2025). FaceAge. *Lancet Digital Health*. DOI: 10.1016/S2589-7500(25)00042-1. PMID 40345937.
- Haugg et al. (2025). FAHR-Face. arXiv:2506.14909.
- Kuprashevich & Tolstykh (2023/2024). MiVOLO. arXiv:2307.04616. MiVOLOv2 release notes, SaluteDevices / WildChlamydia GitHub.
- Paplhám & Franc (2024). A Call to Reflect on Evaluation Practices for Age Estimation. *CVPR 2024*, 1196–1205. arXiv:2307.04570.
- Karaev et al. (2026). Out-of-the-box age estimation: VLMs vs specialised models. arXiv:2602.07815.
- Puglisi et al. (2024). SynthBA. *IEEE MetroXRAINE 2024*. arXiv:2406.00365.
- Wood et al. (2024). Optimising brain age estimation through transfer learning. *Human Brain Mapping* 45(4):e26625. DOI: 10.1002/hbm.26625. PMID 38433665.
- Kim et al. (2025). Deep learning brain age for routine clinical MRI. *npj Aging*. PMID 40730571. DOI: 10.1038/s41514-025-00260-x.
- Tak et al. (2026). BrainIAC. *Nature Neuroscience*. DOI: 10.1038/s41593-026-02202-6.
- Peng et al. (2021). SFCN. *Medical Image Analysis* 68:101871.
- Barbano et al. (2023). Contrastive learning for regression in multi-site brain age prediction. *ISBI 2023*. arXiv:2211.08326. (Anatomical Foundation Models follow-up: arXiv:2408.07079.)
- Dufumier et al. (2022). OpenBHB. *NeuroImage* 259:119637.
- Pomponio et al. (2020). NeuroHarmonize. *NeuroImage* 208:116450.
- Fortin et al. (2017). ComBat harmonisation for MRI. *NeuroImage* 161:149–170.
- Smith et al. (2019). Estimation of brain age delta… *NeuroImage* 200:528–539.
- Beheshti et al. (2019). Bias-adjustment in neuroimaging-based brain age frameworks. PMID 31795063.
- Zhang et al. (2023). Age-level bias correction in brain age prediction. PMC9860514.
- Dular & Špiclin (2025). Bias and generalizability of brain age prediction models. *Imaging Neuroscience*.
- Dular et al. (2024/2025). Benchmarking reproducibility of brain MRI pipelines on SIMON + SRPBS. arXiv:2504.15931.
- Schwarz et al. (2019). Identification of anonymous MRI participants with face-recognition. *NEJM* 381:1684–1686.
- Abramian et al. (2024). Re-identification of anonymised MRI head images. *eClinicalMedicine*. DOI: 10.1016/j.eclinm.2024.102509.
- Pitfalls of defacing whole-head MRI (2025). arXiv:2501.18834.
- Belsky et al. (2019). Brain aging and accelerated biological aging at midlife. *Molecular Psychiatry* 27:768–775.
- Cole et al. (2020). Multimodality neuroimaging brain-age in UK Biobank. *NeuroImage* 210:116504.
- Mouches et al. (2023). BASE: Brain Age Standardized Evaluation. PMID 38065279.
- Koo & Li (2016). Guideline for reporting ICC. *J Chiropr Med* 15(2):155–163.
- McGraw & Wong (1996). Forming inferences about some intraclass correlation coefficients. *Psychol Methods* 1(1):30–46.
- Yin et al. (2025). UK Biobank brain clocks. *NeuroImage* 313.
- Hahn et al. (2025). Brain-age models with lower accuracy have higher sensitivity for disease detection. *PLOS Biology*. DOI: 10.1371/journal.pbio.3003451.
