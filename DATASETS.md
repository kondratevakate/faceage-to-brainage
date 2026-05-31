# Datasets relevant to face-age ↔ brain-age from MRI

A curated catalog of imaging datasets that matter for this project. Sorted by *what they let you do for face-from-MRI work*, not by acquisition modality.

The project's central premise is **paired face-age and brain-age estimation from the same non-defaced T1 volume** ([README](README.md)). That premise needs data along four axes:

1. **MRI with face preserved** — the actual input modality. Most public brain MRI is defaced; we need the small subset that isn't.
2. **Face photo datasets with chronological age labels** — for sanity-checking the face-age branch against the canonical photo-based literature (DEX, MiVOLO, FaceAge).
3. **Multi-view / 3D face datasets** — for testing whether multi-view MRI renders (the nine-view input in this project) carry signal comparable to true multi-camera capture.
4. **Cosmetology / multi-spectral skin imaging** — calibrates what age signal MRI-derived face renders **can't** reach (skin tone, vascular pattern, UV damage) and where the morphological-only ceiling sits.

Plus a fifth axis — **volumetric tissue segmentation references** — for the segmentation steps in the pipeline.

---

## 1. Brain MRI datasets with face preserved (no-deface)

These are the few public T1 collections where the face render pipeline can actually be applied directly. Sort: chronological (newest first).

| Year | Dataset | N | Modalities | Defaced? | Access | Notes |
|---|---|---|---|---|---|---|
| 2024 | [**SHARM**](https://arxiv.org/pdf/2309.06677) (Segmented Head Anatomical Reference Models) | 196 | T1 + 15 tissue labels (skin, fat, eyeballs, cartilage, mandible…) | **No** | Open | Best ground-truth for tissue-level face structures |
| 2024 | [**GRACE**](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00090/119208/) | 177 | T1 + 11 tissues (WM, GM, eyes, CSF, air, vessel, bone, skin, fat) | **No** | Open model + reference segs | Built for older adults; useful for the high-age tail |
| 2025 | [**Full-Head Segmentation w/ abnormal anatomy**](https://arxiv.org/abs/2501.18716) | 98 | T1 + 7 tissues; stroke / DOC clinical cases | **No** | Open | Tests how the face render survives pathology |
| 2014 | [**studyforrest**](https://www.studyforrest.org/) ([OpenNeuro ds000113](https://openneuro.org/datasets/ds000113)) | 20 | 7T T1 + T2 + DTI + SWI + AV-movie fMRI | **No** | OpenNeuro, PDDL | 7T resolution → very high-fidelity face renders |
| 2008+ | [**IXI Dataset**](https://brain-development.org/ixi-dataset/) | ~600 (581 T1) | T1, T2, PD, MRA, DTI (15 dir) | **No** | CC BY-SA 3.0 | **Classic open non-defaced collection.** Three London sites (Hammersmith 3T Philips, Guy's 1.5T Philips, IoP 1.5T GE). Age + sex metadata included. *This is the natural primary training set for face-from-MRI work.* |
| varies | [**OASIS-1**](https://www.oasis-brains.org/) | 416 | T1 (cross-sectional 18–96 yr) | **Originally no, distributed with optional defacing** | Open | Some downloads ship pre-defaced; check release version |
| 2019 | [**OASIS-3**](https://www.oasis-brains.org/) | 1098 | T1, T2, DTI, rsfMRI, ASL, PET | **Distributed both ways** | Application | Some sessions pre-defaced; raw scans available with DUA. Heavily studied for defacing-effect papers |
| — | [**Kirby-21**](https://www.nitrc.org/projects/multimodal/) | 21 | Multimodal | **No** | Open | Small but used as the canonical defacing-evaluation cohort |
| — | [**MIDA model** (Iacono 2015)](https://itis.swiss/virtual-population/regional-human-models/mida-model/) | 1 (template) | 7T, 153 structures | **No** (single donor) | Open | Reference phantom, not a cohort — useful for sanity-checking renders |

**Practical guidance for this project:**
- IXI is the obvious first cohort (~600 non-defaced T1 with age labels, CC BY-SA).
- SHARM / GRACE give per-pixel tissue ground truth — useful if the face branch wants supervised tissue labels rather than just rendered mesh.
- OASIS-3 needs care: some distributions are defaced. Pull raw + check `if facemask in filename`.
- For high-age tail and clinical edge cases, lean on GRACE (older adults) and Full-Head-Abnormal (pathology).

**Reference reading on defacing effects** (relevant because most candidate cohorts ship defaced and the bias matters):
- [Schwarz et al. NEJM 2019](https://www.nejm.org/doi/full/10.1056/NEJMc1908881) — 70% re-identification accuracy via Azure Face API on rendered MRI faces. The whole reason this project even works.
- [Buimer et al. 2023](https://www.biorxiv.org/content/10.1101/2023.04.28.538724) — brain-age models predict **better** from the face portion than from the brain. Direct motivation for paired face/brain age.
- [Schwarz mri_reface 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10502400/) — defacing distorts Freesurfer outputs and AD biomarkers.
- [Re-identification risk with whole-head defacing 2025](https://arxiv.org/pdf/2501.18834) — residual risk after defacing on OASIS-1.

---

## 2. Face photo datasets with age labels

Use these to (a) baseline what the face-age literature ceiling is on photos, (b) cross-validate AI face-age models you run on MRI renders against the same models on real photos of similar age range.

| Year | Dataset | N images / N subjects | Age labels | Access |
|---|---|---|---|---|
| 2015 | [**IMDB-WIKI**](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) | 523k images / 20k celebrities | Chronological (scraped DOB) | Open research, noisy labels |
| 2017 | [**APPA-REAL**](http://chalearnlap.cvc.uab.es/dataset/26/description/) | 7591 | Apparent age (≥38 raters/image) + chronological | Open challenge |
| 2014 | [**Adience**](https://talhassner.github.io/home/projects/Adience/Adience-data.html) | 26k | Age group (8 bins) + gender | Open |
| 2007 | [**MORPH-II**](https://uncw.edu/oic/tech/morph.html) | 55k / 13k subjects | Chronological, longitudinal mugshots | Licence fee |
| 2017 | [**UTKFace**](https://susanqq.github.io/UTKFace/) | 23k | Chronological 0–116 yr | Open |
| 2022 | [**MiVOLO test sets**](https://github.com/WildChlamydia/MiVOLO) | aggregated | MAE ~4.3 yr ceiling on photos | Open checkpoints |
| 2025 | [**FaceAge (Bontempi)**](https://www.thelancet.com/journals/landig/article/PIIS2589-7500%2825%2900042-7) | 58k training | **Biological age, mortality-linked** (not chronological) | Open weights via paper |

**Caveat to flag in any cross-comparison:** FaceAge is a biological-age model and predicts cancer patients to look ~5 yr older than chronological. It is *not* a fair chronological benchmark — MiVOLO is the right open baseline for chronological MAE comparison (the README already addresses this).

---

## 3. Multi-view / 3D face capture datasets (for evaluating the nine-view render approach)

Critical test: do nine MRI-derived views provide signal comparable to nine *true* camera views? Compare your render pipeline's output to data acquired with synchronized multi-camera rigs.

| Year | Dataset | N subjects | Views | Resolution | Notes |
|---|---|---|---|---|---|
| 2023 | [**RenderMe-360**](https://renderme-360.github.io/) | 500 | 60 synchronized 2K cameras, 360° | 2K, 30 fps | **800k video sequences, 250M+ frames.** Annotations: camera params, FLAME fit, 2D/3D landmarks, scan, text |
| 2023 | [**NeRSemble**](https://github.com/tobias-kirschstein/nersemble) | small set, 10 sequences | 16 calibrated forward-facing | HD multi-view video + FLAME meshes | Extreme expressions + speech — useful for testing how mid-face deformation affects age cues |
| 2020 | [**FaceScape**](https://facescape.nju.edu.cn/) | 3592 | 68 viewpoints | 4344 × 2896, 0.9 TB | 20 expressions × 400k frames; topologically uniform mesh |
| 2016 | [**Headspace**](https://www-users.cs.york.ac.uk/nep/research/Headspace/) | 1519 | Full 3D head scan + 2D photo | — | One of the largest "true" 3D head cohorts |
| 2016 | [**3D Facial Norms** (FaceBase)](https://www.facebase.org/facial_norms/) | 2454 | 3dMD stereophotogrammetry + 24 landmarks | sub-mm error | Includes anthropometric measurements; FaceBase governance |
| 2006 | [**BU-3DFE**](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) | 100 (18–70 yr) | 3D shape + paired 2D texture (±45°) | — | 6 expressions × 4 intensities; small but has age range that overlaps with brain-age cohorts |
| 2008 | [Bosphorus](http://bosphorus.ee.boun.edu.tr/) | 105 | 3D face + expression labels | — | Compact open research set |
| 2003 | [FRGC v2.0](https://www.nist.gov/programs-projects/face-recognition-grand-challenge-frgc) | 466 | 3D + paired 2D | — | Restricted research access |
| 2025 | [**INCRAN**](https://papers.miccai.org/miccai-2025/0601-Paper2543.html) (infant cranial) | n/a | Photogrammetry + **paired CT** | — | First public 3D-face + CT head model (infants only). Closest existing analogue to what we'd want for adults |
| 2019 | [FFHQ-UV](https://github.com/csbhr/FFHQ-UV) | 70k (derived from FFHQ) | UV-texture atlases | — | Normalized facial UV for 3DMM-based reconstruction |

**Use for this project:** RenderMe-360 and FaceScape are the right benchmarks for "what does the upper-bound representation look like when you have real cameras." If the MRI nine-view pipeline cannot match performance against derivatives of these on age tasks, the limit is geometry/resolution not method.

---

## 4. Cosmetology & multi-spectral / multi-lamp facial imaging

Defines the upper bound on **what skin/surface signal you cannot recover from MRI**, no matter how good the renderer. This matters because the README explicitly flags skin tone, periocular detail, scleral color, and fat-redistribution texture as the most age-informative cues and the ones most vulnerable to generative hallucination.

| System / dataset | What it captures | Why it matters here |
|---|---|---|
| [**VISIA (Canfield)**](https://www.canfieldsci.com/imaging-systems/visia-complexion-analysis/) | Standard + cross-polarized + parallel-polarized + UV photography; RBX algorithm separates red/brown chromophores; quantifies spots, pores, wrinkles, evenness, porphyrins, UV-spots | Industry-standard cosmetology box. **Open data is essentially absent** (commercial system); but research using VISIA-derived measures is published in dermatology |
| [**VISIA-3D**](https://www.canfieldsci.com/imaging-systems/visia-complexion-analysis/) | Adds 3D facial geometry | Demonstrates feasibility of combining cross-polarized skin + 3D — analogue to the multi-view-MRI + skin-photo hybrid this project could test |
| [**OBSERV** (Sylton)](https://sylton.com/observ/) | UV, polarized, parallel-polarized, "True-Reflectance" | Competitor to VISIA |
| [**Reveal Imager** (Canfield)](https://www.canfieldsci.com/imaging-systems/reveal-imager/) | Cross-polarized + standard | Cheaper VISIA-lite |
| [Quantificare 3D LifeViz](https://www.quantificare.com/) | 3D facial stereophotogrammetry for before/after | Clinical aesthetic research workflow |
| [**PolFace** (CVPR 2023)](https://dazinovic.github.io/polface/) | Smartphone-captured cross + parallel polarized, dark room, point-light source. Separates diffuse and specular. Recovers 4K albedo + specular + normals | **Closest thing to open research data on multi-lamp face imaging.** Demonstrates that you don't need a commercial box for the principle |
| [Multispectral facial color imaging (PubMed 19123654)](https://pubmed.ncbi.nlm.nih.gov/19123654/) | Multimodal facial color analysis for skin lesions | Early benchmark; non-clinical use |
| [PhotoAgeClock (Bobrov 2018)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6286834/) | Eye-corner cropped photo → biological age | Demonstrates that a **tiny crop** of high-quality photo outperforms methylation clocks. Suggests the eye-corner detail MRI cannot render is doing a lot of the work in face-age models |

**Open-data gap, flagged honestly:** there is no large public corpus of *paired* VISIA/OBSERV + MRI on the same subjects. The closest analogue is using PolFace-style smartphone polarization on MRI subjects you control. For this paper that's out of scope, but worth a sentence in Discussion as the natural next acquisition.

---

## 5. Volumetric tissue segmentation references

For the segmentation pieces of the pipeline — facial soft tissue, fat compartments, mandible, skin.

### MRI segmentation

| Dataset | N | Labels | Notes |
|---|---|---|---|
| [SHARM](https://arxiv.org/pdf/2309.06677) | 196 | 15 tissues | Best face-tissue MRI ground truth |
| [GRACE](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00090/119208/) | 177 | 11 tissues | Older adults |
| [HaN-Seg](https://han-seg2023.grand-challenge.org/) | 56 | 30 OARs, paired CT+MR | Head & neck cancer; rare paired modality |
| [TotalSegmentator MRI v2](https://github.com/wasserth/TotalSegmentator) | ~600 | 50+ structures | Generalist whole-body MR segmenter |
| [MyoSegmenTUM](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198200) | 21 + 4 | Thigh muscles, water-fat MRI | If extending to muscle/fat compartments |
| [UK Biobank whole-body atlases](https://www.nature.com/articles/s43856-024-00670-0) | 6 atlases / 50k | Adipose, organs | Population-level normative reference |
| [VIBESegmentator (NAKO + UKB)](https://link.springer.com/article/10.1007/s00330-025-12035-9) | ~50k | Full-body multi-organ + muscle | Most generalizable open model |

### CT segmentation (for cross-validation of MRI face derivation against CT ground truth)

| Dataset | N | Labels | Notes |
|---|---|---|---|
| [TotalSegmentator v2](https://github.com/wasserth/TotalSegmentator) | 1228 | 117 anatomical structures | Workhorse general body |
| [SAROS](https://www.nature.com/articles/s41597-024-03337-6) | 900 | 13 regions + 6 body parts; adipose/muscle via HU | TCIA-derived |
| [FLARE 2023](https://arxiv.org/abs/2408.12534) | 4650 (40+ centers) | 13 organs + pan-cancer lesion | Largest multi-center abdominal CT |
| [HaN-Seg](https://han-seg2023.grand-challenge.org/) | 56 | 30 OARs | Repeated here because of paired CT+MR |
| [DentalSegmentator](https://www.sciencedirect.com/science/article/pii/S0300571224002999) | 470 CT/CBCT | Maxilla, mandible, teeth, mandibular canal | Lower-third face anatomy |
| [CBCT 4938 scans (Nat Commun 2022)](https://www.nature.com/articles/s41467-022-29637-2) | 4938 | Tooth + alveolar bone | Largest dental CBCT cohort |
| [PMCanalSeg](https://www.nature.com/articles/s41597-026-06620-w) | 191 CBCT orthognathic | Canals | Surgical use case |
| [AutoPET I/II/III](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=127664574) | 900–1500 | Whole-body PET + CT, tumor lesions | TCIA |
| [RSNA Cervical Spine Fracture](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection) | 3112 | C-spine vertebrae | Face often in FOV |

### Multimodal ground-truth references

| Resource | What | Why |
|---|---|---|
| [Visible Human Project — Male / Female](https://www.nlm.nih.gov/research/visible/visible_human.html) | CT + MRI + 1mm RGB cryosection (M) / 0.33mm (F) on the same cadaver | The only public resource that lets you verify MRI→face derivation against ground-truth photographic anatomy on the same body |
| [Chinese Visible Human](https://www.researchgate.net/publication/8627672) | CT + cryosection | Second-body validation |
| [Nelly phantom](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8664205/) | VHP-derived surface phantom | For RF/coil simulation; structural reference |
| [ITIS Virtual Population](https://itis.swiss/virtual-population/) | Multiple full-body voxel models | High-resolution computational phantoms |

---

## 6. Known gaps that matter for face-age work

These are the things that would close the loop for this research direction and that *no public dataset* currently provides:

1. **Paired non-defaced T1 MRI + true 3D face photogrammetry + matched age labels** on adults at any scale. INCRAN does it for infants. Nothing for adults publicly.
2. **Paired non-defaced MRI + VISIA/OBSERV/PolFace skin imaging** on the same subjects. Would directly answer "how much of the face-age signal is morphology (MRI-recoverable) vs surface (only photo-recoverable)."
3. **Longitudinal non-defaced MRI in healthy adults with multi-year follow-up + paired photos**. The closest pieces (UK Biobank repeat imaging visit + photographic face album) exist but UK Biobank brain MRI is defaced and there's no facial photo released.
4. **Cosmetology intervention longitudinal (botox / fillers / facelift) with MRI** before / after. Entirely in-house at clinics.
5. **Trans-affirming hormone therapy facial MRI longitudinal**. Both face and brain change measurably; no public cohort.

Items 2 and 3 are the ones that would most directly extend this project beyond MRI-only morphology.

---

## How this catalog ties back to the paper

For each section of the paper, the relevant rows here are:

| Paper section | Relevant rows |
|---|---|
| Training the face-from-MRI branch | §1 (IXI primary; SHARM / GRACE / OASIS-3 for tissue labels and high-age tail) |
| Baseline face-age literature comparison | §2 (MiVOLO on UTKFace / APPA-REAL; FaceAge for biological reference) |
| Multi-view ablation | §3 (RenderMe-360, FaceScape for "true-multi-view" upper bound) |
| Discussion of what MRI face cannot recover | §4 (PolFace, PhotoAgeClock as evidence skin/eye-corner cues matter) |
| Brain-age branch | §5 MRI segmentation rows + Belsky 2019 / Cole 2020 in `papers/related_works/literature_review.md` |
| Limitations / future work | §6 (gap list) |

---

## Contributing

PRs welcome to add datasets, fix dead links, or update access status. Sort additions chronologically (newest first) within their section and link the primary publication or landing page.
