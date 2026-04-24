# Clinical Facial Datasets for the FaceAge-to-BrainAge Project

A survey of public and requestable datasets pairing facial photographs or 3D face surfaces with clinical metadata, imaging, or grading. Compiled to evaluate alignment with an MRI-derived face-surface pipeline that aims to predict age and link surface morphology to internal anatomy.

Date of survey: April 2026. All URLs below were located via live web searches; where the dataset is demonstrably behind an application or DUA wall, that is stated explicitly. Where no reliable public source for a given specific sub-dataset could be located, the entry is marked "not identified publicly".

---

## Executive summary table

| Dataset | Domain | N (subjects) | Age range | Modalities | Age labels | License / access | Primary URL |
|---|---|---|---|---|---|---|---|
| FaceBase 3D Facial Norms (3DFN) | Craniofacial / norms | 2454 | 3-40 y | 3D stereophoto surface + landmarks + genotype | Yes (continuous) | Open summary; individual-level via DUA+IRB | facebase.org/resources/human/facial_norms |
| FaceBase Consortium (full) | Craniofacial / syndromes | 1000s across datasets (e.g. FB00000861, 892, 491) | Pediatric to adult | 3D surface, photos, DNA, CT in some subsets | Yes in most | Controlled-access, DUA+IRB | facebase.org |
| Headspace / LYHM (York) | Head shape / 3D | 1519 (1212 in LYHM build) | 1-89 y | 3D head scans (3dMD, latex cap) + texture | Yes (continuous) | Free, academic non-commercial EULA | www-users.york.ac.uk/~np7/research/Headspace |
| MEEI Facial Palsy Standard Set | Facial nerve palsy | 51 patients + 9 controls | Adult | Standardised photos + videos | No; House-Brackmann, Sunnybrook, eFACE grading | Open-source; annotated DB by IRB review | sircharlesbell.com |
| AAOF Craniofacial Growth Legacy Collection | Orthodontic / growth | 762 longitudinal subjects across 9 studies | 2-25+ y | Lateral ceph, hand-wrist X-rays, photos, casts, records | Yes (longitudinal) | Free registration, open viewer | aaoflegacycollection.org |
| FDTooth | Dental CBCT + intraoral photos | 241 patients | 9-55 y | CBCT (DICOM) + intraoral JPEG + bbox labels | Yes | PhysioNet credentialed (CITI) | physionet.org/content/fdtooth/1.0.0 |
| MMDental | Dental CBCT + EHR | 660 patients | 5-86 y | 3D CBCT (161,200 slices) + 2,125 medical records | Yes | Figshare (Springer Nature) | springernature.figshare.com (id 28505276) |
| Wang 400 / ISBI 2015 Ceph | Cephalometric | 400 | 6-60 y | Lateral cephalogram + 19 landmarks | Yes | Free (challenge release) | ntust.edu.tw/~cweiwang/ISBI2015 |
| UFBA / OdontoAI Panoramic | Dental panoramic | ~4000 images | Mixed | Panoramic X-ray + tooth labels | Partial | Open; test set private | github.com/IvisionLab/OdontoAI-Open-Panoramic-Radiographs |
| Teeth3DS+ | Intraoral 3D | 900 patients (1800 scans) | Orthodontic patient range | 3D OBJ upper+lower jaws | Partial | Open (OSF), MICCAI challenge | crns-smartvision.github.io/teeth3ds |
| SCIN (Google/Stanford) | Dermatology crowd | 5033 contributors / 10,408 imgs | Mostly adult; skewed <40 | Self-taken photos + Fitzpatrick + Monk | Self-reported age bins | Open (CC BY-NC) | github.com/google-research-datasets/scin |
| Fitzpatrick 17k | Dermatology | 16,577 clinical images | Not consistent | Clinical photos + 114 conditions + Fitzpatrick | No chronological age | CC-BY (atlas-sourced) | github.com/mattgroh/fitzpatrick17k |
| DDI (Diverse Dermatology Images) | Dermatology | 656 images | Adult | Clinical photos + skin type | No chronological age | Stanford DUA | ddi-dataset.github.io |
| Chinese cosmetic patient DB (Zhang 2023) | Cosmetology | 1821 patients / 10,529 images | Adult | Clinical-cosmetic facial photos | Yes | Private (institutional) | DOI:10.1111/srt.13402 |
| PhotoAgeClock | Periorbital aging | 8414 images | Adult | High-res eye-corner crops | Yes (chronological) | Not publicly released | aging-us.com/article/101629 |
| Blepharoptosis image sets | Periorbital | 10,555 patients / 250,534 images | Mean 57 y | Periocular photos | Yes | Institutional (Taiwan) | nature.com/articles/s41598-023-44686-3 |
| MORPH-II | General face aging | 13,000 individuals / 55,134 images | 16-77 y | Mugshot photos | Yes | Academic license (UNC) | faceaginggroup.com |
| IMDb-Wiki | Face aging | ~500K images | 0-100 y | Celebrity web photos | Yes (noisy) | Free for research | data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki |
| UTKFace | Face aging | 23,708 images | 0-116 y | Aligned cropped faces | Yes | Research-only | susanqq.github.io/UTKFace |
| AgeDB | Face aging | 16,516 images / 570 celebrities | 0-101 y | In-the-wild photos | Yes | Academic | ibug.doc.ic.ac.uk/resources/agedb |
| FG-NET | Face aging longitudinal | 82 subjects / 1002 images | 0-69 y | Photos spanning decades | Yes | Free for research | yanweifu.github.io/FG_NET_data |
| BU-3DFE / BU-4DFE | 3D face expression | 100 / 101 subjects | 18-70 y | 3D scans + expressions | Yes (bins) | Binghamton EULA | cs.binghamton.edu/~lijun/Research/3DFE |
| Bosphorus 3D | 3D face | 105 subjects / 4666 scans | Adult | 3D + textures + expressions + occlusions | No clinical | Academic registration | bosphorus.ee.boun.edu.tr |
| Visible Human Project | Anatomy (1M, 1F) | 2 cadavers | Adult | Cryosection, CT, MRI, photo | N/A | Public domain | nlm.nih.gov/research/visible |
| Brazilian panoramic age DS | Forensic dental age | 12,827 images | 2.25-96.5 y | Panoramic X-rays + chronological age | Yes | Research | nature.com/articles/s41598-024-70621-1 |
| Ceph-29 landmark DB | Cephalometric | 1000 ceph images / 7 devices | Orthodontic | Lateral ceph + 29 landmarks | Partial | Open (Nature SciData 2025) | nature.com/articles/s41597-025-05542-3 |
| FACE-R (Hungary) | Forensic skull+face | 400 living individuals | Adult | Full-head CT + 3D face scan, same person | Yes | Institutional | PubMed 24020394 |

---

## 1. Cosmetology / aesthetic medicine / dermatology

### 1.1 Clinical skin-aging grading scales and their validation image sets

The four standard photonumeric scales used in aesthetic medicine - **SCINEXA**, **Griffiths (Larnier) photonumeric**, **Glogau**, and the **Allergan / Merz photonumeric family** - were each developed and validated on defined image pools, but in essentially every case those pools are *not* released as datasets. What is published are the reference plate-images used to anchor the scale, not per-subject photographs with metadata.

- **Allergan Fine Lines Scale, Forehead Lines Scale, Facial Skin Texture Scale**: validated with panels of ~289-330 live subjects over two sessions (Dermatologic Surgery supplement, 2016). Reference photos live inside the published PDFs; the subject-level images are not publicly distributed.
- **Merz Aesthetic Scales** (crow's feet, perioral, lower-face laxity, etc.): 5-point photonumeric scales validated by expert panels; again only the anchor images are distributed. No subject-level dataset found.
- **SCINEXA / Vierkötter et al.**: 23-item clinical rating aggregating intrinsic and extrinsic aging; validated in the SALIA cohort (~400 Caucasian women). The SALIA cohort is a research cohort, not an openly released dataset.

**Practical conclusion.** The scale-validation studies are excellent *labelling schemes* that could be applied to any release of clinical photos; they are not themselves downloadable datasets.

### 1.2 Open dermatology datasets with some face content

- **SCIN (Skin Condition Image Network)** - Google Research + Stanford. 10,408 self-captured images from 5,033 US internet users (2023). Includes self-reported age band, sex, Fitzpatrick self-estimate, Monk Skin Tone, body location, dermatologist condition labels. Many are facial crops. Hosted on GCS bucket `dx-scin-public-data`. License: CC-BY-NC. Age is reported in bins, not continuous. Excellent for Fitzpatrick/skin-tone balancing; poor for age regression.
- **Fitzpatrick 17k** - 16,577 images from DermaAmin and Atlas Dermatologico with Fitzpatrick I-VI labels and 114 condition labels. Does **not** include chronological age. Useful as a Fitzpatrick prior only.
- **Diverse Dermatology Images (DDI)** - Stanford-curated 656 pathology-confirmed images skewed to dark skin. No age labels.
- **ISIC archive** - very large (>100K) but it is a lesion-centred, not face-centred, dataset; most images are cropped moles. Not directly useful for face-morphology aging.
- **SD-198** - 198-class dermatology benchmark; no skin-type or reliable age label; mostly body, not face.

### 1.3 Face-aging academic benchmarks with biological / cosmetic context

None of MORPH-II, IMDb-Wiki, UTKFace, AgeDB, or FG-NET are "clinical" - they are celebrity scrapes or mugshot collections with chronological-age labels. However they are the standard training base for face-age CNNs and already power the AIM-Harvard **FaceAge** model (used in the current project's parent pipeline). Key properties:
  - MORPH-II: 55K mugshots, 13K individuals, 16-77 y. The most clinically-uniform photography conditions.
  - FG-NET: 82 subjects, 1002 images; the only classic benchmark with genuine longitudinal per-subject spans (decades). Very small.
  - UTKFace, AgeDB, IMDb-Wiki: in-the-wild; noisy age labels on IMDb-Wiki but millions available.

### 1.4 Before/after cosmetic-procedure archives

No open database of before/after photographs for botox, filler, rhinoplasty, or facelift was identified in any repository with research-grade access. These exist as proprietary collections inside clinics and industry (Allergan/AbbVie internal archives, Galderma clinical-trial arms, Merz trial arms). None are public.

### 1.5 Ethnic-specific cosmetology databases

- **Zhang et al. 2023** (*Skin Research and Technology*, DOI 10.1111/srt.13402). Large-scale Chinese cosmetic-patient database: 10,529 images from 1,821 patients, used to train two facial-age predictors. The paper reports that existing public face-age models are Caucasian-biased. **The dataset itself is not publicly released.**
- **Singapore/Malaysia Cross-sectional Genetics Epidemiology Study** - 1,081 ethnic Chinese young adults with photonumeric wrinkle grading. Cohort data, not an open dataset.
- **Chinese Female Facial Aging Stages cohort** (Frontiers in Medicine 2022) - grading study, not release.
- **Japanese VISIA color + UV image pairs** (184 individuals) used to train UV-photo Net (Sci Reports 2020). Not publicly released.

### 1.6 Commercial skin-analysis systems

- **VISIA / VISIA-CR / VISIA 3D (Canfield Scientific)**. Proprietary. Uses internal multi-ethnic aging database for TruSkin Age; researchers can license the device but the reference database is not distributed.
- **OBSERV 520** (Sylton). Diagnostic imaging device; no public dataset.

### 1.7 Periorbital / eye region aging

- **PhotoAgeClock** (Bobrov et al., *Aging* 2018). 8,414 high-res eye-corner crops with chronological-age labels; MAE 2.3 y. Dataset itself is not publicly released; model available.
- **Blepharoptosis CNN (Hung 2023, *Sci Reports*)**. 250,534 periocular/facial images from 10,555 patients over 25 years at a tertiary oculoplastic clinic. Mean age 57 y. Institutional-access only; not a public release.
- **Taiwan ptosis set** (500 images). Single-institution.

### 1.8 Lip / perioral aging

No dedicated public dataset found. Perioral wrinkle grading is part of the Merz Perioral Wrinkle Scale but, as above, not released as a subject-level resource.

---

## 2. Dental / orthodontic / oral and maxillofacial

### 2.1 CBCT datasets

- **MMDental** (Zhang et al. 2025, *Sci Data*). 660 patients, **5-86 y**, 161,200 CBCT slices + 2,125 expert medical records. Hosted on Springer Nature figshare (id 28505276). No paired face photograph.
- **FDTooth** (Nature *Sci Data* 2025). 241 patients, **9-55 y**. **Paired** intraoral photographs (high-res JPEG 5760x3840) and CBCT (DICOM), 1800 bounding-box annotations for fenestration/dehiscence. PhysioNet credentialed. *This is the rare modality-paired dental resource, but the "photograph" is intraoral, not a frontal face portrait.*
- **CTooth+** and **ToothFairy** (MICCAI 2023-2024). Tooth-segmentation CBCT releases; no face photos.
- **CBCTLiTS** - hepatic, not dental; listed only to disambiguate.

### 2.2 Panoramic and cephalometric radiographs

- **UFBA-UESC / OdontoAI** (IvisionLab). ~4,000 panoramic X-rays, partial release of 2,000 (650 scratch + 1350 HITL). Tooth-level labels. No paired photograph. No age in the public portion.
- **Wang 400 / ISBI 2015 Cephalometric Challenge**. 400 lateral cephalograms, **6-60 y**, 19 manual landmarks, 150/150/100 split. Host: ntust.edu.tw/~cweiwang/ISBI2015.
- **CL-Detection 2023** Grand Challenge. Multi-device cephalometric landmark detection.
- **Benchmark dataset for Ceph Landmark + CVM Stage Classification** (Nature *Sci Data* 2025). 1000 lateral cephalograms, 7 imaging devices, 29 landmarks.
- **Brazilian panoramic age set** (Sci Reports 2024). 12,827 panoramic X-rays, **2.25-96.5 y**, explicit chronological-age labels. Excellent for dental-age estimation.
- **Kaggle Panoramic Dental X-ray TFRecords** - community release of a legacy UFBA subset.

### 2.3 Orthodontic pre/post and intraoral 3D

- **Teeth3DS+ / 3DTeethSeg / 3DTeethLand** (MICCAI 2022 and 2024). 900 patients, 1,800 intraoral 3D scans (upper+lower), OSF-hosted, OBJ format. Ages skew orthodontic (juvenile-young adult).
- **Pre/post orthodontic 3D dental model dataset** (Beijing Stomatological Hospital, Nature *Sci Data* 2024). 435 patients, 1,060 paired pre/post 3D models. No paired face photo.

### 2.4 Dental-panoramic + face photo paired data

**None identified publicly.** The FDTooth pair is panoramic-substitute CBCT vs intraoral photo. A face-portrait + panoramic pair dataset is a clear gap.

### 2.5 Age estimation from teeth (Demirjian-style)

Multiple publications reference internal cohorts (e.g., 4,800 panoramics, mean 10.64 y, for YOLOv11-Demirjian; 12,827 Brazilian panoramics). The **Brazilian panoramic age dataset** appears genuinely released; most others are not.

### 2.6 Smile aesthetics + face

- **MetiSmile (Shining 3D)**: device for fused facial + intraoral + CBCT capture. Commercial, no public dataset.
- **Digital-Smile-Design smartphone workflows**: case reports, no dataset.

### 2.7 TMJ / jaw MRI

**No open TMJ-MRI release** was located. Single-institution cohorts exist in segmentation papers (e.g., 3,693 MR images from 542 patients - not released). TMJ imaging appears to be a gap.

### 2.8 Cleft lip/palate

- **FaceBase FB00000892** - "Genetic Determinants of Orofacial Shape and Relationship to Cleft Lip/Palate", 3D white-light photogrammetry of North American children (Denver, San Francisco). Controlled access via FaceBase DUA.
- **FB00000861, FB00000491** - additional FaceBase syndromic cohorts used in the 3D syndromic facial atlas.
- No purely open release of cleft pre/post 3D was identified.

---

## 3. Facial / plastic / craniofacial surgery

### 3.1 Orthognathic surgery (pre/post 3D + CBCT)

Despite many published papers combining CBCT + 3dMD stereophotogrammetry before and after surgery (typically 1 week pre and 6 months post), **no open-access release of a pre/post orthognathic dataset was identified.** All such cohorts are single-institution (Nijmegen, Leuven, Utrecht, Shanghai Ninth People's, Seoul). Soft-tissue prediction papers (Computers in Biology and Medicine 2025) train on private data.

This is arguably the single biggest gap vs the parent project's long-term goal: linking face surface to underlying bone.

### 3.2 FaceBase Consortium and 3D Facial Norms

- **FaceBase 3D Facial Norms (3DFN)**: 2,454 participants, **3-40 y**, 3dMD two-pod stereophotogrammetry + 3D landmarks + anthropometric measures + genotypes. Summary-level data open; individual-level behind DUA + IRB. Age is continuous. *This is the dataset most aligned with "predict age from face surface" in an explicitly clinical/craniofacial frame.*
- **FaceBase syndrome atlases** - controlled-access releases covering Crouzon, Sotos, etc., with expected-phenotype visualisations per age/sex.

### 3.3 Headspace + Liverpool-York Head Model (LYHM)

- **Headspace**: 1,519 3D head scans (latex cap for hair control), 3dMDhead 5-view system, **ages 1-89** (non-uniform, twenties-heavy), ~180K vertices per mesh, OBJ + BMP texture. Collected at Alder Hey Children's Hospital, UK. Free for University non-commercial research (academic-verification EULA).
- **LYHM**: the 3DMM built from 1,212 balanced-sex Headspace subjects. The only publicly available full-head morphable model that spans the lifespan.

Strongly recommended as complementary to MRI-derived surfaces, since both are skin/head-surface meshes with known age.

### 3.4 Facial nerve palsy / paralysis

- **MEEI Facial Palsy Photo and Video Standard Set** (Greene 2020, *Laryngoscope*). 51 patients with paralysis + 9 controls. Standardised protocol of eight facial movements. Graded with House-Brackmann, Sunnybrook, and eFACE. The standard-set photos and videos are open on sircharlesbell.com; the **annotated** database is controlled by MEEI IRB on a case-by-case basis.
- **IEEE DataPort Facial Paralysis Dataset** - separate, smaller, less standardised release.

### 3.5 Cleft pre/post 3D photogrammetry

FaceBase FB00000892 as above; various single-centre papers (Utrecht, Trieste, Nijmegen) but no purely open release.

### 3.6 Burn / trauma + 3D mesh

**No open burn before/after 3D-mesh dataset was identified.** Case reports and single-patient reconstructions dominate.

### 3.7 Facial trauma CT + photo pairs

**None identified publicly.**

### 3.8 Skull-face forensic reconstruction

- **FACE-R** (Hungary, Palkovics 2013). 400 living individuals, paired full-head CT + 3D face scan. Upright imaging is unusual and makes this one of the best face-to-skull registered cohorts. Institutional-access.
- **Skull-to-Face (arXiv 2403.16207)** cites several proprietary skull-face corpora.
- Brazilian soft-tissue thickness atlases exist as *measurement tables*, not image releases.

### 3.9 Pediatric craniofacial longitudinal

- **AAOF Craniofacial Growth Legacy Collection** (Bolton-Brush, Burlington, Denver, Fels, Forsyth Twins, Iowa, Mathews, Michigan, Oregon). 762 longitudinal subjects, annual+ records from ~age 2 through mid-20s. Lateral cephs, hand-wrist films, plaster casts, **photos**, anthropometrics, medical history. Free with registration. The closest thing to a public longitudinal paired cephalogram+photo+growth archive.
- **NIH Pediatric MRI Data Repository** (~500 children, newborn to young adult). Useful for head MRI baseline, not paired face photos.

### 3.10 Syndrome-specific (Treacher Collins, Apert, Crouzon, hemifacial microsomia)

Covered inside FaceBase releases FB00000491, 861, 892 and used to build the 3D syndromic facial atlas (*J Med Genet* 2024). Controlled access.

### 3.11 Cadaveric / Visible Human

- **NLM Visible Human Project**. Two full-body cadavers (Male 1994, Female 1995) with cryosection, CT, MRI including head. Public domain. Only 2 subjects - essentially a reference, not a training set.

---

## 4. Cross-referenced specific searches (results)

Each of the ten specific probes in the task yielded the items tabulated above. Notable findings:
- "CBCT dataset public download" -> MMDental, FDTooth, CTooth+, ToothFairy.
- "orthognathic pre-post 3D stereophotography dataset" -> no public release identified.
- "facial aging dataset dermatology" -> SCIN, Fitzpatrick 17k, DDI; no face-specific clinical aging set public.
- "craniofacial FaceBase dataset" -> 3DFN + syndrome releases; individual-level access controlled.
- "cleft lip 3D photogrammetry dataset" -> FaceBase FB00000892 only.
- "facial nerve paralysis video dataset" -> MEEI standard set (standard-set photos/videos open; annotations controlled).
- "wrinkle severity grading dataset" -> scales exist; no open subject-level set.
- "Allergan facial assessment photonumeric scale dataset" -> validation paper panels only; no subject release.
- "panoramic dental radiograph public dataset" -> OdontoAI/UFBA, Brazilian 12,827-image set, Ceph-29 set.
- "forensic facial reconstruction open dataset" -> FACE-R (institutional).

---

## 5. Prioritised recommendations for the FaceAge-to-BrainAge project

Ranking assumes the project:
- renders face surfaces from T1 MRI (hairless, closed eyes),
- wants to regress age / aging biomarkers,
- wants eventually to relate face morphology to internal anatomy for surgical applications,
- needs chronological-age labels and, ideally, paired imaging of internal structures.

### Rank 1: Headspace / LYHM (York)

**Rationale.** Closest modality match to MRI-derived face surfaces. Subjects wore latex caps to suppress hair, which directly mirrors the hairless render a T1 MRI produces; closed-eye stereophotogrammetry is common in the dataset. Age range 1-89 covers the clinical lifespan; 1,212 sex-balanced subjects support a 3DMM. Free for academic research. Provides a shape-only reference distribution against which MRI-surface age predictors can be benchmarked.

### Rank 2: FaceBase 3D Facial Norms (3DFN)

**Rationale.** 2,454 subjects, **3-40 y**, with 3D stereophotogrammetric surfaces, landmarks, anthropometric measurements, and genotypes. The only public norm dataset that enforces a standardised frontal clinical capture with age labels at scale. Directly supports age regression and per-landmark aging-rate analysis. Individual-level access requires DUA+IRB but is routinely granted to academic teams. Strong bridge between "face surface" and "craniofacial clinical norms", aligned with surgical-planning use cases.

### Rank 3: AAOF Craniofacial Growth Legacy Collection

**Rationale.** Only public dataset that pairs photographs with cephalometric X-rays longitudinally, from childhood into adulthood. Allows face-surface <-> skull-surface correspondence learning (via ceph) and gives the only genuinely open pediatric-to-adult longitudinal reference. Free registration. Limitation: 2D photo, not 3D. Still essential for modelling the age-trajectory of facial landmarks under known growth conditions.

### Rank 4: FDTooth (PhysioNet)

**Rationale.** The rare dataset with CBCT + paired photograph (albeit intraoral) in the same subject, with continuous age 9-55. If the broader project ever wants to learn skin-to-bone correspondence via CBCT, FDTooth provides a compact, well-documented, credentialed-access template. Credentialed-access overhead is modest (CITI training + PhysioNet application).

### Rank 5 (tie): MMDental and Wang 400 / ISBI 2015 Cephalometric

**Rationale.** MMDental (660 patients, 5-86 y) adds a wide-age CBCT reference to the project with structured medical records, supporting any future age-biomarker work from internal structures. Wang 400 gives a clean, well-benchmarked 2D cephalometric landmark set for pipeline validation. Together they cover skull/tooth internals over the same age span the MRI face-surface project targets.

### Honourable mentions

- **MEEI Facial Palsy** - if the project ever pivots to dynamic facial features (symmetry, movement as aging correlates), this is the standard.
- **Brazilian panoramic age set** (12,827 images, 2.25-96.5 y) - excellent *independent* chronological-age ground-truth via teeth, useful as a cross-modality age sanity check.
- **MORPH-II** for pure age-estimation pre-training given its photographic uniformity.
- **FACE-R (Hungary)** - if institutional access can be negotiated, it is uniquely aligned with the surgical-planning long-term aim because it pairs CT + face scan on the **same living individuals in upright pose**.

---

## 6. Access and compliance notes

The project is a healthcare-adjacent pipeline that will ingest identifiable facial morphology, so dataset selection should observe:

- **IRB / DUA before download.** FaceBase (individual-level), MEEI annotated database, and most hospital releases require a data use agreement with named PI and IRB approval letter.
- **PhysioNet credentialing.** FDTooth requires CITI "Data or Specimens Only" training and a PhysioNet credentialed account. Budget 1-2 weeks.
- **Academic-only EULAs.** Headspace/LYHM, BU-3DFE, Bosphorus are verifiable-employee licenses - covered by a university email and faculty signature.
- **GDPR / HIPAA impact.** Every dataset containing an identifiable facial photograph is biometric / special-category data under GDPR Art. 9. Keep raw photos off any publicly addressable storage; store derived embeddings and mesh-geometry only where possible. Per the project's global compliance rule-set, logs must contain subject IDs only, never image derivatives.
- **Breach notification windows.** HIPAA 60 days, GDPR 72 h - documented on go-live.

---

## 7. Gaps that would be worth closing

1. **Face-portrait + panoramic/cephalogram pair public dataset** - does not appear to exist.
2. **Orthognathic pre/post 3D stereophotogrammetry open release** - very active research area, but all corpora are private.
3. **Cosmetic before/after (injectable, filler, rhinoplasty) public dataset** - a clear gap, likely because the photos are commercial assets.
4. **Ethnic-balanced clinical face-aging release** - Zhang 2023's Chinese dataset remains private; there is no counterpart for Indian or African cohorts.
5. **TMJ-MRI public release** - none found.
6. **Paired face-MRI + face-photo of the *same* individual** - not identified as a public resource. The parent project's own data may in fact be the only accessible example.
