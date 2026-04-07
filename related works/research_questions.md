# Research Questions and Future Work

Current as of April 7, 2026.

This memo answers the main forward-looking questions around chronological-age accuracy, gap interpretation, 2D-to-3D face reconstruction, and the practical value of non-defaced MRI datasets for this repository.

## Executive Takeaways

- Brain-age MRI models are usually the stronger chronological-age estimators, because they are trained directly on chronological age from standardized MRI volumes.
- `FaceAge` should not be treated as a fair chronological-age baseline. It is explicitly framed as a biological-age and health-risk model, so larger deviations from chronological age are part of the point rather than a failure mode.
- Large positive prediction gaps are most consistently reported in disease-enriched cohorts. Close agreement with chronological age is most likely in healthy controls and in-distribution cohorts after bias correction.
- A `2D -> 3D avatar -> age` pipeline is scientifically interesting for morphology, pose normalization, and multi-view rendering, but there is no strong evidence yet that it outperforms strong direct 2D age estimation.

## 1. Which is more accurate for chronological age: 2D face age or 3D MRI brain age?

The short answer is: for chronological age, brain-age MRI models usually win. The comparison is not apples-to-apples, though, because the strongest face-side model currently wired into this repo, `FaceAge`, is a biological-age model rather than a chronological-age benchmark.

| Family | Representative work | Target | Input | Reported accuracy / signal | Interpretation for this repo |
| --- | --- | --- | --- | --- | --- |
| Brain age | SFCN | Chronological age | 3D T1w MRI | MAE `2.14 years` on UK Biobank T1 MRI | Strong open chronological-age baseline for the brain branch |
| Brain age | Kim et al. 2025 | Chronological age plus clinically meaningful brain-age gap | Axial 2D T1w MRI | MAE `2.73 years` after bias correction on clinical cognitively unimpaired scans; ensemble MAE `2.23 years` | Excellent clinical chron-age result, but not openly runnable |
| Face age | MiVOLO | Apparent or chronological age | RGB face image | Official repo reports face-only MAE around `4.22-4.29 years` on IMDB-cleaned and UTKFace checkpoints | Best open comparator if the goal is pure face-based chronological age from 2D renders |
| Face age | FaceAge | Biological age and health-associated face age | RGB face image | Trained on `58,851` healthy individuals; cancer patients averaged `4.79 years` older than chronological age | Best scientific fit for this repo's current face branch, but not a fair chron-age leaderboard entry |

### Chronological age accuracy

If the task is strictly "predict the calendar age as closely as possible," brain-age MRI models currently have the cleaner claim. Open T1-based models such as `SFCN` are trained directly for that target and report lower MAE than strong open face-photo baselines. Clinical MRI work such as `Kim et al. 2025` suggests that well-designed MRI models can remain competitive even outside research-grade 3D acquisitions.

### Biological age and health sensitivity

If the task is "capture clinically meaningful deviation from chronological age," then `FaceAge` becomes much more interesting. Its central result is not low MAE around chronological age, but that looking older than one's chronological age carries prognostic information. For this repository, that means face-side and brain-side models should not be judged on one axis only.

### What this means for the repo

For a fair experiment here, the best setup is not a single mixed leaderboard. It is two comparisons run side by side on the same non-defaced cohort:

- `Chronological-age track`: `SFCN` versus a face model such as `MiVOLO`
- `Biological-gap track`: `FaceAge` versus brain-age-gap models analyzed for disease sensitivity and risk stratification

## 2. Who shows the largest gap between chronological and predicted age, and who matches closely?

Large positive gaps usually appear where pathology, frailty, or systemic stress pushes the model away from "healthy normative aging." Small gaps usually appear in healthy controls and in-distribution cohorts after bias correction.

| Pattern | Example evidence | Interpretation |
| --- | --- | --- |
| Largest face-age gaps | `FaceAge` reported that patients with cancer looked `4.79 years` older on average than their chronological age | Face-side positive gap seems especially informative when systemic illness is present |
| Large brain-age gaps in inflammatory disease | In a multicenter RRMS study, mean brain-age gap reached `13.0 +/- 14.7 years`; another MS study reported `4.4 +/- 6.6 years` | Multiple sclerosis is one of the clearest settings where MRI-derived brain age shifts older |
| Large brain-age gaps in severe psychiatric disease | Schizophrenia showed predicted brain age `6-8 years` older than chronological age across two samples | Severe psychiatric cohorts can show strong structural aging-like signal |
| Moderate positive brain-age gaps in neurodegeneration | `Kim et al. 2025` reported mean corrected brain-age gap `3.10 years` in Alzheimer's disease | Neurodegenerative disease shifts the brain branch older, though not always as dramatically as inflammatory cohorts |
| Closest agreement | In the same `Kim et al. 2025` study, cognitively unimpaired subjects had mean corrected brain-age gap `0.09 years` | Best agreement is expected in healthy, bias-corrected, in-distribution subjects |

The open question that still matters most for this repository is not whether either branch can deviate from chronological age. That is already well established. The unresolved question is whether `face_age_gap` and `brain_age_gap` co-vary inside the same person when both are derived from the same MRI session.

The literature remains mixed on that point:

- `Belsky/Elliott et al.` support the idea that an older-looking organism also shows older-brain characteristics at midlife.
- `Cole et al. 2020` found that multimodal brain-PAD was not significantly associated with subjective facial aging in UK Biobank.

That tension is exactly why this repository matters. It replaces separate photographs and subjective ratings with two machine-derived signals from one acquisition.

## 3. Would age estimation from a 3D face mesh/avatar be better than direct 2D image age estimation?

This question splits into three different claims.

| Claim | Current status | Practical meaning |
| --- | --- | --- |
| Direct 2D age estimation is mature | Strongly supported | The strongest currently testable path remains direct age prediction from rendered 2D face images |
| `2D -> 3D` face reconstruction exists | Strongly supported | There are multiple open systems that reconstruct meshes or parametric heads from one image |
| `2D -> 3D avatar -> age` is better than direct 2D age | Not established | This remains a hypothesis, not a demonstrated advantage |

The closest open ecosystem to the proposed avatar branch is `FaceScape`: it combines a large 3D face dataset, single-view 3D reconstruction, and subject metadata including age. That is useful for future method design. It still does not prove that age predicted from a reconstructed avatar is better than age predicted directly from the input face image.

The likely advantages of a 3D avatar branch are conceptual rather than proven:

- better control of pose and view
- cleaner isolation of geometry from texture
- ability to render multiple canonical views from one reconstructed shape

The likely risks are just as important:

- single-image reconstruction may smooth away age-relevant high-frequency detail
- reconstruction error may add noise before the age model even starts
- many 3D reconstruction benchmarks optimize geometry, not preservation of age signal

`3D -> 2D` is not the hard problem here. Once a mesh exists, rendering novel views is standard computer graphics. The hard question is whether the reconstructed mesh preserves enough age-relevant information to outperform the original 2D render.

## 4. What exists for 2D -> 3D face avatar reconstruction, and how good is it?

### Models and benchmarks

| Resource | What it does | What it outputs | What its benchmark actually measures | Relevance here |
| --- | --- | --- | --- | --- |
| `DECA` | Monocular detailed 3D face reconstruction | Parametric head with detailed facial geometry | Geometric fidelity on datasets such as NoW, plus animation-ready detail | Good candidate if the goal is age from detailed reconstructed morphology |
| `EMOCA` | Emotion-aware monocular 3D face reconstruction | 3D face with stronger expression capture than DECA | Better reconstruction of emotional and expressive faces, not age prediction | Useful if expression robustness matters in photo inputs |
| `3DDFA_V2` | Fast dense 3D face alignment and mesh export | 3DMM parameters, mesh, pose, depth, OBJ/PLY outputs | Speed, alignment stability, and dense face alignment quality | Attractive as a lightweight engineering baseline |
| `NoW` | Standard evaluation benchmark for single-image 3D face reconstruction | Evaluation protocol rather than a model | Scan-to-mesh distance after rigid alignment | Important because it measures geometry, not age preservation |
| `FaceScape` | Large 3D face dataset plus single-view reconstruction benchmark | Detailed 3D scans, multi-view images, age and gender metadata, benchmark tools | Reconstruction quality from single-view input | Closest open ecosystem to an avatar-first future direction |

### What the reported results actually mean

`NoW` is the key cautionary benchmark. It evaluates how closely a predicted mesh matches a 3D scan after rigid alignment. That is a geometry benchmark, not an age-estimation benchmark. A model can score well on `NoW` and still fail to preserve the age signal needed by a downstream age regressor.

`DECA` and `EMOCA` are therefore promising as reconstruction backbones, not as evidence that 3D age prediction is solved. `3DDFA_V2` is even further toward engineering utility: it gives fast and stable alignment, but the output is a simplified morphable-model representation rather than a validated age-sensitive facial biomarker.

`FaceScape` matters because it releases both detailed 3D data and single-view reconstruction infrastructure. It is the closest open benchmark family to the question "if I only have one face image, how faithful can my reconstructed 3D face become?" It is still not a direct answer to "will that improve chronological-age prediction?"

### What we know from true 3D face age work

Age prediction from genuine 3D facial data already exists.

- In adults, `Chen et al. 2015` used more than `300` three-dimensional facial images across ages `17-77` years, built a robust facial-age predictor, and showed that people of the same chronological age could differ by about `+/-6 years` in facial age, with those deviations supported by health indicators.
- In children and adolescents, `Matthews et al. 2018` reported `MAE = 1.19 years` for age estimation from 3D facial prototypes and showed that synthetically grown faces matched later scans within `3 mm` over most of the face.

That distinction matters:

- `Age from true 3D facial scans` exists and can be strong
- `Age from a reconstructed 3D avatar made from one 2D image` is still much less established

For this repository, MRI-derived face surfaces are actually closer in spirit to the first category than the second. The MRI volume already contains 3D morphology. That makes a direct MRI-surface or MRI-mesh age model scientifically more attractive than a generic selfie-to-avatar pipeline.

## 5. What is the opportunity in non-defaced MRI datasets?

The main bottleneck is not only whether facial anatomy is present, but whether the data-use terms allow face reconstruction, avatar generation, or any derivative representation that could identify participants.

| Dataset | Face information status | Age / cohort | Modalities | Potential for this project | Policy / caveat |
| --- | --- | --- | --- | --- | --- |
| IXI | Publicly available raw head MRI; recent defacing work explicitly analyzed `581 nondefaced T1w images` from IXI | Healthy adults, roughly `20-86` years | T1, T2, PD, plus MRA and DTI | High-potential and near-term | Best open testbed for paired face/brain work; already aligned with this repo |
| OASIS-3 | Longitudinal aging and Alzheimer's MRI archive with raw head images, but current access terms explicitly prohibit facial recognition and derivative facial reconstruction | Older adults, normal aging and AD spectrum | Multimodal MRI plus PET and clinical data | Promising but policy-sensitive | Strong disease-age cohort, but face/avatar work is restricted by the data-use agreement |
| ADNI | Mixed picture across releases; `ADNI4` now de-faces all applicable brain images before public release | Older adults, MCI and AD spectrum | Structural MRI, FLAIR, other MRI, PET, clinical data | Mixed; useful only after release-level verification | Distinguish legacy ADNI from `ADNI4`; do not assume current public facial availability |
| ABIDE / ABIDE II | Official ABIDE II paper states that contributors removed PHI including face information and coordinating centers ensured complete facial removal; ears were also removed | Mostly pediatric and young-adult autism/control cohorts | Structural MRI, resting-state fMRI, some diffusion MRI | Low for face-based work | Still useful for brain-only age work, but weak candidate for avatar or face-age experiments |

The practical ranking is therefore:

1. `IXI` for immediate paired face/brain experiments
2. `OASIS` only if the project stays strictly within data-use terms or gets explicit permission for any face-side derivative work
3. legacy `ADNI` only after careful release-level verification
4. `ABIDE / ABIDE II` mainly for brain-side benchmarking, not face-side modeling

## Future Work

| Experiment | Why it matters | Minimum data needed | What would count as success |
| --- | --- | --- | --- |
| Direct chron-age benchmark | Separates pure chronological-age accuracy from biological-age sensitivity | One non-defaced cohort with T1 MRI, chronological age, and train/test split shared across face and brain branches | Brain and face models are compared on identical subjects with transparent MAE and calibration |
| Gap phenotype analysis | Tests who really drives large positive `face_age_gap` and `brain_age_gap` | Chronological age plus metadata such as diagnosis, BMI, smoking, frailty, treatment status, and scanner/site | Clear enrichment patterns emerge rather than gap noise driven only by scanner artifacts |
| 3D avatar ablation | Directly tests whether a mesh/avatar intermediate helps or hurts age estimation | MRI-derived face surface, direct 2D render, and one open 3D reconstruction or mesh-based age pipeline | Avatar-based branch beats or complements direct 2D prediction on held-out subjects |
| Rendering loop | Uses the 3D surface to generate multiple canonical views instead of a single frontal PNG | One reconstructed MRI face mesh plus a renderer and one face-age model | Multi-view aggregation improves robustness or calibration over a single frontal render |
| Dataset expansion | Extends beyond IXI if policy and disease labels justify it | Verified-access OASIS or legacy ADNI releases plus explicit documentation of allowed use | New cohort adds either disease contrast or longitudinal signal without violating access terms |
| Joint modeling | Tests whether face and brain provide complementary aging information | Subjects with paired `chron_age`, `face_age`, `brain_age`, and downstream outcomes | Combined model outperforms either branch alone for prognosis, frailty, or disease discrimination |

## Sources

### Age estimation and gap interpretation

- FaceAge paper: <https://pubmed.ncbi.nlm.nih.gov/40345937/>
- FaceAge official project page: <https://aim.mgh.harvard.edu/faceage>
- FaceAge official repo: <https://github.com/AIM-Harvard/FaceAge>
- MiVOLO official repo: <https://github.com/WildChlamydia/MiVOLO>
- SFCN paper: <https://pubmed.ncbi.nlm.nih.gov/33197716/>
- SFCN official repo: <https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain>
- Kim et al. 2025, routine clinical MRI scans: <https://www.nature.com/articles/s41514-025-00260-x>
- Belsky/Elliott et al., brain-age in midlife: <https://www.nature.com/articles/s41380-019-0626-7>
- Cole et al. 2020, multimodality brain age in UK Biobank: <https://pubmed.ncbi.nlm.nih.gov/32380363/>
- Wei et al. 2023, brain age gap in NMOSD and multiple sclerosis: <https://pubmed.ncbi.nlm.nih.gov/36216455/>
- Hogestol et al. 2019, multiple sclerosis brain age gap: <https://pubmed.ncbi.nlm.nih.gov/31114541/>
- Shahab et al. 2019, schizophrenia brain age: <https://pubmed.ncbi.nlm.nih.gov/30635616/>

### 2D-to-3D face reconstruction and true 3D face age

- DECA official project page: <https://is.mpg.de/ncs/code/deca-learning-an-animatable-detailed-3d-face-model-from-in-the-wild-images>
- DECA official repo: <https://github.com/YadiraF/DECA>
- EMOCA official repo: <https://github.com/radekd91/emoca>
- EMOCA paper: <https://openaccess.thecvf.com/content/CVPR2022/html/Danecek_EMOCA_Emotion_Driven_Monocular_Face_Capture_and_Animation_CVPR_2022_paper.html>
- 3DDFA_V2 official repo: <https://github.com/cleardusk/3DDFA_V2>
- NoW benchmark: <https://now.is.tue.mpg.de/>
- FaceScape official project page: <https://nju-3dv.github.io/projects/FaceScape/>
- FaceScape paper: <https://doi.org/10.1109/CVPR42600.2020.00068>
- 3D facial morphologies as aging markers: <https://pubmed.ncbi.nlm.nih.gov/25828530/>
- Matthews et al. 2018, age estimation and growth from 3D facial prototypes: <https://pubmed.ncbi.nlm.nih.gov/29567544/>

### Dataset access and face-policy constraints

- IXI official dataset page: <https://brain-development.org/ixi-dataset/>
- IXI nondefaced analysis in defacing literature: <https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003149>
- OASIS request-access and data use terms: <https://sites.wustl.edu/oasisbrains/request-access/>
- OASIS project page: <https://dev.oasis-brains.org/>
- ADNI4 face de-identification paper summary: <https://mayoclinic.elsevierpure.com/en/publications/implementation-and-validation-of-face-de-identification-de-facing>
- ADNI and OASIS defacing study context: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7759440/>
- ABIDE II data release paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5349246/>
