# Journal Targeting Memo: MRI-Derived Face Age vs Brain Age

Date checked: 2026-06-07

## Current State

The current manuscript is a methodological and reproducibility paper, not yet
a clinical biomarker paper.

Core evidence in the repo:

- SIMON: 99 T1 scans of one subject across 36 scanners, age 29.6-46.4.
- Brain branch: SynthBA is scanner-stable but heavily biased on SIMON; weak
  longitudinal slope remains detectable.
- Face branch: FaceAge-on-renders and BioFace3D morphometrics do not track
  within-subject aging in SIMON.
- IXI: raw face-gap/brain-gap correlation exists, but collapses after
  controlling for chronological age. This is a useful null result, not a
  clinical biomarker result.

Working claim:

> A same T1 scan can be used to extract a face-derived and a brain-derived age
> estimate, but off-the-shelf models transfer poorly; apparent cross-sectional
> gap correlation is largely an age-bias confound, while SIMON exposes scanner
> stability versus biological validity.

## Source Basis

Local source files:

- `README.md`
- `papers/manuscript/manuscript.tex`
- `papers/manuscript/references.bib`
- `papers/related_works/literature_review.md`
- `papers/related_works/sota_design.md`
- `papers/related_works/data_methods_map.md`

External source checks:

- NeuroImage scope: https://www.sciencedirect.com/journal/neuroimage
- NeuroImage: Reports scope: https://www.sciencedirect.com/journal/neuroimage-reports
- Medical Image Analysis scope: https://www.sciencedirect.com/journal/medical-image-analysis
- IEEE TMI scope: https://ieeetmi.org/scope/
- Communications Medicine scope: https://www.nature.com/commsmed/aims
- npj Digital Medicine scope: https://www.nature.com/npjdigitalmed/aims
- Frontiers in Neuroimaging scope: https://www.frontiersin.org/journals/neuroimaging/about
- Scientific Reports scope: https://www.nature.com/srep/about/aims

## Where Our Cited Work Is Published

### Core Draft References

| Reference | Venue | Lane |
|---|---|---|
| Bontempi et al. 2025, FaceAge | The Lancet Digital Health | clinical digital health / oncology prognosis |
| Peng et al. 2021, SFCN brain age | Medical Image Analysis | medical image analysis / ML method |
| Puglisi et al. 2024, SynthBA | IEEE MetroXRAINE proceedings | engineering / imaging AI conference |
| Hoopes et al. 2022, SynthStrip | NeuroImage | neuroimaging methods |
| Duchesne et al. 2019, SIMON | Scientific Data | dataset descriptor |
| Smith et al. 2019, brain age delta | NeuroImage | neuroimaging methods / biomarker statistics |
| Sullivan & Kaszynski 2019, PyVista | Journal of Open Source Software | software |
| IXI dataset | dataset website | data resource |

### Broader Related-Work Venues

| Topic cluster | Venues represented in our notes |
|---|---|
| Brain-age and neuroimaging ML | NeuroImage, Medical Image Analysis, Human Brain Mapping, Nature Neuroscience, npj Aging, Imaging Neuroscience, ISBI, IEEE MetroXRAINE |
| Face-age / face-health | The Lancet Digital Health, IJCV, CVPR, arXiv, Skin Research and Technology |
| Face-brain / biological aging links | Molecular Psychiatry, BMJ, British Journal of Dermatology, NeuroImage, PLOS Biology |
| Privacy / MRI face re-identification | NEJM, eClinicalMedicine, NeuroImage, Imaging Neuroscience, arXiv |
| Statistical/reporting standards | Radiology: AI, Nature Medicine, Science Advances, JMLR, PLOS Medicine, Nature Human Behaviour |

Interpretation: the citation graph is not pointing to one aging journal. It is
split between neuroimaging methods, medical image analysis, and clinical digital
health. The present evidence is strongest in the first two lanes.

## Venue Fit Ranking

### 1. NeuroImage: Reports - best near-term journal fit

Best fit if the paper remains a rigorous negative/methodological report built
around SIMON + IXI.

Why it fits:

- It is a NeuroImage companion journal.
- Its scope includes methods, databases, theory/conceptual positions, negative
  or null data papers, and replication studies.
- It explicitly accepts scientifically sound neuroimaging work even when the
  finding is negative.

What the paper must emphasize:

- Same-scan face/brain age is a new neuroimaging analysis question.
- The negative result is informative because it prevents overclaiming
  photo-trained face-age transfer to MRI-derived surfaces.
- SIMON is not "small n" in the scanner-stress-test sense, but it is n=1 for
  biological generalization; say this plainly.

Submission shape:

- Original Research or Methods.
- Title should include "same-scan", "scanner reproducibility", and "negative
  transfer" rather than "biological age biomarker".

### 2. Imaging Neuroscience - best aspirational neuroimaging journal

Best fit after adding one or two external validation cohorts beyond SIMON/IXI.

Why it fits:

- It targets major advances in brain imaging methods and neuroimaging-based
  understanding of brain structure/function.
- It is culturally close to the NeuroImage community and more aligned with open
  neuroimaging practice.

What is missing now:

- Current face signal is weak and the biological result is mostly negative.
- The paper needs stronger neuroimaging contribution: e.g. SRPBS/Cao/Huang
  test-retest, OpenBHB/CamCAN external validation, or a robust public pipeline
  with failure-mode analysis.

Submission shape:

- Methods/resource paper.
- Strong open-code, preprocessing QC, and test-retest reliability framing.

### 3. Medical Image Analysis - high-impact but currently too ambitious

Best fit only if the paper becomes an algorithmic medical image analysis paper,
not merely an evaluation of borrowed models.

Why it could fit:

- MedIA explicitly covers MRI, visualization, feature extraction, longitudinal
  studies, shape measurements, statistical shape analysis, and computational
  anatomy.
- Our face-surface rendering + morphometric branch lives in that space.

What is missing now:

- MedIA usually expects a clear methodological advance, broad validation, and
  strong experiments.
- A pipeline using existing models with n=1 longitudinal stress test is not
  enough.

Upgrade needed:

- New robust MRI-face surface extraction method or direct MRI-surface age model.
- Multiple datasets, baselines, ablations, confidence intervals, and open code.

### 4. IEEE Transactions on Medical Imaging - only if method novelty becomes central

Fit condition:

- Make the main contribution a new image-analysis method with substantial
  technical novelty.

Risk:

- TMI explicitly redirects application papers without significant methodological
  innovation.
- The current manuscript is mainly a model-transfer/failure-mode analysis.

### 5. Frontiers in Neuroimaging - pragmatic fallback

Best fit if the goal is a publishable, open-access methods/code article with
moderate selectivity.

Why it fits:

- Sections include Brain Imaging Methods, Computational Neuroimaging, and
  Neuroimaging Analysis and Protocols.
- The journal allows methods and technology/code-oriented work, but requires
  appropriate validation for public-data computational studies.

Risk:

- Lower prestige than NeuroImage/Imaging Neuroscience/MedIA.

### 6. Scientific Reports - broad fallback, not the best signal

Possible fit if the final manuscript is technically sound and broadly framed as
an engineering/health-sciences negative result.

Why it fits:

- Broad scope includes natural sciences, medicine, and engineering.

Risk:

- The venue does not create field-specific neuroimaging prestige.
- Reviewers may be less specialized and ask for more generic validation.

### 7. Communications Medicine / npj Digital Medicine - not now

Not a good current fit.

Why:

- Communications Medicine wants clinical/translational advances important to
  medicine or public health.
- npj Digital Medicine explicitly emphasizes validated clinical applications of
  AI/digital biomarkers and generally does not consider small-scale preliminary
  studies.

Upgrade needed:

- A real clinical outcome: dementia, survival, treatment response, diagnosis, or
  prospectively useful risk stratification.
- External clinical cohorts and EQUATOR/TRIPOD+AI-style reporting.

### 8. The Lancet Digital Health / Nature Aging / npj Aging - no for this version

These are not realistic for the current evidence.

Why:

- FaceAge reached The Lancet Digital Health because it had thousands of cancer
  patients and survival endpoints.
- Aging journals need a biological-aging insight, not just model behavior on MRI
  renders.

Upgrade needed:

- Disease/health-span endpoint, mechanistic aging interpretation, and strong
  external validation.

## Citation Issues Found

1. FaceAge DOI inconsistency:
   - `references.bib` uses `10.1016/j.landig.2025.03.002`.
   - `literature_review.md` and duplicated notes use
     `10.1016/S2589-7500(25)00042-1`.
   - Crossref resolves `10.1016/j.landig.2025.03.002`; the other DOI did not
     resolve in the check. Use the Crossref-resolving DOI unless the journal
     article page contradicts it.

2. Abramian / Steeg eClinicalMedicine DOI issue:
   - `sota_design.md` lists `10.1016/j.eclinm.2024.102509` for MRI
     re-identification.
   - That DOI resolves to a different eClinicalMedicine article.
   - The re-identification paper is Steeg et al. 2024,
     `10.1016/j.eclinm.2024.102930`.

## Recommendation

Primary near-term journal target:

> NeuroImage: Reports

Primary upgraded target after one validation sprint:

> Imaging Neuroscience

Stretch target only after a genuine algorithmic contribution:

> Medical Image Analysis

Do not target clinical digital-health or aging journals until the paper has
clinical outcomes or disease-relevant validation.

## Acceptance-Rate vs Impact-Factor Trade-Off

Important caveat: most journals below do not publish official acceptance rates.
The "fit-adjusted acceptance likelihood" below is therefore not an official
journal statistic. It is a practical estimate for this manuscript, based on
scope fit, selectivity, current evidence strength, and whether the present
story is positive, negative, methodological, or clinical.

### Best Trade-Offs

| Rank | Journal | 2024 JIF / metric | Official acceptance rate | Fit-adjusted likelihood for current paper | Trade-off |
|---|---:|---:|---:|---:|---|
| 1 | NeuroImage: Reports | No JIF yet; CiteScore 3.6 | Not published | Medium-high | Best near-term fit; lower prestige but explicitly friendly to null/replication/methods |
| 2 | Scientific Reports | JIF 3.9; 5-year JIF 4.3 | Broadly reported around ~50%, but verify at submission | Medium | Highest probability among reputable indexed options; weaker field signal |
| 3 | Frontiers in Neuroimaging | No mature JIF found; ESCI/Scopus indexed | Not published | Medium | Good methods/code fit; prestige lower than NeuroImage family |
| 4 | NeuroImage | JIF 4.5; CiteScore 10.8 | Not published | Low-medium | Better prestige than Reports; current paper may be too narrow unless validation is strengthened |
| 5 | Imaging Neuroscience | No JIF yet | Not published | Low-medium now; medium after validation sprint | Strong community prestige despite no JIF; needs a cleaner neuroimaging-method contribution |
| 6 | Brain Informatics | JIF 4.5 | Not published | Medium | Balanced IF/speed, but weaker conceptual fit than NeuroImage: Reports |
| 7 | Communications Medicine | JIF 6.3 | Not published | Low | IF attractive, but clinical/translational bar is not met yet |
| 8 | Medical Image Analysis | JIF 11.8; CiteScore 26.6 | Not published | Very low now; low after validation unless method novelty is added | Excellent impact, poor acceptance odds for current pipeline/evaluation paper |
| 9 | npj Digital Medicine | JIF 15.1 | Not published | Very low | High IF, but current evidence lacks clinical deployment/outcome validation |
| 10 | npj Aging | JIF ~6.0 in 2024 JCR-indexed listings | Not published | Very low | Aging framing is tempting, but the paper is about model transfer/reproducibility, not aging biology |

### Practical Frontier

| Strategy | Expected journal | Why |
|---|---|---|
| Maximize probability while staying reputable | Scientific Reports or Frontiers in Neuroimaging | Soundness/methods-friendly, lower novelty bar |
| Best balance for this manuscript | NeuroImage: Reports | Strongest match to null/methods/reproducibility; no JIF yet but field-relevant |
| Best prestige without overreaching | NeuroImage | JIF 4.5 and direct field fit, but needs stronger validation than current draft |
| Best long-game target | Imaging Neuroscience | No JIF yet, but likely respected by neuroimaging readers; needs validation sprint |
| High-impact stretch | Medical Image Analysis | Only if we add real algorithmic novelty and broad experiments |
| Do not target now | npj Digital Medicine, Communications Medicine, Lancet Digital Health | Clinical outcome/translation bar not met |

### Recommendation by Risk Appetite

Conservative:

> NeuroImage: Reports first. If rejected, transfer/reshape to Scientific Reports
> or Frontiers in Neuroimaging.

Balanced:

> Strengthen with one external test-retest cohort and submit to NeuroImage.
> If desk-rejected, send to NeuroImage: Reports.

Ambitious:

> Add a direct MRI-surface age model or robust surface-extraction method, plus
> multiple datasets and ablations, then target Medical Image Analysis.

My recommendation remains the balanced-conservative route:

1. Add one validation sprint.
2. Submit to NeuroImage.
3. Use NeuroImage: Reports as the high-fit fallback.

This gives a reasonable shot at a field-recognized journal without wasting the
manuscript on clinical/digital-health venues where the core evidence is
misaligned.

## Required Upgrade Before Journal Submission

Minimum viable journal package:

1. Add one external test-retest cohort beyond SIMON.
   - Best low-friction options from local notes: Cao 2015/BNU1, Huang 2016/BNU2,
     Maclaren OpenNeuro ds000239, or SRPBS Traveling Subjects.
2. Add at least one chronological face-age comparator.
   - MiVOLO face-only is the practical open baseline.
3. Run at least two brain-age baselines.
   - SynthBA plus MIDIBrainAge or SFCN.
4. Report raw and age-bias-corrected metrics.
   - MAE, bias, Pearson/Spearman, bootstrap CIs.
5. Report repeatability, not only accuracy.
   - ICC(2,1), Bland-Altman limits, repeatability coefficient.
6. Make the null result the point.
   - The paper should not imply that MRI-derived face age is already a
     biological clock. The defensible contribution is showing where and why the
     transfer fails.

## Candidate Titles

1. Same-Scan Face and Brain Age from T1 MRI: A Scanner Reproducibility and
   Negative-Transfer Study
2. Does the MRI Face Know the Brain's Age? Same-Scan Face-Age and Brain-Age
   Estimation Under Test-Retest Stress
3. MRI-Derived Face Age Does Not Track Brain Age After Age-Bias Correction:
   Evidence from Same-Scan Neuroimaging
