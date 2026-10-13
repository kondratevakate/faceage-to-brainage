# Clinical face-derived biomarkers search log

Companion to
[clinical_face_evidence_review.md](clinical_face_evidence_review.md).

## Scope and date

Search performed 2026-10-13. This is a reproducible rapid search covering the
clinical use of facial images and facial video **outside** facial age
estimation, which is logged separately in [search_log.md](search_log.md).

Purpose: establish what a face is claimed to measure clinically, how strong
that evidence is, which claims have been retracted and why, and what
permission regime applies — in order to constrain our own stage-one and
stage-two design decisions and to set the claim ceiling for any FaceAge result
we publish.

## Sources

- **PubMed** for primary studies, meta-analyses, and the retraction record
  (`Retracted Publication[pt]` and `Retraction Notice[pt]`).
- **Consensus** semantic search over Semantic Scholar / OpenAlex / PubMed, used
  where PubMed indexing is thin (computer-vision venues, some 2026 journals).
- **Publisher retraction notices** via PMC for the stated reasons, because
  PubMed carries no abstract for retraction records.
- **arXiv abstract pages** for benchmark and preprint specifications.
- **Notre Dame CVRL datasets page** for twin-dataset composition and access
  terms.
- **Apple security documentation** for the Face ID training-data claim, used
  only to separate biometric verification from clinical phenotyping.

## Search strings

PubMed, after shortening — note that PubMed ANDs every term, so long natural
language queries returned zero and had to be reduced:

```text
facial emotion recognition autism meta-analysis
facial emotion recognition schizophrenia meta-analysis
hypomimia computer vision Parkinson
facial expression depression automated analysis
amyotrophic lateral sclerosis facial action units bulbar
Toronto NeuroFace dataset orofacial assessment
automated pain assessment facial action units dementia
remote photoplethysmography skin tone bias
contactless blood pressure estimation facial video
smartphone image conjunctival pallor anemia detection
facial recognition technology medical ethics privacy
GestaltMatcher facial phenotype rare disorders
Genetic syndromes screening facial recognition technology Orphanet
comparison facial emotion processing neurological psychiatric conditions
monozygotic dizygotic twins facial morphology heritability 3D
smartphone photogrammetry 3D facial anthropometry accuracy validation
autism facial images deep learning Retracted Publication
artificial intelligence face recognition detect emotions children autism
Automated identification autism spectrum disorder facial images explainable deep learning
Ambiguous facial expression detection autism
Retraction Note facial expressions detect Parkinson disease videos collected online
```

Consensus:

```text
deep learning facial photographs diagnosis of genetic syndromes accuracy
facial emotion recognition deficits in autism spectrum disorder meta-analysis
hypomimia Parkinson's disease automated facial expression analysis computer vision
deep learning biological age estimation from face photographs clinical outcomes
facial video deep learning differentiating bipolar disorder from major depressive disorder
```

Web, for benchmark and dataset specifications:

```text
NoW benchmark monocular 3D face reconstruction error millimeters DECA MICA FLAME comparison
REALY benchmark region-wise 3D face reconstruction evaluation identity shape accuracy
twin face image dataset ND-TWINS-2009-2010 Twins Days face recognition benchmark
TwinsUK cohort 3D facial imaging twins face scans available researchers
```

## Inclusion rules

- Human facial photographs, facial video, facial surfaces, or head scans used
  as the measurement.
- A reported reference standard: molecular diagnosis, a validated clinical
  scale, a structured diagnostic interview, a medical device, a laboratory
  value, or a physical 3D reference.
- Meta-analyses and systematic reviews retained to establish effect sizes for
  behavioural constructs.
- Benchmark specification papers accepted for reconstruction-error figures.
- Retracted papers included **only** in the retraction audit, never as
  evidence.

## Exclusion rules

- Disease, trait, or behaviour inference from a face without a reference
  standard and external validation.
- Accuracy figures from datasets whose provenance the authors describe as a
  public web or Kaggle collection with medical labels, unless cited as a
  provenance warning.
- Secondary news coverage or vendor marketing when a primary source exists.
- Out of scope this round: facial dermatology, ophthalmology, oncology pain
  assessment, endocrine facial phenotypes, forensic age estimation.

## Verification notes

Every numerical claim in the review was read from an abstract, retraction
notice, or benchmark page retrieved in the same session. No claim was written
from recall. Specific verification events worth recording:

1. **Parkinson retraction.** PubMed flags
   `10.1038/s41746-021-00502-8` as `Retracted Publication`; the notice
   `10.1038/s41746-023-00761-7` was located separately. The frequently cited
   95.6% accuracy figure therefore cannot be used.
2. **Retraction cluster is larger than reported elsewhere.** The filter
   `autism AND facial AND images AND deep learning AND Retracted
   Publication[pt]` returns **11** records, not the three that circulate in
   secondary summaries. The common thread is visible in the retracted papers'
   own abstracts: a Kaggle facial collection of 2,940-3,014 images in a 1:1
   split.
3. **Retraction reasons required publisher text.** PubMed retraction records
   carry no abstract. Notice text was retrieved from PMC for PLOS ONE
   (`PMC12742781`) and for *Scientific Reports* 2024 (`PMC11574120`). For
   `10.1038/s41598-025-11847-5` retrieval failed: nature.com returns HTTP 303
   to an authorisation endpoint and PMC returned a reCAPTCHA interstitial.
   Reasons and author position for that record are therefore **unverified** and
   are marked as such in the review.
4. **Author-agreement direction.** The PLOS ONE notice states "All authors did
   not agree with the retraction"; the *Scientific Reports* 2024 notice states
   "The Authors agree with this retraction." Secondary summaries had these
   reversed.
5. **Emoface journal and figures corrected.** The journal is *npj Mental Health
   Research*, not *Nature Mental Health*; accuracies are 95.29% (BD) and
   87.05% (MDD).
6. **DeepGestalt training-set size corrected** to "over 17,000 images" per the
   paper's own abstract.
7. **Bediou et al. 2012 corrected** to *Frontiers in Psychology* 3:98, and
   identified as a re-analysis of four of the authors' own prior studies by
   Cohen's d rather than a primary comparative study.
8. **Mean-face baseline is cross-paper.** The NoW figure for the average FLAME
   mesh (1.92 mm mean, 1.53 mm with metric shape correction) comes from an
   in-paper comparison whose table layout the retrieving source flagged as
   ambiguous. It is reported in the review with that caveat. Resolving it
   properly requires reading the MICA paper's metric table directly — **open
   action**.
9. **TwinsUK sex imbalance.** One description of the imaging collection gives
   1,521 twins aged 16.5-87.3 with 98.6% female. Access terms for the 3D scans
   could not be confirmed from published sources; published studies state only
   written informed consent for academic use. Contacting the study team is an
   **open action**, tracked in `papers/outreach/`.
10. **Notre Dame access terms** were read from the CVRL datasets page: a
    per-dataset licence signed by someone with authority to make legal
    commitments for the institution (not a student or postdoc), submitted from
    an institutional address, delivered via Globus, with online redistribution
    outside the research community forbidden. Commercial terms are not stated
    on the page and must be read in the individual agreements.
11. **FaceApp disambiguation.** Two unrelated products share the name. The
    *Bellus3D FaceApp* evaluated against 3dMD in the cleft-palate validation
    study is a 3D facial-capture application; the Wireless Lab FaceApp is a
    photo editor with no relation to facial recognition or phone unlocking. The
    training data of the latter could not be verified and no claim is made
    about it. Apple's facial-matching networks are documented by Apple as built
    from over a billion images including infrared and depth, collected with
    participants' informed consent, but the dataset is unpublished and the
    figure is not independently checkable; it is recorded only to note that
    biometric verification embeddings are trained to be *invariant* to state,
    which is the opposite objective to health phenotyping.

## Claims that could not be verified

Listed in full in the review's
[Unverified and excluded claims](clinical_face_evidence_review.md#unverified-and-excluded-claims)
table so that they are not reintroduced into project documents. They include a
Parkinson accuracy pair, an ALS action-unit result, two ASD effect sizes, an
anaemia video AUC, a cardiometabolic facial-video claim attributed to ESC 2026,
and a stroke triage AUC.

## Coverage limits

- No duplicate screening, no full-database export, no formal meta-analysis, no
  risk-of-bias scoring, no author contact.
- PubMed's term-ANDing behaviour means recall depends on query brevity; several
  topics were reached only after shortening, and other phrasings may surface
  additional records.
- Consensus returns ten results per query on the current plan, so coverage of
  computer-vision venues is indicative rather than exhaustive.
- The retraction audit reflects PubMed indexing on the search date. Retractions
  are added continuously; this query should be re-run before manuscript
  submission.

## Re-run instructions

The retraction audit is the part most likely to go stale. Re-run before any
submission that cites face-based diagnostic literature:

```text
PubMed: autism AND facial AND images AND deep learning AND Retracted Publication[pt]
PubMed: (face OR facial) AND (deep learning OR machine learning) AND Retracted Publication[pt]
```

Then check each newly flagged record against the provenance rule in the
review's `dataset_ethics_status` table before citing it for any purpose other
than the audit itself.
