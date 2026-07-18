# FaceAge search log

## Scope and date

Search performed 2026-07-18. This is a reproducible rapid search for protocol
design, not an exhaustive systematic-review search.

## Sources

- PubMed for biomedical primary studies and reviews.
- Crossref and DOI landing pages for bibliographic verification.
- CVF and ECVA proceedings for computer-vision methods.
- Official DECA, NoW, MICA, FLAME, FaceBase, Headspace, and FaceScape pages for
  benchmark definitions, access conditions, and model provenance.

## Search strings

```text
("perceived age" OR "facial age" OR "face age") AND
  (mortality OR survival OR health OR biomarker)

("facial age estimation" OR "apparent age estimation") AND
  (deep learning OR machine learning) AND (validation OR benchmark)

("3D facial morphology" OR "3D face") AND
  (aging OR longitudinal OR age trajectory)

("3D face reconstruction") AND
  (metrical OR scan-to-mesh OR benchmark OR repeatability)

(face OR facial) AND (biomarker OR health) AND
  (scoping review OR systematic review)
```

Additional exact-title searches were run for FaceAge, DEX, DECA, MICA, FLAME,
NoW, the 3D Facial Norms Database, Headspace/LYHM, and FaceScape.

## Inclusion rules

- Human face photographs, facial surfaces, or head scans.
- A reported age, health outcome, repeated measurement, longitudinal interval,
  or physical 3D reference.
- Peer-reviewed primary evidence preferred; official benchmark papers accepted.
- Reviews retained to map the field and identify risk-of-bias patterns.

## Exclusion rules

- Disease, personality, behaviour, ethnicity, or social-trait inference without
  a suitable reference outcome and external validation.
- Render-quality studies without geometric or task evaluation.
- Age-progression demonstrations without quantitative held-out evaluation.
- Secondary news reports when the primary paper was available.

## Verification notes

- The verified FaceAge article is *Lancet Digital Health* 2025, article 100870,
  DOI `10.1016/j.landig.2025.03.002`.
- NoW has separate metrical and non-metrical leaderboards; results cannot be
  mixed because one permits scale alignment.
- FaceBase individual-level meshes are controlled access; summary-level data are
  open. Headspace access and licensing must be confirmed before acquisition.
- Search results were screened by one reviewer. Counts were not used to construct
  a PRISMA flow, and no claim of exhaustive coverage is made.
