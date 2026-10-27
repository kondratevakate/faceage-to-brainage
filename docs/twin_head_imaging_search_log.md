# Twin head-imaging data discovery search log

**Version:** 1.0, 2026-07-20.

**Scope:** scoping data-discovery audit, not a systematic review or proof that
unpublished collections do not exist.

## Objective

Identify twin or sibling-family datasets that can support FaceAge, BrainAge, or
paired face-brain research. Eligible visual representations include MRI, fMRI
with an anatomical T1/T2, CT/CBCT, PET attenuation CT, MEG/EEG-linked MRI and
head digitization, 2D clinical photographs, stereophotogrammetry, structured
light, laser scans, video, cephalograms, dental models, fundus/OCT, iris and
thermal imaging.

The primary target is a stable participant/family identifier linking at least
one face/head representation to chronological age. The highest-value target
links a usable facial representation and non-defaced T1 MRI in the same person.

## Sources searched

- PubMed/PMC, publisher full-text pages and reference lists;
- institutional twin-registry, cohort, laboratory and data-access pages;
- OpenNeuro and stable DataLad/GitHub mirrors, NDA, NIST BDbC and national
  research-data catalogues;
- grant and institutional repositories in Japan, China, Taiwan, South Korea,
  Australia, Europe, Latin America and Africa;
- AAOF craniofacial legacy collections;
- commercial dataset catalogues and custom collection vendors;
- forward/backward citation chaining from twin-registry reviews and imaging
  papers.

Dynamic or login-only search cards were not accepted as evidence unless an
independent paper, institutional page or working vendor sample route confirmed
their contents.

## Concept blocks

Queries combined terms from each relevant block:

```text
(twin OR twins OR monozygotic OR dizygotic OR co-twin OR sibling registry)
AND
(MRI OR magnetic resonance OR fMRI OR CT OR CBCT OR PET OR MEG OR T1 OR T2)
AND/OR
(face OR facial OR craniofacial OR head OR photograph OR photogrammetry OR
 stereophotogrammetry OR 3DMD OR VECTRA OR structured light OR cephalogram OR
 dental cast OR palate OR landmark OR fundus OR OCT OR iris OR thermal)
AND
(dataset OR cohort OR registry OR archive OR repository OR data access)
```

Additional searches used modality-led terms because the facial data often
appear only as an ancillary acquisition: `fMRI anatomical T1`, `MEG head shape
MRI`, `PET attenuation CT`, `vocal tract MRI`, `orthodontic twin archive`,
`perceived age twins`, `medical photography twins`, and `craniofacial growth
twin study`.

Local-language variants included Chinese (`双胞胎`, `磁共振`, `颅面`, `三维人脸`),
Japanese (`双生児`, `MRI`, `顔面`, `三次元`), Korean (`쌍둥이`, `자기공명`, `두개안면`),
Spanish (`gemelos`, `resonancia`, `cara`, `craneofacial`), Portuguese
(`gêmeos`, `ressonância`, `face`) and French terms used in African institutional
searches. These searches mostly returned duplicates or papers without a raw-data
route.

## Inclusion and classification

Include a resource when at least one durable source confirms: a twin/family
design, an image or 3D head representation, a sample unit, and a plausible data
holder. Each row is classified as `OPEN`, `LICENSE`, `DUA`, `COMMERCIAL`,
`PAPER`, `UNKNOWN FACE` or `NO FACE` in the
[main catalogue](twin_head_imaging_datasets.md).

Exclude or downgrade:

- model/code releases with no participant-level images;
- empty marketplace cards or broken template pages;
- aggregate landmark tables when no image archive or holder can be identified;
- an fMRI cohort whose anatomical volume is absent or explicitly defaced for a
  facial-reconstruction use case;
- claims of same-subject linkage inferred only from a shared institution.

## Final-search additions

Two important resources appeared only after expanding beyond `dataset` queries:

1. **RATSS, Sweden:** >400 participants currently; a published sample has 372
   twins with multimodal MRI and 280 with professional medical facial
   photographs. It is the strongest newly identified direct paired lead.
2. **Forsyth/Moorrees Twin Study, USA:** nearly 500 pairs/533 families with
   annual craniofacial growth records, facial photographs, cephalograms and 3D
   study casts. It is a high-priority longitudinal FaceAge archive.
3. **EMIF-AD PreclinAD / NTR, Netherlands:** 67 complete MZ pairs have a
   published same-person fundus-photography and MRI analysis; the wider twin
   protocol includes OCT, fundus autofluorescence, rs-fMRI, PET and MEG.
4. **Monash/Australian Twin Registry T2D study:** 35 people from 18 twin pairs
   have bilateral OCT, retinal photographs and multimodal 3T MRI.

The final pass also consolidated multiple Korean cephalometric papers into a
single Samsung Medical Center source archive of 553 participants, rather than
treating each article subset as an independent dataset.

## Coverage assessment

| Domain | Confidence | Residual blind spot |
|---|---|---|
| Named published twin MRI/fMRI cohorts | High | Original non-defaced volumes may be retained privately. |
| Published 2D/3D twin face and orthodontic cohorts | High-moderate | Historical archives may be undigitized or described only in theses. |
| Same-person face-MRI overlap | Moderate-low | Often known only to the holder and absent from papers. |
| Asia, Latin America and Africa | Moderate for published records | Local-language theses, hospital PACS and inactive registries remain hard to index. |
| Commercial/off-the-shelf twin data | Low-moderate | Vendor catalogues change and custom collections are usually confidential. |

Therefore, “all” can only mean saturation of the documented search space. It
cannot mean absence of unadvertised hospital, forensic, dental or commercial
collections.

## Completion gates

The discovery phase is operationally complete when all are true:

1. The query/source log and catalogue are versioned and their links audited.
2. Every P0/P1 resource has a named holder, working contact route and one exact
   unresolved access question.
3. At least ten holders receive a first contact and two timed follow-ups.
4. Every holder is asked both about same-ID modality overlap and about unlisted
   related collections or colleagues to contact.
5. A second citation/holder snowball wave yields no new eligible P0/P1 resource.
6. New discoveries are logged with date and reason rather than silently added.

After these gates, discovery changes from an open-ended search into quarterly
surveillance. The access campaign and response tracker are maintained in
[`data/twin_dataset_outreach.csv`](../data/twin_dataset_outreach.csv).
