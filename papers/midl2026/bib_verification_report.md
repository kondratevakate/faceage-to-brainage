# Citation verification report -- papers\midl2026\midl-samplebibliography.bib

**Summary:** 7 entries -- mismatch=1, needs_review=1, substituted=1, verified_by_doi=4

## [SUBST] duchesne2019simon -- substituted (risk 68/100)

**Field classification** (C=Correct, P=Partial, S=Substituted, F=Fabricated, M=Missing, X=N/A): author=S, doi=C, title=P, year=C

- escalated from 'verified_by_doi' to 'substituted' based on field-level classification
- author overlap with resolved record is only 31% (threshold 60% for 3+ author entries) -- likely wrong paper

**Match found:**
- source: crossref
- doi: 10.1038/s41597-019-0262-8
- title: Structural and functional multi-platform MRI series of a single human volunteer over more than fifteen years
- year: 2019
- type: journal-article
- authors: ['Duchesne, Simon', 'Dieumegarde, Louis', 'Chouinard, Isabelle', 'Farokhian, Farnaz', 'Badhwar, Amanpreet', 'Bellec, Pierre', 'Tétreault, Pascal', 'Descoteaux, Maxime', 'Boré, Arnaud', 'Houde, Jean-Christophe', 'Beaulieu, Christian', 'Potvin, Olivier']

## [WARN] ixi -- mismatch (risk 54/100)

- closest match similarity 0.19 too low: 'Figure 7: Denoising results of different models on an image of the IXI dataset w' (DOI: 10.7717/peerjcs.1579/fig-7)

## [REVIEW] puglisi2024synthba -- needs_review (risk 38/100)

**Field classification** (C=Correct, P=Partial, S=Substituted, F=Fabricated, M=Missing, X=N/A): author=P, doi=M, title=P, year=C

- arXiv ID 2406.00365 did not resolve
- escalated from 'verified_by_title' to 'needs_review' based on field-level classification

**Match found:**
- source: crossref/openalex
- title: SynthBA: Reliable Brain Age Estimation Across Multiple MRI Sequences and Resolutions
- similarity: 0.898
- doi: 10.1109/metroxraine62247.2024.10796114
- year: 2024
- authors: ['Puglisi, Lemuel', 'Rondinella, Alessia', 'De Meo, Linda', 'Guarnera, Francesco', 'Battiato, Sebastiano', 'Ravì, Daniele']

**Suggested replacement:**
```bibtex
@inproceedings{puglisi2024synthba,
  title   = {SynthBA: Reliable Brain Age Estimation Across Multiple MRI Sequences and Resolutions},
  author  = {Puglisi, Lemuel and Rondinella, Alessia and De Meo, Linda and Guarnera, Francesco and Battiato, Sebastiano and Ravì, Daniele},
  booktitle = {2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)},
  year    = {2024},
  doi     = {10.1109/metroxraine62247.2024.10796114}
}
```

## [OK] bontempi2025faceage -- verified_by_doi (risk 0/100)

**Field classification** (C=Correct, P=Partial, S=Substituted, F=Fabricated, M=Missing, X=N/A): author=C, doi=C, title=P, year=C

**Match found:**
- source: crossref
- doi: 10.1016/j.landig.2025.03.002
- title: FaceAge, a deep learning system to estimate biological age from face photographs to improve prognostication: a model development and validation study
- year: 2025
- type: journal-article
- authors: ['Bontempi, Dennis', 'Zalay, Osbert', 'Bitterman, Danielle S', 'Birkbak, Nicolai', 'Shyr, Derek', 'Haugg, Fridolin', 'Qian, Jack M', 'Roberts, Hannah', 'Perni, Subha', 'Prudente, Vasco', 'Pai, Suraj', 'Dekker, Andre', 'Haibe-Kains, Benjamin', 'Guthier, Christian', 'Balboni, Tracy', 'Warren, Laura', 'Krishan, Monica', 'Kann, Benjamin H', 'Swanton, Charles', 'De Ruysscher, Dirk', 'Mak, Raymond H', 'Aerts, Hugo J W L']

## [OK] peng2021sfcn -- verified_by_doi (risk 0/100)

**Field classification** (C=Correct, P=Partial, S=Substituted, F=Fabricated, M=Missing, X=N/A): author=C, doi=C, title=C, year=C

**Match found:**
- source: crossref
- doi: 10.1016/j.media.2020.101871
- title: Accurate brain age prediction with lightweight deep neural networks
- year: 2021
- type: journal-article
- authors: ['Peng, Han', 'Gong, Weikang', 'Beckmann, Christian F.', 'Vedaldi, Andrea', 'Smith, Stephen M.']

## [OK] pyvista -- verified_by_doi (risk 0/100)

**Field classification** (C=Correct, P=Partial, S=Substituted, F=Fabricated, M=Missing, X=N/A): author=C, doi=C, title=P, year=C

**Match found:**
- source: crossref
- doi: 10.21105/joss.01450
- title: PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)
- year: 2019
- type: journal-article
- authors: ['Sullivan, C.', 'Kaszynski, Alexander']

## [OK] synthstrip -- verified_by_doi (risk 0/100)

**Field classification** (C=Correct, P=Partial, S=Substituted, F=Fabricated, M=Missing, X=N/A): author=C, doi=C, title=C, year=C

**Match found:**
- source: crossref
- doi: 10.1016/j.neuroimage.2022.119474
- title: SynthStrip: skull-stripping for any brain image
- year: 2022
- type: journal-article
- authors: ['Hoopes, Andrew', 'Mora, Jocelyn S.', 'Dalca, Adrian V.', 'Fischl, Bruce', 'Hoffmann, Malte']
