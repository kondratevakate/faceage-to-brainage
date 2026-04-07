# Face Age Related Works

This page focuses on open face-age systems that could plausibly be tested on the frontal RGB face renders produced by this repository.

## Testable now

| Work | Predicts | Training data | Expected input | Public access | Relevance to MRI renders | Integration effort |
| --- | --- | --- | --- | --- | --- | --- |
| FaceAge | Biological age / face-derived health age | Official repo describes a curated and rebalanced IMDb-Wiki-derived training set of 56,304 age-labelled images, with curated UTK validation/demo data; the paper reports large-scale clinical validation in oncology cohorts | RGB face crop or portrait photo | Paper + official GitHub + official Drive weight link | Highest relevance: this repo already renders frontal RGB face PNGs and already wraps FaceAge inference | Existing wrapper in `src/face_age.py` |
| MiVOLO | Apparent or chronological age plus gender | IMDB-cleaned, UTKFace, Lagenda | RGB face-only crop or face+body photo | Paper + official GitHub + HuggingFace/Drive checkpoints | Useful as a modern open baseline, but many checkpoints rely on body context that MRI renders do not contain | New wrapper needed; start with face-only checkpoints |

### FaceAge

- Why it matters here: it is already the primary face branch in this repository, and its expected input is closest to the rendered face PNGs produced from T1 MRI.
- Input shape: standard RGB face photography, typically via a detected or pre-cropped face.
- Dataset note: the official repository explicitly documents an IMDb-Wiki-derived curated training set and curated UTK validation/demo data. The paper-level clinical story is broader than the public demo assets, so it is worth keeping those two levels separate in documentation.
- Repo fit: best available open option for immediate testing on MRI renders.

### MiVOLO

- Why it matters here: it is a strong modern open age-estimation baseline with actively maintained checkpoints.
- Input shape: either face-only or face+body RGB photos. For this repository, only face-only checkpoints are a good first match.
- Dataset note: the official repo exposes checkpoints and data instructions for IMDB-cleaned, UTKFace, and Lagenda.
- Repo fit: promising comparator to FaceAge, but it needs a new wrapper and careful checkpoint choice because body-conditioned models are a poor fit for MRI-rendered faces.

## Promising but not openly runnable

| Work | What looks promising | Why it is not in the testable bucket |
| --- | --- | --- |
| FAHR-Face / FAHR-FaceAge | It is the most obvious next-generation follow-up to FaceAge and targets health recognition from face photographs at much larger scale | During this review, no confirmed official public checkpoint download path was found. Until that changes, it should stay out of the runnable shortlist |

### FAHR-Face / FAHR-FaceAge

- Why it matters here: it comes from the same broader research line as FaceAge and is conceptually the strongest candidate for a better face-side backbone.
- Input shape: face photographs, not MRI.
- Dataset note: the paper reports large-scale pretraining and age-task fine-tuning on public face-photo corpora.
- Repo fit: scientifically interesting, but not implementation-ready for this repository without a confirmed official checkpoint release.

## Sources

- FaceAge paper and official repo: <https://github.com/AIM-Harvard/FaceAge>
- MiVOLO paper and official repo: <https://github.com/WildChlamydia/MiVOLO>
- FAHR-Face cited from the official FaceAge repository and preprint listing: <https://github.com/AIM-Harvard/FaceAge>
