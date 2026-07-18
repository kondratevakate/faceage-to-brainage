# Manuscript

This directory contains the venue-independent manuscript and its reproducible
supporting artifacts. No journal or conference target is currently selected.

Primary files:

- `manuscript.tex` - current article source;
- `references.bib` - bibliography;
- `ex_render.png`, `simon_chron_vs_predicted.pdf`, and
  `synthba_training_vs_predictions.pdf` - manuscript figures;
- `HYPOTHESIS.md`, `FINDINGS.md`, and `RESULTS.tsv` - analysis provenance;
- `pipeline.drawio` - editable pipeline diagram.

Build with:

```bash
make
```

The manuscript is a longitudinal single-subject stress test. It does not
establish population accuracy, clinical utility, or a validated biological-age
biomarker. Venue-specific formatting should be added only after a target is
chosen.
