# E05 — SynthBA on IXI (subset)

- **Author**: KK
- **Date**: 2026-04-14
- **Status**: data-saved but **incomplete cohort** (105 / ~581)
- **Notebook**: [`notebooks/07_synthba_colab.ipynb`](../../notebooks/07_synthba_colab.ipynb)

## Question
What is SynthBA's chronological-age MAE on IXI?

## Method
SynthStrip skull strip → SynthBA T1 model on the stripped volume,
default SynthBA preprocessing.

## Result
| metric | value |
|---|---|
| N | **105** |
| MAE | **6.33 yr** |
| RMSE | 7.82 |
| Pearson r | 0.84 |
| bias | −4.15 yr |

Output: [`papers/tables/ixi_brainage_synthba.csv`](../tables/ixi_brainage_synthba.csv).

## What's left undone (CRITICAL)
- **Pipeline stopped at 105 / 581.** The Colab batch was scoped or
  interrupted; no per-subject failure log exists. See
  [E13](E13_data_audit.md). Re-running on the full cohort is action
  item **A1**.
- No bias-corrected MAE reported (Smith et al. 2019 standard).
- No comparison against SFCN, MIDIBrainAge, BrainIAC on the same N.

## Next
[E07](E07_gap_correlation_raw.md) used these 105 to compute the gap
correlation that turned out to be confounded ([E11](E11_gap_confound_check.md)).
