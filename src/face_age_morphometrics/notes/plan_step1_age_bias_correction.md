# Implementation Plan — Step 1: Age-Bias Correction

**Status:** Not started  
**Priority:** Critical prerequisite for all face-age gap analyses (H1, H2, H3)  
**Estimated effort:** ~1–2 hours

---

## Background

The Ridge model exhibits strong regression dilution: error correlates with true age at
r=−0.759 (p<0.001). The model overestimates young subjects (+9 yr for ages 20–35) and
underestimates old subjects (−11.5 yr for ages 65–86).

Applying a standard linear correction (de Lange & Cole 2020; Liang et al. 2019):

| | MAE | Pearson r |
|---|---|---|
| Raw Ridge | 8.31 yr | 0.739 |
| Corrected Ridge | **5.09 yr** | **0.902** |

This is a prerequisite before computing any face-age gap or running H2/H3.

---

## The maths

### Fitting (train set only)

Fit a linear regression of prediction error on true age using training predictions:

```
error  =  b0  +  b1 × age_true
```

where `error = age_pred − age_true`.

Computed from existing train predictions:
- `b1 = −0.4368`  (slope: model underestimates by 0.437 yr per year of true age)
- `b0 = +21.75`   (intercept)

Equivalently, the raw prediction follows `age_pred = 0.563 × age_true + 21.75`.

### Correction formula A — when true age is known (IXI evaluation, SIMON)

```
age_pred_corrected  =  age_pred  −  (b0  +  b1 × age_true)
                     =  age_pred  +  0.437 × age_true  −  21.75
```

Use this for IXI splits (evaluation) and for SIMON (where the subject's chronological
age at each session is available from the phenotype CSV).

### Correction formula B — inference without true age

Invert the regression `age_pred = (1 + b1) × age_true + b0` to obtain:

```
age_pred_corrected  =  (age_pred  −  b0)  /  (1  +  b1)
                     =  (age_pred  −  21.75)  /  0.563
```

Use this when predicting on subjects whose true age is entirely unknown.

Both formulas must be saved and documented so downstream code always uses the
train-fitted correction, never re-fits on val/test/SIMON.

---

## Files to create / modify

### 1. `scripts/apply_age_correction.py`  ← new file

**Inputs:**
- `data/models/benchmark/ridge_{split}_predictions.csv` for split in {train, val, test}
- `data/models/benchmark/ridge_model.joblib` (for reference, not re-fitted)

**Actions:**
1. Load train predictions.
2. Fit `np.polyfit(age_true, error, 1)` → `(b1, b0)`. This must use **train only**.
3. Save correction params to `data/models/benchmark/ridge_correction.json`:
   ```json
   {
     "model":       "ridge",
     "b0":          21.7528,
     "b1":         -0.4368,
     "train_mean_age": 49.3,
     "formula_A":  "corrected = raw_pred - (b0 + b1 * age_true)",
     "formula_B":  "corrected = (raw_pred - b0) / (1 + b1)  [use when age_true unknown]",
     "reference":  "de Lange & Cole (2020) NeuroImage"
   }
   ```
4. Apply formula A to train / val / test → add `age_pred_corrected` column to existing
   prediction CSVs (do not overwrite; add column in place).
5. Print before/after metrics for all three splits.

**CLI:**
```bash
python scripts/apply_age_correction.py
# optional: apply to a custom predictions CSV
python scripts/apply_age_correction.py --predictions path/to/custom_preds.csv --out path/to/output.csv
```

---

### 2. `notebooks/project_overview.ipynb` — new Section 6e  ← add cells

Add after the existing Section 6d (verdict) the following cells:

**Cell A — markdown header:**
```
### 6e. Age-Bias Correction
```

**Cell B — run the script:**
```python
!python ../scripts/apply_age_correction.py
```

**Cell C — before/after metrics table:**
Load both `ridge_test_predictions.csv` (raw) and the corrected column.
Display a 2-row comparison table: Raw vs Corrected for MAE, RMSE, R², r, bias.

**Cell D — before/after scatter plot (2 panels):**
- Left: raw predictions vs true age (coloured by site).
- Right: corrected predictions vs true age (same colour scheme).
- Both panels: identity line, MAE annotation.
- Saved as `data/models/benchmark/ridge_correction_scatter.png`.

**Cell E — age-group bias plot (2 panels):**
- Left: bias by age group before correction (bar chart).
- Right: bias by age group after correction (should be near zero for all groups).
- Saved as `data/models/benchmark/ridge_correction_bias_by_age.png`.

**Cell F — markdown interpretation:**
Document what the correction does biologically, cite de Lange & Cole 2020,
and note that all subsequent analyses (face-age gap, H2, H3) use corrected predictions.

---

### 3. `notebooks/07_predict_new_dataset.ipynb` — update Section 5  ← modify

In the "Scale + Predict" section, after computing `y_pred`:

1. Load `ridge_correction.json`.
2. If session ages are available in `valid_df` (SIMON phenotype CSV must include `age`):
   - Apply formula A: `age_pred_corrected = y_pred − (b0 + b1 × age)`.
3. Else:
   - Apply formula B: `age_pred_corrected = (y_pred − b0) / (1 + b1)`.
4. Add `age_pred_corrected` column to the results CSV.

**SIMON note:** `load_simon_metadata()` in `src/utils.py` currently does not return age.
The SIMON phenotype CSV must be located and the `age` column added to the session table
before formula A can be used. If phenotype data is unavailable, formula B is the fallback.

---

## Implementation order

1. Write `scripts/apply_age_correction.py` and run it. Verify the correction JSON is saved
   and existing CSVs gain the `age_pred_corrected` column.
2. Add cells to `project_overview.ipynb` (Section 6e). Run all cells, check plots.
3. Update `notebooks/07_predict_new_dataset.ipynb` Section 5 to load and apply correction.
4. Update memory: mark Step 1 complete, note corrected artefact paths.

---

## Acceptance criteria

- [ ] `data/models/benchmark/ridge_correction.json` exists with b0, b1, formula strings.
- [ ] All three prediction CSVs (`ridge_{train,val,test}_predictions.csv`) have an
      `age_pred_corrected` column.
- [ ] Test set: MAE ≤ 5.5 yr, r ≥ 0.88 after correction.
- [ ] Age-group bias plot shows near-zero mean error for all four age bins.
- [ ] Notebook 07 loads the correction JSON and applies it (formula A or B) before saving.

---

## Reference

de Lange, A.-M. G. & Cole, J. H. (2020). Commentary: Correction procedures in
brain-age prediction. *NeuroImage: Clinical*, 26, 102229.
