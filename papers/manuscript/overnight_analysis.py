"""
Overnight autoresearch — single-hypothesis test.

Hypothesis (H1): The IXI face-gap × brain-gap Pearson r = 0.31
is predominantly driven by shared age-bias.

Prediction:
    accept H1 iff partial Pearson r(face_gap, brain_gap | chron_age) <= 0.15
    reject H1 iff partial r >= 0.20
    ambiguous otherwise.

Secondary: longitudinal slope analysis on SIMON — does any method track
the subject's aging at all?

Tertiary: per-age-band IXI correlation stratification.

All numerical results are appended to RESULTS.tsv so we can track the
ratchet across sessions.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
TABLES = HERE.parent / "tables"
RESULTS_TSV = HERE / "RESULTS.tsv"


# ── helpers ──────────────────────────────────────────────────────────────────
def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson partial correlation of x and y controlling for z."""
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float("nan")


def log_result(tag: str, **kwargs) -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not RESULTS_TSV.exists()
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    cols = ["ts", "tag"] + list(kwargs.keys())
    row = [ts, tag] + [f"{v:.4g}" if isinstance(v, float) else str(v)
                        for v in kwargs.values()]
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("\t".join(cols) + "\n")
        f.write("\t".join(row) + "\n")


def extract_key(s: str) -> str:
    m = re.search(r"(ses-\d+_run-\d+)", s)
    return m.group(1) if m else s


# ═════════════════════════════════════════════════════════════════════════════
# PRIMARY HYPOTHESIS — IXI gap correlation age-bias confound
# ═════════════════════════════════════════════════════════════════════════════
def run_primary() -> dict:
    print("=" * 74)
    print("PRIMARY — IXI gap correlation, age-bias confound check")
    print("=" * 74)

    gap = pd.read_csv(TABLES / "ixi_gap_correlation.csv",
                       on_bad_lines="skip")
    # trailing SUMMARY row in the CSV — drop any non-numeric ixi_num
    gap = gap[pd.to_numeric(gap["ixi_num"], errors="coerce").notna()].copy()
    gap["true_age"] = gap["true_age"].astype(float)
    gap["face_age_gap"] = gap["face_age_gap"].astype(float)
    gap["brain_age_gap"] = gap["brain_age_gap"].astype(float)
    n = len(gap)
    print(f"\nLoaded N = {n} paired subjects")
    print(f"Chronological age: {gap['true_age'].min():.1f}–"
          f"{gap['true_age'].max():.1f} y, "
          f"mean {gap['true_age'].mean():.1f} ± "
          f"{gap['true_age'].std():.1f}")

    x = gap["face_age_gap"].values
    y = gap["brain_age_gap"].values
    z = gap["true_age"].values

    # (1) raw
    r_raw, p_raw = stats.pearsonr(x, y)
    rho_raw, prho = stats.spearmanr(x, y)
    print(f"\n[raw]               Pearson r = {r_raw:+.3f} (p={p_raw:.3g})")
    print(f"                     Spearman ρ = {rho_raw:+.3f} (p={prho:.3g})")

    # (2) each gap vs chron_age
    face_slope, face_icept, fr, fp, _ = stats.linregress(z, x)
    brain_slope, brain_icept, br, bp, _ = stats.linregress(z, y)
    print(f"\n[age-bias]          face_gap  vs age  r = {fr:+.3f} "
          f"(slope {face_slope:+.3f}, p={fp:.3g})")
    print(f"                     brain_gap vs age  r = {br:+.3f} "
          f"(slope {brain_slope:+.3f}, p={bp:.3g})")

    # (3) bias-corrected residuals
    x_corr = x - (face_slope * z + face_icept)
    y_corr = y - (brain_slope * z + brain_icept)
    r_corr, p_corr = stats.pearsonr(x_corr, y_corr)
    print(f"\n[bias-corrected]    r(residuals) = {r_corr:+.3f} "
          f"(p={p_corr:.3g})")

    # (4) partial correlation controlling for age
    pc = partial_corr(x, y, z)
    print(f"[partial r | age]   = {pc:+.3f}")

    # (5) decision
    print("\n── Decision ────────────────────────────────────────────")
    print(f"  raw r              : {r_raw:+.3f}")
    print(f"  bias-corrected r   : {r_corr:+.3f}")
    print(f"  partial r | age    : {pc:+.3f}")

    if pc <= 0.15:
        decision = "ACCEPT H1 — the claim is age-bias-confounded."
    elif pc >= 0.20:
        decision = "REJECT H1 — shared-aging claim survives age control."
    else:
        decision = "AMBIGUOUS — report both raw and partial, add sensitivity."
    print(f"  VERDICT: {decision}")

    # save corrected gaps
    gap["face_gap_corr"] = x_corr
    gap["brain_gap_corr"] = y_corr
    gap.to_csv(HERE / "ixi_gap_corrected.csv", index=False)

    result = {
        "n": n,
        "r_raw": float(r_raw),
        "r_bias_corrected": float(r_corr),
        "partial_r_given_age": float(pc),
        "face_gap_age_slope": float(face_slope),
        "brain_gap_age_slope": float(brain_slope),
        "decision": decision,
    }
    log_result("primary_gap_confound", **result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECONDARY — SIMON longitudinal slope analysis
# ═════════════════════════════════════════════════════════════════════════════
def run_secondary() -> list[dict]:
    print("\n" + "=" * 74)
    print("SECONDARY — SIMON longitudinal tracking")
    print("=" * 74)

    brain = pd.read_csv(TABLES / "simon_brainage_synthba.csv")
    brain["key"] = brain["scan_key"].apply(
        lambda x: extract_key(x.split("|")[-1]))

    face_mv = pd.read_csv(TABLES / "simon_faceage_multiview_raw.csv")
    face_mv["key"] = face_mv["subject_id"].apply(extract_key)

    face_morph = pd.read_csv(TABLES / "simon_faceage_morphometrics.csv")
    face_morph["key"] = face_morph["subject_id"].apply(extract_key)

    simon = (brain[["key", "chron_age", "predicted_age"]]
             .rename(columns={"predicted_age": "brain_pred"})
             .merge(face_morph[["key", "predicted_age"]]
                    .rename(columns={"predicted_age": "morph_pred"}),
                    on="key", how="outer")
             .merge(face_mv[["key", "predicted_age"]].dropna()
                    .rename(columns={"predicted_age": "mv_pred"}),
                    on="key", how="outer"))

    print(f"\n{'method':<22s}  {'N':>3s}  {'slope':>7s}  {'SE':>6s}  "
          f"{'p':>9s}  {'R²':>5s}  {'bias':>7s}")
    print("-" * 74)

    rows = []
    for label, col in [("SynthBA (brain)", "brain_pred"),
                       ("Face morphometrics", "morph_pred"),
                       ("FaceAge multiview", "mv_pred")]:
        s = simon.dropna(subset=[col, "chron_age"])
        x, y = s["chron_age"].values, s[col].values
        slope, icept, r, p, se = stats.linregress(x, y)
        bias = (y - x).mean()
        print(f"{label:<22s}  {len(s):>3d}  {slope:>+7.3f}  {se:>6.3f}  "
              f"{p:>9.2e}  {r**2:>5.3f}  {bias:>+7.2f}")
        rows.append({"method": label, "n": len(s), "slope": float(slope),
                     "slope_se": float(se), "slope_p": float(p),
                     "r2": float(r ** 2), "bias": float(bias)})
        log_result("secondary_simon_slope", method=label, n=len(s),
                   slope=float(slope), slope_p=float(p),
                   r2=float(r ** 2), bias=float(bias))

    print("\nReading:  slope → 1  tracks aging;  slope → 0  frozen (OOD).")

    best = max(rows, key=lambda r: r["slope"])
    print(f"\nBest longitudinal tracker: {best['method']} "
          f"(slope {best['slope']:+.3f}, p={best['slope_p']:.2g})")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# TERTIARY — IXI correlation stratified by age band
# ═════════════════════════════════════════════════════════════════════════════
def run_tertiary() -> list[dict]:
    print("\n" + "=" * 74)
    print("TERTIARY — IXI gap correlation stratified by age")
    print("=" * 74)

    gap = pd.read_csv(TABLES / "ixi_gap_correlation.csv",
                       on_bad_lines="skip")
    gap = gap[pd.to_numeric(gap["ixi_num"], errors="coerce").notna()].copy()
    for c in ("true_age", "face_age_gap", "brain_age_gap"):
        gap[c] = gap[c].astype(float)
    bins = [(20, 45), (45, 65), (65, 86)]

    print(f"\n{'age band':<10s}  {'N':>3s}  {'raw r':>7s}  "
          f"{'partial r|age':>14s}  {'p_raw':>8s}")
    print("-" * 50)

    rows = []
    for lo, hi in bins:
        s = gap[(gap["true_age"] >= lo) & (gap["true_age"] < hi)]
        if len(s) < 10:
            print(f"{lo}-{hi}       {len(s):>3d}  (skipped, N<10)")
            continue
        r, p = stats.pearsonr(s["face_age_gap"], s["brain_age_gap"])
        pc = partial_corr(s["face_age_gap"].values,
                          s["brain_age_gap"].values,
                          s["true_age"].values)
        print(f"{lo:>3d}-{hi:<3d}  {len(s):>5d}  {r:>+7.3f}  "
              f"{pc:>+14.3f}  {p:>8.2g}")
        rows.append({"band": f"{lo}-{hi}", "n": len(s),
                     "r_raw": float(r), "partial_r": float(pc),
                     "p": float(p)})
        log_result("tertiary_ixi_strat", band=f"{lo}-{hi}",
                   n=len(s), r_raw=float(r), partial_r=float(pc),
                   p=float(p))

    return rows


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    primary = run_primary()
    secondary = run_secondary()
    tertiary = run_tertiary()

    print("\n" + "=" * 74)
    print("FINAL SUMMARY")
    print("=" * 74)
    print(f"  IXI raw r                = {primary['r_raw']:+.3f}")
    print(f"  IXI partial r | age      = {primary['partial_r_given_age']:+.3f}")
    print(f"  IXI bias-corrected r     = {primary['r_bias_corrected']:+.3f}")
    print(f"  VERDICT on H1: {primary['decision']}")
    print()
    for r in secondary:
        print(f"  SIMON {r['method']:<22s} slope = {r['slope']:+.3f} "
              f"(p={r['slope_p']:.2g})")
    print()
    print(f"  Results log → {RESULTS_TSV}")
