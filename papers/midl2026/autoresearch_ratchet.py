"""
Autoresearch ratchet — 10 hypotheses, each pre-registered with a decision
rule, each logging to the shared RESULTS.tsv so the Karpathy-style
accept/reject loop can be read post-hoc.

Primary finding from overnight_analysis.py:
    - IXI gap correlation is confounded by shared age-bias
      (partial r | age = -0.015, raw r = +0.31)

This ratchet tests the robustness of that finding (H4, H12, H13), looks
for alternative explanations in the data (H5, H6, H8, H9), sanity-checks
the pipeline (H7, H10), and tests the calibrated subset (H11).

Exit status of each hypothesis (ACCEPT / REJECT / AMBIGUOUS / NA) is
written to the log.  No hypothesis is permitted to rewrite the paper
without a matching note in FINDINGS.md.
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


# ── shared helpers ───────────────────────────────────────────────────────────
def log(tag: str, verdict: str, **kw):
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    header_needed = not RESULTS_TSV.exists()
    cols = ["ts", "tag", "verdict"] + list(kw.keys())
    row = [ts, tag, verdict] + [
        f"{v:.4g}" if isinstance(v, float) else str(v) for v in kw.values()
    ]
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("\t".join(cols) + "\n")
        f.write("\t".join(row) + "\n")
    banner = {"ACCEPT": "[+]", "REJECT": "[-]", "AMBIGUOUS": "[?]",
              "NA": "[.]"}.get(verdict, "[ ]")
    extras = "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in kw.items())
    print(f"{banner} {tag}  verdict={verdict}  {extras}")


def partial_corr(x, y, z):
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in
                     ((x, y), (x, z), (y, z)))
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float("nan")


def load_gap():
    df = pd.read_csv(TABLES / "ixi_gap_correlation.csv",
                     on_bad_lines="skip")
    df = df[pd.to_numeric(df["ixi_num"], errors="coerce").notna()].copy()
    for c in ("true_age", "face_age_gap", "brain_age_gap",
              "predicted_age", "brain_predicted_age"):
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df


def extract_key(s):
    m = re.search(r"(ses-\d+_run-\d+)", s)
    return m.group(1) if m else s


def load_simon():
    brain = pd.read_csv(TABLES / "simon_brainage_synthba.csv")
    brain["key"] = brain["scan_key"].apply(
        lambda x: extract_key(x.split("|")[-1]))
    mv = pd.read_csv(TABLES / "simon_faceage_multiview_raw.csv")
    mv["key"] = mv["subject_id"].apply(extract_key)
    morph = pd.read_csv(TABLES / "simon_faceage_morphometrics.csv")
    morph["key"] = morph["subject_id"].apply(extract_key)
    return brain, mv, morph


def ixi_site(subject_id: str) -> str:
    m = re.search(r"-(Guys|HH|IOP)-", subject_id)
    return m.group(1) if m else "unknown"


# ═════════════════════════════════════════════════════════════════════════════
# H4 — non-linear bias correction still nulls shared-aging
# ═════════════════════════════════════════════════════════════════════════════
def h4_nonlinear_bias():
    """If the age-bias is non-linear, maybe linear correction over-corrects
    and hides a real residual signal.  Try polynomial orders 2 and 3."""
    gap = load_gap()
    age = gap["true_age"].values
    fg  = gap["face_age_gap"].values
    bg  = gap["brain_age_gap"].values

    out = {}
    for deg in (1, 2, 3):
        p_face  = np.polyfit(age, fg, deg)
        p_brain = np.polyfit(age, bg, deg)
        r_face  = fg - np.polyval(p_face,  age)
        r_brain = bg - np.polyval(p_brain, age)
        r, p = stats.pearsonr(r_face, r_brain)
        out[f"r_deg{deg}"] = float(r)
        out[f"p_deg{deg}"] = float(p)

    max_abs = max(abs(out[f"r_deg{d}"]) for d in (1, 2, 3))
    if max_abs >= 0.20:
        v = "REJECT"   # some non-linear fit reveals residual signal
    elif max_abs <= 0.15:
        v = "ACCEPT"   # null robust to non-linear bias correction
    else:
        v = "AMBIGUOUS"
    log("H4_nonlinear_bias", v, max_abs_r_any_deg=max_abs, **out)


# ═════════════════════════════════════════════════════════════════════════════
# H5 — IXI brain-age bias is site-dependent
# ═════════════════════════════════════════════════════════════════════════════
def h5_ixi_site_effect():
    """Hypothesis: SynthBA bias differs across IXI sites (Guys/HH/IOP).
    If F-test p < 0.05 then site effect exists and our 'OOD' narrative is
    compound with site confound."""
    b = pd.read_csv(TABLES / "ixi_brainage_synthba.csv")
    mv = pd.read_csv(TABLES / "ixi_faceage_multiview_raw.csv")
    # brain table uses ixi_num; map via multiview using subject_id
    b["subject_id"] = b["subject_id"].astype(str)
    # brain CSV has numeric subject_id only — derive site from multiview
    # by matching to ixi_faceage_splits.csv if available
    splits = pd.read_csv(TABLES / "ixi_faceage_splits.csv")
    splits["ixi_num"] = splits["subject_id"].str.extract(r"IXI(\d+)-").astype(int)
    splits["site"] = splits["subject_id"].apply(ixi_site)

    merged = b.merge(splits[["ixi_num", "site"]].drop_duplicates(),
                      left_on="subject_id", right_on="ixi_num", how="left")
    merged = merged.dropna(subset=["site", "brain_age_gap"])
    by_site = {s: g["brain_age_gap"].values
                for s, g in merged.groupby("site") if len(g) > 5}
    if len(by_site) < 2:
        log("H5_ixi_site_effect", "NA", reason="insufficient_sites",
            n_sites=len(by_site))
        return
    F, p = stats.f_oneway(*by_site.values())
    means = {s: float(np.mean(v)) for s, v in by_site.items()}
    counts = {s: len(v) for s, v in by_site.items()}
    v = "ACCEPT" if p < 0.05 else "REJECT"
    log("H5_ixi_site_effect", v, F=float(F), p=float(p),
        means=str(means), counts=str(counts))


# ═════════════════════════════════════════════════════════════════════════════
# H6 — SIMON SynthBA predictions explained by nothing but noise
# ═════════════════════════════════════════════════════════════════════════════
def h6_simon_explanatory_power():
    """How much of the tiny SynthBA slope on SIMON is just drift in the
    subject's biology vs scanner or session effect?  Simple variance
    decomposition: compare variance by session vs by residual."""
    brain, _, _ = load_simon()
    b = brain.dropna(subset=["chron_age", "predicted_age"]).copy()
    by_sess = b.groupby("session_id")["predicted_age"].agg(["mean", "std", "count"])
    within_ss = by_sess["std"].replace([np.inf, np.nan], 0)
    between_ss = by_sess["mean"].std()
    within_mean = float(within_ss.mean())
    total_sd = float(b["predicted_age"].std())

    # ICC-like: between-session / total
    var_between = between_ss ** 2
    var_total = total_sd ** 2
    icc = float(var_between / var_total) if var_total > 0 else float("nan")

    # Slope of predicted vs chron_age (sanity check vs overnight_analysis.py)
    slope, icept, r, p, se = stats.linregress(b["chron_age"], b["predicted_age"])
    r2 = float(r ** 2)

    v = "ACCEPT" if r2 < 0.05 else "REJECT"
    # ACCEPT = "prediction is essentially noise + bias (R² < 5%)"
    log("H6_simon_pred_noise", v, r2_chron=r2, icc_between_session=icc,
        within_session_sd_mean=within_mean, total_sd=total_sd)


# ═════════════════════════════════════════════════════════════════════════════
# H7 — SIMON reproducibility coefficient of variation
# ═════════════════════════════════════════════════════════════════════════════
def h7_reproducibility_cv():
    """Coefficient of variation (SD / |mean|) for each method across
    all SIMON scans.  Lower CV = more reproducible."""
    brain, mv, morph = load_simon()
    out = {}
    for label, df, col in [
        ("SynthBA", brain.dropna(subset=["predicted_age"]), "predicted_age"),
        ("FaceAge_mv", mv.dropna(subset=["predicted_age"]), "predicted_age"),
        ("FaceMorph", morph.dropna(subset=["predicted_age"]), "predicted_age"),
    ]:
        m = float(df[col].mean())
        s = float(df[col].std())
        cv = s / abs(m) if m else float("nan")
        out[f"{label}_mean"] = m
        out[f"{label}_sd"] = s
        out[f"{label}_cv"] = cv
    v = ("ACCEPT" if out["SynthBA_cv"] < out["FaceAge_mv_cv"] / 2
         and out["SynthBA_cv"] < out["FaceMorph_cv"] / 2
         else "REJECT")
    log("H7_simon_cv", v, **out)


# ═════════════════════════════════════════════════════════════════════════════
# H8 — IXI age-stratified sign concordance of the two gaps
# ═════════════════════════════════════════════════════════════════════════════
def h8_sign_concordance():
    """If the raw r=0.31 is pure age-bias (both negative slopes on age),
    then gap signs concord with chron_age:
    young subjects → both gaps positive, old subjects → both gaps negative,
    middle → mixed.  Test sign concordance per age band."""
    g = load_gap()
    bands = [(20, 35), (35, 50), (50, 86)]
    rows = {}
    for lo, hi in bands:
        sub = g[(g["true_age"] >= lo) & (g["true_age"] < hi)]
        if len(sub) < 8:
            continue
        conc = np.mean(np.sign(sub["face_age_gap"]) ==
                        np.sign(sub["brain_age_gap"]))
        rows[f"band_{lo}_{hi}_conc"] = float(conc)
        rows[f"band_{lo}_{hi}_n"] = len(sub)
    # accept if youngest band has highest concordance (age-bias prediction)
    concs = {k: v for k, v in rows.items() if k.endswith("_conc")}
    v = ("ACCEPT" if concs and max(concs, key=concs.get) ==
         f"band_{bands[0][0]}_{bands[0][1]}_conc"
         else "REJECT")
    log("H8_sign_concordance", v, **rows)


# ═════════════════════════════════════════════════════════════════════════════
# H9 — SIMON within-run vs within-session vs between-session variance
# ═════════════════════════════════════════════════════════════════════════════
def h9_simon_variance_decomp():
    """Three variance sources on SIMON SynthBA predictions:
    within run (noise) < within session < between session.
    If all three are similar, scanner dominates; if within-run is large,
    model is stochastic; if between-session dominates, time+scanner matter."""
    brain, _, _ = load_simon()
    b = brain.dropna(subset=["predicted_age"]).copy()
    # sessions with multiple runs
    multi = b.groupby("session_id").filter(lambda g: len(g) > 1)
    if len(multi) == 0:
        log("H9_variance_decomp", "NA", reason="no_multi_run_session")
        return
    within_run_sd = float(multi.groupby("session_id")["predicted_age"]
                          .std().mean())
    all_sess_means = b.groupby("session_id")["predicted_age"].mean()
    between_session_sd = float(all_sess_means.std())
    total_sd = float(b["predicted_age"].std())

    # ratio of between-session to within-session (ICC-style)
    ratio = (between_session_sd / within_run_sd
             if within_run_sd > 0 else float("nan"))
    v = "ACCEPT" if ratio > 2 else "REJECT"   # between > within = scanner-dominated
    log("H9_variance_decomp", v,
        within_run_sd=within_run_sd,
        between_session_sd=between_session_sd,
        total_sd=total_sd, ratio_between_within=ratio,
        n_multirun_sessions=int((b.groupby("session_id").size() > 1).sum()))


# ═════════════════════════════════════════════════════════════════════════════
# H10 — UTK sanity check: FaceAge MAE on real photos
# ═════════════════════════════════════════════════════════════════════════════
def h10_utk_sanity():
    """UTK is the FaceAge training-adjacent set.  MAE here must be small
    (single-digit years) or the pipeline is broken."""
    try:
        u = pd.read_csv(TABLES / "utk_faceage_qa_ext.csv")
    except FileNotFoundError:
        log("H10_utk_sanity", "NA", reason="no_ext_file")
        return
    u = u.dropna(subset=["faceage", "age"])
    mae = float((u["faceage"] - u["age"]).abs().mean())
    bias = float((u["faceage"] - u["age"]).mean())
    r, p = stats.pearsonr(u["age"], u["faceage"])
    v = "ACCEPT" if mae < 10 else "REJECT"
    log("H10_utk_sanity", v, N=len(u), mae=mae, bias=bias,
        pearson_r=float(r), p=float(p))


# ═════════════════════════════════════════════════════════════════════════════
# H11 — calibrated FaceAge gap vs brain gap — does calibration recover signal?
# ═════════════════════════════════════════════════════════════════════════════
def h11_calibrated_gap():
    """Paper's linear calibration: ŷ = 1.010 ŷ_face - 17.049 fit on 20,
    applied to 80.  If calibration removes age-bias, the calibrated gap
    may be cleaner.  Test whether partial r(cal_gap, brain_gap | age)
    rises above 0.15."""
    splits = pd.read_csv(TABLES / "ixi_faceage_splits.csv")
    splits["ixi_num"] = splits["subject_id"].str.extract(r"IXI(\d+)").astype(int)
    test = splits[splits["split"] == "test"].copy()

    # brain data keyed by ixi_num
    brain = pd.read_csv(TABLES / "ixi_brainage_synthba.csv")
    brain["ixi_num"] = brain["subject_id"]

    m = test.merge(brain[["ixi_num", "brain_age_gap", "chron_age"]],
                   on="ixi_num", how="inner")
    if len(m) < 10:
        log("H11_calibrated_gap", "NA", reason="insufficient_overlap",
            n=len(m))
        return
    # apply paper calibration
    m["face_cal"] = 1.010 * m["predicted_age"] - 17.049
    # use chron_age from splits (true_age); fall back to brain chron_age
    m["true_age"] = m["true_age"].fillna(m["chron_age"])
    m["face_cal_gap"] = m["face_cal"] - m["true_age"]
    m = m.dropna(subset=["face_cal_gap", "brain_age_gap", "true_age"])

    r_raw, p_raw = stats.pearsonr(m["face_cal_gap"], m["brain_age_gap"])
    pc = partial_corr(m["face_cal_gap"].values,
                      m["brain_age_gap"].values,
                      m["true_age"].values)
    v = ("ACCEPT" if abs(pc) >= 0.15 else "REJECT"
         if abs(pc) <= 0.10 else "AMBIGUOUS")
    log("H11_calibrated_gap", v, N=len(m), r_raw=float(r_raw),
        p_raw=float(p_raw), partial_r=pc)


# ═════════════════════════════════════════════════════════════════════════════
# H12 — Spearman-rank partial (robust to outliers, distribution)
# ═════════════════════════════════════════════════════════════════════════════
def h12_rank_partial():
    """Use rank-partialling (convert to ranks, then compute partial Pearson
    on ranks = Spearman partial).  Robust to non-linearity and outliers."""
    g = load_gap()
    fr = stats.rankdata(g["face_age_gap"])
    br = stats.rankdata(g["brain_age_gap"])
    ar = stats.rankdata(g["true_age"])
    rho_raw, p_raw = stats.spearmanr(g["face_age_gap"],
                                       g["brain_age_gap"])
    rho_partial = partial_corr(fr, br, ar)
    v = ("ACCEPT" if abs(rho_partial) >= 0.15 else "REJECT"
         if abs(rho_partial) <= 0.10 else "AMBIGUOUS")
    # ACCEPT = rank-partial reveals signal the linear test missed
    # REJECT = still ~0 after rank partial → confirms the null
    v = "REJECT" if abs(rho_partial) < 0.10 else "ACCEPT"
    log("H12_rank_partial", v, rho_raw=float(rho_raw),
        partial_rho=rho_partial, p_raw=float(p_raw))


# ═════════════════════════════════════════════════════════════════════════════
# H13 — permutation test: is the partial r = 0 null robust?
# ═════════════════════════════════════════════════════════════════════════════
def h13_permutation_null():
    """Permute face_gap while holding brain_gap and true_age.  Compare
    observed partial r to null distribution."""
    g = load_gap()
    rng = np.random.default_rng(0)
    obs = partial_corr(g["face_age_gap"].values,
                        g["brain_age_gap"].values,
                        g["true_age"].values)

    n_perm = 10_000
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(g["face_age_gap"].values)
        nulls[i] = partial_corr(perm, g["brain_age_gap"].values,
                                 g["true_age"].values)
    p = float(((np.abs(nulls) >= abs(obs)).sum() + 1) / (n_perm + 1))
    v = "ACCEPT" if p > 0.05 else "REJECT"
    # ACCEPT = observed partial r is indistinguishable from null (confirms confound)
    log("H13_permutation_null", v, obs_partial_r=float(obs),
        null_mean=float(nulls.mean()), null_sd=float(nulls.std()),
        perm_p=p, n_perm=n_perm)


# ═════════════════════════════════════════════════════════════════════════════
HYPOTHESES = [h4_nonlinear_bias, h5_ixi_site_effect, h6_simon_explanatory_power,
              h7_reproducibility_cv, h8_sign_concordance,
              h9_simon_variance_decomp, h10_utk_sanity,
              h11_calibrated_gap, h12_rank_partial, h13_permutation_null]

if __name__ == "__main__":
    print("=" * 76)
    print(f"AUTORESEARCH RATCHET — {len(HYPOTHESES)} hypotheses")
    print("=" * 76)
    for fn in HYPOTHESES:
        try:
            fn()
        except Exception as e:
            log(f"{fn.__name__}", "NA", error=repr(e))
            print(f"    !! {fn.__name__} raised: {e}")
    print("\nDone. See RESULTS.tsv")
