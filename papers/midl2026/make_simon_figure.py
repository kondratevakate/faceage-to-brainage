"""
Generate Figure 1 for the MIDL 2026 short paper:
Chronological age vs predicted age on the SIMON dataset
for three methods: SynthBA (brain), Face Morphometrics, FaceAge Multiview.
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── load data ────────────────────────────────────────────────────────────────

def extract_key(s):
    m = re.search(r'(ses-\d+_run-\d+)', s)
    return m.group(1) if m else s

brain = pd.read_csv("../tables/simon_brainage_synthba.csv")
brain['key'] = brain['scan_key'].apply(lambda x: extract_key(x.split('|')[-1]))

face_mv = pd.read_csv("../tables/simon_faceage_multiview_raw.csv")
face_mv['key'] = face_mv['subject_id'].apply(extract_key)

face_morph = pd.read_csv("../tables/simon_faceage_morphometrics.csv")
face_morph['key'] = face_morph['subject_id'].apply(extract_key)

merged = (
    brain[['key', 'chron_age', 'predicted_age']]
    .rename(columns={'predicted_age': 'brain_pred'})
    .merge(face_morph[['key', 'predicted_age']].rename(columns={'predicted_age': 'morph_pred'}),
           on='key', how='outer')
    .merge(face_mv[['key', 'predicted_age']].dropna().rename(columns={'predicted_age': 'mv_pred'}),
           on='key', how='outer')
)

# ── palette (colorblind-safe) ─────────────────────────────────────────────────
C_BRAIN = "#2166AC"   # blue
C_MORPH = "#4DAC26"   # green
C_MV    = "#D6604D"   # red-orange

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

age_range = np.array([28, 48])
ax.plot(age_range, age_range, color='black', lw=1.2, ls='--', label='Perfect prediction', zorder=1)

kw = dict(alpha=0.55, s=28, zorder=2, linewidths=0.3, edgecolors='white')

b = merged.dropna(subset=['brain_pred', 'chron_age'])
ax.scatter(b['chron_age'], b['brain_pred'],  color=C_BRAIN, marker='o', **kw)

m = merged.dropna(subset=['morph_pred', 'chron_age'])
ax.scatter(m['chron_age'], m['morph_pred'],  color=C_MORPH, marker='s', **kw)

v = merged.dropna(subset=['mv_pred', 'chron_age'])
ax.scatter(v['chron_age'], v['mv_pred'],     color=C_MV,    marker='^', **kw)

# mean prediction lines (horizontal)
for pred_col, color in [('brain_pred', C_BRAIN), ('morph_pred', C_MORPH), ('mv_pred', C_MV)]:
    sub = merged.dropna(subset=[pred_col])
    mean_val = sub[pred_col].mean()
    ax.axhline(mean_val, color=color, lw=0.8, ls=':', alpha=0.7)

# ── legend ───────────────────────────────────────────────────────────────────
def stats(df, col):
    sub = df.dropna(subset=[col, 'chron_age'])
    mae  = (sub[col] - sub['chron_age']).abs().mean()
    bias = (sub[col] - sub['chron_age']).mean()
    sd   = sub[col].std()
    return len(sub), mae, bias, sd

nb, mae_b, bias_b, sd_b = stats(merged, 'brain_pred')
nm, mae_m, bias_m, sd_m = stats(merged, 'morph_pred')
nv, mae_v, bias_v, sd_v = stats(merged, 'mv_pred')

legend_handles = [
    mlines.Line2D([], [], color='black', ls='--', lw=1.2, label='Perfect prediction'),
    plt.scatter([], [], color=C_BRAIN, marker='o', s=40, alpha=0.7,
                label=f'SynthBA (brain)  MAE={mae_b:.1f} yr, SD={sd_b:.2f} yr, bias={bias_b:+.1f} yr'),
    plt.scatter([], [], color=C_MORPH, marker='s', s=40, alpha=0.7,
                label=f'Face morphometrics  MAE={mae_m:.1f} yr, SD={sd_m:.2f} yr, bias={bias_m:+.1f} yr'),
    plt.scatter([], [], color=C_MV,    marker='^', s=40, alpha=0.7,
                label=f'FaceAge renders  MAE={mae_v:.1f} yr, SD={sd_v:.2f} yr, bias={bias_v:+.1f} yr'),
]

ax.legend(handles=legend_handles, fontsize=7.5, loc='upper left',
          framealpha=0.9, edgecolor='#cccccc')

ax.set_xlabel('Chronological age (years)', fontsize=11)
ax.set_ylabel('Predicted age (years)', fontsize=11)
ax.set_title('SIMON: one subject, 36 scanners, 99 scans', fontsize=11)
ax.set_xlim(28, 48)
ax.set_ylim(16, 72)
ax.tick_params(labelsize=9)
ax.grid(True, lw=0.4, alpha=0.4)

fig.tight_layout()
out = "simon_chron_vs_predicted.pdf"
fig.savefig(out, dpi=300, bbox_inches='tight')
out_png = "simon_chron_vs_predicted.png"
fig.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
print(f"Saved: {out_png}")

# ── Figure 2: SynthBA training distribution vs SIMON predictions ─────────────
#
# Reconstruction of the SynthBA training-age distribution as a weighted
# mixture of dataset-level Gaussians.  Per-dataset N is taken directly
# from Table I of Puglisi et al. 2024 (arXiv:2406.00365).  The Gaussian
# parameters per dataset are approximate, derived from each cohort's
# publicly reported age statistics:
#
#   ADNI  — Alzheimer's Disease Neuroimaging Initiative, elderly cohort
#   AIBL  — Australian Imaging, Biomarkers and Lifestyle, elderly cohort
#   HCP   — Human Connectome Project Young Adults (HCP-YA), 22–35 yr
#   IXI   — Information eXtraction from Images, broad healthy adults
#   CoRR  — Consortium for Reliability and Reproducibility, broad lifespan
#
# The sum reproduces the bimodal shape with peaks at ~25 and ~74 yr that
# Puglisi et al. 2024 report textually.

SYNTHBA_TRAIN = {
    # name      (mu,  sigma,  n,     color,     ls)
    'HCP':      (28.5, 3.7,  1105, '#9B59B6', '-'),   # young adults, tight
    'CoRR':     (25.0, 15.0, 1461, '#3498DB', '-'),   # broad lifespan, peak young
    'IXI':      (48.0, 16.0,  563, '#1ABC9C', '-'),   # broad healthy adults
    'ADNI':     (75.0,  7.0,  784, '#E67E22', '-'),   # AD-research, elderly
    'AIBL':     (73.0,  7.0,  192, '#E74C3C', '--'),  # overlaps ADNI -> dashed
}
N_TOTAL = sum(v[2] for v in SYNTHBA_TRAIN.values())
assert N_TOTAL == 4105, f"per-dataset N must sum to 4105 (got {N_TOTAL})"

def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

age_grid = np.linspace(0, 100, 1000)
train_density = np.zeros_like(age_grid)
per_dataset_curves = {}
for name, (mu, sigma, n, color, ls) in SYNTHBA_TRAIN.items():
    g = (n / N_TOTAL) * gauss(age_grid, mu, sigma)
    per_dataset_curves[name] = (g, color, n, ls)
    train_density += g

# Sanity check: find peaks of the reconstructed mixture.
from scipy.signal import find_peaks
peak_idx, _ = find_peaks(train_density, distance=20)
peak_ages = age_grid[peak_idx]
print(f"reconstructed peaks at: {[f'{a:.1f}' for a in peak_ages]} yr (paper reports 25 & 74)")

fig2, (ax2_top, ax2_bot) = plt.subplots(
    2, 1, figsize=(6.2, 4.4), sharex=True,
    gridspec_kw={'height_ratios': [2.2, 1.0], 'hspace': 0.08},
)

# ── Top panel: training distribution ────────────────────────────────────────
# SIMON age band first so it sits behind everything else.
ax2_top.axvspan(29.6, 46.4, color=C_BRAIN, alpha=0.10,
                label='SIMON chronological age (29.6–46.4 yr)')

# Per-dataset Gaussian components.
for name, (g, color, n, ls) in per_dataset_curves.items():
    ax2_top.plot(age_grid, g, color=color, lw=1.0, alpha=0.85, ls=ls,
                 label=f"{name}  (n={n})")

# Weighted sum.
ax2_top.fill_between(age_grid, train_density, color='#444444', alpha=0.18,
                     zorder=1)
ax2_top.plot(age_grid, train_density, color='#222222', lw=1.5,
             label=f"SynthBA training mixture  "
                   f"(n={N_TOTAL}, peaks {peak_ages[0]:.0f} & {peak_ages[-1]:.0f} yr)")

ax2_top.set_ylabel('Training density', fontsize=10)
ax2_top.set_ylim(bottom=0)
ax2_top.tick_params(labelsize=9)
ax2_top.grid(True, lw=0.4, alpha=0.4)
ax2_top.legend(fontsize=6.8, loc='upper right', framealpha=0.92,
               edgecolor='#cccccc', ncol=1)

# ── Bottom panel: predictions ───────────────────────────────────────────────
ax2_bot.axvspan(29.6, 46.4, color=C_BRAIN, alpha=0.10)
brain_pred = merged['brain_pred'].dropna().values
ax2_bot.hist(brain_pred, bins=np.arange(15, 75, 1.0), density=True,
             color=C_BRAIN, alpha=0.70, edgecolor='white', linewidth=0.4,
             label=f"SynthBA predictions on SIMON  "
                   f"(n={len(brain_pred)}, {brain_pred.mean():.1f}±{brain_pred.std():.2f} yr)")
# Annotate the gap between predictions and SIMON age band.
ax2_bot.annotate(
    '', xy=(brain_pred.mean(), 0.18), xytext=(36, 0.18),
    arrowprops=dict(arrowstyle='->', color='#444444', lw=1.0),
)
ax2_bot.text(31.5, 0.20, f"{36 - brain_pred.mean():.0f}-yr offset\nfrom SIMON mid-range",
             fontsize=7.5, color='#444444', ha='left', va='bottom')

ax2_bot.set_xlabel('Age (years)', fontsize=11)
ax2_bot.set_ylabel('Prediction density', fontsize=10)
ax2_bot.set_xlim(0, 100)
ax2_bot.set_ylim(0, 0.42)
ax2_bot.tick_params(labelsize=9)
ax2_bot.grid(True, lw=0.4, alpha=0.4)
ax2_bot.legend(fontsize=7.0, loc='upper right', framealpha=0.92,
               edgecolor='#cccccc')

fig2.tight_layout()
out_c = "synthba_training_vs_predictions.pdf"
fig2.savefig(out_c, dpi=300, bbox_inches='tight')
out_c_png = "synthba_training_vs_predictions.png"
fig2.savefig(out_c_png, dpi=200, bbox_inches='tight')
print(f"Saved: {out_c}")
print(f"Saved: {out_c_png}")
