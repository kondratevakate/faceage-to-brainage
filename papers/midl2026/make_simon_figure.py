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
out = "../figures/simon_chron_vs_predicted.pdf"
fig.savefig(out, dpi=300, bbox_inches='tight')
out_png = "../figures/simon_chron_vs_predicted.png"
fig.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
print(f"Saved: {out_png}")
