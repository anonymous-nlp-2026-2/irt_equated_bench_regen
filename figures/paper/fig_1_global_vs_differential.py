"""
fig_1_global_vs_differential — Global vs Differential Contamination concept diagram.

Two-panel ICC illustration: left panel shows overlapping curves (global contamination,
no DIF), right panel shows diverging curves (differential contamination, DIF detected).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.8,
})

COL_REF = '#0072B2'
COL_FOCAL = '#D55E00'

OUTDIR = Path(__file__).parent
STEM = 'fig_1_global_vs_differential'


def icc(theta, a=1.5, b=0.0):
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


theta = np.linspace(-3, 3, 300)

# --- Left panel curves (identical) ---
y_ref_L = icc(theta, a=1.5, b=0.0)
y_foc_L = icc(theta, a=1.5, b=0.0)

# --- Right panel curves (DIF: focal shifted left) ---
y_ref_R = icc(theta, a=1.5, b=0.0)
y_foc_R = icc(theta, a=1.5, b=-1.2)

# --- Figure ---
fig, (ax_L, ax_R) = plt.subplots(
    1, 2, figsize=(11, 4.5), sharey=True,
    gridspec_kw={'wspace': 0.25}
)

bbox_style = dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', edgecolor='#CCCCCC',
                  alpha=0.92)

# ====== LEFT PANEL: Global Contamination ======
ax_L.plot(theta, y_ref_L, color=COL_REF, linestyle='-', linewidth=2.2,
          label='Reference cohort (2023)')
ax_L.plot(theta, y_foc_L, color=COL_FOCAL, linestyle='--', linewidth=2.2,
          label='Focal cohort (2024)')

ax_L.set_title('Global Contamination', fontweight='bold', pad=10)
ax_L.set_xlabel(r'Ability ($\theta$)')
ax_L.set_ylabel('P(correct)')
ax_L.set_xlim(-3, 3)
ax_L.set_ylim(0, 1.0)

ax_L.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9,
            edgecolor='#CCCCCC')

# Annotations — left panel
ax_L.text(0.04, 0.92, r'$\Delta_\mathrm{MH} \approx 0$  (no DIF)',
          transform=ax_L.transAxes, fontsize=9.5, color='#444444',
          bbox=bbox_style, va='top')

ax_L.text(0.04, 0.78,
          'Li et al.: ✓ contaminated\nDIF: ✗ no group difference',
          transform=ax_L.transAxes, fontsize=8.5, color='#555555',
          bbox=bbox_style, va='top', linespacing=1.5)

# Subtle note about web overlap
ax_L.annotate('Item on Common Crawl since 2020\n→ both cohorts trained on it',
              xy=(0.0, icc(0.0)), xytext=(-2.4, 0.78),
              fontsize=8, color='#666666', fontstyle='italic',
              arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8),
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='#DDDDDD', alpha=0.85))

# ====== RIGHT PANEL: Differential Contamination ======
ax_R.plot(theta, y_ref_R, color=COL_REF, linestyle='-', linewidth=2.2)
ax_R.plot(theta, y_foc_R, color=COL_FOCAL, linestyle='--', linewidth=2.2)

# Shaded DIF area between curves
ax_R.fill_between(theta, y_ref_R, y_foc_R, where=(y_foc_R > y_ref_R),
                  color=COL_FOCAL, alpha=0.18, label='_nolegend_')
ax_R.fill_between(theta, y_ref_R, y_foc_R, where=(y_foc_R <= y_ref_R),
                  color=COL_FOCAL, alpha=0.18, label='_nolegend_')

# DIF magnitude label inside shaded area
mid_theta = -0.6
mid_y = (icc(mid_theta, b=0.0) + icc(mid_theta, b=-1.2)) / 2
ax_R.annotate('DIF\nmagnitude', xy=(mid_theta, mid_y),
              fontsize=8.5, color=COL_FOCAL, fontweight='bold',
              ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COL_FOCAL, alpha=0.85, linewidth=0.8))

ax_R.set_title('Differential Contamination', fontweight='bold', pad=10)
ax_R.set_xlabel(r'Ability ($\theta$)')
ax_R.set_xlim(-3, 3)

# Annotations — right panel
ax_R.text(0.04, 0.92, r'$|\Delta_\mathrm{MH}| \geq 1.5$  (ETS C)',
          transform=ax_R.transAxes, fontsize=9.5, color='#444444',
          bbox=bbox_style, va='top')

ax_R.text(0.04, 0.78,
          'Li et al.: may or may not flag\nDIF: ✓ detected (ETS C)',
          transform=ax_R.transAxes, fontsize=8.5, color='#555555',
          bbox=bbox_style, va='top', linespacing=1.5)

# Arrow noting focal curve is shifted
ax_R.annotate('Focal curve shifted left\n(easier for focal group)',
              xy=(-1.2, 0.5), xytext=(-2.6, 0.28),
              fontsize=8, color=COL_FOCAL, fontstyle='italic',
              arrowprops=dict(arrowstyle='->', color=COL_FOCAL, lw=0.9),
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='#DDDDDD', alpha=0.85))

# ====== BOTTOM ANNOTATION spanning both panels ======
fig.text(0.5, -0.02,
         'Enrichment ≈ 1.0: most MMLU items are globally contaminated (left), '
         'not differentially (right)',
         ha='center', va='top', fontsize=10, color='#333333',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAFAFA',
                   edgecolor='#BBBBBB', alpha=0.95))

# ====== SAVE ======
for ext in ('pdf', 'png'):
    outpath = OUTDIR / f'{STEM}.{ext}'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Saved: {outpath}')

plt.close(fig)
