"""Fig 4: Downstream impact — rank change scatter (full vs stable-only benchmark)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

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
    'lines.linewidth': 1.8,
})

# Color mapping (Okabe-Ito based)
FAMILY_COLORS = {
    'llama': '#0072B2',
    'mistral': '#D55E00',
    'phi': '#009E73',
    'qwen': '#CC79A7',
    'other': '#56B4E9',
}

def assign_color(family):
    f = str(family).lower()
    if 'llama' in f:
        return FAMILY_COLORS['llama']
    if 'mistral' in f or 'mixtral' in f:
        return FAMILY_COLORS['mistral']
    if 'phi' in f:
        return FAMILY_COLORS['phi']
    if 'qwen' in f:
        return FAMILY_COLORS['qwen']
    return FAMILY_COLORS['other']

def short_name(model_name):
    parts = model_name.split('/')
    return parts[-1] if len(parts) > 1 else model_name

# Load data
DATA_PATH = '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts/plan_017/data/rank_movers.csv'
df = pd.read_csv(DATA_PATH)
df = df[df['cohort'] == 'all'].copy()

# Split into background vs highlighted
threshold = 200
bg = df[df['abs_rank_change'] <= threshold]
fg = df[df['abs_rank_change'] > threshold]

fig, ax = plt.subplots(figsize=(7, 6.5))

# Diagonal reference line
lims = [0, df[['rank_full', 'rank_stable']].max().max() + 50]
ax.plot(lims, lims, '--', color='#888888', linewidth=0.8, zorder=1, alpha=0.6)

# Background points
ax.scatter(bg['rank_full'], bg['rank_stable'], s=3, alpha=0.15, color='#999999',
           edgecolors='none', zorder=2, rasterized=True)

# Highlighted points colored by family
fg_colors = fg['family'].apply(assign_color)
ax.scatter(fg['rank_full'], fg['rank_stable'], s=15, alpha=0.6, c=fg_colors,
           edgecolors='none', zorder=3)

# Annotate extreme movers: 3 positive + 2 negative, hand-tuned offsets
annotations = [
    # (model_name_substring, label, xytext_offset)
    # Positive movers (above diagonal)
    ('14B-Glacier-Stack',       '14B-Glacier-Stack\n(Δ = +851)',      (60, -10)),
    ('DeepMount00/phi_3',       'phi-3\n(Δ = +809)',                  (50, 60)),
    ('llama-3-dragonmaid',      'llama-3-dragonmaid-8B\n(Δ = +793)', (50, 30)),
    # Negative movers (below diagonal)
    ('Blurstral-7b',            'Blurstral-7b\n(Δ = −587)',           (40, -40)),
    ('flammen26-mistral',       'flammen26-mistral-7B\n(Δ = −542)',   (-80, -50)),
]

for substr, label, offset in annotations:
    match = df[df['model_name'].str.contains(substr, na=False)]
    if match.empty:
        continue
    row = match.iloc[0]
    x, y = row['rank_full'], row['rank_stable']
    ax.annotate(
        label, (x, y),
        xytext=offset, textcoords='offset points',
        fontsize=7, color='#333333',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8,
                        connectionstyle='arc3,rad=0.15'),
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
        zorder=5, linespacing=1.2,
    )

# Annotation box
stats_text = (
    r'$\rho$ = 0.996 (full vs stable)' '\n'
    r'DIF removal: $z$ = $-$14.9 (mean)' '\n'
    r'DIF items: 1.09$\times$ cross-cohort ratio'
)
props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cccccc', linewidth=0.6)
ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=6)

# Legend for highlighted families
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS['llama'],
           markersize=6, label='LLaMA'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS['mistral'],
           markersize=6, label='Mistral'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS['phi'],
           markersize=6, label='Phi'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS['qwen'],
           markersize=6, label='Qwen'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS['other'],
           markersize=6, label='Other'),
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.85,
          edgecolor='#cccccc', title='|Δrank| > 200', title_fontsize=9)

# Axis labels
ax.set_xlabel('Full-Benchmark Rank (12,508 items)')
ax.set_ylabel('Stable-Only Rank (8,590 non-DIF items)')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')

# Save
out_base = '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/figures/paper/fig_4_downstream_impact'
fig.savefig(f'{out_base}.pdf', format='pdf')
fig.savefig(f'{out_base}.png', format='png')
plt.close(fig)
print(f"Saved: {out_base}.pdf and .png")
