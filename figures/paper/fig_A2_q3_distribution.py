#!/usr/bin/env python3
"""fig_A2_q3_distribution — Q3 residual correlation per-subject distribution."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BLUE = '#0072B2'
ORANGE_RED = '#D55E00'
CROSS_SUBJECT_MEAN = 2.2
WITHIN_SUBJECT_MEAN = 11.3
HIGHLIGHT_THRESHOLD = 20.0

data_path = pathlib.Path(
    '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen'
    '/artifacts/plan_013_q3_local_independence/data/q3_summary.csv'
)
out_dir = pathlib.Path(
    '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen'
    '/figures/paper'
)

df = pd.read_csv(data_path)
df['label'] = df['subject'].str.replace('_', ' ').str.title()
df = df.sort_values('pct_above_020', ascending=True).reset_index(drop=True)

colors = [ORANGE_RED if v >= HIGHLIGHT_THRESHOLD else BLUE for v in df['pct_above_020']]

fig, ax = plt.subplots(figsize=(7, 14))

ax.barh(range(len(df)), df['pct_above_020'], color=colors, height=0.7, zorder=2)

ax.axvline(CROSS_SUBJECT_MEAN, color='grey', linestyle='--', linewidth=1.2, zorder=3)
ax.axvline(WITHIN_SUBJECT_MEAN, color=BLUE, linestyle='--', linewidth=1.2, zorder=3)

n = len(df)
ax.annotate(
    f'Cross-subject = {CROSS_SUBJECT_MEAN}%',
    xy=(CROSS_SUBJECT_MEAN, n - 0.5), xytext=(0, 8),
    textcoords='offset points', fontsize=7.5, color='#555555',
    va='bottom', ha='center', clip_on=False,
    arrowprops=dict(arrowstyle='-', color='#555555', lw=0.6),
)
ax.annotate(
    f'Within-subject mean = {WITHIN_SUBJECT_MEAN}%',
    xy=(WITHIN_SUBJECT_MEAN, n - 0.5), xytext=(0, 24),
    textcoords='offset points', fontsize=7.5, color=BLUE,
    va='bottom', ha='center', clip_on=False,
    arrowprops=dict(arrowstyle='-', color=BLUE, lw=0.6),
)

ax.text(
    0.97, 0.03,
    'Mann–Whitney $p = 2.3 \\times 10^{-9}$\nWithin / Cross ratio ≈ 5×',
    transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9),
)

ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['label'])
ax.set_xlabel('Item Pairs Exceeding |Q3| > 0.20 (%)')
ax.set_xlim(0, df['pct_above_020'].max() * 1.08)
ax.set_ylim(-0.6, len(df) - 0.4)

ax.grid(axis='x', alpha=0.25, zorder=1)

pdf_path = out_dir / 'fig_A2_q3_distribution.pdf'
png_path = out_dir / 'fig_A2_q3_distribution.png'
fig.savefig(pdf_path)
fig.savefig(png_path)
plt.close(fig)
print(f'Saved: {pdf_path}')
print(f'Saved: {png_path}')
