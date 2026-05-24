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
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'external': '#0072B2',
    'standard': '#D55E00',
    'anchor':   '#009E73',
}
MARKERS = {
    'external': 'o',
    'standard': 's',
    'anchor':   '^',
}
LINESTYLES = {
    'external': '-',
    'standard': '--',
    'anchor':   '-.',
}
LABELS = {
    'external': 'External',
    'standard': 'Standard',
    'anchor':   'Anchor',
}

data_path = '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts/plan_011_semi_synthetic_validation/data/dose_response_summary_779a18.csv'
df = pd.read_csv(data_path)
df = df[df['cohort_type'] == 'temporal'].copy()

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(5, 6.67),  # 3:4 aspect
    gridspec_kw={'height_ratios': [7, 3]},
    sharex=True,
)
fig.subplots_adjust(hspace=0.08)

# --- Regime band: p < 0.3 "Below practical detection" ---
for ax in [ax_top, ax_bot]:
    ax.axvspan(-0.03, 0.3, alpha=0.08, color='#888888', zorder=0)
ax_top.text(0.13, 0.97, 'Below practical\ndetection', transform=ax_top.get_xaxis_transform(),
            va='top', ha='center', fontsize=8, color='#999999', fontstyle='italic')

# --- Top panel: TPR vs p ---
for variant in ['external', 'standard', 'anchor']:
    sub = df[df['mh_variant'] == variant].sort_values('p')
    ax_top.errorbar(
        sub['p'], sub['tpr_mean'], yerr=sub['tpr_std'],
        color=COLORS[variant], marker=MARKERS[variant], linestyle=LINESTYLES[variant],
        markersize=6, capsize=3, capthick=1.2, zorder=3,
    )

# Baseline dashed line for external at p=0
baseline_tpr = df[(df['mh_variant'] == 'external') & (df['p'] == 0.0)]['tpr_mean'].values[0]
ax_top.axhline(y=baseline_tpr, color=COLORS['external'], linestyle=':', linewidth=0.9, alpha=0.4, zorder=1)

# Direct labels at rightmost point (p=1.0)
for variant in ['external', 'standard', 'anchor']:
    sub = df[(df['mh_variant'] == variant) & (df['p'] == 1.0)]
    y_val = sub['tpr_mean'].values[0]
    offsets = {'external': (6, -2), 'standard': (6, -10), 'anchor': (6, 6)}
    ox, oy = offsets[variant]
    t = ax_top.annotate(
        LABELS[variant], (1.0, y_val), xytext=(ox, oy), textcoords='offset points',
        fontsize=9, color=COLORS[variant], fontweight='bold',
    )
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

# Annotate ΔTPR at p=0.5 for external
ext_05 = df[(df['mh_variant'] == 'external') & (df['p'] == 0.5)]['tpr_mean'].values[0]
delta_tpr = ext_05 - baseline_tpr
ax_top.annotate(
    f'$\\Delta$TPR = +{delta_tpr:.3f}',
    xy=(0.5, ext_05), xytext=(-60, 15), textcoords='offset points',
    fontsize=9, color=COLORS['external'],
    arrowprops=dict(arrowstyle='->', color=COLORS['external'], lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['external'], alpha=0.9),
)

ax_top.set_ylabel('True Positive Rate (TPR)')
ax_top.set_ylim(0.15, 1.08)

# --- Bottom panel: FPR vs p ---
for variant in ['external', 'standard', 'anchor']:
    sub = df[df['mh_variant'] == variant].sort_values('p')
    ax_bot.errorbar(
        sub['p'], sub['fpr_mean'], yerr=sub['fpr_std'],
        color=COLORS[variant], marker=MARKERS[variant], linestyle=LINESTYLES[variant],
        markersize=6, capsize=3, capthick=1.2, zorder=3,
    )

# Direct labels for FPR panel
for variant in ['external', 'anchor']:
    sub = df[(df['mh_variant'] == variant) & (df['p'] == 1.0)]
    y_val = sub['fpr_mean'].values[0]
    offsets = {'external': (6, 4), 'anchor': (6, -10)}
    ox, oy = offsets[variant]
    t = ax_bot.annotate(
        LABELS[variant], (1.0, y_val), xytext=(ox, oy), textcoords='offset points',
        fontsize=9, color=COLORS[variant], fontweight='bold',
    )
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

# Annotate FPR inflation for standard at p=1.0
std_fpr_10 = df[(df['mh_variant'] == 'standard') & (df['p'] == 1.0)]['fpr_mean'].values[0]
ax_bot.annotate(
    'FPR inflation',
    xy=(1.0, std_fpr_10), xytext=(-65, 12), textcoords='offset points',
    fontsize=9, color=COLORS['standard'],
    arrowprops=dict(arrowstyle='->', color=COLORS['standard'], lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['standard'], alpha=0.9),
)

ax_bot.set_ylabel('False Positive Rate (FPR)')
ax_bot.set_xlabel('Exploitation Rate ($p$)')
ax_bot.set_ylim(0.2, 0.72)
ax_bot.set_xticks([0, 0.3, 0.5, 0.7, 1.0])

out_base = '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/figures/paper/fig_2_dose_response'
fig.savefig(f'{out_base}.pdf')
fig.savefig(f'{out_base}.png')
plt.close()
print(f'Saved: {out_base}.pdf')
print(f'Saved: {out_base}.png')
