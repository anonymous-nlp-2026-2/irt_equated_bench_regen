import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

C_CROSS = '#0072B2'
C_WITHIN = '#D55E00'

csv_path = (
    '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen'
    '/artifacts/plan_016_within_subject_semi_synthetic/data'
    '/within_subject_dose_response_e6d86f.csv'
)
df = pd.read_csv(csv_path)

df_p0 = df[df['p'] == 0.0]
df_p5 = df[df['p'] == 0.5]

records = []
for subj in df['subject'].unique():
    for mt in ['cross', 'within']:
        row_p0 = df_p0[(df_p0['subject'] == subj) & (df_p0['matching_type'] == mt)].iloc[0]
        row_p5 = df_p5[(df_p5['subject'] == subj) & (df_p5['matching_type'] == mt)].iloc[0]
        records.append({
            'subject': subj,
            'matching_type': mt,
            'delta_tpr': row_p5['delta_tpr'],
            'delta_fpr': row_p5['fpr_mean'] - row_p0['fpr_mean'],
        })

data = pd.DataFrame(records)

cross_order = (
    data[data['matching_type'] == 'cross']
    .sort_values('delta_tpr', ascending=True)['subject']
    .tolist()
)

label_map = {
    'business_ethics': 'Business Ethics',
    'human_aging': 'Human Aging',
    'professional_medicine': 'Prof. Medicine',
    'formal_logic': 'Formal Logic',
    'high_school_statistics': 'HS Statistics',
}

y_pos = {subj: i for i, subj in enumerate(cross_order)}

fig, (ax_l, ax_r) = plt.subplots(
    1, 2, figsize=(7.5, 5), sharey=True,
    gridspec_kw={'wspace': 0.12}
)

# ── Left panel: ΔTPR ──
for subj in cross_order:
    i = y_pos[subj]
    cv = data[(data['subject'] == subj) & (data['matching_type'] == 'cross')]['delta_tpr'].values[0]
    wv = data[(data['subject'] == subj) & (data['matching_type'] == 'within')]['delta_tpr'].values[0]

    ax_l.plot([cv, wv], [i, i], color='#AAAAAA', linewidth=1.2, zorder=1)
    ax_l.scatter(cv, i, color=C_CROSS, s=65, zorder=3, edgecolors=C_CROSS, linewidths=1.2)
    ax_l.scatter(wv, i, facecolors='white', edgecolors=C_WITHIN, s=65, zorder=3, linewidths=1.5)

ax_l.axvline(x=0.3, color='#999999', linestyle='--', linewidth=0.9, zorder=0)
ax_l.text(0.3, len(cross_order) - 0.25, 'threshold', ha='center', va='bottom',
          fontsize=8.5, color='#777777')

ax_l.axvline(x=0, color='#DDDDDD', linestyle='-', linewidth=0.7, zorder=0)

ax_l.set_yticks(range(len(cross_order)))
ax_l.set_yticklabels([label_map[s] for s in cross_order])
ax_l.set_xlabel(r'$\Delta$TPR ($p = 0.5$)')
ax_l.set_ylim(-0.6, len(cross_order) - 0.4)

be_idx = y_pos['business_ethics']
ax_l.axhspan(be_idx - 0.4, be_idx + 0.4, color='#FFF3E0', alpha=0.6, zorder=0)

hs_idx = y_pos['high_school_statistics']
hs_cross = data[(data['subject'] == 'high_school_statistics') & (data['matching_type'] == 'cross')]['delta_tpr'].values[0]
hs_within = data[(data['subject'] == 'high_school_statistics') & (data['matching_type'] == 'within')]['delta_tpr'].values[0]

ax_l.annotate('Cross-subject', xy=(hs_cross, hs_idx),
              xytext=(hs_cross + 0.01, hs_idx + 0.32),
              fontsize=8.5, color=C_CROSS, va='bottom', ha='left')
ax_l.annotate('Within-subject', xy=(hs_within, hs_idx),
              xytext=(hs_within - 0.01, hs_idx + 0.32),
              fontsize=8.5, color=C_WITHIN, va='bottom', ha='right')

# ── Right panel: ΔFPR ──
for subj in cross_order:
    i = y_pos[subj]
    cv = data[(data['subject'] == subj) & (data['matching_type'] == 'cross')]['delta_fpr'].values[0]
    wv = data[(data['subject'] == subj) & (data['matching_type'] == 'within')]['delta_fpr'].values[0]

    ax_r.plot([cv, wv], [i, i], color='#AAAAAA', linewidth=1.2, zorder=1)
    ax_r.scatter(cv, i, color=C_CROSS, s=65, zorder=3, edgecolors=C_CROSS, linewidths=1.2)
    ax_r.scatter(wv, i, facecolors='white', edgecolors=C_WITHIN, s=65, zorder=3, linewidths=1.5)

ax_r.axvline(x=0, color='#999999', linestyle='--', linewidth=0.9, zorder=0)
ax_r.set_xlabel(r'$\Delta$FPR ($p = 0.5$)')

ax_r.axhspan(be_idx - 0.4, be_idx + 0.4, color='#FFF3E0', alpha=0.6, zorder=0)

fl_idx = y_pos['formal_logic']
fl_cross_fpr = data[(data['subject'] == 'formal_logic') & (data['matching_type'] == 'cross')]['delta_fpr'].values[0]
fl_within_fpr = data[(data['subject'] == 'formal_logic') & (data['matching_type'] == 'within')]['delta_fpr'].values[0]

ax_r.annotate('Cross-subject', xy=(fl_cross_fpr, fl_idx),
              xytext=(fl_cross_fpr - 0.003, fl_idx + 0.32),
              fontsize=8.5, color=C_CROSS, va='bottom', ha='right')
ax_r.annotate('Within-subject', xy=(fl_within_fpr, fl_idx),
              xytext=(fl_within_fpr + 0.003, fl_idx + 0.32),
              fontsize=8.5, color=C_WITHIN, va='bottom', ha='left')

out_dir = (
    '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen'
    '/figures/paper'
)
fig.savefig(f'{out_dir}/fig_3_within_subject_matching.pdf')
fig.savefig(f'{out_dir}/fig_3_within_subject_matching.png')
print(f'Saved to {out_dir}')
plt.close()
