import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
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

DOMAIN_COLORS = {
    'STEM':            '#0072B2',
    'Humanities':      '#E69F00',
    'Social Science':  '#009E73',
    'Professional':    '#CC79A7',
    'Other':           '#56B4E9',
}

SUBJECT_DOMAIN = {
    'abstract_algebra':                    'STEM',
    'anatomy':                             'STEM',
    'astronomy':                           'STEM',
    'business_ethics':                     'Professional',
    'clinical_knowledge':                  'Professional',
    'college_biology':                     'STEM',
    'college_chemistry':                   'STEM',
    'college_computer_science':            'STEM',
    'college_mathematics':                 'STEM',
    'college_medicine':                    'Professional',
    'college_physics':                     'STEM',
    'computer_security':                   'STEM',
    'conceptual_physics':                  'STEM',
    'econometrics':                        'Social Science',
    'electrical_engineering':              'STEM',
    'elementary_mathematics':              'STEM',
    'formal_logic':                        'Humanities',
    'global_facts':                        'Other',
    'high_school_biology':                 'STEM',
    'high_school_chemistry':               'STEM',
    'high_school_computer_science':        'STEM',
    'high_school_european_history':        'Humanities',
    'high_school_geography':               'Social Science',
    'high_school_government_and_politics': 'Social Science',
    'high_school_macroeconomics':          'Social Science',
    'high_school_mathematics':             'STEM',
    'high_school_microeconomics':          'Social Science',
    'high_school_physics':                 'STEM',
    'high_school_psychology':              'Social Science',
    'high_school_statistics':              'STEM',
    'high_school_us_history':              'Humanities',
    'high_school_world_history':           'Humanities',
    'human_aging':                         'Other',
    'human_sexuality':                     'Social Science',
    'international_law':                   'Humanities',
    'jurisprudence':                       'Humanities',
    'logical_fallacies':                   'Humanities',
    'machine_learning':                    'STEM',
    'management':                          'Professional',
    'marketing':                           'Professional',
    'medical_genetics':                    'STEM',
    'miscellaneous':                       'Other',
    'moral_disputes':                      'Humanities',
    'moral_scenarios':                     'Humanities',
    'nutrition':                           'Other',
    'philosophy':                          'Humanities',
    'prehistory':                          'Humanities',
    'professional_accounting':             'Professional',
    'professional_law':                    'Humanities',
    'professional_medicine':               'Professional',
    'professional_psychology':             'Social Science',
    'public_relations':                    'Professional',
    'security_studies':                    'Social Science',
    'sociology':                           'Social Science',
    'us_foreign_policy':                   'Social Science',
    'virology':                            'STEM',
    'world_religions':                     'Humanities',
}

def format_subject(name):
    return name.replace('_', ' ').title()

DATA_PATH = '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts/plan_001_real_data_dif_mmlu/data/domain_specificity.csv'
OUT_DIR = '/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/figures/paper'

df = pd.read_csv(DATA_PATH)
df['domain'] = df['subject'].map(SUBJECT_DOMAIN)
df['label'] = df['subject'].apply(format_subject)
df = df.sort_values('pct_c', ascending=True)  # ascending so highest is at top in barh

fig, ax = plt.subplots(figsize=(8, 16))

colors = [DOMAIN_COLORS[d] for d in df['domain']]
bars = ax.barh(range(len(df)), df['pct_c'], color=colors, height=0.7, zorder=2)

ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['label'], fontsize=8.5)
ax.set_xlabel('ETS C Items (%)')
ax.set_xlim(0, 55)

mean_c = 31.3
ax.axvline(mean_c, color='#333333', linestyle='--', linewidth=1.0, zorder=1)
ax.text(mean_c + 0.5, len(df) - 1.5, f'Mean = {mean_c}%', fontsize=9,
        color='#333333', va='top')

for i, (val, bar) in enumerate(zip(df['pct_c'], bars)):
    ax.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=7.5, color='#333333')

legend_patches = [mpatches.Patch(color=c, label=l) for l, c in DOMAIN_COLORS.items()]
ax.legend(handles=legend_patches, loc='lower right', framealpha=0.9, edgecolor='#cccccc')

ax.grid(axis='x', alpha=0.2, zorder=0)
ax.set_ylim(-0.5, len(df) - 0.5)

plt.tight_layout()

pdf_path = os.path.join(OUT_DIR, 'fig_A1_subject_heatmap.pdf')
png_path = os.path.join(OUT_DIR, 'fig_A1_subject_heatmap.png')
plt.savefig(pdf_path)
plt.savefig(png_path)
plt.close()

print(f'Saved: {pdf_path}')
print(f'Saved: {png_path}')
