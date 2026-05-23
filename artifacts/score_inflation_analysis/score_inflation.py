"""
Score Inflation Quantification & DIF-Corrected Scoring Analysis.
Computes original vs stable (DIF-free) scores, rank impacts, and equating adjustments.
"""
import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
ART = OUT.parent

# ── Load data ────────────────────────────────────────────────────────────────

rm_data = np.load(ART / 'response_matrix.npz', allow_pickle=True)
R = sparse.csr_matrix((rm_data['data'], rm_data['indices'], rm_data['indptr']),
                       shape=tuple(rm_data['shape']))

idx = np.load(ART / 'response_matrix_index.npz', allow_pickle=True)
model_names = idx['model_names']
item_ids = idx['item_ids']

meta = pd.read_csv(ART / 'model_metadata.csv')
dif = pd.read_csv(ART / 'plan_001' / 'dif_results_temporal.csv')

# ── Derive model_type (base vs instruct) ─────────────────────────────────────

def classify_type(row):
    name_lower = row['model_name'].lower()
    if any(kw in name_lower for kw in ['instruct', 'chat', 'rlhf']):
        return 'instruct'
    if row['training_method'] not in ['unknown']:
        return 'instruct'
    return 'base'

meta['model_type'] = meta.apply(classify_type, axis=1)

# ── Identify DIF-C items ─────────────────────────────────────────────────────

dif_c_items = set(dif.loc[dif['ets_class'] == 'C', 'item_id'].values)
item_id_list = list(item_ids)
dif_c_mask = np.array([iid in dif_c_items for iid in item_id_list])
stable_mask = ~dif_c_mask

n_total = len(item_id_list)
n_dif_c = dif_c_mask.sum()
n_stable = stable_mask.sum()
print(f"Items: total={n_total}, DIF-C={n_dif_c} ({100*n_dif_c/n_total:.1f}%), stable={n_stable}")

# ── Compute scores ───────────────────────────────────────────────────────────

R_dense = R.toarray()  # 5227 × 12508, int8 with -1 = missing

valid_mask = (R_dense >= 0)  # 0 or 1 are valid; -1 is missing

correct_all = np.where(valid_mask, R_dense, 0).sum(axis=1)
attempted_all = valid_mask.sum(axis=1)
original_score = correct_all / np.maximum(attempted_all, 1)

correct_stable = np.where(valid_mask & stable_mask[None, :], R_dense, 0).sum(axis=1)
attempted_stable = (valid_mask & stable_mask[None, :]).sum(axis=1)
stable_score = correct_stable / np.maximum(attempted_stable, 1)

score_change = stable_score - original_score

# ── Build per-model dataframe ────────────────────────────────────────────────

df = pd.DataFrame({
    'model_name': model_names,
    'original_score': original_score,
    'stable_score': stable_score,
    'score_change': score_change,
    'n_attempted_all': attempted_all,
    'n_attempted_stable': attempted_stable,
})

df = df.merge(meta[['model_name', 'cohort_temporal', 'model_type', 'family']], on='model_name', how='left')

# Rankings
df['original_rank'] = df['original_score'].rank(ascending=False, method='min').astype(int)
df['stable_rank'] = df['stable_score'].rank(ascending=False, method='min').astype(int)
df['rank_displacement'] = df['original_rank'] - df['stable_rank']

# ── Analysis 1 & 2: Score inflation by cohort and type ───────────────────────

def group_stats(group):
    n = len(group)
    mu = group['score_change'].mean()
    sd = group['score_change'].std()
    se = sd / np.sqrt(n)
    ci_lo = mu - 1.96 * se
    ci_hi = mu + 1.96 * se
    return pd.Series({
        'n': n,
        'mean_original': group['original_score'].mean(),
        'mean_stable': group['stable_score'].mean(),
        'mean_score_change': mu,
        'sd_score_change': sd,
        'ci95_lo': ci_lo,
        'ci95_hi': ci_hi,
        'median_score_change': group['score_change'].median(),
        'mean_rank_displacement': group['rank_displacement'].mean(),
        'sd_rank_displacement': group['rank_displacement'].std(),
    })

cohort_summary = df.groupby('cohort_temporal').apply(group_stats).reset_index()
type_summary = df.groupby(['cohort_temporal', 'model_type']).apply(group_stats).reset_index()

print("\n=== Cohort Summary ===")
print(cohort_summary.to_string(index=False))
print("\n=== Type Summary ===")
print(type_summary.to_string(index=False))

# ── Analysis 3: Ranking impact ──────────────────────────────────────────────

rho_all, p_rho = stats.spearmanr(df['original_rank'], df['stable_rank'])
print(f"\nOverall Spearman ρ (original vs stable ranking): {rho_all:.6f} (p={p_rho:.2e})")

for cohort in ['2023', '2024']:
    sub = df[df['cohort_temporal'] == cohort]
    rho_c, _ = stats.spearmanr(sub['original_rank'], sub['stable_rank'])
    print(f"  Cohort {cohort}: ρ = {rho_c:.6f}")

# Top displaced
top_beneficiaries = df.nlargest(20, 'rank_displacement')  # rank improved most in stable
top_victims = df.nsmallest(20, 'rank_displacement')  # rank worsened most in stable
top_displaced = pd.concat([
    top_beneficiaries.assign(direction='DIF_victim_rank_improves'),
    top_victims.assign(direction='DIF_beneficiary_rank_drops'),
])

print("\n=== Top-20 DIF Victims (rank improves when DIF items removed) ===")
print(top_beneficiaries[['model_name', 'cohort_temporal', 'model_type', 'original_score',
                          'stable_score', 'score_change', 'original_rank', 'stable_rank',
                          'rank_displacement']].to_string(index=False))

print("\n=== Top-20 DIF Beneficiaries (rank drops when DIF items removed) ===")
print(top_victims[['model_name', 'cohort_temporal', 'model_type', 'original_score',
                    'stable_score', 'score_change', 'original_rank', 'stable_rank',
                    'rank_displacement']].to_string(index=False))

# Characteristics of displaced models
print("\n=== Displacement characteristics ===")
for direction, group in [('victims', top_beneficiaries), ('beneficiaries', top_victims)]:
    print(f"\n{direction}:")
    print(f"  cohort: {group['cohort_temporal'].value_counts().to_dict()}")
    print(f"  type: {group['model_type'].value_counts().to_dict()}")
    print(f"  families: {group['family'].value_counts().head(5).to_dict()}")

# ── Analysis 4: Equating adjustment ─────────────────────────────────────────

print("\n=== Equating Adjustment ===")
for cohort in ['2023', '2024', 'other']:
    sub = df[df['cohort_temporal'] == cohort]
    shift = sub['score_change'].mean()
    print(f"  Cohort {cohort}: mean_shift = {shift:+.4f} ({shift*100:+.2f} pp)")

# Cross-cohort ranking comparison (2023 vs 2024 only)
df_23 = df[df['cohort_temporal'] == '2023'].copy()
df_24 = df[df['cohort_temporal'] == '2024'].copy()
df_cross = pd.concat([df_23, df_24])

rho_orig_cross, _ = stats.spearmanr(df_cross['original_score'].rank(ascending=False),
                                      df_cross['original_rank'])
rho_stable_cross, _ = stats.spearmanr(df_cross['stable_score'].rank(ascending=False),
                                        df_cross['stable_rank'])

# Adjusted scores with cohort-specific shifts
shifts = df.groupby('cohort_temporal')['score_change'].mean().to_dict()
df['adjusted_score'] = df.apply(lambda r: r['original_score'] + shifts.get(r['cohort_temporal'], 0), axis=1)
df['adjusted_rank'] = df['adjusted_score'].rank(ascending=False, method='min').astype(int)

rho_adj, _ = stats.spearmanr(df['original_rank'], df['adjusted_rank'])
rho_stable_adj, _ = stats.spearmanr(df['stable_rank'], df['adjusted_rank'])
print(f"\nRanking correlations:")
print(f"  original vs stable:   ρ = {rho_all:.6f}")
print(f"  original vs adjusted: ρ = {rho_adj:.6f}")
print(f"  stable vs adjusted:   ρ = {rho_stable_adj:.6f}")

# ── Effect size: Cohen's d for 2023 vs 2024 score change ────────────────────

sc_23 = df_23['score_change'].values
sc_24 = df_24['score_change'].values
pooled_sd = np.sqrt(((len(sc_23)-1)*sc_23.std()**2 + (len(sc_24)-1)*sc_24.std()**2) /
                     (len(sc_23)+len(sc_24)-2))
cohens_d = (sc_24.mean() - sc_23.mean()) / pooled_sd
t_stat, t_p = stats.ttest_ind(sc_24, sc_23)
print(f"\n2024 vs 2023 score_change:")
print(f"  Cohen's d = {cohens_d:.4f}")
print(f"  t-test: t={t_stat:.3f}, p={t_p:.2e}")

# ── Save CSVs ────────────────────────────────────────────────────────────────

df[['model_name', 'cohort_temporal', 'model_type', 'family',
    'original_score', 'stable_score', 'score_change',
    'original_rank', 'stable_rank', 'rank_displacement',
    'adjusted_score', 'adjusted_rank']].to_csv(OUT / 'model_scores.csv', index=False)

cohort_summary.to_csv(OUT / 'cohort_summary.csv', index=False)
type_summary.to_csv(OUT / 'type_summary.csv', index=False)

top_displaced[['model_name', 'cohort_temporal', 'model_type', 'family', 'direction',
               'original_score', 'stable_score', 'score_change',
               'original_rank', 'stable_rank', 'rank_displacement']].to_csv(OUT / 'top_displaced.csv', index=False)

print("\nCSVs saved.")

# ── Visualizations ───────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'legend.frameon': False,
    'figure.dpi': 150,
})

# 1. Scatter: original vs stable score by cohort
fig, ax = plt.subplots(figsize=(7, 6))
colors = {'2023': '#2166ac', '2024': '#b2182b', 'other': '#999999'}
for cohort in ['other', '2023', '2024']:
    sub = df[df['cohort_temporal'] == cohort]
    ax.scatter(sub['original_score'], sub['stable_score'],
               c=colors[cohort], alpha=0.25, s=8, label=cohort, rasterized=True)
lo, hi = df['original_score'].min() - 0.02, df['original_score'].max() + 0.02
ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5, label='y = x')
ax.set_xlabel('Original Score (all items)')
ax.set_ylabel('Stable Score (non-DIF-C items)')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.legend(loc='lower right', fontsize=9)
ax.set_title('Original vs. DIF-Corrected Scores')
fig.tight_layout()
fig.savefig(OUT / 'score_scatter.pdf', bbox_inches='tight')
fig.savefig(OUT / 'score_scatter.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 2. Rank displacement histogram by cohort
fig, ax = plt.subplots(figsize=(8, 5))
for cohort in ['2023', '2024']:
    sub = df[df['cohort_temporal'] == cohort]
    ax.hist(sub['rank_displacement'], bins=80, alpha=0.55, color=colors[cohort],
            label=f'{cohort} (n={len(sub)})', density=True)
ax.axvline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Rank Displacement (original_rank − stable_rank)')
ax.set_ylabel('Density')
ax.set_title('Rank Displacement Distribution by Cohort')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'rank_displacement.pdf', bbox_inches='tight')
fig.savefig(OUT / 'rank_displacement.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 3. Violin: score_change by cohort × model_type
fig, ax = plt.subplots(figsize=(8, 5.5))
groups = []
labels = []
positions = []
violin_colors = []
pos_map = {'2023': [1, 2], '2024': [4, 5]}
type_order = ['base', 'instruct']
color_map = {('2023', 'base'): '#4393c3', ('2023', 'instruct'): '#2166ac',
             ('2024', 'base'): '#d6604d', ('2024', 'instruct'): '#b2182b'}

for cohort in ['2023', '2024']:
    for i, mtype in enumerate(type_order):
        sub = df[(df['cohort_temporal'] == cohort) & (df['model_type'] == mtype)]
        if len(sub) < 5:
            continue
        groups.append(sub['score_change'].values * 100)  # convert to pp
        labels.append(f'{cohort}\n{mtype}')
        positions.append(pos_map[cohort][i])
        violin_colors.append(color_map[(cohort, mtype)])

parts = ax.violinplot(groups, positions=positions, showmeans=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(violin_colors[i])
    pc.set_alpha(0.7)
parts['cmeans'].set_color('black')

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
ax.set_ylabel('Score Change (pp): stable − original')
ax.set_title('Score Inflation/Deflation by Cohort × Model Type')
fig.tight_layout()
fig.savefig(OUT / 'score_change_violin.pdf', bbox_inches='tight')
fig.savefig(OUT / 'score_change_violin.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Figures saved.")

# ── Generate report ──────────────────────────────────────────────────────────

report_lines = []
report_lines.append("# Score Inflation & DIF-Corrected Scoring Analysis\n")
report_lines.append(f"**Date**: 2026-05-23\n")
report_lines.append(f"**Dataset**: {n_total} items, {len(df)} models\n")
report_lines.append(f"**DIF-C items**: {n_dif_c} ({100*n_dif_c/n_total:.1f}%)\n")
report_lines.append(f"**Stable items**: {n_stable} ({100*n_stable/n_total:.1f}%)\n")

report_lines.append("\n## 1. Score Change Summary\n")
report_lines.append("### By Cohort\n")
report_lines.append("| Cohort | N | Mean Original | Mean Stable | Mean Δ (pp) | SD (pp) | 95% CI |\n")
report_lines.append("|--------|---|--------------|-------------|-------------|---------|--------|\n")
for _, row in cohort_summary.iterrows():
    report_lines.append(
        f"| {row['cohort_temporal']} | {int(row['n'])} | "
        f"{row['mean_original']:.4f} | {row['mean_stable']:.4f} | "
        f"{row['mean_score_change']*100:+.2f} | {row['sd_score_change']*100:.2f} | "
        f"[{row['ci95_lo']*100:+.2f}, {row['ci95_hi']*100:+.2f}] |\n"
    )

report_lines.append("\n### By Cohort × Model Type\n")
report_lines.append("| Cohort | Type | N | Mean Δ (pp) | SD (pp) | 95% CI |\n")
report_lines.append("|--------|------|---|-------------|---------|--------|\n")
for _, row in type_summary.iterrows():
    report_lines.append(
        f"| {row['cohort_temporal']} | {row['model_type']} | {int(row['n'])} | "
        f"{row['mean_score_change']*100:+.2f} | {row['sd_score_change']*100:.2f} | "
        f"[{row['ci95_lo']*100:+.2f}, {row['ci95_hi']*100:+.2f}] |\n"
    )

report_lines.append(f"\n### Effect Size\n")
report_lines.append(f"- 2024 vs 2023 score_change Cohen's d = {cohens_d:.4f}\n")
report_lines.append(f"- Welch's t-test: t = {t_stat:.3f}, p = {t_p:.2e}\n")

report_lines.append("\n## 2. Ranking Impact\n")
report_lines.append(f"- Overall Spearman ρ (original vs stable): **{rho_all:.6f}**\n")
report_lines.append(f"- Original vs adjusted: ρ = {rho_adj:.6f}\n")
report_lines.append(f"- Stable vs adjusted: ρ = {rho_stable_adj:.6f}\n")

report_lines.append("\n### Top-20 DIF Victims (rank improves most)\n")
report_lines.append("| Model | Cohort | Type | Orig Score | Stable Score | Δ pp | Rank Disp |\n")
report_lines.append("|-------|--------|------|-----------|-------------|------|----------|\n")
for _, row in top_beneficiaries.iterrows():
    report_lines.append(
        f"| {row['model_name'][:50]} | {row['cohort_temporal']} | {row['model_type']} | "
        f"{row['original_score']:.4f} | {row['stable_score']:.4f} | "
        f"{row['score_change']*100:+.2f} | {int(row['rank_displacement']):+d} |\n"
    )

report_lines.append("\n### Top-20 DIF Beneficiaries (rank drops most)\n")
report_lines.append("| Model | Cohort | Type | Orig Score | Stable Score | Δ pp | Rank Disp |\n")
report_lines.append("|-------|--------|------|-----------|-------------|------|----------|\n")
for _, row in top_victims.iterrows():
    report_lines.append(
        f"| {row['model_name'][:50]} | {row['cohort_temporal']} | {row['model_type']} | "
        f"{row['original_score']:.4f} | {row['stable_score']:.4f} | "
        f"{row['score_change']*100:+.2f} | {int(row['rank_displacement']):+d} |\n"
    )

report_lines.append("\n## 3. Equating Adjustment\n")
for cohort in ['2023', '2024', 'other']:
    sub = df[df['cohort_temporal'] == cohort]
    shift = sub['score_change'].mean()
    report_lines.append(f"- **{cohort}**: adjusted_score = original_score + ({shift:+.4f})\n")

report_lines.append("\n## 4. Interpretation\n")

# Compute interpretation-relevant numbers
sc_2023_mean = df[df['cohort_temporal']=='2023']['score_change'].mean()
sc_2024_mean = df[df['cohort_temporal']=='2024']['score_change'].mean()
diff_pp = (sc_2024_mean - sc_2023_mean) * 100

report_lines.append(
    f"Removing the {n_dif_c} DIF-C items ({100*n_dif_c/n_total:.1f}% of all items) "
    f"shifts model scores by an average of {df['score_change'].mean()*100:+.2f} pp overall. "
    f"The direction of this shift differs by cohort: 2023 models see a mean change of "
    f"{sc_2023_mean*100:+.2f} pp while 2024 models see {sc_2024_mean*100:+.2f} pp "
    f"(differential: {diff_pp:+.2f} pp).\n\n"
)

report_lines.append(
    f"Despite measurable score changes, the overall ranking is highly preserved "
    f"(ρ = {rho_all:.4f}), indicating that DIF primarily shifts absolute scores "
    f"rather than relative ordering. However, individual models can experience "
    f"rank displacements exceeding ±{int(abs(df['rank_displacement']).quantile(0.99))} positions "
    f"(99th percentile).\n"
)

(OUT / 'score_inflation_report.md').write_text(''.join(report_lines))
print("Report saved.")
print("\nDone.")
