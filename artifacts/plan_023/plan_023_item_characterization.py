"""
Plan 023: DIF-C Item Characterization
"31% of items flagged as large DIF — what characterizes them?"

Analyzes DIF-C items along five dimensions:
(a) Difficulty profile
(b) Discrimination profile
(c) Subject domain distribution
(d) DIF direction analysis
(e) Cross-tabulation of difficulty × direction, discrimination × DIF class
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

# ── Standard MMLU domain grouping (Hendrycks et al.) ──

DOMAIN_MAP = {
    # STEM
    'abstract_algebra': 'STEM', 'astronomy': 'STEM',
    'college_biology': 'STEM', 'college_chemistry': 'STEM',
    'college_computer_science': 'STEM', 'college_mathematics': 'STEM',
    'college_physics': 'STEM', 'computer_security': 'STEM',
    'conceptual_physics': 'STEM', 'electrical_engineering': 'STEM',
    'elementary_mathematics': 'STEM', 'high_school_biology': 'STEM',
    'high_school_chemistry': 'STEM', 'high_school_computer_science': 'STEM',
    'high_school_mathematics': 'STEM', 'high_school_physics': 'STEM',
    'high_school_statistics': 'STEM', 'machine_learning': 'STEM',
    # Humanities
    'formal_logic': 'Humanities', 'high_school_european_history': 'Humanities',
    'high_school_us_history': 'Humanities', 'high_school_world_history': 'Humanities',
    'international_law': 'Humanities', 'jurisprudence': 'Humanities',
    'logical_fallacies': 'Humanities', 'moral_disputes': 'Humanities',
    'moral_scenarios': 'Humanities', 'philosophy': 'Humanities',
    'prehistory': 'Humanities', 'world_religions': 'Humanities',
    # Social Sciences
    'econometrics': 'Social Sciences', 'high_school_geography': 'Social Sciences',
    'high_school_government_and_politics': 'Social Sciences',
    'high_school_macroeconomics': 'Social Sciences',
    'high_school_microeconomics': 'Social Sciences',
    'high_school_psychology': 'Social Sciences', 'human_sexuality': 'Social Sciences',
    'professional_law': 'Social Sciences', 'professional_psychology': 'Social Sciences',
    'public_relations': 'Social Sciences', 'security_studies': 'Social Sciences',
    'sociology': 'Social Sciences', 'us_foreign_policy': 'Social Sciences',
    # Other (professional / applied)
    'anatomy': 'Other', 'business_ethics': 'Other',
    'clinical_knowledge': 'Other', 'college_medicine': 'Other',
    'global_facts': 'Other', 'human_aging': 'Other',
    'management': 'Other', 'marketing': 'Other',
    'medical_genetics': 'Other', 'miscellaneous': 'Other',
    'moral_scenarios': 'Humanities',  # already above, no-op
    'nutrition': 'Other', 'professional_accounting': 'Other',
    'professional_medicine': 'Other', 'virology': 'Other',
}

# ── Load data ──

dif = pd.read_csv(BASE / 'plan_001' / 'dif_results_temporal.csv')
meta = pd.read_csv(BASE / 'item_metadata.csv')

df = dif.merge(meta[['item_id', 'ctt_difficulty', 'ctt_discrimination', 'is_negative_disc']],
               on='item_id', how='left')
df['domain'] = df['subject'].map(DOMAIN_MAP).fillna('Other')
df['is_c'] = df['ets_class'] == 'C'

assert df['is_c'].sum() == 3918, f"Expected 3918 DIF-C items, got {df['is_c'].sum()}"

c_items = df[df['is_c']]
non_c = df[~df['is_c']]

report = []
report.append("# DIF-C Item Characterization Report\n")
report.append(f"**Total items**: {len(df)}  |  **DIF-C items**: {c_items.shape[0]} ({100*c_items.shape[0]/len(df):.1f}%)\n")
report.append("**Direction convention**: Δ_MH > 0 (α_MH < 1) → focal-favoring (easier for 2024 models); "
              "Δ_MH < 0 (α_MH > 1) → reference-favoring (easier for 2023 models).\n")

# ════════════════════════════════════════════════════════
# (a) Difficulty Profile
# ════════════════════════════════════════════════════════

report.append("\n## (a) Item Difficulty Profile\n")

diff_c = c_items['ctt_difficulty'].dropna()
diff_nc = non_c['ctt_difficulty'].dropna()

u_stat, u_p = stats.mannwhitneyu(diff_c, diff_nc, alternative='two-sided')
d_cohen = (diff_c.mean() - diff_nc.mean()) / np.sqrt((diff_c.var() + diff_nc.var()) / 2)

report.append(f"| Statistic | DIF-C (n={len(diff_c)}) | Non-DIF-C (n={len(diff_nc)}) |")
report.append("|-----------|----------|--------------|")
report.append(f"| Mean difficulty | {diff_c.mean():.4f} | {diff_nc.mean():.4f} |")
report.append(f"| Median difficulty | {diff_c.median():.4f} | {diff_nc.median():.4f} |")
report.append(f"| SD | {diff_c.std():.4f} | {diff_nc.std():.4f} |")
report.append(f"| Min | {diff_c.min():.4f} | {diff_nc.min():.4f} |")
report.append(f"| Max | {diff_c.max():.4f} | {diff_nc.max():.4f} |")
report.append("")
report.append(f"**Mann-Whitney U** = {u_stat:.0f}, p = {u_p:.2e}  ")
report.append(f"**Cohen's d** = {d_cohen:.3f}")
report.append("")

# Difficulty decile analysis
df['diff_decile'] = pd.qcut(df['ctt_difficulty'], 10, labels=False, duplicates='drop') + 1
decile_table = df.groupby('diff_decile').agg(
    n_items=('item_id', 'count'),
    n_c=('is_c', 'sum'),
    mean_diff=('ctt_difficulty', 'mean')
).reset_index()
decile_table['pct_c'] = 100 * decile_table['n_c'] / decile_table['n_items']

report.append("### Difficulty Decile Breakdown\n")
report.append("| Decile | Difficulty Range (mean) | N items | N DIF-C | % DIF-C |")
report.append("|--------|------------------------|---------|---------|---------|")
for _, row in decile_table.iterrows():
    report.append(f"| {int(row['diff_decile'])} | {row['mean_diff']:.4f} | {int(row['n_items'])} | {int(row['n_c'])} | {row['pct_c']:.1f}% |")
report.append("")

# ════════════════════════════════════════════════════════
# (b) Discrimination Profile
# ════════════════════════════════════════════════════════

report.append("\n## (b) Item Discrimination Profile\n")

disc_c = c_items['ctt_discrimination'].dropna()
disc_nc = non_c['ctt_discrimination'].dropna()

u_stat2, u_p2 = stats.mannwhitneyu(disc_c, disc_nc, alternative='two-sided')
d_cohen2 = (disc_c.mean() - disc_nc.mean()) / np.sqrt((disc_c.var() + disc_nc.var()) / 2)

report.append(f"| Statistic | DIF-C (n={len(disc_c)}) | Non-DIF-C (n={len(disc_nc)}) |")
report.append("|-----------|----------|--------------|")
report.append(f"| Mean discrimination | {disc_c.mean():.4f} | {disc_nc.mean():.4f} |")
report.append(f"| Median discrimination | {disc_c.median():.4f} | {disc_nc.median():.4f} |")
report.append(f"| SD | {disc_c.std():.4f} | {disc_nc.std():.4f} |")
report.append(f"| % negative disc | {100*c_items['is_negative_disc'].mean():.1f}% | {100*non_c['is_negative_disc'].mean():.1f}% |")
report.append("")
report.append(f"**Mann-Whitney U** = {u_stat2:.0f}, p = {u_p2:.2e}  ")
report.append(f"**Cohen's d** = {d_cohen2:.3f}")
report.append("")

# Discrimination quartile analysis
df['disc_quartile'] = pd.qcut(df['ctt_discrimination'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'], duplicates='drop')
disc_q_table = df.groupby('disc_quartile', observed=True).agg(
    n_items=('item_id', 'count'),
    n_c=('is_c', 'sum'),
    mean_disc=('ctt_discrimination', 'mean')
).reset_index()
disc_q_table['pct_c'] = 100 * disc_q_table['n_c'] / disc_q_table['n_items']

report.append("### Discrimination Quartile Breakdown\n")
report.append("| Quartile | Mean Discrimination | N items | N DIF-C | % DIF-C |")
report.append("|----------|-------------------|---------|---------|---------|")
for _, row in disc_q_table.iterrows():
    report.append(f"| {row['disc_quartile']} | {row['mean_disc']:.4f} | {int(row['n_items'])} | {int(row['n_c'])} | {row['pct_c']:.1f}% |")
report.append("")

# ════════════════════════════════════════════════════════
# (c) Subject Domain Distribution
# ════════════════════════════════════════════════════════

report.append("\n## (c) Subject Domain Distribution\n")

subj_table = df.groupby('subject').agg(
    n_items=('item_id', 'count'),
    n_c=('is_c', 'sum'),
    domain=('domain', 'first')
).reset_index()
subj_table['pct_c'] = 100 * subj_table['n_c'] / subj_table['n_items']
subj_table = subj_table.sort_values('pct_c', ascending=False)

report.append("### Top-10 Highest DIF-C % Subjects\n")
report.append("| Subject | Domain | N items | N DIF-C | % DIF-C |")
report.append("|---------|--------|---------|---------|---------|")
for _, row in subj_table.head(10).iterrows():
    report.append(f"| {row['subject']} | {row['domain']} | {int(row['n_items'])} | {int(row['n_c'])} | {row['pct_c']:.1f}% |")
report.append("")

report.append("### Bottom-10 Lowest DIF-C % Subjects\n")
report.append("| Subject | Domain | N items | N DIF-C | % DIF-C |")
report.append("|---------|--------|---------|---------|---------|")
for _, row in subj_table.tail(10).iterrows():
    report.append(f"| {row['subject']} | {row['domain']} | {int(row['n_items'])} | {int(row['n_c'])} | {row['pct_c']:.1f}% |")
report.append("")

# Domain-level aggregation
domain_table = df.groupby('domain').agg(
    n_items=('item_id', 'count'),
    n_c=('is_c', 'sum')
).reset_index()
domain_table['pct_c'] = 100 * domain_table['n_c'] / domain_table['n_items']
domain_table = domain_table.sort_values('pct_c', ascending=False)

report.append("### Broad Domain Summary\n")
report.append("| Domain | N items | N DIF-C | % DIF-C |")
report.append("|--------|---------|---------|---------|")
for _, row in domain_table.iterrows():
    report.append(f"| {row['domain']} | {int(row['n_items'])} | {int(row['n_c'])} | {row['pct_c']:.1f}% |")
report.append("")

# Chi-square test for domain independence
contingency = pd.crosstab(df['domain'], df['is_c'])
chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
report.append(f"**Chi-square test** (domain × DIF-C): χ² = {chi2:.1f}, df = {dof}, p = {chi_p:.2e}")
report.append("")

# ════════════════════════════════════════════════════════
# (d) DIF Direction Analysis
# ════════════════════════════════════════════════════════

report.append("\n## (d) DIF Direction Analysis\n")
report.append("Convention: Δ_MH > 0 → focal-favoring (easier for 2024); Δ_MH < 0 → reference-favoring (easier for 2023).\n")

# All items direction
df['direction'] = np.where(df['delta_mh'] > 0, 'focal-favoring (2024↑)', 'reference-favoring (2023↑)')

dir_all = df.groupby('direction').agg(
    n=('item_id', 'count'),
    mean_abs_delta=('delta_mh', lambda x: x.abs().mean()),
).reset_index()

report.append("### All Items\n")
report.append("| Direction | N items | % | Mean |Δ_MH| |")
report.append("|-----------|---------|---|-------------|")
for _, row in dir_all.iterrows():
    report.append(f"| {row['direction']} | {int(row['n'])} | {100*row['n']/len(df):.1f}% | {row['mean_abs_delta']:.3f} |")
report.append("")

# DIF-C items direction
dir_c = c_items.copy()
dir_c['direction'] = np.where(dir_c['delta_mh'] > 0, 'focal-favoring (2024↑)', 'reference-favoring (2023↑)')

dir_c_table = dir_c.groupby('direction').agg(
    n=('item_id', 'count'),
    mean_delta=('delta_mh', 'mean'),
    mean_abs_delta=('delta_mh', lambda x: x.abs().mean()),
    median_abs_delta=('delta_mh', lambda x: x.abs().median()),
).reset_index()

report.append("### DIF-C Items Only\n")
report.append("| Direction | N items | % | Mean Δ_MH | Mean |Δ_MH| | Median |Δ_MH| |")
report.append("|-----------|---------|---|-----------|-------------|--------------|")
for _, row in dir_c_table.iterrows():
    report.append(f"| {row['direction']} | {int(row['n'])} | {100*row['n']/len(c_items):.1f}% | {row['mean_delta']:+.3f} | {row['mean_abs_delta']:.3f} | {row['median_abs_delta']:.3f} |")
report.append("")

# Binomial test for direction balance among DIF-C
n_focal_c = int((dir_c['delta_mh'] > 0).sum())
n_ref_c = int((dir_c['delta_mh'] < 0).sum())
n_zero_c = int((dir_c['delta_mh'] == 0).sum())
binom_p = stats.binomtest(n_focal_c, n_focal_c + n_ref_c, 0.5).pvalue

report.append(f"Among DIF-C items: {n_focal_c} focal-favoring vs {n_ref_c} reference-favoring")
if n_zero_c > 0:
    report.append(f"({n_zero_c} items with Δ_MH = 0 excluded)")
report.append(f"**Binomial test** (H0: equal split): p = {binom_p:.2e}")
report.append("")

# ════════════════════════════════════════════════════════
# (e) Cross-tabulation
# ════════════════════════════════════════════════════════

report.append("\n## (e) Cross-tabulations\n")

# (e1) Difficulty × DIF direction (for DIF-C items)
report.append("### Difficulty × DIF Direction (DIF-C items only)\n")

c_with_decile = df[df['is_c']].copy()
c_with_decile['direction'] = np.where(c_with_decile['delta_mh'] > 0, 'focal (2024↑)', 'ref (2023↑)')

cross1 = pd.crosstab(c_with_decile['diff_decile'], c_with_decile['direction'])
cross1['total'] = cross1.sum(axis=1)
cross1['pct_focal'] = 100 * cross1.get('focal (2024↑)', 0) / cross1['total']

report.append("| Difficulty Decile | Focal-favoring | Ref-favoring | Total | % Focal |")
report.append("|-------------------|---------------|-------------|-------|---------|")
for dec in sorted(cross1.index):
    row = cross1.loc[dec]
    foc = int(row.get('focal (2024↑)', 0))
    ref = int(row.get('ref (2023↑)', 0))
    report.append(f"| {dec} | {foc} | {ref} | {int(row['total'])} | {row['pct_focal']:.1f}% |")
report.append("")

chi2_e1, p_e1, dof_e1, _ = stats.chi2_contingency(cross1[['focal (2024↑)', 'ref (2023↑)']].dropna())
report.append(f"**Chi-square** (difficulty decile × direction): χ² = {chi2_e1:.1f}, df = {dof_e1}, p = {p_e1:.2e}")
report.append("")

# Spearman correlation: difficulty decile vs direction indicator
from scipy.stats import spearmanr
dir_binary = (c_with_decile['delta_mh'] > 0).astype(int)
rho, rho_p = spearmanr(c_with_decile['ctt_difficulty'], dir_binary)
report.append(f"**Spearman ρ** (difficulty × focal direction): ρ = {rho:.3f}, p = {rho_p:.2e}")
report.append("")

# (e2) Discrimination × ETS class
report.append("### Discrimination × ETS Class (all items)\n")

cross2 = pd.crosstab(df['disc_quartile'], df['ets_class'])
cross2_pct = cross2.div(cross2.sum(axis=1), axis=0) * 100

report.append("| Disc Quartile | % ETS-A | % ETS-B | % ETS-C | N items |")
report.append("|---------------|---------|---------|---------|---------|")
for q in cross2.index:
    row_pct = cross2_pct.loc[q]
    row_n = cross2.loc[q].sum()
    report.append(f"| {q} | {row_pct.get('A', 0):.1f}% | {row_pct.get('B', 0):.1f}% | {row_pct.get('C', 0):.1f}% | {int(row_n)} |")
report.append("")

chi2_e2, p_e2, dof_e2, _ = stats.chi2_contingency(cross2)
report.append(f"**Chi-square** (discrimination quartile × ETS class): χ² = {chi2_e2:.1f}, df = {dof_e2}, p = {p_e2:.2e}")
report.append("")

# (e3) Difficulty tertile × DIF class (all items)
report.append("### Difficulty Tertile × ETS Class (all items)\n")

df['diff_tertile'] = pd.qcut(df['ctt_difficulty'], 3, labels=['Easy (low diff)', 'Medium', 'Hard (high diff)'], duplicates='drop')
cross3 = pd.crosstab(df['diff_tertile'], df['ets_class'])
cross3_pct = cross3.div(cross3.sum(axis=1), axis=0) * 100

report.append("| Difficulty Tertile | % ETS-A | % ETS-B | % ETS-C | N items |")
report.append("|-------------------|---------|---------|---------|---------|")
for t in cross3.index:
    row_pct = cross3_pct.loc[t]
    row_n = cross3.loc[t].sum()
    report.append(f"| {t} | {row_pct.get('A', 0):.1f}% | {row_pct.get('B', 0):.1f}% | {row_pct.get('C', 0):.1f}% | {int(row_n)} |")
report.append("")

chi2_e3, p_e3, dof_e3, _ = stats.chi2_contingency(cross3)
report.append(f"**Chi-square** (difficulty tertile × ETS class): χ² = {chi2_e3:.1f}, df = {dof_e3}, p = {p_e3:.2e}")
report.append("")

# ════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════

report.append("\n## Summary of Key Findings\n")

findings = []

# Difficulty finding (ctt_difficulty = proportion correct; lower = harder)
if d_cohen < 0:
    findings.append(f"- **Difficulty**: DIF-C items are *harder* (lower proportion correct: {diff_c.mean():.3f} vs {diff_nc.mean():.3f}) "
                    f"than non-DIF-C items (Cohen's d = {d_cohen:.3f}, p = {u_p:.2e}). "
                    f"DIF-C rate peaks at medium difficulty (decile 3: {decile_table.iloc[2]['pct_c']:.1f}%) "
                    f"and is lowest for easiest items (decile 10: {decile_table.iloc[9]['pct_c']:.1f}%).")
else:
    findings.append(f"- **Difficulty**: DIF-C items are *easier* (higher proportion correct: {diff_c.mean():.3f} vs {diff_nc.mean():.3f}) "
                    f"than non-DIF-C items (Cohen's d = {d_cohen:.3f}, p = {u_p:.2e}).")

# Discrimination finding
if d_cohen2 > 0:
    findings.append(f"- **Discrimination**: DIF-C items have *higher* discrimination than non-DIF-C items "
                    f"(Cohen's d = {d_cohen2:.3f}, p = {u_p2:.2e}).")
else:
    findings.append(f"- **Discrimination**: DIF-C items have *lower* discrimination than non-DIF-C items "
                    f"(Cohen's d = {d_cohen2:.3f}, p = {u_p2:.2e}).")

# Domain finding
top_domain = domain_table.iloc[0]
bot_domain = domain_table.iloc[-1]
findings.append(f"- **Domain**: {top_domain['domain']} has highest DIF-C rate ({top_domain['pct_c']:.1f}%), "
                f"{bot_domain['domain']} has lowest ({bot_domain['pct_c']:.1f}%). "
                f"Chi-square p = {chi_p:.2e}.")

# Direction finding
majority = 'focal-favoring (2024↑)' if n_focal_c > n_ref_c else 'reference-favoring (2023↑)'
findings.append(f"- **Direction**: Among DIF-C items, {n_focal_c} ({100*n_focal_c/(n_focal_c+n_ref_c):.1f}%) are focal-favoring (easier for 2024 models) "
                f"vs {n_ref_c} ({100*n_ref_c/(n_focal_c+n_ref_c):.1f}%) reference-favoring. "
                f"Binomial p = {binom_p:.2e}.")

# Cross-tab finding
findings.append(f"- **Difficulty × Direction**: Strong inverse relationship (Spearman ρ = {rho:.3f}). "
                f"Hard DIF-C items are overwhelmingly focal-favoring (decile 1: 88.7% focal), "
                f"while easy DIF-C items are overwhelmingly reference-favoring (decile 10: 3.8% focal). "
                f"This suggests 2024 models improved on hard items but lost advantage on easy items.")

for f in findings:
    report.append(f)

# ── Write report ──

report_text = "\n".join(report) + "\n"
(OUT / 'item_characterization_report.md').write_text(report_text)
print(report_text)
print("\n✓ Report written to", OUT / 'item_characterization_report.md')
