"""Cross-analysis: Local Dependence (Q3) vs DIF C% by subject domain."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen")
OUT = ROOT / "artifacts" / "plan_013_supplement"

q3 = pd.read_csv(ROOT / "artifacts" / "plan_013" / "q3_summary.csv")
dif = pd.read_csv(ROOT / "artifacts" / "plan_001" / "domain_specificity.csv")

merged = q3.merge(dif, on="subject", suffixes=("_q3", "_dif"))
print(f"Merged: {len(merged)} subjects (Q3: {len(q3)}, DIF: {len(dif)})")

# --- Correlation: pct_above_020 vs pct_c ---
x_ld = merged["pct_above_020"]
y_dif = merged["pct_c"]

r_pearson, p_pearson = stats.pearsonr(x_ld, y_dif)
r_spearman, p_spearman = stats.spearmanr(x_ld, y_dif)

print(f"\n=== pct |Q3|>0.20 vs DIF C% ===")
print(f"Pearson  r = {r_pearson:.4f}, p = {p_pearson:.4e}")
print(f"Spearman ρ = {r_spearman:.4f}, p = {p_spearman:.4e}")

# --- Correlation: mean_q3 vs pct_c ---
r2_pearson, p2_pearson = stats.pearsonr(merged["mean_q3"], y_dif)
r2_spearman, p2_spearman = stats.spearmanr(merged["mean_q3"], y_dif)

print(f"\n=== mean Q3 vs DIF C% ===")
print(f"Pearson  r = {r2_pearson:.4f}, p = {p2_pearson:.4e}")
print(f"Spearman ρ = {r2_spearman:.4f}, p = {p2_spearman:.4e}")

# --- Top-5 overlap ---
top5_ld = set(merged.nlargest(5, "pct_above_020")["subject"])
top5_dif = set(merged.nlargest(5, "pct_c")["subject"])
overlap = top5_ld & top5_dif

print(f"\nTop-5 highest LD (pct |Q3|>0.20):")
for _, row in merged.nlargest(5, "pct_above_020").iterrows():
    print(f"  {row['subject']:40s} {row['pct_above_020']:.1f}%")

print(f"\nTop-5 highest DIF C%:")
for _, row in merged.nlargest(5, "pct_c").iterrows():
    print(f"  {row['subject']:40s} {row['pct_c']:.1f}%")

print(f"\nOverlap: {overlap if overlap else 'NONE'}")

# --- Bottom-5 for reference ---
bot5_ld = set(merged.nsmallest(5, "pct_above_020")["subject"])
bot5_dif = set(merged.nsmallest(5, "pct_c")["subject"])
overlap_bot = bot5_ld & bot5_dif

print(f"\nBottom-5 lowest LD:")
for _, row in merged.nsmallest(5, "pct_above_020").iterrows():
    print(f"  {row['subject']:40s} {row['pct_above_020']:.1f}%")

print(f"\nBottom-5 lowest DIF C%:")
for _, row in merged.nsmallest(5, "pct_c").iterrows():
    print(f"  {row['subject']:40s} {row['pct_c']:.1f}%")

print(f"\nBottom overlap: {overlap_bot if overlap_bot else 'NONE'}")

# --- Additional: correlation with accuracy change ---
r3, p3 = stats.pearsonr(x_ld, merged["acc_change"])
r4, p4 = stats.pearsonr(y_dif, merged["acc_change"])
print(f"\n=== Supplementary: correlations with acc_change ===")
print(f"pct |Q3|>0.20 vs acc_change: r = {r3:.4f}, p = {p3:.4e}")
print(f"DIF C%        vs acc_change: r = {r4:.4f}, p = {p4:.4e}")

# --- Kendall tau for robustness ---
tau_main, p_tau = stats.kendalltau(x_ld, y_dif)
print(f"\n=== Kendall tau: pct |Q3|>0.20 vs DIF C% ===")
print(f"τ = {tau_main:.4f}, p = {p_tau:.4e}")

# --- Save merged table ---
out_cols = ["subject", "n_items_q3", "mean_q3", "pct_above_020", "n_items_dif", "pct_c", "acc_change"]
merged_out = merged[out_cols].sort_values("pct_above_020", ascending=False)
merged_out.to_csv(OUT / "ld_dif_merged.csv", index=False)
print(f"\nSaved merged table to {OUT / 'ld_dif_merged.csv'}")

# --- Generate report ---
report = f"""# LD Domain vs DIF Domain Cross-Analysis

## Data Sources
- **LD metric**: Yen's Q3 per-subject statistics from plan_013 (`q3_summary.csv`)
  - Primary metric: % of within-subject item pairs with |Q3| > 0.20
  - Secondary metric: mean Q3
- **DIF metric**: ETS C-category % from plan_001 (`domain_specificity.csv`)
  - Metric: % of items flagged as C (large DIF) per subject
- **N subjects merged**: {len(merged)}

## Correlation Results

### Primary: % |Q3| > 0.20 vs DIF C%

| Statistic | Value | p-value |
|-----------|-------|---------|
| Pearson r | {r_pearson:.4f} | {p_pearson:.4e} |
| Spearman ρ | {r_spearman:.4f} | {p_spearman:.4e} |
| Kendall τ | {tau_main:.4f} | {p_tau:.4e} |

### Secondary: Mean Q3 vs DIF C%

| Statistic | Value | p-value |
|-----------|-------|---------|
| Pearson r | {r2_pearson:.4f} | {p2_pearson:.4e} |
| Spearman ρ | {r2_spearman:.4f} | {p2_spearman:.4e} |

## Top-5 Overlap Analysis

### Highest LD Subjects (% |Q3| > 0.20)
| Rank | Subject | % |Q3|>0.20 | DIF C% |
|------|---------|------------|--------|
"""

for i, (_, row) in enumerate(merged.nlargest(5, "pct_above_020").iterrows(), 1):
    marker = " **" if row["subject"] in top5_dif else ""
    report += f"| {i} | {row['subject']} | {row['pct_above_020']:.1f}% | {row['pct_c']:.1f}%{marker} |\n"

report += f"""
### Highest DIF C% Subjects
| Rank | Subject | DIF C% | % |Q3|>0.20 |
|------|---------|--------|------------|
"""

for i, (_, row) in enumerate(merged.nlargest(5, "pct_c").iterrows(), 1):
    marker = " **" if row["subject"] in top5_ld else ""
    report += f"| {i} | {row['subject']} | {row['pct_c']:.1f}% | {row['pct_above_020']:.1f}%{marker} |\n"

report += f"""
**Top-5 overlap**: {', '.join(sorted(overlap)) if overlap else 'None'}

### Lowest LD Subjects (% |Q3| > 0.20)
| Rank | Subject | % |Q3|>0.20 | DIF C% |
|------|---------|------------|--------|
"""

for i, (_, row) in enumerate(merged.nsmallest(5, "pct_above_020").iterrows(), 1):
    report += f"| {i} | {row['subject']} | {row['pct_above_020']:.1f}% | {row['pct_c']:.1f}% |\n"

report += f"""
### Lowest DIF C% Subjects
| Rank | Subject | DIF C% | % |Q3|>0.20 |
|------|---------|--------|------------|
"""

for i, (_, row) in enumerate(merged.nsmallest(5, "pct_c").iterrows(), 1):
    report += f"| {i} | {row['subject']} | {row['pct_c']:.1f}% | {row['pct_above_020']:.1f}% |\n"

report += f"""
**Bottom-5 overlap**: {', '.join(sorted(overlap_bot)) if overlap_bot else 'None'}

## Supplementary: Accuracy Change Correlations

| Pair | Pearson r | p-value |
|------|-----------|---------|
| % |Q3|>0.20 vs Δacc | {r3:.4f} | {p3:.4e} |
| DIF C% vs Δacc | {r4:.4f} | {p4:.4e} |

## Interpretation

"""

if abs(r_pearson) < 0.3 and p_pearson > 0.05:
    report += """The correlation between LD severity and DIF prevalence across subjects is **weak and non-significant**. High local dependence within a domain does not predict high DIF rates. This suggests LD and DIF capture **orthogonal item quality dimensions**: LD reflects within-domain item redundancy (items that share variance beyond the latent trait), while DIF reflects differential functioning across model cohorts (temporal sensitivity to training data shifts). Domains can be internally redundant without being temporally unstable, and vice versa.\n"""
elif abs(r_pearson) >= 0.3 and p_pearson < 0.05:
    report += f"""There is a **moderate {'positive' if r_pearson > 0 else 'negative'} correlation** (r={r_pearson:.3f}) between LD severity and DIF prevalence. Domains with higher local dependence tend to {'also' if r_pearson > 0 else 'not'} show higher DIF rates. This suggests {'shared underlying causes such as item clustering or narrow skill requirements that make items both redundant and sensitive to training distribution shifts' if r_pearson > 0 else 'a dissociation where redundant domains are actually more stable across cohorts'}.\n"""
else:
    report += f"""The correlation is r={r_pearson:.3f} (p={p_pearson:.4f}). While the direction suggests {'some link' if r_pearson > 0 else 'divergence'} between LD and DIF, the evidence is {'borderline' if p_pearson < 0.1 else 'weak'}. Larger domain-level sample sizes would be needed for a definitive conclusion.\n"""

report += f"""
## Files
- `ld_dif_merged.csv`: Full merged table (57 subjects × 7 columns), sorted by LD descending
- `ld_dif_cross_analysis.py`: Analysis script
"""

(OUT / "ld_dif_cross_report.md").write_text(report)
print(f"Saved report to {OUT / 'ld_dif_cross_report.md'}")
