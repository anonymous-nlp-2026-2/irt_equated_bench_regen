"""R14 SF9: Cook's D outlier check for GSM1K DIF correlation."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

ROOT = Path(".")
DATA = ROOT / "artifacts/plan_004/gsm1k_dif_correlation.csv"
OUT = ROOT / "artifacts/r14_sf9_gsm1k_cooks"

df = pd.read_csv(DATA)
# Drop any empty rows
df = df.dropna(subset=["gap", "dif_advantage"])
N = len(df)
print(f"N = {N} models")

# --- 1. OLS regression: dif_advantage ~ gap ---
X = sm.add_constant(df["gap"].values)
y = df["dif_advantage"].values
model = sm.OLS(y, X).fit()
print(f"\nOLS: beta_gap = {model.params[1]:.4f}, R² = {model.rsquared:.4f}")

# --- 2. Cook's D ---
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
df["cooks_d"] = cooks_d

threshold = 4.0 / N
print(f"\nCook's D threshold (4/N): {threshold:.4f}")
print(f"Max Cook's D: {cooks_d.max():.4f} ({df.loc[cooks_d.argmax(), 'model_name_gsm1k']})")
print(f"Median Cook's D: {np.median(cooks_d):.4f}")
n_outliers = (cooks_d > threshold).sum()
print(f"Models with Cook's D > threshold: {n_outliers}")

outlier_df = df[df["cooks_d"] > threshold].sort_values("cooks_d", ascending=False)
if len(outlier_df) > 0:
    print("\nOutliers:")
    for _, row in outlier_df.iterrows():
        print(f"  {row['model_name_gsm1k']}: Cook's D = {row['cooks_d']:.4f}, gap = {row['gap']:.3f}, dif_advantage = {row['dif_advantage']:.4f}")

# --- 3. Leave-one-out analysis ---
loo_results = []
for i in range(N):
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    r_loo, p_loo = stats.pearsonr(df["gap"].values[mask], df["dif_advantage"].values[mask])
    loo_results.append({"idx": i, "loo_r": r_loo, "loo_p": p_loo})

loo_df = pd.DataFrame(loo_results)
df["loo_r"] = loo_df["loo_r"].values
df["loo_p"] = loo_df["loo_p"].values

# Full correlation
r_full, p_full = stats.pearsonr(df["gap"], df["dif_advantage"])
print(f"\nFull correlation: r = {r_full:.4f}, p = {p_full:.4f}")

# Most influential (largest |Δr|)
df["delta_r"] = df["loo_r"] - r_full
most_influential_idx = df["delta_r"].abs().idxmax()
mi = df.loc[most_influential_idx]
print(f"\nMost influential model (max |Δr|):")
print(f"  {mi['model_name_gsm1k']}: Δr = {mi['delta_r']:+.4f}, LOO r = {mi['loo_r']:.4f}, LOO p = {mi['loo_p']:.4f}")

print(f"\nLOO r range: [{df['loo_r'].min():.4f}, {df['loo_r'].max():.4f}]")
print(f"LOO p range: [{df['loo_p'].min():.4f}, {df['loo_p'].max():.4f}]")
all_sig = (df["loo_p"] < 0.05).all()
print(f"All LOO p < 0.05? {all_sig}")

# --- 4. Remove all outliers and recompute ---
if n_outliers > 0:
    clean_mask = df["cooks_d"] <= threshold
    r_clean, p_clean = stats.pearsonr(df.loc[clean_mask, "gap"], df.loc[clean_mask, "dif_advantage"])
    print(f"\nAfter removing {n_outliers} outlier(s): r = {r_clean:.4f}, p = {p_clean:.4f}, N = {clean_mask.sum()}")
else:
    r_clean, p_clean = r_full, p_full
    print("\nNo outliers to remove.")

# --- 5. Write CSV ---
out_csv = OUT / "cooks_d_results.csv"
df[["model_id", "model_name_gsm1k", "gap", "dif_advantage", "cooks_d", "loo_r", "loo_p"]].to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")

# --- 6. Write report ---
robust = all_sig and (n_outliers == 0 or p_clean < 0.05)

report = f"""# Cook's D Outlier Analysis — GSM1K DIF Correlation

## Setup

- **Regression**: `dif_advantage ~ gap` (OLS)
- **N**: {N} models
- **Full correlation**: r = {r_full:.3f}, p = {p_full:.4f}
- **Cook's D threshold**: 4/N = {threshold:.3f}

## Cook's D Distribution

| Statistic | Value |
|-----------|-------|
| Max | {cooks_d.max():.4f} ({df.loc[cooks_d.argmax(), 'model_name_gsm1k']}) |
| Median | {np.median(cooks_d):.4f} |
| Mean | {cooks_d.mean():.4f} |
| Models > threshold | {n_outliers} / {N} |

"""

if n_outliers > 0:
    report += "## Influential Outliers (Cook's D > 4/N)\n\n"
    report += "| Model | Cook's D | gap | dif_advantage |\n"
    report += "|-------|----------|-----|---------------|\n"
    for _, row in outlier_df.iterrows():
        report += f"| {row['model_name_gsm1k']} | {row['cooks_d']:.4f} | {row['gap']:.3f} | {row['dif_advantage']:.4f} |\n"
    report += "\n"
else:
    report += "## Influential Outliers\n\nNo models exceed the Cook's D threshold.\n\n"

report += f"""## Leave-One-Out Analysis

| Statistic | Value |
|-----------|-------|
| LOO r range | [{df['loo_r'].min():.3f}, {df['loo_r'].max():.3f}] |
| LOO p range | [{df['loo_p'].min():.4f}, {df['loo_p'].max():.4f}] |
| All LOO p < 0.05 | {'Yes' if all_sig else 'No'} |

**Most influential model**: {mi['model_name_gsm1k']}
- Removing it shifts r from {r_full:.3f} to {mi['loo_r']:.3f} (Δr = {mi['delta_r']:+.3f})
- LOO p = {mi['loo_p']:.4f}

"""

# Top 5 by |Δr|
report += "### Top 5 by |Δr|\n\n"
report += "| Model | Δr | LOO r | LOO p |\n"
report += "|-------|----|-------|-------|\n"
top5 = df.reindex(df["delta_r"].abs().sort_values(ascending=False).index).head(5)
for _, row in top5.iterrows():
    report += f"| {row['model_name_gsm1k']} | {row['delta_r']:+.4f} | {row['loo_r']:.3f} | {row['loo_p']:.4f} |\n"
report += "\n"

if n_outliers > 0:
    report += f"""## Sensitivity: Removing All Outliers

- After removing {n_outliers} outlier(s): r = {r_clean:.3f}, p = {p_clean:.4f}, N = {clean_mask.sum()}
- Correlation {'remains significant' if p_clean < 0.05 else 'becomes non-significant'} (p {'<' if p_clean < 0.05 else '>'} 0.05)

"""

report += f"""## Conclusion

The correlation r = {r_full:.3f} (p = {p_full:.4f}) is **{'robust' if robust else 'NOT robust'}** to outlier removal.
"""

if robust:
    if n_outliers == 0:
        report += "No observations exceed the Cook's D threshold (4/N), and no single model removal changes significance.\n"
    else:
        report += f"Although {n_outliers} model(s) exceed the Cook's D threshold, removing them {'strengthens' if abs(r_clean) > abs(r_full) else 'weakens but preserves'} the correlation (r = {r_clean:.3f}, p = {p_clean:.4f}).\n"
else:
    report += "Outlier removal changes the significance of the correlation. The finding should be interpreted with caution.\n"

out_report = OUT / "cooks_d_report.md"
out_report.write_text(report)
print(f"Wrote {out_report}")
