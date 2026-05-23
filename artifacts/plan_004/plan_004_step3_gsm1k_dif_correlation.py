"""
Plan 004 Step 3: GSM1K Contamination Gap × DIF Flag Correlation
Core hypothesis: models with larger GSM8K-GSM1K gap show disproportionate
advantage on DIF-C items (suspected contaminated items).
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE

# ── 1. Load data ──────────────────────────────────────────────────────────────

dif_df = pd.read_csv(ARTIFACTS / 'gsm8k_dif_results.csv')
gsm1k_df = pd.read_csv(ARTIFACTS / 'gsm1k_model_accuracy.csv')
resp_df = pd.read_csv(ARTIFACTS.parent / 'metabench_data' / 'benchmark-data' / 'gsm8k.csv')
meta_df = pd.read_csv(ARTIFACTS.parent / 'model_metadata.csv')

# ── 2. DIF-C items (standard matching) ────────────────────────────────────────

dif_c_items = dif_df[(dif_df['matching_type'] == 'standard') & (dif_df['ets_class'] == 'C')]['item_id'].values
dif_non_c_items = dif_df[(dif_df['matching_type'] == 'standard') & (dif_df['ets_class'].isin(['A', 'B']))]['item_id'].values
print(f"DIF-C items: {len(dif_c_items)}, Non-DIF (A+B) items: {len(dif_non_c_items)}")

# ── 3. Matched models (have metabench ID + contamination gap) ─────────────────

matched = gsm1k_df[
    (gsm1k_df['model_name_metabench'] != 'unmatched') &
    (gsm1k_df['contamination_gap'].notna())
].copy()
print(f"Matched models with gap: {len(matched)}")

# ── 4. Pivot response matrix ─────────────────────────────────────────────────

resp_df['correct'] = resp_df['correct'].map({'True': 1, 'False': 0, True: 1, False: 0}).astype(int)
resp_pivot = resp_df.pivot_table(index='source', columns='item', values='correct', aggfunc='first')
print(f"Response matrix: {resp_pivot.shape[0]} models × {resp_pivot.shape[1]} items")

# ── 5. Compute DIF advantage per matched model ───────────────────────────────

dif_c_cols = [c for c in dif_c_items if c in resp_pivot.columns]
non_c_cols = [c for c in dif_non_c_items if c in resp_pivot.columns]
print(f"DIF-C items in response matrix: {len(dif_c_cols)}, Non-DIF items: {len(non_c_cols)}")

records = []
for _, row in matched.iterrows():
    model_id = row['model_name_metabench']
    if model_id not in resp_pivot.index:
        print(f"  SKIP (not in response matrix): {model_id}")
        continue
    model_resp = resp_pivot.loc[model_id]
    acc_dif_c = model_resp[dif_c_cols].mean()
    acc_non_dif = model_resp[non_c_cols].mean()
    dif_advantage = acc_dif_c - acc_non_dif

    fam = meta_df.loc[meta_df['model_name'] == model_id, 'family'].values
    family = fam[0] if len(fam) > 0 else 'unknown'

    records.append({
        'model_id': model_id,
        'model_name_gsm1k': row['model_name_gsm1k'],
        'model_family': family,
        'gsm8k_acc': row['gsm8k_accuracy'],
        'gsm1k_acc': row['gsm1k_accuracy'],
        'gap': row['contamination_gap'],
        'acc_dif_c': round(acc_dif_c, 6),
        'acc_non_dif': round(acc_non_dif, 6),
        'dif_advantage': round(dif_advantage, 6),
    })

result_df = pd.DataFrame(records).sort_values('gap', ascending=False).reset_index(drop=True)
result_df.to_csv(ARTIFACTS / 'gsm1k_dif_correlation.csv', index=False)
print(f"\nFinal matched models: {len(result_df)}")
print(result_df[['model_name_gsm1k', 'model_family', 'gap', 'dif_advantage']].to_string())

# ── 6. Main correlations ─────────────────────────────────────────────────────

gap = result_df['gap'].values
dif_adv = result_df['dif_advantage'].values

r_pearson, p_pearson = stats.pearsonr(gap, dif_adv)
r_spearman, p_spearman = stats.spearmanr(gap, dif_adv)
print(f"\n=== Main Analysis ===")
print(f"Pearson  r={r_pearson:.4f}, p={p_pearson:.4f}")
print(f"Spearman r={r_spearman:.4f}, p={p_spearman:.4f}")

# ── 7. Partial correlation (controlling for GSM8K overall accuracy) ───────────

gsm8k_acc = result_df['gsm8k_acc'].values

# Residualize dif_advantage on gsm8k_acc
slope_da, intercept_da, _, _, _ = stats.linregress(gsm8k_acc, dif_adv)
resid_da = dif_adv - (slope_da * gsm8k_acc + intercept_da)

# Residualize gap on gsm8k_acc
slope_gap, intercept_gap, _, _, _ = stats.linregress(gsm8k_acc, gap)
resid_gap = gap - (slope_gap * gsm8k_acc + intercept_gap)

r_partial, p_partial = stats.pearsonr(resid_gap, resid_da)
print(f"\n=== Partial Correlation (controlling GSM8K accuracy) ===")
print(f"Partial r={r_partial:.4f}, p={p_partial:.4f}")

# ── 8. Permutation test ──────────────────────────────────────────────────────

np.random.seed(42)
all_items = list(dif_c_cols) + list(non_c_cols)
n_dif_c = len(dif_c_cols)
n_perm = 1000

perm_rs = []
for i in range(n_perm):
    rand_items = np.random.choice(all_items, size=n_dif_c, replace=False)
    rest_items = [it for it in all_items if it not in rand_items]
    rand_advs = []
    for _, row in result_df.iterrows():
        model_id = row['model_id']
        model_resp = resp_pivot.loc[model_id]
        acc_rand = model_resp[rand_items].mean()
        acc_rest = model_resp[rest_items].mean()
        rand_advs.append(acc_rand - acc_rest)
    r_perm, _ = stats.pearsonr(result_df['gap'].values, rand_advs)
    perm_rs.append(r_perm)

perm_rs = np.array(perm_rs)
perm_p = np.mean(np.abs(perm_rs) >= np.abs(r_pearson))
print(f"\n=== Permutation Test (n={n_perm}) ===")
print(f"Observed r={r_pearson:.4f}")
print(f"Null distribution: mean={perm_rs.mean():.4f}, std={perm_rs.std():.4f}")
print(f"Null 95% CI: [{np.percentile(perm_rs, 2.5):.4f}, {np.percentile(perm_rs, 97.5):.4f}]")
print(f"Empirical p-value (two-tailed): {perm_p:.4f}")

# ── 9. Outlier analysis (Cook's distance + leave-one-out) ────────────────────

from numpy.linalg import lstsq

X = np.column_stack([np.ones(len(gap)), gap])
beta, _, _, _ = lstsq(X, dif_adv, rcond=None)
y_hat = X @ beta
residuals = dif_adv - y_hat
n = len(gap)
p_param = 2
mse = np.sum(residuals**2) / (n - p_param)
H = X @ np.linalg.inv(X.T @ X) @ X.T
h = np.diag(H)
cooks_d = (residuals**2 / (p_param * mse)) * (h / (1 - h)**2)

print(f"\n=== Outlier Analysis (Cook's Distance) ===")
for i in np.argsort(-cooks_d)[:5]:
    print(f"  {result_df.iloc[i]['model_name_gsm1k']}: Cook's D={cooks_d[i]:.4f}, gap={gap[i]:.3f}, dif_adv={dif_adv[i]:.4f}")

# Leave-one-out
print(f"\n=== Leave-One-Out Analysis ===")
loo_results = []
for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    r_loo, _ = stats.pearsonr(gap[mask], dif_adv[mask])
    loo_results.append((result_df.iloc[i]['model_name_gsm1k'], r_loo, r_pearson - r_loo))

loo_results.sort(key=lambda x: abs(x[2]), reverse=True)
for name, r_loo, delta in loo_results[:5]:
    print(f"  Remove {name}: r={r_loo:.4f} (Δr={delta:+.4f})")

# ── 10. Model family analysis ────────────────────────────────────────────────

print(f"\n=== Model Family Summary ===")
family_stats = result_df.groupby('model_family').agg(
    n=('gap', 'count'),
    mean_gap=('gap', 'mean'),
    mean_dif_adv=('dif_advantage', 'mean'),
).sort_values('mean_gap', ascending=False)
print(family_stats.to_string())

# ── 11. Scatter plot ─────────────────────────────────────────────────────────

FAMILY_COLORS = {
    'phi-2': '#e41a1c', 'phi-3': '#e41a1c',
    'gemma': '#377eb8', 'codegemma': '#377eb8',
    'llama-3': '#4daf4a', 'codellama': '#4daf4a',
    'mixtral': '#984ea3',
    'falcon': '#ff7f00',
    'yi': '#a65628',
    'deepseek': '#f781bf',
    'smaug': '#999999',
    'dbrx': '#666666',
}

def get_color(fam):
    for key, col in FAMILY_COLORS.items():
        if key in str(fam).lower():
            return col
    return '#333333'

def get_display_family(fam):
    fam = str(fam).lower()
    if 'phi' in fam: return 'Phi'
    if 'gemma' in fam or 'codegemma' in fam: return 'Gemma'
    if 'llama' in fam or 'codellama' in fam: return 'Llama/CodeLlama'
    if 'mixtral' in fam: return 'Mixtral'
    if 'falcon' in fam: return 'Falcon'
    if 'yi' in fam: return 'Yi'
    if 'deepseek' in fam: return 'DeepSeek'
    if 'mistral' in fam and 'mixtral' not in fam: return 'Mistral'
    return 'Other'

result_df['display_family'] = result_df['model_family'].apply(get_display_family)
result_df['color'] = result_df['model_family'].apply(get_color)

fig, ax = plt.subplots(figsize=(10, 7))

# Plot by family for legend
for fam in sorted(result_df['display_family'].unique()):
    sub = result_df[result_df['display_family'] == fam]
    ax.scatter(sub['gap'], sub['dif_advantage'], c=sub['color'].values,
               s=60, label=fam, zorder=3, edgecolors='white', linewidth=0.5)

# Regression line + CI band
slope, intercept, r_val, p_val, se = stats.linregress(gap, dif_adv)
x_line = np.linspace(gap.min() - 0.005, gap.max() + 0.005, 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'k-', linewidth=1.5, alpha=0.7)

# 95% CI band
x_mean = gap.mean()
ss_x = np.sum((gap - x_mean)**2)
se_fit = np.sqrt(mse * (1/n + (x_line - x_mean)**2 / ss_x))
t_crit = stats.t.ppf(0.975, n - 2)
ax.fill_between(x_line, y_line - t_crit * se_fit, y_line + t_crit * se_fit,
                alpha=0.15, color='gray')

# Label each point
for _, row in result_df.iterrows():
    short_name = row['model_name_gsm1k']
    if len(short_name) > 20:
        short_name = short_name[:18] + '..'
    ax.annotate(short_name, (row['gap'], row['dif_advantage']),
                fontsize=6.5, alpha=0.8,
                xytext=(4, 4), textcoords='offset points')

ax.set_xlabel('Contamination Gap (GSM8K − GSM1K accuracy)', fontsize=12)
ax.set_ylabel('DIF Advantage (acc on DIF-C − acc on non-DIF)', fontsize=12)
ax.set_title(f'GSM1K Contamination Gap vs. DIF-C Item Advantage\n'
             f'Pearson r={r_pearson:.3f} (p={p_pearson:.3f}), '
             f'Spearman ρ={r_spearman:.3f} (p={p_spearman:.3f})',
             fontsize=12)
ax.legend(loc='best', fontsize=9, framealpha=0.8)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()

fig.savefig(ARTIFACTS / 'fig_gsm1k_dif_correlation.png', dpi=200, bbox_inches='tight')
fig.savefig(ARTIFACTS / 'fig_gsm1k_dif_correlation.pdf', bbox_inches='tight')
print(f"\nFigure saved.")

# ── 12. Permutation null distribution histogram ──────────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.hist(perm_rs, bins=40, color='#cccccc', edgecolor='gray', alpha=0.8, density=True)
ax2.axvline(r_pearson, color='red', linewidth=2, label=f'Observed r={r_pearson:.3f}')
ax2.axvline(np.percentile(perm_rs, 2.5), color='blue', linewidth=1, linestyle='--', label='Null 95% CI')
ax2.axvline(np.percentile(perm_rs, 97.5), color='blue', linewidth=1, linestyle='--')
ax2.set_xlabel('Pearson r (random item sets vs. gap)', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title(f'Permutation Test: Null Distribution (n={n_perm})\nEmpirical p={perm_p:.3f}', fontsize=11)
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(ARTIFACTS / 'fig_gsm1k_dif_perm_null.png', dpi=200, bbox_inches='tight')
print(f"Permutation null figure saved.")

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Models analyzed: {len(result_df)}")
print(f"DIF-C items: {len(dif_c_cols)}, Non-DIF items: {len(non_c_cols)}")
print(f"Pearson  r={r_pearson:.4f}, p={p_pearson:.4f}")
print(f"Spearman ρ={r_spearman:.4f}, p={p_spearman:.4f}")
print(f"Partial  r={r_partial:.4f}, p={p_partial:.4f} (controlling GSM8K acc)")
print(f"Permutation p={perm_p:.4f} (two-tailed, n={n_perm})")
if abs(r_pearson) > 0.3:
    print(f">>> BREAKTHROUGH: |r| > 0.3 — DIF flags correlate with contamination")
elif abs(r_pearson) < 0.1:
    print(f">>> WEAK: |r| < 0.1 — DIF advantage appears driven by capability, not contamination")
else:
    print(f">>> MODERATE: 0.1 ≤ |r| ≤ 0.3 — suggestive but not conclusive")
