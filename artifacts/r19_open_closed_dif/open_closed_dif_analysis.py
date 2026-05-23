#!/usr/bin/env python3
"""
R19 Exp 3 — Open vs Closed-Source DIF analysis.

Classifies METABENCH models as open-source vs closed-source/API-only,
then runs temporal MH-DIF (2023 vs 2024) per group.

Finding: METABENCH contains 0 closed-source models (all from HF Open LLM
Leaderboard). Fallback analysis uses distilled-from-API vs non-distilled split.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist, spearmanr
from pathlib import Path
import warnings
import re

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("artifacts")
OUT = BASE / "r19_open_closed_dif"
OUT.mkdir(exist_ok=True)

N_STRATA = 5
SEED = 42

# ════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════
rm = np.load(BASE / "response_matrix.npz")
X = sparse.csr_matrix((rm['data'], rm['indices'], rm['indptr']),
                       shape=tuple(rm['shape']))
idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
model_names = list(idx['model_names'])
item_ids = list(idx['item_ids'])
meta = pd.read_csv(BASE / "model_metadata.csv")

print(f"Loaded: {X.shape[0]} models × {X.shape[1]} items")

# ════════════════════════════════════════════════════════════════════════
#  MODEL CLASSIFICATION: open vs closed-source
# ════════════════════════════════════════════════════════════════════════

# True closed-source/API-only: models NOT hosted on HF with public weights.
# In METABENCH (HF Open LLM Leaderboard data), all models have public weights.
# We detect closed-source by checking for API-only model names without HF org prefixes.
CLOSED_SOURCE_EXACT = {
    # OpenAI API-only models (if present)
    'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini',
    'gpt-3.5-turbo', 'o1-mini', 'o1-preview', 'o3-mini',
    # Would also match claude-3-*, gemini-*, etc. if present
}

def classify_model(name):
    """Classify a model as open-source, closed-source, or distilled-from-api."""
    name_lower = name.lower()
    base = name.split('/')[-1] if '/' in name else name

    # Check exact matches for API-only models
    if base.lower() in CLOSED_SOURCE_EXACT:
        return 'closed'

    # API-only models typically have NO HF org/ prefix
    # and match known API provider patterns
    if '/' not in name:
        for p in ['gpt-4', 'gpt-3.5', 'claude-', 'gemini-', 'palm-']:
            if p in name_lower:
                return 'closed'

    # Everything on HF with org/ prefix = open-source (public weights)
    return 'open'


def is_distilled_from_api(name):
    """Detect models fine-tuned on closed-source model outputs."""
    name_lower = name.lower()
    distill_signals = [
        'gpt-4', 'gpt-3.5', 'gpt4', 'gpt35',
        'chatgpt', 'claude-distil', 'orca',
        'alpaca-gpt4', 'sharegpt',
    ]
    for sig in distill_signals:
        if sig in name_lower:
            return True
    return False


# Classify all models
classifications = []
for m in model_names:
    source = classify_model(m)
    distilled = is_distilled_from_api(m)
    classifications.append({
        'model_name': m,
        'source_type': source,
        'distilled_from_api': distilled
    })

class_df = pd.DataFrame(classifications)
class_df = class_df.merge(meta[['model_name', 'cohort_temporal']], on='model_name', how='left')

# Save classification
class_df.to_csv(OUT / "model_source_classification.csv", index=False)

n_open = (class_df['source_type'] == 'open').sum()
n_closed = (class_df['source_type'] == 'closed').sum()
n_distilled = class_df['distilled_from_api'].sum()

print(f"\n=== Source Classification ===")
print(f"Open-source:    {n_open}")
print(f"Closed-source:  {n_closed}")
print(f"Distilled from API: {n_distilled}")

# Cross-tab with cohort
print("\n=== Source × Cohort ===")
print(pd.crosstab(class_df['source_type'], class_df['cohort_temporal'], margins=True))
print("\n=== Distilled × Cohort ===")
print(pd.crosstab(class_df['distilled_from_api'], class_df['cohort_temporal'], margins=True))


# ════════════════════════════════════════════════════════════════════════
#  PER-ITEM MH-DIF (returns item-level stats)
# ════════════════════════════════════════════════════════════════════════
def mh_dif_per_item(X_ref, X_foc, n_strata=N_STRATA):
    """MH-DIF returning per-item delta_mh, p-value, and C classification."""
    n_ref, J = X_ref.shape
    n_foc = X_foc.shape[0]

    tot_ref = X_ref.sum(axis=1).astype(np.float64)
    tot_foc = X_foc.sum(axis=1).astype(np.float64)
    s_all = np.concatenate([tot_ref, tot_foc])

    edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
    n_bins = len(edges) - 1
    if n_bins < 1:
        return pd.DataFrame({
            'delta_mh': np.zeros(J),
            'p_value': np.ones(J),
            'is_c': np.zeros(J, dtype=bool),
            'difficulty': np.zeros(J),
        })

    k_ref = np.digitize(tot_ref, edges[1:-1])
    k_foc = np.digitize(tot_foc, edges[1:-1])

    n1 = np.zeros(n_bins, dtype=np.float64)
    n0 = np.zeros(n_bins, dtype=np.float64)
    A = np.zeros((n_bins, J), dtype=np.float64)
    C = np.zeros((n_bins, J), dtype=np.float64)

    for k in range(n_bins):
        mr = (k_ref == k)
        mf = (k_foc == k)
        n1[k] = mr.sum()
        n0[k] = mf.sum()
        if mr.any():
            A[k] = X_ref[mr].sum(axis=0).astype(np.float64)
        if mf.any():
            C[k] = X_foc[mf].sum(axis=0).astype(np.float64)

    B = n1[:, None] - A
    D = n0[:, None] - C
    Nk = (n1 + n0)[:, None]
    m1 = A + C
    m0 = B + D

    valid = ((n1[:, None] > 0) & (n0[:, None] > 0) &
             (Nk > 1) & (m1 > 0) & (m0 > 0))

    Rj = np.where(valid, A * D / Nk, 0).sum(axis=0)
    Sj = np.where(valid, B * C / Nk, 0).sum(axis=0)

    finite_mask = (Rj > 0) & (Sj > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_j = np.where(finite_mask, -2.35 * np.log(Rj / Sj), 0)

    EA = np.where(valid, n1[:, None] * m1 / Nk, 0)
    sum_diff = np.where(valid, A - EA, 0).sum(axis=0)

    Nk_m1 = np.where(Nk > 1, Nk - 1, 1)
    sum_var = np.where(
        valid,
        n1[:, None] * n0[:, None] * m1 * m0 / (Nk ** 2 * Nk_m1),
        0,
    ).sum(axis=0)

    chi2_j = np.where(
        sum_var > 0,
        np.maximum(np.abs(sum_diff) - 0.5, 0) ** 2 / np.maximum(sum_var, 1e-30),
        0,
    )
    pval_j = np.where(sum_var > 0, 1 - chi2_dist.cdf(chi2_j, df=1), 1.0)

    is_c = finite_mask & (np.abs(delta_j) >= 1.5) & (pval_j < 0.05)

    # item difficulty = overall p-value (proportion correct)
    all_X = np.vstack([X_ref, X_foc])
    difficulty = all_X.mean(axis=0).astype(np.float64)

    return pd.DataFrame({
        'delta_mh': delta_j,
        'p_value': pval_j,
        'is_c': is_c,
        'difficulty': difficulty,
    })


def run_temporal_dif(group_mask, group_name):
    """Run temporal MH-DIF (2023 vs 2024) for a subset of models."""
    cohorts = class_df['cohort_temporal'].values

    ref_mask = group_mask & (cohorts == '2023')
    foc_mask = group_mask & (cohorts == '2024')

    n_ref = ref_mask.sum()
    n_foc = foc_mask.sum()

    print(f"\n{'='*60}")
    print(f"  {group_name}: N_ref(2023)={n_ref}, N_foc(2024)={n_foc}")
    print(f"{'='*60}")

    if n_ref < 5 or n_foc < 5:
        print(f"  SKIPPED: insufficient models (need >= 5 per cohort)")
        return None

    X_dense = X.toarray()
    X_ref = X_dense[ref_mask]
    X_foc = X_dense[foc_mask]

    results = mh_dif_per_item(X_ref, X_foc)
    results['item_id'] = item_ids

    J = len(item_ids)
    c_pct = results['is_c'].sum() / J * 100
    n_c = results['is_c'].sum()

    # Direction of DIF among C items
    c_items = results[results['is_c']]
    n_favor_ref = (c_items['delta_mh'] > 0).sum()  # favors 2023
    n_favor_foc = (c_items['delta_mh'] < 0).sum()  # favors 2024

    # Correlation between difficulty and delta_MH
    finite = results['delta_mh'] != 0
    if finite.sum() > 10:
        rho, rho_p = spearmanr(results.loc[finite, 'difficulty'],
                               results.loc[finite, 'delta_mh'])
    else:
        rho, rho_p = np.nan, np.nan

    print(f"  C%: {c_pct:.1f}% ({n_c}/{J})")
    print(f"  Direction: {n_favor_ref} favor 2023, {n_favor_foc} favor 2024")
    print(f"  ρ(difficulty, Δ_MH): {rho:.3f} (p={rho_p:.4f})")

    return {
        'group': group_name,
        'n_ref': n_ref,
        'n_foc': n_foc,
        'c_pct': c_pct,
        'n_c': n_c,
        'n_favor_ref': n_favor_ref,
        'n_favor_foc': n_favor_foc,
        'rho': rho,
        'rho_p': rho_p,
        'item_results': results,
    }


# ════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ════════════════════════════════════════════════════════════════════════
results_all = {}

# 1. Open-source full set (all models are open-source)
open_mask = (class_df['source_type'] == 'open').values
r = run_temporal_dif(open_mask, "All Models (Open-Source)")
if r:
    results_all['all_open'] = r

# 2. Closed-source (expected: N=0)
closed_mask = (class_df['source_type'] == 'closed').values
if closed_mask.sum() > 0:
    r = run_temporal_dif(closed_mask, "Closed-Source")
    if r:
        results_all['closed'] = r
else:
    print(f"\n{'='*60}")
    print(f"  Closed-Source: N=0 — METABENCH has no API-only models")
    print(f"  (all data from HF Open LLM Leaderboard)")
    print(f"{'='*60}")

# 3. Fallback: Distilled-from-API vs Non-distilled
distilled_mask = class_df['distilled_from_api'].values
nondistilled_mask = ~class_df['distilled_from_api'].values

r = run_temporal_dif(distilled_mask, "Distilled-from-API")
if r:
    results_all['distilled'] = r

r = run_temporal_dif(nondistilled_mask, "Non-Distilled")
if r:
    results_all['non_distilled'] = r

# ════════════════════════════════════════════════════════════════════════
#  DIF ITEM OVERLAP (Jaccard) — distilled vs non-distilled
# ════════════════════════════════════════════════════════════════════════
if 'distilled' in results_all and 'non_distilled' in results_all:
    c_dist = set(results_all['distilled']['item_results']
                 .loc[results_all['distilled']['item_results']['is_c'], 'item_id'])
    c_nondist = set(results_all['non_distilled']['item_results']
                    .loc[results_all['non_distilled']['item_results']['is_c'], 'item_id'])

    if len(c_dist | c_nondist) > 0:
        jaccard = len(c_dist & c_nondist) / len(c_dist | c_nondist)
    else:
        jaccard = 0.0

    print(f"\n=== DIF Item Overlap ===")
    print(f"Distilled C items: {len(c_dist)}")
    print(f"Non-distilled C items: {len(c_nondist)}")
    print(f"Intersection: {len(c_dist & c_nondist)}")
    print(f"Jaccard: {jaccard:.3f}")

# ════════════════════════════════════════════════════════════════════════
#  SIZE-MATCHED CONTROL
# ════════════════════════════════════════════════════════════════════════
if 'distilled' in results_all:
    n_dist_ref = results_all['distilled']['n_ref']
    n_dist_foc = results_all['distilled']['n_foc']

    rng = np.random.default_rng(SEED)
    cohorts = class_df['cohort_temporal'].values
    nondist_ref_idx = np.where(nondistilled_mask & (cohorts == '2023'))[0]
    nondist_foc_idx = np.where(nondistilled_mask & (cohorts == '2024'))[0]

    if len(nondist_ref_idx) >= n_dist_ref and len(nondist_foc_idx) >= n_dist_foc:
        n_boot = 50
        control_c_pcts = []
        X_dense = X.toarray()

        print(f"\n=== Size-Matched Control (N={n_boot} resamples) ===")
        for i in range(n_boot):
            sub_ref = rng.choice(nondist_ref_idx, size=n_dist_ref, replace=False)
            sub_foc = rng.choice(nondist_foc_idx, size=n_dist_foc, replace=False)
            res = mh_dif_per_item(X_dense[sub_ref], X_dense[sub_foc])
            control_c_pcts.append(res['is_c'].sum() / len(item_ids) * 100)

        ctrl_mean = np.mean(control_c_pcts)
        ctrl_std = np.std(control_c_pcts)
        ctrl_ci = np.percentile(control_c_pcts, [2.5, 97.5])
        dist_c = results_all['distilled']['c_pct']

        print(f"Distilled C%: {dist_c:.1f}%")
        print(f"Control C% (mean ± sd): {ctrl_mean:.1f}% ± {ctrl_std:.1f}%")
        print(f"Control 95% CI: [{ctrl_ci[0]:.1f}%, {ctrl_ci[1]:.1f}%]")
        print(f"Distilled {'OUTSIDE' if dist_c < ctrl_ci[0] or dist_c > ctrl_ci[1] else 'within'} control CI")


# ════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ════════════════════════════════════════════════════════════════════════
report_lines = [
    "# R19 Exp 3 — Open vs Closed-Source DIF\n",
    "## Key Finding\n",
    "METABENCH contains **zero closed-source/API-only models**. All 5,227 models",
    "come from the HuggingFace Open LLM Leaderboard, which by definition only",
    "evaluates models with publicly available weights.\n",
    "Models with names containing 'gpt-4' or 'gpt-3.5' are open-source models",
    "fine-tuned on data distilled from those API models, not the API models themselves.\n",
    "## Model Classification\n",
    f"| Category | N |",
    f"|----------|---|",
    f"| Open-source | {n_open} |",
    f"| Closed-source | {n_closed} |",
    f"| Distilled-from-API | {n_distilled} |",
    f"| Non-distilled | {len(model_names) - n_distilled} |",
    "",
]

# Cohort breakdown
report_lines += [
    "\n## Cohort Breakdown\n",
    "| Group | 2023 | 2024 | Other |",
    "|-------|------|------|-------|",
]
for label, mask in [('All Open', open_mask), ('Distilled', distilled_mask),
                    ('Non-distilled', nondistilled_mask)]:
    n23 = (mask & (class_df['cohort_temporal'].values == '2023')).sum()
    n24 = (mask & (class_df['cohort_temporal'].values == '2024')).sum()
    nother = (mask & (class_df['cohort_temporal'].values == 'other')).sum()
    report_lines.append(f"| {label} | {n23} | {n24} | {nother} |")

# DIF results
report_lines += ["\n## Temporal MH-DIF Results (2023 → 2024)\n"]

for key, label in [('all_open', 'All Models'), ('distilled', 'Distilled-from-API'),
                   ('non_distilled', 'Non-Distilled')]:
    if key in results_all:
        r = results_all[key]
        report_lines += [
            f"### {label}\n",
            f"- N_ref (2023): {r['n_ref']}, N_foc (2024): {r['n_foc']}",
            f"- C%: {r['c_pct']:.1f}% ({r['n_c']}/{len(item_ids)} items)",
            f"- Direction: {r['n_favor_ref']} favor 2023, {r['n_favor_foc']} favor 2024",
            f"- ρ(difficulty, Δ_MH): {r['rho']:.3f} (p={r['rho_p']:.4f})",
            "",
        ]

# Overlap
if 'distilled' in results_all and 'non_distilled' in results_all:
    report_lines += [
        "## DIF Item Overlap\n",
        f"- Distilled C items: {len(c_dist)}",
        f"- Non-distilled C items: {len(c_nondist)}",
        f"- Intersection: {len(c_dist & c_nondist)}",
        f"- Jaccard similarity: {jaccard:.3f}",
        "",
    ]

# Size-matched control
if 'distilled' in results_all and 'control_c_pcts' in dir():
    report_lines += [
        "## Size-Matched Control\n",
        f"- Distilled C%: {dist_c:.1f}%",
        f"- Control C% (mean ± sd): {ctrl_mean:.1f}% ± {ctrl_std:.1f}%",
        f"- Control 95% CI: [{ctrl_ci[0]:.1f}%, {ctrl_ci[1]:.1f}%]",
        f"- Distilled {'**outside**' if dist_c < ctrl_ci[0] or dist_c > ctrl_ci[1] else 'within'} control CI",
        "",
    ]

report_lines += [
    "\n## Implication for Paper\n",
    "The open-vs-closed analysis is not feasible with METABENCH data. This is a",
    "limitation of the dataset, not of the method. If API-only model responses were",
    "available (e.g., from LMSYS Chatbot Arena or custom evaluation), the same",
    "MH-DIF pipeline could be applied directly.",
]

report_text = "\n".join(report_lines)
(OUT / "open_closed_dif_report.md").write_text(report_text)
print(f"\n\nReport saved to {OUT / 'open_closed_dif_report.md'}")
print(f"Classification saved to {OUT / 'model_source_classification.csv'}")
print("\nDone.")
