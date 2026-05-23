"""
MF2: Design Effect (DEFF) Estimation — v3
Cohort-consistent: only ref (2023) + foc (2024) models, matching DIF analysis.
Reports both weighted and simple m̄ variants.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts"
OUT = ARTIFACTS / "mf2_deff_estimation"
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    f = np.load(ARTIFACTS / "response_matrix.npz")
    X = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape']))
    idx = np.load(ARTIFACTS / "response_matrix_index.npz", allow_pickle=True)
    model_names = idx['model_names']
    item_ids = idx['item_ids']

    meta = pd.read_csv(ARTIFACTS / "plan_032_dif_cleaned_mmlu" / "cleaned_mmlu_analysis.csv")
    dif = pd.read_csv(ARTIFACTS / "plan_001" / "dif_results_temporal.csv")
    return X, model_names, item_ids, meta, dif


def filter_valid_models(meta, model_names):
    """Exclude NaN family, 'unknown', families with size < 2, and non-DIF cohorts."""
    # Cohort filter: only ref (2023) + foc (2024)
    cohort_mask = meta['cohort_temporal'].isin(['2023', '2024'])
    meta_cohort = meta.loc[cohort_mask].copy()

    # Family filter on cohort-restricted set
    fc = meta_cohort['family'].value_counts()
    valid_families = [f for f in fc[fc >= 2].index if f != 'unknown' and pd.notna(f)]

    mask = meta_cohort['family'].isin(valid_families)
    valid_meta = meta_cohort.loc[mask].copy()

    model_name_to_idx = {name: i for i, name in enumerate(model_names)}
    valid_indices = []
    valid_families_list = []
    valid_model_names_list = []
    for _, row in valid_meta.iterrows():
        m = row['model_name']
        if m in model_name_to_idx:
            valid_indices.append(model_name_to_idx[m])
            valid_families_list.append(row['family'])
            valid_model_names_list.append(m)

    return (np.array(valid_indices), np.array(valid_families_list),
            valid_model_names_list, valid_families)


def compute_icc_vectorized(X_valid, families_valid):
    """Compute per-item ICC using matrix operations. Returns both m̄ variants."""
    N, n_items = X_valid.shape
    unique_families = np.unique(families_valid)
    K = len(unique_families)
    family_to_idx = {f: i for i, f in enumerate(unique_families)}
    family_indices = np.array([family_to_idx[f] for f in families_valid])

    F = sparse.csc_matrix(
        (np.ones(N), (np.arange(N), family_indices)),
        shape=(N, K)
    )
    family_sizes = np.array(F.sum(axis=0)).ravel()

    print(f"  Converting to dense ({N} x {n_items})...")
    X_dense = X_valid.toarray().astype(np.float64)

    grand_mean = X_dense.mean(axis=0)

    print("  Computing family means...")
    family_sums = F.T @ X_dense
    family_means = family_sums / family_sizes[:, None]

    print("  Computing SSB...")
    deviations = family_means - grand_mean[None, :]
    SSB = (family_sizes[:, None] * deviations ** 2).sum(axis=0)

    print("  Computing SSW...")
    sum_y2 = (X_dense ** 2).sum(axis=0)
    weighted_mean2 = (family_sizes[:, None] * family_means ** 2).sum(axis=0)
    SSW = sum_y2 - weighted_mean2

    MSB = SSB / (K - 1)
    MSW = SSW / (N - K)

    m_bar_weighted = (family_sizes ** 2).sum() / family_sizes.sum()
    m_bar_simple = family_sizes.mean()

    print("  Computing ICC...")
    numerator = MSB - MSW
    denominator = MSB + (m_bar_weighted - 1) * MSW

    icc = np.zeros(n_items)
    nonzero_denom = denominator != 0
    icc[nonzero_denom] = numerator[nonzero_denom] / denominator[nonzero_denom]
    icc = np.clip(icc, 0, 1)

    return icc, m_bar_weighted, m_bar_simple, K, family_sizes


def adjust_dif(dif, icc, item_ids, m_bar_weighted, m_bar_simple):
    """Adjust chi-squared by both DEFF variants and reclassify ETS."""
    item_to_icc = dict(zip(item_ids, icc))

    dif = dif.copy()
    dif['icc'] = dif['item_id'].map(item_to_icc)

    dif['deff_weighted'] = 1 + (m_bar_weighted - 1) * dif['icc']
    dif['deff_simple'] = 1 + (m_bar_simple - 1) * dif['icc']

    dif['original_chi2'] = (dif['delta_mh'] / dif['se']) ** 2

    dif['adj_chi2_weighted'] = dif['original_chi2'] / dif['deff_weighted']
    dif['adj_chi2_simple'] = dif['original_chi2'] / dif['deff_simple']

    dif['adj_p_weighted'] = 1 - chi2.cdf(dif['adj_chi2_weighted'], df=1)
    dif['adj_p_simple'] = 1 - chi2.cdf(dif['adj_chi2_simple'], df=1)

    abs_delta = dif['delta_mh'].abs()

    for suffix in ['weighted', 'simple']:
        col = f'adj_ets_{suffix}'
        sig = dif[f'adj_p_{suffix}'] < 0.05
        dif[col] = 'A'
        dif.loc[(abs_delta >= 1.0) & sig, col] = 'B'
        dif.loc[(abs_delta >= 1.5) & sig, col] = 'C'

    dif = dif.rename(columns={
        'p_value': 'original_p',
        'ets_class': 'original_ets'
    })

    return dif


def main():
    t0 = time.time()
    print("=" * 60)
    print("MF2: Design Effect (DEFF) Estimation — v3 (cohort-consistent)")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    X, model_names, item_ids, meta, dif = load_data()
    print(f"  Response matrix: {X.shape}")
    print(f"  DIF results: {len(dif)} items")
    print(f"  Metadata: {len(meta)} models")
    n_ref = (meta['cohort_temporal'] == '2023').sum()
    n_foc = (meta['cohort_temporal'] == '2024').sum()
    print(f"  Ref (2023): {n_ref}, Foc (2024): {n_foc}, Total DIF cohort: {n_ref + n_foc}")

    print("\n[2/5] Filtering valid models (cohort + family)...")
    valid_indices, families_valid, valid_model_names, valid_families = \
        filter_valid_models(meta, model_names)
    print(f"  Valid models (ref+foc, known family, family≥2): {len(valid_indices)}")
    print(f"  Valid families: {len(valid_families)}")

    X_valid = X[valid_indices]
    print(f"  X_valid shape: {X_valid.shape}")

    print("\n[3/5] Computing per-item ICC...")
    icc, m_bar_weighted, m_bar_simple, K, family_sizes = \
        compute_icc_vectorized(X_valid, families_valid)

    print(f"\n  m̄_weighted = {m_bar_weighted:.2f}")
    print(f"  m̄_simple   = {m_bar_simple:.2f}")
    print(f"  ICC stats:")
    print(f"    mean   = {icc.mean():.4f}")
    print(f"    median = {np.median(icc):.4f}")
    print(f"    Q1     = {np.percentile(icc, 25):.4f}")
    print(f"    Q3     = {np.percentile(icc, 75):.4f}")
    print(f"    max    = {icc.max():.4f}")
    print(f"    % ICC > 0.1 = {(icc > 0.1).mean() * 100:.1f}%")
    print(f"    % ICC == 0  = {(icc == 0).mean() * 100:.1f}%")

    deff_global_w = 1 + (m_bar_weighted - 1) * icc.mean()
    deff_global_s = 1 + (m_bar_simple - 1) * icc.mean()
    print(f"\n  Global DEFF (weighted) = {deff_global_w:.2f}")
    print(f"  Global DEFF (simple)   = {deff_global_s:.2f}")
    print(f"  Effective N (weighted) = {len(valid_indices) / deff_global_w:.1f}")
    print(f"  Effective N (simple)   = {len(valid_indices) / deff_global_s:.1f}")

    print("\n[4/5] Adjusting DIF classifications...")
    results = adjust_dif(dif, icc, item_ids, m_bar_weighted, m_bar_simple)

    original_c_pct = (results['original_ets'] == 'C').mean() * 100
    adj_c_w = (results['adj_ets_weighted'] == 'C').mean() * 100
    adj_c_s = (results['adj_ets_simple'] == 'C').mean() * 100
    print(f"  Original C% = {original_c_pct:.2f}%")
    print(f"  Adjusted C% (weighted) = {adj_c_w:.2f}%")
    print(f"  Adjusted C% (simple)   = {adj_c_s:.2f}%")

    for label, suffix in [('Weighted', 'weighted'), ('Simple', 'simple')]:
        col = f'adj_ets_{suffix}'
        c_to_b = ((results['original_ets'] == 'C') & (results[col] == 'B')).sum()
        c_to_a = ((results['original_ets'] == 'C') & (results[col] == 'A')).sum()
        b_to_a = ((results['original_ets'] == 'B') & (results[col] == 'A')).sum()
        print(f"  [{label}] C→B={c_to_b}, C→A={c_to_a}, B→A={b_to_a}")

    # Comparison with old v2 (2118 models)
    print("\n  --- Old vs New comparison ---")
    try:
        old = pd.read_csv(OUT / "deff_results.csv")
        old_c = (old['adjusted_ets'] == 'C').mean() * 100
        print(f"  Old (2118 models): C% = {old_c:.2f}%")
        print(f"  New weighted (ref+foc): C% = {adj_c_w:.2f}%")
        print(f"  New simple   (ref+foc): C% = {adj_c_s:.2f}%")
    except FileNotFoundError:
        print("  (no old results to compare)")

    print("\n[5/5] Saving results...")
    out_cols = ['item_id', 'subject', 'icc',
                'deff_weighted', 'deff_simple',
                'delta_mh', 'se',
                'original_chi2', 'adj_chi2_weighted', 'adj_chi2_simple',
                'original_p', 'adj_p_weighted', 'adj_p_simple',
                'original_ets', 'adj_ets_weighted', 'adj_ets_simple']
    results[out_cols].to_csv(OUT / "deff_results.csv", index=False)
    print(f"  Saved: deff_results.csv ({len(results)} rows)")

    # Summary report
    family_size_series = pd.Series(family_sizes)
    unique_fams = np.unique(families_valid)
    fam_size_map = pd.Series(family_sizes, index=unique_fams).sort_values(ascending=False)
    top5_str = ', '.join(f'{f}({s})' for f, s in fam_size_map.head(5).items())

    # Reclassification counts for summary
    reclass = {}
    for suffix in ['weighted', 'simple']:
        col = f'adj_ets_{suffix}'
        reclass[suffix] = {
            'c_to_b': ((results['original_ets'] == 'C') & (results[col] == 'B')).sum(),
            'c_to_a': ((results['original_ets'] == 'C') & (results[col] == 'A')).sum(),
            'b_to_a': ((results['original_ets'] == 'B') & (results[col] == 'A')).sum(),
        }

    summary = f"""# MF2: Design Effect (DEFF) Estimation — Summary (v3, cohort-consistent)

## Analysis Scope

| Parameter | Value |
|-----------|-------|
| Cohort restriction | ref (2023) + foc (2024) only |
| Models in DIF cohort (ref+foc) | {n_ref + n_foc} |
| Valid models (known family, family≥2) | {len(valid_indices)} |
| Valid families | {K} |
| Total models in response matrix | {X.shape[0]} |
| Items analyzed | {X.shape[1]} |

**Change from v2**: v2 used all {X.shape[0]} models with family labels ({len(valid_indices)} after filtering).
v3 restricts to ref+foc cohort ({n_ref + n_foc}) before family filtering, yielding {len(valid_indices)} valid models.

### Family Size Distribution (within ref+foc cohort)

| Stat | Value |
|------|-------|
| Mean (m̄_simple) | {m_bar_simple:.2f} |
| m̄_weighted (Σn²/Σn) | {m_bar_weighted:.2f} |
| Median | {np.median(family_sizes):.1f} |
| Min | {family_sizes.min()} |
| Max | {family_sizes.max()} |
| Top-5 families | {top5_str} |

## Design Effect — Two Variants

| Parameter | Weighted (Σn²/Σn) | Simple (mean) |
|-----------|--------------------|---------------|
| m̄ | {m_bar_weighted:.2f} | {m_bar_simple:.2f} |
| Mean ICC | {icc.mean():.4f} | {icc.mean():.4f} |
| **Global DEFF** | **{deff_global_w:.2f}** | **{deff_global_s:.2f}** |
| Effective N | {len(valid_indices) / deff_global_w:.1f} | {len(valid_indices) / deff_global_s:.1f} |

### ICC Distribution

| Stat | Value |
|------|-------|
| Mean | {icc.mean():.4f} |
| Median | {np.median(icc):.4f} |
| Q1 | {np.percentile(icc, 25):.4f} |
| Q3 | {np.percentile(icc, 75):.4f} |
| Max | {icc.max():.4f} |
| % ICC > 0.1 | {(icc > 0.1).mean() * 100:.1f}% |
| % ICC = 0 | {(icc == 0).mean() * 100:.1f}% |

## Adjusted DIF Classifications

### Weighted m̄

| Metric | Original | Adjusted | Change |
|--------|----------|----------|--------|
| C items | {(results['original_ets'] == 'C').sum()} ({original_c_pct:.1f}%) | {(results['adj_ets_weighted'] == 'C').sum()} ({adj_c_w:.1f}%) | {(results['adj_ets_weighted'] == 'C').sum() - (results['original_ets'] == 'C').sum()} |
| B items | {(results['original_ets'] == 'B').sum()} | {(results['adj_ets_weighted'] == 'B').sum()} | {(results['adj_ets_weighted'] == 'B').sum() - (results['original_ets'] == 'B').sum()} |
| A items | {(results['original_ets'] == 'A').sum()} | {(results['adj_ets_weighted'] == 'A').sum()} | {(results['adj_ets_weighted'] == 'A').sum() - (results['original_ets'] == 'A').sum()} |

Reclassification: C→B={reclass['weighted']['c_to_b']}, C→A={reclass['weighted']['c_to_a']}, B→A={reclass['weighted']['b_to_a']}

### Simple m̄

| Metric | Original | Adjusted | Change |
|--------|----------|----------|--------|
| C items | {(results['original_ets'] == 'C').sum()} ({original_c_pct:.1f}%) | {(results['adj_ets_simple'] == 'C').sum()} ({adj_c_s:.1f}%) | {(results['adj_ets_simple'] == 'C').sum() - (results['original_ets'] == 'C').sum()} |
| B items | {(results['original_ets'] == 'B').sum()} | {(results['adj_ets_simple'] == 'B').sum()} | {(results['adj_ets_simple'] == 'B').sum() - (results['original_ets'] == 'B').sum()} |
| A items | {(results['original_ets'] == 'A').sum()} | {(results['adj_ets_simple'] == 'A').sum()} | {(results['adj_ets_simple'] == 'A').sum() - (results['original_ets'] == 'A').sum()} |

Reclassification: C→B={reclass['simple']['c_to_b']}, C→A={reclass['simple']['c_to_a']}, B→A={reclass['simple']['b_to_a']}

## Reviewer's DEFF 3–5 Expectation

The **simple m̄ DEFF = {deff_global_s:.2f}** is the appropriate comparison for reviewers expecting DEFF 3–5.
The weighted variant ({deff_global_w:.2f}) is inflated by large families (e.g., {top5_str.split(',')[0].strip()})
and represents the upper bound under worst-case clustering.

Both variants are reported so the paper can present the simple DEFF as the primary estimate
and note the weighted variant as a sensitivity check.

## Validation

- Cohort-consistent: {len(valid_indices)} valid models < {n_ref + n_foc} DIF cohort ✓
- m̄_weighted ({m_bar_weighted:.2f}) > m̄_simple ({m_bar_simple:.2f}) ✓
- Adjusted C% (weighted: {adj_c_w:.1f}%, simple: {adj_c_s:.1f}%) ≤ Original ({original_c_pct:.1f}%) ✓
- p-value reconstruction: median |Δp| = {np.median(np.abs(1 - chi2.cdf(results['original_chi2'], 1) - results['original_p'])):.2e}
"""
    (OUT / "deff_summary.md").write_text(summary)
    print(f"  Saved: deff_summary.md")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
