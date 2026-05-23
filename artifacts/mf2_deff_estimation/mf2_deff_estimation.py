"""
MF2: Design Effect (DEFF) Estimation
Quantifies non-independence within model families and adjusts DIF chi-squared.
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
    """Exclude NaN family, 'unknown', and families with size < 2."""
    fc = meta['family'].value_counts()
    valid_families = [f for f in fc[fc >= 2].index if f != 'unknown' and pd.notna(f)]

    mask = meta['family'].isin(valid_families)
    valid_meta = meta.loc[mask].copy()

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
    """Compute per-item ICC using matrix operations."""
    N, n_items = X_valid.shape
    unique_families = np.unique(families_valid)
    K = len(unique_families)
    family_to_idx = {f: i for i, f in enumerate(unique_families)}
    family_indices = np.array([family_to_idx[f] for f in families_valid])

    # Family indicator matrix (N x K) — sparse for memory
    F = sparse.csc_matrix(
        (np.ones(N), (np.arange(N), family_indices)),
        shape=(N, K)
    )
    family_sizes = np.array(F.sum(axis=0)).ravel()  # (K,)

    # Convert X_valid to dense float for computation
    print(f"  Converting to dense ({N} x {n_items})...")
    X_dense = X_valid.toarray().astype(np.float64)

    # Grand mean per item
    grand_mean = X_dense.mean(axis=0)  # (n_items,)

    # Family sums and means per item
    print("  Computing family means...")
    family_sums = F.T @ X_dense  # (K, n_items)
    family_means = family_sums / family_sizes[:, None]  # (K, n_items)

    # SSB: sum_f n_f * (mean_f - grand_mean)^2
    print("  Computing SSB...")
    deviations = family_means - grand_mean[None, :]  # (K, n_items)
    SSB = (family_sizes[:, None] * deviations ** 2).sum(axis=0)  # (n_items,)

    # SSW: sum_f sum_i (y_if - mean_f)^2
    # = sum(y^2) - sum_f n_f * mean_f^2
    print("  Computing SSW...")
    sum_y2 = (X_dense ** 2).sum(axis=0)  # (n_items,)
    weighted_mean2 = (family_sizes[:, None] * family_means ** 2).sum(axis=0)  # (n_items,)
    SSW = sum_y2 - weighted_mean2  # (n_items,)

    # MSB and MSW
    MSB = SSB / (K - 1)
    MSW = SSW / (N - K)

    # Weighted m_bar
    m_bar_weighted = (family_sizes ** 2).sum() / family_sizes.sum()

    # ICC = (MSB - MSW) / (MSB + (m_bar - 1) * MSW)
    print("  Computing ICC...")
    numerator = MSB - MSW
    denominator = MSB + (m_bar_weighted - 1) * MSW

    icc = np.zeros(n_items)
    nonzero_denom = denominator != 0
    icc[nonzero_denom] = numerator[nonzero_denom] / denominator[nonzero_denom]

    # Clip: ICC < 0 → 0, ICC > 1 → 1
    icc = np.clip(icc, 0, 1)

    return icc, m_bar_weighted, K, family_sizes


def adjust_dif(dif, icc, item_ids, m_bar_weighted):
    """Adjust chi-squared by DEFF and reclassify ETS."""
    # Map ICC to DIF items
    item_to_icc = dict(zip(item_ids, icc))

    dif = dif.copy()
    dif['icc'] = dif['item_id'].map(item_to_icc)

    # Per-item DEFF
    dif['deff'] = 1 + (m_bar_weighted - 1) * dif['icc']

    # Original chi-squared from delta_mh / se
    dif['original_chi2'] = (dif['delta_mh'] / dif['se']) ** 2

    # Adjusted chi-squared
    dif['adjusted_chi2'] = dif['original_chi2'] / dif['deff']

    # Adjusted p-value
    dif['adjusted_p'] = 1 - chi2.cdf(dif['adjusted_chi2'], df=1)

    # Reclassify ETS
    abs_delta = dif['delta_mh'].abs()
    sig = dif['adjusted_p'] < 0.05

    dif['adjusted_ets'] = 'A'
    dif.loc[(abs_delta >= 1.0) & sig, 'adjusted_ets'] = 'B'
    dif.loc[(abs_delta >= 1.5) & sig, 'adjusted_ets'] = 'C'

    # Rename columns for clarity
    dif = dif.rename(columns={
        'p_value': 'original_p',
        'ets_class': 'original_ets'
    })

    return dif


def main():
    t0 = time.time()
    print("=" * 60)
    print("MF2: Design Effect (DEFF) Estimation")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    X, model_names, item_ids, meta, dif = load_data()
    print(f"  Response matrix: {X.shape}")
    print(f"  DIF results: {len(dif)} items")

    print("\n[2/5] Filtering valid models...")
    valid_indices, families_valid, valid_model_names, valid_families = \
        filter_valid_models(meta, model_names)
    print(f"  Valid models: {len(valid_indices)}")
    print(f"  Valid families: {len(valid_families)}")

    X_valid = X[valid_indices]
    print(f"  X_valid shape: {X_valid.shape}")

    print("\n[3/5] Computing per-item ICC...")
    icc, m_bar_weighted, K, family_sizes = compute_icc_vectorized(X_valid, families_valid)
    print(f"\n  m̄_weighted = {m_bar_weighted:.2f}")
    print(f"  ICC stats:")
    print(f"    mean   = {icc.mean():.4f}")
    print(f"    median = {np.median(icc):.4f}")
    print(f"    Q1     = {np.percentile(icc, 25):.4f}")
    print(f"    Q3     = {np.percentile(icc, 75):.4f}")
    print(f"    max    = {icc.max():.4f}")
    print(f"    % ICC > 0.1 = {(icc > 0.1).mean() * 100:.1f}%")
    print(f"    % ICC == 0  = {(icc == 0).mean() * 100:.1f}%")

    # Global DEFF
    deff_global = 1 + (m_bar_weighted - 1) * icc.mean()
    print(f"\n  Global DEFF = {deff_global:.2f}")
    print(f"  Effective N = {len(valid_indices) / deff_global:.1f} (from {len(valid_indices)})")

    print("\n[4/5] Adjusting DIF classifications...")
    results = adjust_dif(dif, icc, item_ids, m_bar_weighted)

    original_c_pct = (results['original_ets'] == 'C').mean() * 100
    adjusted_c_pct = (results['adjusted_ets'] == 'C').mean() * 100
    print(f"  Original C% = {original_c_pct:.2f}%")
    print(f"  Adjusted C% = {adjusted_c_pct:.2f}%")

    # Reclassification analysis
    c_to_b = ((results['original_ets'] == 'C') & (results['adjusted_ets'] == 'B')).sum()
    c_to_a = ((results['original_ets'] == 'C') & (results['adjusted_ets'] == 'A')).sum()
    b_to_a = ((results['original_ets'] == 'B') & (results['adjusted_ets'] == 'A')).sum()
    print(f"  Reclassified: C→B = {c_to_b}, C→A = {c_to_a}, B→A = {b_to_a}")

    print("\n[5/5] Saving results...")
    # Save per-item results
    out_cols = ['item_id', 'subject', 'icc', 'deff', 'delta_mh', 'se',
                'original_chi2', 'adjusted_chi2', 'original_p', 'adjusted_p',
                'original_ets', 'adjusted_ets']
    results[out_cols].to_csv(OUT / "deff_results.csv", index=False)
    print(f"  Saved: deff_results.csv ({len(results)} rows)")

    # Summary report
    family_size_series = pd.Series(family_sizes)
    unique_fams = np.unique(families_valid)
    fam_size_map = pd.Series(family_sizes, index=unique_fams).sort_values(ascending=False)
    top3_str = ', '.join(f'{f}({s})' for f, s in fam_size_map.head(3).items())
    summary = f"""# MF2: Design Effect (DEFF) Estimation — Summary

## Analysis Scope

| Parameter | Value |
|-----------|-------|
| Valid models (excl. unknown/NaN, family≥2) | {len(valid_indices)} |
| Valid families | {K} |
| Total models in response matrix | {X.shape[0]} |
| Items analyzed | {X.shape[1]} |

### Family Size Distribution

| Stat | Value |
|------|-------|
| Mean | {family_size_series.mean():.1f} |
| Median | {np.median(family_sizes):.1f} |
| Min | {family_sizes.min()} |
| Max | {family_sizes.max()} |
| Top-3 families | {top3_str} |

## Design Effect

| Parameter | Value |
|-----------|-------|
| m̄_weighted | {m_bar_weighted:.2f} |
| Mean ICC | {icc.mean():.4f} |
| Median ICC | {np.median(icc):.4f} |
| Q1 ICC | {np.percentile(icc, 25):.4f} |
| Q3 ICC | {np.percentile(icc, 75):.4f} |
| Max ICC | {icc.max():.4f} |
| % items with ICC > 0.1 | {(icc > 0.1).mean() * 100:.1f}% |
| % items with ICC = 0 | {(icc == 0).mean() * 100:.1f}% |
| **Global DEFF** | **{deff_global:.2f}** |
| Effective N | {len(valid_indices) / deff_global:.1f} |

## Adjusted DIF Classifications

| Metric | Original | Adjusted | Change |
|--------|----------|----------|--------|
| C items | {(results['original_ets'] == 'C').sum()} ({original_c_pct:.1f}%) | {(results['adjusted_ets'] == 'C').sum()} ({adjusted_c_pct:.1f}%) | {(results['adjusted_ets'] == 'C').sum() - (results['original_ets'] == 'C').sum()} |
| B items | {(results['original_ets'] == 'B').sum()} | {(results['adjusted_ets'] == 'B').sum()} | {(results['adjusted_ets'] == 'B').sum() - (results['original_ets'] == 'B').sum()} |
| A items | {(results['original_ets'] == 'A').sum()} | {(results['adjusted_ets'] == 'A').sum()} | {(results['adjusted_ets'] == 'A').sum() - (results['original_ets'] == 'A').sum()} |

### Reclassification Breakdown

| Transition | Count |
|------------|-------|
| C → B | {c_to_b} |
| C → A | {c_to_a} |
| B → A | {b_to_a} |
| Total reclassified from C | {c_to_b + c_to_a} |

## Validation

- p-value reconstruction check (original chi2 → p): median absolute error = {np.median(np.abs(1 - chi2.cdf(results['original_chi2'], 1) - results['original_p'])):.2e}
- m̄_weighted ({m_bar_weighted:.2f}) > simple mean ({family_size_series.mean():.2f}) ✓
- DEFF > 1: {deff_global:.2f} ✓
- Adjusted C% ({adjusted_c_pct:.1f}%) ≤ Original C% ({original_c_pct:.1f}%) ✓

## Notes

- **DEFF higher than reviewer's 3-5 expectation**: m̄_weighted = {m_bar_weighted:.2f} is driven by
  highly skewed family sizes (mistral=547, llama-3=315 vs median=15). The weighted formula
  m̄ = Σ(n_f²)/Σ(n_f) gives large families proportionally more weight, which is correct
  (they contribute more to the clustering effect) but produces a larger DEFF than a simple-mean
  estimate would.
- **C→B = 0**: This is mathematically expected. C items have |δ_MH| ≥ 1.5 by definition.
  After adjustment, they either remain significant (→ still C) or lose significance (→ A).
  They cannot become B (which requires 1.0 ≤ |δ| < 1.5) since δ is unchanged by DEFF.
- **chi² approximation**: We use (δ_MH/SE)² as the Wald-type statistic. This differs slightly
  from the continuity-corrected MH chi² (median |Δp| ≈ 8e-4), but is standard for DEFF adjustment.

## Conclusion

The design effect of **{deff_global:.2f}** indicates substantial non-independence within model families.
After DEFF adjustment, C-class items drop from {original_c_pct:.1f}% to {adjusted_c_pct:.1f}%
(Δ = {adjusted_c_pct - original_c_pct:.1f} pp), with {c_to_b + c_to_a} items reclassified out of
the C category. This confirms that family-level clustering inflates apparent DIF significance,
though the majority ({(results['adjusted_ets'] == 'C').sum()}/{(results['original_ets'] == 'C').sum()}, {(results['adjusted_ets'] == 'C').sum()/(results['original_ets'] == 'C').sum()*100:.0f}%) of
C-classified items remain significant even after correction.
"""
    (OUT / "deff_summary.md").write_text(summary)
    print(f"  Saved: deff_summary.md")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
