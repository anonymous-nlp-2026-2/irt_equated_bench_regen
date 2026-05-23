"""
Plan 013: Yen's Q3 Local Independence Diagnostic for MMLU Items
Q3(i,j) = corr(residual_i, residual_j) after removing ability (theta) effect
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

ARTIFACTS = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

def load_data():
    R = sp.load_npz(ARTIFACTS / 'response_matrix.npz')
    idx = np.load(ARTIFACTS / 'response_matrix_index.npz', allow_pickle=True)
    meta = pd.read_csv(ARTIFACTS / 'item_metadata.csv')
    return R, idx['model_names'], idx['item_ids'], meta

def estimate_theta(R):
    """Logit(proportion_correct) as theta estimate."""
    total = np.asarray(R.sum(axis=1)).ravel().astype(np.float64)
    n_items = R.shape[1]  # all models answered all items; sparse only stores 1s
    p = total / n_items
    theta = np.log((p + 0.001) / (1 - p + 0.001))
    return theta

def compute_residuals(R_dense, theta, n_bins=20):
    """
    Compute residuals r_ij = X_ij - E(X_ij | theta_j).
    E is estimated by binning theta into n_bins groups and computing
    conditional item accuracy per bin.
    """
    n_models, n_items = R_dense.shape
    bin_edges = np.percentile(theta, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1
    bin_edges[-1] += 1
    bin_idx = np.digitize(theta, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    expected = np.zeros_like(R_dense, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        group_mean = R_dense[mask].mean(axis=0)
        expected[mask] = group_mean

    residuals = R_dense - expected
    return residuals

def q3_for_pairs(residuals, pairs):
    """Compute Q3 for a list of (i, j) column index pairs."""
    q3_vals = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        ri = residuals[:, i]
        rj = residuals[:, j]
        if ri.std() < 1e-10 or rj.std() < 1e-10:
            q3_vals[k] = np.nan
        else:
            q3_vals[k] = np.corrcoef(ri, rj)[0, 1]
    return q3_vals

def q3_for_pairs_vectorized(residuals, pairs, batch_size=5000):
    """Vectorized Q3 computation in batches."""
    pairs = np.array(pairs)
    q3_vals = np.empty(len(pairs))

    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        batch = pairs[start:end]
        ri = residuals[:, batch[:, 0]]  # (n_models, batch)
        rj = residuals[:, batch[:, 1]]

        ri_m = ri - ri.mean(axis=0, keepdims=True)
        rj_m = rj - rj.mean(axis=0, keepdims=True)

        ri_std = np.sqrt((ri_m ** 2).sum(axis=0))
        rj_std = np.sqrt((rj_m ** 2).sum(axis=0))

        denom = ri_std * rj_std
        valid = denom > 1e-10

        cov = (ri_m * rj_m).sum(axis=0)
        q3_batch = np.where(valid, cov / denom, np.nan)
        q3_vals[start:end] = q3_batch

    return q3_vals

def sample_pairs(item_indices_by_subject, n_within=2000, n_cross=2000, rng=None):
    """Sample within-subject and cross-subject item pairs."""
    if rng is None:
        rng = np.random.default_rng(42)

    within_pairs = []
    within_subjects = []
    for subj, indices in item_indices_by_subject.items():
        if len(indices) < 2:
            continue
        n_possible = len(indices) * (len(indices) - 1) // 2
        for _ in range(min(n_possible, 200)):
            i, j = rng.choice(indices, size=2, replace=False)
            within_pairs.append((min(i, j), max(i, j)))
            within_subjects.append(subj)

    within_pairs_set = set(within_pairs)
    if len(within_pairs) > n_within:
        sel = rng.choice(len(within_pairs), size=n_within, replace=False)
        within_pairs = [within_pairs[s] for s in sel]
        within_subjects = [within_subjects[s] for s in sel]

    all_items = []
    item_to_subject = {}
    for subj, indices in item_indices_by_subject.items():
        for idx in indices:
            all_items.append(idx)
            item_to_subject[idx] = subj

    cross_pairs = []
    attempts = 0
    while len(cross_pairs) < n_cross and attempts < n_cross * 20:
        i, j = rng.choice(all_items, size=2, replace=False)
        if item_to_subject[i] != item_to_subject[j]:
            cross_pairs.append((min(i, j), max(i, j)))
        attempts += 1

    return within_pairs, within_subjects, cross_pairs

def full_within_subject_q3(residuals, indices, max_pairs=50000):
    """Compute all within-subject Q3 for a subject's items."""
    n = len(indices)
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            pairs.append((indices[a], indices[b]))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    q3_vals = q3_for_pairs_vectorized(residuals, pairs)
    return q3_vals

def main():
    print("=" * 60)
    print("Plan 013: Yen's Q3 Local Independence Diagnostic")
    print("=" * 60)

    t0 = time.time()

    # Load data
    print("\n[1/6] Loading data...")
    R, model_names, item_ids, meta = load_data()
    n_models, n_items = R.shape
    print(f"  Response matrix: {n_models} models × {n_items} items")

    # Filter items with extreme p-values
    extreme_mask = (meta['empirical_p'] < 0.05) | (meta['empirical_p'] > 0.95)
    n_extreme = extreme_mask.sum()
    print(f"  Items with p < 0.05 or p > 0.95: {n_extreme} (will be excluded from Q3)")

    valid_items = meta[~extreme_mask]['item_id'].values
    valid_item_set = set(valid_items)
    item_id_list = list(item_ids)
    valid_col_indices = [i for i, iid in enumerate(item_id_list) if iid in valid_item_set]
    print(f"  Valid items for Q3: {len(valid_col_indices)}")

    # Build subject -> column index mapping (only valid items)
    item_id_to_col = {iid: i for i, iid in enumerate(item_id_list)}
    meta_valid = meta[~extreme_mask].copy()
    subject_to_cols = {}
    for _, row in meta_valid.iterrows():
        subj = row['subject']
        col = item_id_to_col.get(row['item_id'])
        if col is not None:
            subject_to_cols.setdefault(subj, []).append(col)

    # Theta estimation
    print("\n[2/6] Estimating theta...")
    theta = estimate_theta(R)
    print(f"  Theta range: [{theta.min():.2f}, {theta.max():.2f}], mean={theta.mean():.2f}")

    # Compute residuals
    print("\n[3/6] Computing residuals (dense matrix)...")
    R_dense = R.toarray().astype(np.float64)
    residuals = compute_residuals(R_dense, theta, n_bins=20)
    del R_dense
    print(f"  Residuals shape: {residuals.shape}, mean={residuals.mean():.6f}")

    # Validation on virology
    print("\n[4/6] Validation on virology subject...")
    virology_cols = subject_to_cols.get('virology', [])
    print(f"  Virology items: {len(virology_cols)}")
    if len(virology_cols) >= 2:
        vir_q3 = full_within_subject_q3(residuals, virology_cols)
        vir_valid = vir_q3[~np.isnan(vir_q3)]
        print(f"  Virology Q3: n_pairs={len(vir_valid)}, mean={vir_valid.mean():.4f}, "
              f"median={np.median(vir_valid):.4f}, std={vir_valid.std():.4f}")
        print(f"  |Q3| > 0.20: {(np.abs(vir_valid) > 0.20).mean():.1%}")
        print(f"  Q3 range: [{vir_valid.min():.4f}, {vir_valid.max():.4f}]")

    # Sampled within-subject vs cross-subject Q3
    print("\n[5/6] Sampling within-subject and cross-subject pairs...")
    rng = np.random.default_rng(42)
    within_pairs, within_subjects, cross_pairs = sample_pairs(
        subject_to_cols, n_within=3000, n_cross=3000, rng=rng
    )
    print(f"  Within-subject pairs: {len(within_pairs)}")
    print(f"  Cross-subject pairs: {len(cross_pairs)}")

    q3_within = q3_for_pairs_vectorized(residuals, within_pairs)
    q3_cross = q3_for_pairs_vectorized(residuals, cross_pairs)

    q3_within_valid = q3_within[~np.isnan(q3_within)]
    q3_cross_valid = q3_cross[~np.isnan(q3_cross)]

    print(f"\n  Within-subject Q3:")
    print(f"    n={len(q3_within_valid)}, mean={q3_within_valid.mean():.4f}, "
          f"median={np.median(q3_within_valid):.4f}, std={q3_within_valid.std():.4f}")
    print(f"    |Q3| > 0.20: {(np.abs(q3_within_valid) > 0.20).mean():.1%}")

    print(f"  Cross-subject Q3:")
    print(f"    n={len(q3_cross_valid)}, mean={q3_cross_valid.mean():.4f}, "
          f"median={np.median(q3_cross_valid):.4f}, std={q3_cross_valid.std():.4f}")
    print(f"    |Q3| > 0.20: {(np.abs(q3_cross_valid) > 0.20).mean():.1%}")

    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(q3_within_valid, q3_cross_valid, alternative='two-sided')
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_pval:.2e}")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(q3_within_valid), len(q3_cross_valid)
    r_rb = 1 - 2 * u_stat / (n1 * n2)
    print(f"  Rank-biserial r = {r_rb:.4f}")

    # Save within vs cross summary
    within_cross_df = pd.DataFrame({
        'type': ['within_subject', 'cross_subject'],
        'mean_q3': [q3_within_valid.mean(), q3_cross_valid.mean()],
        'std_q3': [q3_within_valid.std(), q3_cross_valid.std()],
        'median_q3': [np.median(q3_within_valid), np.median(q3_cross_valid)],
        'n_pairs': [len(q3_within_valid), len(q3_cross_valid)],
        'pct_above_020': [
            (np.abs(q3_within_valid) > 0.20).mean() * 100,
            (np.abs(q3_cross_valid) > 0.20).mean() * 100
        ],
        'q3_p25': [np.percentile(q3_within_valid, 25), np.percentile(q3_cross_valid, 25)],
        'q3_p75': [np.percentile(q3_within_valid, 75), np.percentile(q3_cross_valid, 75)],
        'q3_p95': [np.percentile(q3_within_valid, 95), np.percentile(q3_cross_valid, 95)],
        'q3_min': [q3_within_valid.min(), q3_cross_valid.min()],
        'q3_max': [q3_within_valid.max(), q3_cross_valid.max()],
    })
    within_cross_df.to_csv(OUT / 'q3_within_vs_cross.csv', index=False)
    print(f"\n  Saved q3_within_vs_cross.csv")

    # Per-subject full Q3 analysis for top subjects
    print("\n[6/6] Per-subject Q3 analysis...")
    subject_sizes = {s: len(cols) for s, cols in subject_to_cols.items()}
    sorted_subjects = sorted(subject_sizes.keys(), key=lambda s: -subject_sizes[s])

    subject_results = []
    for subj in sorted_subjects:
        cols = subject_to_cols[subj]
        n_items_subj = len(cols)
        n_possible = n_items_subj * (n_items_subj - 1) // 2

        if n_items_subj < 2:
            continue

        if n_possible <= 50000:
            q3_vals = full_within_subject_q3(residuals, cols)
        else:
            sampled_pairs = []
            for _ in range(min(n_possible, 10000)):
                i, j = rng.choice(cols, size=2, replace=False)
                sampled_pairs.append((min(i, j), max(i, j)))
            sampled_pairs = list(set(sampled_pairs))
            q3_vals = q3_for_pairs_vectorized(residuals, sampled_pairs)

        valid = q3_vals[~np.isnan(q3_vals)]
        if len(valid) == 0:
            continue

        subject_results.append({
            'subject': subj,
            'n_items': n_items_subj,
            'mean_q3': valid.mean(),
            'median_q3': np.median(valid),
            'std_q3': valid.std(),
            'pct_above_020': (np.abs(valid) > 0.20).mean() * 100,
            'q3_max': valid.max(),
            'n_pairs_sampled': len(valid),
        })

        if len(subject_results) % 10 == 0:
            print(f"  Processed {len(subject_results)} subjects...")

    subject_df = pd.DataFrame(subject_results)
    subject_df = subject_df.sort_values('mean_q3', ascending=False)
    subject_df.to_csv(OUT / 'q3_summary.csv', index=False)
    print(f"  Saved q3_summary.csv ({len(subject_df)} subjects)")

    # Print top/bottom subjects
    print(f"\n  Top 10 subjects by mean Q3:")
    for _, row in subject_df.head(10).iterrows():
        print(f"    {row['subject']:40s} n={row['n_items']:4d}  "
              f"mean_Q3={row['mean_q3']:.4f}  |Q3|>0.20: {row['pct_above_020']:.1f}%")

    print(f"\n  Bottom 5 subjects by mean Q3:")
    for _, row in subject_df.tail(5).iterrows():
        print(f"    {row['subject']:40s} n={row['n_items']:4d}  "
              f"mean_Q3={row['mean_q3']:.4f}  |Q3|>0.20: {row['pct_above_020']:.1f}%")

    # Q3 vs subject size correlation
    corr_size_q3, pval_size_q3 = stats.spearmanr(subject_df['n_items'], subject_df['mean_q3'])
    print(f"\n  Spearman corr(subject_size, mean_Q3) = {corr_size_q3:.4f}, p = {pval_size_q3:.2e}")

    # Generate report
    print("\n" + "=" * 60)
    print("Generating report...")

    global_q3_all = np.concatenate([q3_within_valid, q3_cross_valid])

    report = f"""# Plan 013: Yen's Q3 Local Independence Diagnostic

## Summary

Yen's Q3 statistic measures residual correlations between item pairs after removing the ability (θ) effect. Values |Q3| > 0.20 indicate significant local dependence (LD). This analysis tests whether MMLU items within the same subject violate local independence — a core assumption of IRT and MH-DIF — which could inflate false positive rates.

## Method

- **Ability estimation**: θ = logit(proportion_correct) per model, with Laplace smoothing
- **Residuals**: θ binned into 20 quantile groups; E(X_ij | θ) = conditional item accuracy within each bin; residual = observed − expected
- **Q3**: Pearson correlation of residual vectors across {n_models} models
- **Sampling**: {len(q3_within_valid)} within-subject pairs, {len(q3_cross_valid)} cross-subject pairs (random)
- **Item filtering**: Excluded {n_extreme} items with empirical_p < 0.05 or > 0.95 (unreliable Q3); {len(valid_col_indices)} items retained

## Results

### 1. Within-Subject vs Cross-Subject Q3

| Metric | Within-Subject | Cross-Subject |
|--------|---------------|---------------|
| N pairs | {len(q3_within_valid)} | {len(q3_cross_valid)} |
| Mean Q3 | {q3_within_valid.mean():.4f} | {q3_cross_valid.mean():.4f} |
| Median Q3 | {np.median(q3_within_valid):.4f} | {np.median(q3_cross_valid):.4f} |
| Std Q3 | {q3_within_valid.std():.4f} | {q3_cross_valid.std():.4f} |
| P25 | {np.percentile(q3_within_valid, 25):.4f} | {np.percentile(q3_cross_valid, 25):.4f} |
| P75 | {np.percentile(q3_within_valid, 75):.4f} | {np.percentile(q3_cross_valid, 75):.4f} |
| P95 | {np.percentile(q3_within_valid, 95):.4f} | {np.percentile(q3_cross_valid, 95):.4f} |
| Max | {q3_within_valid.max():.4f} | {q3_cross_valid.max():.4f} |
| |Q3| > 0.20 | {(np.abs(q3_within_valid) > 0.20).mean():.1%} | {(np.abs(q3_cross_valid) > 0.20).mean():.1%} |

**Mann-Whitney U test**: U = {u_stat:.0f}, p = {u_pval:.2e}, rank-biserial r = {r_rb:.4f}

### 2. Top 10 Subjects by Mean Q3

| Subject | N Items | Mean Q3 | Median Q3 | |Q3|>0.20 (%) | Max Q3 |
|---------|---------|---------|-----------|--------------|--------|
"""
    for _, row in subject_df.head(10).iterrows():
        report += f"| {row['subject']} | {row['n_items']:.0f} | {row['mean_q3']:.4f} | {row['median_q3']:.4f} | {row['pct_above_020']:.1f} | {row['q3_max']:.4f} |\n"

    report += f"""
### 3. Bottom 5 Subjects by Mean Q3

| Subject | N Items | Mean Q3 | Median Q3 | |Q3|>0.20 (%) |
|---------|---------|---------|-----------|--------------|
"""
    for _, row in subject_df.tail(5).iterrows():
        report += f"| {row['subject']} | {row['n_items']:.0f} | {row['mean_q3']:.4f} | {row['median_q3']:.4f} | {row['pct_above_020']:.1f} |\n"

    report += f"""
### 4. Q3 vs Subject Size

Spearman correlation between subject size (n_items) and mean Q3: ρ = {corr_size_q3:.4f}, p = {pval_size_q3:.2e}

### 5. Validation: Virology Subject (Full Q3)

"""
    if len(virology_cols) >= 2:
        report += f"""- N items: {len(virology_cols)}
- N pairs (full): {len(vir_valid)}
- Mean Q3: {vir_valid.mean():.4f}
- Median Q3: {np.median(vir_valid):.4f}
- Std Q3: {vir_valid.std():.4f}
- |Q3| > 0.20: {(np.abs(vir_valid) > 0.20).mean():.1%}
- Range: [{vir_valid.min():.4f}, {vir_valid.max():.4f}]
"""

    # Determine overall severity
    overall_mean = q3_within_valid.mean()
    within_pct_020 = (np.abs(q3_within_valid) > 0.20).mean() * 100
    cross_pct_020 = (np.abs(q3_cross_valid) > 0.20).mean() * 100

    report += f"""
## Conclusions

1. **Within-subject Q3 ({q3_within_valid.mean():.4f}) vs cross-subject Q3 ({q3_cross_valid.mean():.4f})**: """

    diff = q3_within_valid.mean() - q3_cross_valid.mean()
    if diff > 0.05:
        report += f"Within-subject pairs show substantially higher residual correlations (Δ = {diff:.4f}), confirming local dependence within MMLU subjects."
    elif diff > 0.02:
        report += f"Within-subject pairs show moderately higher residual correlations (Δ = {diff:.4f}), indicating some local dependence within MMLU subjects."
    elif diff > 0:
        report += f"Within-subject pairs show slightly higher residual correlations (Δ = {diff:.4f}), suggesting mild local dependence."
    else:
        report += f"No meaningful difference between within-subject and cross-subject Q3 (Δ = {diff:.4f})."

    report += f"""

2. **Proportion |Q3| > 0.20**: {within_pct_020:.1f}% of within-subject pairs vs {cross_pct_020:.1f}% of cross-subject pairs exceed the conventional LD threshold. """

    if within_pct_020 > 20:
        report += "This is severe — a large fraction of within-subject item pairs are locally dependent."
    elif within_pct_020 > 10:
        report += "This is moderate — a non-trivial fraction of within-subject item pairs show LD."
    elif within_pct_020 > 5:
        report += "This is mild but present."
    else:
        report += "This is minimal."

    report += f"""

3. **Implications for MH-DIF**: """

    if within_pct_020 > 10 or diff > 0.03:
        report += """Local dependence within subjects inflates variance of the MH statistic, leading to elevated FPR. This is consistent with the observed ~31% baseline FPR in plan_011/012. A bifactor or testlet model that accounts for subject-level clustering would be more appropriate than standard MH-DIF for this data."""
    else:
        report += """The level of local dependence observed is unlikely to fully account for the ~31% baseline FPR seen in plan_011/012. Other factors (e.g., multidimensionality, DIF in the matching variable) may play a larger role."""

    report += f"""

4. **Subject-size effect**: Spearman ρ = {corr_size_q3:.4f} (p = {pval_size_q3:.2e}) between subject size and mean Q3. """

    if abs(corr_size_q3) > 0.3:
        report += "Larger subjects tend to have " + ("higher" if corr_size_q3 > 0 else "lower") + " local dependence."
    else:
        report += "No strong relationship between subject size and degree of local dependence."

    report += "\n"

    elapsed = time.time() - t0
    report += f"\n---\n*Generated in {elapsed:.1f}s with {n_models} models and {n_items} items.*\n"

    with open(OUT / 'plan_013_report.md', 'w') as f:
        f.write(report)
    print(f"  Saved plan_013_report.md")

    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")

if __name__ == '__main__':
    main()
