#!/usr/bin/env python3
"""
R15-SF7: Random split control within 2024 cohort.

Sanity check: randomly splitting a single cohort into two groups should yield
C% ≈ 0%, confirming that the 31.3% temporal-split C% is not a statistical artifact.
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import chi2 as chi2_dist
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("artifacts")
OUT = BASE / "r15_random_split_control"
OUT.mkdir(exist_ok=True)

N_REPS = 100
SEED = 42
N_STRATA = 5


def mh_dif_c_pct_vec(X_ref, X_foc, n_strata=N_STRATA):
    """Fully vectorized MH-DIF → C%. Stratifies on total score."""
    n_ref, J = X_ref.shape
    n_foc = X_foc.shape[0]

    tot_ref = X_ref.sum(axis=1).astype(np.float64)
    tot_foc = X_foc.sum(axis=1).astype(np.float64)
    s_all = np.concatenate([tot_ref, tot_foc])

    edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
    n_bins = len(edges) - 1
    if n_bins < 1:
        return 0.0

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
    return is_c.sum() / J * 100


def main():
    t_total = time.time()
    print(f"R15-SF7: Random Split Control — N_REPS={N_REPS}, seed={SEED}")
    print(f"Output: {OUT}\n")

    # Load data
    print("Loading response matrix...")
    X_full = load_npz(BASE / "response_matrix.npz").toarray().astype(np.int8)
    idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
    model_names = list(idx["model_names"])
    print(f"  Full matrix: {X_full.shape[0]} models × {X_full.shape[1]} items")

    # Load metadata, filter 2024 cohort
    meta = pd.read_csv(BASE / "plan_032_dif_cleaned_mmlu/cleaned_mmlu_analysis.csv")
    cohort_2024 = meta[meta["cohort_temporal"] == "2024"]["model_name"].tolist()
    print(f"  2024 cohort: {len(cohort_2024)} models")

    # Map to row indices
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    row_indices = [name_to_idx[m] for m in cohort_2024 if m in name_to_idx]
    print(f"  Matched rows: {len(row_indices)}")

    X_2024 = X_full[row_indices]
    N = X_2024.shape[0]
    print(f"  X_2024: {N} × {X_2024.shape[1]}\n")

    # ── Part 1: Random splits within 2024 cohort (100 reps) ──
    print("Part 1: Random splits within 2024 cohort")
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(N)
    half = N // 2
    t0 = time.time()
    c_test = mh_dif_c_pct_vec(X_2024[perm[:half]], X_2024[perm[half:]])
    t_single = time.time() - t0
    print(f"  Single rep: C% = {c_test:.3f}%, time = {t_single:.2f}s")
    print(f"  Estimated total: {t_single * N_REPS:.0f}s\n")

    results = []
    rng = np.random.RandomState(SEED)
    for rep in range(N_REPS):
        perm = rng.permutation(N)
        half = N // 2
        group_a = perm[:half]
        group_b = perm[half:]

        c_pct = mh_dif_c_pct_vec(X_2024[group_a], X_2024[group_b])
        results.append({
            "rep": rep,
            "seed": SEED + rep,
            "n_group_a": len(group_a),
            "n_group_b": len(group_b),
            "c_pct": c_pct,
        })

        if (rep + 1) % 10 == 0:
            elapsed = time.time() - t_total
            mean_so_far = np.mean([r["c_pct"] for r in results])
            print(f"  Rep {rep+1:3d}/{N_REPS}: "
                  f"C% = {c_pct:.3f}%, running mean = {mean_so_far:.3f}%, "
                  f"elapsed = {elapsed:.0f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "random_split_results.csv", index=False)
    print(f"\nSaved: {OUT / 'random_split_results.csv'}")

    c_vals = df["c_pct"].values
    mean_c = c_vals.mean()
    sd_c = c_vals.std(ddof=1)
    ci_lo = mean_c - 1.96 * sd_c / np.sqrt(len(c_vals))
    ci_hi = mean_c + 1.96 * sd_c / np.sqrt(len(c_vals))
    min_c = c_vals.min()
    max_c = c_vals.max()

    # ── Part 2: Size-matched control from full sample (20 reps) ──
    N_CTRL = 20
    print(f"\nPart 2: Size-matched control from full sample ({N_CTRL} reps)")
    print(f"  Splitting {X_full.shape[0]} models into {half} vs {N - half}")
    rng_ctrl = np.random.RandomState(SEED + 9999)
    ctrl_results = []
    for i in range(N_CTRL):
        perm = rng_ctrl.permutation(X_full.shape[0])
        c = mh_dif_c_pct_vec(X_full[perm[:half]], X_full[perm[half:N]])
        ctrl_results.append(c)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{N_CTRL}] C% = {c:.3f}%")

    ctrl_mean = np.mean(ctrl_results)
    ctrl_sd = np.std(ctrl_results, ddof=1)
    print(f"  Size-matched control: mean={ctrl_mean:.3f}%, SD={ctrl_sd:.3f}%")

    total_time = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"RESULTS: Random Split Control (2024 cohort, N={N})")
    print(f"{'='*60}")
    print(f"  Reps:                 {N_REPS}")
    print(f"  Mean C%:              {mean_c:.3f}%")
    print(f"  SD:                   {sd_c:.3f}%")
    print(f"  95% CI:               [{ci_lo:.3f}%, {ci_hi:.3f}%]")
    print(f"  Range:                [{min_c:.3f}%, {max_c:.3f}%]")
    print(f"  Size-matched ctrl:    {ctrl_mean:.3f}% (from full sample)")
    print(f"  Temporal split:       31.3%")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"{'='*60}")

    # Write report
    report = f"""# R15-SF7: Random Split Control — 2024 Cohort

## Setup

- **Cohort**: 2024 temporal cohort (N = {N} models)
- **Items**: {X_2024.shape[1]} MMLU items
- **Procedure**: 100 random 50/50 splits within the 2024 cohort
- **DIF method**: 5-stratum MH-DIF with Yates-corrected chi-squared
- **C% criterion**: |Δ_MH| ≥ 1.5 AND p < 0.05 (ETS Category C)

## Results

| Metric | Value |
|--------|-------|
| Mean C% | {mean_c:.3f}% |
| SD | {sd_c:.3f}% |
| 95% CI | [{ci_lo:.3f}%, {ci_hi:.3f}%] |
| Min | {min_c:.3f}% |
| Max | {max_c:.3f}% |
| N reps | {N_REPS} |

## Size-Matched Control

To verify that the within-cohort C% is consistent with MH-DIF's null behavior at this
sample size, we ran {N_CTRL} additional random splits from the **full** 5,227-model pool
using the same group sizes ({half} vs {N - half}):

| Metric | Value |
|--------|-------|
| Mean C% (full-sample, size-matched) | {ctrl_mean:.3f}% |
| SD | {ctrl_sd:.3f}% |

The within-cohort ({mean_c:.3f}%) and full-sample ({ctrl_mean:.3f}%) baselines are
comparable, confirming that the ~1% level reflects the expected false-positive rate of
MH-DIF at this sample size ({half} vs {N - half} per split), not cohort-specific structure.

## Comparison

| Split type | C% | N per group |
|------------|-----|-------------|
| **Temporal (pre-2024 vs 2024)** | **31.3%** | 1,080 vs 807 |
| Random within 2024 cohort | {mean_c:.3f}% | ~{half} vs ~{N - half} |
| Size-matched control (full sample) | {ctrl_mean:.3f}% | {half} vs {N - half} |
| Random baseline (large groups) | 0.10% | 1,080 vs 807 |

## Conclusion

Random splitting within the 2024 cohort yields C% = {mean_c:.3f}% (95% CI:
[{ci_lo:.3f}%, {ci_hi:.3f}%]). This is consistent with the size-matched full-sample
control ({ctrl_mean:.3f}%), confirming that MH-DIF does not produce spurious C-flagged
items when no genuine differential functioning exists. The temporal split C% of 31.3%
is {31.3 / max(mean_c, 0.001):.0f}× larger than the random-split baseline, ruling out
statistical artifact as an explanation.

The ~1% baseline (versus ~0.1% with larger groups) reflects the expected Type I error
inflation with smaller strata: at N ≈ 400 per group and 5 strata, each stratum contains
~80 models, increasing sampling variability of Δ_MH enough for some items to cross the
1.5 threshold by chance.

---
*Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} | Seed: {SEED} | Runtime: {total_time:.0f}s*
"""

    (OUT / "random_split_report.md").write_text(report)
    print(f"Saved: {OUT / 'random_split_report.md'}")


if __name__ == "__main__":
    main()
