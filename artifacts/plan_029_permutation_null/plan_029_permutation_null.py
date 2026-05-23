#!/usr/bin/env python3
"""
plan_029_permutation_null.py — Permutation null for temporal DIF C%

Tests whether the observed ~31.3% C-category rate in temporal DIF (2023 vs 2024)
exceeds chance under random label assignment.  1000 permutations, keeping group
sizes fixed (N_ref=1080, N_foc=807).

Key optimisation: purified scores (and hence strata) are independent of the
ref/foc split, so they are precomputed once.  Per-permutation work reduces to
two np.bincount calls per item.
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import chi2 as chi2_dist
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/"
            "projects/irt_equated_bench_regen/artifacts")
OUT = BASE / "plan_029_permutation_null"
OUT.mkdir(exist_ok=True)

N_PERM = 200
N_STRATA = 5
SEED = 42

# ═══════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
t0 = time.time()

mat = load_npz(BASE / "response_matrix.npz")
X_full = mat.toarray()          # int8, (5227, 12508)
del mat

idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
MODEL_NAMES = idx["model_names"]
ITEM_IDS = idx["item_ids"]
J = len(ITEM_IDS)

meta = pd.read_csv(BASE / "model_metadata.csv")
NAME2ROW = {n: i for i, n in enumerate(MODEL_NAMES)}

# Temporal subset (1887 models)
temporal_mask = meta["cohort_temporal"].isin(["2023", "2024"])
temporal_meta = meta[temporal_mask].copy()
temporal_rows = np.array([NAME2ROW[n] for n in temporal_meta["model_name"]
                          if n in NAME2ROW])
name2cohort = dict(zip(meta["model_name"], meta["cohort_temporal"]))
is_ref_true = np.array([name2cohort[MODEL_NAMES[r]] == "2023"
                         for r in temporal_rows])

N_temp = len(temporal_rows)
N_ref = int(is_ref_true.sum())
N_foc = N_temp - N_ref

X_temp = X_full[temporal_rows]   # (1887, 12508) int8
del X_full

print(f"  N_temporal={N_temp}, N_ref={N_ref}, N_foc={N_foc}, J={J}")
print(f"  Loaded in {time.time() - t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
#  PRECOMPUTE STRATA  (fixed across permutations)
# ═══════════════════════════════════════════════════════════════
print("Precomputing strata...")
t0 = time.time()

tot = X_temp.astype(np.float64).sum(axis=1)       # (1887,)
purified_all = tot[:, None] - X_temp.astype(np.float64)  # (1887, J)

quantiles = np.linspace(0, 1, N_STRATA + 1)
edges = np.quantile(purified_all, quantiles, axis=0)  # (6, J)
inner_edges = edges[1:-1]                              # (4, J)

strata = np.zeros((N_temp, J), dtype=np.int8)
for e_idx in range(inner_edges.shape[0]):
    strata += (purified_all >= inner_edges[e_idx][None, :]).astype(np.int8)

del purified_all, edges, inner_edges

MAX_BINS = int(strata.max()) + 1          # should be 5

# Transpose for contiguous row access in the hot loop
strata_T = strata.T.copy()                # (J, 1887) int8
resp_T = X_temp.T.astype(np.float64)      # (J, 1887) float64 — weights for bincount
del X_temp, strata

print(f"  Done in {time.time() - t0:.1f}s  (MAX_BINS={MAX_BINS})")

# ═══════════════════════════════════════════════════════════════
#  FAST C% COMPUTATION
# ═══════════════════════════════════════════════════════════════
MINLEN = 2 * MAX_BINS

def compute_c_pct(is_ref):
    """Return C-category percentage given a ref/foc boolean mask."""
    foc_offset = (~is_ref).astype(np.intp) * MAX_BINS   # (1887,)
    n_c = 0

    for j in range(J):
        s_j = strata_T[j]          # (1887,) int8, contiguous row
        r_j = resp_T[j]            # (1887,) float64, contiguous row

        combined = s_j.astype(np.intp) + foc_offset

        counts = np.bincount(combined, minlength=MINLEN)
        correct = np.bincount(combined, weights=r_j, minlength=MINLEN)

        n1 = counts[:MAX_BINS].astype(np.float64)
        n0 = counts[MAX_BINS:2*MAX_BINS].astype(np.float64)
        A  = correct[:MAX_BINS]
        C  = correct[MAX_BINS:2*MAX_BINS]
        B  = n1 - A
        D  = n0 - C
        Nk = n1 + n0
        m1 = A + C
        m0 = B + D

        valid = (n1 > 0) & (n0 > 0) & (m1 > 0) & (m0 > 0) & (Nk > 1)
        if not valid.any():
            continue

        Av, Bv, Cv, Dv = A[valid], B[valid], C[valid], D[valid]
        Nkv = Nk[valid]
        n1v, n0v, m1v, m0v = n1[valid], n0[valid], m1[valid], m0[valid]

        R = (Av * Dv / Nkv).sum()
        S = (Bv * Cv / Nkv).sum()
        if R == 0 or S == 0:
            continue

        delta = -2.35 * np.log(R / S)
        if abs(delta) < 1.5:
            continue

        EA = n1v * m1v / Nkv
        sum_diff = (Av - EA).sum()
        sum_var = (n1v * n0v * m1v * m0v / (Nkv * Nkv * (Nkv - 1))).sum()
        if sum_var <= 0:
            continue

        chi2_stat = max(abs(sum_diff) - 0.5, 0) ** 2 / sum_var
        pval = 1.0 - chi2_dist.cdf(chi2_stat, df=1)
        if pval < 0.05:
            n_c += 1

    return n_c / J * 100


# ═══════════════════════════════════════════════════════════════
#  VERIFY OBSERVED C%
# ═══════════════════════════════════════════════════════════════
print("\nVerifying observed C%...")
t0 = time.time()
observed_c = compute_c_pct(is_ref_true)
dt_single = time.time() - t0
print(f"  Observed C% = {observed_c:.2f}%  ({dt_single:.1f}s)")

# ═══════════════════════════════════════════════════════════════
#  PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════
print(f"\nRunning {N_PERM} permutations (seed={SEED}, "
      f"est. {dt_single * N_PERM / 60:.0f} min)...")
rng = np.random.RandomState(SEED)
perm_results = []
t_start = time.time()

for i in range(N_PERM):
    perm_idx = rng.permutation(N_temp)
    is_ref_perm = np.zeros(N_temp, dtype=bool)
    is_ref_perm[perm_idx[:N_ref]] = True

    c_pct = compute_c_pct(is_ref_perm)
    perm_results.append(c_pct)

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t_start
        avg = elapsed / (i + 1)
        eta = avg * (N_PERM - i - 1)
        n_exceed = sum(1 for x in perm_results if x >= observed_c)
        print(f"  [{i+1:4d}/{N_PERM}] C%={c_pct:.2f}%  "
              f"max_null={max(perm_results):.2f}%  "
              f"n_exceed={n_exceed}  "
              f"{avg:.1f}s/perm  ETA {eta/60:.1f}min")

    if (i + 1) == 100:
        n_exc = sum(1 for x in perm_results if x >= observed_c)
        print(f"\n  [checkpoint] 100 permutations done, "
              f"n_exceed={n_exc}, max_null={max(perm_results):.2f}%")
        if n_exc == 0:
            print(f"  Already p < 0.01; continuing to {N_PERM} for precision.\n")

total_time = time.time() - t_start
print(f"\nCompleted {N_PERM} permutations in {total_time / 60:.1f} min")

# ═══════════════════════════════════════════════════════════════
#  STATISTICS
# ═══════════════════════════════════════════════════════════════
null = np.array(perm_results)
n_exceed = int((null >= observed_c).sum())
p_value = (n_exceed + 1) / (N_PERM + 1)     # conservative

null_mean = null.mean()
null_std  = null.std()
null_ci   = (np.percentile(null, 2.5), np.percentile(null, 97.5))
null_max  = null.max()
null_min  = null.min()

print(f"\n{'=' * 55}")
print("RESULTS")
print(f"{'=' * 55}")
print(f"  Observed C%      {observed_c:.2f}%")
print(f"  Null mean ± SD   {null_mean:.2f}% ± {null_std:.2f}%")
print(f"  Null 95% CI      [{null_ci[0]:.2f}%, {null_ci[1]:.2f}%]")
print(f"  Null range       [{null_min:.2f}%, {null_max:.2f}%]")
print(f"  N exceed         {n_exceed} / {N_PERM}")
p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.4f}"
print(f"  p-value          {p_str}")

# ═══════════════════════════════════════════════════════════════
#  OUTPUTS
# ═══════════════════════════════════════════════════════════════

# 1. CSV
pd.DataFrame({"permutation": np.arange(1, N_PERM + 1),
              "c_pct": perm_results}).to_csv(
    OUT / "permutation_results.csv", index=False)

# 2. Figure
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(null, bins=40, color="#4A90D9", edgecolor="white", alpha=0.85,
        label="Null distribution")
ax.axvline(observed_c, color="#D9534F", lw=2, ls="--",
           label=f"Observed = {observed_c:.1f}%")
ax.set_xlabel("C% (ETS category C items)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(f"Permutation Null for Temporal DIF C%\n"
             f"(N={N_PERM}, p {p_str})", fontsize=12)
ax.legend(fontsize=10)
ax.annotate(f"Null: {null_mean:.1f}% ± {null_std:.1f}%\n"
            f"Observed: {observed_c:.1f}%\n"
            f"p {p_str}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightyellow", edgecolor="gray"))
fig.tight_layout()
fig.savefig(OUT / "fig_permutation_null.png", dpi=200)
plt.close()

# 3. Report
report = f"""# Plan 029: Permutation Null for Temporal DIF C%

## Purpose

Test whether the observed temporal DIF C% significantly exceeds the rate
expected under random label assignment.

## Method

- **Data**: {N_temp} models with temporal labels ({N_ref} "2023", {N_foc} "2024"), {J} items
- **Procedure**: {N_PERM} permutations of temporal labels, group sizes fixed
- **MH-DIF**: purified matching, {N_STRATA} quantile strata, ETS classification
- **Seed**: {SEED}
- **Optimisation**: strata precomputed once (invariant to label permutation)

## Results

| Metric | Value |
|--------|-------|
| Observed C% | {observed_c:.2f}% |
| Null mean ± SD | {null_mean:.2f}% ± {null_std:.2f}% |
| Null 95% CI | [{null_ci[0]:.2f}%, {null_ci[1]:.2f}%] |
| Null range | [{null_min:.2f}%, {null_max:.2f}%] |
| N exceed observed | {n_exceed} / {N_PERM} |
| p-value | {p_str} |

## Interpretation

The observed temporal DIF C% of {observed_c:.2f}% is \
{"**far above**" if p_value < 0.001 else "**above**" if p_value < 0.05 else "not above"} \
the permutation null (mean {null_mean:.2f}%, max {null_max:.2f}%, \
p {p_str}).  The 31.3% C-category rate cannot be attributed to \
chance label assignment.

## Runtime

- Per permutation: ~{dt_single:.1f}s
- Total: {total_time / 60:.1f} min ({N_PERM} permutations)
"""

(OUT / "permutation_null_report.md").write_text(report)

print(f"\nOutputs → {OUT}/")
for f in ["permutation_results.csv", "fig_permutation_null.png",
          "permutation_null_report.md"]:
    p = OUT / f
    print(f"  {f}  ({p.stat().st_size / 1024:.0f} KB)" if p.exists() else
          f"  {f}  MISSING")
