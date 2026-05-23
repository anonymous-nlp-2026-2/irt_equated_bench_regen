#!/usr/bin/env python3
"""
plan_017 — Downstream Impact Analysis: DIF Item Removal → Model Ranking Stability

Answers the "so what?" question: does removing temporally unstable (DIF) items
produce more stable model rankings?

Parts:
  1. Full vs Stable-only ranking comparison (Spearman ρ, split-half reliability)
  2. Progressive removal by |Δ_MH| magnitude
  3. Random removal baseline (20 replicates per removal fraction)
  4. Rank-mover analysis (models with largest rank shifts)
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import spearmanr
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

BASE = Path("artifacts")
OUT = BASE / "plan_017"
OUT.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
print("=== Loading data ===")
t0 = time.time()

mat = load_npz(BASE / "response_matrix.npz")
X = mat.toarray().astype(np.int16)  # (5227, 12508)
del mat

idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
MODEL_NAMES = idx["model_names"]
ITEM_IDS = idx["item_ids"]

model_meta = pd.read_csv(BASE / "model_metadata.csv")
dif_results = pd.read_csv(BASE / "plan_001" / "dif_results_temporal.csv")

N_MODELS, J = X.shape
print(f"  Matrix: {N_MODELS} models × {J} items")
print(f"  Loaded in {time.time()-t0:.1f}s")

# Build lookup maps
name2row = {n: i for i, n in enumerate(MODEL_NAMES)}
item2col = {iid: i for i, iid in enumerate(ITEM_IDS)}

# ════════════════════════════════════════════════════════════════════════════
#  COHORT ASSIGNMENT
# ════════════════════════════════════════════════════════════════════════════
cohort_map = dict(zip(model_meta["model_name"], model_meta["cohort_temporal"]))
family_map = dict(zip(model_meta["model_name"], model_meta["family"]))

rows_2023 = np.array([name2row[n] for n in MODEL_NAMES if cohort_map.get(n) == "2023"])
rows_2024 = np.array([name2row[n] for n in MODEL_NAMES if cohort_map.get(n) == "2024"])
rows_all = np.arange(N_MODELS)

print(f"  2023 cohort: {len(rows_2023)} models")
print(f"  2024 cohort: {len(rows_2024)} models")

# ════════════════════════════════════════════════════════════════════════════
#  DIF ITEM CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
dif_item_ids = dif_results["item_id"].values
dif_ets = dif_results["ets_class"].values
dif_delta = np.abs(dif_results["delta_mh"].values)

# Map to column indices
dif_col_indices = np.array([item2col[iid] for iid in dif_item_ids])

ets_A = dif_ets == "A"
ets_B = dif_ets == "B"
ets_C = dif_ets == "C"
stable_mask = ets_A | ets_B  # ETS A+B = stable
dif_mask = ets_C             # ETS C = large DIF

stable_cols = dif_col_indices[stable_mask]
dif_cols = dif_col_indices[dif_mask]

print(f"\n=== DIF Classification ===")
print(f"  ETS A (negligible): {ets_A.sum()}")
print(f"  ETS B (moderate):   {ets_B.sum()}")
print(f"  ETS C (large):      {ets_C.sum()}")
print(f"  Stable (A+B):       {len(stable_cols)} items")
print(f"  DIF (C):            {len(dif_cols)} items")

# Sort items by |Δ_MH| descending for progressive removal
sort_idx = np.argsort(-dif_delta)
sorted_cols_by_dif = dif_col_indices[sort_idx]
sorted_delta_by_dif = dif_delta[sort_idx]


# ════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def compute_scores(X_subset, col_indices):
    """Compute total scores for a subset of models on selected columns."""
    return X_subset[:, col_indices].sum(axis=1).astype(np.float64)


def rank_models(scores):
    """Compute dense ranks (1 = highest score)."""
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def spearman_rho(r1, r2):
    """Spearman correlation between two rank vectors."""
    rho, pval = spearmanr(r1, r2)
    return rho, pval


def split_half_reliability(X_subset, col_indices, n_splits=50):
    """Split-half reliability: correlate scores from two random halves of items."""
    n_items = len(col_indices)
    rhos = []
    for _ in range(n_splits):
        perm = np.random.permutation(n_items)
        half1 = col_indices[perm[:n_items // 2]]
        half2 = col_indices[perm[n_items // 2:]]
        s1 = compute_scores(X_subset, half1)
        s2 = compute_scores(X_subset, half2)
        rho, _ = spearmanr(s1, s2)
        rhos.append(rho)
    return np.mean(rhos), np.std(rhos)


def bootstrap_spearman(r1, r2, n_boot=200):
    """Bootstrap CI for Spearman ρ."""
    n = len(r1)
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rho, _ = spearmanr(r1[idx], r2[idx])
        rhos.append(rho)
    rhos = np.array(rhos)
    return np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)


# ════════════════════════════════════════════════════════════════════════════
#  PART 1: FULL vs STABLE-ONLY RANKING
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 1: Full vs Stable-only Ranking ===")
all_cols = np.arange(J)

results_part1 = []
for cohort_name, row_idx in [("2023", rows_2023), ("2024", rows_2024), ("all", rows_all)]:
    X_sub = X[row_idx]

    scores_full = compute_scores(X_sub, all_cols)
    scores_stable = compute_scores(X_sub, stable_cols)

    rank_full = rank_models(scores_full)
    rank_stable = rank_models(scores_stable)

    rho, pval = spearman_rho(rank_full, rank_stable)
    ci_lo, ci_hi = bootstrap_spearman(rank_full, rank_stable)

    shr_full_mean, shr_full_std = split_half_reliability(X_sub, all_cols)
    shr_stable_mean, shr_stable_std = split_half_reliability(X_sub, stable_cols)

    results_part1.append({
        "cohort": cohort_name,
        "n_models": len(row_idx),
        "n_items_full": J,
        "n_items_stable": len(stable_cols),
        "rho_full_vs_stable": rho,
        "rho_ci_lo": ci_lo,
        "rho_ci_hi": ci_hi,
        "p_value": pval,
        "split_half_full_mean": shr_full_mean,
        "split_half_full_std": shr_full_std,
        "split_half_stable_mean": shr_stable_mean,
        "split_half_stable_std": shr_stable_std,
    })
    print(f"  {cohort_name}: ρ(full,stable)={rho:.6f} [{ci_lo:.6f}, {ci_hi:.6f}]  "
          f"SHR_full={shr_full_mean:.4f}  SHR_stable={shr_stable_mean:.4f}")

df_part1 = pd.DataFrame(results_part1)
df_part1.to_csv(OUT / "ranking_comparison.csv", index=False)
print(f"  → {OUT / 'ranking_comparison.csv'}")

# ════════════════════════════════════════════════════════════════════════════
#  PART 2: PROGRESSIVE REMOVAL
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 2: Progressive Removal ===")
removal_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.38]

results_part2 = []
for frac in removal_fracs:
    n_remove = int(frac * J)
    if n_remove == 0:
        remaining_cols = all_cols
    else:
        removed = set(sorted_cols_by_dif[:n_remove])
        remaining_cols = np.array([c for c in all_cols if c not in removed])

    n_remaining = len(remaining_cols)
    delta_threshold = sorted_delta_by_dif[n_remove - 1] if n_remove > 0 else np.nan

    for cohort_name, row_idx in [("2023", rows_2023), ("2024", rows_2024), ("all", rows_all)]:
        X_sub = X[row_idx]

        scores_reduced = compute_scores(X_sub, remaining_cols)
        scores_full = compute_scores(X_sub, all_cols)

        rank_reduced = rank_models(scores_reduced)
        rank_full = rank_models(scores_full)

        rho, pval = spearman_rho(rank_full, rank_reduced)
        ci_lo, ci_hi = bootstrap_spearman(rank_full, rank_reduced)

        results_part2.append({
            "removal_frac": frac,
            "n_removed": n_remove,
            "n_remaining": n_remaining,
            "delta_threshold": delta_threshold,
            "cohort": cohort_name,
            "rho_full_vs_reduced": rho,
            "rho_ci_lo": ci_lo,
            "rho_ci_hi": ci_hi,
        })

    print(f"  frac={frac:.2f}: removed {n_remove}, remaining {n_remaining}")

df_part2 = pd.DataFrame(results_part2)
df_part2.to_csv(OUT / "progressive_removal.csv", index=False)
print(f"  → {OUT / 'progressive_removal.csv'}")

# ════════════════════════════════════════════════════════════════════════════
#  PART 3: RANDOM REMOVAL BASELINE
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 3: Random Removal Baseline ===")
N_RANDOM = 20

results_part3 = []
for frac in removal_fracs:
    n_remove = int(frac * J)
    if n_remove == 0:
        for cohort_name, row_idx in [("2023", rows_2023), ("2024", rows_2024), ("all", rows_all)]:
            results_part3.append({
                "removal_frac": frac,
                "n_removed": 0,
                "cohort": cohort_name,
                "replicate": 0,
                "rho_full_vs_reduced": 1.0,
            })
        continue

    for rep in range(N_RANDOM):
        removed = set(np.random.choice(J, n_remove, replace=False))
        remaining_cols = np.array([c for c in all_cols if c not in removed])

        for cohort_name, row_idx in [("2023", rows_2023), ("2024", rows_2024), ("all", rows_all)]:
            X_sub = X[row_idx]
            scores_reduced = compute_scores(X_sub, remaining_cols)
            scores_full = compute_scores(X_sub, all_cols)

            rank_reduced = rank_models(scores_reduced)
            rank_full = rank_models(scores_full)

            rho, _ = spearman_rho(rank_full, rank_reduced)
            results_part3.append({
                "removal_frac": frac,
                "n_removed": n_remove,
                "cohort": cohort_name,
                "replicate": rep,
                "rho_full_vs_reduced": rho,
            })

    print(f"  frac={frac:.2f}: {N_RANDOM} replicates done")

df_part3 = pd.DataFrame(results_part3)
df_part3.to_csv(OUT / "random_baseline.csv", index=False)
print(f"  → {OUT / 'random_baseline.csv'}")

# ════════════════════════════════════════════════════════════════════════════
#  PART 4: RANK MOVERS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 4: Rank Movers Analysis ===")

rank_movers = []
for cohort_name, row_idx in [("2023", rows_2023), ("2024", rows_2024), ("all", rows_all)]:
    X_sub = X[row_idx]
    names = MODEL_NAMES[row_idx]

    scores_full = compute_scores(X_sub, all_cols)
    scores_stable = compute_scores(X_sub, stable_cols)

    rank_full = rank_models(scores_full)
    rank_stable = rank_models(scores_stable)

    rank_diff = rank_stable - rank_full  # positive = dropped in stable-only ranking

    for i in range(len(row_idx)):
        rank_movers.append({
            "cohort": cohort_name,
            "model_name": names[i],
            "family": family_map.get(names[i], ""),
            "training_method": model_meta.set_index("model_name")["training_method"].get(names[i], ""),
            "accuracy_full": float(scores_full[i]) / J,
            "accuracy_stable": float(scores_stable[i]) / len(stable_cols),
            "rank_full": int(rank_full[i]),
            "rank_stable": int(rank_stable[i]),
            "rank_change": int(rank_diff[i]),
            "abs_rank_change": int(abs(rank_diff[i])),
        })

df_movers = pd.DataFrame(rank_movers)
df_movers.to_csv(OUT / "rank_movers.csv", index=False)

# Top 20 movers per cohort
print("\n  Top 20 rank movers per cohort (by |rank_change|):")
for cohort_name in ["2023", "2024", "all"]:
    sub = df_movers[df_movers["cohort"] == cohort_name].nlargest(20, "abs_rank_change")
    print(f"\n  --- {cohort_name} cohort ---")
    for _, row in sub.head(10).iterrows():
        print(f"    {row['model_name'][:60]:60s}  Δrank={row['rank_change']:+5d}  "
              f"family={str(row['family'])[:20]:20s}  method={str(row['training_method'])[:10]}")

print(f"\n  → {OUT / 'rank_movers.csv'}")

# ════════════════════════════════════════════════════════════════════════════
#  PART 5: CROSS-COHORT RANKING CONSISTENCY (SUPER-FAMILY)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 5: Cross-Cohort Ranking Consistency ===")

SUPER_FAMILY = {
    "llama-2": "llama", "llama-3": "llama",
    "qwen": "qwen", "qwen-1.5": "qwen", "qwen-2": "qwen",
    "yi": "yi", "yi-1.5": "yi",
    "phi-2": "phi", "phi-3": "phi",
    "gemma": "gemma", "gemma-2": "gemma",
    "falcon": "falcon", "falcon-2": "falcon",
    "mistral": "mistral", "mixtral": "mistral",
    "wizardlm": "wizardlm", "wizardlm-2": "wizardlm",
    "openhermes": "hermes", "nous-hermes-2": "hermes",
}

def get_super_family(fam):
    if not fam or fam == "unknown" or pd.isna(fam):
        return None
    return SUPER_FAMILY.get(fam, fam)

sfam_2023 = {}
sfam_2024 = {}
for r in rows_2023:
    fam = family_map.get(MODEL_NAMES[r], "")
    sf = get_super_family(fam)
    if sf:
        sfam_2023.setdefault(sf, []).append(r)
for r in rows_2024:
    fam = family_map.get(MODEL_NAMES[r], "")
    sf = get_super_family(fam)
    if sf:
        sfam_2024.setdefault(sf, []).append(r)

shared_families = sorted(set(sfam_2023.keys()) & set(sfam_2024.keys()))
print(f"  Super-families in both cohorts: {len(shared_families)} — {shared_families}")

cross_cohort_results = []
for sf in shared_families:
    r23 = sfam_2023[sf]
    r24 = sfam_2024[sf]

    acc_full_23 = X[r23][:, all_cols].mean()
    acc_full_24 = X[r24][:, all_cols].mean()
    acc_stable_23 = X[r23][:, stable_cols].mean()
    acc_stable_24 = X[r24][:, stable_cols].mean()
    acc_dif_23 = X[r23][:, dif_cols].mean()
    acc_dif_24 = X[r24][:, dif_cols].mean()

    cross_cohort_results.append({
        "super_family": sf,
        "n_2023": len(r23),
        "n_2024": len(r24),
        "acc_full_2023": acc_full_23,
        "acc_full_2024": acc_full_24,
        "acc_stable_2023": acc_stable_23,
        "acc_stable_2024": acc_stable_24,
        "acc_dif_2023": acc_dif_23,
        "acc_dif_2024": acc_dif_24,
        "delta_full": acc_full_24 - acc_full_23,
        "delta_stable": acc_stable_24 - acc_stable_23,
        "delta_dif": acc_dif_24 - acc_dif_23,
    })

df_cross = pd.DataFrame(cross_cohort_results)
if len(df_cross) > 0:
    print(f"  Mean |Δacc| full:   {df_cross['delta_full'].abs().mean():.4f}")
    print(f"  Mean |Δacc| stable: {df_cross['delta_stable'].abs().mean():.4f}")
    print(f"  Mean |Δacc| DIF:    {df_cross['delta_dif'].abs().mean():.4f}")
    for _, row in df_cross.iterrows():
        print(f"    {row['super_family']:12s}  n23={row['n_2023']:3d}  n24={row['n_2024']:3d}  "
              f"Δfull={row['delta_full']:+.4f}  Δstable={row['delta_stable']:+.4f}  Δdif={row['delta_dif']:+.4f}")

# ════════════════════════════════════════════════════════════════════════════
#  PART 6: DIF vs RANDOM COMPARISON SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 6: DIF vs Random Removal Comparison ===")

comparison_rows = []
for frac in removal_fracs:
    if frac == 0.0:
        continue
    for cohort_name in ["2023", "2024", "all"]:
        dif_rho = df_part2[(df_part2["removal_frac"] == frac) &
                           (df_part2["cohort"] == cohort_name)]["rho_full_vs_reduced"].values[0]

        rand_rhos = df_part3[(df_part3["removal_frac"] == frac) &
                             (df_part3["cohort"] == cohort_name)]["rho_full_vs_reduced"].values
        rand_mean = rand_rhos.mean()
        rand_std = rand_rhos.std()

        # How many SDs below random is DIF removal?
        z_score = (dif_rho - rand_mean) / rand_std if rand_std > 0 else 0.0

        comparison_rows.append({
            "removal_frac": frac,
            "cohort": cohort_name,
            "rho_dif_removal": dif_rho,
            "rho_random_mean": rand_mean,
            "rho_random_std": rand_std,
            "z_score": z_score,
            "dif_lower_than_random": dif_rho < rand_mean,
        })
        print(f"  frac={frac:.2f} {cohort_name:4s}: DIF ρ={dif_rho:.6f}  "
              f"Random ρ={rand_mean:.6f}±{rand_std:.6f}  z={z_score:+.2f}")

df_comparison = pd.DataFrame(comparison_rows)

# ════════════════════════════════════════════════════════════════════════════
#  PART 7: TRAINING METHOD ANALYSIS OF RANK MOVERS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 7: Training Method Pattern in Rank Movers ===")

method_meta = model_meta.set_index("model_name")["training_method"].to_dict()

for cohort_name in ["2023", "2024", "all"]:
    sub = df_movers[df_movers["cohort"] == cohort_name].copy()
    top_movers = sub.nlargest(50, "abs_rank_change")

    methods = top_movers["training_method"].fillna("unknown").value_counts()
    all_methods = sub["training_method"].fillna("unknown").value_counts()

    print(f"\n  {cohort_name} — top-50 movers training method distribution:")
    for m in methods.index[:5]:
        pct_movers = methods[m] / len(top_movers) * 100
        pct_all = all_methods.get(m, 0) / len(sub) * 100
        print(f"    {m:20s}: {pct_movers:5.1f}% in movers vs {pct_all:5.1f}% overall")

# ════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Generating Report ===")

report = []
report.append("# plan_017: Downstream Impact Analysis — DIF Item Removal → Ranking Stability\n")
report.append("## Summary\n")
report.append("This analysis evaluates whether removing temporally unstable (DIF) items from MMLU")
report.append("produces more stable and reliable model rankings.\n")

report.append("\n## Data\n")
report.append(f"- Response matrix: {N_MODELS} models × {J} items")
report.append(f"- Temporal cohorts: 2023 ({len(rows_2023)} models), 2024 ({len(rows_2024)} models)")
report.append(f"- DIF classification: ETS A={ets_A.sum()}, B={ets_B.sum()}, C={ets_C.sum()}")
report.append(f"- Stable items (A+B): {len(stable_cols)} ({len(stable_cols)/J*100:.1f}%)")
report.append(f"- DIF items (C): {len(dif_cols)} ({len(dif_cols)/J*100:.1f}%)\n")

report.append("\n## Part 1: Full vs Stable-Only Ranking\n")
report.append("| Cohort | N models | ρ(full, stable) | 95% CI | SHR full | SHR stable |")
report.append("|--------|----------|-----------------|--------|----------|------------|")
for _, row in df_part1.iterrows():
    report.append(f"| {row['cohort']} | {row['n_models']} | {row['rho_full_vs_stable']:.6f} | "
                  f"[{row['rho_ci_lo']:.6f}, {row['rho_ci_hi']:.6f}] | "
                  f"{row['split_half_full_mean']:.4f}±{row['split_half_full_std']:.4f} | "
                  f"{row['split_half_stable_mean']:.4f}±{row['split_half_stable_std']:.4f} |")

report.append("\n\n## Part 2: Progressive DIF Removal\n")
report.append("| Frac | Removed | Remaining | Cohort | ρ(full, reduced) | 95% CI |")
report.append("|------|---------|-----------|--------|------------------|--------|")
for _, row in df_part2.iterrows():
    report.append(f"| {row['removal_frac']:.2f} | {row['n_removed']} | {row['n_remaining']} | "
                  f"{row['cohort']} | {row['rho_full_vs_reduced']:.6f} | "
                  f"[{row['rho_ci_lo']:.6f}, {row['rho_ci_hi']:.6f}] |")

report.append("\n\n## Part 3: DIF vs Random Removal Comparison\n")
report.append("Negative z-score = DIF removal causes *more* rank disruption than random.\n")
report.append("| Frac | Cohort | ρ DIF | ρ Random (mean±std) | z-score |")
report.append("|------|--------|-------|---------------------|---------|")
for _, row in df_comparison.iterrows():
    report.append(f"| {row['removal_frac']:.2f} | {row['cohort']} | {row['rho_dif_removal']:.6f} | "
                  f"{row['rho_random_mean']:.6f}±{row['rho_random_std']:.6f} | {row['z_score']:+.2f} |")

report.append("\n\n## Part 4: Top Rank Movers (all models, full → stable-only)\n")
report.append("### Biggest rank *drops* (benefited from DIF items)\n")
top_drops = df_movers[df_movers["cohort"] == "all"].nlargest(20, "rank_change")
report.append("| Model | Family | Method | Rank Full | Rank Stable | ΔRank |")
report.append("|-------|--------|--------|-----------|-------------|-------|")
for _, row in top_drops.iterrows():
    report.append(f"| {row['model_name'][:50]} | {str(row['family'])[:15]} | "
                  f"{str(row['training_method'])[:10]} | {row['rank_full']} | "
                  f"{row['rank_stable']} | {row['rank_change']:+d} |")

report.append("\n### Biggest rank *gains* (hurt by DIF items)\n")
top_gains = df_movers[df_movers["cohort"] == "all"].nsmallest(20, "rank_change")
report.append("| Model | Family | Method | Rank Full | Rank Stable | ΔRank |")
report.append("|-------|--------|--------|-----------|-------------|-------|")
for _, row in top_gains.iterrows():
    report.append(f"| {row['model_name'][:50]} | {str(row['family'])[:15]} | "
                  f"{str(row['training_method'])[:10]} | {row['rank_full']} | "
                  f"{row['rank_stable']} | {row['rank_change']:+d} |")

if len(df_cross) > 0:
    report.append("\n\n## Part 5: Cross-Cohort Super-Family Accuracy Shifts\n")
    report.append(f"Super-families in both cohorts: {len(df_cross)}\n")
    report.append(f"- Mean |Δaccuracy| on full benchmark: {df_cross['delta_full'].abs().mean():.4f}")
    report.append(f"- Mean |Δaccuracy| on stable items:   {df_cross['delta_stable'].abs().mean():.4f}")
    report.append(f"- Mean |Δaccuracy| on DIF items:      {df_cross['delta_dif'].abs().mean():.4f}")
    delta_ratio = df_cross['delta_stable'].abs().mean() / df_cross['delta_full'].abs().mean()
    delta_ratio_dif = df_cross['delta_dif'].abs().mean() / df_cross['delta_full'].abs().mean()
    report.append(f"- Ratio stable/full: {delta_ratio:.4f}")
    report.append(f"- Ratio DIF/full: {delta_ratio_dif:.4f}")
    if delta_ratio_dif > 1:
        report.append("- DIF items show *larger* cross-cohort accuracy swings → they capture temporal instability.")
    report.append("\n| Super-family | N23 | N24 | Δacc full | Δacc stable | Δacc DIF |")
    report.append("|-------------|-----|-----|-----------|-------------|----------|")
    for _, row in df_cross.sort_values("delta_full", key=abs, ascending=False).iterrows():
        report.append(f"| {row['super_family'][:20]} | {row['n_2023']} | {row['n_2024']} | "
                      f"{row['delta_full']:+.4f} | {row['delta_stable']:+.4f} | {row['delta_dif']:+.4f} |")

report.append("\n\n## Key Findings\n")

# Summarize
rho_2023 = df_part1[df_part1["cohort"] == "2023"]["rho_full_vs_stable"].values[0]
rho_2024 = df_part1[df_part1["cohort"] == "2024"]["rho_full_vs_stable"].values[0]
shr_full_all = df_part1[df_part1["cohort"] == "all"]["split_half_full_mean"].values[0]
shr_stable_all = df_part1[df_part1["cohort"] == "all"]["split_half_stable_mean"].values[0]

report.append(f"1. **Ranking correlation (full vs stable-only)**: 2023 ρ={rho_2023:.6f}, 2024 ρ={rho_2024:.6f}")
report.append(f"2. **Split-half reliability**: full={shr_full_all:.4f}, stable={shr_stable_all:.4f}")

# Check DIF vs random pattern
n_dif_lower = df_comparison["dif_lower_than_random"].sum()
n_total = len(df_comparison)
report.append(f"3. **DIF removal vs random**: DIF removal caused more rank disruption than random in "
              f"{n_dif_lower}/{n_total} conditions")

mean_z = df_comparison["z_score"].mean()
report.append(f"4. **Average z-score (DIF vs random)**: {mean_z:+.2f}")

if len(df_cross) > 0:
    delta_ratio = df_cross['delta_stable'].abs().mean() / df_cross['delta_full'].abs().mean()
    delta_ratio_dif = df_cross['delta_dif'].abs().mean() / df_cross['delta_full'].abs().mean()
    report.append(f"5. **Cross-cohort stability**: DIF items show {delta_ratio_dif:.2f}× the accuracy shift of full benchmark; "
                  f"stable items show {delta_ratio:.2f}×")

# Top mover patterns
top50_all = df_movers[df_movers["cohort"] == "all"].nlargest(50, "abs_rank_change")
method_counts = top50_all["training_method"].fillna("unknown").value_counts()
report.append(f"6. **Top-50 rank movers** dominant training methods: "
              + ", ".join(f"{m} ({c})" for m, c in method_counts.head(3).items()))

report_text = "\n".join(report)
with open(OUT / "plan_017_report.md", "w") as f:
    f.write(report_text)
print(f"  → {OUT / 'plan_017_report.md'}")

print(f"\n=== All done in {time.time()-t0:.1f}s ===")
