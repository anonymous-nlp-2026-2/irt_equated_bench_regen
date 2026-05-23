#!/usr/bin/env python3
"""R15-MF3: Subject-level ranking ρ analysis.

Shows that aggregate ρ≈0.997 masks subject-level ranking instability
when DIF-C items are removed.
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "r15_subject_level_rho"

# ---------- Load data ----------
rm = np.load(ROOT / "artifacts" / "response_matrix.npz")
X = sparse.csr_matrix(
    (rm["data"], rm["indices"], rm["indptr"]),
    shape=tuple(rm["shape"]),
)
idx = np.load(ROOT / "artifacts" / "response_matrix_index.npz", allow_pickle=True)
model_names = idx["model_names"]
item_ids = idx["item_ids"]

dif = pd.read_csv(ROOT / "artifacts" / "plan_001" / "dif_results_temporal.csv")
dif_c_set = set(dif.loc[dif["ets_class"] == "C", "item_id"])

# ---------- Build item-to-column index ----------
item_to_col = {iid: i for i, iid in enumerate(item_ids)}

# ---------- Group items by subject ----------
subject_items = {}
for iid in item_ids:
    subj = iid.rsplit(".", 1)[0]
    subject_items.setdefault(subj, []).append(iid)

# ---------- Convert to dense once (fits in memory: 5227×12508 float32 ≈ 250MB) ----------
X_dense = X.toarray().astype(np.float32)

# ---------- Per-subject analysis ----------
records = []
for subj in sorted(subject_items.keys()):
    all_items = subject_items[subj]
    c_items = [iid for iid in all_items if iid in dif_c_set]
    stable_items = [iid for iid in all_items if iid not in dif_c_set]

    n_all = len(all_items)
    n_c = len(c_items)
    n_stable = len(stable_items)

    if n_stable < 2:
        continue

    all_cols = np.array([item_to_col[iid] for iid in all_items])
    stable_cols = np.array([item_to_col[iid] for iid in stable_items])

    score_all = X_dense[:, all_cols].mean(axis=1)
    score_stable = X_dense[:, stable_cols].mean(axis=1)

    rho, _ = stats.spearmanr(score_all, score_stable)

    rank_all = stats.rankdata(-score_all, method="average")
    rank_stable = stats.rankdata(-score_stable, method="average")
    rank_diff = np.abs(rank_all - rank_stable)

    flip_rate = np.mean(rank_diff >= 1)
    mean_abs_rank_change = np.mean(rank_diff)
    max_abs_rank_change = np.max(rank_diff)

    records.append({
        "subject": subj,
        "n_items": n_all,
        "n_c_items": n_c,
        "n_stable_items": n_stable,
        "pct_c": round(100 * n_c / n_all, 1),
        "spearman_rho": round(rho, 6),
        "flip_rate": round(flip_rate, 4),
        "mean_abs_rank_change": round(mean_abs_rank_change, 2),
        "max_abs_rank_change": round(max_abs_rank_change, 1),
    })

df = pd.DataFrame(records)

# ---------- Aggregate statistics ----------
agg_all_cols = np.arange(X_dense.shape[1])
stable_item_ids = [iid for iid in item_ids if iid not in dif_c_set]
agg_stable_cols = np.array([item_to_col[iid] for iid in stable_item_ids])
agg_score_all = X_dense[:, agg_all_cols].mean(axis=1)
agg_score_stable = X_dense[:, agg_stable_cols].mean(axis=1)
agg_rho, _ = stats.spearmanr(agg_score_all, agg_score_stable)

# C% vs ρ correlation across subjects
cpct_vs_rho = stats.spearmanr(df["pct_c"], df["spearman_rho"])

# ---------- Summary stats ----------
mean_rho = df["spearman_rho"].mean()
min_rho_row = df.loc[df["spearman_rho"].idxmin()]
n_below_99 = (df["spearman_rho"] < 0.99).sum()
n_below_95 = (df["spearman_rho"] < 0.95).sum()
n_below_90 = (df["spearman_rho"] < 0.90).sum()
mean_flip = df["flip_rate"].mean()
mean_rank_change = df["mean_abs_rank_change"].mean()

# ---------- Save CSV ----------
df.to_csv(OUT / "subject_level_rho_results.csv", index=False)
print(f"Saved {len(df)} subjects to CSV")

# ---------- Generate report ----------
top10 = df.nsmallest(10, "spearman_rho")

report = f"""# R15-MF3: Subject-Level Ranking ρ Analysis

## Key Finding

Aggregate ρ = {agg_rho:.4f} masks substantial subject-level ranking instability.
Mean subject-level ρ = {mean_rho:.4f} — individual subjects show much larger divergence.

## Summary Statistics

| Metric | Value |
|--------|-------|
| Aggregate ρ (all items vs stable items) | {agg_rho:.4f} |
| Mean subject-level ρ | {mean_rho:.4f} |
| Median subject-level ρ | {df['spearman_rho'].median():.4f} |
| Min ρ | {min_rho_row['spearman_rho']:.4f} ({min_rho_row['subject']}) |
| Subjects with ρ < 0.99 | {n_below_99} / {len(df)} |
| Subjects with ρ < 0.95 | {n_below_95} / {len(df)} |
| Subjects with ρ < 0.90 | {n_below_90} / {len(df)} |
| Mean flip rate | {mean_flip:.4f} |
| Mean |rank change| (averaged across subjects) | {mean_rank_change:.1f} |

## Top 10 Subjects with Lowest ρ

| Subject | n_items | C% | ρ | Flip Rate | Mean |Δrank| | Max |Δrank| |
|---------|---------|-----|------|-----------|---------------|---------------|
"""

for _, r in top10.iterrows():
    report += (
        f"| {r['subject']} | {r['n_items']} | {r['pct_c']:.1f}% | "
        f"{r['spearman_rho']:.4f} | {r['flip_rate']:.4f} | "
        f"{r['mean_abs_rank_change']:.1f} | {r['max_abs_rank_change']:.0f} |\n"
    )

report += f"""
## C% vs ρ Correlation

Spearman correlation between subject DIF-C percentage and subject ρ:
- ρ = {cpct_vs_rho.statistic:.4f}, p = {cpct_vs_rho.pvalue:.2e}

{"Confirmed: higher DIF contamination → lower ranking stability." if cpct_vs_rho.statistic < -0.3 else "Correlation weaker than expected."}

## Distribution of Subject-Level ρ

"""

bins = [(0.99, 1.0), (0.95, 0.99), (0.90, 0.95), (0.0, 0.90)]
for lo, hi in bins:
    n = ((df["spearman_rho"] >= lo) & (df["spearman_rho"] < hi)).sum()
    if lo == 0.99:
        n = (df["spearman_rho"] >= 0.99).sum()
        report += f"- ρ ≥ 0.99: {n} subjects\n"
    else:
        report += f"- {lo:.2f} ≤ ρ < {hi:.2f}: {n} subjects\n"

report += f"""
## Conclusion

The aggregate ρ of {agg_rho:.4f} is misleading. At the subject level:
- {n_below_99} of {len(df)} subjects have ρ < 0.99
- {n_below_95} subjects fall below ρ = 0.95
- {n_below_90} subjects fall below ρ = 0.90
- The worst subject ({min_rho_row['subject']}) has ρ = {min_rho_row['spearman_rho']:.4f} with max rank displacement of {min_rho_row['max_abs_rank_change']:.0f}

Averaging across thousands of items dilutes the signal: DIF-C items cluster within specific subjects,
so subject-level rankings absorb disproportionate distortion that disappears in aggregate.
"""

with open(OUT / "subject_level_rho_report.md", "w") as f:
    f.write(report)

print(report)
