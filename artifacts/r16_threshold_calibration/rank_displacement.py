"""Rank displacement analysis across ETS thresholds with random-removal baseline."""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from pathlib import Path

ART = Path(__file__).resolve().parent.parent
OUT = ART / "r16_threshold_calibration"
OUT.mkdir(exist_ok=True)

THRESHOLDS = [1.0, 1.5, 2.0, 2.5]
N_RANDOM_TRIALS = 20
RNG = np.random.default_rng(42)

# --- Load data ---
rm_data = np.load(ART / "response_matrix.npz", allow_pickle=True)
R = sparse.csr_matrix(
    (rm_data["data"], rm_data["indices"], rm_data["indptr"]),
    shape=tuple(rm_data["shape"]),
)
idx = np.load(ART / "response_matrix_index.npz", allow_pickle=True)
model_names = idx["model_names"]
item_ids = idx["item_ids"]

R_dense = R.toarray().astype(np.int8)  # -1=missing, 0=wrong, 1=correct

dif = pd.read_csv(ART / "plan_001" / "dif_results_temporal.csv")

# Build item_id -> column index mapping
item_id_to_col = {iid: i for i, iid in enumerate(item_ids)}
dif_item_cols = dif["item_id"].map(item_id_to_col)
dif = dif[dif_item_cols.notna()].copy()
dif["col_idx"] = dif["item_id"].map(item_id_to_col).astype(int)


def compute_accuracy(R_dense, col_mask):
    """Compute per-model accuracy using only columns in col_mask."""
    sub = R_dense[:, col_mask]
    valid = sub >= 0
    n_valid = valid.sum(axis=1)
    n_correct = ((sub == 1) & valid).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(n_valid > 0, n_correct / n_valid, 0.0)
    return acc


def compute_rankings(acc):
    """Rank by accuracy descending (rank 1 = highest accuracy). Ties use average."""
    return stats.rankdata(-acc, method="average")


def compute_metrics(rank_orig, rank_clean):
    disp = np.abs(rank_orig - rank_clean)
    tau, _ = stats.kendalltau(rank_orig, rank_clean)
    return {
        "avg_abs_rank_displacement": disp.mean(),
        "max_abs_rank_displacement": disp.max(),
        "pairwise_flip_rate": (1 - tau) / 2,
    }


# --- Original ranking (all items) ---
all_cols = np.arange(R_dense.shape[1])
acc_orig = compute_accuracy(R_dense, all_cols)
rank_orig = compute_rankings(acc_orig)

n_total_items = R_dense.shape[1]

results = []

for thresh in THRESHOLDS:
    print(f"Threshold {thresh}...")

    flagged_mask = (dif["delta_mh"].abs() >= thresh) & (dif["p_value"] < 0.05)
    flagged_cols = set(dif.loc[flagged_mask, "col_idx"].values)
    n_flagged = len(flagged_cols)
    pct_flagged = n_flagged / n_total_items

    # Cleaned ranking
    stable_cols = np.array([c for c in range(n_total_items) if c not in flagged_cols])
    acc_clean = compute_accuracy(R_dense, stable_cols)
    rank_clean = compute_rankings(acc_clean)

    m = compute_metrics(rank_orig, rank_clean)

    # Random removal baseline
    rand_metrics = {"avg_abs_rank_displacement": [], "max_abs_rank_displacement": [], "pairwise_flip_rate": []}
    for trial in range(N_RANDOM_TRIALS):
        rand_remove = set(RNG.choice(n_total_items, size=n_flagged, replace=False))
        rand_cols = np.array([c for c in range(n_total_items) if c not in rand_remove])
        acc_rand = compute_accuracy(R_dense, rand_cols)
        rank_rand = compute_rankings(acc_rand)
        rm_ = compute_metrics(rank_orig, rank_rand)
        for k in rand_metrics:
            rand_metrics[k].append(rm_[k])

    row = {
        "threshold": thresh,
        "n_flagged": n_flagged,
        "pct_flagged": round(pct_flagged, 4),
    }
    for k in ["avg_abs_rank_displacement", "max_abs_rank_displacement", "pairwise_flip_rate"]:
        row[k] = round(m[k], 4)
        rm_arr = np.array(rand_metrics[k])
        row[f"random_mean_{k}"] = round(rm_arr.mean(), 4)
        row[f"random_std_{k}"] = round(rm_arr.std(), 4)
        if rm_arr.std() > 0:
            row[f"z_{k}"] = round((m[k] - rm_arr.mean()) / rm_arr.std(), 2)
        else:
            row[f"z_{k}"] = float("nan")

    results.append(row)
    print(f"  flagged={n_flagged} ({pct_flagged:.1%}), avg_disp={m['avg_abs_rank_displacement']:.1f}, "
          f"flip_rate={m['pairwise_flip_rate']:.4f}")

df = pd.DataFrame(results)
df.to_csv(OUT / "rank_displacement_by_threshold.csv", index=False)
print(f"\nSaved to {OUT / 'rank_displacement_by_threshold.csv'}")

# --- Generate report ---
lines = ["# Rank Displacement by ETS Threshold\n"]
lines.append(f"Models: {R_dense.shape[0]}, Items: {n_total_items}, Random trials: {N_RANDOM_TRIALS}\n")

lines.append("## Results\n")
lines.append("| Threshold | Flagged | % | Avg Disp | Max Disp | Flip Rate | Rand Avg Disp | Rand Flip | Z (Disp) | Z (Flip) |")
lines.append("|-----------|---------|---|----------|----------|-----------|---------------|-----------|----------|----------|")
for _, r in df.iterrows():
    lines.append(
        f"| {r['threshold']:.1f} | {int(r['n_flagged'])} | {r['pct_flagged']:.1%} "
        f"| {r['avg_abs_rank_displacement']:.1f} | {int(r['max_abs_rank_displacement'])} "
        f"| {r['pairwise_flip_rate']:.4f} "
        f"| {r['random_mean_avg_abs_rank_displacement']:.1f} "
        f"| {r['random_mean_pairwise_flip_rate']:.4f} "
        f"| {r['z_avg_abs_rank_displacement']:.1f} "
        f"| {r['z_pairwise_flip_rate']:.1f} |"
    )

lines.append("\n## Interpretation\n")
lines.append("- **Avg Disp**: mean absolute rank change per model after removing C-flagged items")
lines.append("- **Flip Rate**: fraction of model pairs whose relative ordering changes (from Kendall tau)")
lines.append("- **Z-score**: how many SDs the DIF-based removal exceeds random removal of equal count")
lines.append("- Z > 2 indicates DIF-flagged items cause significantly more ranking instability than random items\n")

report = "\n".join(lines)
(OUT / "rank_displacement_report.md").write_text(report)
print(f"Saved report to {OUT / 'rank_displacement_report.md'}")
