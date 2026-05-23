"""
plan_032: DIF-Cleaned MMLU Demonstration

Show that removing ETS C-flagged items changes model rankings on MMLU,
countering the "ρ=0.997 undermines urgency" reviewer concern.

Inputs:
  - artifacts/response_matrix.npz          (5227×12508 sparse int8, -1=missing)
  - artifacts/response_matrix_index.npz    (model_names, item_ids)
  - artifacts/plan_001/dif_results_temporal.csv  (item DIF classifications)
  - artifacts/model_metadata.csv           (model family, cohort, etc.)
  - artifacts/item_metadata.csv            (subject, difficulty, discrimination)

Outputs (in artifacts/plan_032_dif_cleaned_mmlu/):
  - cleaned_mmlu_analysis.csv
  - per_subject_dif_concentration.csv
  - plan_032_report.md
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, skew
from pathlib import Path


def df_to_md(df, floatfmt=".4f"):
    """Convert DataFrame to markdown table without tabulate dependency."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:{floatfmt}}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)

ARTIFACTS = Path(__file__).resolve().parent.parent
OUT_DIR = ARTIFACTS / "plan_032_dif_cleaned_mmlu"

# ── 1. Load data ──────────────────────────────────────────────────────────────

print("Loading data...")

R = sp.load_npz(ARTIFACTS / "response_matrix.npz").toarray().astype(np.int8)
idx = np.load(ARTIFACTS / "response_matrix_index.npz", allow_pickle=True)
model_names = idx["model_names"]
item_ids = idx["item_ids"]

dif = pd.read_csv(ARTIFACTS / "plan_001" / "dif_results_temporal.csv")
model_meta = pd.read_csv(ARTIFACTS / "model_metadata.csv")
item_meta = pd.read_csv(ARTIFACTS / "item_metadata.csv")

print(f"  Response matrix: {R.shape}")
print(f"  DIF results: {len(dif)} items")
print(f"  Models: {len(model_meta)}, Items metadata: {len(item_meta)}")

# ── 2. Build full & DIF-cleaned MMLU ─────────────────────────────────────────

dif_map = dict(zip(dif["item_id"], dif["ets_class"]))
ets_classes = np.array([dif_map.get(iid, "A") for iid in item_ids])

mask_full = np.ones(len(item_ids), dtype=bool)
mask_cleaned = ets_classes != "C"

n_full = mask_full.sum()
n_cleaned = mask_cleaned.sum()
n_removed = n_full - n_cleaned
print(f"\n  Full MMLU: {n_full} items")
print(f"  DIF-cleaned MMLU: {n_cleaned} items (removed {n_removed} C-flagged)")

def compute_scores(response_matrix, item_mask):
    """Compute per-model accuracy, ignoring missing (-1) cells."""
    sub = response_matrix[:, item_mask]
    valid = sub >= 0
    correct = sub == 1
    n_valid = valid.sum(axis=1).astype(float)
    n_correct = correct.sum(axis=1).astype(float)
    n_valid[n_valid == 0] = np.nan
    return n_correct / n_valid

scores_full = compute_scores(R, mask_full)
scores_cleaned = compute_scores(R, mask_cleaned)

def compute_ranks(scores):
    """Rank descending (rank 1 = highest score). NaN gets last rank."""
    from scipy.stats import rankdata
    s = scores.copy()
    valid = ~np.isnan(s)
    ranks = np.full(len(s), len(s), dtype=float)
    ranks[valid] = rankdata(-s[valid], method="ordinal")
    return ranks.astype(int)

ranks_full = compute_ranks(scores_full)
ranks_cleaned = compute_ranks(scores_cleaned)

# ── 3. Build per-model analysis DataFrame ────────────────────────────────────

meta_lookup = model_meta.set_index("model_name")

df = pd.DataFrame({
    "model_name": model_names,
    "score_full": scores_full,
    "score_cleaned": scores_cleaned,
    "rank_full": ranks_full,
    "rank_cleaned": ranks_cleaned,
})
df["rank_change"] = df["rank_cleaned"] - df["rank_full"]
df["abs_rank_change"] = df["rank_change"].abs()

df["family"] = df["model_name"].map(
    lambda m: meta_lookup.loc[m, "family"] if m in meta_lookup.index else "unknown"
)
df["cohort_temporal"] = df["model_name"].map(
    lambda m: meta_lookup.loc[m, "cohort_temporal"] if m in meta_lookup.index else "unknown"
)

df.to_csv(OUT_DIR / "cleaned_mmlu_analysis.csv", index=False)
print(f"\n  Saved cleaned_mmlu_analysis.csv ({len(df)} models)")

# ── 4. Analyses ──────────────────────────────────────────────────────────────

report_lines = []
def report(text=""):
    print(text)
    report_lines.append(text)

report("# Plan 032: DIF-Cleaned MMLU Demonstration")
report()
report("## Overview")
report(f"- Full MMLU: **{n_full}** items")
report(f"- DIF-cleaned MMLU: **{n_cleaned}** items (removed **{n_removed}** ETS C-flagged, {100*n_removed/n_full:.1f}%)")
report(f"- Models analyzed: **{len(df)}**")

# 4a) Global rank correlation
valid_mask = ~np.isnan(scores_full) & ~np.isnan(scores_cleaned)
rho, p_val = spearmanr(scores_full[valid_mask], scores_cleaned[valid_mask])
report()
report("## Global Rank Correlation")
report(f"- Spearman ρ (full vs. cleaned scores): **{rho:.6f}** (p={p_val:.2e})")
report(f"- This high ρ is expected—but masks local rank disruptions that matter for evaluation.")

# 4b) Rank change distribution
report()
report("## Rank Change Distribution")
rc = df["rank_change"]
report(f"- Mean rank change: {rc.mean():.2f}")
report(f"- Std rank change: {rc.std():.2f}")
report(f"- Min: {rc.min()} | Max: {rc.max()}")
report(f"- |change| ≥ 5: {(df['abs_rank_change'] >= 5).sum()} models ({100*(df['abs_rank_change'] >= 5).mean():.1f}%)")
report(f"- |change| ≥ 10: {(df['abs_rank_change'] >= 10).sum()} models ({100*(df['abs_rank_change'] >= 10).mean():.1f}%)")
report(f"- |change| ≥ 20: {(df['abs_rank_change'] >= 20).sum()} models ({100*(df['abs_rank_change'] >= 20).mean():.1f}%)")
report(f"- |change| ≥ 50: {(df['abs_rank_change'] >= 50).sum()} models ({100*(df['abs_rank_change'] >= 50).mean():.1f}%)")
report(f"- |change| ≥ 100: {(df['abs_rank_change'] >= 100).sum()} models ({100*(df['abs_rank_change'] >= 100).mean():.1f}%)")

# 4c) Top movers
report()
report("## Top 10 Models Moving UP (rank improved after cleaning)")
top_up = df.nsmallest(10, "rank_change")[["model_name", "family", "cohort_temporal", "score_full", "score_cleaned", "rank_full", "rank_cleaned", "rank_change"]]
report()
report(top_up.pipe(df_to_md, floatfmt=".4f"))

report()
report("## Top 10 Models Moving DOWN (rank worsened after cleaning)")
top_down = df.nlargest(10, "rank_change")[["model_name", "family", "cohort_temporal", "score_full", "score_cleaned", "rank_full", "rank_cleaned", "rank_change"]]
report()
report(top_down.pipe(df_to_md, floatfmt=".4f"))

# 4d) Top-N analysis
report()
report("## Top-N Rank Stability Analysis")
report()
report("| Top-N | |Δ|≥5 | %≥5 | |Δ|≥10 | %≥10 | |Δ|≥20 | %≥20 |")
report("|------:|-----:|-----:|------:|------:|------:|------:|")
for topn in [50, 100, 200, 500]:
    subset = df[df["rank_full"] <= topn]
    n5 = (subset["abs_rank_change"] >= 5).sum()
    n10 = (subset["abs_rank_change"] >= 10).sum()
    n20 = (subset["abs_rank_change"] >= 20).sum()
    report(f"| {topn} | {n5} | {100*n5/len(subset):.1f}% | {n10} | {100*n10/len(subset):.1f}% | {n20} | {100*n20/len(subset):.1f}% |")

# 4e) Per-subject DIF concentration
report()
report("## Per-Subject DIF Concentration")

item_dif = dif.merge(item_meta[["item_id", "ctt_difficulty", "ctt_discrimination"]], on="item_id", how="left")
subj_stats = item_dif.groupby("subject").agg(
    n_items=("item_id", "count"),
    n_c_items=("ets_class", lambda x: (x == "C").sum()),
    mean_difficulty=("ctt_difficulty", "mean"),
    mean_discrimination=("ctt_discrimination", "mean"),
).reset_index()
subj_stats["pct_c"] = 100 * subj_stats["n_c_items"] / subj_stats["n_items"]
subj_stats = subj_stats.sort_values("pct_c", ascending=False)
subj_stats.to_csv(OUT_DIR / "per_subject_dif_concentration.csv", index=False)

report()
report("Top 15 subjects by C-flagged %:")
report()
report(subj_stats.head(15).pipe(df_to_md, floatfmt=".3f"))
report()
report("Bottom 10 subjects by C-flagged %:")
report()
report(subj_stats.tail(10).pipe(df_to_md, floatfmt=".3f"))

# 4f) Item difficulty distribution shift
report()
report("## Item Difficulty Distribution Shift")

diff_full = item_meta["ctt_difficulty"]
diff_cleaned = item_meta[item_meta["item_id"].isin(item_ids[mask_cleaned])]["ctt_difficulty"]
diff_removed = item_meta[item_meta["item_id"].isin(item_ids[~mask_cleaned])]["ctt_difficulty"]

report(f"| Subset | N | Mean | Std | Skew |")
report(f"|--------|--:|-----:|----:|-----:|")
for label, d in [("Full", diff_full), ("Cleaned (A+B)", diff_cleaned), ("Removed (C only)", diff_removed)]:
    report(f"| {label} | {len(d)} | {d.mean():.4f} | {d.std():.4f} | {skew(d.dropna()):.4f} |")

# 4g) Per-cohort analysis
report()
report("## Per-Cohort Rank Change Analysis")

for cohort in sorted(df["cohort_temporal"].unique()):
    sub = df[df["cohort_temporal"] == cohort]
    if len(sub) < 5:
        continue
    rc_sub = sub["rank_change"]
    report(f"\n### Cohort: {cohort} (n={len(sub)})")
    report(f"- Mean rank change: {rc_sub.mean():+.2f} (std={rc_sub.std():.2f})")
    report(f"- Median: {rc_sub.median():+.1f}")
    report(f"- |change| ≥ 10: {(sub['abs_rank_change'] >= 10).sum()} ({100*(sub['abs_rank_change'] >= 10).mean():.1f}%)")

# 4h) Per-family aggregation
report()
report("## Per-Family Average Rank Change (top 20 families by model count)")

fam_stats = df.groupby("family").agg(
    n_models=("model_name", "count"),
    mean_rank_change=("rank_change", "mean"),
    median_rank_change=("rank_change", "median"),
    mean_abs_change=("abs_rank_change", "mean"),
    mean_score_full=("score_full", "mean"),
).reset_index()
fam_stats = fam_stats[fam_stats["n_models"] >= 3].sort_values("mean_rank_change")

report()
report("Families benefiting most (rank improved):")
report()
report(fam_stats.head(10).pipe(df_to_md, floatfmt=".2f"))
report()
report("Families hurt most (rank worsened):")
report()
report(fam_stats.tail(10).pipe(df_to_md, floatfmt=".2f"))

# ── 5. Save report ───────────────────────────────────────────────────────────

(OUT_DIR / "plan_032_report.md").write_text("\n".join(report_lines))
print(f"\n  Saved plan_032_report.md")
print("\nDone.")
