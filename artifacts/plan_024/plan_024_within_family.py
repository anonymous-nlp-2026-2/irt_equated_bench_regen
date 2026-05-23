#!/usr/bin/env python3
"""
plan_024 — Within-Family Deep Dive: Llama-2 vs Llama-3

Part 1: Llama-2 vs Llama-3 full analysis (score distributions, ability-matched DIF, per-subject breakdown)
Part 2: Other family pairs — descriptive statistics only (underpowered)
Part 3: 2023 vs 2024 temporal cohort distributions for comparison
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import chi2 as chi2_dist, mannwhitneyu
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts")
OUT = BASE / "plan_024"
OUT.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════
print("=== Loading data ===")
mat = load_npz(BASE / "response_matrix.npz")
X = mat.toarray().astype(np.int16)
del mat

idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
MODEL_NAMES = idx["model_names"]
ITEM_IDS = idx["item_ids"]

model_meta = pd.read_csv(BASE / "model_metadata.csv")
item_meta = pd.read_csv(BASE / "item_metadata.csv")

NAME2ROW = {n: i for i, n in enumerate(MODEL_NAMES)}
N_MODELS, J = X.shape
print(f"  Matrix {N_MODELS}×{J}")


# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════
def get_rows(family):
    names = model_meta.loc[model_meta["family"] == family, "model_name"]
    return np.array([NAME2ROW[n] for n in names if n in NAME2ROW])

def total_scores(rows):
    return X[rows].sum(axis=1).astype(np.float64) / J

def kl_divergence(p_hist, q_hist):
    p = p_hist / p_hist.sum()
    q = q_hist / q_hist.sum()
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

def overlap_coefficient(p_hist, q_hist):
    p = p_hist / p_hist.sum()
    q = q_hist / q_hist.sum()
    return float(np.sum(np.minimum(p, q)))

def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def distribution_stats(scores_a, scores_b, label_a, label_b, n_bins=50):
    edges = np.linspace(0, 1, n_bins + 1)
    ha, _ = np.histogram(scores_a, bins=edges)
    hb, _ = np.histogram(scores_b, bins=edges)
    ha_smooth = ha + 1e-10
    hb_smooth = hb + 1e-10
    U, p_mw = mannwhitneyu(scores_a, scores_b, alternative="two-sided")
    return {
        "group_a": label_a, "group_b": label_b,
        "n_a": len(scores_a), "n_b": len(scores_b),
        "mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b)),
        "sd_a": float(np.std(scores_a, ddof=1)), "sd_b": float(np.std(scores_b, ddof=1)),
        "median_a": float(np.median(scores_a)), "median_b": float(np.median(scores_b)),
        "kl_a_b": kl_divergence(ha_smooth, hb_smooth),
        "kl_b_a": kl_divergence(hb_smooth, ha_smooth),
        "overlap_coeff": overlap_coefficient(ha_smooth, hb_smooth),
        "mann_whitney_U": float(U), "mann_whitney_p": float(p_mw),
        "cohens_d": cohens_d(scores_a, scores_b),
    }


# ════════════════════════════════════════════════════════════════════════
#  MH-DIF CORE (from plan_001)
# ════════════════════════════════════════════════════════════════════════
def _nan_row(j):
    return dict(col_idx=j, alpha_mh=np.nan, delta_mh=np.nan,
                se=np.nan, chi2=np.nan, p_value=np.nan, ets_class="A")

def mh_dif(X_ref, X_foc, n_strata=5, col_mask=None):
    _Nr, Jc = X_ref.shape
    tot_ref = X_ref.sum(axis=1).astype(np.float64)
    tot_foc = X_foc.sum(axis=1).astype(np.float64)
    cols = np.where(col_mask)[0] if col_mask is not None else np.arange(Jc)
    rows = []
    t0 = time.time()

    for pos, j in enumerate(cols):
        if pos % 2500 == 0 and pos > 0:
            print(f"    {pos}/{len(cols)}  ({time.time()-t0:.0f}s)")

        s_ref = tot_ref - X_ref[:, j].astype(np.float64)
        s_foc = tot_foc - X_foc[:, j].astype(np.float64)

        s_all = np.concatenate([s_ref, s_foc])
        edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
        if len(edges) < 2:
            rows.append(_nan_row(j))
            continue
        n_bins = len(edges) - 1
        k_ref = np.digitize(s_ref, edges[1:-1])
        k_foc = np.digitize(s_foc, edges[1:-1])

        R = S = 0.0
        sum_diff = sum_var = 0.0
        pr = ps_qr = qs = 0.0
        n_ok = 0

        for k in range(n_bins):
            mr = (k_ref == k)
            mf = (k_foc == k)
            n1 = int(mr.sum())
            n0 = int(mf.sum())
            if n1 == 0 or n0 == 0:
                continue

            A = float(X_ref[mr, j].sum())
            B = float(n1 - A)
            C = float(X_foc[mf, j].sum())
            D = float(n0 - C)
            Nk = float(n1 + n0)
            m1 = A + C
            m0 = B + D
            if m1 == 0 or m0 == 0 or Nk <= 1:
                continue
            n_ok += 1
            Rk = A * D / Nk
            Sk = B * C / Nk
            R += Rk
            S += Sk
            EA = n1 * m1 / Nk
            sum_diff += A - EA
            sum_var += n1 * n0 * m1 * m0 / (Nk * Nk * (Nk - 1))
            Pk = (A + D) / Nk
            Qk = (B + C) / Nk
            pr += Pk * Rk
            ps_qr += Pk * Sk + Qk * Rk
            qs += Qk * Sk

        if n_ok == 0 or (R == 0 and S == 0):
            rows.append(_nan_row(j))
            continue

        if S == 0:
            alpha, delta, se = np.inf, -np.inf, np.nan
        elif R == 0:
            alpha, delta, se = 0.0, np.inf, np.nan
        else:
            alpha = R / S
            delta = -2.35 * np.log(alpha)
            v = pr / (2 * R * R) + ps_qr / (2 * R * S) + qs / (2 * S * S)
            se = 2.35 * np.sqrt(v) if v > 0 else np.nan

        if sum_var > 0:
            chi2_stat = max(abs(sum_diff) - 0.5, 0) ** 2 / sum_var
            pval = 1 - chi2_dist.cdf(chi2_stat, df=1)
        else:
            chi2_stat = pval = np.nan

        ad = abs(delta) if np.isfinite(delta) else 0.0
        if ad < 1.0:
            ets = "A"
        elif ad >= 1.5 and (not np.isnan(pval)) and pval < 0.05:
            ets = "C"
        else:
            ets = "B"

        rows.append(dict(col_idx=j, alpha_mh=alpha, delta_mh=delta,
                         se=se, chi2=chi2_stat, p_value=pval, ets_class=ets))

    elapsed = time.time() - t0
    print(f"    done — {len(cols)} items in {elapsed:.1f}s")
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
#  PART 1: Llama-2 vs Llama-3 Full Analysis
# ════════════════════════════════════════════════════════════════════════
print("\n=== Part 1: Llama-2 vs Llama-3 ===")

rows_l2 = get_rows("llama-2")
rows_l3 = get_rows("llama-3")
print(f"  Llama-2: N={len(rows_l2)}, Llama-3: N={len(rows_l3)}")

scores_l2 = total_scores(rows_l2)
scores_l3 = total_scores(rows_l3)

# 1a. Distribution stats
dist_llama = distribution_stats(scores_l2, scores_l3, "Llama-2", "Llama-3")
print(f"  Llama-2 mean={dist_llama['mean_a']:.4f} SD={dist_llama['sd_a']:.4f}")
print(f"  Llama-3 mean={dist_llama['mean_b']:.4f} SD={dist_llama['sd_b']:.4f}")
print(f"  KL(L2||L3)={dist_llama['kl_a_b']:.4f}, KL(L3||L2)={dist_llama['kl_b_a']:.4f}")
print(f"  Overlap={dist_llama['overlap_coeff']:.4f}, Cohen's d={dist_llama['cohens_d']:.4f}")
print(f"  Mann-Whitney p={dist_llama['mann_whitney_p']:.2e}")

# 1b. Unmatched DIF (full groups)
print("\n  --- Unmatched MH-DIF (Llama-2 ref, Llama-3 foc) ---")
X_l2 = X[rows_l2]
X_l3 = X[rows_l3]
dif_unmatched = mh_dif(X_l2, X_l3, n_strata=5)
dif_unmatched["item_id"] = [ITEM_IDS[i] for i in dif_unmatched["col_idx"]]
dif_unmatched = dif_unmatched.merge(item_meta[["item_id", "subject"]], on="item_id", how="left")

c_pct_unmatched = (dif_unmatched["ets_class"] == "C").mean() * 100
b_pct_unmatched = (dif_unmatched["ets_class"] == "B").mean() * 100
a_pct_unmatched = (dif_unmatched["ets_class"] == "A").mean() * 100
print(f"  Unmatched C%={c_pct_unmatched:.1f}%, B%={b_pct_unmatched:.1f}%, A%={a_pct_unmatched:.1f}%")

# 1c. Ability-Matched DIF
print("\n  --- Ability-Matched MH-DIF (quintile matching, 5 seeds) ---")
n_seeds = 5
matched_results = []

for seed in range(n_seeds):
    rng = np.random.RandomState(42 + seed)
    # quintile bins on pooled scores
    all_scores = np.concatenate([scores_l2, scores_l3])
    q_edges = np.quantile(all_scores, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    q_edges[-1] += 1e-6

    q_l2 = np.digitize(scores_l2, q_edges[1:-1])
    q_l3 = np.digitize(scores_l3, q_edges[1:-1])

    matched_l3_idx = []
    matched_l2_idx = []
    for q in range(5):
        idx_l2_q = np.where(q_l2 == q)[0]
        idx_l3_q = np.where(q_l3 == q)[0]
        n_match = min(len(idx_l2_q), len(idx_l3_q))
        if n_match == 0:
            continue
        sel_l2 = rng.choice(idx_l2_q, size=n_match, replace=False)
        sel_l3 = rng.choice(idx_l3_q, size=n_match, replace=False)
        matched_l2_idx.extend(sel_l2)
        matched_l3_idx.extend(sel_l3)

    matched_l2_idx = np.array(matched_l2_idx)
    matched_l3_idx = np.array(matched_l3_idx)
    X_l2_m = X[rows_l2[matched_l2_idx]]
    X_l3_m = X[rows_l3[matched_l3_idx]]

    print(f"  Seed {seed}: matched N_ref={len(matched_l2_idx)}, N_foc={len(matched_l3_idx)}")
    print(f"    Matched Llama-2 mean={total_scores(rows_l2[matched_l2_idx]).mean():.4f}, "
          f"Llama-3 mean={total_scores(rows_l3[matched_l3_idx]).mean():.4f}")

    dif_m = mh_dif(X_l2_m, X_l3_m, n_strata=5)
    c_pct = (dif_m["ets_class"] == "C").mean() * 100
    b_pct = (dif_m["ets_class"] == "B").mean() * 100
    matched_results.append({"seed": seed, "n_matched": len(matched_l2_idx),
                            "c_pct": c_pct, "b_pct": b_pct})
    print(f"    C%={c_pct:.1f}%, B%={b_pct:.1f}%")

matched_df = pd.DataFrame(matched_results)
mean_c = matched_df["c_pct"].mean()
sd_c = matched_df["c_pct"].std()
print(f"\n  Matched C%: {mean_c:.1f} ± {sd_c:.1f} (vs unmatched {c_pct_unmatched:.1f}%)")

# 1d. Per-subject C% breakdown (top-10 subjects by item count)
print("\n  --- Per-subject C% (Llama-2 vs Llama-3, unmatched) ---")
subj_counts = dif_unmatched["subject"].value_counts()
top10_subj = subj_counts.head(10).index.tolist()

per_subject = []
for subj in sorted(dif_unmatched["subject"].unique()):
    sub = dif_unmatched[dif_unmatched["subject"] == subj]
    n_items = len(sub)
    n_c = (sub["ets_class"] == "C").sum()
    per_subject.append({"subject": subj, "n_items": n_items,
                        "n_C": n_c, "C_pct": n_c / n_items * 100 if n_items > 0 else 0})

per_subject_df = pd.DataFrame(per_subject).sort_values("C_pct", ascending=False)
per_subject_df.to_csv(OUT / "llama_per_subject_dif.csv", index=False)
print("  Top-10 by C%:")
for _, r in per_subject_df.head(10).iterrows():
    print(f"    {r['subject']:40s}  C%={r['C_pct']:5.1f}  ({r['n_C']}/{r['n_items']})")


# ════════════════════════════════════════════════════════════════════════
#  PART 2: Other Family Pairs (Descriptive Only)
# ════════════════════════════════════════════════════════════════════════
print("\n=== Part 2: Other Family Pairs (Descriptive) ===")

family_pairs = [
    ("phi-2", "phi-3"),
    ("qwen", "qwen-1.5"),
    ("gemma", "gemma-2"),
    ("mistral", "mixtral"),
]

other_pairs = []
for fam_a, fam_b in family_pairs:
    rows_a = get_rows(fam_a)
    rows_b = get_rows(fam_b)
    if len(rows_a) == 0 or len(rows_b) == 0:
        print(f"  {fam_a} vs {fam_b}: skipped (empty group)")
        continue
    sa = total_scores(rows_a)
    sb = total_scores(rows_b)
    stats = distribution_stats(sa, sb, fam_a, fam_b)
    stats["underpowered"] = min(len(rows_a), len(rows_b)) < 50
    other_pairs.append(stats)
    print(f"  {fam_a}(N={len(rows_a)}) vs {fam_b}(N={len(rows_b)}): "
          f"mean {stats['mean_a']:.3f} vs {stats['mean_b']:.3f}, "
          f"overlap={stats['overlap_coeff']:.3f}, d={stats['cohens_d']:.3f}"
          f"{'  [UNDERPOWERED]' if stats['underpowered'] else ''}")

other_pairs_df = pd.DataFrame(other_pairs)
other_pairs_df.to_csv(OUT / "other_family_pairs.csv", index=False)


# ════════════════════════════════════════════════════════════════════════
#  PART 3: 2023 vs 2024 Temporal Cohort
# ════════════════════════════════════════════════════════════════════════
print("\n=== Part 3: 2023 vs 2024 Temporal Cohort ===")

rows_2023 = np.array([NAME2ROW[n] for n in
    model_meta.loc[model_meta["cohort_temporal"] == "2023", "model_name"] if n in NAME2ROW])
rows_2024 = np.array([NAME2ROW[n] for n in
    model_meta.loc[model_meta["cohort_temporal"] == "2024", "model_name"] if n in NAME2ROW])

scores_2023 = total_scores(rows_2023)
scores_2024 = total_scores(rows_2024)
dist_temporal = distribution_stats(scores_2023, scores_2024, "2023", "2024")

print(f"  2023(N={len(rows_2023)}): mean={dist_temporal['mean_a']:.4f} SD={dist_temporal['sd_a']:.4f}")
print(f"  2024(N={len(rows_2024)}): mean={dist_temporal['mean_b']:.4f} SD={dist_temporal['sd_b']:.4f}")
print(f"  KL(23||24)={dist_temporal['kl_a_b']:.4f}, overlap={dist_temporal['overlap_coeff']:.4f}")
print(f"  Cohen's d={dist_temporal['cohens_d']:.4f}, Mann-Whitney p={dist_temporal['mann_whitney_p']:.2e}")


# ════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS + REPORT
# ════════════════════════════════════════════════════════════════════════
dif_unmatched.to_csv(OUT / "llama_dif_unmatched.csv", index=False)
matched_df.to_csv(OUT / "llama_matched_dif_summary.csv", index=False)

# Build report
report_lines = [
    "# Plan 024: Within-Family Deep Dive — Llama-2 vs Llama-3",
    "",
    "## Part 1: Llama-2 vs Llama-3",
    "",
    "### Score Distributions",
    "",
    f"| Metric | Llama-2 (N={len(rows_l2)}) | Llama-3 (N={len(rows_l3)}) |",
    "|--------|------------|------------|",
    f"| Mean accuracy | {dist_llama['mean_a']:.4f} | {dist_llama['mean_b']:.4f} |",
    f"| SD | {dist_llama['sd_a']:.4f} | {dist_llama['sd_b']:.4f} |",
    f"| Median | {dist_llama['median_a']:.4f} | {dist_llama['median_b']:.4f} |",
    "",
    f"- **KL(Llama-2 || Llama-3)** = {dist_llama['kl_a_b']:.4f}",
    f"- **KL(Llama-3 || Llama-2)** = {dist_llama['kl_b_a']:.4f}",
    f"- **Overlap coefficient** = {dist_llama['overlap_coeff']:.4f}",
    f"- **Mann-Whitney U** = {dist_llama['mann_whitney_U']:.0f}, p = {dist_llama['mann_whitney_p']:.2e}",
    f"- **Cohen's d** = {dist_llama['cohens_d']:.4f}",
    "",
    "### Unmatched MH-DIF",
    "",
    f"- A% = {a_pct_unmatched:.1f}%, B% = {b_pct_unmatched:.1f}%, **C% = {c_pct_unmatched:.1f}%**",
    "",
    "### Ability-Matched MH-DIF (Quintile Matching, 5 Seeds)",
    "",
    "| Seed | N matched | C% | B% |",
    "|------|-----------|-----|-----|",
]
for _, r in matched_df.iterrows():
    report_lines.append(f"| {int(r['seed'])} | {int(r['n_matched'])} | {r['c_pct']:.1f} | {r['b_pct']:.1f} |")

report_lines += [
    "",
    f"**Mean matched C% = {mean_c:.1f} ± {sd_c:.1f}** (vs unmatched {c_pct_unmatched:.1f}%)",
    "",
]

if mean_c < c_pct_unmatched * 0.6:
    interpretation = (f"Matched C% ({mean_c:.1f}%) is substantially lower than unmatched ({c_pct_unmatched:.1f}%), "
                      f"indicating ability confound is a major contributor to the observed DIF.")
elif mean_c > c_pct_unmatched * 0.8:
    interpretation = (f"Matched C% ({mean_c:.1f}%) remains close to unmatched ({c_pct_unmatched:.1f}%), "
                      f"indicating genuine within-family DIF beyond ability differences.")
else:
    interpretation = (f"Matched C% ({mean_c:.1f}%) is moderately lower than unmatched ({c_pct_unmatched:.1f}%), "
                      f"suggesting both ability confound and genuine within-family DIF contribute.")

report_lines += [
    f"**Interpretation:** {interpretation}",
    "",
    "### Per-Subject C% (Top 10, Unmatched)",
    "",
    "| Subject | N items | N C | C% |",
    "|---------|---------|-----|-----|",
]
for _, r in per_subject_df.head(10).iterrows():
    report_lines.append(f"| {r['subject']} | {r['n_items']} | {r['n_C']} | {r['C_pct']:.1f} |")

report_lines += [
    "",
    "## Part 2: Other Family Pairs (Descriptive Only)",
    "",
    "| Pair | N_a | N_b | Mean_a | Mean_b | Overlap | Cohen's d | Underpowered |",
    "|------|-----|-----|--------|--------|---------|-----------|-------------|",
]
for _, r in other_pairs_df.iterrows():
    report_lines.append(
        f"| {r['group_a']} vs {r['group_b']} | {r['n_a']} | {r['n_b']} | "
        f"{r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['overlap_coeff']:.3f} | "
        f"{r['cohens_d']:.3f} | {'Yes' if r['underpowered'] else 'No'} |"
    )

report_lines += [
    "",
    "DIF analysis omitted for pairs with N < 50 in at least one group.",
    "",
    "## Part 3: 2023 vs 2024 Temporal Cohort",
    "",
    f"| Metric | 2023 (N={len(rows_2023)}) | 2024 (N={len(rows_2024)}) |",
    "|--------|------|------|",
    f"| Mean accuracy | {dist_temporal['mean_a']:.4f} | {dist_temporal['mean_b']:.4f} |",
    f"| SD | {dist_temporal['sd_a']:.4f} | {dist_temporal['sd_b']:.4f} |",
    f"| Median | {dist_temporal['median_a']:.4f} | {dist_temporal['median_b']:.4f} |",
    "",
    f"- **KL(2023 || 2024)** = {dist_temporal['kl_a_b']:.4f}",
    f"- **KL(2024 || 2023)** = {dist_temporal['kl_b_a']:.4f}",
    f"- **Overlap coefficient** = {dist_temporal['overlap_coeff']:.4f}",
    f"- **Mann-Whitney U** = {dist_temporal['mann_whitney_U']:.0f}, p = {dist_temporal['mann_whitney_p']:.2e}",
    f"- **Cohen's d** = {dist_temporal['cohens_d']:.4f}",
    "",
    "### Comparison with Llama-2 vs Llama-3",
    "",
    "| Metric | Llama-2 vs 3 | 2023 vs 2024 |",
    "|--------|-------------|-------------|",
    f"| KL divergence (fwd) | {dist_llama['kl_a_b']:.4f} | {dist_temporal['kl_a_b']:.4f} |",
    f"| Overlap coefficient | {dist_llama['overlap_coeff']:.4f} | {dist_temporal['overlap_coeff']:.4f} |",
    f"| Cohen's d | {dist_llama['cohens_d']:.4f} | {dist_temporal['cohens_d']:.4f} |",
    f"| Mann-Whitney p | {dist_llama['mann_whitney_p']:.2e} | {dist_temporal['mann_whitney_p']:.2e} |",
    "",
]

with open(OUT / "within_family_report.md", "w") as f:
    f.write("\n".join(report_lines))

print(f"\n=== Done. Outputs in {OUT} ===")
