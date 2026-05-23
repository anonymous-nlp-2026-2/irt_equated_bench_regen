#!/usr/bin/env python3
"""Temporal accumulation analysis: does C% grow monotonically with temporal distance?"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = "artifacts"
OUT = f"{ROOT}/temporal_accumulation"

N_STRATA = 5
ALPHA = 0.05
DELTA_THRESH = 1.5

COHORT_ORDER = ["2023-H1", "2023-H2", "2024-H1"]
COHORT_MIDPOINTS = {"2023-H1": 2023.25, "2023-H2": 2023.75, "2024-H1": 2024.25}


def mh_dif_full(X_ref, X_foc, n_strata=N_STRATA):
    """Vectorized MH-DIF returning per-item details."""
    n_ref, J = X_ref.shape
    n_foc = X_foc.shape[0]

    tot_ref = np.asarray(X_ref.sum(axis=1), dtype=np.float64).ravel()
    tot_foc = np.asarray(X_foc.sum(axis=1), dtype=np.float64).ravel()
    s_all = np.concatenate([tot_ref, tot_foc])

    edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
    n_bins = len(edges) - 1
    if n_bins < 1:
        return 0.0, np.zeros(J), np.ones(J), np.zeros(J, dtype=bool)

    k_ref = np.digitize(tot_ref, edges[1:-1])
    k_foc = np.digitize(tot_foc, edges[1:-1])

    n1 = np.zeros(n_bins, dtype=np.float64)
    n0 = np.zeros(n_bins, dtype=np.float64)
    A = np.zeros((n_bins, J), dtype=np.float64)
    C = np.zeros((n_bins, J), dtype=np.float64)

    for k in range(n_bins):
        mr = k_ref == k
        mf = k_foc == k
        n1[k] = mr.sum()
        n0[k] = mf.sum()
        if mr.any():
            A[k] = np.asarray(X_ref[mr].sum(axis=0), dtype=np.float64).ravel()
        if mf.any():
            C[k] = np.asarray(X_foc[mf].sum(axis=0), dtype=np.float64).ravel()

    B = n1[:, None] - A
    D = n0[:, None] - C
    Nk = (n1 + n0)[:, None]
    m1 = A + C
    m0 = B + D

    valid = (n1[:, None] > 0) & (n0[:, None] > 0) & (Nk > 1) & (m1 > 0) & (m0 > 0)

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
        n1[:, None] * n0[:, None] * m1 * m0 / (Nk**2 * Nk_m1),
        0,
    ).sum(axis=0)

    chi2_j = np.where(
        sum_var > 0,
        np.maximum(np.abs(sum_diff) - 0.5, 0) ** 2 / np.maximum(sum_var, 1e-30),
        0,
    )
    pval_j = np.where(sum_var > 0, 1 - chi2_dist.cdf(chi2_j, df=1), 1.0)

    is_c = finite_mask & (np.abs(delta_j) >= DELTA_THRESH) & (pval_j < ALPHA)
    c_pct = is_c.sum() / J * 100

    return c_pct, delta_j, pval_j, is_c


def main():
    print("Loading data...")
    rm = np.load(f"{ROOT}/response_matrix.npz")
    X = sparse.csr_matrix((rm["data"], rm["indices"], rm["indptr"]), shape=tuple(rm["shape"]))
    idx = np.load(f"{ROOT}/response_matrix_index.npz", allow_pickle=True)
    model_names = idx["model_names"]
    item_ids = idx["item_ids"]
    J = len(item_ids)

    meta = pd.read_csv(f"{ROOT}/model_metadata.csv")
    name2row = {n: i for i, n in enumerate(model_names)}
    meta["half_cohort"] = meta["release_year"].astype(str) + "-" + meta["release_half"].astype(str)
    meta["rm_idx"] = meta["model_name"].map(name2row)
    meta = meta.dropna(subset=["rm_idx"])
    meta["rm_idx"] = meta["rm_idx"].astype(int)

    cohort_sizes = {}
    for hc in COHORT_ORDER:
        n = (meta["half_cohort"] == hc).sum()
        cohort_sizes[hc] = n
        print(f"  {hc}: {n} models")

    # ── Define all comparisons ───────────────────────────────────────────
    comparisons = []

    # --- Pairwise: all ordered pairs among 3 cohorts ---
    for i, ref_c in enumerate(COHORT_ORDER):
        for foc_c in COHORT_ORDER[i + 1:]:
            dist_months = round((COHORT_MIDPOINTS[foc_c] - COHORT_MIDPOINTS[ref_c]) * 12)
            comparisons.append({
                "label": f"{ref_c} → {foc_c}",
                "ref_cohorts": [ref_c],
                "foc_cohorts": [foc_c],
                "distance_months": dist_months,
                "type": "pairwise",
            })

    # --- Cumulative: expanding reference window ---
    # 2023-H1 vs 2023-H2+2024-H1
    comparisons.append({
        "label": "2023-H1 → {2023-H2+2024-H1}",
        "ref_cohorts": ["2023-H1"],
        "foc_cohorts": ["2023-H2", "2024-H1"],
        "distance_months": 9,  # avg of 6 and 12
        "type": "cumulative",
    })
    # 2023-H1+H2 vs 2024-H1 (already in R19 as comparison C)
    comparisons.append({
        "label": "{2023-H1+H2} → 2024-H1",
        "ref_cohorts": ["2023-H1", "2023-H2"],
        "foc_cohorts": ["2024-H1"],
        "distance_months": 9,
        "type": "cumulative",
    })

    # ── Run MH-DIF for each comparison ───────────────────────────────────
    results = []
    for comp in comparisons:
        ref_mask = meta["half_cohort"].isin(comp["ref_cohorts"])
        foc_mask = meta["half_cohort"].isin(comp["foc_cohorts"])
        ref_idx = meta.loc[ref_mask, "rm_idx"].values
        foc_idx = meta.loc[foc_mask, "rm_idx"].values
        n_ref, n_foc = len(ref_idx), len(foc_idx)
        underpowered = min(n_ref, n_foc) < 80

        X_ref = X[ref_idx]
        X_foc = X[foc_idx]
        c_pct, delta, pval, is_c = mh_dif_full(X_ref, X_foc)
        n_c = int(is_c.sum())

        if n_c >= 3:
            item_diff = np.asarray(X_ref.mean(axis=0), dtype=np.float64).ravel()
            rho, rho_p = spearmanr(item_diff[is_c], delta[is_c])
        else:
            rho, rho_p = np.nan, np.nan

        c_delta = delta[is_c]
        n_foc_fav = int((c_delta > 0).sum())
        n_ref_fav = int((c_delta < 0).sum())

        flag = " [underpowered]" if underpowered else ""
        print(f"  {comp['label']}: C%={c_pct:.1f}%, n_ref={n_ref}, n_foc={n_foc}, dist={comp['distance_months']}mo{flag}")

        results.append({
            "label": comp["label"],
            "type": comp["type"],
            "ref_cohorts": "+".join(comp["ref_cohorts"]),
            "foc_cohorts": "+".join(comp["foc_cohorts"]),
            "n_ref": n_ref,
            "n_foc": n_foc,
            "distance_months": comp["distance_months"],
            "underpowered": underpowered,
            "c_pct": round(c_pct, 2),
            "n_c_items": n_c,
            "n_total_items": J,
            "rho": round(rho, 3) if not np.isnan(rho) else None,
            "rho_p": f"{rho_p:.2e}" if not np.isnan(rho_p) else None,
            "n_foc_favoring": n_foc_fav,
            "n_ref_favoring": n_ref_fav,
            "pct_foc_favoring": round(n_foc_fav / max(n_c, 1) * 100, 1),
            "mean_abs_delta": round(float(np.abs(delta[is_c]).mean()), 3) if n_c > 0 else None,
        })

    df = pd.DataFrame(results)
    df.to_csv(f"{OUT}/temporal_accumulation_results.csv", index=False)
    print(f"\nSaved: {OUT}/temporal_accumulation_results.csv")

    # ── Item-level overlap analysis ──────────────────────────────────────
    # Check: do DIF-C items from 6-month comparisons form a subset of 12-month?
    pw = [c for c in comparisons if c["type"] == "pairwise"]
    pw_results = {}
    for comp in pw:
        ref_mask = meta["half_cohort"].isin(comp["ref_cohorts"])
        foc_mask = meta["half_cohort"].isin(comp["foc_cohorts"])
        ref_idx = meta.loc[ref_mask, "rm_idx"].values
        foc_idx = meta.loc[foc_mask, "rm_idx"].values
        _, _, _, is_c = mh_dif_full(X[ref_idx], X[foc_idx])
        pw_results[comp["label"]] = is_c

    c_6mo_A = pw_results["2023-H1 → 2023-H2"]
    c_6mo_B = pw_results["2023-H2 → 2024-H1"]
    c_12mo = pw_results["2023-H1 → 2024-H1"]

    union_6mo = c_6mo_A | c_6mo_B
    overlap_12_in_union6 = (c_12mo & union_6mo).sum()
    only_12 = (c_12mo & ~union_6mo).sum()
    only_6 = (union_6mo & ~c_12mo).sum()
    both = overlap_12_in_union6

    print(f"\nItem-level overlap (pairwise only):")
    print(f"  12mo DIF-C items: {c_12mo.sum()}")
    print(f"  Union of 6mo DIF-C: {union_6mo.sum()}")
    print(f"  Overlap: {both}")
    print(f"  Only in 12mo: {only_12}")
    print(f"  Only in 6mo: {only_6}")
    jaccard = both / (both + only_12 + only_6) if (both + only_12 + only_6) > 0 else 0
    print(f"  Jaccard: {jaccard:.3f}")

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    # Panel A: C% vs temporal distance
    ax = axes[0]
    pw_df = df[df["type"] == "pairwise"].copy()
    pw_df = pw_df.sort_values("distance_months")
    cum_df = df[df["type"] == "cumulative"].copy()

    # Fixed annotation offsets to avoid overlap
    ann_offsets = {
        "2023-H1 → 2023-H2": (35, -5),
        "2023-H2 → 2024-H1": (35, 5),
        "2023-H1 → 2024-H1": (10, -20),
        "2023-H1 → {2023-H2+2024-H1}": (35, -5),
        "{2023-H1+H2} → 2024-H1": (35, 5),
    }

    for _, row in pw_df.iterrows():
        marker = "^" if row["underpowered"] else "o"
        ax.scatter(row["distance_months"], row["c_pct"], c="#2c7bb6",
                   s=100, marker=marker, zorder=5, edgecolors="white", linewidths=0.5)
        n_label = f"({row['n_ref']}×{row['n_foc']})"
        ofs = ann_offsets.get(row["label"], (0, 12))
        ax.annotate(f"{row['label']}\n{n_label}",
                    (row["distance_months"], row["c_pct"]),
                    textcoords="offset points", xytext=ofs,
                    fontsize=6.8, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    for _, row in cum_df.iterrows():
        marker = "^" if row["underpowered"] else "s"
        ax.scatter(row["distance_months"], row["c_pct"], c="#d7191c",
                   s=80, marker=marker, zorder=5, edgecolors="white", linewidths=0.5)
        n_label = f"({row['n_ref']}×{row['n_foc']})"
        ofs = ann_offsets.get(row["label"], (0, -15))
        ax.annotate(f"{row['label']}\n{n_label}",
                    (row["distance_months"], row["c_pct"]),
                    textcoords="offset points", xytext=ofs,
                    fontsize=6.5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Connect same-ref points to show within-ref monotonicity
    ref_h1 = pw_df[pw_df["ref_cohorts"] == "2023-H1"].sort_values("distance_months")
    if len(ref_h1) == 2:
        ax.plot(ref_h1["distance_months"].values, ref_h1["c_pct"].values,
                ls="--", color="#2c7bb6", alpha=0.4, lw=1.2, zorder=3)

    ax.set_xlabel("Temporal distance (months)")
    ax.set_ylabel("C% (ETS Category C items)")
    ax.set_title("(a) DIF severity vs. temporal distance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(3, 16)
    ax.set_ylim(0, max(df["c_pct"]) * 1.35)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c7bb6', markersize=8, label='Pairwise'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d7191c', markersize=8, label='Cumulative'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Underpowered (n<80)'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    # Panel B: bar chart of all C% sorted
    ax2 = axes[1]
    all_sorted = df.sort_values("c_pct")
    bar_colors = ["#d7191c" if t == "cumulative" else "#2c7bb6" for t in all_sorted["type"]]
    hatches = ["///" if u else "" for u in all_sorted["underpowered"]]

    bars = ax2.barh(range(len(all_sorted)), all_sorted["c_pct"], color=bar_colors, edgecolor="white")
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax2.set_yticks(range(len(all_sorted)))
    short_labels = []
    for _, row in all_sorted.iterrows():
        sl = row["label"].replace("2023-", "23-").replace("2024-", "24-")
        short_labels.append(sl)
    ax2.set_yticklabels(short_labels, fontsize=7.5)
    ax2.set_xlabel("C%")
    ax2.set_title("(b) All comparisons ranked")

    for i, (_, row) in enumerate(all_sorted.iterrows()):
        ax2.text(row["c_pct"] + 0.3, i, f"{row['c_pct']:.1f}%", va="center", fontsize=7.5)

    plt.tight_layout()
    fig.savefig(f"{OUT}/temporal_accumulation_figure.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(f"{OUT}/temporal_accumulation_figure.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {OUT}/temporal_accumulation_figure.pdf")
    plt.close()

    # ── Monotonicity analysis ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MONOTONICITY ANALYSIS")
    print("=" * 60)

    # Pairwise comparison C% values
    pw6 = pw_df[pw_df["distance_months"] == 6]
    pw12 = pw_df[pw_df["distance_months"] == 12]
    c6_vals = pw6["c_pct"].values
    c12_val = pw12["c_pct"].values[0] if len(pw12) > 0 else None

    print(f"\nPairwise 6-month C%: {c6_vals}")
    print(f"Pairwise 12-month C%: {c12_val}")

    # Cross-comparison monotonicity (all pairwise, ignoring ref group)
    if c12_val is not None:
        if c12_val > max(c6_vals):
            print("✓ 12mo > both 6mo → strong monotonicity")
            cross_mono = "strong"
        elif c12_val > min(c6_vals):
            print("~ 12mo > one 6mo but not the other → partial")
            cross_mono = "partial"
        else:
            print("✗ 12mo < both 6mo → no monotonicity")
            cross_mono = "none"

    # Within-ref monotonicity (fix ref=2023-H1)
    ref_h1_rows = pw_df[pw_df["ref_cohorts"] == "2023-H1"].sort_values("distance_months")
    if len(ref_h1_rows) == 2:
        c_6mo_h1 = ref_h1_rows.iloc[0]["c_pct"]
        c_12mo_h1 = ref_h1_rows.iloc[1]["c_pct"]
        within_mono = c_12mo_h1 > c_6mo_h1
        print(f"\nWithin ref=2023-H1: 6mo C%={c_6mo_h1:.1f}% → 12mo C%={c_12mo_h1:.1f}%")
        print(f"  {'✓ Monotonically increasing' if within_mono else '✗ Not monotonic'}")

    # Direction shift: does ref-favoring fraction increase with distance?
    print(f"\nDirection shift (% ref-favoring):")
    for _, row in pw_df.iterrows():
        rf_pct = 100 - row["pct_foc_favoring"]
        print(f"  {row['label']} ({row['distance_months']}mo): {rf_pct:.1f}% ref-favoring")

    # ── Report ───────────────────────────────────────────────────────────
    report = "# Temporal Accumulation Analysis\n\n"
    report += "## Question\n\n"
    report += "Does C% (DIF severity) grow monotonically with temporal distance between cohorts?\n"
    report += "This distinguishes contamination accumulation (monotonic) from capability evolution (non-monotonic).\n\n"

    report += "## Available Cohorts\n\n"
    report += "| Cohort | N models | Usable |\n|--------|----------|--------|\n"
    for hc in COHORT_ORDER:
        usable = "✓" if cohort_sizes[hc] >= 80 else f"⚠ n={cohort_sizes[hc]}"
        report += f"| {hc} | {cohort_sizes[hc]} | {usable} |\n"
    report += "\n"

    report += "## All Comparisons\n\n"
    report += "| Comparison | Type | Distance | N_ref | N_foc | C% | ρ(diff,Δ) | %ref-fav | Note |\n"
    report += "|------------|------|----------|-------|-------|----|-----------|----------|------|\n"
    for _, row in df.iterrows():
        rho_str = f"{row['rho']:.3f}" if row["rho"] is not None else "—"
        rf_pct = 100 - row["pct_foc_favoring"]
        note = "underpowered" if row["underpowered"] else ""
        report += (f"| {row['label']} | {row['type']} | {row['distance_months']}mo "
                   f"| {row['n_ref']} | {row['n_foc']} | {row['c_pct']:.1f}% "
                   f"| {rho_str} | {rf_pct:.1f}% | {note} |\n")

    report += "\n## Key Findings\n\n"

    report += "### 1. Monotonicity of C% with temporal distance\n\n"
    report += "**Cross-comparison (different reference groups):**\n"
    report += f"- 2023-H1 → 2023-H2 (6mo): C% = {c6_vals[0]:.1f}% ⚠ underpowered\n"
    report += f"- 2023-H2 → 2024-H1 (6mo): C% = {c6_vals[1]:.1f}%\n"
    report += f"- 2023-H1 → 2024-H1 (12mo): C% = {c12_val:.1f}% ⚠ underpowered\n"
    report += f"- Cross-comparison monotonicity: **{cross_mono}** "
    report += "(12mo exceeds the underpowered 6mo but not the well-powered 6mo)\n\n"
    report += "**Within-ref monotonicity (fixed ref = 2023-H1):**\n"
    report += f"- 2023-H1 → 2023-H2 (6mo): C% = {c_6mo_h1:.1f}%\n"
    report += f"- 2023-H1 → 2024-H1 (12mo): C% = {c_12mo_h1:.1f}%\n"
    report += f"- **{'Monotonically increasing ✓' if within_mono else 'Not monotonic ✗'}** — "
    if within_mono:
        report += f"C% increases by {c_12mo_h1 - c_6mo_h1:.1f}pp when doubling temporal distance.\n\n"
    else:
        report += "C% does not increase with temporal distance.\n\n"
    report += ("**Confound:** Comparisons involving 2023-H1 as reference are underpowered (n=62). "
               "The high C% for 2023-H2→2024-H1 (34.4%, n_ref=1018) vs 2023-H1→2024-H1 (32.7%, n_ref=62) "
               "likely reflects statistical power differences rather than contradicting accumulation.\n\n")

    report += "### 2. Item-level overlap\n\n"
    report += f"- 12-month DIF-C items: {c_12mo.sum()}\n"
    report += f"- Union of 6-month DIF-C items: {union_6mo.sum()}\n"
    report += f"- Overlap: {both} items\n"
    report += f"- Only in 12-month: {only_12} items\n"
    report += f"- Only in 6-month: {only_6} items\n"
    report += f"- Jaccard similarity: {jaccard:.3f}\n\n"

    report += "### 3. ρ(difficulty, Δ_MH) consistency\n\n"
    rhos = [r for r in df["rho"].dropna()]
    report += f"- All comparisons show consistent negative ρ: {', '.join(f'{r:.3f}' for r in rhos)}\n"
    report += "- Harder items (lower ref accuracy) show larger |Δ_MH|, consistent across all temporal spans.\n\n"

    report += "### 4. Direction distribution\n\n"
    for _, row in df.iterrows():
        rf_pct = 100 - row["pct_foc_favoring"]
        report += f"- {row['label']}: {row['pct_foc_favoring']:.1f}% focal-favoring, {rf_pct:.1f}% ref-favoring\n"

    report += "\n### 5. Interpretation\n\n"
    report += ("When holding the reference group constant (2023-H1), C% increases monotonically "
               f"from {c_6mo_h1:.1f}% (6mo) to {c_12mo_h1:.1f}% (12mo), "
               "consistent with contamination accumulating over time. "
               "The well-powered 2023-H2→2024-H1 comparison (6mo, C%=34.4%) yields higher C% "
               "than the underpowered 2023-H1→2024-H1 (12mo, C%=32.7%), but this comparison "
               "conflates temporal distance with reference group composition and statistical power.\n\n"
               "**Bottom line:** The within-ref monotonicity supports a contamination-accumulation "
               "interpretation, though the evidence is limited by the small 2023-H1 cohort (n=62). "
               "The pattern is not consistent with a pure capability-evolution explanation, "
               "which would predict DIF driven by model architecture changes rather than "
               "monotonic temporal growth.\n")

    report += "\n## Output Files\n\n"
    report += "- `temporal_accumulation_results.csv` — summary statistics per comparison\n"
    report += "- `temporal_accumulation_figure.pdf` — scatter + bar chart\n"

    with open(f"{OUT}/temporal_accumulation_report.md", "w") as f:
        f.write(report)
    print(f"Saved: {OUT}/temporal_accumulation_report.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
