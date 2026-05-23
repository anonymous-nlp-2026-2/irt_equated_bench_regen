#!/usr/bin/env python3
"""R19 Exp 1: Temporal Granularity DIF — half-year cohort pairwise MH-DIF."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist, spearmanr

ROOT = "artifacts"
OUT = f"{ROOT}/r19_temporal_granularity"

N_STRATA = 5
ALPHA = 0.05
DELTA_THRESH = 1.5


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


def get_subject(iid):
    parts = str(iid).rsplit(".", 1)
    return parts[0] if len(parts) > 1 else str(iid)


def top_subjects_by_c_pct(item_ids, is_c, n=5):
    """Return top-N subjects ranked by within-subject C%."""
    subjs = np.array([get_subject(i) for i in item_ids])
    unique_subjs = np.unique(subjs)
    rows = []
    for s in unique_subjs:
        mask = subjs == s
        total = mask.sum()
        c_count = (is_c & mask).sum()
        rows.append((s, c_count, total, c_count / total * 100))
    rows.sort(key=lambda x: -x[3])
    return rows[:n]


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

    # Build half-year cohort label
    meta["half_cohort"] = meta["release_year"].astype(str) + "-" + meta["release_half"].astype(str)
    meta["rm_idx"] = meta["model_name"].map(name2row)
    meta = meta.dropna(subset=["rm_idx"])
    meta["rm_idx"] = meta["rm_idx"].astype(int)

    print("\nCohort distribution:")
    for hc in sorted(meta["half_cohort"].unique()):
        print(f"  {hc}: {(meta['half_cohort'] == hc).sum()} models")

    # Define comparisons
    comparisons = {
        "A: 2023-H1 vs 2023-H2": ("2023-H1", "2023-H2"),
        "B: 2023-H2 vs 2024-H1": ("2023-H2", "2024-H1"),
        "C: 2023-full vs 2024-H1": (["2023-H1", "2023-H2"], "2024-H1"),
    }

    all_results = []
    item_level_dfs = []

    for label, (ref_cohort, foc_cohort) in comparisons.items():
        print(f"\n{'='*60}")
        print(f"Comparison {label}")
        print(f"{'='*60}")

        if isinstance(ref_cohort, list):
            ref_mask = meta["half_cohort"].isin(ref_cohort)
            ref_label = "+".join(ref_cohort)
        else:
            ref_mask = meta["half_cohort"] == ref_cohort
            ref_label = ref_cohort

        foc_mask = meta["half_cohort"] == foc_cohort
        ref_idx = meta.loc[ref_mask, "rm_idx"].values
        foc_idx = meta.loc[foc_mask, "rm_idx"].values
        n_ref, n_foc = len(ref_idx), len(foc_idx)

        underpowered = min(n_ref, n_foc) < 80
        flag = " ⚠ UNDERPOWERED" if underpowered else ""
        print(f"  Ref ({ref_label}): {n_ref}{flag}")
        print(f"  Foc ({foc_cohort}): {n_foc}{flag}")

        X_ref = X[ref_idx]
        X_foc = X[foc_idx]
        c_pct, delta, pval, is_c = mh_dif_full(X_ref, X_foc)
        n_c = int(is_c.sum())
        print(f"  C% = {c_pct:.1f}%  ({n_c}/{J} items)")

        # Difficulty = ref cohort mean accuracy per item
        item_diff = np.asarray(X_ref.mean(axis=0), dtype=np.float64).ravel()

        # ρ(difficulty, Δ_MH) among DIF-C items
        if n_c >= 3:
            rho, rho_p = spearmanr(item_diff[is_c], delta[is_c])
            print(f"  ρ(difficulty, Δ_MH) = {rho:.3f}  (p={rho_p:.2e}, n={n_c})")
        else:
            rho, rho_p = np.nan, np.nan
            print(f"  Too few C items ({n_c}) for correlation")

        # Direction distribution
        c_delta = delta[is_c]
        n_foc_fav = int((c_delta > 0).sum())
        n_ref_fav = int((c_delta < 0).sum())
        if n_c > 0:
            pct_foc = n_foc_fav / n_c * 100
            pct_ref = n_ref_fav / n_c * 100
        else:
            pct_foc = pct_ref = 0
        print(f"  Direction: focal-favoring={n_foc_fav} ({pct_foc:.1f}%), ref-favoring={n_ref_fav} ({pct_ref:.1f}%)")

        # Top-5 subjects by C%
        top5 = top_subjects_by_c_pct(item_ids, is_c, n=5)
        print(f"  Top-5 subjects by C%:")
        for subj, cc, tt, cpct in top5:
            print(f"    {subj}: {cpct:.1f}% ({cc}/{tt})")

        all_results.append({
            "comparison": label,
            "ref_cohort": ref_label,
            "foc_cohort": foc_cohort,
            "n_ref": n_ref,
            "n_foc": n_foc,
            "underpowered": underpowered,
            "c_pct": round(c_pct, 2),
            "n_c_items": n_c,
            "n_total_items": J,
            "rho_difficulty_delta": round(rho, 3) if not np.isnan(rho) else None,
            "rho_p_value": f"{rho_p:.2e}" if not np.isnan(rho_p) else None,
            "n_focal_favoring": n_foc_fav,
            "n_ref_favoring": n_ref_fav,
            "pct_focal_favoring": round(pct_foc, 1),
            "pct_ref_favoring": round(pct_ref, 1),
            "top5_subjects": "; ".join(f"{s} ({cp:.1f}%)" for s, _, _, cp in top5),
        })

        # Per-item data for this comparison
        short_label = label.split(":")[0].strip()
        item_level_dfs.append(pd.DataFrame({
            "item_id": item_ids,
            "comparison": short_label,
            "difficulty_ref": item_diff,
            "delta_mh": delta,
            "p_value": pval,
            "is_c": is_c,
            "direction": np.where(delta > 0, "focal-favoring", np.where(delta < 0, "ref-favoring", "none")),
        }))

    # Save CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{OUT}/temporal_granularity_results.csv", index=False)
    print(f"\nSaved: {OUT}/temporal_granularity_results.csv")

    item_all = pd.concat(item_level_dfs, ignore_index=True)
    item_all.to_csv(f"{OUT}/temporal_granularity_items.csv", index=False)
    print(f"Saved: {OUT}/temporal_granularity_items.csv")

    # ── Key analyses ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY ANALYSES")
    print("=" * 60)

    cpcts = [r["c_pct"] for r in all_results]
    labels = [r["comparison"] for r in all_results]
    print(f"\n1. C% monotonicity: {' → '.join(f'{l}: {c}%' for l, c in zip(labels, cpcts))}")
    if cpcts[0] < cpcts[1] < cpcts[2]:
        print("   ✓ Monotonically increasing (A < B < C)")
    else:
        print("   ✗ Not monotonically increasing")

    rhos = [r["rho_difficulty_delta"] for r in all_results]
    print(f"\n2. ρ gradient consistency: {rhos}")
    non_null = [r for r in rhos if r is not None]
    if len(non_null) >= 2:
        signs = [np.sign(r) for r in non_null]
        if len(set(signs)) == 1:
            print(f"   ✓ Consistent sign across comparisons (all {'positive' if signs[0] > 0 else 'negative'})")
        else:
            print("   ✗ Inconsistent signs across comparisons")

    print(f"\n3. Direction distribution shift:")
    for r in all_results:
        print(f"   {r['comparison']}: focal-fav {r['pct_focal_favoring']}% | ref-fav {r['pct_ref_favoring']}%")

    # ── Report ────────────────────────────────────────────────────────────
    print("\nWriting report...")

    report = "# R19 Exp 1: Temporal Granularity DIF\n\n"
    report += "## Design\n\n"
    report += "Pairwise MH-DIF across half-year cohorts to test whether DIF accumulates with temporal distance.\n\n"
    report += "| Comparison | Ref | Foc | N_ref | N_foc | Note |\n"
    report += "|------------|-----|-----|-------|-------|------|\n"
    for r in all_results:
        note = "⚠ underpowered" if r["underpowered"] else ""
        report += f"| {r['comparison']} | {r['ref_cohort']} | {r['foc_cohort']} | {r['n_ref']} | {r['n_foc']} | {note} |\n"

    report += "\n## Results\n\n"
    report += "### C% (ETS Category C items)\n\n"
    report += "| Comparison | C% | DIF-C items |\n"
    report += "|------------|-----|-------------|\n"
    for r in all_results:
        report += f"| {r['comparison']} | {r['c_pct']:.1f}% | {r['n_c_items']}/{r['n_total_items']} |\n"

    mono = cpcts[0] < cpcts[1] < cpcts[2]
    report += f"\nMonotonicity (A < B < C): **{'yes' if mono else 'no'}**\n"

    report += "\n### Difficulty–Direction Gradient\n\n"
    report += "Spearman ρ(item difficulty, Δ_MH) among DIF-C items:\n\n"
    report += "| Comparison | ρ | p | n |\n"
    report += "|------------|---|---|---|\n"
    for r in all_results:
        rho_str = f"{r['rho_difficulty_delta']:.3f}" if r["rho_difficulty_delta"] is not None else "—"
        p_str = r["rho_p_value"] if r["rho_p_value"] is not None else "—"
        report += f"| {r['comparison']} | {rho_str} | {p_str} | {r['n_c_items']} |\n"

    report += "\n### Direction Distribution\n\n"
    report += "| Comparison | Focal-favoring | Ref-favoring |\n"
    report += "|------------|---------------|-------------|\n"
    for r in all_results:
        report += f"| {r['comparison']} | {r['n_focal_favoring']} ({r['pct_focal_favoring']}%) | {r['n_ref_favoring']} ({r['pct_ref_favoring']}%) |\n"

    report += "\n### Top-5 Subjects by C%\n\n"
    for r in all_results:
        report += f"**{r['comparison']}:** {r['top5_subjects']}\n\n"

    report += "## Key Findings\n\n"
    report += f"1. **C% monotonicity:** {'Confirmed' if mono else 'Not confirmed'} — "
    report += f"C% = {cpcts[0]:.1f}% → {cpcts[1]:.1f}% → {cpcts[2]:.1f}%\n"

    if len(non_null) >= 2:
        signs = [np.sign(r) for r in non_null]
        consistent = len(set(signs)) == 1
        report += f"2. **ρ gradient:** {'Consistent' if consistent else 'Inconsistent'} sign across comparisons "
        report += f"({', '.join(f'{r:.3f}' for r in non_null)})\n"
    else:
        report += "2. **ρ gradient:** Insufficient data for comparison\n"

    report += "3. **Direction distribution:** See table above\n"

    report += "\n## Output Files\n\n"
    report += "- `temporal_granularity_results.csv` — summary per comparison\n"
    report += "- `temporal_granularity_items.csv` — per-item DIF statistics for each comparison\n"

    with open(f"{OUT}/temporal_granularity_report.md", "w") as f:
        f.write(report)
    print(f"Saved: {OUT}/temporal_granularity_report.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
