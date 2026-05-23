#!/usr/bin/env python3
"""Leave-One-Family-Out (LOFO) robustness check for temporal MH-DIF C%."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2 as chi2_dist

BASE = "artifacts"
N_STRATA = 5
TOP_K = 5


def mh_dif_c_pct_vec(X_ref, X_foc, n_strata=N_STRATA):
    """Vectorized MH-DIF -> C%. Stratifies on total score."""
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
    mat = sp.load_npz(f"{BASE}/response_matrix.npz")
    idx = np.load(f"{BASE}/response_matrix_index.npz", allow_pickle=True)
    model_names = list(idx["model_names"])
    name_to_idx = {n: i for i, n in enumerate(model_names)}

    df = pd.read_csv(f"{BASE}/plan_032_dif_cleaned_mmlu/cleaned_mmlu_analysis.csv")

    ref_models = df[df["cohort_temporal"] == "2023"]["model_name"].tolist()
    foc_models = df[df["cohort_temporal"] == "2024"]["model_name"].tolist()
    ref_indices = np.array([name_to_idx[m] for m in ref_models])
    foc_indices = np.array([name_to_idx[m] for m in foc_models])
    ref_set = set(ref_indices)
    foc_set = set(foc_indices)

    print(f"Ref (2023): {len(ref_indices)}, Foc (2024): {len(foc_indices)}")

    # Full baseline
    X_ref_full = mat[ref_indices, :].toarray()
    X_foc_full = mat[foc_indices, :].toarray()
    c_pct_full = mh_dif_c_pct_vec(X_ref_full, X_foc_full)
    print(f"Full baseline C%: {c_pct_full:.2f}%")

    # Top-K families across ref+foc
    active = df[df["cohort_temporal"].isin(["2023", "2024"])]
    top_families = active["family"].value_counts().head(TOP_K).index.tolist()
    print(f"Top-{TOP_K} families: {top_families}")

    results = []
    for fam in top_families:
        fam_models = df[df["family"] == fam]["model_name"].tolist()
        fam_idx_set = set(name_to_idx[m] for m in fam_models if m in name_to_idx)

        ref_lofo = np.array([i for i in ref_indices if i not in fam_idx_set])
        foc_lofo = np.array([i for i in foc_indices if i not in fam_idx_set])

        n_ref_removed = len(ref_indices) - len(ref_lofo)
        n_foc_removed = len(foc_indices) - len(foc_lofo)

        if len(ref_lofo) == 0 or len(foc_lofo) == 0:
            print(f"  {fam}: skip (empty cohort after removal)")
            continue

        X_ref_lofo = mat[ref_lofo, :].toarray()
        X_foc_lofo = mat[foc_lofo, :].toarray()
        c_pct_lofo = mh_dif_c_pct_vec(X_ref_lofo, X_foc_lofo)
        delta = c_pct_lofo - c_pct_full

        print(f"  {fam}: removed {n_ref_removed}R+{n_foc_removed}F = {n_ref_removed+n_foc_removed}, "
              f"C% = {c_pct_lofo:.2f}% (Δ = {delta:+.2f}%)")

        results.append({
            "family": fam,
            "n_models_total": len(fam_models),
            "n_models_ref": n_ref_removed,
            "n_models_foc": n_foc_removed,
            "c_pct_full": round(c_pct_full, 2),
            "c_pct_lofo": round(c_pct_lofo, 2),
            "delta_c_pct": round(delta, 2),
        })

        del X_ref_lofo, X_foc_lofo

    out_dir = f"{BASE}/lofo_robustness"
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{out_dir}/lofo_results.csv", index=False)

    # Write markdown report
    lines = [
        "# LOFO Robustness Check: Temporal MH-DIF C%\n",
        f"**Full baseline C%**: {c_pct_full:.2f}%  ",
        f"**Ref cohort**: 2023 ({len(ref_indices)} models)  ",
        f"**Foc cohort**: 2024 ({len(foc_indices)} models)\n",
        "## Results\n",
        "| Family | N removed (ref+foc) | C% full | C% LOFO | ΔC% |",
        "|--------|--------------------:|--------:|--------:|----:|",
    ]
    for r in results:
        n_rem = r["n_models_ref"] + r["n_models_foc"]
        lines.append(
            f"| {r['family']} | {n_rem} ({r['n_models_ref']}R+{r['n_models_foc']}F) "
            f"| {r['c_pct_full']:.2f} | {r['c_pct_lofo']:.2f} | {r['delta_c_pct']:+.2f} |"
        )

    deltas = [abs(r["delta_c_pct"]) for r in results]
    max_delta = max(deltas)
    mean_delta = np.mean(deltas)
    lines.append(f"\n**Max |ΔC%|**: {max_delta:.2f}pp  ")
    lines.append(f"**Mean |ΔC%|**: {mean_delta:.2f}pp\n")

    if max_delta < 3.0:
        lines.append("**Conclusion**: C% is robust to single-family removal — "
                      f"max perturbation {max_delta:.2f}pp across top-{TOP_K} families.")
    else:
        worst = max(results, key=lambda r: abs(r["delta_c_pct"]))
        lines.append(f"**Conclusion**: Removing {worst['family']} shifts C% by "
                      f"{worst['delta_c_pct']:+.2f}pp, indicating moderate sensitivity "
                      f"to this family.")

    with open(f"{out_dir}/lofo_results.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved to {out_dir}/lofo_results.csv and lofo_results.md")


if __name__ == "__main__":
    main()
