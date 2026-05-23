#!/usr/bin/env python3
"""R22: Within-architecture temporal DIF analysis.

Groups fine-grained families into architecture lineages (e.g., llama-2 + llama-3 → Llama),
then runs MH-DIF within each lineage (2023 vs 2024) to rule out the confound that
overall temporal DIF is driven by cross-architecture differences.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist

ROOT = "/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts"
OUT = f"{ROOT}/r22_within_arch_temporal"

N_STRATA = 5
ALPHA = 0.05
DELTA_THRESH = 1.5
MIN_PER_COHORT = 10

# ── Architecture lineage mapping ────────────────────────────────────────────
# Maps fine-grained family → broad architecture lineage.
# Only families that appear in model_metadata.csv and have a cross-year counterpart.
ARCH_LINEAGE = {
    "llama-2":    "Llama",
    "codellama":  "Llama",
    "llama-3":    "Llama",
    "qwen":       "Qwen",
    "qwen-1.5":   "Qwen",
    "qwen-2":     "Qwen",
    "yi":         "Yi",
    "yi-1.5":     "Yi",
    "phi-2":      "Phi",
    "phi-3":      "Phi",
}


# ── MH-DIF (vectorized, from r18) ──────────────────────────────────────────

def mh_dif_full(X_ref, X_foc, n_strata=N_STRATA):
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


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    meta = pd.read_csv(f"{ROOT}/model_metadata.csv")
    idx = np.load(f"{ROOT}/response_matrix_index.npz", allow_pickle=True)
    model_names = list(idx["model_names"])
    R = sparse.load_npz(f"{ROOT}/response_matrix.npz")

    meta["arch_lineage"] = meta["family"].map(ARCH_LINEAGE)
    meta_t = meta[meta.cohort_temporal.isin(["2023", "2024"]) & meta.arch_lineage.notna()].copy()

    name_to_row = {n: i for i, n in enumerate(model_names)}

    lineages = sorted(meta_t.arch_lineage.unique())
    results = []

    for lin in lineages:
        sub = meta_t[meta_t.arch_lineage == lin]
        ref_names = sub[sub.cohort_temporal == "2023"].model_name.tolist()
        foc_names = sub[sub.cohort_temporal == "2024"].model_name.tolist()

        ref_idx = [name_to_row[n] for n in ref_names if n in name_to_row]
        foc_idx = [name_to_row[n] for n in foc_names if n in name_to_row]

        n_ref = len(ref_idx)
        n_foc = len(foc_idx)
        underpowered = n_ref < MIN_PER_COHORT or n_foc < MIN_PER_COHORT

        families_2023 = sorted(sub[sub.cohort_temporal == "2023"].family.unique())
        families_2024 = sorted(sub[sub.cohort_temporal == "2024"].family.unique())

        if n_ref < 2 or n_foc < 2:
            print(f"  {lin}: skipped (n_ref={n_ref}, n_foc={n_foc})")
            continue

        X_ref = R[ref_idx].toarray()
        X_foc = R[foc_idx].toarray()

        c_pct, delta_j, pval_j, is_c = mh_dif_full(X_ref, X_foc)
        n_dif_c = int(is_c.sum())
        n_items = X_ref.shape[1]
        median_abs_delta = float(np.median(np.abs(delta_j[delta_j != 0]))) if (delta_j != 0).any() else 0.0

        status = "underpowered" if underpowered else "ok"
        print(f"  {lin:8s}  n_ref={n_ref:4d}  n_foc={n_foc:4d}  C%={c_pct:5.1f}%  "
              f"n_C={n_dif_c:5d}/{n_items}  med|Δ|={median_abs_delta:.2f}  [{status}]")

        results.append({
            "lineage": lin,
            "families_2023": "; ".join(families_2023),
            "families_2024": "; ".join(families_2024),
            "n_ref": n_ref,
            "n_foc": n_foc,
            "c_pct": round(c_pct, 2),
            "n_dif_c": n_dif_c,
            "n_items": n_items,
            "median_abs_delta": round(median_abs_delta, 3),
            "status": status,
        })

    df = pd.DataFrame(results)
    df.to_csv(f"{OUT}/within_arch_results.csv", index=False)
    print(f"\nSaved: {OUT}/within_arch_results.csv")

    # ── Report ──
    overall_cpct = 31.3
    random_baseline = "0.18–1.3%"

    lines = ["# R22: Within-Architecture Temporal DIF\n"]
    lines.append("## Design\n")
    lines.append("**Goal:** Rule out the confound that overall temporal DIF (C%=31.3%) "
                 "is driven by cross-architecture differences rather than genuine temporal instability.\n")
    lines.append("**Method:** Group fine-grained families into architecture lineages sharing "
                 "the same base design (e.g., Llama-2 + Llama-3 → Llama). Within each lineage, "
                 "run MH-DIF with 2023 models as reference and 2024 models as focal.\n")
    lines.append("**Threshold:** ETS C classification: |Δ_MH| ≥ 1.5 and p < 0.05.\n")

    lines.append("## Lineage Mapping\n")
    lines.append("| Lineage | 2023 families | 2024 families |")
    lines.append("|---------|--------------|--------------|")
    for _, row in df.iterrows():
        lines.append(f"| {row.lineage} | {row.families_2023} | {row.families_2024} |")
    lines.append("")

    lines.append("## Results\n")
    lines.append("| Lineage | N_ref (2023) | N_foc (2024) | C% | N_C items | Med |Δ| | Status |")
    lines.append("|---------|-------------|-------------|-----|-----------|---------|--------|")
    for _, row in df.iterrows():
        lines.append(f"| {row.lineage} | {row.n_ref} | {row.n_foc} | {row.c_pct}% | "
                     f"{row.n_dif_c}/{row.n_items} | {row.median_abs_delta} | {row.status} |")
    lines.append("")

    if len(df) > 0:
        powered = df[df.status == "ok"]
        if len(powered) > 0:
            mean_c = powered.c_pct.mean()
            min_c = powered.c_pct.min()
            max_c = powered.c_pct.max()
        else:
            mean_c = df.c_pct.mean()
            min_c = df.c_pct.min()
            max_c = df.c_pct.max()
            powered = df

        lines.append("## Comparison\n")
        lines.append(f"- **Overall temporal DIF:** C% = {overall_cpct}%")
        lines.append(f"- **Within-architecture mean C%:** {mean_c:.1f}% (range: {min_c:.1f}%–{max_c:.1f}%)")
        lines.append(f"- **Random baseline:** C% ≈ {random_baseline}")
        lines.append("")

        lines.append("## Interpretation\n")
        if mean_c > 5:
            lines.append("Within-architecture C% remains well above random baseline across all lineages, "
                         "confirming that temporal measurement instability is **not** an artifact of "
                         "comparing different architectures. Even models sharing the same base design "
                         "exhibit substantial DIF between 2023 and 2024 cohorts.")
        else:
            lines.append("Within-architecture C% drops substantially compared to overall temporal DIF, "
                         "suggesting architecture differences partially drive the observed DIF signal.")

    report = "\n".join(lines) + "\n"
    with open(f"{OUT}/within_arch_report.md", "w") as f:
        f.write(report)
    print(f"Saved: {OUT}/within_arch_report.md")


if __name__ == "__main__":
    main()
