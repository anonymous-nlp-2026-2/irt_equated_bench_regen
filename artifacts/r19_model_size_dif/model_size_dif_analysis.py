#!/usr/bin/env python3
"""
R19 Exp 2 — Model-size DIF decomposition.

Split models into Small (<7B), Medium (7B–30B), Large (>30B) by parameter count
extracted from model names, then run temporal MH-DIF (2023 vs 2024) per group.
"""

import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist, spearmanr
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("artifacts")
OUT = ROOT / "r19_model_size_dif"
OUT.mkdir(exist_ok=True)

N_STRATA = 5

# ════════════════════════════════════════════════════════════════════════
#  Parameter extraction
# ════════════════════════════════════════════════════════════════════════

MOE_PATTERNS = {
    re.compile(r'(\d+)x(\d+\.?\d*)[bB]', re.I): lambda m: int(m.group(1)) * float(m.group(2)),
}

def extract_params_b(name: str) -> float | None:
    """Extract parameter count in billions from model name."""
    short = name.split("/")[-1].lower()

    # MoE patterns: 8x7B, 8x22B
    for pat, fn in MOE_PATTERNS.items():
        m = pat.search(short)
        if m:
            return fn(m)

    # Standard: 70b, 7.1b, 0.5b
    m = re.search(r'(\d+\.?\d*)\s*[bB](?!\w*ase|it|ench|lock|oost|yte)', short)
    if m:
        val = float(m.group(1))
        if val < 0.01 or val > 2000:
            return None
        return val

    # Millions: 248m, 400m → convert to B
    m = re.search(r'(\d+\.?\d*)\s*[mM](?:oE|$|-|_)', short)
    if m:
        val = float(m.group(1))
        if 10 <= val <= 50000:
            return val / 1000
    return None


def classify_size(params_b: float | None) -> str:
    if params_b is None:
        return "unknown"
    if params_b < 7:
        return "small"
    if params_b <= 30:
        return "medium"
    return "large"


# ════════════════════════════════════════════════════════════════════════
#  MH-DIF (returns per-item arrays)
# ════════════════════════════════════════════════════════════════════════

def mh_dif_per_item(X_ref, X_foc, n_strata=N_STRATA):
    """Vectorized MH-DIF returning per-item delta, p-value, is_c."""
    n_ref, J = X_ref.shape
    n_foc = X_foc.shape[0]

    tot_ref = X_ref.sum(axis=1).astype(np.float64)
    tot_foc = X_foc.sum(axis=1).astype(np.float64)
    s_all = np.concatenate([tot_ref, tot_foc])

    edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
    n_bins = len(edges) - 1
    if n_bins < 1:
        return np.zeros(J), np.ones(J), np.zeros(J, dtype=bool)

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
    return delta_j, pval_j, is_c


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    rm = np.load(f"{ROOT}/response_matrix.npz")
    X = sparse.csr_matrix((rm['data'], rm['indices'], rm['indptr']),
                          shape=tuple(rm['shape'])).toarray().astype(np.int8)
    idx = np.load(f"{ROOT}/response_matrix_index.npz", allow_pickle=True)
    model_names = idx['model_names']
    item_ids = idx['item_ids']
    J = len(item_ids)

    meta = pd.read_csv(ROOT / "model_metadata.csv")
    name2row = {n: i for i, n in enumerate(model_names)}

    # Extract params and classify
    records = []
    for _, row in meta.iterrows():
        nm = row['model_name']
        params = extract_params_b(nm)
        sz = classify_size(params)
        records.append({
            'model_name': nm,
            'params_b': params,
            'size_group': sz,
            'cohort_temporal': row['cohort_temporal'],
        })
    df = pd.DataFrame(records)

    # Save classification
    df.to_csv(OUT / "model_size_classification.csv", index=False)

    # Print size distribution
    print("=== Model Size Classification ===")
    ct = pd.crosstab(df['size_group'], df['cohort_temporal'], margins=True)
    print(ct)
    print()

    # Item difficulty (overall proportion correct)
    difficulty = X.mean(axis=0).astype(np.float64)

    # Run DIF per size group
    results = []
    dif_items_per_group = {}

    for grp in ['small', 'medium', 'large']:
        sub = df[(df['size_group'] == grp)]
        ref_names = sub[sub['cohort_temporal'] == '2023']['model_name'].values
        foc_names = sub[sub['cohort_temporal'] == '2024']['model_name'].values

        ref_rows = [name2row[n] for n in ref_names if n in name2row]
        foc_rows = [name2row[n] for n in foc_names if n in name2row]

        n_ref = len(ref_rows)
        n_foc = len(foc_rows)

        print(f"--- {grp.upper()} (N_ref={n_ref}, N_foc={n_foc}) ---")

        if n_ref < 5 or n_foc < 5:
            print("  Too few models, skipping.\n")
            results.append({
                'size_group': grp, 'n_ref': n_ref, 'n_foc': n_foc,
                'c_pct': np.nan, 'n_c_items': 0,
                'rho_diff_delta': np.nan, 'rho_p': np.nan,
                'n_focal_favoring': 0, 'n_ref_favoring': 0,
            })
            continue

        X_ref = X[ref_rows]
        X_foc = X[foc_rows]

        delta_j, pval_j, is_c = mh_dif_per_item(X_ref, X_foc)

        c_pct = is_c.sum() / J * 100
        n_c = is_c.sum()
        dif_items_per_group[grp] = set(np.where(is_c)[0])

        # Direction: delta > 0 → reference-favoring, delta < 0 → focal-favoring
        delta_c = delta_j[is_c]
        n_ref_fav = (delta_c > 0).sum()
        n_foc_fav = (delta_c < 0).sum()

        # Correlation: difficulty vs delta among C items
        if n_c >= 3:
            diff_c = difficulty[is_c]
            rho, rho_p = spearmanr(diff_c, delta_c)
        else:
            rho, rho_p = np.nan, np.nan

        print(f"  C% = {c_pct:.2f}%  ({n_c}/{J} items)")
        print(f"  Ref-favoring: {n_ref_fav}, Focal-favoring: {n_foc_fav}")
        print(f"  ρ(difficulty, Δ_MH) among C items: {rho:.3f}  (p={rho_p:.4f})")
        print()

        results.append({
            'size_group': grp,
            'n_ref': n_ref,
            'n_foc': n_foc,
            'c_pct': round(c_pct, 2),
            'n_c_items': int(n_c),
            'rho_diff_delta': round(rho, 4) if not np.isnan(rho) else np.nan,
            'rho_p': round(rho_p, 6) if not np.isnan(rho_p) else np.nan,
            'n_focal_favoring': int(n_foc_fav),
            'n_ref_favoring': int(n_ref_fav),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "model_size_results.csv", index=False)

    # Cross-group Jaccard overlap
    print("=== Cross-Group DIF Item Overlap (Jaccard) ===")
    groups_with_items = [(g, s) for g, s in dif_items_per_group.items() if len(s) > 0]
    jaccard_rows = []
    for i in range(len(groups_with_items)):
        for j in range(i + 1, len(groups_with_items)):
            g1, s1 = groups_with_items[i]
            g2, s2 = groups_with_items[j]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            jac = inter / union if union > 0 else 0
            print(f"  J({g1}, {g2}) = {jac:.3f}  (|∩|={inter}, |∪|={union})")
            jaccard_rows.append({
                'group_1': g1, 'group_2': g2,
                'intersection': inter, 'union': union,
                'jaccard': round(jac, 4),
            })

    # Generate report
    report = generate_report(res_df, ct, jaccard_rows, df)
    (OUT / "model_size_dif_report.md").write_text(report)
    print(f"\nAll outputs saved to {OUT}")


def generate_report(res_df, ct, jaccard_rows, clf_df):
    lines = ["# R19 Exp 2 — Model-Size DIF Decomposition\n"]

    lines.append("## 1. Model Classification\n")
    lines.append("Models classified by parameter count extracted from names.\n")
    lines.append("- **Small**: < 7B")
    lines.append("- **Medium**: 7B–30B")
    lines.append("- **Large**: > 30B")
    lines.append("- **Unknown**: parameter count not extractable\n")
    lines.append("### Distribution by cohort\n")
    lines.append("```")
    lines.append(ct.to_string())
    lines.append("```\n")

    # Percentage of unknowns
    n_unk = (clf_df['size_group'] == 'unknown').sum()
    n_tot = len(clf_df)
    lines.append(f"Unknown: {n_unk}/{n_tot} ({n_unk/n_tot*100:.1f}%)\n")

    lines.append("## 2. Per-Group MH-DIF Results\n")
    lines.append("Temporal DIF: reference = 2023, focal = 2024.\n")
    lines.append("| Group | N_ref | N_foc | C% | N_C items | Ref-fav | Foc-fav | ρ(diff,Δ) | ρ p-val |")
    lines.append("|-------|-------|-------|----|-----------|---------|---------|-----------|---------|")
    for _, r in res_df.iterrows():
        rho_str = f"{r['rho_diff_delta']:.3f}" if not pd.isna(r['rho_diff_delta']) else "—"
        rho_p_str = f"{r['rho_p']:.4f}" if not pd.isna(r['rho_p']) else "—"
        lines.append(
            f"| {r['size_group']} | {r['n_ref']} | {r['n_foc']} | "
            f"{r['c_pct']:.2f} | {r['n_c_items']} | "
            f"{r['n_ref_favoring']} | {r['n_focal_favoring']} | "
            f"{rho_str} | {rho_p_str} |"
        )
    lines.append("")

    lines.append("## 3. Cross-Group DIF Item Overlap\n")
    if jaccard_rows:
        lines.append("| Pair | |∩| | |∪| | Jaccard |")
        lines.append("|------|-----|-----|---------|")
        for jr in jaccard_rows:
            lines.append(
                f"| {jr['group_1']}–{jr['group_2']} | "
                f"{jr['intersection']} | {jr['union']} | {jr['jaccard']:.3f} |"
            )
    else:
        lines.append("Not enough groups with DIF items.\n")
    lines.append("")

    lines.append("## 4. Interpretation\n")
    lines.append("*To be filled after reviewing results.*\n")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
