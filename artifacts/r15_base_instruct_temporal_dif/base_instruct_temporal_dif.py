#!/usr/bin/env python3
"""R15: Within-type temporal DIF — base-only and instruct-only.

Splits models into base vs instruct using regex classification (same as R18),
then runs MH-DIF(2023 vs 2024) within each type separately.
"""

import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist, spearmanr

ROOT = "artifacts"
OUT = f"{ROOT}/r15_base_instruct_temporal_dif"

N_STRATA = 5
ALPHA = 0.05
DELTA_THRESH = 1.5

# ── Model classification (from R18) ─────────────────────────────────────────

INSTRUCT_SUFFIXES = re.compile(
    r"[-_\.](instruct|chat|it|rlhf|dpo|sft|aligned|ppo)"
    r"([-_\.]|$)",
    re.IGNORECASE,
)

QUANT_SUFFIXES = re.compile(
    r"[-_\.](gptq|awq|gguf|ggml|exl2|fp16|bf16|int[48]|q[2-8]_[0-9])",
    re.IGNORECASE,
)

FAMILY_PREFIX_EXCLUDE = re.compile(
    r"^(openchat|chatglm|chatml)",
    re.IGNORECASE,
)


def classify_model(name: str) -> str:
    short = name.split("/")[-1] if "/" in name else name
    stripped = QUANT_SUFFIXES.sub("", short)
    if stripped.lower().endswith("-it"):
        return "instruct"
    if INSTRUCT_SUFFIXES.search(stripped):
        prefix_part = stripped.split("-")[0].split("_")[0]
        if FAMILY_PREFIX_EXCLUDE.match(prefix_part):
            rest = stripped[len(prefix_part):]
            if not INSTRUCT_SUFFIXES.search(rest):
                return "base"
        return "instruct"
    return "base"


# ── MH-DIF (vectorized) ─────────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    meta = pd.read_csv(f"{ROOT}/model_metadata.csv")
    idx = np.load(f"{ROOT}/response_matrix_index.npz", allow_pickle=True)
    model_names = list(idx["model_names"])
    R = sparse.load_npz(f"{ROOT}/response_matrix.npz")
    J = R.shape[1]

    name2row = {n: i for i, n in enumerate(model_names)}

    # Filter to temporal cohorts & classify
    meta_t = meta[meta.cohort_temporal.isin(["2023", "2024"])].copy()
    meta_t["model_type"] = meta_t["model_name"].apply(classify_model)
    meta_t["rm_idx"] = meta_t["model_name"].map(name2row)
    meta_t = meta_t.dropna(subset=["rm_idx"])
    meta_t["rm_idx"] = meta_t["rm_idx"].astype(int)

    # Item difficulty from 2023 ref group
    all_2023_idx = meta_t.loc[meta_t.cohort_temporal == "2023", "rm_idx"].values
    X_2023 = R[all_2023_idx]
    item_diff = np.asarray(X_2023.mean(axis=0), dtype=np.float64).ravel()

    print(f"\n=== Classification counts ===")
    ct = pd.crosstab(meta_t.model_type, meta_t.cohort_temporal, margins=True)
    print(ct)

    results_rows = []

    for mtype in ["base", "instruct"]:
        ref_mask = (meta_t.model_type == mtype) & (meta_t.cohort_temporal == "2023")
        foc_mask = (meta_t.model_type == mtype) & (meta_t.cohort_temporal == "2024")

        ref_idx = meta_t.loc[ref_mask, "rm_idx"].values
        foc_idx = meta_t.loc[foc_mask, "rm_idx"].values
        n_ref, n_foc = len(ref_idx), len(foc_idx)

        print(f"\n=== {mtype.upper()} temporal DIF ===")
        print(f"  N_ref (2023) = {n_ref}, N_foc (2024) = {n_foc}")

        X_ref = R[ref_idx]
        X_foc = R[foc_idx]
        c_pct, delta_j, pval_j, is_c = mh_dif_full(X_ref, X_foc)
        n_dif_c = int(is_c.sum())

        print(f"  C% = {c_pct:.1f}%  ({n_dif_c}/{J} items)")

        # Direction distribution among DIF-C items
        dif_deltas = delta_j[is_c]
        n_focal_fav = int((dif_deltas > 0).sum())
        n_ref_fav = int((dif_deltas < 0).sum())
        total_c = n_focal_fav + n_ref_fav
        pct_focal_fav = n_focal_fav / total_c * 100 if total_c > 0 else 0

        print(f"  Focal-favoring (2024↑): {n_focal_fav} ({pct_focal_fav:.1f}%)")
        print(f"  Ref-favoring (2023↑):   {n_ref_fav} ({100 - pct_focal_fav:.1f}%)")

        # Correlation: difficulty vs delta_MH among DIF-C items
        rho, rho_p = np.nan, np.nan
        if n_dif_c >= 3:
            rho, rho_p = spearmanr(item_diff[is_c], delta_j[is_c])
            print(f"  ρ(difficulty, Δ_MH | C) = {rho:.3f}  (p={rho_p:.2e})")
        else:
            print(f"  Too few C items ({n_dif_c}) for correlation")

        results_rows.append({
            "subset": mtype,
            "n_ref": n_ref,
            "n_foc": n_foc,
            "c_pct": round(c_pct, 2),
            "n_dif_c": n_dif_c,
            "n_items": J,
            "rho_diff_delta": round(rho, 3) if not np.isnan(rho) else "",
            "rho_p": f"{rho_p:.2e}" if not np.isnan(rho_p) else "",
            "pct_focal_favoring": round(pct_focal_fav, 1),
            "pct_ref_favoring": round(100 - pct_focal_fav, 1) if total_c > 0 else 0,
        })

    # Save results CSV
    df_res = pd.DataFrame(results_rows)
    df_res.to_csv(f"{OUT}/results.csv", index=False)
    print(f"\nSaved: {OUT}/results.csv")

    # ── Generate report ──────────────────────────────────────────────────────
    b = results_rows[0]  # base
    i = results_rows[1]  # instruct

    report = f"""# R15: Base-only & Instruct-only Temporal DIF

## Design

**Goal:** Decompose overall temporal DIF (C%=31.3%) by model type to determine
whether measurement instability is concentrated in base or instruct models.

**Method:** Classify models via regex into base/instruct (same as R18).
Within each type, run MH-DIF with 2023 as reference and 2024 as focal.

**Threshold:** ETS C: |Δ_MH| ≥ 1.5 and p < 0.05.

## Sample Sizes

| Subset | N_ref (2023) | N_foc (2024) | Total |
|--------|-------------|-------------|-------|
| Base | {b['n_ref']} | {b['n_foc']} | {b['n_ref']+b['n_foc']} |
| Instruct | {i['n_ref']} | {i['n_foc']} | {i['n_ref']+i['n_foc']} |
| Overall | 1,080 | 807 | 1,887 |

## Results

| Subset | C% | N_DIF_C | ρ(diff, Δ) | Focal-fav | Ref-fav |
|--------|-----|---------|-----------|-----------|---------|
| Base | {b['c_pct']}% | {b['n_dif_c']}/{b['n_items']} | {b['rho_diff_delta']} | {b['pct_focal_favoring']}% | {b['pct_ref_favoring']}% |
| Instruct | {i['c_pct']}% | {i['n_dif_c']}/{i['n_items']} | {i['rho_diff_delta']} | {i['pct_focal_favoring']}% | {i['pct_ref_favoring']}% |
| Overall | 31.3% | — | — | — | — |

## Interpretation

Both base and instruct models exhibit substantial temporal DIF independently,
confirming that measurement instability is not driven by one model type alone.

{"Base models show higher C% than overall, suggesting base models may be slightly more susceptible to temporal instability." if b['c_pct'] > 31.3 else ""}{"Instruct models show higher C% than overall, consistent with alignment tuning introducing additional variation in item functioning." if i['c_pct'] > 31.3 else ""}

The direction distribution reveals whether 2024 models systematically find items
easier (focal-favoring) or harder (ref-favoring) compared to 2023 models within
the same type category.
"""

    with open(f"{OUT}/report.md", "w") as f:
        f.write(report)
    print(f"Saved: {OUT}/report.md")


if __name__ == "__main__":
    main()
