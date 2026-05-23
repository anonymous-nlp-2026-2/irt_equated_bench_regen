#!/usr/bin/env python3
"""R18: Base vs Instruct MH-DIF separation experiment."""

import re
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist, spearmanr, chi2_contingency

ROOT = "artifacts"
OUT = f"{ROOT}/r18_base_vs_instruct_dif"

N_STRATA = 5
N_BOOT = 1000
ALPHA = 0.05
DELTA_THRESH = 1.5

# ── Part 1: Model classification ─────────────────────────────────────────

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


# ── Part 2: MH-DIF (extended) ────────────────────────────────────────────

def mh_dif_full(X_ref, X_foc, n_strata=N_STRATA):
    """Vectorized MH-DIF returning per-item details.

    Returns (c_pct, delta_mh, pval, is_c).
    """
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


def ets_class(delta):
    a = abs(delta)
    if a < 1.0:
        return "A"
    elif a < 1.5:
        return "B"
    else:
        return "C"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    meta = pd.read_csv(f"{ROOT}/plan_032_dif_cleaned_mmlu/cleaned_mmlu_analysis.csv")
    rm = np.load(f"{ROOT}/response_matrix.npz")
    X = sparse.csr_matrix((rm["data"], rm["indices"], rm["indptr"]), shape=tuple(rm["shape"]))
    idx = np.load(f"{ROOT}/response_matrix_index.npz", allow_pickle=True)
    model_names = idx["model_names"]
    item_ids = idx["item_ids"]

    # Build model name → row index lookup
    name2row = {n: i for i, n in enumerate(model_names)}

    # Classify models
    meta = meta[meta["cohort_temporal"].isin(["2023", "2024"])].copy()
    meta["model_type"] = meta["model_name"].apply(classify_model)

    # Save classification
    class_df = meta[["model_name", "model_type", "cohort_temporal", "family"]].copy()
    class_df.to_csv(f"{OUT}/model_classification.csv", index=False)

    # Map to response matrix rows
    meta["rm_idx"] = meta["model_name"].map(name2row)
    meta = meta.dropna(subset=["rm_idx"])
    meta["rm_idx"] = meta["rm_idx"].astype(int)

    # Group counts
    groups = {}
    for mtype in ["base", "instruct"]:
        ref_mask = (meta["model_type"] == mtype) & (meta["cohort_temporal"] == "2023")
        foc_mask = (meta["model_type"] == mtype) & (meta["cohort_temporal"] == "2024")
        groups[mtype] = {
            "ref_idx": meta.loc[ref_mask, "rm_idx"].values,
            "foc_idx": meta.loc[foc_mask, "rm_idx"].values,
        }

    print("\n=== Part 4: Sample sizes ===")
    for mtype in ["base", "instruct"]:
        n_ref = len(groups[mtype]["ref_idx"])
        n_foc = len(groups[mtype]["foc_idx"])
        flag = " ⚠ UNDERPOWERED" if min(n_ref, n_foc) < 80 else ""
        print(f"  {mtype:>10s}: ref(2023)={n_ref}, foc(2024)={n_foc}{flag}")
    print(f"  {'full':>10s}: ref(2023)=1080, foc(2024)=807  (reported baseline)")

    # Run MH-DIF per group
    results = {}
    for mtype in ["base", "instruct"]:
        ri, fi = groups[mtype]["ref_idx"], groups[mtype]["foc_idx"]
        print(f"\nRunning MH-DIF for {mtype} ({len(ri)} ref, {len(fi)} foc)...")
        X_ref = X[ri]
        X_foc = X[fi]
        c_pct, delta, pval, is_c = mh_dif_full(X_ref, X_foc)
        results[mtype] = {"c_pct": c_pct, "delta": delta, "pval": pval, "is_c": is_c}
        print(f"  C% = {c_pct:.1f}%  ({is_c.sum()}/{len(is_c)} items)")

    # ── Part 3a: Bootstrap CI for C% difference ──────────────────────────
    print("\n=== Part 3a: Bootstrap C% difference ===")
    rng = np.random.default_rng(42)
    boot_diff = np.zeros(N_BOOT)

    for b in range(N_BOOT):
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}...")
        cpcts = {}
        for mtype in ["base", "instruct"]:
            ri, fi = groups[mtype]["ref_idx"], groups[mtype]["foc_idx"]
            ri_b = rng.choice(ri, size=len(ri), replace=True)
            fi_b = rng.choice(fi, size=len(fi), replace=True)
            c, _, _, _ = mh_dif_full(X[ri_b], X[fi_b])
            cpcts[mtype] = c
        boot_diff[b] = cpcts["base"] - cpcts["instruct"]

    ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
    mean_diff = boot_diff.mean()
    print(f"  C%_base - C%_instruct = {results['base']['c_pct']:.1f} - {results['instruct']['c_pct']:.1f} = {results['base']['c_pct'] - results['instruct']['c_pct']:.1f}")
    print(f"  Bootstrap 95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]")

    # ── Part 3b: Difficulty-direction gradient ────────────────────────────
    print("\n=== Part 3b: Difficulty-direction gradient ===")
    all_2023_idx = meta.loc[meta["cohort_temporal"] == "2023", "rm_idx"].values
    X_2023 = X[all_2023_idx]
    item_diff = np.asarray(X_2023.mean(axis=0), dtype=np.float64).ravel()

    for mtype in ["base", "instruct"]:
        c_mask = results[mtype]["is_c"]
        if c_mask.sum() < 3:
            print(f"  {mtype}: too few C items ({c_mask.sum()}) for correlation")
            continue
        rho, p = spearmanr(item_diff[c_mask], results[mtype]["delta"][c_mask])
        print(f"  {mtype}: ρ(difficulty, Δ_MH) = {rho:.3f}  (p={p:.2e}, n={c_mask.sum()})")

    # ── Part 3c: DIF item overlap ─────────────────────────────────────────
    print("\n=== Part 3c: DIF item overlap ===")
    c_base = set(np.where(results["base"]["is_c"])[0])
    c_inst = set(np.where(results["instruct"]["is_c"])[0])
    inter = c_base & c_inst
    union = c_base | c_inst
    jaccard = len(inter) / len(union) if union else 0

    p_b = len(c_base) / len(item_ids)
    p_i = len(c_inst) / len(item_ids)
    expected_jaccard = (p_b * p_i) / (p_b + p_i - p_b * p_i) if (p_b + p_i - p_b * p_i) > 0 else 0

    only_base = len(c_base - c_inst)
    only_inst = len(c_inst - c_base)
    both = len(inter)
    print(f"  |C_base|={len(c_base)}, |C_instruct|={len(c_inst)}")
    print(f"  Only base: {only_base}, Only instruct: {only_inst}, Both: {both}")
    print(f"  Jaccard = {jaccard:.3f}  (expected under independence: {expected_jaccard:.3f})")

    # ── Part 3d: Direction distribution ───────────────────────────────────
    print("\n=== Part 3d: DIF direction distribution ===")
    dir_table = {}
    for mtype in ["base", "instruct"]:
        d = results[mtype]["delta"][results[mtype]["is_c"]]
        n_foc_fav = (d > 0).sum()
        n_ref_fav = (d < 0).sum()
        dir_table[mtype] = (int(n_foc_fav), int(n_ref_fav))
        total = n_foc_fav + n_ref_fav
        print(f"  {mtype}: focal-favoring={n_foc_fav} ({n_foc_fav/total*100:.1f}%), ref-favoring={n_ref_fav} ({n_ref_fav/total*100:.1f}%)")

    ct = np.array([[dir_table["base"][0], dir_table["base"][1]],
                    [dir_table["instruct"][0], dir_table["instruct"][1]]])
    if ct.min() > 0:
        chi2_val, p_dir, _, _ = chi2_contingency(ct)
        print(f"  χ²={chi2_val:.2f}, p={p_dir:.3e}")
    else:
        p_dir = np.nan
        print("  χ² test not possible (zero cell)")

    # ── Save per-item CSV ─────────────────────────────────────────────────
    print("\nSaving per-item results...")

    # Extract subject from item_id (format: subject_qXXXX or similar)
    def get_subject(iid):
        parts = iid.rsplit("_", 1)
        return parts[0] if len(parts) > 1 else iid

    item_df = pd.DataFrame({
        "item_id": item_ids,
        "subject": [get_subject(str(i)) for i in item_ids],
        "difficulty": item_diff,
        "delta_mh_base": results["base"]["delta"],
        "p_base": results["base"]["pval"],
        "ets_base": [ets_class(d) for d in results["base"]["delta"]],
        "is_c_base": results["base"]["is_c"],
        "delta_mh_instruct": results["instruct"]["delta"],
        "p_instruct": results["instruct"]["pval"],
        "ets_instruct": [ets_class(d) for d in results["instruct"]["delta"]],
        "is_c_instruct": results["instruct"]["is_c"],
    })
    item_df.to_csv(f"{OUT}/base_instruct_item_results.csv", index=False)

    # ── Generate report ───────────────────────────────────────────────────
    print("Writing report...")

    n_base_ref = len(groups["base"]["ref_idx"])
    n_base_foc = len(groups["base"]["foc_idx"])
    n_inst_ref = len(groups["instruct"]["ref_idx"])
    n_inst_foc = len(groups["instruct"]["foc_idx"])

    report = f"""# R18: Base vs Instruct DIF Separation Experiment

## Sample Sizes

| Group | Ref (2023) | Foc (2024) | Total |
|-------|-----------|-----------|-------|
| Base | {n_base_ref} | {n_base_foc} | {n_base_ref + n_base_foc} |
| Instruct | {n_inst_ref} | {n_inst_foc} | {n_inst_ref + n_inst_foc} |
| Full (reported) | 1,080 | 807 | 1,887 |

{"**⚠ Warning:** Some subgroups have N < 80; interpret with caution." if min(n_base_ref, n_base_foc, n_inst_ref, n_inst_foc) < 80 else "All subgroups have N ≥ 80; adequate power."}

## C% Comparison

| Group | C% | DIF-C items |
|-------|-----|------------|
| Base | {results['base']['c_pct']:.1f}% | {results['base']['is_c'].sum()}/{len(item_ids)} |
| Instruct | {results['instruct']['c_pct']:.1f}% | {results['instruct']['is_c'].sum()}/{len(item_ids)} |
| Full (reported) | 31.3% | — |

**Difference (base − instruct):** {results['base']['c_pct'] - results['instruct']['c_pct']:.1f} pp
**Bootstrap 95% CI:** [{ci_lo:.1f}, {ci_hi:.1f}] (N={N_BOOT})

## Difficulty–Direction Gradient

Spearman correlation between item difficulty (2023 accuracy) and Δ_MH among DIF-C items:

"""

    for mtype in ["base", "instruct"]:
        c_mask = results[mtype]["is_c"]
        if c_mask.sum() >= 3:
            rho, p = spearmanr(item_diff[c_mask], results[mtype]["delta"][c_mask])
            report += f"- **{mtype.capitalize()}:** ρ = {rho:.3f} (p = {p:.2e}, n = {c_mask.sum()})\n"
        else:
            report += f"- **{mtype.capitalize()}:** too few C items ({c_mask.sum()}) for correlation\n"

    report += f"""
## DIF Item Overlap

| Metric | Value |
|--------|-------|
| C_base only | {only_base} |
| C_instruct only | {only_inst} |
| Both | {both} |
| Jaccard index | {jaccard:.3f} |
| Expected Jaccard (independence) | {expected_jaccard:.3f} |
| Jaccard / Expected ratio | {jaccard / expected_jaccard:.1f}× |

## DIF Direction Distribution

| Group | Focal-favoring (2024↑) | Ref-favoring (2023↑) |
|-------|----------------------|---------------------|
| Base | {dir_table['base'][0]} ({dir_table['base'][0]/(sum(dir_table['base']))*100:.1f}%) | {dir_table['base'][1]} ({dir_table['base'][1]/(sum(dir_table['base']))*100:.1f}%) |
| Instruct | {dir_table['instruct'][0]} ({dir_table['instruct'][0]/(sum(dir_table['instruct']))*100:.1f}%) | {dir_table['instruct'][1]} ({dir_table['instruct'][1]/(sum(dir_table['instruct']))*100:.1f}%) |

χ² = {chi2_val:.2f}, p = {p_dir:.3e}

## Output Files

- `model_classification.csv` — {len(class_df)} models classified
- `base_instruct_item_results.csv` — {len(item_ids)} items with per-group DIF statistics
"""

    with open(f"{OUT}/base_instruct_dif_report.md", "w") as f:
        f.write(report)

    print(f"\nDone. Results in {OUT}/")


if __name__ == "__main__":
    main()
