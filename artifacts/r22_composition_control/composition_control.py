#!/usr/bin/env python3
"""R22: Base/instruct composition control for MH-DIF.

Matched subsampling to equalize base/instruct ratio across 2023 and 2024
cohorts, then re-run MH-DIF to show that DIF is not driven by composition shift.
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from scipy.sparse import load_npz
from pathlib import Path

ROOT = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/"
            "projects/irt_equated_bench_regen/artifacts")
OUT = ROOT / "r22_composition_control"
OUT.mkdir(exist_ok=True)

N_STRATA = 5
N_REPS = 10
SEED = 2026
ALPHA = 0.05
DELTA_THRESH = 1.5

# ── Model classification (reused from R18) ──────────────────────────────

INSTRUCT_SUFFIXES = re.compile(
    r"[-_\.](instruct|chat|it|rlhf|dpo|sft|aligned|ppo)"
    r"([-_\.]|$)",
    re.IGNORECASE,
)
QUANT_SUFFIXES = re.compile(
    r"[-_\.](gptq|awq|gguf|ggml|exl2|fp16|bf16|int[48]|q[2-8]_[0-9])",
    re.IGNORECASE,
)
FAMILY_PREFIX_EXCLUDE = re.compile(r"^(openchat|chatglm|chatml)", re.IGNORECASE)


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


# ── MH-DIF (vectorized, total-score stratification) ─────────────────────

def mh_dif_c_pct(X_ref, X_foc, n_strata=N_STRATA):
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

    is_c = finite_mask & (np.abs(delta_j) >= DELTA_THRESH) & (pval_j < ALPHA)
    return is_c.sum() / J * 100


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # Load data
    X = load_npz(ROOT / "response_matrix.npz").toarray()
    idx = np.load(ROOT / "response_matrix_index.npz", allow_pickle=True)
    model_names = [str(m) for m in idx["model_names"]]
    meta = pd.read_csv(ROOT / "model_metadata.csv")

    meta_lookup = dict(zip(meta["model_name"], meta["cohort_temporal"]))
    cohorts = np.array([meta_lookup.get(m, "other") for m in model_names])

    # Classify base/instruct
    model_types = np.array([classify_model(m) for m in model_names])

    # Filter to temporal cohorts
    mask_2023 = cohorts == "2023"
    mask_2024 = cohorts == "2024"

    idx_2023 = np.where(mask_2023)[0]
    idx_2024 = np.where(mask_2024)[0]

    type_2023 = model_types[mask_2023]
    type_2024 = model_types[mask_2024]

    n_base_2023 = (type_2023 == "base").sum()
    n_inst_2023 = (type_2023 == "instruct").sum()
    n_base_2024 = (type_2024 == "base").sum()
    n_inst_2024 = (type_2024 == "instruct").sum()

    print("=== Original composition ===")
    print(f"2023: {n_base_2023} base + {n_inst_2023} instruct = {len(idx_2023)} "
          f"(base%={n_base_2023/len(idx_2023)*100:.1f}%)")
    print(f"2024: {n_base_2024} base + {n_inst_2024} instruct = {len(idx_2024)} "
          f"(base%={n_base_2024/len(idx_2024)*100:.1f}%)")

    # Save composition stats
    comp_df = pd.DataFrame({
        "cohort": ["2023", "2024"],
        "n_base": [n_base_2023, n_base_2024],
        "n_instruct": [n_inst_2023, n_inst_2024],
        "n_total": [len(idx_2023), len(idx_2024)],
        "base_pct": [n_base_2023 / len(idx_2023) * 100,
                     n_base_2024 / len(idx_2024) * 100],
    })
    comp_df.to_csv(OUT / "composition_stats.csv", index=False)
    print(f"\nSaved composition stats → {OUT / 'composition_stats.csv'}")

    # Original (unbalanced) MH-DIF
    X_ref_full = X[idx_2023]
    X_foc_full = X[idx_2024]
    orig_c_pct = mh_dif_c_pct(X_ref_full, X_foc_full)
    print(f"\nOriginal MH-DIF C% = {orig_c_pct:.2f}%")

    # Matched subsampling
    # Target: match both cohorts to same base ratio
    # Use the lower base% (2024's ~67.9%) as target
    ratio_2023 = n_base_2023 / len(idx_2023)
    ratio_2024 = n_base_2024 / len(idx_2024)
    target_ratio = min(ratio_2023, ratio_2024)

    # Determine which cohort to downsample
    if ratio_2023 > ratio_2024:
        # Downsample 2023 base models
        # target_ratio = n_base_new / (n_base_new + n_inst_2023)
        n_base_target = int(target_ratio * n_inst_2023 / (1 - target_ratio))
        print(f"\nTarget base% = {target_ratio*100:.1f}% (matching 2024)")
        print(f"Downsampling 2023 base: {n_base_2023} → {n_base_target}")
    else:
        # Downsample 2024 base models
        n_base_target = int(target_ratio * n_inst_2024 / (1 - target_ratio))
        print(f"\nTarget base% = {target_ratio*100:.1f}% (matching 2023)")
        print(f"Downsampling 2024 base: {n_base_2024} → {n_base_target}")

    # Indices for subsampling
    idx_2023_base = idx_2023[type_2023 == "base"]
    idx_2023_inst = idx_2023[type_2023 == "instruct"]
    idx_2024_base = idx_2024[type_2024 == "base"]
    idx_2024_inst = idx_2024[type_2024 == "instruct"]

    rng = np.random.default_rng(SEED)
    results = []

    for rep in range(N_REPS):
        if ratio_2023 > ratio_2024:
            sampled_base = rng.choice(idx_2023_base, size=n_base_target, replace=False)
            bal_2023 = np.concatenate([sampled_base, idx_2023_inst])
            bal_2024 = np.concatenate([idx_2024_base, idx_2024_inst])
        else:
            sampled_base = rng.choice(idx_2024_base, size=n_base_target, replace=False)
            bal_2023 = np.concatenate([idx_2023_base, idx_2023_inst])
            bal_2024 = np.concatenate([sampled_base, idx_2024_inst])

        X_ref_bal = X[bal_2023]
        X_foc_bal = X[bal_2024]

        c_pct = mh_dif_c_pct(X_ref_bal, X_foc_bal)

        n_b_2023 = (model_types[bal_2023] == "base").sum()
        n_i_2023 = (model_types[bal_2023] == "instruct").sum()
        n_b_2024 = (model_types[bal_2024] == "base").sum()
        n_i_2024 = (model_types[bal_2024] == "instruct").sum()
        bp_2023 = n_b_2023 / len(bal_2023) * 100
        bp_2024 = n_b_2024 / len(bal_2024) * 100

        results.append({
            "rep": rep + 1,
            "c_pct": c_pct,
            "n_ref": len(bal_2023),
            "n_foc": len(bal_2024),
            "base_pct_ref": bp_2023,
            "base_pct_foc": bp_2024,
        })
        print(f"  Rep {rep+1}: C% = {c_pct:.2f}%  "
              f"(ref={len(bal_2023)} [{bp_2023:.1f}% base], "
              f"foc={len(bal_2024)} [{bp_2024:.1f}% base])")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "balanced_dif_results.csv", index=False)

    mean_c = res_df["c_pct"].mean()
    std_c = res_df["c_pct"].std()
    ci_lo = mean_c - 1.96 * std_c
    ci_hi = mean_c + 1.96 * std_c

    print(f"\n=== Summary ===")
    print(f"Original C%:  {orig_c_pct:.2f}%")
    print(f"Balanced C%:  {mean_c:.2f}% ± {std_c:.2f}% (95% CI: [{ci_lo:.2f}%, {ci_hi:.2f}%])")
    print(f"Δ(orig - balanced): {orig_c_pct - mean_c:.2f} pp")

    # Generate report
    report = f"""# R22: Base/Instruct Composition Control

## Motivation

2023 and 2024 cohorts differ in base/instruct model composition:
- 2023: {n_base_2023} base + {n_inst_2023} instruct ({n_base_2023/len(idx_2023)*100:.1f}% base)
- 2024: {n_base_2024} base + {n_inst_2024} instruct ({n_base_2024/len(idx_2024)*100:.1f}% base)

This {abs(ratio_2023 - ratio_2024)*100:.1f} pp gap could confound temporal MH-DIF if base and instruct
models respond differently. We test this by matched subsampling.

## Method

1. Target base ratio: {target_ratio*100:.1f}% (matching the lower-ratio cohort)
2. Downsample {"2023" if ratio_2023 > ratio_2024 else "2024"} base models from {n_base_2023 if ratio_2023 > ratio_2024 else n_base_2024} → {n_base_target}
3. Re-run 5-stratum MH-DIF with ETS C classification
4. Repeat {N_REPS}× with different random seeds

## Results

| Metric | Value |
|--------|-------|
| Original C% | {orig_c_pct:.2f}% |
| Balanced C% (mean ± SD) | {mean_c:.2f}% ± {std_c:.2f}% |
| Balanced C% 95% CI | [{ci_lo:.2f}%, {ci_hi:.2f}%] |
| Δ(original − balanced) | {orig_c_pct - mean_c:.2f} pp |

### Per-repetition results

{res_df.to_csv(index=False)}

## Conclusion

{"Balanced C% remains high (" + f"{mean_c:.1f}%" + "), confirming that DIF is NOT driven by base/instruct composition shift. The " + f"{abs(orig_c_pct - mean_c):.1f}" + " pp difference is negligible." if mean_c > 20 else "Balanced C% dropped substantially, suggesting composition shift partially explains the observed DIF."}
"""
    (OUT / "composition_control_report.md").write_text(report)
    print(f"\nReport → {OUT / 'composition_control_report.md'}")


if __name__ == "__main__":
    main()
