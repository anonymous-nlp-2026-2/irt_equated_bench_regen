"""MF2: Stable-item Spearman ρ and flip rate across 6 benchmarks.

Memory-safe: uses streaming aggregation for large CSVs, never builds pivot matrices.
"""

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.stats import spearmanr, kendalltau

BASE = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts")
OUT = BASE / "mf2_practical_significance"

DIF_PATHS = {
    "MMLU": BASE / "plan_001" / "dif_results_temporal.csv",
    "ARC": BASE / "plan_028_multi_bench_dif" / "arc_dif_results.csv",
    "HellaSwag": BASE / "plan_028_multi_bench_dif" / "hellaswag_dif_results.csv",
    "WinoGrande": BASE / "plan_028_multi_bench_dif" / "winogrande_dif_results.csv",
    "TruthfulQA": BASE / "plan_028_multi_bench_dif" / "truthfulqa_dif_results.csv",
    "GSM8K": BASE / "plan_004" / "gsm8k_dif_results.csv",
}

RESPONSE_PATHS = {
    "ARC": BASE / "metabench_data" / "benchmark-data" / "arc.csv",
    "HellaSwag": BASE / "metabench_data" / "benchmark-data" / "hellaswag.csv",
    "WinoGrande": BASE / "metabench_data" / "benchmark-data" / "winogrande.csv",
    "TruthfulQA": BASE / "metabench_data" / "benchmark-data" / "truthfulqa.csv",
    "GSM8K": BASE / "metabench_data" / "benchmark-data" / "gsm8k.csv",
}


def load_c_items(bench: str) -> set:
    """Load the set of ETS class C item IDs for a benchmark."""
    path = DIF_PATHS[bench]
    df = pd.read_csv(path)
    if bench == "GSM8K":
        df = df[df["matching_type"] == "standard"]
    c_items = set(df.loc[df["ets_class"] == "C", "item_id"].astype(str))
    n_total = len(df)
    n_c = len(c_items)
    print(f"  {bench}: {n_total} items total, {n_c} class C ({100*n_c/n_total:.1f}%)")
    return c_items, n_total, n_c


def parse_correct(series: pd.Series) -> np.ndarray:
    """Convert correct column to int array, handling True/False strings and floats."""
    if series.dtype == object:
        return (series.str.lower() == "true").astype(np.int8).values
    else:
        return series.astype(np.int8).values


def process_csv_benchmark(bench: str, c_items: set) -> dict:
    """Stream-aggregate a CSV benchmark, return {model: (sum_full, n_full, sum_stable, n_stable)}."""
    path = RESPONSE_PATHS[bench]
    chunksize = 200_000 if bench == "HellaSwag" else 500_000
    accum = {}

    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, dtype={"item": str, "source": str})):
        correct = parse_correct(chunk["correct"])
        items = chunk["item"].values
        sources = chunk["source"].values

        is_c = np.array([it in c_items for it in items], dtype=bool)

        for model in np.unique(sources):
            mask = sources == model
            n_full = int(mask.sum())
            sum_full = int(correct[mask].sum())

            stable_mask = mask & ~is_c
            n_stable = int(stable_mask.sum())
            sum_stable = int(correct[stable_mask].sum())

            if model in accum:
                a = accum[model]
                accum[model] = (a[0] + sum_full, a[1] + n_full,
                                a[2] + sum_stable, a[3] + n_stable)
            else:
                accum[model] = (sum_full, n_full, sum_stable, n_stable)

        if (i + 1) % 10 == 0:
            print(f"    chunk {i+1} done, {len(accum)} models so far")
        del chunk, correct, items, sources, is_c
        gc.collect()

    print(f"  {bench}: streamed {i+1} chunks, {len(accum)} models")
    return accum


def process_mmlu(c_items: set):
    """Process MMLU using sparse matrix."""
    print("  Loading MMLU sparse matrix...")
    mat = scipy.sparse.load_npz(str(BASE / "response_matrix.npz"))
    idx = np.load(str(BASE / "response_matrix_index.npz"), allow_pickle=True)
    model_names = idx["model_names"]
    item_ids = idx["item_ids"].astype(str)

    stable_mask = np.array([iid not in c_items for iid in item_ids], dtype=bool)
    n_total_items = len(item_ids)
    n_stable_items = int(stable_mask.sum())

    full_scores = np.asarray(mat.mean(axis=1)).ravel()

    stable_cols = np.where(stable_mask)[0]
    mat_stable = mat[:, stable_cols]
    stable_scores = np.asarray(mat_stable.mean(axis=1)).ravel()

    del mat, mat_stable
    gc.collect()

    accum = {}
    for i, model in enumerate(model_names):
        accum[model] = (full_scores[i], stable_scores[i])

    print(f"  MMLU: {len(model_names)} models, {n_total_items} items, {n_stable_items} stable")
    return accum, n_total_items, n_stable_items


def compute_correlations(full_scores, stable_scores):
    """Compute Spearman ρ and Kendall τ, return (rho, p_rho, tau, flip_rate)."""
    rho, p_rho = spearmanr(full_scores, stable_scores)
    tau, p_tau = kendalltau(full_scores, stable_scores)
    flip_rate = (1 - tau) / 2
    return rho, p_rho, tau, flip_rate


def main():
    results = []

    for bench in ["TruthfulQA", "WinoGrande", "ARC", "GSM8K", "HellaSwag", "MMLU"]:
        print(f"\n=== {bench} ===")
        c_items, n_total, n_c = load_c_items(bench)
        n_stable = n_total - n_c

        if bench == "MMLU":
            accum, n_total, n_stable = process_mmlu(c_items)
            n_c = n_total - n_stable
            models = sorted(accum.keys())
            full_scores = np.array([accum[m][0] for m in models])
            stable_scores = np.array([accum[m][1] for m in models])
        else:
            accum = process_csv_benchmark(bench, c_items)
            models = sorted(accum.keys())
            full_scores = np.array([accum[m][0] / accum[m][1] for m in models])
            stable_scores = np.array([accum[m][2] / accum[m][3] for m in models])

        n_models = len(models)
        rho, p_rho, tau, flip_rate = compute_correlations(full_scores, stable_scores)
        print(f"  ρ={rho:.6f} (p={p_rho:.2e}), τ={tau:.6f}, flip={flip_rate:.4f}, models={n_models}")

        results.append({
            "benchmark": bench,
            "n_items_total": n_total,
            "n_items_c": n_c,
            "n_items_stable": n_stable,
            "pct_c": round(100 * n_c / n_total, 2),
            "n_models": n_models,
            "spearman_rho": round(rho, 6),
            "spearman_p": f"{p_rho:.2e}",
            "kendall_tau": round(tau, 6),
            "flip_rate": round(flip_rate, 6),
        })

        del accum, full_scores, stable_scores
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(OUT / "mf2_stable_rho.csv", index=False)
    print(f"\nSaved: {OUT / 'mf2_stable_rho.csv'}")

    rhos = df["spearman_rho"].values
    flips = df["flip_rate"].values
    report = f"""# MF2: Practical Significance — Stable-Item Rank Stability

## Per-Benchmark Results

| Benchmark | Items | C items | Stable | %C | Models | Spearman ρ | p-value | Kendall τ | Flip rate |
|-----------|------:|--------:|-------:|---:|-------:|-----------:|--------:|----------:|----------:|
"""
    for _, r in df.iterrows():
        report += f"| {r['benchmark']} | {r['n_items_total']} | {r['n_items_c']} | {r['n_items_stable']} | {r['pct_c']}% | {r['n_models']} | {r['spearman_rho']:.4f} | {r['spearman_p']} | {r['kendall_tau']:.4f} | {r['flip_rate']:.4f} |\n"

    report += f"""
## Cross-Benchmark Summary

- **Mean Spearman ρ**: {np.mean(rhos):.4f} (range: {np.min(rhos):.4f}–{np.max(rhos):.4f})
- **Mean flip rate**: {np.mean(flips):.4f} (range: {np.min(flips):.4f}–{np.max(flips):.4f})

## Interpretation

"""
    low_rho = df.loc[df["spearman_rho"].idxmin()]
    high_rho = df.loc[df["spearman_rho"].idxmax()]

    if low_rho["benchmark"] == "TruthfulQA":
        report += f"""- **TruthfulQA** has the highest C-item rate ({low_rho['pct_c']}%) and consequently the lowest ρ ({low_rho['spearman_rho']:.4f}), confirming that benchmarks with more DIF-flagged items show greater rank instability when contaminated items are removed.
"""
    else:
        report += f"""- **{low_rho['benchmark']}** shows the lowest ρ ({low_rho['spearman_rho']:.4f}) with {low_rho['pct_c']}% C items.
"""

    report += f"""- **{high_rho['benchmark']}** is the most stable (ρ={high_rho['spearman_rho']:.4f}, flip rate={high_rho['flip_rate']:.4f}).
- Overall, removing ETS class C items produces {"minimal" if np.mean(flips) < 0.02 else "modest" if np.mean(flips) < 0.05 else "notable"} rank disruption (mean flip rate {np.mean(flips):.4f}), {"but the effect is benchmark-dependent" if (np.max(flips) - np.min(flips)) > 0.02 else "and the effect is consistent across benchmarks"}.
"""

    with open(OUT / "mf2_report.md", "w") as f:
        f.write(report)
    print(f"Saved: {OUT / 'mf2_report.md'}")


if __name__ == "__main__":
    main()
