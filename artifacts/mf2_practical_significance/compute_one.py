"""Process a single benchmark for MF2 analysis. Run as: python compute_one.py <benchmark_name>"""

import gc
import json
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


def load_c_items(bench):
    df = pd.read_csv(DIF_PATHS[bench])
    if bench == "GSM8K":
        df = df[df["matching_type"] == "standard"]
    c_items = set(df.loc[df["ets_class"] == "C", "item_id"].astype(str))
    n_total = len(df)
    n_c = len(c_items)
    print(f"  {n_total} items, {n_c} class C ({100*n_c/n_total:.1f}%)")
    del df; gc.collect()
    return c_items, n_total, n_c


def parse_correct(series):
    if series.dtype == object:
        return (series.str.lower() == "true").astype(np.int8).values
    return series.astype(np.int8).values


def process_csv(bench, c_items):
    path = RESPONSE_PATHS[bench]
    chunksize = 100_000 if bench == "HellaSwag" else 200_000
    accum = {}

    reader = pd.read_csv(path, chunksize=chunksize, dtype={"item": str, "source": str})
    for i, chunk in enumerate(reader):
        correct = parse_correct(chunk["correct"])
        items = chunk["item"].values
        sources = chunk["source"].values
        is_c = np.array([it in c_items for it in items], dtype=bool)

        for model in np.unique(sources):
            mask = sources == model
            sm = mask & ~is_c
            rec = accum.get(model, (0, 0, 0, 0))
            accum[model] = (
                rec[0] + int(correct[mask].sum()),
                rec[1] + int(mask.sum()),
                rec[2] + int(correct[sm].sum()),
                rec[3] + int(sm.sum()),
            )

        del chunk, correct, items, sources, is_c, mask, sm
        gc.collect()

        if (i + 1) % 20 == 0:
            print(f"    chunk {i+1}, {len(accum)} models")

    reader.close()
    del reader; gc.collect()
    print(f"  {i+1} chunks, {len(accum)} models")
    return accum


def process_mmlu(c_items):
    print("  Loading sparse matrix...")
    mat = scipy.sparse.load_npz(str(BASE / "response_matrix.npz"))
    idx = np.load(str(BASE / "response_matrix_index.npz"), allow_pickle=True)
    model_names = idx["model_names"]
    item_ids = idx["item_ids"].astype(str)

    stable_mask = np.array([iid not in c_items for iid in item_ids], dtype=bool)
    n_total = len(item_ids)
    n_stable = int(stable_mask.sum())

    full_scores = np.asarray(mat.mean(axis=1)).ravel()
    mat_stable = mat[:, np.where(stable_mask)[0]]
    stable_scores = np.asarray(mat_stable.mean(axis=1)).ravel()

    del mat, mat_stable, idx; gc.collect()
    print(f"  {len(model_names)} models, {n_total} items, {n_stable} stable")
    return model_names, full_scores, stable_scores, n_total, n_stable


def main():
    bench = sys.argv[1]
    print(f"=== {bench} ===")
    c_items, n_total, n_c = load_c_items(bench)
    n_stable = n_total - n_c

    if bench == "MMLU":
        model_names, full_scores, stable_scores, n_total, n_stable = process_mmlu(c_items)
        n_c = n_total - n_stable
        n_models = len(model_names)
    else:
        accum = process_csv(bench, c_items)
        models = sorted(accum.keys())
        full_scores = np.array([accum[m][0] / accum[m][1] for m in models])
        stable_scores = np.array([accum[m][2] / accum[m][3] for m in models])
        n_models = len(models)
        del accum; gc.collect()

    rho, p_rho = spearmanr(full_scores, stable_scores)
    tau, _ = kendalltau(full_scores, stable_scores)
    flip_rate = (1 - tau) / 2

    result = {
        "benchmark": bench,
        "n_items_total": int(n_total),
        "n_items_c": int(n_c),
        "n_items_stable": int(n_stable),
        "pct_c": round(100 * n_c / n_total, 2),
        "n_models": n_models,
        "spearman_rho": round(float(rho), 6),
        "spearman_p": f"{p_rho:.2e}",
        "kendall_tau": round(float(tau), 6),
        "flip_rate": round(float(flip_rate), 6),
    }

    out_path = OUT / f"result_{bench.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"  rho={rho:.6f}, tau={tau:.6f}, flip={flip_rate:.4f}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
