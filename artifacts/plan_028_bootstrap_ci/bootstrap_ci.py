#!/usr/bin/env python3
"""
bootstrap_ci.py — Bootstrap 95% CI for MH-DIF C% across all 6 benchmarks.

Resamples *models* (not items) with replacement from each cohort,
re-runs MH-DIF on each bootstrap sample, records C%.

Outputs:
  bootstrap_ci_results.csv       — per-benchmark C%, CI_lower, CI_upper
  bootstrap_distributions.csv    — per-bootstrap C% for distribution plots
  bootstrap_ci_report.md         — human-readable report
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from pathlib import Path
import time
import warnings
import gc
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("artifacts")
DATA = BASE / "metabench_data" / "benchmark-data"
OUT = BASE / "plan_028_bootstrap_ci"
OUT.mkdir(exist_ok=True)

N_BOOTSTRAP = 1000
SEED = 42


# ════════════════════════════════════════════════════════════════════════════
#  MH-DIF CORE (from plan_028, with inf-delta fix)
# ════════════════════════════════════════════════════════════════════════════
def _nan_row(j):
    return dict(col_idx=j, ets_class="A")


def mh_dif_c_pct(X_ref, X_foc, n_strata=5):
    """
    Purified MH-DIF → returns only C%.
    Stripped version that skips storing full per-item stats for speed.
    """
    tot_ref = X_ref.sum(axis=1).astype(np.float64)
    tot_foc = X_foc.sum(axis=1).astype(np.float64)
    J = X_ref.shape[1]
    n_c = 0

    for j in range(J):
        s_ref = tot_ref - X_ref[:, j].astype(np.float64)
        s_foc = tot_foc - X_foc[:, j].astype(np.float64)

        s_all = np.concatenate([s_ref, s_foc])
        edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
        if len(edges) < 2:
            continue
        n_bins = len(edges) - 1
        k_ref = np.digitize(s_ref, edges[1:-1])
        k_foc = np.digitize(s_foc, edges[1:-1])

        R = S = 0.0
        sum_diff = sum_var = 0.0
        n_ok = 0

        for k in range(n_bins):
            mr = (k_ref == k)
            mf = (k_foc == k)
            n1 = int(mr.sum())
            n0 = int(mf.sum())
            if n1 == 0 or n0 == 0:
                continue

            A = float(X_ref[mr, j].sum())
            B = float(n1 - A)
            C = float(X_foc[mf, j].sum())
            D = float(n0 - C)
            Nk = float(n1 + n0)
            m1 = A + C
            m0 = B + D

            if m1 == 0 or m0 == 0 or Nk <= 1:
                continue
            n_ok += 1

            Rk = A * D / Nk
            Sk = B * C / Nk
            R += Rk
            S += Sk

            EA = n1 * m1 / Nk
            sum_diff += A - EA
            sum_var += n1 * n0 * m1 * m0 / (Nk * Nk * (Nk - 1))

        if n_ok == 0 or (R == 0 and S == 0):
            continue

        if S == 0:
            delta = -np.inf
        elif R == 0:
            delta = np.inf
        else:
            alpha = R / S
            delta = -2.35 * np.log(alpha)

        if sum_var > 0:
            chi2_stat = max(abs(sum_diff) - 0.5, 0) ** 2 / sum_var
            pval = 1 - chi2_dist.cdf(chi2_stat, df=1)
        else:
            pval = np.nan

        if not np.isfinite(delta):
            ets = "A"
        elif abs(delta) >= 1.5 and pval < 0.05:
            ets = "C"
        elif abs(delta) >= 1.0:
            ets = "B"
        else:
            ets = "A"

        if ets == "C":
            n_c += 1

    return n_c / J * 100


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
def load_response_matrix(benchmark_name):
    """Load CSV → (response_matrix, model_names, item_ids)."""
    path = DATA / f"{benchmark_name}.csv"
    print(f"  Loading {path.name}...")

    file_size_mb = path.stat().st_size / 1024 / 1024
    print(f"  File size: {file_size_mb:.0f} MB")

    if file_size_mb > 500:
        print("  Using chunked loading (large file)...")
        sources_set, items_set = set(), set()
        for chunk in pd.read_csv(path, chunksize=2_000_000, usecols=["source", "item"]):
            sources_set.update(chunk["source"].unique())
            items_set.update(chunk["item"].unique())

        sources = sorted(sources_set)
        items = sorted(items_set)
        src2idx = {s: i for i, s in enumerate(sources)}
        itm2idx = {t: i for i, t in enumerate(items)}
        print(f"  Discovered {len(sources)} sources × {len(items)} items")

        X = np.full((len(sources), len(items)), -1, dtype=np.int8)
        for chunk in pd.read_csv(path, chunksize=2_000_000):
            if chunk["correct"].dtype == object:
                vals = chunk["correct"].map({"True": 1, "False": 0, True: 1, False: 0}).values
            else:
                vals = chunk["correct"].astype(np.int8).values
            si = chunk["source"].map(src2idx).values
            ii = chunk["item"].map(itm2idx).values
            X[si, ii] = vals
        del chunk

        complete_mask = np.all(X >= 0, axis=1)
        X = X[complete_mask]
        model_names = np.array(sources)[complete_mask]
        item_ids = np.array(items)
    else:
        df = pd.read_csv(path)
        if df["correct"].dtype == object:
            df["correct"] = df["correct"].map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)
        else:
            df["correct"] = df["correct"].astype(int)

        pivot = df.pivot(index="source", columns="item", values="correct")
        pivot = pivot.dropna(axis=0)
        model_names = pivot.index.values
        item_ids = pivot.columns.values
        X = pivot.values.astype(np.int8)
        del df, pivot

    print(f"  Matrix: {X.shape[0]} models × {X.shape[1]} items")
    return X, model_names, item_ids


def load_mmlu_from_npz():
    """Load pre-built MMLU sparse matrix (matches plan_001 exactly)."""
    from scipy.sparse import load_npz
    print("  Loading MMLU from response_matrix.npz...")
    t0 = time.time()

    mat = load_npz(BASE / "response_matrix.npz")
    X = mat.toarray().astype(np.int8)
    del mat

    idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
    model_names = idx["model_names"]
    item_ids = idx["item_ids"]

    print(f"  MMLU matrix: {X.shape[0]} models × {X.shape[1]} items ({time.time()-t0:.0f}s)")
    return X, model_names, item_ids


# ════════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════
def run_bootstrap(X, ref_idx, foc_idx, n_bootstrap, rng, bench_name):
    """Bootstrap resample models, compute C% each time."""
    n_ref = len(ref_idx)
    n_foc = len(foc_idx)

    # Original C%
    print(f"  Computing original C%...")
    t0 = time.time()
    original_c = mh_dif_c_pct(X[ref_idx], X[foc_idx])
    print(f"  Original C% = {original_c:.2f}% ({time.time()-t0:.1f}s)")

    per_iter_est = time.time() - t0
    total_est = per_iter_est * n_bootstrap
    print(f"  Estimated total: {total_est/60:.0f} min for {n_bootstrap} bootstraps")

    boot_cs = np.zeros(n_bootstrap)
    t_start = time.time()

    for b in range(n_bootstrap):
        # Resample models with replacement within each group
        boot_ref = ref_idx[rng.randint(0, n_ref, size=n_ref)]
        boot_foc = foc_idx[rng.randint(0, n_foc, size=n_foc)]

        boot_cs[b] = mh_dif_c_pct(X[boot_ref], X[boot_foc])

        if (b + 1) % 100 == 0 or b == 0:
            elapsed = time.time() - t_start
            rate = (b + 1) / elapsed
            eta = (n_bootstrap - b - 1) / rate / 60
            print(f"    [{bench_name}] bootstrap {b+1}/{n_bootstrap}  "
                  f"C%={boot_cs[b]:.2f}%  mean={boot_cs[:b+1].mean():.2f}%  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}min)")

    ci_lower = np.percentile(boot_cs, 2.5)
    ci_upper = np.percentile(boot_cs, 97.5)
    boot_mean = boot_cs.mean()
    boot_std = boot_cs.std()

    print(f"  {bench_name}: C% = {original_c:.2f}% [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print(f"  Bootstrap mean = {boot_mean:.2f}% (bias = {boot_mean - original_c:+.2f}pp)")

    return dict(
        benchmark=bench_name,
        original_c_pct=original_c,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        boot_mean=boot_mean,
        boot_std=boot_std,
        n_bootstrap=n_bootstrap,
        n_ref=n_ref,
        n_foc=n_foc,
        n_items=X.shape[1],
    ), boot_cs


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--benchmarks", nargs="*", default=None,
                        help="Subset of benchmarks to run (default: all 6)")
    args = parser.parse_args()

    n_boot = args.n_bootstrap
    print(f"Bootstrap CI analysis — N={n_boot}, seed={SEED}")
    print(f"Output: {OUT}\n")

    model_meta = pd.read_csv(BASE / "model_metadata.csv")
    cohort_map = dict(zip(model_meta["model_name"], model_meta["cohort_temporal"]))

    ALL_BENCHMARKS = ["mmlu", "arc", "hellaswag", "winogrande", "truthfulqa", "gsm8k"]
    benchmarks = args.benchmarks if args.benchmarks else ALL_BENCHMARKS

    # Per-benchmark seeds for reproducibility regardless of batch ordering
    BENCH_SEEDS = {b: SEED + i for i, b in enumerate(ALL_BENCHMARKS)}

    all_summaries = []
    all_distributions = []

    for bench in benchmarks:
        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {bench.upper()}")
        print(f"{'='*60}")

        # Load data
        if bench == "mmlu":
            X, model_names, item_ids = load_mmlu_from_npz()
        else:
            X, model_names, item_ids = load_response_matrix(bench)

        name2row = {n: i for i, n in enumerate(model_names)}

        ref_idx = np.array([name2row[n] for n in model_names
                            if cohort_map.get(n) == "2023"])
        foc_idx = np.array([name2row[n] for n in model_names
                            if cohort_map.get(n) == "2024"])
        print(f"  Cohort split: ref(2023)={len(ref_idx)}, foc(2024)={len(foc_idx)}")

        if len(ref_idx) < 50 or len(foc_idx) < 50:
            print(f"  SKIP: insufficient models (<50 per cohort)")
            continue

        rng = np.random.RandomState(BENCH_SEEDS[bench])
        summary, boot_cs = run_bootstrap(X, ref_idx, foc_idx, n_boot, rng, bench)
        all_summaries.append(summary)

        boot_dist = pd.DataFrame({"bootstrap_id": range(len(boot_cs)), "c_pct": boot_cs})
        boot_dist["benchmark"] = bench

        # Save per-benchmark files (safe for parallel runs)
        pd.DataFrame([summary]).to_csv(OUT / f"bootstrap_{bench}_summary.csv", index=False)
        boot_dist.to_csv(OUT / f"bootstrap_{bench}_dist.csv", index=False)
        print(f"  Saved: bootstrap_{bench}_summary.csv, bootstrap_{bench}_dist.csv")

        for b, c in enumerate(boot_cs):
            all_distributions.append(dict(benchmark=bench, bootstrap_id=b, c_pct=c))

        del X
        gc.collect()

    # ── Merge all per-benchmark files into combined outputs ──
    _merge_results()

    # ── Final summary ──
    results_df = pd.DataFrame(all_summaries)
    print(f"\n{'='*60}")
    print("BOOTSTRAP CI SUMMARY")
    print(f"{'='*60}")
    for _, r in results_df.iterrows():
        print(f"  {r.benchmark.upper():12s}  C% = {r.original_c_pct:5.1f}%  "
              f"[{r.ci_lower:5.1f}%, {r.ci_upper:5.1f}%]  "
              f"(boot mean={r.boot_mean:.1f}%, std={r.boot_std:.1f}%)")


def _merge_results():
    """Merge all per-benchmark files into combined outputs."""
    import glob as globmod
    summary_files = sorted(globmod.glob(str(OUT / "bootstrap_*_summary.csv")))
    dist_files = sorted(globmod.glob(str(OUT / "bootstrap_*_dist.csv")))

    if summary_files:
        results_df = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
        results_df.to_csv(OUT / "bootstrap_ci_results.csv", index=False)
        print(f"\nMerged {len(summary_files)} benchmarks → bootstrap_ci_results.csv")

        _write_report(results_df)
        print("Saved: bootstrap_ci_report.md")

    if dist_files:
        dist_df = pd.concat([pd.read_csv(f) for f in dist_files], ignore_index=True)
        dist_df.to_csv(OUT / "bootstrap_distributions.csv", index=False)
        print(f"Merged {len(dist_files)} benchmarks → bootstrap_distributions.csv")


def _write_report(results_df):
    L = []
    def w(s=""):
        L.append(s)

    w("# Bootstrap 95% CI for MH-DIF C%")
    w()
    w("## Method")
    w()
    w(f"- Bootstrap resampling of **models** (not items), {int(results_df.iloc[0].n_bootstrap)} iterations")
    w("- Within each bootstrap: resample N_ref models with replacement from 2023 cohort,")
    w("  resample N_foc models with replacement from 2024 cohort")
    w("- Re-run purified MH-DIF (5 strata) on resampled groups")
    w("- 95% CI = [2.5th, 97.5th] percentile of bootstrap C% distribution")
    w(f"- Random seed = {SEED}")
    w()

    w("## Results")
    w()
    w("| Benchmark | Items | N_ref | N_foc | C% | 95% CI | Boot Mean | Boot SD |")
    w("|-----------|-------|-------|-------|----|--------|-----------|---------|")
    for _, r in results_df.iterrows():
        w(f"| {r.benchmark.upper()} | {int(r.n_items)} | {int(r.n_ref)} | {int(r.n_foc)} | "
          f"{r.original_c_pct:.1f}% | [{r.ci_lower:.1f}%, {r.ci_upper:.1f}%] | "
          f"{r.boot_mean:.1f}% | {r.boot_std:.1f}% |")
    w()

    w("## Validation")
    w()
    for _, r in results_df.iterrows():
        bias = r.boot_mean - r.original_c_pct
        ci_width = r.ci_upper - r.ci_lower
        w(f"- **{r.benchmark.upper()}**: bias = {bias:+.2f}pp, CI width = {ci_width:.1f}pp")
    w()

    (OUT / "bootstrap_ci_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
