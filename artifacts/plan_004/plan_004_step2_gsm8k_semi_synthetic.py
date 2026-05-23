#!/usr/bin/env python3
"""
plan_004_step2 — GSM8K Semi-Synthetic DIF Injection

Replicates plan_011 methodology on GSM8K: inject known differential contamination
into real response data and test whether MH-DIF detects it.

Two matching variants:
  - standard: purified GSM8K total score (contaminated by injection)
  - external: average proportion correct on ARC+HellaSwag+TruthfulQA+Winogrande

Cohort: temporal, ref=2023, foc=2024
Injection: 20% of GSM8K items, p ∈ {0.0, 0.3, 0.5, 0.7, 1.0}, 5 seeds

Inputs:
  metabench_data/benchmark-data/gsm8k.csv
  model_metadata.csv
  metabench_data/benchmark-data/{arc,hellaswag,truthfulqa,winogrande}.csv

Outputs (in plan_004/):
  gsm8k_dose_response.csv, gsm8k_dose_response_summary.csv,
  plan_004_step2_report.md
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist, fisher_exact
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("artifacts")
BENCH_DIR = BASE / "metabench_data" / "benchmark-data"
OUT = BASE / "plan_004"
OUT.mkdir(exist_ok=True)

INJECTION_PS = [0.0, 0.3, 0.5, 0.7, 1.0]
N_SEEDS = 5
CONTAM_FRAC = 0.20

MH_VARIANTS = ["standard", "external"]


# ════════════════════════════════════════════════════════════════════════════
#  MH-DIF CORE
# ════════════════════════════════════════════════════════════════════════════
def _nan_row(j):
    return dict(col_idx=j, alpha_mh=np.nan, delta_mh=np.nan,
                se=np.nan, chi2=np.nan, p_value=np.nan, ets_class="A")


def mh_dif(X_ref, X_foc, n_strata=5,
           external_scores_ref=None, external_scores_foc=None):
    """Purified Mantel-Haenszel DIF for all columns."""
    J = X_ref.shape[1]
    use_external = external_scores_ref is not None

    if not use_external:
        tot_ref = X_ref.sum(axis=1).astype(np.float64)
        tot_foc = X_foc.sum(axis=1).astype(np.float64)

    rows = []
    for j in range(J):
        if use_external:
            s_ref = external_scores_ref.astype(np.float64)
            s_foc = external_scores_foc.astype(np.float64)
        else:
            s_ref = tot_ref - X_ref[:, j].astype(np.float64)
            s_foc = tot_foc - X_foc[:, j].astype(np.float64)

        s_all = np.concatenate([s_ref, s_foc])
        edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
        if len(edges) < 2:
            rows.append(_nan_row(j))
            continue
        n_bins = len(edges) - 1

        k_ref = np.digitize(s_ref, edges[1:-1])
        k_foc = np.digitize(s_foc, edges[1:-1])

        R = S = 0.0
        sum_diff = 0.0
        sum_var = 0.0
        pr = ps_qr = qs = 0.0
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

            Pk = (A + D) / Nk
            Qk = (B + C) / Nk
            pr += Pk * Rk
            ps_qr += Pk * Sk + Qk * Rk
            qs += Qk * Sk

        if n_ok == 0 or (R == 0 and S == 0):
            rows.append(_nan_row(j))
            continue

        if S == 0:
            alpha, delta, se = np.inf, -np.inf, np.nan
        elif R == 0:
            alpha, delta, se = 0.0, np.inf, np.nan
        else:
            alpha = R / S
            delta = -2.35 * np.log(alpha)
            v = pr / (2 * R * R) + ps_qr / (2 * R * S) + qs / (2 * S * S)
            se = 2.35 * np.sqrt(v) if v > 0 else np.nan

        if sum_var > 0:
            chi2_stat = max(abs(sum_diff) - 0.5, 0) ** 2 / sum_var
            pval = 1 - chi2_dist.cdf(chi2_stat, df=1)
        else:
            chi2_stat = pval = np.nan

        ad = abs(delta) if np.isfinite(delta) else 999.0
        if ad < 1.0:
            ets = "A"
        elif ad >= 1.5 and (not np.isnan(pval)) and pval < 0.05:
            ets = "C"
        else:
            ets = "B"

        rows.append(dict(col_idx=j, alpha_mh=alpha, delta_mh=delta,
                         se=se, chi2=chi2_stat, p_value=pval, ets_class=ets))

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
print("=== Loading GSM8K data ===")
t0 = time.time()
gsm = pd.read_csv(BENCH_DIR / "gsm8k.csv")
gsm["correct"] = gsm["correct"].map({"True": 1, "False": 0, True: 1, False: 0}).astype(np.int8)
gsm_pivot = gsm.pivot(index="source", columns="item", values="correct")
gsm_pivot = gsm_pivot.fillna(0).astype(np.int8)

MODEL_NAMES = np.array(gsm_pivot.index.tolist())
ITEM_IDS = np.array(gsm_pivot.columns.tolist())
X_full = gsm_pivot.values
N_MODELS, J_TOTAL = X_full.shape
print(f"  Response matrix: {N_MODELS} × {J_TOTAL}")
del gsm, gsm_pivot

NAME2ROW = {n: i for i, n in enumerate(MODEL_NAMES)}
N_CONTAMINATED = int(round(J_TOTAL * CONTAM_FRAC))
N_CLEAN = J_TOTAL - N_CONTAMINATED
print(f"  Contaminated items: {N_CONTAMINATED} ({CONTAM_FRAC*100:.0f}%), clean: {N_CLEAN}")

# Model metadata
model_meta = pd.read_csv(BASE / "model_metadata.csv")


# ════════════════════════════════════════════════════════════════════════════
#  EXTERNAL MATCHING SCORES
# ════════════════════════════════════════════════════════════════════════════
def compute_benchmark_scores(filepath, benchmark_name):
    """Compute per-model proportion correct from a benchmark CSV."""
    print(f"  Loading {benchmark_name}...")
    t0b = time.time()
    scores = {}
    counts = {}
    for chunk in pd.read_csv(filepath, chunksize=500_000):
        chunk["correct"] = chunk["correct"].map({"True": 1, "False": 0, True: 1, False: 0})
        grouped = chunk.groupby("source")["correct"].agg(["sum", "count"])
        for model, row in grouped.iterrows():
            if model in scores:
                scores[model] += row["sum"]
                counts[model] += row["count"]
            else:
                scores[model] = row["sum"]
                counts[model] = row["count"]
    result = {m: scores[m] / counts[m] for m in scores if counts[m] > 0}
    print(f"    {benchmark_name}: {len(result)} models, {time.time()-t0b:.1f}s")
    return result


print("\n=== Computing external matching scores ===")
ext_benchmarks = ["arc", "hellaswag", "truthfulqa", "winogrande"]
ext_scores_raw = {}
for bm in ext_benchmarks:
    ext_scores_raw[bm] = compute_benchmark_scores(BENCH_DIR / f"{bm}.csv", bm)

ext_combined = {}
for model in MODEL_NAMES:
    bm_scores = [ext_scores_raw[bm][model] for bm in ext_benchmarks if model in ext_scores_raw[bm]]
    if len(bm_scores) >= 2:
        ext_combined[model] = np.mean(bm_scores)
print(f"  Models with external scores (≥2 benchmarks): {len(ext_combined)}")
del ext_scores_raw


# ════════════════════════════════════════════════════════════════════════════
#  COHORT CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Cohort construction ===")


def rows_for(col, values):
    names = model_meta.loc[model_meta[col].isin(values), "model_name"]
    return np.array([NAME2ROW[n] for n in names if n in NAME2ROW])


ref_idx = rows_for("cohort_temporal", ["2023"])
foc_idx = rows_for("cohort_temporal", ["2024"])
print(f"  Temporal: ref=2023 (N={len(ref_idx)}), foc=2024 (N={len(foc_idx)})")

# External subset: models with external scores
ref_names_ext = [MODEL_NAMES[i] for i in ref_idx if MODEL_NAMES[i] in ext_combined]
foc_names_ext = [MODEL_NAMES[i] for i in foc_idx if MODEL_NAMES[i] in ext_combined]
ref_idx_ext = np.array([NAME2ROW[n] for n in ref_names_ext])
foc_idx_ext = np.array([NAME2ROW[n] for n in foc_names_ext])
print(f"  External subset: ref (N={len(ref_idx_ext)}), foc (N={len(foc_idx_ext)})")

# Focal injection subset: models with accuracy overlap with ref
gsm8k_acc = X_full.mean(axis=1)
ref_acc = gsm8k_acc[ref_idx]
foc_acc = gsm8k_acc[foc_idx]
overlap_min = max(ref_acc.min(), foc_acc.min())
overlap_max = min(ref_acc.max(), foc_acc.max())

if overlap_min <= overlap_max:
    foc_inject_mask = np.array([(gsm8k_acc[i] >= overlap_min) and (gsm8k_acc[i] <= overlap_max)
                                 for i in foc_idx])
    foc_inject_idx = foc_idx[foc_inject_mask]
else:
    foc_inject_idx = foc_idx.copy()

# Same for external subset
if overlap_min <= overlap_max:
    foc_inject_ext_mask = np.array([(gsm8k_acc[i] >= overlap_min) and (gsm8k_acc[i] <= overlap_max)
                                     for i in foc_idx_ext])
    foc_inject_idx_ext = foc_idx_ext[foc_inject_ext_mask]
else:
    foc_inject_idx_ext = foc_idx_ext.copy()

print(f"  Accuracy overlap: [{overlap_min:.4f}, {overlap_max:.4f}]")
print(f"  Focal inject (standard): {len(foc_inject_idx)} models")
print(f"  Focal inject (external): {len(foc_inject_idx_ext)} models")

# Precompute external scores arrays
ext_ref_scores = np.array([ext_combined[MODEL_NAMES[i]] for i in ref_idx_ext])
ext_foc_scores = np.array([ext_combined[MODEL_NAMES[i]] for i in foc_idx_ext])


# ════════════════════════════════════════════════════════════════════════════
#  INJECTION + MH-DIF
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Injection + MH-DIF ===")

total_runs = len(INJECTION_PS) * N_SEEDS * len(MH_VARIANTS)
run_count = 0
t_start = time.time()

injection_rows = []

for p in INJECTION_PS:
    for seed in range(1, N_SEEDS + 1):
        rng = np.random.RandomState(seed)

        contam_local_idx = rng.choice(J_TOTAL, N_CONTAMINATED, replace=False)
        contam_set = set(contam_local_idx.tolist())
        clean_local_idx = np.array([i for i in range(J_TOTAL) if i not in contam_set])

        X_inj = X_full.copy()
        n_flipped = 0

        if p > 0:
            for local_j in contam_local_idx:
                for model_row in foc_inject_idx:
                    if X_inj[model_row, local_j] == 0:
                        if rng.random() < p:
                            X_inj[model_row, local_j] = 1
                            n_flipped += 1

        orig_density = X_full[np.ix_(foc_inject_idx, contam_local_idx)].mean()
        new_density = X_inj[np.ix_(foc_inject_idx, contam_local_idx)].mean()

        for variant in MH_VARIANTS:
            run_count += 1

            if variant == "standard":
                X_ref_v = X_inj[ref_idx]
                X_foc_v = X_inj[foc_idx]
                dif_results = mh_dif(X_ref_v, X_foc_v, n_strata=5)
            else:
                X_ref_v = X_inj[ref_idx_ext]
                X_foc_v = X_inj[foc_idx_ext]
                dif_results = mh_dif(X_ref_v, X_foc_v, n_strata=5,
                                     external_scores_ref=ext_ref_scores,
                                     external_scores_foc=ext_foc_scores)

            contam_mask = dif_results["col_idx"].isin(contam_set)
            contam_results = dif_results[contam_mask]
            clean_results = dif_results[~contam_mask]

            n_c_contam = (contam_results["ets_class"] == "C").sum()
            n_c_clean = (clean_results["ets_class"] == "C").sum()
            tpr = n_c_contam / N_CONTAMINATED
            fpr = n_c_clean / N_CLEAN

            table = [[n_c_contam, N_CONTAMINATED - n_c_contam],
                     [n_c_clean, N_CLEAN - n_c_clean]]
            fisher_or, fisher_p = fisher_exact(table)

            injection_rows.append(dict(
                p=p, seed=seed, matching_type=variant,
                n_items=J_TOTAL, n_contaminated=N_CONTAMINATED, n_clean=N_CLEAN,
                n_ref=len(ref_idx) if variant == "standard" else len(ref_idx_ext),
                n_foc=len(foc_idx) if variant == "standard" else len(foc_idx_ext),
                n_foc_inject=len(foc_inject_idx) if variant == "standard" else len(foc_inject_idx_ext),
                tpr=tpr, fpr=fpr,
                n_c_contam=n_c_contam, n_c_clean=n_c_clean,
                fisher_or=fisher_or, fisher_p=fisher_p,
                n_responses_flipped=n_flipped,
                orig_density=orig_density, new_density=new_density,
            ))

            elapsed = time.time() - t_start
            print(f"  [{run_count}/{total_runs}] {variant:8s} p={p} seed={seed}: "
                  f"TPR={tpr:.3f} FPR={fpr:.3f} flipped={n_flipped} ({elapsed:.0f}s)")


# ════════════════════════════════════════════════════════════════════════════
#  AGGREGATE & SAVE
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Aggregating results ===")

inj_df = pd.DataFrame(injection_rows)
inj_df.to_csv(OUT / "gsm8k_dose_response.csv", index=False)
print(f"  Saved gsm8k_dose_response.csv ({len(inj_df)} rows)")

summary_rows = []
for (p_val, variant), grp in inj_df.groupby(["p", "matching_type"]):
    base_grp = inj_df[(inj_df["p"] == 0.0) & (inj_df["matching_type"] == variant)]
    base_tpr = base_grp["tpr"].mean()
    summary_rows.append(dict(
        p=p_val, matching_type=variant,
        tpr_mean=grp["tpr"].mean(), tpr_std=grp["tpr"].std(),
        fpr_mean=grp["fpr"].mean(), fpr_std=grp["fpr"].std(),
        delta_tpr=grp["tpr"].mean() - base_tpr,
        fisher_or_mean=grp["fisher_or"].mean(),
        fisher_p_median=grp["fisher_p"].median(),
        n_flipped_mean=grp["n_responses_flipped"].mean(),
    ))
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "gsm8k_dose_response_summary.csv", index=False)
print(f"  Saved gsm8k_dose_response_summary.csv ({len(summary_df)} rows)")


# ════════════════════════════════════════════════════════════════════════════
#  REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Generating report ===")

L = []
def w(s=""):
    L.append(s)

w("# Plan 004 Step 2: GSM8K Semi-Synthetic DIF Injection")
w()
w("## 1. Objective")
w()
w("Inject known differential contamination into GSM8K response data and test whether")
w("MH-DIF detects it. Validates that the DIF methodology works on GSM8K (single-domain,")
w("math benchmark) and enables comparison with MMLU results from plan_011.")
w()

w("## 2. Setup")
w()
w(f"- **Benchmark**: GSM8K ({J_TOTAL} items)")
w(f"- **Contaminated items**: {N_CONTAMINATED} ({CONTAM_FRAC*100:.0f}% of {J_TOTAL})")
w(f"- **Injection probabilities**: {INJECTION_PS}")
w(f"- **Seeds per p-level**: {N_SEEDS}")
w(f"- **Cohort**: temporal, ref=2023 (N={len(ref_idx)}), foc=2024 (N={len(foc_idx)})")
w(f"- **External cohort**: ref (N={len(ref_idx_ext)}), foc (N={len(foc_idx_ext)})")
w(f"- **Focal inject (standard)**: {len(foc_inject_idx)} models in accuracy overlap [{overlap_min:.4f}, {overlap_max:.4f}]")
w(f"- **Focal inject (external)**: {len(foc_inject_idx_ext)} models")
w()

w("### Matching variants")
w()
w("| Variant | Matching variable | Properties |")
w("|---------|-------------------|------------|")
w("| standard | purified GSM8K total score | Contaminated by injection (20% items affected) |")
w("| external | avg proportion correct on ARC+HellaSwag+TruthfulQA+Winogrande | Immune to GSM8K injection |")
w()

w("## 3. Dose-Response Results")
w()
for variant in MH_VARIANTS:
    w(f"### {variant.capitalize()} matching")
    w()
    w("| p | TPR (mean±std) | FPR (mean±std) | ΔTPR | Fisher OR | Fisher p |")
    w("|---|----------------|----------------|------|-----------|----------|")
    sub = summary_df[summary_df["matching_type"] == variant].sort_values("p")
    for _, r in sub.iterrows():
        w(f"| {r.p} | {r.tpr_mean:.3f}±{r.tpr_std:.3f} | "
          f"{r.fpr_mean:.3f}±{r.fpr_std:.3f} | {r.delta_tpr:+.3f} | "
          f"{r.fisher_or_mean:.2f} | {r.fisher_p_median:.4f} |")
    w()

w("## 4. Comparison with MMLU (plan_011)")
w()
w("| Metric | MMLU plan_011 (external) | GSM8K (external) |")
w("|--------|--------------------------|------------------|")

ext_sub = summary_df[summary_df["matching_type"] == "external"]
ext_p0 = ext_sub[ext_sub["p"] == 0.0]
ext_p05 = ext_sub[ext_sub["p"] == 0.5]
ext_p1 = ext_sub[ext_sub["p"] == 1.0]

mmlu_baseline_fpr = 0.314
mmlu_dtpr_05 = 0.508
mmlu_dtpr_10 = 0.674

gsm_fpr0 = ext_p0.iloc[0]["fpr_mean"] if len(ext_p0) > 0 else np.nan
gsm_dtpr05 = ext_p05.iloc[0]["delta_tpr"] if len(ext_p05) > 0 else np.nan
gsm_dtpr10 = ext_p1.iloc[0]["delta_tpr"] if len(ext_p1) > 0 else np.nan

w(f"| Baseline FPR (p=0) | {mmlu_baseline_fpr:.3f} | {gsm_fpr0:.3f} |")
w(f"| ΔTPR at p=0.5 | +{mmlu_dtpr_05:.3f} | {gsm_dtpr05:+.3f} |")
w(f"| ΔTPR at p=1.0 | +{mmlu_dtpr_10:.3f} | {gsm_dtpr10:+.3f} |")
w(f"| Items | 500 (subset) | {J_TOTAL} (all) |")
w(f"| Contaminated | 100 (20%) | {N_CONTAMINATED} (~20%) |")
w()

w("## 5. Standard vs External Matching")
w()
w("| p | Standard TPR | External TPR | Standard FPR | External FPR |")
w("|---|-------------|-------------|-------------|-------------|")
for p_val in INJECTION_PS:
    std_r = summary_df[(summary_df["p"] == p_val) & (summary_df["matching_type"] == "standard")]
    ext_r = summary_df[(summary_df["p"] == p_val) & (summary_df["matching_type"] == "external")]
    if len(std_r) > 0 and len(ext_r) > 0:
        w(f"| {p_val} | {std_r.iloc[0].tpr_mean:.3f} | {ext_r.iloc[0].tpr_mean:.3f} | "
          f"{std_r.iloc[0].fpr_mean:.3f} | {ext_r.iloc[0].fpr_mean:.3f} |")
w()

w("## 6. Density Changes")
w()
w("| p | Orig density | Post density | Responses flipped |")
w("|---|-------------|-------------|-------------------|")
for p_val in INJECTION_PS:
    if p_val == 0:
        continue
    sub = inj_df[(inj_df["p"] == p_val) & (inj_df["matching_type"] == "standard")]
    if len(sub) > 0:
        w(f"| {p_val} | {sub.orig_density.mean():.4f} | {sub.new_density.mean():.4f} | "
          f"{sub.n_responses_flipped.mean():.0f} |")
w()

w("## 7. Verdict")
w()

for variant in MH_VARIANTS:
    sub = summary_df[summary_df["matching_type"] == variant]
    p0 = sub[sub["p"] == 0.0]
    p1 = sub[sub["p"] == 1.0]
    if len(p0) == 0 or len(p1) == 0:
        continue
    tpr0 = p0.iloc[0]["tpr_mean"]
    fpr0 = p0.iloc[0]["fpr_mean"]
    tpr1 = p1.iloc[0]["tpr_mean"]
    fpr1 = p1.iloc[0]["fpr_mean"]
    d_tpr = tpr1 - tpr0

    w(f"### {variant.capitalize()} matching")
    w()
    w(f"- Baseline (p=0): TPR={tpr0:.3f}, FPR={fpr0:.3f}")
    w(f"- Full injection (p=1): TPR={tpr1:.3f}, FPR={fpr1:.3f}")
    w(f"- ΔTPR = {d_tpr:+.3f}")
    w()

    if d_tpr > 0.3 and tpr1 > 0.5:
        w(f"**SUCCESS**: injection raises TPR by {d_tpr:.3f} with clear dose-response.")
    elif d_tpr > 0.1:
        w(f"**PARTIAL**: detectable signal (ΔTPR={d_tpr:.3f}) but modest.")
    else:
        w(f"**FAIL**: ΔTPR={d_tpr:.3f} too small to indicate reliable detection.")
    w()

w("## 8. Implications")
w()
w("### Cross-benchmark generalization")
w()
w("If GSM8K semi-synthetic results are comparable to MMLU plan_011, this confirms")
w("MH-DIF methodology generalizes beyond MMLU to single-domain math benchmarks.")
w()
w("### Matching variable contamination")
w()
w("Standard matching uses GSM8K total score, which is corrupted when 20% of items")
w("are injected. This should inflate FPR compared to external matching.")
w("External matching (other benchmarks) is immune to GSM8K injection and should")
w("provide cleaner separation between contaminated and clean items.")
w()

report_text = "\n".join(L) + "\n"
(OUT / "plan_004_step2_report.md").write_text(report_text)
print("  Report written.")

# Console summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for variant in MH_VARIANTS:
    print(f"\n--- {variant} matching ---")
    sub = summary_df[summary_df["matching_type"] == variant].sort_values("p")
    for _, r in sub.iterrows():
        print(f"  p={r.p}: TPR={r.tpr_mean:.3f}±{r.tpr_std:.3f}  "
              f"FPR={r.fpr_mean:.3f}±{r.fpr_std:.3f}  ΔTPR={r.delta_tpr:+.3f}")

total_elapsed = time.time() - t_start
print(f"\nTotal time: {total_elapsed:.0f}s")
print(f"All outputs saved to {OUT}")
