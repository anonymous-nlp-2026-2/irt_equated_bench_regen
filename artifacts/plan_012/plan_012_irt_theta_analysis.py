#!/usr/bin/env python3
"""
plan_012_irt_theta_analysis.py — IRT θ-Conditioning MH-DIF Analysis

Uses IRT-based θ estimates (logit of proportion correct) instead of raw total
scores for MH-DIF stratification. Tests whether finer ability matching reduces
the high baseline FPR from plan_011 (temporal ~31%, ability ~49%) while
preserving detection power for differential contamination.

Variants tested:
  - theta_eq_K   : equal-width bins on θ (K=10,15,20)
  - theta_qt_K   : quantile bins on θ (K=10,15,20)
  - external_5   : plan_011 baseline (5-quantile bins on full-matrix total score)

Outputs (all in plan_012/):
  theta_estimates.csv, dose_response_theta.csv, enrichment_theta.csv,
  plan_012_report.md
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import chi2 as chi2_dist, fisher_exact
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("artifacts")
OUT = BASE / "plan_012"
OUT.mkdir(exist_ok=True)

INJECTION_PS = [0.0, 0.3, 0.5, 0.7, 1.0]
N_SEEDS = 5
N_BASELINE = 500
N_CONTAMINATED = 100
STRATA_COUNTS = [10, 15, 20]


# ════════════════════════════════════════════════════════════════════════════
#  MH-DIF CORE
# ════════════════════════════════════════════════════════════════════════════
def _nan_row(j):
    return dict(col_idx=j, alpha_mh=np.nan, delta_mh=np.nan,
                se=np.nan, chi2=np.nan, p_value=np.nan, ets_class="A")


def _compute_mh_item(X_ref_j, X_foc_j, k_ref, k_foc, n_bins):
    """Compute MH statistics for a single item given pre-computed bin assignments."""
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

        A = float(X_ref_j[mr].sum())
        B = float(n1 - A)
        C = float(X_foc_j[mf].sum())
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
        return None

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

    ad = abs(delta) if np.isfinite(delta) else np.inf
    if ad < 1.0:
        ets = "A"
    elif ad >= 1.5 and (not np.isnan(pval)) and pval < 0.05:
        ets = "C"
    else:
        ets = "B"

    return dict(alpha_mh=alpha, delta_mh=delta, se=se,
                chi2=chi2_stat, p_value=pval, ets_class=ets)


def mh_dif_with_bins(X_ref, X_foc, scores_ref, scores_foc, n_strata, bin_mode="equal"):
    """
    MH-DIF with pre-computed ability scores.

    bin_mode:
      "equal"   — equal-width bins on score scale
      "quantile" — quantile bins (equal-count)
    """
    J = X_ref.shape[1]
    s_all = np.concatenate([scores_ref, scores_foc])

    if bin_mode == "equal":
        edges = np.linspace(s_all.min() - 0.001, s_all.max() + 0.001, n_strata + 1)
        n_bins = n_strata
    else:
        edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
        n_bins = len(edges) - 1

    if n_bins < 2:
        return pd.DataFrame([_nan_row(j) for j in range(J)])

    k_ref = np.digitize(scores_ref, edges[1:-1])
    k_foc = np.digitize(scores_foc, edges[1:-1])

    rows = []
    for j in range(J):
        result = _compute_mh_item(X_ref[:, j], X_foc[:, j], k_ref, k_foc, n_bins)
        if result is None:
            rows.append(_nan_row(j))
        else:
            result["col_idx"] = j
            rows.append(result)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
print("=== Loading data ===")
mat = load_npz(BASE / "response_matrix.npz")
idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
MODEL_NAMES = idx["model_names"]
ITEM_IDS = idx["item_ids"]
ITEM2COL = {iid: i for i, iid in enumerate(ITEM_IDS)}

model_meta = pd.read_csv(BASE / "model_metadata.csv")
li_labels = pd.read_csv(BASE / "mmlu_li2024_proxy_labels.csv")
ability_dif = pd.read_csv(BASE / "plan_001" / "dif_results_ability.csv")

NAME2ROW = {n: i for i, n in enumerate(MODEL_NAMES)}
N_MODELS, J_FULL = mat.shape
print(f"  Matrix: {N_MODELS} x {J_FULL}")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1: IRT θ ESTIMATION
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 1: IRT theta estimation ===")
totals_all = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
p_correct = (totals_all + 0.5) / (J_FULL + 1.0)
theta_all = np.log(p_correct / (1.0 - p_correct))

print(f"  θ range: [{theta_all.min():.3f}, {theta_all.max():.3f}]")
print(f"  θ mean: {theta_all.mean():.3f}, std: {theta_all.std():.3f}")

corr = np.corrcoef(theta_all, totals_all)[0, 1]
print(f"  Correlation(θ, total_score): {corr:.4f}")

theta_df = pd.DataFrame({
    "model_name": MODEL_NAMES,
    "theta": theta_all,
    "total_score": totals_all,
    "method": "logit_laplace"
})
theta_df.to_csv(OUT / "theta_estimates.csv", index=False)
print(f"  Saved theta_estimates.csv ({len(theta_df)} rows)")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2: SELECT BASELINE ITEMS (same as plan_011)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 2: Selecting baseline items ===")
merged = ability_dif.merge(li_labels[["item_id", "contamination_type"]], on="item_id", how="inner")
baseline_pool = merged[(merged["contamination_type"] == "clean") & (merged["ets_class"] == "A")]
print(f"  Baseline pool (clean + ETS A): {len(baseline_pool)}")

baseline_items = baseline_pool.sample(N_BASELINE, random_state=42)
baseline_item_ids = baseline_items["item_id"].values
baseline_col_indices = np.array([ITEM2COL[iid] for iid in baseline_item_ids])
print(f"  Selected {N_BASELINE} baseline items")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3: DEFINE COHORTS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 3: Defining cohorts ===")


def get_row_indices(col, values):
    names = model_meta.loc[model_meta[col].isin(values), "model_name"]
    return np.array([NAME2ROW[n] for n in names if n in NAME2ROW])


cohort_defs = {}

# (a) Ability cohort
ref_ability_idx = get_row_indices("cohort_accuracy_quintile", [1, 2])
foc_ability_idx = get_row_indices("cohort_accuracy_quintile", [4, 5])

ref_acc = model_meta.loc[model_meta["cohort_accuracy_quintile"].isin([1, 2]), "accuracy"]
foc_acc = model_meta.loc[model_meta["cohort_accuracy_quintile"].isin([4, 5]), "accuracy"]
ab_overlap_min = max(ref_acc.min(), foc_acc.min())
ab_overlap_max = min(ref_acc.max(), foc_acc.max())

if ab_overlap_min <= ab_overlap_max:
    foc_ab_inject_mask = (
        model_meta["cohort_accuracy_quintile"].isin([4, 5]) &
        (model_meta["accuracy"] >= ab_overlap_min) &
        (model_meta["accuracy"] <= ab_overlap_max)
    )
    foc_ab_inject_idx = np.array([NAME2ROW[n] for n in model_meta.loc[foc_ab_inject_mask, "model_name"] if n in NAME2ROW])
else:
    foc_ab_inject_idx = foc_ability_idx

print(f"  Ability: ref={len(ref_ability_idx)} (Q1-2), foc={len(foc_ability_idx)} (Q4-5)")

cohort_defs["ability"] = dict(
    ref_idx=ref_ability_idx, foc_idx=foc_ability_idx,
    foc_inject_idx=foc_ab_inject_idx
)

# (b) Temporal cohort
ref_temporal_idx = get_row_indices("cohort_temporal", ["2023"])
foc_temporal_idx = get_row_indices("cohort_temporal", ["2024"])

ref_t_acc = model_meta.loc[model_meta["cohort_temporal"] == "2023", "accuracy"]
foc_t_acc = model_meta.loc[model_meta["cohort_temporal"] == "2024", "accuracy"]
t_overlap_min = max(ref_t_acc.min(), foc_t_acc.min())
t_overlap_max = min(ref_t_acc.max(), foc_t_acc.max())

foc_t_inject_mask = (
    (model_meta["cohort_temporal"] == "2024") &
    (model_meta["accuracy"] >= t_overlap_min) &
    (model_meta["accuracy"] <= t_overlap_max)
)
foc_t_inject_idx = np.array([NAME2ROW[n] for n in model_meta.loc[foc_t_inject_mask, "model_name"] if n in NAME2ROW])

print(f"  Temporal: ref={len(ref_temporal_idx)} (2023), foc={len(foc_temporal_idx)} (2024)")
print(f"    Inject: {len(foc_t_inject_idx)} focal models in accuracy overlap")

cohort_defs["temporal"] = dict(
    ref_idx=ref_temporal_idx, foc_idx=foc_temporal_idx,
    foc_inject_idx=foc_t_inject_idx
)


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4: SEMI-SYNTHETIC INJECTION + θ-CONDITIONED MH-DIF
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 4: Semi-Synthetic Injection + MH-DIF ===")

X_base = mat[:, baseline_col_indices].toarray().astype(np.int8)
full_totals = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
print(f"  Baseline submatrix: {X_base.shape}, density: {X_base.mean():.4f}")

# Build variant list: theta_eq_K, theta_qt_K, external_5
MH_VARIANTS = []
for ns in STRATA_COUNTS:
    MH_VARIANTS.append(f"theta_eq_{ns}")
    MH_VARIANTS.append(f"theta_qt_{ns}")
MH_VARIANTS.append("external_5")

injection_rows = []

total_runs = len(INJECTION_PS) * N_SEEDS * len(cohort_defs) * len(MH_VARIANTS)
run_count = 0
t_start = time.time()

for cohort_name, cdef in cohort_defs.items():
    ref_idx = cdef["ref_idx"]
    foc_idx = cdef["foc_idx"]
    foc_inject_idx = cdef["foc_inject_idx"]

    theta_ref = theta_all[ref_idx]
    theta_foc = theta_all[foc_idx]
    ext_ref = full_totals[ref_idx]
    ext_foc = full_totals[foc_idx]

    for p in INJECTION_PS:
        for seed in range(1, N_SEEDS + 1):
            rng = np.random.RandomState(seed)

            contam_local_idx = rng.choice(N_BASELINE, N_CONTAMINATED, replace=False)
            contam_set = set(contam_local_idx.tolist())

            X_inj = X_base.copy()
            n_flipped = 0

            if p > 0:
                for local_j in contam_local_idx:
                    for model_row in foc_inject_idx:
                        if X_inj[model_row, local_j] == 0:
                            if rng.random() < p:
                                X_inj[model_row, local_j] = 1
                                n_flipped += 1

            X_ref = X_inj[ref_idx]
            X_foc = X_inj[foc_idx]

            for variant in MH_VARIANTS:
                run_count += 1

                if variant.startswith("theta_eq_"):
                    ns = int(variant.split("_")[2])
                    dif_results = mh_dif_with_bins(
                        X_ref, X_foc, theta_ref, theta_foc,
                        n_strata=ns, bin_mode="equal"
                    )
                    strata_count = ns
                elif variant.startswith("theta_qt_"):
                    ns = int(variant.split("_")[2])
                    dif_results = mh_dif_with_bins(
                        X_ref, X_foc, theta_ref, theta_foc,
                        n_strata=ns, bin_mode="quantile"
                    )
                    strata_count = ns
                elif variant == "external_5":
                    dif_results = mh_dif_with_bins(
                        X_ref, X_foc, ext_ref, ext_foc,
                        n_strata=5, bin_mode="quantile"
                    )
                    strata_count = 5

                contam_mask = dif_results["col_idx"].isin(contam_set)
                contam_results = dif_results[contam_mask]
                clean_results = dif_results[~contam_mask]

                n_c_contam = (contam_results["ets_class"] == "C").sum()
                n_c_clean = (clean_results["ets_class"] == "C").sum()
                tpr = n_c_contam / N_CONTAMINATED
                fpr = n_c_clean / (N_BASELINE - N_CONTAMINATED)

                table = [[n_c_contam, N_CONTAMINATED - n_c_contam],
                         [n_c_clean, (N_BASELINE - N_CONTAMINATED) - n_c_clean]]
                fisher_or, fisher_p = fisher_exact(table)

                injection_rows.append(dict(
                    p=p, seed=seed, cohort_type=cohort_name,
                    mh_variant=variant,
                    strata_count=strata_count,
                    tpr=tpr, fpr=fpr,
                    fisher_or=fisher_or, fisher_p=fisher_p,
                    n_responses_flipped=n_flipped,
                ))

                elapsed = time.time() - t_start
                if run_count % 35 == 0 or (p in [0.0, 1.0] and seed == 1 and "theta_eq_10" in variant):
                    print(f"  [{run_count}/{total_runs}] {cohort_name} {variant:14s} p={p} seed={seed}: "
                          f"TPR={tpr:.3f} FPR={fpr:.3f} ({elapsed:.0f}s)")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5: AGGREGATE & SAVE
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 5: Aggregating results ===")

inj_df = pd.DataFrame(injection_rows)

summary_rows = []
for (p_val, cohort, variant), grp in inj_df.groupby(["p", "cohort_type", "mh_variant"]):
    summary_rows.append(dict(
        p=p_val, cohort_type=cohort, mh_variant=variant,
        strata_count=grp["strata_count"].iloc[0],
        tpr_mean=grp["tpr"].mean(), tpr_std=grp["tpr"].std(),
        fpr_mean=grp["fpr"].mean(), fpr_std=grp["fpr"].std(),
        fisher_or_mean=grp["fisher_or"].mean(),
        fisher_p_median=grp["fisher_p"].median(),
    ))
summary_df = pd.DataFrame(summary_rows)

inj_df.to_csv(OUT / "injection_results.csv", index=False)
summary_df.to_csv(OUT / "dose_response_theta.csv", index=False)
print(f"  Saved injection_results.csv ({len(inj_df)} rows)")
print(f"  Saved dose_response_theta.csv ({len(summary_df)} rows)")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 6: REAL DATA DIF — ENRICHMENT AGAINST LI ET AL. LABELS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 6: Real data DIF (no injection) — enrichment vs Li et al. ===")

li_item_ids = li_labels["item_id"].values
li_col_indices = np.array([ITEM2COL[iid] for iid in li_item_ids if iid in ITEM2COL])
li_item_ids_valid = np.array([iid for iid in li_item_ids if iid in ITEM2COL])

li_labels_valid = li_labels[li_labels["item_id"].isin(li_item_ids_valid)].copy()
li_labels_valid = li_labels_valid.set_index("item_id").loc[li_item_ids_valid].reset_index()

X_li = mat[:, li_col_indices].toarray().astype(np.int8)
print(f"  Li et al. items submatrix: {X_li.shape}")

enrichment_rows = []

for cohort_name in ["temporal"]:
    cdef = cohort_defs[cohort_name]
    ref_idx = cdef["ref_idx"]
    foc_idx = cdef["foc_idx"]

    X_ref_li = X_li[ref_idx]
    X_foc_li = X_li[foc_idx]

    theta_ref = theta_all[ref_idx]
    theta_foc = theta_all[foc_idx]
    ext_ref = full_totals[ref_idx]
    ext_foc = full_totals[foc_idx]

    for variant in MH_VARIANTS:
        if variant.startswith("theta_eq_"):
            ns = int(variant.split("_")[2])
            dif_results = mh_dif_with_bins(X_ref_li, X_foc_li, theta_ref, theta_foc,
                                           n_strata=ns, bin_mode="equal")
            sc = ns
        elif variant.startswith("theta_qt_"):
            ns = int(variant.split("_")[2])
            dif_results = mh_dif_with_bins(X_ref_li, X_foc_li, theta_ref, theta_foc,
                                           n_strata=ns, bin_mode="quantile")
            sc = ns
        elif variant == "external_5":
            dif_results = mh_dif_with_bins(X_ref_li, X_foc_li, ext_ref, ext_foc,
                                           n_strata=5, bin_mode="quantile")
            sc = 5

        dif_results["item_id"] = li_item_ids_valid[dif_results["col_idx"].values]
        dif_merged = dif_results.merge(li_labels_valid[["item_id", "contamination_type"]], on="item_id")

        is_contam = dif_merged["contamination_type"] != "clean"
        is_clean = dif_merged["contamination_type"] == "clean"
        is_c = dif_merged["ets_class"] == "C"

        rate_contam = (is_c & is_contam).sum() / is_contam.sum() if is_contam.sum() > 0 else 0
        rate_clean = (is_c & is_clean).sum() / is_clean.sum() if is_clean.sum() > 0 else 0
        enrichment = rate_contam / rate_clean if rate_clean > 0 else np.inf

        enrichment_rows.append(dict(
            cohort=cohort_name,
            mh_variant=variant,
            strata_count=sc,
            n_items=len(dif_merged),
            n_contam=int(is_contam.sum()),
            n_clean=int(is_clean.sum()),
            rate_contam=rate_contam,
            rate_clean=rate_clean,
            fpr_clean=rate_clean,
            enrichment=enrichment,
        ))

        print(f"  {variant:14s}: rate_contam={rate_contam:.4f}, "
              f"rate_clean={rate_clean:.4f}, enrichment={enrichment:.3f}")

enrichment_df = pd.DataFrame(enrichment_rows)
enrichment_df.to_csv(OUT / "enrichment_theta.csv", index=False)
print(f"  Saved enrichment_theta.csv ({len(enrichment_df)} rows)")


# ════════════════════════════════════════════════════════════════════════════
#  REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Generating report ===")

L = []
def w(s=""):
    L.append(s)

w("# Plan 012: IRT θ-Conditioning MH-DIF Analysis")
w()
w("## 1. Objective")
w()
w("Replace raw total scores with IRT-based θ estimates as the MH-DIF stratification")
w("variable. Test whether finer ability matching reduces the high baseline FPR from")
w("plan_011 (temporal ~31%, ability ~49%) while preserving detection power (TPR).")
w()

w("## 2. θ Estimation Method")
w()
w("**Method**: Logit of proportion correct with Laplace smoothing")
w("```")
w("θ = log((total + 0.5) / (N_items + 1 - total - 0.5))")
w("```")
w(f"- N models: {N_MODELS}")
w(f"- N items (full matrix): {J_FULL}")
w(f"- θ range: [{theta_all.min():.3f}, {theta_all.max():.3f}]")
w(f"- θ mean±std: {theta_all.mean():.3f}±{theta_all.std():.3f}")
w(f"- Correlation(θ, total_score): {corr:.4f}")
w()
w("This is a monotone transformation of the Rasch model sufficient statistic.")
w("Two binning strategies are compared:")
w("- **Equal-width (eq)**: equal intervals on the logit scale — respects IRT metric")
w("- **Quantile (qt)**: equal-count bins — ensures balanced strata")
w()

w("## 3. Experimental Design")
w()
w(f"- Baseline items: {N_BASELINE}")
w(f"- Contaminated per run: {N_CONTAMINATED}")
w(f"- Injection probabilities: {INJECTION_PS}")
w(f"- Seeds per p: {N_SEEDS}")
w(f"- Strata counts tested: {STRATA_COUNTS}")
w()
w("### Cohort definitions")
w()
for cn, cd in cohort_defs.items():
    w(f"- **{cn}**: ref N={len(cd['ref_idx'])}, foc N={len(cd['foc_idx'])}, "
      f"foc inject N={len(cd['foc_inject_idx'])}")
w()
w("### MH variants")
w()
w("| Variant | Stratification | Description |")
w("|---------|---------------|-------------|")
for ns in STRATA_COUNTS:
    w(f"| theta_eq_{ns} | Equal-width θ bins | {ns} equal intervals on logit scale |")
    w(f"| theta_qt_{ns} | Quantile θ bins | {ns} equal-count bins on θ |")
w("| external_5 | Quantile total-score | 5 bins on full-matrix total (plan_011 baseline) |")
w()

# ─── Results tables ───
w("## 4. Semi-Synthetic Results")
w()

for cohort in ["temporal", "ability"]:
    w(f"### {cohort.capitalize()} cohort")
    w()

    # Baseline FPR table
    w("#### Baseline FPR (p=0)")
    w()
    w("| Variant | FPR (mean±std) |")
    w("|---------|----------------|")
    sub = summary_df[(summary_df["p"] == 0.0) & (summary_df["cohort_type"] == cohort)]
    for _, r in sub.sort_values("mh_variant").iterrows():
        w(f"| {r.mh_variant} | {r.fpr_mean:.3f}±{r.fpr_std:.3f} |")
    w()

    # Dose-response table (selected variants)
    w("#### Dose-response (key variants)")
    w()
    key_variants = ["theta_eq_10", "theta_qt_10", "theta_qt_20", "external_5"]
    w("| Variant | p | TPR (mean±std) | FPR (mean±std) | ΔTPR | Fisher OR | Fisher p |")
    w("|---------|---|----------------|----------------|------|-----------|----------|")
    for variant in key_variants:
        sub_v = summary_df[(summary_df["cohort_type"] == cohort) & (summary_df["mh_variant"] == variant)]
        if len(sub_v) == 0:
            continue
        base_tpr = sub_v[sub_v["p"] == 0.0]["tpr_mean"].values
        base_tpr = base_tpr[0] if len(base_tpr) > 0 else 0
        for _, r in sub_v.sort_values("p").iterrows():
            d_tpr = r.tpr_mean - base_tpr
            w(f"| {r.mh_variant} | {r.p} | {r.tpr_mean:.3f}±{r.tpr_std:.3f} | "
              f"{r.fpr_mean:.3f}±{r.fpr_std:.3f} | {d_tpr:+.3f} | "
              f"{r.fisher_or_mean:.2f} | {r.fisher_p_median:.2e} |")
    w()

# ─── Success criteria ───
w("## 5. Success Criteria Evaluation")
w()
w("**Target**: baseline FPR < 0.10, TPR > 0.70 at p=0.5 (temporal cohort)")
w()

for variant in MH_VARIANTS:
    sub = summary_df[(summary_df["cohort_type"] == "temporal") & (summary_df["mh_variant"] == variant)]
    fpr0 = sub[sub["p"] == 0.0]["fpr_mean"].values
    tpr05 = sub[sub["p"] == 0.5]["tpr_mean"].values
    if len(fpr0) > 0 and len(tpr05) > 0:
        fpr_ok = "PASS" if fpr0[0] < 0.10 else "FAIL"
        tpr_ok = "PASS" if tpr05[0] > 0.70 else "FAIL"
        w(f"- **{variant}**: FPR={fpr0[0]:.3f} [{fpr_ok}], TPR@p=0.5={tpr05[0]:.3f} [{tpr_ok}]")
w()

# ─── Enrichment ───
w("## 6. Enrichment Against Li et al. Labels (No Injection)")
w()
w("| Variant | Strata | Rate(contam) | Rate(clean) | Enrichment |")
w("|---------|--------|-------------|-------------|------------|")
for _, r in enrichment_df.iterrows():
    w(f"| {r.mh_variant} | {r.strata_count} | {r.rate_contam:.4f} | "
      f"{r.rate_clean:.4f} | {r.enrichment:.3f} |")
w()
w("(plan_001 reference: enrichment = 0.96 with 5-strata total-score MH)")
w()

# ─── Analysis ───
w("## 7. Analysis & Conclusions")
w()

# Find best variant
best_variant = None
best_fpr = 1.0
for variant in MH_VARIANTS:
    sub = summary_df[(summary_df["cohort_type"] == "temporal") &
                     (summary_df["mh_variant"] == variant) & (summary_df["p"] == 0.0)]
    if len(sub) > 0:
        fpr = sub.iloc[0]["fpr_mean"]
        if fpr < best_fpr:
            best_fpr = fpr
            best_variant = variant

ext_sub = summary_df[(summary_df["cohort_type"] == "temporal") &
                     (summary_df["mh_variant"] == "external_5") & (summary_df["p"] == 0.0)]
ext_fpr = ext_sub.iloc[0]["fpr_mean"] if len(ext_sub) > 0 else None

w("### θ vs total-score stratification")
w()
if ext_fpr is not None:
    w(f"- Plan_011 external (5 quantile strata, total score) FPR: {ext_fpr:.3f}")
w(f"- Best θ variant ({best_variant}) FPR: {best_fpr:.3f}")
if ext_fpr is not None:
    w(f"- FPR difference: {best_fpr - ext_fpr:+.3f}")
w()

w("### Equal-width vs quantile binning")
w()
for ns in STRATA_COUNTS:
    eq_sub = summary_df[(summary_df["cohort_type"] == "temporal") &
                        (summary_df["mh_variant"] == f"theta_eq_{ns}") & (summary_df["p"] == 0.0)]
    qt_sub = summary_df[(summary_df["cohort_type"] == "temporal") &
                        (summary_df["mh_variant"] == f"theta_qt_{ns}") & (summary_df["p"] == 0.0)]
    if len(eq_sub) > 0 and len(qt_sub) > 0:
        w(f"- K={ns}: equal-width FPR={eq_sub.iloc[0]['fpr_mean']:.3f}, "
          f"quantile FPR={qt_sub.iloc[0]['fpr_mean']:.3f}")
w()

w("### Why FPR remains elevated")
w()
w("Possible explanations if FPR stays high despite θ-conditioning:")
w("1. Real DIF exists between cohorts — genuine differential item functioning")
w("   (2023 vs 2024 models genuinely differ on certain items beyond overall ability)")
w("2. Cohorts differ on dimensions not captured by a single θ parameter")
w("   (e.g., training data composition, architecture effects)")
w("3. The logit(p) θ estimate has correlation >.999 with raw total scores,")
w("   so quantile bins on θ produce nearly identical strata to quantile bins on totals")
w("4. MH-DIF's power to distinguish real DIF from ability confounding is limited")
w("   when the grouping variable (year/quintile) correlates strongly with ability")
w()

w("### Recommendations")
w()
w("Based on these results:")
w()
sub_t = summary_df[summary_df["cohort_type"] == "temporal"]
for variant in ["theta_eq_10", "theta_qt_10", "theta_qt_20", "external_5"]:
    v_sub = sub_t[sub_t["mh_variant"] == variant]
    fpr0 = v_sub[v_sub["p"] == 0.0]["fpr_mean"].values
    tpr05 = v_sub[v_sub["p"] == 0.5]["tpr_mean"].values
    tpr1 = v_sub[v_sub["p"] == 1.0]["tpr_mean"].values
    if len(fpr0) > 0 and len(tpr05) > 0 and len(tpr1) > 0:
        w(f"- {variant}: FPR₀={fpr0[0]:.3f}, TPR@0.5={tpr05[0]:.3f}, TPR@1.0={tpr1[0]:.3f}")
w()

(OUT / "plan_012_report.md").write_text("\n".join(L) + "\n")
print("  Report written to plan_012_report.md")


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Baseline FPR (p=0)")
print("=" * 70)
for cohort in ["temporal", "ability"]:
    print(f"\n  {cohort}:")
    sub = summary_df[(summary_df["p"] == 0.0) & (summary_df["cohort_type"] == cohort)]
    for _, r in sub.sort_values("mh_variant").iterrows():
        print(f"    {r.mh_variant:14s}: FPR={r.fpr_mean:.3f}±{r.fpr_std:.3f}")

print("\n" + "=" * 70)
print("SUMMARY — TPR at p=0.5 (temporal)")
print("=" * 70)
sub = summary_df[(summary_df["p"] == 0.5) & (summary_df["cohort_type"] == "temporal")]
for _, r in sub.sort_values("mh_variant").iterrows():
    print(f"  {r.mh_variant:14s}: TPR={r.tpr_mean:.3f}±{r.tpr_std:.3f}")

print("\n" + "=" * 70)
print("SUMMARY — Enrichment (Li et al.)")
print("=" * 70)
for _, r in enrichment_df.iterrows():
    print(f"  {r.mh_variant:14s}: enrichment={r.enrichment:.3f}")

total_elapsed = time.time() - t_start
print(f"\nTotal time: {total_elapsed:.0f}s")
print(f"All outputs saved to {OUT}")
