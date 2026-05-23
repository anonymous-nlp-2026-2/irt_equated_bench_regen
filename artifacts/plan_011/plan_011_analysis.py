#!/usr/bin/env python3
"""
plan_011_analysis.py — Semi-Synthetic DIF Validation

Injects known differential contamination into real MMLU response data and tests
whether MH-DIF can detect it. Validates the DIF methodology used in plan_001.

Three MH variants:
  (1) standard  — total score on 500 items minus current (same as plan_001)
  (2) anchor    — total score on 400 known-clean items only
  (3) external  — total score on full 12508-item original matrix (best matching)

Inputs:
  response_matrix.npz, response_matrix_index.npz, model_metadata.csv,
  item_metadata.csv, mmlu_li2024_proxy_labels.csv,
  plan_001/dif_results_ability.csv

Outputs (all in plan_011/):
  injection_results.csv, dose_response_summary.csv, per_item_results.csv,
  plan_011_report.md
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import chi2 as chi2_dist, fisher_exact
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts")
OUT = BASE / "plan_011"
OUT.mkdir(exist_ok=True)

INJECTION_PS = [0.0, 0.3, 0.5, 0.7, 1.0]
N_SEEDS = 5
N_BASELINE = 500
N_CONTAMINATED = 100


# ════════════════════════════════════════════════════════════════════════════
#  MH-DIF CORE
# ════════════════════════════════════════════════════════════════════════════
def mh_dif_subset(X_ref, X_foc, n_strata=5, anchor_cols=None,
                  external_scores_ref=None, external_scores_foc=None):
    """
    Purified MH-DIF for all columns.

    Stratification source (in priority order):
      1. external_scores_ref/foc — precomputed ability scores (e.g. full-matrix total)
      2. anchor_cols — total score from specified columns only
      3. None — standard purified (all cols minus current)
    """
    J = X_ref.shape[1]
    use_external = external_scores_ref is not None

    if not use_external:
        if anchor_cols is not None:
            tot_ref = X_ref[:, anchor_cols].sum(axis=1).astype(np.float64)
            tot_foc = X_foc[:, anchor_cols].sum(axis=1).astype(np.float64)
            anchor_set = set(anchor_cols.tolist()) if hasattr(anchor_cols, 'tolist') else set(anchor_cols)
        else:
            tot_ref = X_ref.sum(axis=1).astype(np.float64)
            tot_foc = X_foc.sum(axis=1).astype(np.float64)
            anchor_set = None

    rows = []
    for j in range(J):
        if use_external:
            s_ref = external_scores_ref
            s_foc = external_scores_foc
        elif anchor_set is not None:
            if j in anchor_set:
                s_ref = tot_ref - X_ref[:, j].astype(np.float64)
                s_foc = tot_foc - X_foc[:, j].astype(np.float64)
            else:
                s_ref = tot_ref
                s_foc = tot_foc
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

        ad = abs(delta) if np.isfinite(delta) else np.inf
        if ad < 1.0:
            ets = "A"
        elif ad >= 1.5 and (not np.isnan(pval)) and pval < 0.05:
            ets = "C"
        else:
            ets = "B"

        rows.append(dict(col_idx=j, alpha_mh=alpha, delta_mh=delta,
                         se=se, chi2=chi2_stat, p_value=pval, ets_class=ets))

    return pd.DataFrame(rows)


def _nan_row(j):
    return dict(col_idx=j, alpha_mh=np.nan, delta_mh=np.nan,
                se=np.nan, chi2=np.nan, p_value=np.nan, ets_class="A")


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
print(f"  Matrix: {N_MODELS} × {J_FULL}")

# Precompute full-matrix total scores (for external matching)
print("  Computing full-matrix total scores...")
full_totals = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)  # (5227,)
print(f"  Full total scores: mean={full_totals.mean():.1f}, "
      f"range=[{full_totals.min():.0f}, {full_totals.max():.0f}]")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1: SELECT BASELINE ITEMS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 1: Selecting baseline items ===")
merged = ability_dif.merge(li_labels[["item_id", "contamination_type"]], on="item_id", how="inner")
baseline_pool = merged[(merged["contamination_type"] == "clean") & (merged["ets_class"] == "A")]
print(f"  Baseline pool (clean + ETS A in ability DIF): {len(baseline_pool)}")

baseline_items = baseline_pool.sample(N_BASELINE, random_state=42)
baseline_item_ids = baseline_items["item_id"].values
baseline_col_indices = np.array([ITEM2COL[iid] for iid in baseline_item_ids])
print(f"  Selected {N_BASELINE} baseline items")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2: DEFINE COHORTS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 2: Defining cohorts ===")


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
print(f"    Accuracy overlap: [{ab_overlap_min:.4f}, {ab_overlap_max:.4f}]"
      f" → {'EMPTY, injecting all focal' if ab_overlap_min > ab_overlap_max else f'{len(foc_ab_inject_idx)} focal models'}")

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
print(f"    Accuracy overlap: [{t_overlap_min:.4f}, {t_overlap_max:.4f}]"
      f" → {len(foc_t_inject_idx)} focal models for injection")

cohort_defs["temporal"] = dict(
    ref_idx=ref_temporal_idx, foc_idx=foc_temporal_idx,
    foc_inject_idx=foc_t_inject_idx
)


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3-4: INJECTION + MH-DIF
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 3-4: Injection + MH-DIF ===")

X_base = mat[:, baseline_col_indices].toarray().astype(np.int8)
print(f"  Baseline submatrix: {X_base.shape}, density: {X_base.mean():.4f}")

injection_rows = []
per_item_rows = []

MH_VARIANTS = ["standard", "anchor", "external"]

total_runs = len(INJECTION_PS) * N_SEEDS * len(cohort_defs) * len(MH_VARIANTS)
run_count = 0
t_start = time.time()

for cohort_name, cdef in cohort_defs.items():
    ref_idx = cdef["ref_idx"]
    foc_idx = cdef["foc_idx"]
    foc_inject_idx = cdef["foc_inject_idx"]

    # External scores for this cohort (from full original matrix)
    ext_ref = full_totals[ref_idx]
    ext_foc = full_totals[foc_idx]

    for p in INJECTION_PS:
        for seed in range(1, N_SEEDS + 1):
            rng = np.random.RandomState(seed)

            contam_local_idx = rng.choice(N_BASELINE, N_CONTAMINATED, replace=False)
            contam_set = set(contam_local_idx.tolist())
            clean_local_idx = np.array([i for i in range(N_BASELINE) if i not in contam_set])

            X_inj = X_base.copy()
            n_flipped = 0

            if p > 0:
                for local_j in contam_local_idx:
                    for model_row in foc_inject_idx:
                        if X_inj[model_row, local_j] == 0:
                            if rng.random() < p:
                                X_inj[model_row, local_j] = 1
                                n_flipped += 1

            orig_density = X_base[np.ix_(foc_inject_idx, contam_local_idx)].mean()
            new_density = X_inj[np.ix_(foc_inject_idx, contam_local_idx)].mean()

            X_ref = X_inj[ref_idx]
            X_foc = X_inj[foc_idx]

            for variant in MH_VARIANTS:
                run_count += 1

                if variant == "anchor":
                    dif_results = mh_dif_subset(X_ref, X_foc, n_strata=5,
                                                anchor_cols=clean_local_idx)
                elif variant == "external":
                    dif_results = mh_dif_subset(X_ref, X_foc, n_strata=5,
                                                external_scores_ref=ext_ref,
                                                external_scores_foc=ext_foc)
                else:
                    dif_results = mh_dif_subset(X_ref, X_foc, n_strata=5)

                contam_mask = dif_results["col_idx"].isin(contam_set)
                clean_mask = ~contam_mask

                contam_results = dif_results[contam_mask]
                clean_results = dif_results[clean_mask]

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
                    n_baseline=N_BASELINE, n_contaminated=N_CONTAMINATED,
                    n_clean=N_BASELINE - N_CONTAMINATED,
                    n_foc_inject=len(foc_inject_idx),
                    tpr=tpr, fpr=fpr,
                    fisher_or=fisher_or, fisher_p=fisher_p,
                    n_responses_flipped=n_flipped,
                    orig_density=orig_density, new_density=new_density,
                ))

                # Per-item results only for anchor variant (keeps CSV manageable)
                if variant == "anchor":
                    for _, row in dif_results.iterrows():
                        local_j = int(row["col_idx"])
                        per_item_rows.append(dict(
                            item_id=baseline_item_ids[local_j],
                            p=p, seed=seed, cohort_type=cohort_name,
                            mh_variant=variant,
                            delta_mh=row["delta_mh"], p_value=row["p_value"],
                            ets_class=row["ets_class"],
                            is_injected=local_j in contam_set,
                        ))

                elapsed = time.time() - t_start
                if variant == "external" or (variant == "standard" and p in [0.0, 1.0]):
                    print(f"  [{run_count}/{total_runs}] {cohort_name} {variant:8s} p={p} seed={seed}: "
                          f"TPR={tpr:.3f} FPR={fpr:.3f} flipped={n_flipped} "
                          f"({elapsed:.0f}s)")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5-6: AGGREGATE & SAVE
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 5: Aggregating results ===")

inj_df = pd.DataFrame(injection_rows)
per_item_df = pd.DataFrame(per_item_rows)

summary_rows = []
for (p_val, cohort, variant), grp in inj_df.groupby(["p", "cohort_type", "mh_variant"]):
    summary_rows.append(dict(
        p=p_val, cohort_type=cohort, mh_variant=variant,
        tpr_mean=grp["tpr"].mean(), tpr_std=grp["tpr"].std(),
        fpr_mean=grp["fpr"].mean(), fpr_std=grp["fpr"].std(),
        fisher_or_mean=grp["fisher_or"].mean(),
        fisher_p_median=grp["fisher_p"].median(),
    ))
summary_df = pd.DataFrame(summary_rows)

inj_df.to_csv(OUT / "injection_results.csv", index=False)
summary_df.to_csv(OUT / "dose_response_summary.csv", index=False)
per_item_df.to_csv(OUT / "per_item_results.csv", index=False)
print(f"  Saved injection_results.csv ({len(inj_df)} rows)")
print(f"  Saved dose_response_summary.csv ({len(summary_df)} rows)")
print(f"  Saved per_item_results.csv ({len(per_item_df)} rows)")


# ════════════════════════════════════════════════════════════════════════════
#  REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Generating report ===")

L = []
def w(s=""):
    L.append(s)

w("# Plan 011: Semi-Synthetic DIF Validation")
w()
w("## 1. Objective")
w()
w("Inject known **differential** contamination into real MMLU response data and test")
w("whether MH-DIF can detect it. This validates whether plan_001/001b's failure to find")
w("enrichment was due to Li et al. labels being non-differential (global contamination)")
w("rather than a fundamental limitation of MH-DIF.")
w()

w("## 2. Setup")
w()
w(f"- **Baseline items**: {N_BASELINE} (clean in Li et al. + ETS A in plan_001 ability DIF)")
w(f"- **Contaminated per run**: {N_CONTAMINATED} of {N_BASELINE}")
w(f"- **Injection probabilities**: {INJECTION_PS}")
w(f"- **Seeds per p**: {N_SEEDS}")
w()
w("### Cohort definitions")
w()
for cn, cd in cohort_defs.items():
    w(f"- **{cn}**: ref N={len(cd['ref_idx'])}, foc N={len(cd['foc_idx'])}, "
      f"foc inject N={len(cd['foc_inject_idx'])}")
w()

if ab_overlap_min > ab_overlap_max:
    w("> **Note**: Ability quintiles (Q1-2 vs Q4-5) have no accuracy overlap by construction "
      f"(ref max={ref_acc.max():.4f}, foc min={foc_acc.min():.4f}). "
      "All focal models used for injection. MH stratification handles ability matching.")
    w()

w("### MH variants")
w()
w("| Variant | Matching variable | Rationale |")
w("|---------|-------------------|-----------|")
w("| standard | sum of 500 items − current | Same as plan_001; susceptible to matching variable contamination when 20% items injected |")
w("| anchor | sum of 400 known-clean items | Eliminates matching variable contamination; assumes oracle knowledge of clean items |")
w("| external | sum of all 12,508 items (original, pre-injection) | Best matching precision; immune to injection; requires external ability estimate |")
w()

w("## 3. Results by MH Variant")
w()
for variant in MH_VARIANTS:
    w(f"### {variant.capitalize()} MH")
    w()
    w("| Cohort | p | TPR (mean±std) | FPR (mean±std) | ΔTPR | Fisher OR | Fisher p |")
    w("|--------|---|----------------|----------------|------|-----------|----------|")
    sub = summary_df[summary_df["mh_variant"] == variant]
    for _, r in sub.sort_values(["cohort_type", "p"]).iterrows():
        # ΔTPR = TPR - baseline TPR at p=0
        base_tpr = sub[(sub["cohort_type"] == r.cohort_type) & (sub["p"] == 0.0)]["tpr_mean"].values
        d_tpr = r.tpr_mean - base_tpr[0] if len(base_tpr) > 0 else 0
        w(f"| {r.cohort_type} | {r.p} | {r.tpr_mean:.3f}±{r.tpr_std:.3f} | "
          f"{r.fpr_mean:.3f}±{r.fpr_std:.3f} | {d_tpr:+.3f} | "
          f"{r.fisher_or_mean:.2f} | {r.fisher_p_median:.4f} |")
    w()

w("## 4. Key Comparison: Baseline FPR by Variant")
w()
w("| Cohort | standard | anchor | external |")
w("|--------|----------|--------|----------|")
for cohort in ["ability", "temporal"]:
    vals = []
    for variant in MH_VARIANTS:
        r = summary_df[(summary_df["p"] == 0.0) & (summary_df["cohort_type"] == cohort) &
                       (summary_df["mh_variant"] == variant)]
        vals.append(f"{r.iloc[0]['fpr_mean']:.3f}" if len(r) > 0 else "—")
    w(f"| {cohort} | {vals[0]} | {vals[1]} | {vals[2]} |")
w()
w("If external FPR ≪ standard FPR, the high baseline rate is from matching precision,")
w("not inherent DIF in the selected items.")
w()

w("## 5. Sanity Checks")
w()
w("### Density change (focal × contaminated items)")
w()
w("| Cohort | p | Orig density | Post density | Responses flipped |")
w("|--------|---|--------------|--------------|-------------------|")
for (p_val, cohort), grp in inj_df[inj_df["mh_variant"] == "standard"].groupby(["p", "cohort_type"]):
    if p_val == 0:
        continue
    w(f"| {cohort} | {p_val} | {grp['orig_density'].mean():.4f} | "
      f"{grp['new_density'].mean():.4f} | {grp['n_responses_flipped'].mean():.0f} |")
w()

w("## 6. Verdict")
w()

for variant in MH_VARIANTS:
    w(f"### {variant.capitalize()} MH")
    w()
    sub = summary_df[summary_df["mh_variant"] == variant]

    for cohort in ["ability", "temporal"]:
        c_sub = sub[sub["cohort_type"] == cohort]
        p0 = c_sub[c_sub["p"] == 0.0]
        p1 = c_sub[c_sub["p"] == 1.0]
        if len(p1) == 0 or len(p0) == 0:
            continue

        fpr0 = p0.iloc[0]["fpr_mean"]
        tpr0 = p0.iloc[0]["tpr_mean"]
        tpr1 = p1.iloc[0]["tpr_mean"]
        fpr1 = p1.iloc[0]["fpr_mean"]
        d_tpr = tpr1 - tpr0

        w(f"**{cohort}**: baseline TPR={tpr0:.3f}/FPR={fpr0:.3f}, "
          f"p=1.0 TPR={tpr1:.3f}/FPR={fpr1:.3f}, ΔTPR={d_tpr:+.3f}")

        if d_tpr > 0.3 and tpr1 > 0.5:
            w(f"→ **SUCCESS**: injection raises TPR by {d_tpr:.3f} with clear dose-response")
        elif d_tpr > 0.1:
            w(f"→ **PARTIAL**: detectable signal (ΔTPR={d_tpr:.3f}) but modest")
        else:
            w(f"→ **FAIL**: ΔTPR={d_tpr:.3f} too small to indicate detection")
        w()

w("## 7. Overall Interpretation")
w()
w("### Does MH-DIF detect differential contamination?")
w()

# Compute key numbers for conclusion
ext_t = summary_df[(summary_df["mh_variant"] == "external") & (summary_df["cohort_type"] == "temporal")]
ext_a = summary_df[(summary_df["mh_variant"] == "external") & (summary_df["cohort_type"] == "ability")]

if len(ext_t) > 0:
    ext_t_p0 = ext_t[ext_t["p"] == 0.0]["tpr_mean"].values[0]
    ext_t_p1 = ext_t[ext_t["p"] == 1.0]["tpr_mean"].values[0]
    ext_t_fpr0 = ext_t[ext_t["p"] == 0.0]["fpr_mean"].values[0]
    ext_t_fpr1 = ext_t[ext_t["p"] == 1.0]["fpr_mean"].values[0]
    w(f"- **Temporal + external matching**: TPR rises {ext_t_p0:.3f} → {ext_t_p1:.3f} "
      f"(ΔTPR=+{ext_t_p1-ext_t_p0:.3f}), FPR {ext_t_fpr0:.3f} → {ext_t_fpr1:.3f}")

if len(ext_a) > 0:
    ext_a_p0 = ext_a[ext_a["p"] == 0.0]["tpr_mean"].values[0]
    ext_a_p1 = ext_a[ext_a["p"] == 1.0]["tpr_mean"].values[0]
    ext_a_fpr0 = ext_a[ext_a["p"] == 0.0]["fpr_mean"].values[0]
    ext_a_fpr1 = ext_a[ext_a["p"] == 1.0]["fpr_mean"].values[0]
    w(f"- **Ability + external matching**: TPR rises {ext_a_p0:.3f} → {ext_a_p1:.3f} "
      f"(ΔTPR=+{ext_a_p1-ext_a_p0:.3f}), FPR {ext_a_fpr0:.3f} → {ext_a_fpr1:.3f}")

w()
w("### Implications for plan_001")
w()
w("If semi-synthetic injection is detectable → plan_001's failure to find enrichment")
w("against Li et al. labels is because Li et al. contamination is **non-differential**")
w("(Common Crawl data available to all models), not because MH-DIF is fundamentally limited.")
w()
w("### Matching variable contamination")
w()
w("When 20% of items are injected, standard MH's matching variable (total score) is")
w("corrupted by injected items, inflating FPR. This is a known limitation when DIF items")
w("constitute a large fraction of the test. In practice, iterative purification (exclude")
w("suspected DIF items from matching) or external ability estimates mitigate this.")
w()

(OUT / "plan_011_report.md").write_text("\n".join(L) + "\n")
print("  Report written.")

# Console summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for variant in MH_VARIANTS:
    print(f"\n--- {variant} MH ---")
    sub = summary_df[summary_df["mh_variant"] == variant]
    for _, r in sub.sort_values(["cohort_type", "p"]).iterrows():
        print(f"  {r.cohort_type:8s} p={r.p}: TPR={r.tpr_mean:.3f}±{r.tpr_std:.3f}  "
              f"FPR={r.fpr_mean:.3f}±{r.fpr_std:.3f}")

total_elapsed = time.time() - t_start
print(f"\nTotal time: {total_elapsed:.0f}s")
print(f"All outputs saved to {OUT}")
