#!/usr/bin/env python3
"""
plan_034: Multi-Subject Simultaneous Injection Experiment

Tests Proposition 1 (ΔFPR=0) robustness under multi-subject contamination.
When k subjects are simultaneously injected, the cross-subject matching variable
(total score on remaining 57-k uninjected subjects) degrades. ΔFPR(k) on
non-target subjects should increase with k, revealing the validity boundary.

Inputs:
  - artifacts/response_matrix.npz         (5227 × 12508 sparse binary)
  - artifacts/response_matrix_index.npz   (model_names, item_ids)
  - artifacts/model_metadata.csv          (cohort_temporal, accuracy, ...)
  - artifacts/item_metadata.csv           (item_id, subject, ...)

Outputs:
  - multi_subject_injection_results.csv   (per-target-subject DIF results)
  - non_target_fpr_results.csv            (FPR on uninjected subjects)
  - summary_by_k.csv                      (aggregated ΔFPR/ΔTPR by k,p)
  - plan_034_report.md

Dependencies: numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import chi2 as chi2_dist
from pathlib import Path
import time
import warnings
import argparse

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts")
OUT = BASE / "plan_034_multi_subject_injection"
OUT.mkdir(exist_ok=True)

KS = [1, 2, 3, 5, 10, 20]
PS = [0.1, 0.3, 0.5]
N_SEEDS = 5
N_STRATA = 5
MIN_PER_STRATUM = 5


# ════════════════════════════════════════════════════════════════════════════
#  MH-DIF CORE (from plan_016 with strata merging + external score support)
# ════════════════════════════════════════════════════════════════════════════
def purified_mh_dif(X_ref, X_foc, n_strata=5, external_scores_ref=None,
                    external_scores_foc=None, min_per_stratum=5):
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
        groups_all = np.concatenate([np.zeros(len(s_ref)), np.ones(len(s_foc))])

        edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
        if len(edges) < 2:
            rows.append(_nan_row(j))
            continue

        n_bins = len(edges) - 1
        k_all = np.digitize(s_all, edges[1:-1])
        strata_labels = _merge_small_strata(k_all, groups_all, n_bins, min_per_stratum)

        R = S = 0.0
        sum_diff = 0.0
        sum_var = 0.0
        pr = ps_qr = qs = 0.0
        n_ok = 0

        x_all = np.concatenate([X_ref[:, j], X_foc[:, j]])

        for k in np.unique(strata_labels):
            mask_k = (strata_labels == k)
            mask_ref = mask_k & (groups_all == 0)
            mask_foc = mask_k & (groups_all == 1)

            n1 = int(mask_ref.sum())
            n0 = int(mask_foc.sum())
            if n1 == 0 or n0 == 0:
                continue

            A = float(x_all[mask_ref].sum())
            B = float(n1 - A)
            C = float(x_all[mask_foc].sum())
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
                         se=se, chi2=chi2_stat, p_value=pval, ets_class=ets,
                         n_strata_used=n_ok))

    return pd.DataFrame(rows)


def _nan_row(j):
    return dict(col_idx=j, alpha_mh=np.nan, delta_mh=np.nan,
                se=np.nan, chi2=np.nan, p_value=np.nan, ets_class="A",
                n_strata_used=0)


def _merge_small_strata(k_all, groups_all, n_bins, min_n):
    labels = k_all.copy()
    merged = True
    while merged:
        merged = False
        unique_strata = sorted(np.unique(labels))
        for i, k in enumerate(unique_strata):
            mask_k = (labels == k)
            n_ref = int((mask_k & (groups_all == 0)).sum())
            n_foc = int((mask_k & (groups_all == 1)).sum())
            if n_ref < min_n or n_foc < min_n:
                if i + 1 < len(unique_strata):
                    next_k = unique_strata[i + 1]
                    labels[labels == next_k] = k
                elif i > 0:
                    prev_k = unique_strata[i - 1]
                    labels[labels == k] = prev_k
                merged = True
                break
    return labels


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    mat = load_npz(BASE / "response_matrix.npz")
    idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
    model_names = idx["model_names"]
    item_ids = idx["item_ids"]
    item2col = {iid: i for i, iid in enumerate(item_ids)}

    model_meta = pd.read_csv(BASE / "model_metadata.csv")
    item_meta = pd.read_csv(BASE / "item_metadata.csv")
    name2row = {n: i for i, n in enumerate(model_names)}

    # Temporal cohorts
    ref_names = model_meta.loc[model_meta["cohort_temporal"] == "2023", "model_name"]
    foc_names = model_meta.loc[model_meta["cohort_temporal"] == "2024", "model_name"]
    ref_rows = np.array([name2row[n] for n in ref_names if n in name2row])
    foc_rows = np.array([name2row[n] for n in foc_names if n in name2row])

    # Focal inject subset: 2024 models in accuracy overlap with 2023
    ref_acc = model_meta.loc[model_meta["cohort_temporal"] == "2023", "accuracy"]
    foc_df = model_meta[model_meta["cohort_temporal"] == "2024"]
    t_min = max(ref_acc.min(), foc_df["accuracy"].min())
    t_max = min(ref_acc.max(), foc_df["accuracy"].max())
    foc_inject_mask = (foc_df["accuracy"] >= t_min) & (foc_df["accuracy"] <= t_max)
    foc_inject_rows = np.array([name2row[n] for n in foc_df.loc[foc_inject_mask, "model_name"]
                                if n in name2row])

    # Build subject → column indices mapping
    subject_cols = {}
    subjects = sorted(item_meta["subject"].unique())
    for subj in subjects:
        ids = item_meta.loc[item_meta["subject"] == subj, "item_id"].values
        cols = np.array([item2col[iid] for iid in ids if iid in item2col])
        subject_cols[subj] = cols

    return mat, ref_rows, foc_rows, foc_inject_rows, subjects, subject_cols


# ════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════
def run_experiment(ks=KS, ps=PS, n_seeds=N_SEEDS):
    t0 = time.time()
    mat, ref_rows, foc_rows, foc_inject_rows, subjects, subject_cols = load_data()
    print(f"Data loaded: {len(ref_rows)} ref, {len(foc_rows)} foc, "
          f"{len(foc_inject_rows)} foc_inject, {len(subjects)} subjects")

    all_cols = np.concatenate([subject_cols[s] for s in subjects])

    # Precompute per-subject total scores (original matrix) for ref/foc
    # Enables O(1) matching variable via subtraction instead of re-slicing
    subj_totals_ref = {}
    subj_totals_foc = {}
    for subj in subjects:
        s_cols = subject_cols[subj]
        subj_totals_ref[subj] = np.asarray(mat[ref_rows][:, s_cols].sum(axis=1)).ravel().astype(np.float64)
        subj_totals_foc[subj] = np.asarray(mat[foc_rows][:, s_cols].sum(axis=1)).ravel().astype(np.float64)

    # Full total across all subjects
    full_total_ref = sum(subj_totals_ref.values())
    full_total_foc = sum(subj_totals_foc.values())

    # Precompute dense submatrices per subject for ref/foc (avoids repeated sparse→dense)
    subj_X_ref = {}
    subj_X_foc = {}
    for subj in subjects:
        s_cols = subject_cols[subj]
        subj_X_ref[subj] = mat[ref_rows][:, s_cols].toarray().astype(np.int8)
        subj_X_foc[subj] = mat[foc_rows][:, s_cols].toarray().astype(np.int8)

    # ── Phase 1: baseline (p=0) per-subject FPR with cross-subject matching ──
    print("Computing baseline FPR (p=0) for all subjects...")
    baseline_fpr = {}
    baseline_n_c = {}
    for subj in subjects:
        s_cols = subject_cols[subj]
        cross_ref = full_total_ref - subj_totals_ref[subj]
        cross_foc = full_total_foc - subj_totals_foc[subj]

        n_strata = 3 if len(s_cols) < 100 else N_STRATA
        dif = purified_mh_dif(subj_X_ref[subj], subj_X_foc[subj], n_strata=n_strata,
                              external_scores_ref=cross_ref,
                              external_scores_foc=cross_foc,
                              min_per_stratum=MIN_PER_STRATUM)
        n_c = int((dif["ets_class"] == "C").sum())
        fpr = n_c / len(s_cols) if len(s_cols) > 0 else 0.0
        baseline_fpr[subj] = fpr
        baseline_n_c[subj] = n_c

    print(f"Baseline done. Mean FPR = {np.mean(list(baseline_fpr.values())):.4f}")

    # ── Phase 2: injection runs ──
    target_rows_list = []
    non_target_rows_list = []
    total_runs = len(ks) * len(ps) * n_seeds
    run_i = 0

    for k in ks:
        for p in ps:
            for seed in range(1, n_seeds + 1):
                run_i += 1
                rng = np.random.RandomState(seed)
                target_subjects = sorted(rng.choice(subjects, size=k, replace=False))
                non_target_subjects = [s for s in subjects if s not in target_subjects]

                # Build injected matrix columns for target subjects
                # Collect all target columns for injection
                target_cols_all = np.concatenate([subject_cols[s] for s in target_subjects])

                # Inject: flip incorrect→correct for focal inject models on target items
                mat_dense_targets = mat[:, target_cols_all].toarray().astype(np.int8)
                n_flipped = 0
                if p > 0:
                    block = mat_dense_targets[foc_inject_rows]
                    zeros = (block == 0)
                    flip_mask = zeros & (rng.random(block.shape) < p)
                    n_flipped = int(flip_mask.sum())
                    mat_dense_targets[foc_inject_rows] = block + flip_mask.astype(np.int8)

                # Precompute totals for matching variable construction
                # non_target_total = full_total - sum(target subject originals)
                non_target_total_ref = full_total_ref - sum(subj_totals_ref[s] for s in target_subjects)
                non_target_total_foc = full_total_foc - sum(subj_totals_foc[s] for s in target_subjects)

                # Injected target total (all target subjects combined, after injection)
                injected_target_total = mat_dense_targets.sum(axis=1).astype(np.float64)

                # Per-target-subject injected totals (for leave-one-out matching)
                target_col_to_local = {c: i for i, c in enumerate(target_cols_all)}
                target_subj_injected = {}
                for s in target_subjects:
                    local_idxs = np.array([target_col_to_local[c] for c in subject_cols[s]])
                    target_subj_injected[s] = mat_dense_targets[:, local_idxs].sum(axis=1).astype(np.float64)

                elapsed = time.time() - t0
                print(f"  [{run_i}/{total_runs}] k={k} p={p} seed={seed} "
                      f"flipped={n_flipped} ({elapsed:.0f}s)")

                # ── DIF on each target subject ──
                for subj in target_subjects:
                    s_cols = subject_cols[subj]
                    local_idxs = np.array([target_col_to_local[c] for c in s_cols])
                    X_subj_inj = mat_dense_targets[:, local_idxs]

                    X_ref_s = X_subj_inj[ref_rows]
                    X_foc_s = X_subj_inj[foc_rows]

                    # Matching = non_target_total + other_target_injected
                    other_target_inj = injected_target_total - target_subj_injected[subj]
                    match_ref = non_target_total_ref + other_target_inj[ref_rows]
                    match_foc = non_target_total_foc + other_target_inj[foc_rows]

                    n_strata = 3 if len(s_cols) < 100 else N_STRATA
                    dif = purified_mh_dif(X_ref_s, X_foc_s, n_strata=n_strata,
                                          external_scores_ref=match_ref,
                                          external_scores_foc=match_foc,
                                          min_per_stratum=MIN_PER_STRATUM)

                    n_c = int((dif["ets_class"] == "C").sum())
                    # All items in target subject are "contaminated" (injected)
                    tpr = n_c / len(s_cols) if len(s_cols) > 0 else 0.0
                    delta_tpr = tpr - baseline_fpr[subj]  # baseline is p=0 rate

                    target_rows_list.append(dict(
                        k=k, p=p, seed=seed, target_subject=subj,
                        n_items=len(s_cols), tpr=tpr, fpr=np.nan,
                        delta_tpr=delta_tpr, delta_fpr=np.nan,
                        n_target_c=n_c, n_clean_c=np.nan
                    ))

                # ── DIF on each non-target subject (FPR measurement) ──
                # Sample non-targets to keep runtime manageable for large k
                # For k≤5, test all; for k>5, test 20 random non-targets
                if len(non_target_subjects) <= 20:
                    test_nontargets = non_target_subjects
                else:
                    test_nontargets = sorted(rng.choice(non_target_subjects,
                                                        size=20, replace=False))

                for subj in test_nontargets:
                    s_cols = subject_cols[subj]

                    # Matching = (non_target_total - this subject) + all injected target total
                    match_ref = (non_target_total_ref - subj_totals_ref[subj]
                                 + injected_target_total[ref_rows])
                    match_foc = (non_target_total_foc - subj_totals_foc[subj]
                                 + injected_target_total[foc_rows])

                    n_strata = 3 if len(s_cols) < 100 else N_STRATA
                    dif = purified_mh_dif(subj_X_ref[subj], subj_X_foc[subj], n_strata=n_strata,
                                          external_scores_ref=match_ref,
                                          external_scores_foc=match_foc,
                                          min_per_stratum=MIN_PER_STRATUM)

                    n_c = int((dif["ets_class"] == "C").sum())
                    fpr = n_c / len(s_cols) if len(s_cols) > 0 else 0.0
                    delta_fpr = fpr - baseline_fpr[subj]

                    non_target_rows_list.append(dict(
                        k=k, p=p, seed=seed, non_target_subject=subj,
                        n_items=len(s_cols),
                        fpr=fpr, baseline_fpr=baseline_fpr[subj],
                        delta_fpr=delta_fpr
                    ))

    # ── Save results ──
    target_df = pd.DataFrame(target_rows_list)
    target_df.to_csv(OUT / "multi_subject_injection_results.csv", index=False)

    nontarget_df = pd.DataFrame(non_target_rows_list)
    nontarget_df.to_csv(OUT / "non_target_fpr_results.csv", index=False)

    # ── Summary by k,p ──
    summary_rows = []
    for k in ks:
        for p_val in ps:
            t_sub = target_df[(target_df["k"] == k) & (target_df["p"] == p_val)]
            nt_sub = nontarget_df[(nontarget_df["k"] == k) & (nontarget_df["p"] == p_val)]

            mean_tpr = t_sub["tpr"].mean() if len(t_sub) > 0 else np.nan
            mean_delta_tpr = t_sub["delta_tpr"].mean() if len(t_sub) > 0 else np.nan
            std_delta_tpr = t_sub.groupby("seed")["delta_tpr"].mean().std() if len(t_sub) > 0 else np.nan

            mean_fpr_nt = nt_sub["fpr"].mean() if len(nt_sub) > 0 else np.nan
            mean_delta_fpr_nt = nt_sub["delta_fpr"].mean() if len(nt_sub) > 0 else np.nan
            std_delta_fpr_nt = nt_sub.groupby("seed")["delta_fpr"].mean().std() if len(nt_sub) > 0 else np.nan

            summary_rows.append(dict(
                k=k, p=p_val,
                mean_tpr_target=round(mean_tpr, 4),
                mean_delta_tpr=round(mean_delta_tpr, 4),
                std_delta_tpr=round(std_delta_tpr, 4) if not np.isnan(std_delta_tpr) else np.nan,
                mean_fpr_non_target=round(mean_fpr_nt, 4),
                mean_delta_fpr_non_target=round(mean_delta_fpr_nt, 4),
                std_delta_fpr_non_target=round(std_delta_fpr_nt, 4) if not np.isnan(std_delta_fpr_nt) else np.nan,
            ))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "summary_by_k.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Saved to {OUT}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    return target_df, nontarget_df, summary_df


def generate_report(target_df, nontarget_df, summary_df):
    lines = [
        "# Plan 034: Multi-Subject Simultaneous Injection — Report",
        "",
        "## Method",
        "",
        "We test Proposition 1's robustness boundary by simultaneously injecting",
        "differential contamination into k MMLU subjects (k ∈ {1,2,3,5,10,20}).",
        "For each (k, p, seed) combination:",
        "",
        "1. Randomly select k target subjects from 57 MMLU subjects",
        "2. For each item in the k target subjects, flip incorrect→correct responses",
        "   in the focal group (2024 models within accuracy overlap) with probability p",
        "3. Compute cross-subject MH-DIF for each target and non-target subject",
        "   - Matching variable = total score on all subjects EXCEPT the tested one",
        "   - For target subjects: matching includes injected scores from other targets",
        "   - For non-target subjects: matching includes all injected target scores",
        "4. Measure TPR on target subjects and FPR on non-target subjects",
        "",
        f"Grid: k ∈ {{{','.join(map(str, KS))}}}, p ∈ {{{','.join(map(str, PS))}}}, "
        f"5 seeds per combination = {len(KS)*len(PS)*N_SEEDS} runs",
        "",
        "## Core Results: ΔFPR by k and p",
        "",
    ]

    # Summary table
    lines.append("### Target Subjects (TPR)")
    lines.append("")
    lines.append("| k | p | Mean TPR | Mean ΔTPR | SD(ΔTPR) |")
    lines.append("|---|---|----------|-----------|----------|")
    for _, row in summary_df.iterrows():
        lines.append(f"| {row['k']} | {row['p']} | {row['mean_tpr_target']:.4f} | "
                      f"{row['mean_delta_tpr']:.4f} | "
                      f"{row['std_delta_tpr']:.4f} |")
    lines.append("")

    lines.append("### Non-Target Subjects (FPR — key result)")
    lines.append("")
    lines.append("| k | p | Mean FPR | Mean ΔFPR | SD(ΔFPR) |")
    lines.append("|---|---|----------|-----------|----------|")
    for _, row in summary_df.iterrows():
        lines.append(f"| {row['k']} | {row['p']} | {row['mean_fpr_non_target']:.4f} | "
                      f"{row['mean_delta_fpr_non_target']:.4f} | "
                      f"{row['std_delta_fpr_non_target']:.4f} |")
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Find threshold where ΔFPR becomes notable
    for p_val in PS:
        sub = summary_df[summary_df["p"] == p_val]
        significant_k = sub[sub["mean_delta_fpr_non_target"].abs() > 0.01]
        if len(significant_k) > 0:
            first_k = int(significant_k.iloc[0]["k"])
            lines.append(f"- **p={p_val}**: Non-target ΔFPR exceeds 0.01 at k={first_k}")
        else:
            lines.append(f"- **p={p_val}**: Non-target ΔFPR stays below 0.01 for all k tested")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Proposition 1 guarantees ΔFPR=0 when the matching variable is uncontaminated.")
    lines.append("As k increases, more of the cross-subject total score is contaminated,")
    lines.append("violating the proposition's assumption. The k at which ΔFPR becomes")
    lines.append("practically significant marks the validity boundary of cross-subject DIF auditing.")
    lines.append("")

    report_path = OUT / "plan_034_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan 034: Multi-Subject Injection")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation: k∈{1,5,20}, p=0.3, seed=1")
    args = parser.parse_args()

    if args.quick:
        target_df, nontarget_df, summary_df = run_experiment(
            ks=[1, 5, 20], ps=[0.3], n_seeds=1)
    else:
        target_df, nontarget_df, summary_df = run_experiment()

    generate_report(target_df, nontarget_df, summary_df)
