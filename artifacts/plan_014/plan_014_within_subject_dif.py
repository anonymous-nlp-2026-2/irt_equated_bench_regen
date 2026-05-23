#!/usr/bin/env python3
"""
plan_014: Within-Subject Temporal DIF + Cross-Subject Matching + SIBTEST Check

Part 1: For 5 high-C% subjects, run purified MH-DIF using within-subject total score
Part 2: Re-run with cross-subject total score (all other subjects) as matching variable
Part 3: Check SIBTEST availability (R packages + Python)
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import chi2 as chi2_dist
from pathlib import Path
import subprocess
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts")
OUT = BASE / "plan_014"
OUT.mkdir(exist_ok=True)

SUBJECTS = [
    ("business_ethics", 93, 48.4),
    ("human_aging", 203, 41.9),
    ("formal_logic", 109, 40.4),
    ("professional_medicine", 248, 37.9),
    ("high_school_statistics", 181, 35.4),
]

N_STRATA = 5
MIN_PER_STRATUM = 5


# ════════════════════════════════════════════════════════════════════════════
#  MH-DIF CORE (adapted from plan_011)
# ════════════════════════════════════════════════════════════════════════════
def purified_mh_dif(X_ref, X_foc, n_strata=5, external_scores_ref=None,
                    external_scores_foc=None, min_per_stratum=5):
    """
    Purified MH-DIF for all columns in X_ref/X_foc.
    If external_scores provided, use those for stratification.
    Otherwise use purified within-subject total (sum minus current item).
    Merges adjacent strata if any stratum has < min_per_stratum in either group.
    """
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
    """Merge adjacent strata if either group has < min_n observations."""
    labels = k_all.copy()
    unique_strata = sorted(np.unique(labels))

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
print("=== Loading data ===")
t0 = time.time()

mat = load_npz(BASE / "response_matrix.npz")
idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
MODEL_NAMES = idx["model_names"]
ITEM_IDS = idx["item_ids"]
ITEM2COL = {iid: i for i, iid in enumerate(ITEM_IDS)}

model_meta = pd.read_csv(BASE / "model_metadata.csv")
item_meta = pd.read_csv(BASE / "item_metadata.csv")

NAME2ROW = {n: i for i, n in enumerate(MODEL_NAMES)}
print(f"  Matrix: {mat.shape[0]} × {mat.shape[1]}  ({time.time()-t0:.1f}s)")

# Temporal cohorts
ref_names = model_meta.loc[model_meta["cohort_temporal"] == "2023", "model_name"]
foc_names = model_meta.loc[model_meta["cohort_temporal"] == "2024", "model_name"]
ref_rows = np.array([NAME2ROW[n] for n in ref_names if n in NAME2ROW])
foc_rows = np.array([NAME2ROW[n] for n in foc_names if n in NAME2ROW])
print(f"  Temporal cohorts: ref(2023)={len(ref_rows)}, foc(2024)={len(foc_rows)}")


# ════════════════════════════════════════════════════════════════════════════
#  PART 1 & 2: WITHIN-SUBJECT + CROSS-SUBJECT DIF
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 1 & 2: Within-Subject + Cross-Subject DIF ===")

all_item_ids = set(ITEM_IDS)
results_rows = []
detail_rows = []

for subj_name, expected_n, expected_c_pct in SUBJECTS:
    print(f"\n--- {subj_name} ---")

    subj_items = item_meta[item_meta["subject"] == subj_name]["item_id"].values
    subj_cols = np.array([ITEM2COL[iid] for iid in subj_items if iid in ITEM2COL])
    n_items = len(subj_cols)
    print(f"  Items: {n_items}")

    # Extract dense submatrices for this subject
    X_subj_ref = mat[ref_rows][:, subj_cols].toarray().astype(np.int8)
    X_subj_foc = mat[foc_rows][:, subj_cols].toarray().astype(np.int8)

    # All other subjects' columns (for cross-subject matching)
    other_cols = np.array([ITEM2COL[iid] for iid in ITEM_IDS
                           if iid in ITEM2COL and iid not in set(subj_items)])

    # Cross-subject total scores
    cross_ref = np.asarray(mat[ref_rows][:, other_cols].sum(axis=1)).ravel().astype(np.float64)
    cross_foc = np.asarray(mat[foc_rows][:, other_cols].sum(axis=1)).ravel().astype(np.float64)

    # Adaptive strata count for small subjects
    n_strata = N_STRATA
    min_group = min(len(ref_rows), len(foc_rows))
    if n_items < 50:
        n_strata = 3

    # --- Part 1: Within-subject matching ---
    print(f"  Running within-subject MH-DIF (n_strata={n_strata})...")
    t1 = time.time()
    dif_within = purified_mh_dif(X_subj_ref, X_subj_foc, n_strata=n_strata,
                                  min_per_stratum=MIN_PER_STRATUM)
    t_within = time.time() - t1

    n_a = (dif_within["ets_class"] == "A").sum()
    n_b = (dif_within["ets_class"] == "B").sum()
    n_c = (dif_within["ets_class"] == "C").sum()
    pct_c = 100 * n_c / n_items if n_items > 0 else 0
    pct_a = 100 * n_a / n_items if n_items > 0 else 0

    print(f"  Within: A={n_a} B={n_b} C={n_c} | C%={pct_c:.1f}% ({t_within:.1f}s)")

    results_rows.append(dict(
        subject=subj_name, n_items=n_items, matching_type="within",
        n_ets_a=int(n_a), n_ets_b=int(n_b), n_ets_c=int(n_c),
        pct_c=round(pct_c, 2), pct_a=round(pct_a, 2)
    ))

    for _, row in dif_within.iterrows():
        detail_rows.append(dict(
            subject=subj_name, item_id=subj_items[int(row["col_idx"])],
            matching_type="within",
            delta_mh=row["delta_mh"], p_value=row["p_value"],
            ets_class=row["ets_class"], n_strata_used=row["n_strata_used"]
        ))

    # --- Part 2: Cross-subject matching ---
    print(f"  Running cross-subject MH-DIF...")
    t2 = time.time()
    dif_cross = purified_mh_dif(X_subj_ref, X_subj_foc, n_strata=n_strata,
                                 external_scores_ref=cross_ref,
                                 external_scores_foc=cross_foc,
                                 min_per_stratum=MIN_PER_STRATUM)
    t_cross = time.time() - t2

    n_a_x = (dif_cross["ets_class"] == "A").sum()
    n_b_x = (dif_cross["ets_class"] == "B").sum()
    n_c_x = (dif_cross["ets_class"] == "C").sum()
    pct_c_x = 100 * n_c_x / n_items if n_items > 0 else 0
    pct_a_x = 100 * n_a_x / n_items if n_items > 0 else 0

    print(f"  Cross:  A={n_a_x} B={n_b_x} C={n_c_x} | C%={pct_c_x:.1f}% ({t_cross:.1f}s)")

    results_rows.append(dict(
        subject=subj_name, n_items=n_items, matching_type="cross",
        n_ets_a=int(n_a_x), n_ets_b=int(n_b_x), n_ets_c=int(n_c_x),
        pct_c=round(pct_c_x, 2), pct_a=round(pct_a_x, 2)
    ))

    for _, row in dif_cross.iterrows():
        detail_rows.append(dict(
            subject=subj_name, item_id=subj_items[int(row["col_idx"])],
            matching_type="cross",
            delta_mh=row["delta_mh"], p_value=row["p_value"],
            ets_class=row["ets_class"], n_strata_used=row["n_strata_used"]
        ))

# Save results
results_df = pd.DataFrame(results_rows)
results_df.to_csv(OUT / "within_subject_dif_results.csv", index=False)
print(f"\nSaved within_subject_dif_results.csv ({len(results_df)} rows)")

detail_df = pd.DataFrame(detail_rows)
detail_df.to_csv(OUT / "within_subject_dif_detail.csv", index=False)
print(f"Saved within_subject_dif_detail.csv ({len(detail_df)} rows)")


# ════════════════════════════════════════════════════════════════════════════
#  PART 3: SIBTEST AVAILABILITY
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Part 3: SIBTEST Availability ===")

sibtest_report = []

# Check R
try:
    r_result = subprocess.run(["R", "--version"], capture_output=True, text=True)
    r_available = r_result.returncode == 0
except FileNotFoundError:
    r_available = False

if r_available:
    r_ver = [l for l in r_result.stdout.split("\n") if "R version" in l]
    r_ver_str = r_ver[0].strip() if r_ver else "unknown version"
    print(f"  R: {r_ver_str}")
    sibtest_report.append(f"R available: {r_ver_str}")

    # Check sirt package
    sirt_result = subprocess.run(
        ["R", "-e", "cat(as.character(packageVersion('sirt')))"],
        capture_output=True, text=True
    )
    if sirt_result.returncode == 0 and "Error" not in sirt_result.stderr:
        sirt_ver = sirt_result.stdout.strip().split("\n")[-1]
        print(f"  sirt package: v{sirt_ver}")
        sibtest_report.append(f"sirt package: v{sirt_ver}")

        # Check if SIBTEST function exists
        sibtest_fn = subprocess.run(
            ["R", "-e", "cat(exists('SIBTEST', where='package:sirt'))"],
            capture_output=True, text=True
        )
        has_sibtest = "TRUE" in sibtest_fn.stdout
        print(f"  sirt::SIBTEST: {'available' if has_sibtest else 'not found'}")
        sibtest_report.append(f"sirt::SIBTEST: {'available' if has_sibtest else 'not found'}")
    else:
        print("  sirt package: not installed")
        sibtest_report.append("sirt package: not installed")

    # Check difR package
    difr_result = subprocess.run(
        ["R", "-e", "cat(as.character(packageVersion('difR')))"],
        capture_output=True, text=True
    )
    if difr_result.returncode == 0 and "Error" not in difr_result.stderr:
        difr_ver = difr_result.stdout.strip().split("\n")[-1]
        print(f"  difR package: v{difr_ver}")
        sibtest_report.append(f"difR package: v{difr_ver}")

        difr_sibtest = subprocess.run(
            ["R", "-e", "cat(exists('difSIBTEST', where='package:difR'))"],
            capture_output=True, text=True
        )
        has_difr_sibtest = "TRUE" in difr_sibtest.stdout
        print(f"  difR::difSIBTEST: {'available' if has_difr_sibtest else 'not found'}")
        sibtest_report.append(f"difR::difSIBTEST: {'available' if has_difr_sibtest else 'not found'}")
    else:
        print("  difR package: not installed")
        sibtest_report.append("difR package: not installed")
else:
    print("  R: not available")
    sibtest_report.append("R: not available")

# Check Python SIBTEST
print("  Checking Python SIBTEST packages...")
for pkg in ["pysibtest", "sibtest", "pyirt"]:
    pip_result = subprocess.run(
        ["pip3", "show", pkg], capture_output=True, text=True
    )
    if pip_result.returncode == 0:
        ver = [l for l in pip_result.stdout.split("\n") if l.startswith("Version:")]
        print(f"  Python {pkg}: {ver[0] if ver else 'installed'}")
        sibtest_report.append(f"Python {pkg}: {ver[0] if ver else 'installed'}")
    else:
        sibtest_report.append(f"Python {pkg}: not installed")

# Search PyPI
pip_search = subprocess.run(
    ["pip3", "index", "versions", "pysibtest"],
    capture_output=True, text=True
)
if pip_search.returncode == 0 and "versions" in pip_search.stdout.lower():
    print(f"  pysibtest on PyPI: found")
    sibtest_report.append("pysibtest on PyPI: found")
else:
    sibtest_report.append("pysibtest on PyPI: not found")

print(f"\n  SIBTEST summary: {'; '.join(sibtest_report)}")


# ════════════════════════════════════════════════════════════════════════════
#  REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Generating report ===")

L = []
def w(s=""):
    L.append(s)

w("# Plan 014: Within-Subject Temporal DIF + SIBTEST Availability")
w()
w("## Background")
w()
w("Plan_012 showed single-dimensional θ-conditioning cannot reduce the temporal cohort's")
w("31% baseline FPR. Before launching a full bifactor model, we check whether the high")
w("C% is driven by between-subject ability differences or within-subject DIF.")
w()

w("## Part 1: Within-Subject Temporal DIF")
w()
w("For each subject, MH-DIF uses **within-subject purified total score** as the matching")
w("variable. This isolates DIF to items within the same content domain, removing")
w("between-subject ability confounds.")
w()

w("### Results")
w()
w("| Subject | N items | ETS A | ETS B | ETS C | C% | A% |")
w("|---------|---------|-------|-------|-------|-----|-----|")
within_df = results_df[results_df["matching_type"] == "within"]
for _, r in within_df.iterrows():
    w(f"| {r.subject} | {r.n_items} | {r.n_ets_a} | {r.n_ets_b} | {r.n_ets_c} | "
      f"{r.pct_c:.1f}% | {r.pct_a:.1f}% |")

within_avg_c = within_df["pct_c"].mean()
w()
w(f"**Average within-subject C%: {within_avg_c:.1f}%**")
w()

w("## Part 2: Cross-Subject Matching Comparison")
w()
w("Cross-subject matching uses total score on **all other subjects** (excluding the")
w("current one) as the matching variable. This eliminates within-subject ability")
w("contamination of the matching variable.")
w()

w("### Results")
w()
w("| Subject | N items | Matching | ETS A | ETS B | ETS C | C% | A% |")
w("|---------|---------|----------|-------|-------|-------|-----|-----|")
for subj in [s[0] for s in SUBJECTS]:
    ws = results_df[(results_df["subject"] == subj) & (results_df["matching_type"] == "within")].iloc[0]
    cs = results_df[(results_df["subject"] == subj) & (results_df["matching_type"] == "cross")].iloc[0]
    w(f"| {subj} | {ws.n_items} | within | {ws.n_ets_a} | {ws.n_ets_b} | {ws.n_ets_c} | "
      f"{ws.pct_c:.1f}% | {ws.pct_a:.1f}% |")
    w(f"| | | cross | {cs.n_ets_a} | {cs.n_ets_b} | {cs.n_ets_c} | "
      f"{cs.pct_c:.1f}% | {cs.pct_a:.1f}% |")

cross_df = results_df[results_df["matching_type"] == "cross"]
cross_avg_c = cross_df["pct_c"].mean()
w()
w(f"**Average cross-subject C%: {cross_avg_c:.1f}%**")
w()

w("### Within vs Cross Comparison")
w()
w("| Subject | Within C% | Cross C% | Δ C% |")
w("|---------|-----------|----------|------|")
for subj in [s[0] for s in SUBJECTS]:
    ws = results_df[(results_df["subject"] == subj) & (results_df["matching_type"] == "within")].iloc[0]
    cs = results_df[(results_df["subject"] == subj) & (results_df["matching_type"] == "cross")].iloc[0]
    delta = ws.pct_c - cs.pct_c
    w(f"| {subj} | {ws.pct_c:.1f}% | {cs.pct_c:.1f}% | {delta:+.1f}% |")

w()

w("## Part 3: SIBTEST Availability")
w()
for line in sibtest_report:
    w(f"- {line}")
w()

w("## Conclusions")
w()

# Decision rules
if within_avg_c < 15:
    decision = "LOW"
    w(f"### Within-subject C% = {within_avg_c:.1f}% → **LOW** (<15%)")
    w()
    w("Within-subject DIF is minimal. The high global C% is driven by **between-subject**")
    w("ability differences, not within-subject temporal DIF. A bifactor model addressing")
    w("subject-specific factors would likely resolve this.")
elif within_avg_c <= 25:
    decision = "MEDIUM"
    w(f"### Within-subject C% = {within_avg_c:.1f}% → **MEDIUM** (15-25%)")
    w()
    w("Within-subject DIF exists but is moderate. Some temporal DIF is genuine within")
    w("subjects, but between-subject effects also contribute. Bifactor model is still")
    w("worth pursuing but may not fully eliminate C%.")
else:
    decision = "HIGH"
    w(f"### Within-subject C% = {within_avg_c:.1f}% → **HIGH** (>25%)")
    w()
    w("Within-subject DIF remains high even after isolating by subject. The temporal")
    w("effect is pervasive across content domains, not just an artifact of between-subject")
    w("ability differences. Bifactor alone will not solve this — need additional")
    w("conditioning or a different DIF method (e.g., SIBTEST with suspect/anchor split).")

w()
w("### Cross-subject matching insight")
w()
delta_avg = within_avg_c - cross_avg_c
if abs(delta_avg) < 5:
    w(f"Within and cross-subject C% are similar (Δ={delta_avg:+.1f}%), suggesting the DIF")
    w("signal is robust to the choice of matching variable.")
elif delta_avg > 5:
    w(f"Within C% > cross C% by {delta_avg:.1f}pp. Within-subject matching may be")
    w("inflated by matching variable contamination (items studied in the subject also")
    w("contribute to the total score).")
else:
    w(f"Cross C% > within C% by {-delta_avg:.1f}pp. Cross-subject matching may introduce")
    w("noise from unrelated content domains.")

w()
w(f"### Recommendation")
w()
if decision == "LOW":
    w("Proceed with bifactor model as planned. Subject-specific factors should capture")
    w("the between-subject ability differences driving the inflated global C%.")
elif decision == "MEDIUM":
    w("Bifactor model is reasonable but set expectations that within-subject DIF of")
    w(f"{within_avg_c:.1f}% will persist. Consider SIBTEST as a complementary method")
    w("if it becomes available.")
else:
    w("Bifactor model alone is insufficient. Investigate:")
    w("1. Whether specific item types (difficulty levels) drive within-subject C%")
    w("2. SIBTEST with empirically determined suspect items")
    w("3. Lord's χ² with IRT parameters per subject")

(OUT / "plan_014_report.md").write_text("\n".join(L) + "\n")
print("Report written.")

# Console summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nWithin-subject avg C%: {within_avg_c:.1f}%")
print(f"Cross-subject avg C%:  {cross_avg_c:.1f}%")
print(f"Decision: {decision}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
print(f"All outputs saved to {OUT}")
