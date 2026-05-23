"""
Plan 021: Non-Uniform DIF Robustness Check via Logistic Regression
Swaminathan & Rogers method: tests both uniform and non-uniform DIF
"""

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = "artifacts"
OUT = f"{BASE}/plan_021"

# ── Load data ──
print("Loading data...")
mat = sp.load_npz(f"{BASE}/response_matrix.npz").toarray()  # (5227, 12508)
idx = np.load(f"{BASE}/response_matrix_index.npz", allow_pickle=True)
model_names = idx["model_names"]
item_ids = idx["item_ids"]

item_meta = pd.read_csv(f"{BASE}/item_metadata.csv")
model_meta = pd.read_csv(f"{BASE}/model_metadata.csv")
mh_results = pd.read_csv(f"{BASE}/plan_001/dif_results_temporal.csv")

# Build item_id -> column index mapping
item_id_to_col = {iid: i for i, iid in enumerate(item_ids)}
model_name_to_row = {mn: i for i, mn in enumerate(model_names)}

# ── Define cohorts (temporal: 2023 vs 2024) ──
ref_mask = model_meta["cohort_temporal"].values == "2023"
foc_mask = model_meta["cohort_temporal"].values == "2024"
ref_idx = np.where(ref_mask)[0]
foc_idx = np.where(foc_mask)[0]
print(f"Reference (2023): {len(ref_idx)} models, Focal (2024): {len(foc_idx)} models")

# ── Subject selection ──
subj_stats = mh_results.groupby("subject").apply(
    lambda g: pd.Series({
        "n_items": len(g),
        "n_C": (g.ets_class == "C").sum(),
        "pct_C": 100 * (g.ets_class == "C").mean(),
    })
).reset_index()

# High C% (>40%)
high = subj_stats[subj_stats.pct_C > 40].sort_values("pct_C", ascending=False)
# Medium C% (25-35%)
mid = subj_stats[(subj_stats.pct_C >= 25) & (subj_stats.pct_C <= 35)].sort_values("pct_C", ascending=False)
# Low C% (<25%)
low = subj_stats[subj_stats.pct_C < 25].sort_values("pct_C", ascending=True)

selected = []
# High: take top 3
for _, r in high.head(3).iterrows():
    selected.append((r.subject, r.n_items, r.pct_C, "high_C"))
# Medium: pick a large, a medium, and a small subject
mid_sorted = mid.sort_values("n_items", ascending=False)
if len(mid_sorted) >= 3:
    selected.append((mid_sorted.iloc[0].subject, mid_sorted.iloc[0].n_items, mid_sorted.iloc[0].pct_C, "medium_C (large n)"))
    mid_mid = mid_sorted.iloc[len(mid_sorted)//2]
    selected.append((mid_mid.subject, mid_mid.n_items, mid_mid.pct_C, "medium_C (medium n)"))
    selected.append((mid_sorted.iloc[-1].subject, mid_sorted.iloc[-1].n_items, mid_sorted.iloc[-1].pct_C, "medium_C (small n)"))
# Low: take bottom 3
for _, r in low.head(3).iterrows():
    selected.append((r.subject, r.n_items, r.pct_C, "low_C"))

sel_df = pd.DataFrame(selected, columns=["subject", "n_items", "mh_pct_C", "selection_reason"])
print(f"\nSelected {len(sel_df)} subjects:")
print(sel_df.to_string(index=False))

# ── Logistic Regression DIF per item ──
def lr_dif_test(y, group, theta):
    """
    Logistic regression DIF test (Swaminathan & Rogers).
    Returns: uniform_p, nonuniform_p, converged, beta_G, beta_interaction
    """
    n = len(y)
    if n < 20:
        return np.nan, np.nan, False, np.nan, np.nan

    # Check variance
    if y.std() == 0 or group.std() == 0:
        return np.nan, np.nan, False, np.nan, np.nan

    # Standardize theta for numerical stability
    theta_z = (theta - theta.mean()) / (theta.std() + 1e-8)
    interaction = theta_z * group

    try:
        # Model 1: Y ~ theta
        X1 = sm.add_constant(theta_z)
        m1 = sm.Logit(y, X1).fit(disp=0, maxiter=100, method="newton", warn_convergence=False)

        # Model 2: Y ~ theta + G
        X2 = sm.add_constant(np.column_stack([theta_z, group]))
        m2 = sm.Logit(y, X2).fit(disp=0, maxiter=100, method="newton", warn_convergence=False)

        # Model 3: Y ~ theta + G + theta*G
        X3 = sm.add_constant(np.column_stack([theta_z, group, interaction]))
        m3 = sm.Logit(y, X3).fit(disp=0, maxiter=100, method="newton", warn_convergence=False)

        # LR tests
        lr_uniform = -2 * (m1.llf - m2.llf)
        lr_nonuniform = -2 * (m2.llf - m3.llf)

        p_uniform = chi2.sf(lr_uniform, df=1)
        p_nonuniform = chi2.sf(lr_nonuniform, df=1)

        beta_G = m2.params[2] if len(m2.params) > 2 else np.nan
        beta_int = m3.params[3] if len(m3.params) > 3 else np.nan

        converged = m1.mle_retvals.get("converged", True) and \
                    m2.mle_retvals.get("converged", True) and \
                    m3.mle_retvals.get("converged", True)

        return p_uniform, p_nonuniform, converged, beta_G, beta_int

    except Exception:
        return np.nan, np.nan, False, np.nan, np.nan


# Combine ref + focal responses
cohort_rows = np.concatenate([ref_idx, foc_idx])
group_vec = np.concatenate([np.zeros(len(ref_idx)), np.ones(len(foc_idx))])
sub_mat = mat[cohort_rows, :]  # (n_ref+n_foc, 12508)

all_results = []

for _, sel_row in sel_df.iterrows():
    subj = sel_row.subject
    subj_items = item_meta[item_meta.subject == subj].item_id.values
    subj_cols = [item_id_to_col[iid] for iid in subj_items if iid in item_id_to_col]

    if len(subj_cols) == 0:
        continue

    # Extract subject-level responses
    subj_resp = sub_mat[:, subj_cols]  # (n_models, n_items_in_subject)

    # Total score within subject (theta proxy)
    # Handle missing: response_matrix uses 0 for wrong, 1 for correct
    # But sparse matrix might have 0 as missing — check
    theta = subj_resp.sum(axis=1).astype(float)

    print(f"\n--- {subj} ({len(subj_cols)} items) ---")
    print(f"  Theta range: [{theta.min():.0f}, {theta.max():.0f}], mean={theta.mean():.1f}")

    n_converged = 0
    n_skipped = 0
    subj_item_results = []

    for j, col_idx in enumerate(subj_cols):
        iid = item_ids[col_idx]
        y = subj_resp[:, j].astype(float)

        # Skip zero-variance items
        if y.std() == 0:
            n_skipped += 1
            continue

        p_uni, p_nonuni, conv, beta_g, beta_int = lr_dif_test(y, group_vec, theta)

        if conv:
            n_converged += 1

        subj_item_results.append({
            "item_id": iid,
            "subject": subj,
            "p_uniform": p_uni,
            "p_nonuniform": p_nonuni,
            "converged": conv,
            "beta_G": beta_g,
            "beta_interaction": beta_int,
        })

    print(f"  Converged: {n_converged}/{len(subj_item_results)}, Skipped (0-var): {n_skipped}")

    if subj_item_results:
        sdf = pd.DataFrame(subj_item_results)

        # BH correction within subject
        valid = sdf.p_uniform.notna()
        if valid.sum() > 0:
            _, sdf.loc[valid, "p_uniform_bh"], _, _ = multipletests(
                sdf.loc[valid, "p_uniform"], method="fdr_bh"
            )
            _, sdf.loc[valid, "p_nonuniform_bh"], _, _ = multipletests(
                sdf.loc[valid, "p_nonuniform"], method="fdr_bh"
            )
        else:
            sdf["p_uniform_bh"] = np.nan
            sdf["p_nonuniform_bh"] = np.nan

        # Classify
        sdf["uniform_dif"] = sdf.p_uniform_bh < 0.05
        sdf["nonuniform_dif"] = sdf.p_nonuniform_bh < 0.05

        all_results.append(sdf)

results = pd.concat(all_results, ignore_index=True)

# ── Per-subject summary ──
print("\n\n=== Per-Subject Summary ===")
summary_rows = []
for subj in sel_df.subject:
    sdf = results[results.subject == subj]
    n_total = len(sdf)
    n_conv = sdf.converged.sum()
    valid = sdf[sdf.p_uniform_bh.notna()]
    n_valid = len(valid)

    n_uni_only = ((valid.uniform_dif) & (~valid.nonuniform_dif)).sum()
    n_nonuni_only = ((~valid.uniform_dif) & (valid.nonuniform_dif)).sum()
    n_both = ((valid.uniform_dif) & (valid.nonuniform_dif)).sum()
    n_neither = ((~valid.uniform_dif) & (~valid.nonuniform_dif)).sum()

    uni_pct = 100 * (n_uni_only + n_both) / n_valid if n_valid > 0 else 0
    nonuni_pct = 100 * (n_nonuni_only + n_both) / n_valid if n_valid > 0 else 0

    mh_pct_c = sel_df[sel_df.subject == subj].mh_pct_C.values[0]

    summary_rows.append({
        "subject": subj,
        "n_items": n_total,
        "n_converged": n_conv,
        "n_valid": n_valid,
        "n_uniform_only": n_uni_only,
        "n_nonuniform_only": n_nonuni_only,
        "n_both": n_both,
        "n_neither": n_neither,
        "uniform_pct": round(uni_pct, 1),
        "nonuniform_pct": round(nonuni_pct, 1),
        "mh_pct_C": round(mh_pct_c, 1),
    })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

# ── Aggregate across all subjects ──
print("\n\n=== Aggregate Summary ===")
total_valid = results[results.p_uniform_bh.notna()]
n_tot = len(total_valid)
n_uni = ((total_valid.uniform_dif) & (~total_valid.nonuniform_dif)).sum()
n_nonuni = ((~total_valid.uniform_dif) & (total_valid.nonuniform_dif)).sum()
n_both_agg = ((total_valid.uniform_dif) & (total_valid.nonuniform_dif)).sum()
n_none = ((~total_valid.uniform_dif) & (~total_valid.nonuniform_dif)).sum()

print(f"Total valid items: {n_tot}")
print(f"  Uniform DIF only:     {n_uni} ({100*n_uni/n_tot:.1f}%)")
print(f"  Non-uniform DIF only: {n_nonuni} ({100*n_nonuni/n_tot:.1f}%)")
print(f"  Both:                 {n_both_agg} ({100*n_both_agg/n_tot:.1f}%)")
print(f"  Neither:              {n_none} ({100*n_none/n_tot:.1f}%)")
print(f"  Any non-uniform:      {n_nonuni + n_both_agg} ({100*(n_nonuni+n_both_agg)/n_tot:.1f}%)")

# ── MH concordance ──
print("\n\n=== MH Concordance ===")
mh_lookup = mh_results.set_index("item_id")["ets_class"].to_dict()
results["mh_class"] = results.item_id.map(mh_lookup)

valid_with_mh = results[(results.p_uniform_bh.notna()) & (results.mh_class.notna())]

mh_c_items = valid_with_mh[valid_with_mh.mh_class == "C"]
mh_noc_items = valid_with_mh[valid_with_mh.mh_class != "C"]

if len(mh_c_items) > 0:
    lr_uni_in_mh_c = mh_c_items.uniform_dif.sum()
    lr_nonuni_in_mh_c = mh_c_items.nonuniform_dif.sum()
    concordance = lr_uni_in_mh_c / len(mh_c_items) * 100
    print(f"MH C items: {len(mh_c_items)}")
    print(f"  Also flagged uniform by LR:     {lr_uni_in_mh_c} ({concordance:.1f}%)")
    print(f"  Also flagged non-uniform by LR: {lr_nonuni_in_mh_c} ({100*lr_nonuni_in_mh_c/len(mh_c_items):.1f}%)")

if len(mh_noc_items) > 0:
    lr_uni_in_noc = mh_noc_items.uniform_dif.sum()
    lr_nonuni_in_noc = mh_noc_items.nonuniform_dif.sum()
    print(f"\nMH A/B items: {len(mh_noc_items)}")
    print(f"  Flagged uniform by LR:     {lr_uni_in_noc} ({100*lr_uni_in_noc/len(mh_noc_items):.1f}%)")
    print(f"  Flagged non-uniform by LR: {lr_nonuni_in_noc} ({100*lr_nonuni_in_noc/len(mh_noc_items):.1f}%)")

# ── Save results ──
results.to_csv(f"{OUT}/lr_dif_item_results.csv", index=False)
summary.to_csv(f"{OUT}/lr_dif_subject_summary.csv", index=False)
sel_df.to_csv(f"{OUT}/selected_subjects.csv", index=False)

# ── Generate report ──
report = f"""# Non-Uniform DIF Robustness Check (Plan 021)

## Method
Logistic regression DIF analysis (Swaminathan & Rogers, 1990) on selected MMLU subjects.
- **Reference cohort**: 2023 models (n={len(ref_idx)})
- **Focal cohort**: 2024 models (n={len(foc_idx)})
- **Matching variable**: within-subject total score (θ)
- **Multiple testing correction**: Benjamini-Hochberg FDR within each subject

Three nested models per item:
1. M1 (baseline): logit(P) = β₀ + β₁θ
2. M2 (uniform DIF): logit(P) = β₀ + β₁θ + β₂G
3. M3 (non-uniform DIF): logit(P) = β₀ + β₁θ + β₂G + β₃(θ×G)

Uniform DIF: M2 vs M1 (LR test, p < 0.05 after BH). Non-uniform DIF: M3 vs M2 (LR test, p < 0.05 after BH).

## 1. Subject Selection

| Subject | n_items | MH C% | Selection Reason |
|---------|---------|-------|------------------|
"""

for _, r in sel_df.iterrows():
    report += f"| {r.subject} | {int(r.n_items)} | {r.mh_pct_C:.1f}% | {r.selection_reason} |\n"

report += f"""
## 2. Per-Subject Results

| Subject | n_items | n_conv | Uniform Only | Non-Uniform Only | Both | Neither | Uniform% | Non-Uniform% | MH C% |
|---------|---------|--------|-------------|-----------------|------|---------|----------|-------------|-------|
"""

for _, r in summary.iterrows():
    report += f"| {r.subject} | {int(r.n_items)} | {int(r.n_converged)} | {int(r.n_uniform_only)} | {int(r.n_nonuniform_only)} | {int(r.n_both)} | {int(r.n_neither)} | {r.uniform_pct}% | {r.nonuniform_pct}% | {r.mh_pct_C}% |\n"

report += f"""
## 3. Aggregate Summary

Across all {n_tot} valid items in selected subjects:
- **Uniform DIF only**: {n_uni} ({100*n_uni/n_tot:.1f}%)
- **Non-uniform DIF only**: {n_nonuni} ({100*n_nonuni/n_tot:.1f}%)
- **Both uniform and non-uniform**: {n_both_agg} ({100*n_both_agg/n_tot:.1f}%)
- **Neither**: {n_none} ({100*n_none/n_tot:.1f}%)
- **Any non-uniform DIF**: {n_nonuni + n_both_agg} ({100*(n_nonuni+n_both_agg)/n_tot:.1f}%)

## 4. MH Concordance

"""

if len(mh_c_items) > 0:
    report += f"""Among {len(mh_c_items)} MH C-flagged items in selected subjects:
- **{lr_uni_in_mh_c}** ({concordance:.1f}%) also flagged as uniform DIF by logistic regression
- **{lr_nonuni_in_mh_c}** ({100*lr_nonuni_in_mh_c/len(mh_c_items):.1f}%) also flagged as non-uniform DIF

"""

if len(mh_noc_items) > 0:
    report += f"""Among {len(mh_noc_items)} MH A/B items:
- **{lr_uni_in_noc}** ({100*lr_uni_in_noc/len(mh_noc_items):.1f}%) flagged as uniform DIF by LR (but not by MH)
- **{lr_nonuni_in_noc}** ({100*lr_nonuni_in_noc/len(mh_noc_items):.1f}%) flagged as non-uniform DIF

"""

# Interpretation
nonuni_rate = 100 * (n_nonuni + n_both_agg) / n_tot if n_tot > 0 else 0
report += f"""## 5. Interpretation

Non-uniform DIF was detected in **{n_nonuni + n_both_agg}** out of **{n_tot}** items ({nonuni_rate:.1f}%).
"""

if nonuni_rate < 10:
    report += """
This is a low rate, indicating that the Mantel-Haenszel method's assumption of uniform DIF is largely appropriate for this dataset. The temporal DIF patterns between 2023 and 2024 model cohorts are predominantly shifts in item difficulty (uniform), not differential discrimination (non-uniform). MH did not miss a substantial class of DIF effects.
"""
elif nonuni_rate < 20:
    report += """
A modest proportion of items show non-uniform DIF. While MH captures the dominant DIF pattern (uniform difficulty shifts), there exists a secondary non-uniform component that MH cannot detect. However, the non-uniform rate is not high enough to invalidate the MH-based conclusions.
"""
else:
    report += """
A notable proportion of items show non-uniform DIF. This suggests that the temporal shift between 2023 and 2024 model cohorts involves not just difficulty changes but also differential discrimination patterns. The MH analysis should be interpreted with this caveat.
"""

if len(mh_c_items) > 0:
    report += f"""
**MH-LR concordance**: {concordance:.1f}% of MH C-class items were also identified as uniform DIF by logistic regression, {"indicating strong agreement between the two methods" if concordance > 60 else "indicating moderate agreement"}.
"""

with open(f"{OUT}/nonuniform_dif_report.md", "w") as f:
    f.write(report)

print(f"\n\nResults saved to {OUT}/")
print("  - lr_dif_item_results.csv")
print("  - lr_dif_subject_summary.csv")
print("  - selected_subjects.csv")
print("  - nonuniform_dif_report.md")
