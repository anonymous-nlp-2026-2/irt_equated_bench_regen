"""
plan_033: Subject-Level DIF Concentration Analysis

Analyzes whether DIF in MMLU is concentrated in specific knowledge domains.
Inputs:
  - artifacts/plan_001/dif_results_temporal.csv  (MH DIF results per item)
  - artifacts/item_metadata.csv                  (CTT difficulty/discrimination)
Outputs (all in artifacts/plan_033_subject_dif_concentration/):
  - subject_dif_concentration.csv
  - domain_group_summary.csv
  - dif_prediction_model.csv
  - subject_residuals.csv
  - plan_033_report.md
"""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent

# ── Load data ────────────────────────────────────────────────────────────────

dif = pd.read_csv(BASE / "plan_001" / "dif_results_temporal.csv")
meta = pd.read_csv(BASE / "item_metadata.csv")

df = dif.merge(meta[["item_id", "ctt_difficulty", "ctt_discrimination"]], on="item_id", how="left")
print(f"Merged dataset: {len(df)} items, {df['subject'].nunique()} subjects")

# ── Domain classification ────────────────────────────────────────────────────

STEM = {
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "machine_learning", "medical_genetics", "virology",
}

HUMANITIES = {
    "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "jurisprudence", "logical_fallacies",
    "moral_disputes", "moral_scenarios", "philosophy", "prehistory",
    "world_religions",
}

SOCIAL_SCIENCE = {
    "business_ethics", "clinical_knowledge", "college_medicine", "econometrics",
    "global_facts", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics",
    "high_school_psychology", "human_aging", "human_sexuality",
    "international_law", "management", "marketing", "miscellaneous", "nutrition",
    "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy",
}


def get_domain(subj):
    if subj in STEM:
        return "STEM"
    if subj in HUMANITIES:
        return "Humanities"
    if subj in SOCIAL_SCIENCE:
        return "Social Science"
    return "Other"


df["domain"] = df["subject"].map(get_domain)

all_subjects = set(df["subject"].unique())
classified = STEM | HUMANITIES | SOCIAL_SCIENCE
unclassified = all_subjects - classified
if unclassified:
    print(f"WARNING: unclassified subjects mapped to Other: {unclassified}")

# ── 1. Subject-level DIF concentration ───────────────────────────────────────

def subject_stats(g):
    n = len(g)
    n_a = (g["ets_class"] == "A").sum()
    n_b = (g["ets_class"] == "B").sum()
    n_c = (g["ets_class"] == "C").sum()
    pct_c = n_c / n * 100 if n > 0 else 0

    c_items = g[g["ets_class"] == "C"]
    if len(c_items) > 0:
        pct_favor_focal = (c_items["alpha_mh"] < 1).sum() / len(c_items) * 100
    else:
        pct_favor_focal = np.nan

    return pd.Series({
        "domain": g["domain"].iloc[0],
        "n_items": n,
        "n_a": n_a,
        "n_b": n_b,
        "n_c": n_c,
        "pct_c": round(pct_c, 2),
        "mean_difficulty": round(g["ctt_difficulty"].mean(), 4),
        "std_difficulty": round(g["ctt_difficulty"].std(), 4),
        "mean_discrimination": round(g["ctt_discrimination"].mean(), 4),
        "std_discrimination": round(g["ctt_discrimination"].std(), 4),
        "mean_delta_mh": round(g["delta_mh"].mean(), 4),
        "std_delta_mh": round(g["delta_mh"].std(), 4),
        "pct_favor_focal": round(pct_favor_focal, 2) if not np.isnan(pct_favor_focal) else np.nan,
    })


subj_df = df.groupby("subject").apply(subject_stats).reset_index()
subj_df = subj_df.sort_values("pct_c", ascending=False)
subj_df.to_csv(OUT / "subject_dif_concentration.csv", index=False)

print("\n" + "=" * 70)
print("SUBJECT-LEVEL DIF CONCENTRATION (top 15 by %C)")
print("=" * 70)
print(subj_df[["subject", "domain", "n_items", "n_c", "pct_c", "mean_difficulty",
               "mean_discrimination", "pct_favor_focal"]].head(15).to_string(index=False))

print(f"\nBottom 5 by %C:")
print(subj_df[["subject", "domain", "n_items", "n_c", "pct_c"]].tail(5).to_string(index=False))

# ── 2. Domain group summary ─────────────────────────────────────────────────

def domain_stats(g):
    n = len(g)
    n_c = (g["ets_class"] == "C").sum()
    c_items = g[g["ets_class"] == "C"]
    pct_favor_focal = (c_items["alpha_mh"] < 1).sum() / len(c_items) * 100 if len(c_items) > 0 else np.nan
    return pd.Series({
        "n_subjects": g["subject"].nunique(),
        "n_items": n,
        "n_c": n_c,
        "pct_c": round(n_c / n * 100, 2),
        "mean_difficulty": round(g["ctt_difficulty"].mean(), 4),
        "mean_discrimination": round(g["ctt_discrimination"].mean(), 4),
        "mean_delta_mh": round(g["delta_mh"].mean(), 4),
        "pct_favor_focal": round(pct_favor_focal, 2) if not np.isnan(pct_favor_focal) else np.nan,
    })


domain_df = df.groupby("domain").apply(domain_stats).reset_index()
domain_df = domain_df.sort_values("pct_c", ascending=False)
domain_df.to_csv(OUT / "domain_group_summary.csv", index=False)

print("\n" + "=" * 70)
print("DOMAIN GROUP SUMMARY")
print("=" * 70)
print(domain_df.to_string(index=False))

# STEM vs Humanities test
stem_subj = subj_df[subj_df["domain"] == "STEM"]["pct_c"]
hum_subj = subj_df[subj_df["domain"] == "Humanities"]["pct_c"]
soc_subj = subj_df[subj_df["domain"] == "Social Science"]["pct_c"]

mw_stat, mw_p = stats.mannwhitneyu(stem_subj, hum_subj, alternative="two-sided")
tt_stat, tt_p = stats.ttest_ind(stem_subj, hum_subj, equal_var=False)

print(f"\nSTEM vs Humanities %C comparison:")
print(f"  STEM mean %C:       {stem_subj.mean():.2f}  (n={len(stem_subj)} subjects)")
print(f"  Humanities mean %C:  {hum_subj.mean():.2f}  (n={len(hum_subj)} subjects)")
print(f"  Social Sci mean %C:  {soc_subj.mean():.2f}  (n={len(soc_subj)} subjects)")
print(f"  Mann-Whitney U:      U={mw_stat:.1f}, p={mw_p:.4f}")
print(f"  Welch t-test:        t={tt_stat:.3f}, p={tt_p:.4f}")

# Kruskal-Wallis across all 3 groups
kw_stat, kw_p = stats.kruskal(stem_subj, hum_subj, soc_subj)
print(f"  Kruskal-Wallis (3 groups): H={kw_stat:.3f}, p={kw_p:.4f}")

# ── 3. DIF prediction model ─────────────────────────────────────────────────

df["dif_c"] = (df["ets_class"] == "C").astype(int)
df["n_items_in_subject"] = df.groupby("subject")["item_id"].transform("count")

model_df = df[["dif_c", "ctt_difficulty", "ctt_discrimination", "n_items_in_subject"]].dropna()

# Logistic regression
model = smf.logit("dif_c ~ ctt_difficulty + ctt_discrimination + n_items_in_subject", data=model_df).fit(disp=0)

coef_table = pd.DataFrame({
    "variable": model.params.index,
    "coef": model.params.values,
    "se": model.bse.values,
    "z": model.tvalues.values,
    "p_value": model.pvalues.values,
    "odds_ratio": np.exp(model.params.values),
    "or_ci_lower": np.exp(model.conf_int()[0].values),
    "or_ci_upper": np.exp(model.conf_int()[1].values),
})
coef_table.to_csv(OUT / "dif_prediction_model.csv", index=False)

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION: DIF_C ~ difficulty + discrimination + n_items")
print("=" * 70)
print(f"N = {int(model.nobs)}, Pseudo R² = {model.prsquared:.4f}")
print(coef_table[["variable", "odds_ratio", "or_ci_lower", "or_ci_upper", "p_value"]].to_string(index=False))

# Subject residuals: add subject dummies
df_for_resid = df[["dif_c", "ctt_difficulty", "ctt_discrimination", "subject"]].dropna()
base_pred = smf.logit("dif_c ~ ctt_difficulty + ctt_discrimination", data=df_for_resid).fit(disp=0)
df_for_resid["predicted_c"] = base_pred.predict(df_for_resid)

resid_by_subj = df_for_resid.groupby("subject").agg(
    n_items=("dif_c", "count"),
    observed_c=("dif_c", "sum"),
    expected_c=("predicted_c", "sum"),
).reset_index()
resid_by_subj["excess_c"] = resid_by_subj["observed_c"] - resid_by_subj["expected_c"]
resid_by_subj["observed_pct_c"] = (resid_by_subj["observed_c"] / resid_by_subj["n_items"] * 100).round(2)
resid_by_subj["expected_pct_c"] = (resid_by_subj["expected_c"] / resid_by_subj["n_items"] * 100).round(2)
resid_by_subj["excess_pct"] = (resid_by_subj["excess_c"] / resid_by_subj["n_items"] * 100).round(2)

# Binomial test for significance
def binom_p(row):
    p_exp = row["expected_c"] / row["n_items"] if row["n_items"] > 0 else 0
    p_exp = np.clip(p_exp, 1e-10, 1 - 1e-10)
    return stats.binomtest(int(row["observed_c"]), int(row["n_items"]), p_exp).pvalue

resid_by_subj["binom_p"] = resid_by_subj.apply(binom_p, axis=1)
resid_by_subj = resid_by_subj.sort_values("excess_c", ascending=False)
resid_by_subj["domain"] = resid_by_subj["subject"].map(get_domain)
resid_by_subj.to_csv(OUT / "subject_residuals.csv", index=False)

print("\n" + "=" * 70)
print("SUBJECT RESIDUALS (controlling difficulty + discrimination)")
print("=" * 70)
print("Top 10 excess DIF:")
print(resid_by_subj[["subject", "domain", "n_items", "observed_c", "expected_c",
                      "excess_c", "excess_pct", "binom_p"]].head(10).to_string(index=False))
print("\nBottom 10 (less DIF than expected):")
print(resid_by_subj[["subject", "domain", "n_items", "observed_c", "expected_c",
                      "excess_c", "excess_pct", "binom_p"]].tail(10).to_string(index=False))

# ── 4. Temporal asymmetry per subject ────────────────────────────────────────

c_items = df[df["ets_class"] == "C"].copy()

def temporal_asym(g):
    n = len(g)
    n_favor_focal = (g["alpha_mh"] < 1).sum()
    n_favor_ref = (g["alpha_mh"] > 1).sum()
    pct_favor_focal = n_favor_focal / n * 100 if n > 0 else np.nan
    # Binomial test for asymmetry (H0: 50/50)
    if n >= 3:
        binom_pval = stats.binomtest(n_favor_focal, n, 0.5).pvalue
    else:
        binom_pval = np.nan
    return pd.Series({
        "n_c_items": n,
        "n_favor_2024": n_favor_focal,
        "n_favor_2023": n_favor_ref,
        "pct_favor_2024": round(pct_favor_focal, 1),
        "asym_binom_p": round(binom_pval, 4) if not np.isnan(binom_pval) else np.nan,
    })


asym_df = c_items.groupby("subject").apply(temporal_asym).reset_index()
asym_df["domain"] = asym_df["subject"].map(get_domain)
asym_df = asym_df.sort_values("pct_favor_2024", ascending=False)

# Merge asymmetry into subject_dif_concentration
subj_df = subj_df.merge(
    asym_df[["subject", "n_favor_2024", "n_favor_2023", "asym_binom_p"]],
    on="subject", how="left"
)
subj_df.to_csv(OUT / "subject_dif_concentration.csv", index=False)

print("\n" + "=" * 70)
print("TEMPORAL ASYMMETRY (C-flagged items: favor 2024 vs 2023)")
print("=" * 70)
print("Subjects with strongest 2024-favoring asymmetry:")
top_asym = asym_df[asym_df["n_c_items"] >= 5].head(10)
print(top_asym[["subject", "domain", "n_c_items", "n_favor_2024", "n_favor_2023",
                "pct_favor_2024", "asym_binom_p"]].to_string(index=False))

print("\nSubjects with strongest 2023-favoring asymmetry:")
bot_asym = asym_df[asym_df["n_c_items"] >= 5].tail(10)
print(bot_asym[["subject", "domain", "n_c_items", "n_favor_2024", "n_favor_2023",
                "pct_favor_2024", "asym_binom_p"]].to_string(index=False))

# Overall temporal direction
overall_favor_2024 = (c_items["alpha_mh"] < 1).sum()
overall_favor_2023 = (c_items["alpha_mh"] > 1).sum()
print(f"\nOverall C-items: {overall_favor_2024} favor 2024, {overall_favor_2023} favor 2023 "
      f"({overall_favor_2024/(overall_favor_2024+overall_favor_2023)*100:.1f}% favor 2024)")

# ── 5. Generate report ───────────────────────────────────────────────────────

sig_stem_hum = "significant" if mw_p < 0.05 else "not significant"
top3 = subj_df.nlargest(3, "pct_c")[["subject", "pct_c"]].values
bot3 = subj_df.nsmallest(3, "pct_c")[["subject", "pct_c"]].values

sig_excess = resid_by_subj[resid_by_subj["binom_p"] < 0.05]
excess_high = sig_excess[sig_excess["excess_c"] > 0]
excess_low = sig_excess[sig_excess["excess_c"] < 0]

# Subjects with unidirectional DIF (all or nearly all favor one direction, n>=5)
unidir = asym_df[(asym_df["n_c_items"] >= 5) &
                  ((asym_df["pct_favor_2024"] >= 80) | (asym_df["pct_favor_2024"] <= 20))]

report = f"""# Plan 033: Subject-Level DIF Concentration Analysis

## Overview

Analyzed DIF distribution across {df['subject'].nunique()} MMLU subjects ({len(df)} items total)
to determine whether differential item functioning concentrates in specific knowledge domains.

## 1. Subject-Level DIF Rates

| Metric | Value |
|--------|-------|
| Subjects with C-flagged items | {(subj_df['n_c'] > 0).sum()} / {len(subj_df)} |
| Mean %C across subjects | {subj_df['pct_c'].mean():.1f}% |
| Median %C | {subj_df['pct_c'].median():.1f}% |
| Range | {subj_df['pct_c'].min():.1f}% – {subj_df['pct_c'].max():.1f}% |

**Highest DIF concentration:**
- {top3[0][0]}: {top3[0][1]:.1f}%
- {top3[1][0]}: {top3[1][1]:.1f}%
- {top3[2][0]}: {top3[2][1]:.1f}%

**Lowest DIF concentration:**
- {bot3[0][0]}: {bot3[0][1]:.1f}%
- {bot3[1][0]}: {bot3[1][1]:.1f}%
- {bot3[2][0]}: {bot3[2][1]:.1f}%

## 2. Domain-Level Summary

{domain_df.to_csv(index=False)}

**STEM vs Humanities:**
- STEM mean %C = {stem_subj.mean():.1f}%, Humanities mean %C = {hum_subj.mean():.1f}%
- Mann-Whitney U = {mw_stat:.1f}, p = {mw_p:.4f} ({sig_stem_hum} at α=0.05)
- Welch t = {tt_stat:.3f}, p = {tt_p:.4f}
- Kruskal-Wallis (3 groups): H = {kw_stat:.3f}, p = {kw_p:.4f}

## 3. Predictive Model

Logistic regression: P(DIF_C) ~ difficulty + discrimination + n_items_in_subject

{coef_table[["variable","odds_ratio","or_ci_lower","or_ci_upper","p_value"]].to_csv(index=False)}

Pseudo R² = {model.prsquared:.4f}, N = {int(model.nobs)}

**Subject residuals (controlling difficulty + discrimination):**
- {len(excess_high)} subjects show significantly MORE DIF than expected (p < 0.05)
- {len(excess_low)} subjects show significantly LESS DIF than expected (p < 0.05)

Top excess-DIF subjects:
{excess_high.head(5)[["subject","domain","observed_c","expected_c","excess_c","binom_p"]].to_csv(index=False) if len(excess_high) > 0 else "None"}

## 4. Temporal Asymmetry

Overall: {overall_favor_2024}/{overall_favor_2024+overall_favor_2023} ({overall_favor_2024/(overall_favor_2024+overall_favor_2023)*100:.1f}%) C-items favor 2024 (focal) models.

**Subjects with unidirectional DIF (≥80% one direction, n≥5):**

{unidir[["subject","domain","n_c_items","pct_favor_2024","asym_binom_p"]].to_csv(index=False) if len(unidir) > 0 else "No subjects meet this criterion."}

## Key Findings

1. DIF is {'unevenly' if subj_df['pct_c'].std() > 5 else 'relatively evenly'} distributed across subjects (SD = {subj_df['pct_c'].std():.1f}%).
2. STEM vs Humanities difference is {sig_stem_hum} (p = {mw_p:.4f}).
3. After controlling for item difficulty and discrimination, {len(excess_high)} subjects show excess DIF — suggesting subject content itself contributes to DIF beyond psychometric properties.
4. {overall_favor_2024/(overall_favor_2024+overall_favor_2023)*100:.1f}% of C-flagged items favor 2024 models, with {len(unidir)} subjects showing near-unidirectional asymmetry.
"""

(OUT / "plan_033_report.md").write_text(report)

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)
for f in sorted(OUT.glob("*")):
    if f.is_file() and f.suffix != ".py":
        print(f"  {f.name}")

print("\nDone.")
