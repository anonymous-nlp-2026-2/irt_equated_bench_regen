"""R14 MF1: Domain-Level Ability Profile Analysis.

Compares 2023 vs 2024 cohort accuracy across MMLU domains.
If all domain-level Cohen's d < 0.3 → profile shift is flat,
ruling out multidimensional spurious DIF as a confound.
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "r14_mf1_domain_profile"
OUT.mkdir(parents=True, exist_ok=True)

# --- MMLU Subject → Domain mapping ---
DOMAIN_MAP = {}
for s in [
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "machine_learning", "medical_genetics", "virology",
]:
    DOMAIN_MAP[s] = "STEM"

for s in [
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
    "prehistory", "world_religions",
]:
    DOMAIN_MAP[s] = "Humanities"

for s in [
    "econometrics", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "high_school_psychology",
    "human_sexuality", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy",
]:
    DOMAIN_MAP[s] = "Social Science"

for s in [
    "business_ethics", "clinical_knowledge", "college_medicine",
    "global_facts", "human_aging", "management", "marketing",
    "miscellaneous", "nutrition", "professional_accounting",
    "professional_law", "professional_medicine",
]:
    DOMAIN_MAP[s] = "Other"

# --- Load data ---
print("Loading response matrix...")
X = sparse.load_npz(ROOT / "artifacts" / "response_matrix.npz")
idx = np.load(ROOT / "artifacts" / "response_matrix_index.npz", allow_pickle=True)
model_names = idx["model_names"]
item_ids = idx["item_ids"]

meta = pd.read_csv(ROOT / "artifacts" / "plan_032_dif_cleaned_mmlu" / "cleaned_mmlu_analysis.csv")

# --- Build cohort masks ---
model_to_idx = {name: i for i, name in enumerate(model_names)}
meta_2023 = meta[meta["cohort_temporal"] == "2023"]
meta_2024 = meta[meta["cohort_temporal"] == "2024"]

idx_2023 = np.array([model_to_idx[m] for m in meta_2023["model_name"] if m in model_to_idx])
idx_2024 = np.array([model_to_idx[m] for m in meta_2024["model_name"] if m in model_to_idx])
print(f"Cohort sizes: 2023={len(idx_2023)}, 2024={len(idx_2024)}")

# --- Parse item subjects and map to domains ---
item_subjects = np.array([str(iid).rsplit(".", 1)[0] for iid in item_ids])
unique_subjects = sorted(set(item_subjects))
print(f"Found {len(unique_subjects)} unique subjects")

unmapped = [s for s in unique_subjects if s not in DOMAIN_MAP]
if unmapped:
    print(f"WARNING: unmapped subjects (assigned to Other): {unmapped}")
    for s in unmapped:
        DOMAIN_MAP[s] = "Other"

item_domains = np.array([DOMAIN_MAP[s] for s in item_subjects])

# --- Convert to dense for cohort subsets (memory ~OK for 1887 × 12508 int8) ---
print("Extracting cohort submatrices...")
X_2023 = X[idx_2023].toarray().astype(np.float32)  # (1080, 12508)
X_2024 = X[idx_2024].toarray().astype(np.float32)  # (807, 12508)

# --- Per-domain analysis ---
domains = ["STEM", "Humanities", "Social Science", "Other"]
results = []

for domain in domains:
    col_mask = item_domains == domain
    n_items = col_mask.sum()
    n_subjects = len(set(item_subjects[col_mask]))

    acc_2023 = X_2023[:, col_mask].mean(axis=1)  # per-model accuracy
    acc_2024 = X_2024[:, col_mask].mean(axis=1)

    mean_2023 = acc_2023.mean()
    mean_2024 = acc_2024.mean()
    sd_2023 = acc_2023.std(ddof=1)
    sd_2024 = acc_2024.std(ddof=1)

    # Pooled SD for Cohen's d
    n1, n2 = len(acc_2023), len(acc_2024)
    pooled_sd = np.sqrt(((n1 - 1) * sd_2023**2 + (n2 - 1) * sd_2024**2) / (n1 + n2 - 2))
    d = (mean_2024 - mean_2023) / pooled_sd if pooled_sd > 0 else 0.0

    # 95% CI for d using standard formula
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    d_ci_lower = d - 1.96 * se_d
    d_ci_upper = d + 1.96 * se_d

    results.append({
        "domain": domain,
        "n_items": int(n_items),
        "n_subjects": int(n_subjects),
        "mean_acc_2023": round(float(mean_2023), 4),
        "sd_acc_2023": round(float(sd_2023), 4),
        "mean_acc_2024": round(float(mean_2024), 4),
        "sd_acc_2024": round(float(sd_2024), 4),
        "cohens_d": round(float(d), 4),
        "d_ci_lower": round(float(d_ci_lower), 4),
        "d_ci_upper": round(float(d_ci_upper), 4),
    })
    print(f"  {domain:16s}: d={d:+.4f} [{d_ci_lower:.4f}, {d_ci_upper:.4f}]  "
          f"acc_2023={mean_2023:.4f}  acc_2024={mean_2024:.4f}  n_items={n_items}")

# --- Overall ---
acc_2023_all = X_2023.mean(axis=1)
acc_2024_all = X_2024.mean(axis=1)
mean_all_2023 = acc_2023_all.mean()
mean_all_2024 = acc_2024_all.mean()
sd_all_2023 = acc_2023_all.std(ddof=1)
sd_all_2024 = acc_2024_all.std(ddof=1)
n1, n2 = len(acc_2023_all), len(acc_2024_all)
pooled_sd_all = np.sqrt(((n1 - 1) * sd_all_2023**2 + (n2 - 1) * sd_all_2024**2) / (n1 + n2 - 2))
d_all = (mean_all_2024 - mean_all_2023) / pooled_sd_all
se_d_all = np.sqrt((n1 + n2) / (n1 * n2) + d_all**2 / (2 * (n1 + n2)))

results.append({
    "domain": "Overall",
    "n_items": int(len(item_ids)),
    "n_subjects": int(len(unique_subjects)),
    "mean_acc_2023": round(float(mean_all_2023), 4),
    "sd_acc_2023": round(float(sd_all_2023), 4),
    "mean_acc_2024": round(float(mean_all_2024), 4),
    "sd_acc_2024": round(float(sd_all_2024), 4),
    "cohens_d": round(float(d_all), 4),
    "d_ci_lower": round(float(d_all - 1.96 * se_d_all), 4),
    "d_ci_upper": round(float(d_all + 1.96 * se_d_all), 4),
})
print(f"\n  {'Overall':16s}: d={d_all:+.4f} [{d_all - 1.96*se_d_all:.4f}, {d_all + 1.96*se_d_all:.4f}]  "
      f"acc_2023={mean_all_2023:.4f}  acc_2024={mean_all_2024:.4f}")

# --- Save CSV ---
df = pd.DataFrame(results)
df.to_csv(OUT / "domain_profile_results.csv", index=False)
print(f"\nSaved: {OUT / 'domain_profile_results.csv'}")

# --- Profile flatness ---
domain_ds = [r["cohens_d"] for r in results if r["domain"] != "Overall"]
d_range = max(domain_ds) - min(domain_ds)
d_max = max(abs(d) for d in domain_ds)

# --- Per-subject Cohen's d for supplementary detail ---
subject_results = []
for subj in sorted(unique_subjects):
    col_mask = item_subjects == subj
    n_items = col_mask.sum()
    if n_items < 5:
        continue
    acc_2023 = X_2023[:, col_mask].mean(axis=1)
    acc_2024 = X_2024[:, col_mask].mean(axis=1)
    m1, m2 = acc_2023.mean(), acc_2024.mean()
    s1, s2 = acc_2023.std(ddof=1), acc_2024.std(ddof=1)
    n1, n2 = len(acc_2023), len(acc_2024)
    ps = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    d = (m2 - m1) / ps if ps > 0 else 0.0
    subject_results.append({
        "subject": subj,
        "domain": DOMAIN_MAP[subj],
        "n_items": int(n_items),
        "mean_acc_2023": round(float(m1), 4),
        "mean_acc_2024": round(float(m2), 4),
        "cohens_d": round(float(d), 4),
    })

df_subj = pd.DataFrame(subject_results)
df_subj.to_csv(OUT / "subject_profile_results.csv", index=False)
print(f"Saved: {OUT / 'subject_profile_results.csv'}")

# --- Generate report ---
report_lines = []
report_lines.append("# R14 MF1: Domain-Level Ability Profile Analysis\n")
report_lines.append("## Purpose\n")
report_lines.append(
    "Quantify whether 2023→2024 cohort ability differences are uniform across MMLU domains "
    "(STEM, Humanities, Social Science, Other). If all domain-level Cohen's d < 0.3 and the "
    "profile is flat, multidimensional ability shifts cannot produce spurious DIF.\n"
)

report_lines.append("## Cohort Sizes\n")
report_lines.append(f"- 2023 (Reference): {len(idx_2023)} models")
report_lines.append(f"- 2024 (Focal): {len(idx_2024)} models\n")

report_lines.append("## Domain-Level Results\n")
report_lines.append("| Domain | Items | Subjects | Acc (2023) | Acc (2024) | Cohen's d | 95% CI |")
report_lines.append("|--------|------:|:--------:|:----------:|:----------:|:---------:|:------:|")
for r in results:
    ci = f"[{r['d_ci_lower']:.3f}, {r['d_ci_upper']:.3f}]"
    report_lines.append(
        f"| {r['domain']} | {r['n_items']} | {r['n_subjects']} | "
        f"{r['mean_acc_2023']:.3f} | {r['mean_acc_2024']:.3f} | "
        f"{r['cohens_d']:+.3f} | {ci} |"
    )
report_lines.append("")

report_lines.append("## Profile Flatness\n")
report_lines.append(f"- Max |d| across domains: {d_max:.4f}")
report_lines.append(f"- Range of d across domains: {d_range:.4f}")
report_lines.append(f"- Overall d: {d_all:.4f}\n")

all_below_03 = all(abs(d) < 0.3 for d in domain_ds)
report_lines.append("## Interpretation\n")
if all_below_03:
    report_lines.append(
        f"All domain-level Cohen's d values fall below the 0.3 threshold "
        f"(max |d| = {d_max:.3f}). The inter-domain range is only {d_range:.3f}, "
        f"indicating a nearly flat ability profile shift. This rules out "
        f"multidimensional ability confounds as a source of spurious DIF: "
        f"the 2024 cohort is uniformly slightly {'stronger' if d_all > 0 else 'weaker'} "
        f"across all domains, rather than differentially strong in specific areas."
    )
else:
    report_lines.append(
        f"Some domain-level Cohen's d values exceed 0.3 (max |d| = {d_max:.3f}). "
        f"This suggests non-uniform ability shifts across domains, which could "
        f"contribute to differential item functioning through multidimensional confounds."
    )

report_lines.append("")
report_lines.append("## Per-Subject Detail (top 10 largest |d|)\n")
df_subj_sorted = df_subj.reindex(df_subj["cohens_d"].abs().sort_values(ascending=False).index)
report_lines.append("| Subject | Domain | Items | d |")
report_lines.append("|---------|--------|------:|:---:|")
for _, row in df_subj_sorted.head(10).iterrows():
    report_lines.append(f"| {row['subject']} | {row['domain']} | {row['n_items']} | {row['cohens_d']:+.3f} |")
report_lines.append("")

report = "\n".join(report_lines)
(OUT / "domain_profile_analysis.md").write_text(report)
print(f"Saved: {OUT / 'domain_profile_analysis.md'}")
print("\n--- REPORT ---")
print(report)
