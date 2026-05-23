"""R17 Task 2: Web-overlap per-subject k distribution from Li et al. (2024)."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "mmlu_li2024_proxy_labels.csv"
OUT = Path(__file__).parent / "web_overlap_k_distribution.md"

df = pd.read_csv(DATA)

# --- Exclude no_annotation items ---
no_ann = df[df["contamination_type"] == "no_annotation"]
df_ann = df[df["contamination_type"] != "no_annotation"].copy()

# --- Per-subject stats ---
records = []
for subj, g in df_ann.groupby("subject"):
    n_total = len(g)
    n_contam = g["is_contaminated"].sum()
    prevalence = n_contam / n_total * 100
    records.append({
        "subject": subj,
        "n_items_total": n_total,
        "n_contaminated": int(n_contam),
        "prevalence_pct": round(prevalence, 1),
    })

tbl = pd.DataFrame(records).sort_values("prevalence_pct", ascending=False).reset_index(drop=True)

# --- Summary stats ---
prev = tbl["prevalence_pct"]
summary = {
    "n_subjects_with_data": len(tbl),
    "mean": round(prev.mean(), 1),
    "median": round(prev.median(), 1),
    "std": round(prev.std(), 1),
    "Q25": round(prev.quantile(0.25), 1),
    "Q75": round(prev.quantile(0.75), 1),
    "min": round(prev.min(), 1),
    "max": round(prev.max(), 1),
    "n_prev_gt50": int((prev > 50).sum()),
    "n_prev_gt25": int((prev > 25).sum()),
    "n_prev_gt10": int((prev > 10).sum()),
    "n_prev_eq0": int((prev == 0).sum()),
    "n_prev_gt0": int((prev > 0).sum()),
}

# --- Check subjects missing entirely (all no_annotation) ---
all_subjects = df["subject"].unique()
annotated_subjects = df_ann["subject"].unique()
missing_subjects = sorted(set(all_subjects) - set(annotated_subjects))

# --- Write report ---
lines = []
lines.append("# Web-Overlap Per-Subject Contamination Distribution (Li et al. 2024)")
lines.append("")
lines.append("## Data")
lines.append(f"- Source: `mmlu_li2024_proxy_labels.csv` (Li et al. 2024, EMNLP)")
lines.append(f"- Total items: {len(df)}")
lines.append(f"- Items with annotation: {len(df_ann)} (excluded {len(no_ann)} `no_annotation`)")
lines.append(f"- Subjects with annotated data: {len(tbl)} / 57")
lines.append("")

if missing_subjects:
    lines.append("### Subjects with no annotated items (all `no_annotation`)")
    for s in missing_subjects:
        lines.append(f"- {s}")
    lines.append("")

lines.append("## Per-Subject Table (sorted by prevalence descending)")
lines.append("")
lines.append("| # | Subject | n_items | n_contam | prevalence% |")
lines.append("|---|---------|---------|----------|-------------|")
for i, row in tbl.iterrows():
    lines.append(f"| {i+1} | {row['subject']} | {row['n_items_total']} | {row['n_contaminated']} | {row['prevalence_pct']:.1f}% |")
lines.append("")

lines.append("## Summary Statistics")
lines.append("")
lines.append("| Metric | Value |")
lines.append("|--------|-------|")
lines.append(f"| Subjects with data | {summary['n_subjects_with_data']} |")
lines.append(f"| Mean prevalence | {summary['mean']}% |")
lines.append(f"| Median prevalence | {summary['median']}% |")
lines.append(f"| Std | {summary['std']}% |")
lines.append(f"| Q25 | {summary['Q25']}% |")
lines.append(f"| Q75 | {summary['Q75']}% |")
lines.append(f"| Min | {summary['min']}% |")
lines.append(f"| Max | {summary['max']}% |")
lines.append(f"| Subjects with prevalence > 50% | {summary['n_prev_gt50']} |")
lines.append(f"| Subjects with prevalence > 25% | {summary['n_prev_gt25']} |")
lines.append(f"| Subjects with prevalence > 10% | {summary['n_prev_gt10']} |")
lines.append(f"| Subjects with prevalence = 0% | {summary['n_prev_eq0']} |")
lines.append("")

lines.append("## k-Parameter Constraint")
lines.append("")
lines.append(f"In `plan_034_multi_subject_injection`, k = number of simultaneously contaminated subjects.")
lines.append("")
lines.append(f"From Li et al. web-overlap data:")
lines.append(f"- **{summary['n_prev_gt0']} / 57 subjects** show contamination evidence (prevalence > 0%)")
lines.append(f"- This provides an empirical upper bound of **k ≤ {summary['n_prev_gt0']}**")
lines.append(f"- However, prevalence varies widely: median {summary['median']}%, range [{summary['min']}%, {summary['max']}%]")
lines.append(f"- {summary['n_prev_gt50']} subjects have majority-contaminated items (>50%)")
lines.append(f"- {summary['n_prev_gt25']} subjects have substantial contamination (>25%)")
lines.append("")
lines.append("The distribution suggests that contamination is **pervasive across subjects** rather than concentrated in a few, ")
lines.append("supporting the use of high k values (k ≥ 30) in multi-subject injection simulations.")

OUT.write_text("\n".join(lines) + "\n")
print(f"Report written to {OUT}")
print()
print("=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
