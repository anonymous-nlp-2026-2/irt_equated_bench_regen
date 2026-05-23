"""R16-SF2: Compute uniform + non-uniform DIF union/intersection sizes."""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "artifacts/r16_dif_union"

# Load data
mh = pd.read_csv(ROOT / "artifacts/plan_001/dif_results_temporal.csv")
lr = pd.read_csv(ROOT / "artifacts/plan_021/lr_dif_item_results.csv")
sel = pd.read_csv(ROOT / "artifacts/plan_021/selected_subjects.csv")

print(f"MH-DIF: {len(mh)} rows, {mh['subject'].nunique()} subjects")
print(f"LR-DIF: {len(lr)} rows, {lr['subject'].nunique()} subjects")
print(f"Selected subjects: {sel['subject'].tolist()}")

# Merge on (item_id, subject) — restrict to LR-DIF coverage
merged = lr.merge(mh[['item_id', 'subject', 'ets_class']], on=['item_id', 'subject'], how='left', suffixes=('_lr', '_mh'))
print(f"\nMerged: {len(merged)} rows")
print(f"ets_class from MH in merged: {merged['ets_class'].value_counts().to_dict()}")

# Check mh_class in LR file vs ets_class from MH file
print(f"\nmh_class (from LR file): {lr['mh_class'].value_counts().to_dict()}")

# Define DIF flags
merged['is_uniform_c'] = merged['ets_class'] == 'C'
merged['is_nonuniform_sig'] = merged['nonuniform_dif'] == True
merged['is_nonuniform_effect'] = merged['beta_interaction'].abs() >= 0.5

print(f"\n=== DIF Counts (N={len(merged)}) ===")
print(f"Uniform DIF (ETS C): {merged['is_uniform_c'].sum()} ({merged['is_uniform_c'].mean()*100:.1f}%)")
print(f"Non-uniform DIF (BH sig): {merged['is_nonuniform_sig'].sum()} ({merged['is_nonuniform_sig'].mean()*100:.1f}%)")
print(f"Non-uniform DIF (|β| ≥ 0.5): {merged['is_nonuniform_effect'].sum()} ({merged['is_nonuniform_effect'].mean()*100:.1f}%)")

# Union/Intersection analysis for both non-uniform criteria
for label, nu_col in [("significance (BH p<0.05)", "is_nonuniform_sig"),
                       ("effect size (|β_int| ≥ 0.5)", "is_nonuniform_effect")]:
    u = merged['is_uniform_c']
    n = merged[nu_col]

    both = (u & n).sum()
    only_u = (u & ~n).sum()
    only_n = (~u & n).sum()
    neither = (~u & ~n).sum()
    union = (u | n).sum()
    jaccard = both / union if union > 0 else 0

    print(f"\n--- Non-uniform criterion: {label} ---")
    print(f"Only Uniform:     {only_u} ({only_u/len(merged)*100:.1f}%)")
    print(f"Only Non-uniform: {only_n} ({only_n/len(merged)*100:.1f}%)")
    print(f"Both:             {both} ({both/len(merged)*100:.1f}%)")
    print(f"Neither:          {neither} ({neither/len(merged)*100:.1f}%)")
    print(f"Union:            {union} ({union/len(merged)*100:.1f}%)")
    print(f"Jaccard:          {jaccard:.3f}")

# Extrapolation to full 12508 items
total_mh = len(mh)
for label, nu_col in [("sig", "is_nonuniform_sig"), ("effect", "is_nonuniform_effect")]:
    union_pct = ((merged['is_uniform_c'] | merged[nu_col]).sum()) / len(merged)
    est_union = union_pct * total_mh
    print(f"\nExtrapolation ({label}): {union_pct*100:.1f}% in 9 subjects → ~{est_union:.0f} of {total_mh} items")

# Per-item CSV
out_df = merged[['item_id', 'subject', 'is_uniform_c', 'is_nonuniform_sig', 'is_nonuniform_effect']].copy()
out_df['in_union_sig'] = out_df['is_uniform_c'] | out_df['is_nonuniform_sig']
out_df['in_union_effect'] = out_df['is_uniform_c'] | out_df['is_nonuniform_effect']
out_df.to_csv(OUT / "dif_union_results.csv", index=False)
print(f"\nSaved per-item CSV: {OUT / 'dif_union_results.csv'} ({len(out_df)} rows)")

# Subject-level breakdown
print("\n=== Subject-level breakdown ===")
for subj in sorted(merged['subject'].unique()):
    s = merged[merged['subject'] == subj]
    n_total = len(s)
    n_uc = s['is_uniform_c'].sum()
    n_ns = s['is_nonuniform_sig'].sum()
    n_ne = s['is_nonuniform_effect'].sum()
    n_union_s = (s['is_uniform_c'] | s['is_nonuniform_sig']).sum()
    n_union_e = (s['is_uniform_c'] | s['is_nonuniform_effect']).sum()
    print(f"  {subj:25s}: N={n_total:3d} | UC={n_uc:3d} | NU_sig={n_ns:3d} | NU_eff={n_ne:3d} | Union_sig={n_union_s:3d} ({n_union_s/n_total*100:.0f}%) | Union_eff={n_union_e:3d} ({n_union_e/n_total*100:.0f}%)")

# beta_interaction distribution
print(f"\n=== beta_interaction distribution ===")
bi = merged['beta_interaction'].dropna()
print(f"  mean={bi.mean():.4f}, median={bi.median():.4f}, std={bi.std():.4f}")
print(f"  |β_int| percentiles: 50th={bi.abs().quantile(0.5):.4f}, 75th={bi.abs().quantile(0.75):.4f}, 90th={bi.abs().quantile(0.9):.4f}, 95th={bi.abs().quantile(0.95):.4f}")
print(f"  |β_int| ≥ 0.3: {(bi.abs() >= 0.3).sum()} ({(bi.abs() >= 0.3).mean()*100:.1f}%)")
print(f"  |β_int| ≥ 0.5: {(bi.abs() >= 0.5).sum()} ({(bi.abs() >= 0.5).mean()*100:.1f}%)")
print(f"  |β_int| ≥ 1.0: {(bi.abs() >= 1.0).sum()} ({(bi.abs() >= 1.0).mean()*100:.1f}%)")
