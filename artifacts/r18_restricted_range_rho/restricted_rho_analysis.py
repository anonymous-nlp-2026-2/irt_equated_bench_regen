"""R18: Restricted-range Spearman ρ for difficulty–DIF direction association.

Verifies that ρ(difficulty, delta_MH) ≈ -0.597 on DIF-C items is not
a ceiling/floor artifact by trimming extreme-accuracy items and re-testing.
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from pathlib import Path
from collections import OrderedDict

ROOT = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts")
OUT  = ROOT / "r18_restricted_range_rho"

# ── 1. Load DIF results, keep DIF-C ────────────────────────────────────
dif = pd.read_csv(ROOT / "plan_001/dif_results_temporal.csv")
dif_c = dif[dif["ets_class"] == "C"].copy()
print(f"DIF-C items: {len(dif_c)}")

# ── 2. Compute ref-cohort accuracy per item ─────────────────────────────
meta = pd.read_csv(ROOT / "plan_032_dif_cleaned_mmlu/cleaned_mmlu_analysis.csv")
ref_models = set(meta.loc[meta["cohort_temporal"] == "2023", "model_name"])

rm = np.load(ROOT / "response_matrix.npz")
X = sparse.csr_matrix((rm["data"], rm["indices"], rm["indptr"]), shape=tuple(rm["shape"]))
idx = np.load(ROOT / "response_matrix_index.npz", allow_pickle=True)
model_names = idx["model_names"]
item_ids    = idx["item_ids"]

ref_rows = np.array([i for i, m in enumerate(model_names) if m in ref_models])
X_ref = X[ref_rows].toarray()
ref_acc = X_ref.mean(axis=0)                       # shape (n_items,)
item_acc = pd.Series(ref_acc, index=item_ids)       # item_id → accuracy

# ── 3. Load IRT difficulty ──────────────────────────────────────────────
irt = pd.read_csv(ROOT / "_project/data/metabench_irt_params.csv")
irt_diff = irt.set_index("item")["diff"]            # item → IRT difficulty

# ── 4. Merge into DIF-C frame ──────────────────────────────────────────
dif_c["ref_acc"]  = dif_c["item_id"].map(item_acc)
dif_c["irt_diff"] = dif_c["item_id"].map(irt_diff)
dif_c = dif_c.dropna(subset=["ref_acc"])            # keep items w/ both
print(f"DIF-C with ref_acc: {len(dif_c)}")

# ── 5. Helper: compute ρ for a subset ──────────────────────────────────
def rho_report(df, label):
    n = len(df)
    if n < 5:
        return dict(label=label, n=n, rho_acc=np.nan, p_acc=np.nan,
                    rho_irt=np.nan, p_irt=np.nan,
                    pct_focal=np.nan)
    r1, p1 = stats.spearmanr(df["ref_acc"],  df["delta_mh"])
    sub = df.dropna(subset=["irt_diff"])
    if len(sub) >= 5:
        r2, p2 = stats.spearmanr(sub["irt_diff"], sub["delta_mh"])
    else:
        r2, p2 = np.nan, np.nan
    pct_focal = (df["delta_mh"] > 0).mean() * 100
    return dict(label=label, n=n,
                rho_acc=r1, p_acc=p1,
                rho_irt=r2, p_irt=p2,
                pct_focal=pct_focal)

# ── 6. Restricted-range analyses ────────────────────────────────────────
ranges = OrderedDict([
    ("Full (all DIF-C)",         (None,  None)),
    ("Trim ceiling (≤0.9)",      (None,  0.9)),
    ("Trim floor (≥0.1)",        (0.1,   None)),
    ("Interior [0.1, 0.9]",      (0.1,   0.9)),
    ("Interior [0.2, 0.8]",      (0.2,   0.8)),
    ("Interior [0.3, 0.7]",      (0.3,   0.7)),
])

rows = []
for label, (lo, hi) in ranges.items():
    mask = pd.Series(True, index=dif_c.index)
    if lo is not None:
        mask &= dif_c["ref_acc"] >= lo
    if hi is not None:
        mask &= dif_c["ref_acc"] <= hi
    rows.append(rho_report(dif_c[mask], label))

tbl = pd.DataFrame(rows)
print("\n=== Restricted-Range ρ ===")
print(tbl.to_string(index=False, float_format="{:.4f}".format))

# ── 7. Decile analysis ──────────────────────────────────────────────────
dif_c["diff_decile"] = pd.qcut(dif_c["ref_acc"], 10, labels=False, duplicates="drop")
dec = (dif_c.groupby("diff_decile")
       .agg(n=("delta_mh", "size"),
            mean_acc=("ref_acc", "mean"),
            pct_focal=("delta_mh", lambda s: (s > 0).mean() * 100),
            mean_delta=("delta_mh", "mean"))
       .reset_index())
print("\n=== Decile Breakdown ===")
print(dec.to_string(index=False, float_format="{:.3f}".format))

# Interior deciles (drop first and last)
interior_deciles = dec[(dec["diff_decile"] >= 1) & (dec["diff_decile"] <= 8)]
interior_items = dif_c[(dif_c["diff_decile"] >= 1) & (dif_c["diff_decile"] <= 8)]
rho_interior_dec = rho_report(interior_items, "Interior deciles 1-8")
print(f"\nInterior deciles 1-8: ρ_acc={rho_interior_dec['rho_acc']:.4f}, "
      f"p={rho_interior_dec['p_acc']:.2e}, n={rho_interior_dec['n']}")

# ── 8. Bootstrap CI for the full ρ and key restricted ρ ─────────────────
rng = np.random.default_rng(42)
def bootstrap_rho(x, y, n_boot=10000):
    rhos = np.empty(n_boot)
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rhos[b], _ = stats.spearmanr(x[idx], y[idx])
    return np.percentile(rhos, [2.5, 97.5])

boot_keys = [
    ("Full",           (None,  None)),
    ("Interior [0.1, 0.9]", (0.1, 0.9)),
    ("Interior [0.2, 0.8]", (0.2, 0.8)),
]
print("\n=== Bootstrap 95% CI ===")
for label, (lo, hi) in boot_keys:
    mask = pd.Series(True, index=dif_c.index)
    if lo is not None:
        mask &= dif_c["ref_acc"] >= lo
    if hi is not None:
        mask &= dif_c["ref_acc"] <= hi
    sub = dif_c[mask]
    ci = bootstrap_rho(sub["ref_acc"].values, sub["delta_mh"].values)
    r, p = stats.spearmanr(sub["ref_acc"], sub["delta_mh"])
    print(f"  {label}: ρ={r:.4f}  95%CI [{ci[0]:.4f}, {ci[1]:.4f}]  n={len(sub)}")

# ── 9. Write report ────────────────────────────────────────────────────
report = []
report.append("# R18: Restricted-Range ρ Analysis\n")
report.append("## Motivation\n")
report.append("Verify that Spearman ρ(difficulty, δ_MH) ≈ −0.597 among DIF-C items ")
report.append("is not an artifact of ceiling/floor effects. If the correlation survives ")
report.append("after trimming extreme-accuracy items, the difficulty–direction association ")
report.append("is substantive, not mechanical.\n\n")

report.append("## Restricted-Range Results\n\n")
report.append("| Range | N | ρ (ref-acc) | p | ρ (IRT-diff) | p | % focal-fav |\n")
report.append("|-------|---|------------|---|-------------|---|-------------|\n")
for _, r in tbl.iterrows():
    report.append(f"| {r['label']} | {r['n']:.0f} | {r['rho_acc']:.4f} | {r['p_acc']:.2e} | "
                  f"{r['rho_irt']:.4f} | {r['p_irt']:.2e} | {r['pct_focal']:.1f}% |\n")

report.append("\n## Decile Breakdown\n\n")
report.append("| Decile | N | Mean acc | % focal-fav | Mean δ_MH |\n")
report.append("|--------|---|----------|------------|----------|\n")
for _, d in dec.iterrows():
    report.append(f"| {d['diff_decile']:.0f} | {d['n']:.0f} | {d['mean_acc']:.3f} | "
                  f"{d['pct_focal']:.1f}% | {d['mean_delta']:.3f} |\n")

report.append(f"\nInterior deciles (1–8): ρ = {rho_interior_dec['rho_acc']:.4f}, "
              f"p = {rho_interior_dec['p_acc']:.2e}, N = {rho_interior_dec['n']}\n")

report.append("\n## Bootstrap 95% CIs\n\n")
report.append("| Range | ρ | 95% CI | N |\n")
report.append("|-------|---|--------|---|\n")
for label, (lo, hi) in boot_keys:
    mask = pd.Series(True, index=dif_c.index)
    if lo is not None:
        mask &= dif_c["ref_acc"] >= lo
    if hi is not None:
        mask &= dif_c["ref_acc"] <= hi
    sub = dif_c[mask]
    r, p = stats.spearmanr(sub["ref_acc"], sub["delta_mh"])
    ci = bootstrap_rho(sub["ref_acc"].values, sub["delta_mh"].values)
    report.append(f"| {label} | {r:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {len(sub)} |\n")

report.append("\n## Conclusion\n\n")

# Auto-generate conclusion based on results
full_rho = tbl.iloc[0]["rho_acc"]
interior_rho = tbl[tbl["label"] == "Interior [0.1, 0.9]"].iloc[0]["rho_acc"]
interior_p   = tbl[tbl["label"] == "Interior [0.1, 0.9]"].iloc[0]["p_acc"]
strict_rho   = tbl[tbl["label"] == "Interior [0.2, 0.8]"].iloc[0]["rho_acc"]
strict_p     = tbl[tbl["label"] == "Interior [0.2, 0.8]"].iloc[0]["p_acc"]

if interior_p < 0.05 and strict_p < 0.05:
    report.append(f"The full-sample ρ = {full_rho:.3f} attenuates to {interior_rho:.3f} "
                  f"(interior [0.1, 0.9]) and {strict_rho:.3f} (interior [0.2, 0.8]), "
                  f"but remains statistically significant in both cases "
                  f"(p = {interior_p:.2e} and p = {strict_p:.2e}). "
                  f"**Ceiling/floor effects alone cannot explain the difficulty–direction association.** "
                  f"The correlation is substantive.\n")
elif interior_p < 0.05:
    report.append(f"The full-sample ρ = {full_rho:.3f} attenuates to {interior_rho:.3f} "
                  f"(interior [0.1, 0.9], p = {interior_p:.2e}) which remains significant, "
                  f"though the stricter [0.2, 0.8] range (ρ = {strict_rho:.3f}, p = {strict_p:.2e}) "
                  f"loses significance. Ceiling/floor effects contribute but do not fully explain the association.\n")
else:
    report.append(f"The full-sample ρ = {full_rho:.3f} does not survive restriction. "
                  f"Ceiling/floor effects may substantially drive the observed correlation.\n")

(OUT / "restricted_rho_report.md").write_text("".join(report))
print(f"\nReport written to {OUT / 'restricted_rho_report.md'}")
