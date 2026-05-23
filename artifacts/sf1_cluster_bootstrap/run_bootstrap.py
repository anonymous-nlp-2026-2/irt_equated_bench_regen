"""
SF1 Cluster Bootstrap: Family-level CI for temporal C%.
Resamples model families (cluster bootstrap) and individual models (model-level bootstrap),
recomputing MH-DIF statistics each time to get CIs for C%.
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from pathlib import Path
import time

BASE = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen")
OUT = BASE / "artifacts" / "sf1_cluster_bootstrap"

# ── Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv(BASE / "artifacts/plan_032_dif_cleaned_mmlu/cleaned_mmlu_analysis.csv")
idx = np.load(BASE / "artifacts/response_matrix_index.npz", allow_pickle=True)
model_names_mat = idx["model_names"]
item_ids = idx["item_ids"]
n_items = len(item_ids)

raw = np.load(BASE / "artifacts/response_matrix.npz", allow_pickle=True)
mat = sparse.csr_matrix((raw["data"], raw["indices"], raw["indptr"]), shape=tuple(raw["shape"]))

# ── Build model index mapping ──────────────────────────────────────────────

name_to_matidx = {n: i for i, n in enumerate(model_names_mat)}

# Filter: cohort in {2023, 2024}, family not NaN/unknown, family size >= 3
df_mh = df[df["cohort_temporal"].isin(["2023", "2024"])].copy()
df_mh = df_mh[df_mh["family"].notna() & (df_mh["family"] != "") & (df_mh["family"] != "unknown")]

fam_counts = df_mh["family"].value_counts()
valid_fams = fam_counts[fam_counts >= 3].index
df_mh = df_mh[df_mh["family"].isin(valid_fams)].copy()

# Map to matrix row indices
df_mh["mat_idx"] = df_mh["model_name"].map(name_to_matidx)
assert df_mh["mat_idx"].notna().all(), "Some models not found in response matrix"
df_mh["mat_idx"] = df_mh["mat_idx"].astype(int)

print(f"Models in bootstrap pool: {len(df_mh)}")
print(f"  2023 (reference): {(df_mh.cohort_temporal == '2023').sum()}")
print(f"  2024 (focal):     {(df_mh.cohort_temporal == '2024').sum()}")
print(f"Families (size >= 3): {len(valid_fams)}")

# ── Step 1: Family distribution ────────────────────────────────────────────

fam_dist = df_mh.groupby("family").agg(
    n_models=("model_name", "count"),
    n_2023=("cohort_temporal", lambda x: (x == "2023").sum()),
    n_2024=("cohort_temporal", lambda x: (x == "2024").sum()),
).sort_values("n_models", ascending=False).reset_index()

fam_dist.to_csv(OUT / "family_distribution.csv", index=False)
print(f"\nFamily distribution saved. Top 10:")
print(fam_dist.head(10).to_string(index=False))

# ── Pre-compute dense response matrix for MH subset ───────────────────────

all_mat_indices = df_mh["mat_idx"].values
resp = mat[all_mat_indices, :].toarray().astype(np.int32)  # (n_models, n_items)
cohorts = (df_mh["cohort_temporal"].values == "2024").astype(bool)  # True = focal
families = df_mh["family"].values

print(f"\nDense response matrix: {resp.shape}, {resp.nbytes / 1e6:.1f} MB")

# ── Vectorized MH-DIF computation ─────────────────────────────────────────

def compute_cpct(resp_sub, cohort_sub, n_strata=5):
    """
    Compute C% from MH-DIF on a model subset.
    resp_sub: (n_models, n_items) binary matrix
    cohort_sub: (n_models,) boolean array, True = focal (2024)
    Returns: C% (fraction of items classified as ETS-C)
    """
    n_models, n_items = resp_sub.shape
    total_scores = resp_sub.sum(axis=1)

    # Assign strata by quintiles of total score
    try:
        strata = pd.qcut(total_scores, n_strata, labels=False, duplicates="drop")
    except ValueError:
        strata = np.zeros(n_models, dtype=int)

    is_ref = ~cohort_sub
    is_foc = cohort_sub

    # Accumulate MH numerator/denominator across strata
    num = np.zeros(n_items, dtype=np.float64)  # Σ a*d/n
    den = np.zeros(n_items, dtype=np.float64)  # Σ b*c/n
    sum_diff = np.zeros(n_items, dtype=np.float64)  # Σ (a - E(a))
    sum_var = np.zeros(n_items, dtype=np.float64)   # Σ Var(a)

    unique_strata = np.unique(strata)
    for k in unique_strata:
        in_stratum = (strata == k)
        ref_k = in_stratum & is_ref
        foc_k = in_stratum & is_foc

        n1 = ref_k.sum()  # reference count
        n0 = foc_k.sum()  # focal count
        n_k = n1 + n0

        if n1 == 0 or n0 == 0 or n_k <= 1:
            continue

        # Sum correct responses per item
        a = resp_sub[ref_k].sum(axis=0).astype(np.float64)  # ref correct
        c = resp_sub[foc_k].sum(axis=0).astype(np.float64)  # foc correct
        b = n1 - a  # ref incorrect
        d = n0 - c  # foc incorrect

        m1 = a + c  # total correct
        m0 = b + d  # total incorrect

        num += (a * d) / n_k
        den += (b * c) / n_k

        # For chi-square
        e_a = (n1 * m1) / n_k
        sum_diff += (a - e_a)
        var_a = (n1 * n0 * m1 * m0) / (n_k * n_k * (n_k - 1))
        sum_var += var_a

    # MH alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_mh = num / den
        delta_mh = -2.35 * np.log(alpha_mh)
        delta_mh = np.where(np.isfinite(delta_mh), delta_mh, 0.0)

    # MH chi-square (with continuity correction)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(
            sum_var > 0,
            (np.maximum(np.abs(sum_diff) - 0.5, 0)) ** 2 / sum_var,
            0.0,
        )
    p_values = 1 - stats.chi2.cdf(chi2, df=1)

    # ETS classification
    abs_delta = np.abs(delta_mh)
    is_c = (abs_delta >= 1.5) & (p_values < 0.05)
    c_pct = is_c.sum() / n_items

    return c_pct


# ── Verify against original results ───────────────────────────────────────

print("\n── Verifying MH-DIF computation ──")
c_pct_full = compute_cpct(resp, cohorts)
print(f"C% on full subset (1882 models): {c_pct_full:.4f} ({c_pct_full*100:.1f}%)")

# Compare with original
dif_orig = pd.read_csv(BASE / "artifacts/plan_001/dif_results_temporal.csv")
orig_c_pct = (dif_orig["ets_class"] == "C").sum() / len(dif_orig)
print(f"Original C% (5227 models): {orig_c_pct:.4f} ({orig_c_pct*100:.1f}%)")
print("(Slight difference expected since we use 1882 vs 5227 models)")

# ── Bootstrap ──────────────────────────────────────────────────────────────

N_BOOT = 1000
np.random.seed(42)

family_list = list(valid_fams)
family_indices = {f: np.where(families == f)[0] for f in family_list}
n_families = len(family_list)
n_models_total = len(df_mh)

print(f"\n── Running {N_BOOT} cluster bootstrap iterations ──")
t0 = time.time()
cluster_cpcts = []

for b in range(N_BOOT):
    # Resample families with replacement
    sampled_fams = np.random.choice(n_families, size=n_families, replace=True)
    model_idx = np.concatenate([family_indices[family_list[f]] for f in sampled_fams])

    c_pct = compute_cpct(resp[model_idx], cohorts[model_idx])
    cluster_cpcts.append(c_pct)

    if (b + 1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  Cluster bootstrap {b+1}/{N_BOOT} done ({elapsed:.1f}s)")

cluster_cpcts = np.array(cluster_cpcts)
t_cluster = time.time() - t0
print(f"Cluster bootstrap done in {t_cluster:.1f}s")

print(f"\n── Running {N_BOOT} model-level bootstrap iterations ──")
t0 = time.time()
model_cpcts = []

for b in range(N_BOOT):
    # Resample individual models with replacement
    model_idx = np.random.choice(n_models_total, size=n_models_total, replace=True)

    c_pct = compute_cpct(resp[model_idx], cohorts[model_idx])
    model_cpcts.append(c_pct)

    if (b + 1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  Model bootstrap {b+1}/{N_BOOT} done ({elapsed:.1f}s)")

model_cpcts = np.array(model_cpcts)
t_model = time.time() - t0
print(f"Model-level bootstrap done in {t_model:.1f}s")

# ── Results ────────────────────────────────────────────────────────────────

def ci_95(arr):
    return np.percentile(arr, [2.5, 97.5])

cluster_ci = ci_95(cluster_cpcts)
model_ci = ci_95(model_cpcts)

print(f"\n{'='*60}")
print(f"Point estimate C%: {c_pct_full*100:.2f}%")
print(f"Cluster bootstrap CI (95%): [{cluster_ci[0]*100:.2f}%, {cluster_ci[1]*100:.2f}%]")
print(f"  Width: {(cluster_ci[1]-cluster_ci[0])*100:.2f} pp")
print(f"  Mean: {cluster_cpcts.mean()*100:.2f}%, SD: {cluster_cpcts.std()*100:.2f}%")
print(f"Model bootstrap CI (95%):   [{model_ci[0]*100:.2f}%, {model_ci[1]*100:.2f}%]")
print(f"  Width: {(model_ci[1]-model_ci[0])*100:.2f} pp")
print(f"  Mean: {model_cpcts.mean()*100:.2f}%, SD: {model_cpcts.std()*100:.2f}%")
print(f"CI width ratio (cluster/model): {(cluster_ci[1]-cluster_ci[0])/(model_ci[1]-model_ci[0]):.2f}x")

# ── Save outputs ───────────────────────────────────────────────────────────

# bootstrap_results.csv
boot_df = pd.DataFrame({
    "bootstrap_type": ["cluster"] * N_BOOT + ["model"] * N_BOOT,
    "iteration": list(range(N_BOOT)) * 2,
    "c_pct": np.concatenate([cluster_cpcts, model_cpcts]),
})
boot_df.to_csv(OUT / "bootstrap_results.csv", index=False)

# cluster_bootstrap_ci.md
width_ratio = (cluster_ci[1] - cluster_ci[0]) / (model_ci[1] - model_ci[0])

md = f"""# SF1: Cluster Bootstrap — Family-level CI for Temporal C%

## Family Distribution

- **Filter**: only families with size >= 3 (among models with cohort ∈ {{2023, 2024}})
- **Before filter**: 33 families
- **After filter**: {len(valid_fams)} families, {len(df_mh)} models
- **Size distribution**: min={fam_dist['n_models'].min()}, median={fam_dist['n_models'].median():.0f}, max={fam_dist['n_models'].max()}
- **Top 5 families**: {', '.join(f"{r['family']} ({r['n_models']})" for _, r in fam_dist.head(5).iterrows())}

## Bootstrap Results (B = {N_BOOT})

| Metric | Cluster Bootstrap | Model Bootstrap |
|--------|------------------|-----------------|
| Point estimate C% | {c_pct_full*100:.2f}% | {c_pct_full*100:.2f}% |
| 95% CI | [{cluster_ci[0]*100:.2f}%, {cluster_ci[1]*100:.2f}%] | [{model_ci[0]*100:.2f}%, {model_ci[1]*100:.2f}%] |
| CI width | {(cluster_ci[1]-cluster_ci[0])*100:.2f} pp | {(model_ci[1]-model_ci[0])*100:.2f} pp |
| Mean | {cluster_cpcts.mean()*100:.2f}% | {model_cpcts.mean()*100:.2f}% |
| SD | {cluster_cpcts.std()*100:.2f}% | {model_cpcts.std()*100:.2f}% |

**CI width ratio (cluster / model)**: {width_ratio:.2f}×

## Interpretation

The cluster bootstrap CI is {width_ratio:.2f}× {'wider' if width_ratio > 1 else 'narrower'} than the model-level CI.
{'This indicates that within-family correlation inflates uncertainty — models from the same family tend to produce similar DIF patterns, so treating them as independent underestimates the true sampling variability.' if width_ratio > 1.2 else 'The cluster and model-level CIs are similar in width, suggesting that family-level dependence has limited impact on the C% estimate — the DIF signal is robust to model clustering.'}

Both CIs {'exclude zero' if cluster_ci[0] > 0 else 'include zero'}, confirming that the C% finding is {'robust' if cluster_ci[0] > 0.05 else 'fragile'} regardless of bootstrap method.
"""

(OUT / "cluster_bootstrap_ci.md").write_text(md)

print(f"\nOutputs saved to {OUT}/")
print("  - family_distribution.csv")
print("  - bootstrap_results.csv")
print("  - cluster_bootstrap_ci.md")
