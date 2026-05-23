"""
Mixed-mechanism semi-synthetic injection: difficulty × model_type.
Scheme D: base=difficulty-dependent, instruct=uniform
Scheme E: base=uniform, instruct=difficulty-dependent
"""

import re
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import chi2 as chi2_dist
import time, os

ROOT = "/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts"
OUT = os.path.join(ROOT, "mixed_mechanism_semisynthetic")

N_SEEDS = 20
N_STRATA = 5
FOC_INJECT_FRAC = 0.30
ITEM_INJECT_FRAC = 0.24

# ════════════════════════════════════════════════════════════════════════════
#  MODEL CLASSIFICATION (from R18)
# ════════════════════════════════════════════════════════════════════════════
INSTRUCT_SUFFIXES = re.compile(
    r"[-_\.](instruct|chat|it|rlhf|dpo|sft|aligned|ppo)([-_\.]|$)", re.IGNORECASE)
QUANT_SUFFIXES = re.compile(
    r"[-_\.](gptq|awq|gguf|ggml|exl2|fp16|bf16|int[48]|q[2-8]_[0-9])", re.IGNORECASE)
FAMILY_PREFIX_EXCLUDE = re.compile(r"^(openchat|chatglm|chatml)", re.IGNORECASE)

def classify_model(name: str) -> str:
    short = name.split("/")[-1] if "/" in name else name
    stripped = QUANT_SUFFIXES.sub("", short)
    if stripped.lower().endswith("-it"):
        return "instruct"
    if INSTRUCT_SUFFIXES.search(stripped):
        prefix_part = stripped.split("-")[0].split("_")[0]
        if FAMILY_PREFIX_EXCLUDE.match(prefix_part):
            rest = stripped[len(prefix_part):]
            if not INSTRUCT_SUFFIXES.search(rest):
                return "base"
        return "instruct"
    return "base"


# ════════════════════════════════════════════════════════════════════════════
#  VECTORIZED MH-DIF
# ════════════════════════════════════════════════════════════════════════════
def mh_dif_vec(X_ref, X_foc, n_strata=N_STRATA):
    n_ref, J = X_ref.shape
    n_foc = X_foc.shape[0]

    tot_ref = X_ref.sum(axis=1).astype(np.float64)
    tot_foc = X_foc.sum(axis=1).astype(np.float64)
    s_all = np.concatenate([tot_ref, tot_foc])

    edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
    n_bins = len(edges) - 1
    if n_bins < 1:
        return np.zeros(J), np.ones(J), np.zeros(J, dtype=bool)

    k_ref = np.digitize(tot_ref, edges[1:-1])
    k_foc = np.digitize(tot_foc, edges[1:-1])

    n1 = np.zeros(n_bins, dtype=np.float64)
    n0 = np.zeros(n_bins, dtype=np.float64)
    A = np.zeros((n_bins, J), dtype=np.float64)
    C_mat = np.zeros((n_bins, J), dtype=np.float64)

    for k in range(n_bins):
        mr = (k_ref == k)
        mf = (k_foc == k)
        n1[k] = mr.sum()
        n0[k] = mf.sum()
        if mr.any():
            A[k] = X_ref[mr].sum(axis=0).astype(np.float64)
        if mf.any():
            C_mat[k] = X_foc[mf].sum(axis=0).astype(np.float64)

    B = n1[:, None] - A
    D = n0[:, None] - C_mat
    Nk = (n1 + n0)[:, None]
    m1 = A + C_mat
    m0 = B + D

    valid = ((n1[:, None] > 0) & (n0[:, None] > 0) &
             (Nk > 1) & (m1 > 0) & (m0 > 0))

    Rj = np.where(valid, A * D / Nk, 0).sum(axis=0)
    Sj = np.where(valid, B * C_mat / Nk, 0).sum(axis=0)

    finite_mask = (Rj > 0) & (Sj > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_j = np.where(finite_mask, -2.35 * np.log(Rj / Sj), 0)

    EA = np.where(valid, n1[:, None] * m1 / Nk, 0)
    sum_diff = np.where(valid, A - EA, 0).sum(axis=0)
    Nk_m1 = np.where(Nk > 1, Nk - 1, 1)
    sum_var = np.where(
        valid,
        n1[:, None] * n0[:, None] * m1 * m0 / (Nk ** 2 * Nk_m1), 0
    ).sum(axis=0)

    chi2_j = np.where(
        sum_var > 0,
        np.maximum(np.abs(sum_diff) - 0.5, 0) ** 2 / np.maximum(sum_var, 1e-30), 0)
    pval_j = np.where(sum_var > 0, 1 - chi2_dist.cdf(chi2_j, df=1), 1.0)

    is_c = finite_mask & (np.abs(delta_j) >= 1.5) & (pval_j < 0.05)
    return delta_j, pval_j, is_c


# ════════════════════════════════════════════════════════════════════════════
#  INJECTION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def inject_difficulty_dependent(X_foc, model_mask, item_idx, item_diff, rng):
    """Contamination-like: injection prob ∝ (1 - item_accuracy), hard items more likely."""
    X = X_foc.copy()
    n_flipped = 0
    models = np.where(model_mask)[0]
    for j in item_idx:
        p_inject = max(0.0, min(1.0, (1.0 - item_diff[j] - 0.2) / 0.6))
        for i in models:
            if X[i, j] == 0 and rng.random() < p_inject:
                X[i, j] = 1
                n_flipped += 1
    return X, n_flipped


def inject_uniform(X_foc, model_mask, item_idx, rng, p_const=0.5):
    """Uniform injection probability p=0.5."""
    X = X_foc.copy()
    n_flipped = 0
    models = np.where(model_mask)[0]
    for j in item_idx:
        for i in models:
            if X[i, j] == 0 and rng.random() < p_const:
                X[i, j] = 1
                n_flipped += 1
    return X, n_flipped


def inject_mixed(X_foc, base_foc_mask, instruct_foc_mask,
                 inject_model_mask_base, inject_model_mask_instruct,
                 inject_item_idx, item_diff, rng,
                 base_scheme='difficulty', instruct_scheme='uniform'):
    """
    Mixed injection: different schemes for base vs instruct focal models.
    base_scheme/instruct_scheme: 'difficulty' or 'uniform'
    """
    X = X_foc.copy()
    n_flipped_base = 0
    n_flipped_instruct = 0

    base_models = np.where(inject_model_mask_base)[0]
    instruct_models = np.where(inject_model_mask_instruct)[0]

    for j in inject_item_idx:
        p_diff = max(0.0, min(1.0, (1.0 - item_diff[j] - 0.2) / 0.6))
        p_unif = 0.5

        p_base = p_diff if base_scheme == 'difficulty' else p_unif
        p_inst = p_diff if instruct_scheme == 'difficulty' else p_unif

        for i in base_models:
            if X[i, j] == 0 and rng.random() < p_base:
                X[i, j] = 1
                n_flipped_base += 1

        for i in instruct_models:
            if X[i, j] == 0 and rng.random() < p_inst:
                X[i, j] = 1
                n_flipped_instruct += 1

    return X, n_flipped_base, n_flipped_instruct


# ════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
print("Loading data...")
rm = np.load(f"{ROOT}/response_matrix.npz")
X_sparse = sparse.csr_matrix((rm['data'], rm['indices'], rm['indptr']), shape=tuple(rm['shape']))
idx = np.load(f"{ROOT}/response_matrix_index.npz", allow_pickle=True)
model_names = idx['model_names']
item_ids = idx['item_ids']

meta = pd.read_csv(f"{ROOT}/model_metadata.csv")
name_to_row = {n: i for i, n in enumerate(model_names)}

ref_models = meta[meta['cohort_temporal'] == '2023']['model_name'].values
foc_models = meta[meta['cohort_temporal'] == '2024']['model_name'].values

ref_idx = np.array([name_to_row[n] for n in ref_models if n in name_to_row])
foc_idx = np.array([name_to_row[n] for n in foc_models if n in name_to_row])

N_ref, N_foc, J = len(ref_idx), len(foc_idx), len(item_ids)
N_inject_items = int(J * ITEM_INJECT_FRAC)

# Classify focal models as base/instruct
foc_model_names = np.array([model_names[i] for i in foc_idx])
foc_model_types = np.array([classify_model(n) for n in foc_model_names])
foc_base_mask = (foc_model_types == 'base')
foc_instruct_mask = (foc_model_types == 'instruct')

N_foc_base = foc_base_mask.sum()
N_foc_instruct = foc_instruct_mask.sum()
N_inject_base = int(N_foc_base * FOC_INJECT_FRAC)
N_inject_instruct = int(N_foc_instruct * FOC_INJECT_FRAC)

# Classify ref models
ref_model_names = np.array([model_names[i] for i in ref_idx])
ref_model_types = np.array([classify_model(n) for n in ref_model_names])
ref_base_mask = (ref_model_types == 'base')
ref_instruct_mask = (ref_model_types == 'instruct')

N_ref_base = ref_base_mask.sum()
N_ref_instruct = ref_instruct_mask.sum()

print(f"  Ref: {N_ref} (base={N_ref_base}, instruct={N_ref_instruct})")
print(f"  Foc: {N_foc} (base={N_foc_base}, instruct={N_foc_instruct})")
print(f"  Items: {J}")
print(f"  Injection: {N_inject_base} base models, {N_inject_instruct} instruct models, {N_inject_items} items")

# Densify submatrices
X_ref_base_all = X_sparse[ref_idx].toarray().astype(np.float64)
X_foc_base_all = X_sparse[foc_idx].toarray().astype(np.float64)

# Submatrices by model type (for type-specific DIF)
X_ref_base_type = X_ref_base_all[ref_base_mask]
X_ref_instruct_type = X_ref_base_all[ref_instruct_mask]
X_foc_base_type = X_foc_base_all[foc_base_mask]
X_foc_instruct_type = X_foc_base_all[foc_instruct_mask]

item_difficulty = X_ref_base_all.mean(axis=0)
print(f"  Item difficulty: mean={item_difficulty.mean():.3f}")


# ════════════════════════════════════════════════════════════════════════════
#  BASELINE (no injection)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Baseline ===")

delta_base_all, pval_base_all, is_c_base_all = mh_dif_vec(X_ref_base_all, X_foc_base_all)
delta_base_b, pval_base_b, is_c_base_b = mh_dif_vec(X_ref_base_type, X_foc_base_type)
delta_base_i, pval_base_i, is_c_base_i = mh_dif_vec(X_ref_instruct_type, X_foc_instruct_type)

c_all = is_c_base_all.sum()
c_b = is_c_base_b.sum()
c_i = is_c_base_i.sum()

rho_base_all, _ = stats.spearmanr(item_difficulty[is_c_base_all], delta_base_all[is_c_base_all])
rho_base_b, _ = stats.spearmanr(item_difficulty[is_c_base_b], delta_base_b[is_c_base_b]) if c_b >= 3 else (np.nan, None)
rho_base_i, _ = stats.spearmanr(item_difficulty[is_c_base_i], delta_base_i[is_c_base_i]) if c_i >= 3 else (np.nan, None)

pct_focal_base_all = (is_c_base_all & (delta_base_all > 0)).sum() / c_all * 100
pct_focal_base_b = (is_c_base_b & (delta_base_b > 0)).sum() / c_b * 100 if c_b > 0 else 0
pct_focal_base_i = (is_c_base_i & (delta_base_i > 0)).sum() / c_i * 100 if c_i > 0 else 0

print(f"  Overall: C%={c_all/J*100:.1f}%, ρ={rho_base_all:.3f}, focal%={pct_focal_base_all:.1f}%")
print(f"  Base-only: C%={c_b/J*100:.1f}%, ρ={rho_base_b:.3f}, focal%={pct_focal_base_b:.1f}%")
print(f"  Instruct-only: C%={c_i/J*100:.1f}%, ρ={rho_base_i:.3f}, focal%={pct_focal_base_i:.1f}%")


# ════════════════════════════════════════════════════════════════════════════
#  DIRECTION REVERSAL χ² (helper)
# ════════════════════════════════════════════════════════════════════════════
def direction_reversal_chi2(is_c_base, delta_base, is_c_inst, delta_inst):
    """χ² for difference in focal-favoring proportion between base and instruct DIF-C items."""
    n_c_b = is_c_base.sum()
    n_c_i = is_c_inst.sum()
    if n_c_b == 0 or n_c_i == 0:
        return 0.0, 1.0

    n_focal_b = (is_c_base & (delta_base > 0)).sum()
    n_ref_b = n_c_b - n_focal_b
    n_focal_i = (is_c_inst & (delta_inst > 0)).sum()
    n_ref_i = n_c_i - n_focal_i

    table = np.array([[n_focal_b, n_ref_b], [n_focal_i, n_ref_i]])
    if table.min() == 0:
        return 0.0, 1.0
    chi2_val, p_val, _, _ = stats.chi2_contingency(table, correction=True)
    return chi2_val, p_val


# ════════════════════════════════════════════════════════════════════════════
#  RUN EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Running mixed-mechanism experiments ===")
results = []
t0 = time.time()

schemes = {
    'D_base_diff_instruct_unif': ('difficulty', 'uniform'),
    'E_base_unif_instruct_diff': ('uniform', 'difficulty'),
}

for seed in range(1, N_SEEDS + 1):
    rng = np.random.RandomState(seed * 42)

    # Select injection items (shared across schemes within seed)
    inject_item_idx = rng.choice(J, N_inject_items, replace=False)

    # Select injection models separately for base and instruct
    base_foc_indices = np.where(foc_base_mask)[0]
    instruct_foc_indices = np.where(foc_instruct_mask)[0]

    inject_base_idx = rng.choice(base_foc_indices, N_inject_base, replace=False)
    inject_instruct_idx = rng.choice(instruct_foc_indices, N_inject_instruct, replace=False)

    inject_model_mask_base = np.zeros(N_foc, dtype=bool)
    inject_model_mask_base[inject_base_idx] = True

    inject_model_mask_instruct = np.zeros(N_foc, dtype=bool)
    inject_model_mask_instruct[inject_instruct_idx] = True

    for scheme_name, (base_scheme, instruct_scheme) in schemes.items():
        t1 = time.time()

        # Use a fresh RNG copy per scheme to avoid cross-contamination
        rng_scheme = np.random.RandomState(seed * 42 + hash(scheme_name) % (2**31))

        X_foc_inj, n_flipped_base, n_flipped_instruct = inject_mixed(
            X_foc_base_all, foc_base_mask, foc_instruct_mask,
            inject_model_mask_base, inject_model_mask_instruct,
            inject_item_idx, item_difficulty, rng_scheme,
            base_scheme=base_scheme, instruct_scheme=instruct_scheme)

        n_flipped = n_flipped_base + n_flipped_instruct
        injected_items = set(inject_item_idx)
        inj_arr = np.array(sorted(injected_items))

        # ── Overall MH-DIF ──
        delta_j, pval_j, is_c = mh_dif_vec(X_ref_base_all, X_foc_inj)
        ddelta = delta_j - delta_base_all

        c_pct_all = is_c.sum() / J * 100
        n_c_total = is_c.sum()
        c_idx = np.where(is_c)[0]
        rho_all = stats.spearmanr(item_difficulty[c_idx], delta_j[c_idx])[0] if len(c_idx) >= 3 else np.nan

        n_focal_all = (is_c & (delta_j > 0)).sum()
        pct_focal_all = n_focal_all / n_c_total * 100 if n_c_total > 0 else 0

        rho_ddelta_inj = stats.spearmanr(item_difficulty[inj_arr], ddelta[inj_arr])[0] if len(inj_arr) >= 3 else np.nan

        # ── Base-only MH-DIF ──
        X_foc_inj_base = X_foc_inj[foc_base_mask]
        delta_b, pval_b, is_c_b = mh_dif_vec(X_ref_base_type, X_foc_inj_base)
        ddelta_b = delta_b - delta_base_b

        c_pct_b = is_c_b.sum() / J * 100
        n_c_b = is_c_b.sum()
        c_idx_b = np.where(is_c_b)[0]
        rho_base = stats.spearmanr(item_difficulty[c_idx_b], delta_b[c_idx_b])[0] if len(c_idx_b) >= 3 else np.nan

        n_focal_b = (is_c_b & (delta_b > 0)).sum()
        pct_focal_b = n_focal_b / n_c_b * 100 if n_c_b > 0 else 0

        rho_ddelta_b = stats.spearmanr(item_difficulty[inj_arr], ddelta_b[inj_arr])[0] if len(inj_arr) >= 3 else np.nan

        # ── Instruct-only MH-DIF ──
        X_foc_inj_instruct = X_foc_inj[foc_instruct_mask]
        delta_i, pval_i, is_c_i = mh_dif_vec(X_ref_instruct_type, X_foc_inj_instruct)
        ddelta_i = delta_i - delta_base_i

        c_pct_i = is_c_i.sum() / J * 100
        n_c_i = is_c_i.sum()
        c_idx_i = np.where(is_c_i)[0]
        rho_instruct = stats.spearmanr(item_difficulty[c_idx_i], delta_i[c_idx_i])[0] if len(c_idx_i) >= 3 else np.nan

        n_focal_i = (is_c_i & (delta_i > 0)).sum()
        pct_focal_i = n_focal_i / n_c_i * 100 if n_c_i > 0 else 0

        rho_ddelta_i = stats.spearmanr(item_difficulty[inj_arr], ddelta_i[inj_arr])[0] if len(inj_arr) >= 3 else np.nan

        # ── Direction reversal χ² ──
        chi2_rev, p_rev = direction_reversal_chi2(is_c_b, delta_b, is_c_i, delta_i)

        elapsed = time.time() - t1

        row = dict(
            scheme=scheme_name, seed=seed,
            n_inject_base=N_inject_base,
            n_inject_instruct=N_inject_instruct,
            n_inject_items=len(injected_items),
            n_flipped_base=n_flipped_base,
            n_flipped_instruct=n_flipped_instruct,
            n_flipped_total=n_flipped,
            # Overall
            rho_overall=rho_all,
            rho_ddelta_overall=rho_ddelta_inj,
            c_pct_overall=c_pct_all,
            n_c_overall=n_c_total,
            pct_focal_overall=pct_focal_all,
            # Base-only
            rho_base=rho_base,
            rho_ddelta_base=rho_ddelta_b,
            c_pct_base=c_pct_b,
            n_c_base=n_c_b,
            pct_focal_base=pct_focal_b,
            # Instruct-only
            rho_instruct=rho_instruct,
            rho_ddelta_instruct=rho_ddelta_i,
            c_pct_instruct=c_pct_i,
            n_c_instruct=n_c_i,
            pct_focal_instruct=pct_focal_i,
            # Direction reversal
            chi2_reversal=chi2_rev,
            p_reversal=p_rev,
            elapsed_sec=elapsed,
        )
        results.append(row)
        print(f"  [{scheme_name}] seed={seed}: "
              f"ρ_all={rho_all:.3f}, ρ_base={rho_base:.3f}, ρ_inst={rho_instruct:.3f}, "
              f"χ²={chi2_rev:.1f}, focal%: all={pct_focal_all:.1f}/base={pct_focal_b:.1f}/inst={pct_focal_i:.1f}, "
              f"flips={n_flipped} ({n_flipped_base}B+{n_flipped_instruct}I), {elapsed:.1f}s")

total_time = time.time() - t0
print(f"\nTotal time: {total_time:.0f}s")


# ════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
for scheme_name in schemes:
    sub = df[df['scheme'] == scheme_name]
    fname = f"scheme_{scheme_name[0].lower()}_results.csv"
    sub.to_csv(os.path.join(OUT, fname), index=False)
    print(f"Saved: {fname} ({len(sub)} rows)")

df.to_csv(os.path.join(OUT, "all_results.csv"), index=False)


# ════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ════════════════════════════════════════════════════════════════════════════
obs = dict(rho_overall=-0.597, rho_base=-0.570, rho_instruct=-0.079, chi2=562.7)

lines = ["# Mixed-Mechanism Semi-Synthetic Injection\n"]
lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
lines.append(f"**Config**: {N_SEEDS} seeds, {N_ref} ref models ({N_ref_base} base, {N_ref_instruct} instruct), "
             f"{N_foc} foc models ({N_foc_base} base, {N_foc_instruct} instruct), {J} items\n")
lines.append(f"**Injection**: 30% focal models per type ({N_inject_base} base, {N_inject_instruct} instruct), "
             f"{N_inject_items} items ({ITEM_INJECT_FRAC*100:.0f}%)\n")

lines.append("\n## Observed Data (for comparison)\n")
lines.append(f"| Metric | Observed |")
lines.append(f"|--------|----------|")
lines.append(f"| ρ(difficulty, δ_MH) overall | {obs['rho_overall']:.3f} |")
lines.append(f"| ρ(difficulty, δ_MH) base-only | {obs['rho_base']:.3f} |")
lines.append(f"| ρ(difficulty, δ_MH) instruct-only | {obs['rho_instruct']:.3f} |")
lines.append(f"| Direction reversal χ² | {obs['chi2']:.1f} |")

lines.append("\n## Baseline (no injection)\n")
lines.append(f"| Metric | Overall | Base-only | Instruct-only |")
lines.append(f"|--------|---------|-----------|---------------|")
lines.append(f"| C% | {c_all/J*100:.1f}% | {is_c_base_b.sum()/J*100:.1f}% | {is_c_base_i.sum()/J*100:.1f}% |")
lines.append(f"| ρ(diff, δ) | {rho_base_all:.3f} | {rho_base_b:.3f} | {rho_base_i:.3f} |")
lines.append(f"| % focal-favoring | {pct_focal_base_all:.1f}% | {pct_focal_base_b:.1f}% | {pct_focal_base_i:.1f}% |")

for scheme_name, (bs, ins) in schemes.items():
    sub = df[df['scheme'] == scheme_name]
    letter = scheme_name[0]
    lines.append(f"\n## Scheme {letter} — base={bs}, instruct={ins}\n")

    lines.append(f"| Metric | Overall | Base-only | Instruct-only | Observed |")
    lines.append(f"|--------|---------|-----------|---------------|----------|")

    def fmt(col):
        return f"{sub[col].mean():.3f} ± {sub[col].std():.3f}"

    lines.append(f"| ρ(diff, δ_MH) among C items | {fmt('rho_overall')} | {fmt('rho_base')} | {fmt('rho_instruct')} | {obs['rho_overall']:.3f} / {obs['rho_base']:.3f} / {obs['rho_instruct']:.3f} |")
    lines.append(f"| ρ(diff, Δδ) injected items | {fmt('rho_ddelta_overall')} | {fmt('rho_ddelta_base')} | {fmt('rho_ddelta_instruct')} | — |")
    lines.append(f"| C% | {sub['c_pct_overall'].mean():.1f}±{sub['c_pct_overall'].std():.1f} | {sub['c_pct_base'].mean():.1f}±{sub['c_pct_base'].std():.1f} | {sub['c_pct_instruct'].mean():.1f}±{sub['c_pct_instruct'].std():.1f} | — |")
    lines.append(f"| % focal-favoring | {sub['pct_focal_overall'].mean():.1f}±{sub['pct_focal_overall'].std():.1f} | {sub['pct_focal_base'].mean():.1f}±{sub['pct_focal_base'].std():.1f} | {sub['pct_focal_instruct'].mean():.1f}±{sub['pct_focal_instruct'].std():.1f} | — |")
    lines.append(f"| Direction reversal χ² | {sub['chi2_reversal'].mean():.1f} ± {sub['chi2_reversal'].std():.1f} | — | — | {obs['chi2']:.1f} |")
    lines.append(f"| N DIF-C | {sub['n_c_overall'].mean():.0f}±{sub['n_c_overall'].std():.0f} | {sub['n_c_base'].mean():.0f}±{sub['n_c_base'].std():.0f} | {sub['n_c_instruct'].mean():.0f}±{sub['n_c_instruct'].std():.0f} | — |")
    lines.append(f"| Flips (base/instruct) | {sub['n_flipped_total'].mean():.0f} | {sub['n_flipped_base'].mean():.0f} | {sub['n_flipped_instruct'].mean():.0f} | — |")

# Comparison summary
lines.append(f"\n## Comparison with Observed Data\n")
lines.append(f"| Metric | Observed | Scheme D | Scheme E |")
lines.append(f"|--------|----------|----------|----------|")

d = df[df['scheme'] == 'D_base_diff_instruct_unif']
e = df[df['scheme'] == 'E_base_unif_instruct_diff']

for metric, obs_val, col in [
    ('ρ overall', obs['rho_overall'], 'rho_overall'),
    ('ρ base', obs['rho_base'], 'rho_base'),
    ('ρ instruct', obs['rho_instruct'], 'rho_instruct'),
    ('χ² reversal', obs['chi2'], 'chi2_reversal'),
]:
    d_val = f"{d[col].mean():.3f} ± {d[col].std():.3f}"
    e_val = f"{e[col].mean():.3f} ± {e[col].std():.3f}"
    if col == 'chi2_reversal':
        d_val = f"{d[col].mean():.1f} ± {d[col].std():.1f}"
        e_val = f"{e[col].mean():.1f} ± {e[col].std():.1f}"
    lines.append(f"| {metric} | {obs_val} | {d_val} | {e_val} |")

lines.append(f"\n### Distance from observed (L2 on [ρ_overall, ρ_base, ρ_instruct])\n")
obs_vec = np.array([obs['rho_overall'], obs['rho_base'], obs['rho_instruct']])
d_vec = np.array([d['rho_overall'].mean(), d['rho_base'].mean(), d['rho_instruct'].mean()])
e_vec = np.array([e['rho_overall'].mean(), e['rho_base'].mean(), e['rho_instruct'].mean()])
d_dist = np.sqrt(((d_vec - obs_vec)**2).sum())
e_dist = np.sqrt(((e_vec - obs_vec)**2).sum())
lines.append(f"- Scheme D: L2 = {d_dist:.3f}")
lines.append(f"- Scheme E: L2 = {e_dist:.3f}")
winner = "D" if d_dist < e_dist else "E"
lines.append(f"- **Scheme {winner} is closer to observed pattern**")

lines.append(f"\n### Key questions answered\n")
lines.append(f"1. **Can mixed mechanism reproduce overall ρ≈-0.6?**")
d_rho = d['rho_overall'].mean()
e_rho = e['rho_overall'].mean()
lines.append(f"   - Scheme D: ρ={d_rho:.3f} → {'YES' if abs(d_rho - obs['rho_overall']) < 0.15 else 'NO'}")
lines.append(f"   - Scheme E: ρ={e_rho:.3f} → {'YES' if abs(e_rho - obs['rho_overall']) < 0.15 else 'NO'}")

lines.append(f"2. **Can mixed mechanism reproduce direction reversal (χ²>100)?**")
d_chi = d['chi2_reversal'].mean()
e_chi = e['chi2_reversal'].mean()
lines.append(f"   - Scheme D: χ²={d_chi:.1f} → {'YES' if d_chi > 100 else 'NO'}")
lines.append(f"   - Scheme E: χ²={e_chi:.1f} → {'YES' if e_chi > 100 else 'NO'}")

lines.append(f"3. **Can mixed mechanism simultaneously reproduce BOTH?**")
d_both = abs(d_rho - obs['rho_overall']) < 0.15 and d_chi > 100
e_both = abs(e_rho - obs['rho_overall']) < 0.15 and e_chi > 100
lines.append(f"   - Scheme D: {'YES' if d_both else 'NO'}")
lines.append(f"   - Scheme E: {'YES' if e_both else 'NO'}")

# Also compare with R19 schemes
lines.append(f"\n## Comparison with R19 Schemes (A/B/C)\n")
lines.append(f"| Scheme | ρ overall | ρ base | ρ instruct | χ² reversal | Mechanism |")
lines.append(f"|--------|-----------|--------|------------|-------------|-----------|")
lines.append(f"| A (R19) | — | — | — | — | capability: prob∝accuracy |")
lines.append(f"| B (R19) | — | — | — | — | contamination: uniform p=0.5 |")
lines.append(f"| C (R19) | — | — | — | — | mixed A+B items |")

def fmt_mean_sd(s):
    return f"{s.mean():.3f}±{s.std():.3f}"

lines.append(f"| **D** | {fmt_mean_sd(d['rho_overall'])} | {fmt_mean_sd(d['rho_base'])} | {fmt_mean_sd(d['rho_instruct'])} | {d['chi2_reversal'].mean():.1f}±{d['chi2_reversal'].std():.1f} | base=diff-dep, instruct=uniform |")
lines.append(f"| **E** | {fmt_mean_sd(e['rho_overall'])} | {fmt_mean_sd(e['rho_base'])} | {fmt_mean_sd(e['rho_instruct'])} | {e['chi2_reversal'].mean():.1f}±{e['chi2_reversal'].std():.1f} | base=uniform, instruct=diff-dep |")
lines.append(f"| **Observed** | {obs['rho_overall']:.3f} | {obs['rho_base']:.3f} | {obs['rho_instruct']:.3f} | {obs['chi2']:.1f} | ? |")

report_text = "\n".join(lines) + "\n"
with open(os.path.join(OUT, "mixed_mechanism_report.md"), "w") as f:
    f.write(report_text)

print(f"\nSaved: mixed_mechanism_report.md")
print("\n" + report_text)
