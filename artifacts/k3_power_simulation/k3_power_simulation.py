"""
K=3 vs K=5 Strata Power Simulation for MH-DIF Detection.

Tests whether K=3 strata provides adequate power for contamination detection,
supporting the paper claim "smaller corpora may suffice with K=3".

PSN-IRT empirical parameters: ā=1.2, b̄=-0.84
Grid: K∈{3,5} × N∈{40,60,80,100,200,500} × δ∈{0.6,0.8,1.0,1.2,1.4}
1000 replications per condition, MH-DIF only.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from itertools import product

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

N_ITEMS = 200
N_CONTAMINATED = 20
ALPHA = 0.05
SEED = 42

A_MEAN = 1.2
B_MEAN = -0.84

K_VALUES = [3, 5]
N_VALUES = [40, 60, 80, 100, 200, 500]
DELTA_VALUES = [0.6, 0.8, 1.0, 1.2, 1.4]


def generate_item_params(n_items, rng):
    sigma_a = 0.4
    mu_a = np.log(A_MEAN) - sigma_a**2 / 2
    a = rng.lognormal(mean=mu_a, sigma=sigma_a, size=n_items)
    a = np.clip(a, 0.3, 2.5)
    b = rng.normal(B_MEAN, 1.0, size=n_items)
    return a, b


def generate_responses(theta, a, b, delta=None, contaminated_mask=None, rng=None):
    b_adj = b.copy()
    if delta is not None and contaminated_mask is not None:
        b_adj[contaminated_mask] = b[contaminated_mask] - delta
    logit = a[np.newaxis, :] * (theta[:, np.newaxis] - b_adj[np.newaxis, :])
    prob = expit(logit)
    return (rng.random(prob.shape) < prob).astype(np.int8)


def mantel_haenszel_dif(responses, group, n_strata):
    n_total, n_items = responses.shape
    total_score = responses.sum(axis=1).astype(float)

    try:
        _, bin_edges = pd.qcut(total_score, q=n_strata, retbins=True, duplicates="drop")
    except ValueError:
        bin_edges = np.linspace(total_score.min() - 0.5, total_score.max() + 0.5, n_strata + 1)
    strata = np.digitize(total_score, bin_edges[1:-1])

    ref_mask = group == 0
    foc_mask = group == 1

    numerator = np.zeros(n_items)
    denominator = np.zeros(n_items)
    chi_num = np.zeros(n_items)
    chi_var = np.zeros(n_items)

    for k in np.unique(strata):
        in_stratum = strata == k
        ref_in = ref_mask & in_stratum
        foc_in = foc_mask & in_stratum

        n_r = ref_in.sum()
        n_f = foc_in.sum()
        if n_r == 0 or n_f == 0:
            continue

        n_k = float(n_r + n_f)

        a_k = responses[ref_in].sum(axis=0).astype(float)
        b_k = n_r - a_k
        c_k = responses[foc_in].sum(axis=0).astype(float)
        d_k = n_f - c_k

        numerator += (a_k * d_k) / n_k
        denominator += (b_k * c_k) / n_k

        e_a = n_r * (a_k + c_k) / n_k
        chi_num += a_k - e_a
        v = (n_r * n_f * (a_k + c_k) * (b_k + d_k)) / (n_k**2 * max(n_k - 1, 1))
        chi_var += v

    valid = (denominator > 0) & (numerator > 0)
    delta_mh = np.zeros(n_items)
    alpha_mh = np.ones(n_items)
    alpha_mh[valid] = numerator[valid] / denominator[valid]
    delta_mh[valid] = -2.35 * np.log(alpha_mh[valid])

    pvalues = np.ones(n_items)
    valid_chi = chi_var > 0
    chi_sq = np.zeros(n_items)
    chi_sq[valid_chi] = (np.abs(chi_num[valid_chi]) - 0.5)**2 / chi_var[valid_chi]
    pvalues[valid_chi] = 1.0 - stats.chi2.cdf(chi_sq[valid_chi], df=1)

    ets_class = np.array(["A"] * n_items, dtype="U1")
    abs_delta = np.abs(delta_mh)
    c_mask = (abs_delta >= 1.5) & (pvalues < ALPHA)
    b_mask = (abs_delta >= 1.0) & (pvalues < ALPHA) & ~c_mask
    ets_class[c_mask] = "C"
    ets_class[b_mask] = "B"

    return delta_mh, pvalues, ets_class


def run_single_replication(n_per_group, delta, n_strata, a, b, contaminated_mask, rng):
    theta_ref = rng.normal(0, 1, n_per_group)
    theta_foc = rng.normal(0, 1, n_per_group)

    resp_ref = generate_responses(theta_ref, a, b, rng=rng)
    resp_foc = generate_responses(theta_foc, a, b, delta=delta,
                                  contaminated_mask=contaminated_mask, rng=rng)

    responses = np.vstack([resp_ref, resp_foc])
    group = np.array([0]*n_per_group + [1]*n_per_group, dtype=float)

    delta_mh, pval_mh, ets_class = mantel_haenszel_dif(responses, group, n_strata)
    reject_mh, _, _, _ = multipletests(pval_mh, alpha=ALPHA, method="fdr_bh")

    return {
        "flagged": reject_mh,
        "stat": np.abs(np.nan_to_num(delta_mh, nan=0.0)),
        "ets_class": ets_class,
    }


def compute_metrics(all_results, contaminated_mask):
    pos = contaminated_mask
    neg = ~contaminated_mask
    n_pos = pos.sum()
    n_neg = neg.sum()

    n_reps = len(all_results)
    tpr_arr = np.empty(n_reps)
    fpr_arr = np.empty(n_reps)
    auc_arr = np.empty(n_reps)
    ets_c_tpr_arr = np.empty(n_reps)
    ets_c_fpr_arr = np.empty(n_reps)

    for i, res in enumerate(all_results):
        flagged = res["flagged"]
        stat = res["stat"]

        tpr_arr[i] = flagged[pos].sum() / max(n_pos, 1)
        fpr_arr[i] = flagged[neg].sum() / max(n_neg, 1)

        try:
            auc_arr[i] = roc_auc_score(pos.astype(int), stat)
        except ValueError:
            auc_arr[i] = 0.5

        ets_c = res["ets_class"] == "C"
        ets_c_tpr_arr[i] = (ets_c & pos).sum() / max(n_pos, 1)
        ets_c_fpr_arr[i] = (ets_c & neg).sum() / max(n_neg, 1)

    return {
        "tpr": float(tpr_arr.mean()),
        "fpr": float(fpr_arr.mean()),
        "auc": float(auc_arr.mean()),
        "tpr_se": float(tpr_arr.std() / np.sqrt(n_reps)),
        "fpr_se": float(fpr_arr.std() / np.sqrt(n_reps)),
        "ets_c_tpr": float(ets_c_tpr_arr.mean()),
        "ets_c_fpr": float(ets_c_fpr_arr.mean()),
    }


def run_simulation(n_reps=1000):
    rng = np.random.default_rng(SEED)
    a, b = generate_item_params(N_ITEMS, rng)

    contaminated_idx = rng.choice(N_ITEMS, N_CONTAMINATED, replace=False)
    contaminated_mask = np.zeros(N_ITEMS, dtype=bool)
    contaminated_mask[contaminated_idx] = True

    print(f"Item params: mean(a)={a.mean():.3f}, mean(b)={b.mean():.3f}")
    print(f"Contaminated: {N_CONTAMINATED}/{N_ITEMS} items")
    print()

    conditions = list(product(K_VALUES, N_VALUES, DELTA_VALUES))
    total = len(conditions)
    all_rows = []

    for idx, (k, n, delta) in enumerate(conditions, 1):
        t0 = time.time()
        print(f"[{idx}/{total}] K={k}, N={n}, δ={delta}", end="", flush=True)

        rep_results = []
        for _ in range(n_reps):
            res = run_single_replication(n, delta, k, a, b, contaminated_mask, rng)
            rep_results.append(res)

        m = compute_metrics(rep_results, contaminated_mask)
        dt = time.time() - t0

        row = {
            "K": k,
            "n_per_group": n,
            "delta": delta,
            "tpr": round(m["tpr"], 4),
            "fpr": round(m["fpr"], 4),
            "auc": round(m["auc"], 4),
            "tpr_se": round(m["tpr_se"], 4),
            "fpr_se": round(m["fpr_se"], 4),
            "ets_c_tpr": round(m["ets_c_tpr"], 4),
            "ets_c_fpr": round(m["ets_c_fpr"], 4),
            "n_reps": n_reps,
        }
        all_rows.append(row)
        print(f"  TPR={m['tpr']:.3f} FPR={m['fpr']:.3f} AUC={m['auc']:.3f}  ({dt:.1f}s)")

    return pd.DataFrame(all_rows)


def generate_report(df):
    lines = ["# K=3 Power Simulation Results\n"]
    lines.append("## Setup\n")
    lines.append(f"- **PSN-IRT parameters**: ā={A_MEAN}, b̄={B_MEAN}")
    lines.append(f"- **Items**: {N_ITEMS} total, {N_CONTAMINATED} contaminated (10%)")
    lines.append(f"- **K (strata)**: {K_VALUES}")
    lines.append(f"- **N/cohort**: {N_VALUES}")
    lines.append(f"- **δ (effect sizes)**: {DELTA_VALUES}")
    lines.append(f"- **Replications**: 1000 per condition")
    lines.append(f"- **Method**: Mantel-Haenszel with BH FDR correction")
    lines.append(f"- **Flagging**: ETS C classification (|Δ_MH|≥1.5 & p<0.05)\n")

    lines.append("## Key Finding: K=3 vs K=5 Comparison\n")

    for delta in DELTA_VALUES:
        lines.append(f"### δ = {delta}\n")
        lines.append("| N/cohort | K=3 TPR | K=5 TPR | K=3 FPR | K=5 FPR | K=3 AUC | K=5 AUC |")
        lines.append("|----------|---------|---------|---------|---------|---------|---------|")
        for n in N_VALUES:
            r3 = df[(df["K"] == 3) & (df["n_per_group"] == n) & (df["delta"] == delta)]
            r5 = df[(df["K"] == 5) & (df["n_per_group"] == n) & (df["delta"] == delta)]
            if len(r3) == 0 or len(r5) == 0:
                continue
            r3, r5 = r3.iloc[0], r5.iloc[0]
            lines.append(
                f"| {n:>8} | {r3['tpr']:.3f}   | {r5['tpr']:.3f}   "
                f"| {r3['fpr']:.3f}   | {r5['fpr']:.3f}   "
                f"| {r3['auc']:.3f}   | {r5['auc']:.3f}   |"
            )
        lines.append("")

    lines.append("## Success Criterion Check\n")
    lines.append("**Claim**: K=3 at N≥80 achieves TPR≥0.5\n")

    for delta in DELTA_VALUES:
        k3_n80 = df[(df["K"] == 3) & (df["n_per_group"] >= 80) & (df["delta"] == delta)]
        if len(k3_n80) == 0:
            continue
        min_tpr = k3_n80["tpr"].min()
        status = "PASS" if min_tpr >= 0.5 else "FAIL"
        lines.append(f"- δ={delta}: min TPR(K=3, N≥80) = {min_tpr:.3f} → **{status}**")

    lines.append("\n## Power Degradation: K=3 vs K=5\n")
    lines.append("Average relative TPR change (K=3 vs K=5) across all N:\n")

    for delta in DELTA_VALUES:
        pairs = []
        for n in N_VALUES:
            r3 = df[(df["K"] == 3) & (df["n_per_group"] == n) & (df["delta"] == delta)]
            r5 = df[(df["K"] == 5) & (df["n_per_group"] == n) & (df["delta"] == delta)]
            if len(r3) > 0 and len(r5) > 0:
                t3, t5 = r3.iloc[0]["tpr"], r5.iloc[0]["tpr"]
                if t5 > 0:
                    pairs.append((t3 - t5) / t5)
        if pairs:
            avg_change = np.mean(pairs) * 100
            lines.append(f"- δ={delta}: {avg_change:+.1f}%")

    lines.append("\n## Interpretation\n")
    lines.append("*(Auto-generated; verify against simulation output.)*\n")

    k3_core = df[(df["K"] == 3) & (df["n_per_group"] >= 80) & (df["delta"] >= 1.0)]
    k5_core = df[(df["K"] == 5) & (df["n_per_group"] >= 80) & (df["delta"] >= 1.0)]

    if len(k3_core) > 0 and len(k5_core) > 0:
        k3_mean_tpr = k3_core["tpr"].mean()
        k5_mean_tpr = k5_core["tpr"].mean()
        diff = k3_mean_tpr - k5_mean_tpr
        lines.append(
            f"In the core region (N≥80, δ≥1.0), K=3 achieves mean TPR={k3_mean_tpr:.3f} "
            f"vs K=5 mean TPR={k5_mean_tpr:.3f} (Δ={diff:+.3f}). "
        )
        if abs(diff) < 0.05:
            lines.append("The power difference is negligible, supporting K=3 as sufficient.")
        elif diff < -0.10:
            lines.append("K=3 shows meaningful power loss; K=5 is recommended for small N.")
        else:
            lines.append("K=3 shows modest power difference relative to K=5.")

    return "\n".join(lines)


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print(f"K=3 Power Simulation: {n_reps} replications per condition")
    print(f"Grid: K={K_VALUES} × N={N_VALUES} × δ={DELTA_VALUES}")
    print(f"Total conditions: {len(K_VALUES) * len(N_VALUES) * len(DELTA_VALUES)}")
    print(f"PSN-IRT params: ā={A_MEAN}, b̄={B_MEAN}")
    print()

    t0 = time.time()
    df = run_simulation(n_reps=n_reps)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")

    csv_path = os.path.join(OUT_DIR, "k3_power_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    report = generate_report(df)
    report_path = os.path.join(OUT_DIR, "k3_power_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    print("\n--- Results Summary ---")
    print(df.to_string(index=False))
