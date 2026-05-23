"""
Phase 2: Semi-synthetic DIF power simulation using real MMLU IRT parameters
from PSN-IRT (4PL → 2PL: take a, b only).

Validates Phase 1 findings (synthetic params) with real parameter distributions.
Each replication samples 200 items (w/o replacement) from the 14042-item MMLU pool.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))
N_ITEMS = 200
N_CONTAMINATED = 20
N_STRATA = 5
ALPHA = 0.05
SEED = 42

# Benchmark item counts (alphabetical order in PSN-IRT combine.csv)
BENCHMARK_SIZES = [
    ("ARC-C", 295), ("BBH", 6511), ("Chinese SimpleQA", 3000),
    ("GPQA Diamond", 198), ("GSM8K", 1319), ("HellaSwag", 10042),
    ("HumanEval", 164), ("MATH", 5000), ("MBPP", 500),
    ("MMLU", 14042), ("TheoremQA", 800),
]

sys.path.insert(0, ARTIFACTS_DIR)
try:
    from dif_power_simulation import (
        generate_responses, mantel_haenszel_dif,
        logistic_regression_dif, _batched_logistic_ll,
    )
except ImportError:
    print("WARNING: Could not import from dif_power_simulation.py, using inline copies")
    def generate_responses(theta, a, b, delta=None, contaminated_mask=None):
        b_adj = b.copy()
        if delta is not None and contaminated_mask is not None:
            b_adj[contaminated_mask] = b[contaminated_mask] - delta
        logit = a[np.newaxis, :] * (theta[:, np.newaxis] - b_adj[np.newaxis, :])
        prob = expit(logit)
        return (np.random.random(prob.shape) < prob).astype(np.int8)

    def mantel_haenszel_dif(responses, group, n_strata=N_STRATA):
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

    def _batched_logistic_ll(X, Y, max_iter=25, tol=1e-8):
        N, p = X.shape
        J = Y.shape[1]
        Beta = np.zeros((p, J))
        for _ in range(max_iter):
            Eta = X @ Beta
            np.clip(Eta, -20, 20, out=Eta)
            Mu = expit(Eta)
            W = Mu * (1.0 - Mu)
            np.maximum(W, 1e-12, out=W)
            Resid = Y - Mu
            Grad = X.T @ Resid
            XtWX = np.einsum("na,nj,nb->jab", X, W, X)
            XtWX += np.eye(p)[np.newaxis, :, :] * 1e-10
            try:
                Step = np.linalg.solve(XtWX, Grad.T[:, :, np.newaxis])[:, :, 0].T
            except np.linalg.LinAlgError:
                break
            Beta += Step
            if np.max(np.abs(Step)) < tol:
                break
        Eta = X @ Beta
        np.clip(Eta, -20, 20, out=Eta)
        Mu = expit(Eta)
        np.clip(Mu, 1e-15, 1.0 - 1e-15, out=Mu)
        LL = np.sum(Y * np.log(Mu) + (1.0 - Y) * np.log(1.0 - Mu), axis=0)
        return LL

    def logistic_regression_dif(responses, group):
        n_total, n_items = responses.shape
        Y = responses.astype(np.float64)
        total_score = responses.sum(axis=1).astype(float)
        ts_std = total_score.std()
        if ts_std < 1e-8:
            ts_std = 1.0
        ts_z = (total_score - total_score.mean()) / ts_std
        ones = np.ones(n_total)
        X_reduced = np.column_stack([ones, ts_z])
        X_full = np.column_stack([ones, ts_z, group.astype(float), ts_z * group])
        col_sums = Y.sum(axis=0)
        active = (col_sums > 0) & (col_sums < n_total)
        lr_chi2 = np.zeros(n_items)
        pvalues = np.ones(n_items)
        if active.any():
            Y_act = Y[:, active]
            ll_r = _batched_logistic_ll(X_reduced, Y_act)
            ll_f = _batched_logistic_ll(X_full, Y_act)
            lr_stat = np.maximum(-2.0 * (ll_r - ll_f), 0.0)
            lr_chi2[active] = lr_stat
            pvalues[active] = 1.0 - stats.chi2.cdf(lr_stat, df=2)
        return lr_chi2, pvalues


# ─── Load Real Parameters ──────────────────────────────────────────────────

def load_mmlu_params(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(ARTIFACTS_DIR, "psn_irt_item_parameters.csv")
    df = pd.read_csv(csv_path)
    cumulative = np.cumsum([s for _, s in BENCHMARK_SIZES])
    mmlu_idx = [name for name, _ in BENCHMARK_SIZES].index("MMLU")
    start = cumulative[mmlu_idx - 1] if mmlu_idx > 0 else 0
    end = cumulative[mmlu_idx]
    mmlu = df.iloc[start:end]
    a = mmlu["discrimination"].values.astype(np.float64)
    b = mmlu["difficulty"].values.astype(np.float64)
    valid = (a > 0) & np.isfinite(a) & np.isfinite(b)
    return a[valid], b[valid]


# ─── Single Replication (samples items each time) ──────────────────────────

def run_single_replication(n_per_group, delta, pool_a, pool_b, rng):
    idx = rng.choice(len(pool_a), N_ITEMS, replace=False)
    a = pool_a[idx]
    b = pool_b[idx]

    contaminated_idx = rng.choice(N_ITEMS, N_CONTAMINATED, replace=False)
    contaminated_mask = np.zeros(N_ITEMS, dtype=bool)
    contaminated_mask[contaminated_idx] = True

    theta_ref = rng.normal(0, 1, n_per_group)
    theta_foc = rng.normal(0, 1, n_per_group)

    np.random.seed(rng.integers(0, 2**31))
    resp_ref = generate_responses(theta_ref, a, b)
    resp_foc = generate_responses(theta_foc, a, b, delta=delta, contaminated_mask=contaminated_mask)

    responses = np.vstack([resp_ref, resp_foc])
    group = np.array([0]*n_per_group + [1]*n_per_group, dtype=float)

    results = {}

    # MH
    delta_mh, pval_mh, ets_class = mantel_haenszel_dif(responses, group)
    reject_mh_bh, _, _, _ = multipletests(pval_mh, alpha=ALPHA, method="fdr_bh")
    reject_mh_raw = pval_mh < ALPHA
    results["mh"] = {
        "flagged_raw": reject_mh_raw,
        "flagged_bh": reject_mh_bh,
        "stat": np.abs(np.nan_to_num(delta_mh, nan=0.0)),
        "ets_class": ets_class,
    }

    # LR
    lr_chi2, pval_lr = logistic_regression_dif(responses, group)
    reject_lr_bh, _, _, _ = multipletests(pval_lr, alpha=ALPHA, method="fdr_bh")
    reject_lr_raw = pval_lr < ALPHA
    results["lr"] = {
        "flagged_raw": reject_lr_raw,
        "flagged_bh": reject_lr_bh,
        "stat": lr_chi2,
    }

    return results, contaminated_mask


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(all_results):
    metrics = {}
    for method in ["mh", "lr"]:
        n_reps = len(all_results)
        tpr_raw = np.empty(n_reps)
        fpr_raw = np.empty(n_reps)
        tpr_bh = np.empty(n_reps)
        fpr_bh = np.empty(n_reps)
        auc_arr = np.empty(n_reps)
        ets_c_tpr = np.empty(n_reps) if method == "mh" else None
        ets_c_fpr = np.empty(n_reps) if method == "mh" else None

        for i, (res, contam) in enumerate(all_results):
            pos = contam
            neg = ~contam
            n_pos = pos.sum()
            n_neg = neg.sum()

            r = res[method]
            tpr_raw[i] = r["flagged_raw"][pos].sum() / max(n_pos, 1)
            fpr_raw[i] = r["flagged_raw"][neg].sum() / max(n_neg, 1)
            tpr_bh[i] = r["flagged_bh"][pos].sum() / max(n_pos, 1)
            fpr_bh[i] = r["flagged_bh"][neg].sum() / max(n_neg, 1)

            try:
                auc_arr[i] = roc_auc_score(pos.astype(int), r["stat"])
            except ValueError:
                auc_arr[i] = 0.5

            if method == "mh":
                ets_c = r["ets_class"] == "C"
                ets_c_tpr[i] = (ets_c & pos).sum() / max(n_pos, 1)
                ets_c_fpr[i] = (ets_c & neg).sum() / max(n_neg, 1)

        metrics[method] = {
            "tpr_raw": float(tpr_raw.mean()),
            "fpr_raw": float(fpr_raw.mean()),
            "tpr_bh": float(tpr_bh.mean()),
            "fpr_bh": float(fpr_bh.mean()),
            "auc": float(auc_arr.mean()),
            "ets_c_tpr": float(ets_c_tpr.mean()) if method == "mh" else np.nan,
            "ets_c_fpr": float(ets_c_fpr.mean()) if method == "mh" else np.nan,
        }
    return metrics


# ─── Verdict (Phase 2: based on ETS C metrics) ─────────────────────────────

def assign_verdict(row):
    n = row["n_per_group"]
    delta = row["delta"]
    ets_c_tpr = row["ets_c_tpr"]
    ets_c_fpr = row["ets_c_fpr"]

    if row["method"] != "MH":
        return "N/A"
    if not (isinstance(ets_c_tpr, (int, float)) and np.isfinite(ets_c_tpr)):
        return "N/A"

    if ets_c_fpr > 0.15:
        return "FAIL"
    if delta >= 1.4 and n >= 80:
        if ets_c_tpr >= 0.65 and ets_c_fpr <= 0.10:
            return "PASS"
        if ets_c_tpr >= 0.50:
            return "GRAY"
        return "FAIL"
    return "N/A"


# ─── Main Simulation ───────────────────────────────────────────────────────

def run_simulation(pool_a, pool_b, n_reps=1000):
    ns = [50, 80, 100]
    deltas = [0.8, 1.0, 1.2, 1.4, 1.6]

    rng = np.random.default_rng(SEED)
    all_rows = []
    total_conditions = len(ns) * len(deltas)
    cond_idx = 0

    for n_per_group, delta in product(ns, deltas):
        cond_idx += 1
        t_cond = time.time()
        print(f"[{cond_idx}/{total_conditions}] N={n_per_group}, δ={delta}, reps={n_reps}")

        rep_results = []
        for _ in range(n_reps):
            res, contam = run_single_replication(n_per_group, delta, pool_a, pool_b, rng)
            rep_results.append((res, contam))

        metrics = compute_metrics(rep_results)
        dt = time.time() - t_cond

        for method_name, method_key in [("MH", "mh"), ("LR", "lr")]:
            m = metrics[method_key]
            row = {
                "n_per_group": n_per_group,
                "delta": delta,
                "method": method_name,
                "tpr_raw": round(m["tpr_raw"], 4),
                "fpr_raw": round(m["fpr_raw"], 4),
                "tpr_bh": round(m["tpr_bh"], 4),
                "fpr_bh": round(m["fpr_bh"], 4),
                "ets_c_tpr": round(m["ets_c_tpr"], 4) if not np.isnan(m["ets_c_tpr"]) else "",
                "ets_c_fpr": round(m["ets_c_fpr"], 4) if not np.isnan(m["ets_c_fpr"]) else "",
                "auc": round(m["auc"], 4),
                "n_reps": n_reps,
            }
            row["verdict"] = assign_verdict(row)
            all_rows.append(row)

            ets_tag = ""
            if method_name == "MH" and row["ets_c_tpr"] != "":
                ets_tag = f"  ETS-C: TPR={row['ets_c_tpr']:.4f} FPR={row['ets_c_fpr']:.4f}"
                v = row["verdict"]
                if v != "N/A":
                    ets_tag += f" [{v}]"
            print(f"  {method_name}: raw TPR={m['tpr_raw']:.3f} FPR={m['fpr_raw']:.3f} | "
                  f"BH TPR={m['tpr_bh']:.3f} FPR={m['fpr_bh']:.3f} | AUC={m['auc']:.3f}{ets_tag}")

        print(f"  ({dt:.1f}s)")

    return pd.DataFrame(all_rows)


# ─── Parameter Distribution Plot ───────────────────────────────────────────

def plot_param_distributions(real_a, real_b, outpath):
    rng_synth = np.random.default_rng(SEED)
    synth_a = rng_synth.lognormal(mean=0.0, sigma=0.5, size=10000)
    synth_a = np.clip(synth_a, 0.3, 2.5)
    synth_b = rng_synth.normal(0.0, 1.0, size=10000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(real_a, bins=60, density=True, alpha=0.6, color="#2ca02c", label="Real MMLU (PSN-IRT)")
    axes[0].hist(synth_a, bins=60, density=True, alpha=0.5, color="#1f77b4", label="Synthetic (Phase 1)")
    axes[0].set_xlabel("Discrimination (a)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Item Discrimination")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(-0.2, 3.0)

    axes[1].hist(real_b, bins=60, density=True, alpha=0.6, color="#2ca02c", label="Real MMLU (PSN-IRT)")
    axes[1].hist(synth_b, bins=60, density=True, alpha=0.5, color="#1f77b4", label="Synthetic (Phase 1)")
    axes[1].set_xlabel("Difficulty (b)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Item Difficulty")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(-4.0, 3.0)

    fig.suptitle("Real MMLU vs Synthetic IRT Parameter Distributions", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parameter distribution plot: {outpath}")


# ─── Power Curves Plot ─────────────────────────────────────────────────────

def plot_power_curves(df, outpath):
    ns = sorted(df["n_per_group"].unique())
    colors = {50: "#ff7f0e", 80: "#2ca02c", 100: "#d62728"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    mh = df[df["method"] == "MH"]
    lr = df[df["method"] == "LR"]

    # Row 0: TPR curves
    # Col 0: ETS C TPR (main)
    for n in ns:
        cond = mh[mh["n_per_group"] == n]
        axes[0, 0].plot(cond["delta"], cond["ets_c_tpr"], "o-", color=colors[n], label=f"N={n}")
    axes[0, 0].set_title("MH ETS-C TPR (Primary)")
    axes[0, 0].set_ylabel("TPR")
    axes[0, 0].axhline(0.65, color="green", ls=":", lw=0.8, label="Pass=0.65")
    axes[0, 0].axhline(0.50, color="orange", ls=":", lw=0.8, label="Gray=0.50")
    axes[0, 0].axhline(0.75, color="blue", ls=":", lw=0.8, label="Pass+=0.75")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    axes[0, 0].set_ylim(-0.02, 1.02)

    # Col 1: Raw TPR
    for n in ns:
        cond_mh = mh[mh["n_per_group"] == n]
        cond_lr = lr[lr["n_per_group"] == n]
        axes[0, 1].plot(cond_mh["delta"], cond_mh["tpr_raw"], "o-", color=colors[n], label=f"MH N={n}")
        axes[0, 1].plot(cond_lr["delta"], cond_lr["tpr_raw"], "s--", color=colors[n], label=f"LR N={n}", alpha=0.6)
    axes[0, 1].set_title("Raw TPR (uncorrected)")
    axes[0, 1].set_ylabel("TPR")
    axes[0, 1].legend(fontsize=6, loc="lower right")
    axes[0, 1].set_ylim(-0.02, 1.02)

    # Col 2: BH TPR
    for n in ns:
        cond_mh = mh[mh["n_per_group"] == n]
        cond_lr = lr[lr["n_per_group"] == n]
        axes[0, 2].plot(cond_mh["delta"], cond_mh["tpr_bh"], "o-", color=colors[n], label=f"MH N={n}")
        axes[0, 2].plot(cond_lr["delta"], cond_lr["tpr_bh"], "s--", color=colors[n], label=f"LR N={n}", alpha=0.6)
    axes[0, 2].set_title("BH-corrected TPR")
    axes[0, 2].set_ylabel("TPR")
    axes[0, 2].legend(fontsize=6, loc="lower right")
    axes[0, 2].set_ylim(-0.02, 1.02)

    # Row 1: FPR curves
    # Col 0: ETS C FPR
    for n in ns:
        cond = mh[mh["n_per_group"] == n]
        axes[1, 0].plot(cond["delta"], cond["ets_c_fpr"], "s--", color=colors[n], label=f"N={n}")
    axes[1, 0].set_title("MH ETS-C FPR (Primary)")
    axes[1, 0].set_xlabel("δ (contamination effect, logits)")
    axes[1, 0].set_ylabel("FPR")
    axes[1, 0].axhline(0.10, color="orange", ls=":", lw=0.8)
    axes[1, 0].axhline(0.15, color="red", ls=":", lw=0.8)
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_ylim(-0.005, 0.25)

    # Col 1: Raw FPR
    for n in ns:
        cond_mh = mh[mh["n_per_group"] == n]
        cond_lr = lr[lr["n_per_group"] == n]
        axes[1, 1].plot(cond_mh["delta"], cond_mh["fpr_raw"], "o-", color=colors[n], label=f"MH N={n}")
        axes[1, 1].plot(cond_lr["delta"], cond_lr["fpr_raw"], "s--", color=colors[n], label=f"LR N={n}", alpha=0.6)
    axes[1, 1].set_title("Raw FPR (uncorrected)")
    axes[1, 1].set_xlabel("δ (contamination effect, logits)")
    axes[1, 1].set_ylabel("FPR")
    axes[1, 1].axhline(0.05, color="gray", ls=":", lw=0.8)
    axes[1, 1].axhline(0.15, color="red", ls=":", lw=0.8)
    axes[1, 1].legend(fontsize=6)
    axes[1, 1].set_ylim(-0.005, 0.25)

    # Col 2: BH FPR
    for n in ns:
        cond_mh = mh[mh["n_per_group"] == n]
        cond_lr = lr[lr["n_per_group"] == n]
        axes[1, 2].plot(cond_mh["delta"], cond_mh["fpr_bh"], "o-", color=colors[n], label=f"MH N={n}")
        axes[1, 2].plot(cond_lr["delta"], cond_lr["fpr_bh"], "s--", color=colors[n], label=f"LR N={n}", alpha=0.6)
    axes[1, 2].set_title("BH-corrected FPR")
    axes[1, 2].set_xlabel("δ (contamination effect, logits)")
    axes[1, 2].set_ylabel("FPR")
    axes[1, 2].axhline(0.05, color="gray", ls=":", lw=0.8)
    axes[1, 2].axhline(0.15, color="red", ls=":", lw=0.8)
    axes[1, 2].legend(fontsize=6)
    axes[1, 2].set_ylim(-0.005, 0.25)

    fig.suptitle("Phase 2: DIF Power Curves (Real MMLU Parameters)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved power curves: {outpath}")


# ─── Pass/Fail Evaluation ──────────────────────────────────────────────────

def evaluate_pass_fail(df):
    print("\n" + "=" * 70)
    print("PHASE 2 PASS / GRAY / FAIL EVALUATION (ETS C classification)")
    print("=" * 70)

    mh = df[df["method"] == "MH"].copy()
    mh["ets_c_tpr"] = pd.to_numeric(mh["ets_c_tpr"], errors="coerce")
    mh["ets_c_fpr"] = pd.to_numeric(mh["ets_c_fpr"], errors="coerce")

    fpr_violations = mh[mh["ets_c_fpr"] > 0.15]
    if len(fpr_violations) > 0:
        print(f"\nFPR > 0.15 violations: {len(fpr_violations)}")
        for _, r in fpr_violations.iterrows():
            print(f"  N={int(r['n_per_group'])}, δ={r['delta']}: ETS-C FPR={r['ets_c_fpr']:.4f}")
    else:
        print("\nNo FPR > 0.15 violations.")

    core = mh[(mh["delta"] >= 1.4) & (mh["n_per_group"] >= 80)]
    print(f"\nCore region (δ≥1.4, N≥80):")
    for _, r in core.iterrows():
        v = r["verdict"]
        print(f"  N={int(r['n_per_group'])}, δ={r['delta']}: "
              f"ETS-C TPR={r['ets_c_tpr']:.4f}, FPR={r['ets_c_fpr']:.4f} → {v}")

    n_pass = (core["verdict"] == "PASS").sum()
    n_gray = (core["verdict"] == "GRAY").sum()
    n_fail = (core["verdict"] == "FAIL").sum()
    print(f"  Summary: {n_pass} PASS, {n_gray} GRAY, {n_fail} FAIL / {len(core)} conditions")

    # Pass+
    plus = mh[(mh["delta"] >= 1.4) & (mh["n_per_group"] >= 100)]
    if len(plus) > 0:
        plus_tpr = plus["ets_c_tpr"].values
        all_pass_plus = all(t >= 0.75 for t in plus_tpr if np.isfinite(t))
        min_tpr = plus_tpr.min()
        tag = "PASS+" if all_pass_plus else "NOT MET"
        print(f"\n  Pass+ (δ≥1.4, N≥100): min ETS-C TPR = {min_tpr:.4f} → {tag}")

    verdicts = core["verdict"].values
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif "GRAY" in verdicts:
        overall = "GRAY"
    elif len(verdicts) > 0 and all(v == "PASS" for v in verdicts):
        overall = "PASS"
    else:
        overall = "N/A"

    print(f"\n  *** PHASE 2 OVERALL: {overall} ***")
    print("=" * 70)
    return overall


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print("Phase 2: Semi-synthetic DIF power simulation")
    print("Parameter source: PSN-IRT MMLU items (4PL → 2PL)")
    print(f"Replications: {n_reps}, Items/rep: {N_ITEMS}, Contaminated: {N_CONTAMINATED}")
    print(f"Seed: {SEED}")
    print()

    # Load real parameters
    pool_a, pool_b = load_mmlu_params()
    print(f"MMLU parameter pool: {len(pool_a)} valid items")
    print(f"  a (discrimination): mean={pool_a.mean():.4f}, std={pool_a.std():.4f}, "
          f"min={pool_a.min():.4f}, max={pool_a.max():.4f}, "
          f"skew={stats.skew(pool_a):.4f}")
    print(f"  b (difficulty):     mean={pool_b.mean():.4f}, std={pool_b.std():.4f}, "
          f"min={pool_b.min():.4f}, max={pool_b.max():.4f}, "
          f"skew={stats.skew(pool_b):.4f}")
    print()

    # Distribution comparison plot
    param_plot_path = os.path.join(ARTIFACTS_DIR, "dif_phase2_param_distribution.pdf")
    plot_param_distributions(pool_a, pool_b, param_plot_path)

    # Run simulation
    t0 = time.time()
    df = run_simulation(pool_a, pool_b, n_reps=n_reps)
    elapsed = time.time() - t0
    print(f"\nSimulation completed in {elapsed/60:.1f} minutes")

    # Save CSV
    csv_path = os.path.join(ARTIFACTS_DIR, "dif_phase2_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Power curves
    pdf_path = os.path.join(ARTIFACTS_DIR, "dif_phase2_power_curves.pdf")
    plot_power_curves(df, pdf_path)

    # Results table
    print("\n--- Full Results ---")
    print(df.to_string(index=False))

    # Pass/Fail
    overall = evaluate_pass_fail(df)

    # Phase comparison summary
    print("\n" + "=" * 70)
    print("REAL vs SYNTHETIC PARAMETER COMPARISON")
    print("=" * 70)
    print(f"Real MMLU a: mean={pool_a.mean():.3f}, std={pool_a.std():.3f} | "
          f"Synthetic a: mean≈1.13, std≈0.60 (LogNormal(0,0.5) clipped)")
    print(f"Real MMLU b: mean={pool_b.mean():.3f}, std={pool_b.std():.3f} | "
          f"Synthetic b: mean=0.00, std=1.00 (Normal)")
    print(f"Key difference: real items are EASIER (b shifted left by ~0.84) "
          f"and slightly LESS discriminating")
    print("=" * 70)
