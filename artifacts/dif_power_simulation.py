"""
DIF Power Monte Carlo Simulation for Benchmark Contamination Detection.

Simulates 2PL IRT response data with injected contamination (DIF) on a subset
of items, then evaluates whether Mantel-Haenszel and Logistic Regression DIF
methods can detect the contaminated items with sufficient statistical power.

v2: BH correction, FPR>0.15 kill, GRAY zone, verdict column.
    Vectorized MH (no item loop) + batched IRLS LR for speed.
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

# ─── Data Generation ────────────────────────────────────────────────────────

def generate_item_params(n_items, rng):
    a = rng.lognormal(mean=0.0, sigma=0.5, size=n_items)
    a = np.clip(a, 0.3, 2.5)
    b = rng.normal(0.0, 1.0, size=n_items)
    return a, b


def generate_responses(theta, a, b, delta=None, contaminated_mask=None):
    b_adj = b.copy()
    if delta is not None and contaminated_mask is not None:
        b_adj[contaminated_mask] = b[contaminated_mask] - delta
    logit = a[np.newaxis, :] * (theta[:, np.newaxis] - b_adj[np.newaxis, :])
    prob = expit(logit)
    return (np.random.random(prob.shape) < prob).astype(np.int8)


# ─── Mantel-Haenszel DIF (vectorized across items) ────────────────────────

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


# ─── Logistic Regression DIF (batched IRLS) ───────────────────────────────

def _batched_logistic_ll(X, Y, max_iter=25, tol=1e-8):
    """Fit logistic regression for all items simultaneously via batched IRLS.

    X: (N, p) shared design matrix
    Y: (N, J) response matrix (columns = items)
    Returns: (J,) log-likelihoods
    """
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

    # Skip constant items
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


# ─── Single Replication ─────────────────────────────────────────────────────

def run_single_replication(n_per_group, delta, a, b, contaminated_mask, rng):
    theta_ref = rng.normal(0, 1, n_per_group)
    theta_foc = rng.normal(0, 1, n_per_group)

    np.random.seed(rng.integers(0, 2**31))
    resp_ref = generate_responses(theta_ref, a, b)
    resp_foc = generate_responses(theta_foc, a, b, delta=delta, contaminated_mask=contaminated_mask)

    responses = np.vstack([resp_ref, resp_foc])
    group = np.array([0]*n_per_group + [1]*n_per_group, dtype=float)

    results = {}

    # --- MH ---
    delta_mh, pval_mh, ets_class = mantel_haenszel_dif(responses, group)
    reject_mh, _, _, _ = multipletests(pval_mh, alpha=ALPHA, method="fdr_bh")
    results["mh"] = {
        "flagged": reject_mh,
        "stat": np.abs(np.nan_to_num(delta_mh, nan=0.0)),
        "ets_class": ets_class,
    }

    # --- LR ---
    lr_chi2, pval_lr = logistic_regression_dif(responses, group)
    reject_lr, _, _, _ = multipletests(pval_lr, alpha=ALPHA, method="fdr_bh")
    results["lr"] = {
        "flagged": reject_lr,
        "stat": lr_chi2,
    }

    return results


# ─── Evaluate Metrics ────────────────────────────────────────────────────────

def compute_metrics(all_results, contaminated_mask):
    pos = contaminated_mask
    neg = ~contaminated_mask
    n_pos = pos.sum()
    n_neg = neg.sum()

    metrics = {}
    for method in ["mh", "lr"]:
        tpr_arr = np.empty(len(all_results))
        fpr_arr = np.empty(len(all_results))
        auc_arr = np.empty(len(all_results))
        ets_c_tpr_arr = np.empty(len(all_results)) if method == "mh" else None
        ets_c_fpr_arr = np.empty(len(all_results)) if method == "mh" else None

        for i, res in enumerate(all_results):
            flagged = res[method]["flagged"]
            stat = res[method]["stat"]

            tp = flagged[pos].sum()
            fp = flagged[neg].sum()
            tpr_arr[i] = tp / max(n_pos, 1)
            fpr_arr[i] = fp / max(n_neg, 1)

            try:
                auc_arr[i] = roc_auc_score(pos.astype(int), stat)
            except ValueError:
                auc_arr[i] = 0.5

            if method == "mh":
                ets_c = res[method]["ets_class"] == "C"
                ets_c_tpr_arr[i] = (ets_c & pos).sum() / max(n_pos, 1)
                ets_c_fpr_arr[i] = (ets_c & neg).sum() / max(n_neg, 1)

        metrics[method] = {
            "tpr": float(tpr_arr.mean()),
            "fpr": float(fpr_arr.mean()),
            "auc": float(auc_arr.mean()),
        }
        if method == "mh":
            metrics[method]["ets_c_tpr"] = float(ets_c_tpr_arr.mean())
            metrics[method]["ets_c_fpr"] = float(ets_c_fpr_arr.mean())
        else:
            metrics[method]["ets_c_tpr"] = np.nan
            metrics[method]["ets_c_fpr"] = np.nan

    return metrics


# ─── Verdict Logic ───────────────────────────────────────────────────────────

def assign_row_verdict(row):
    n = row["n_per_group"]
    delta = row["delta"]
    tpr = row["tpr"]
    fpr = row["fpr"]

    if fpr > 0.15:
        return "FAIL"
    if delta >= 1.0 and n >= 80 and tpr < 0.50:
        return "FAIL"
    if delta >= 1.0 and n >= 50:
        if tpr >= 0.70 and fpr <= 0.10:
            return "PASS"
        if tpr >= 0.50:
            return "GRAY"
        return "FAIL"
    return "N/A"


def evaluate_pass_fail(df):
    print("\n" + "=" * 70)
    print("PASS / GRAY / FAIL EVALUATION (BH-corrected FPR)")
    print("=" * 70)

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        print(f"\n--- {method} ---")

        fpr_kills = sub[sub["fpr_exceeds_015"]]
        if len(fpr_kills) > 0:
            print(f"  FPR > 0.15 violations: {len(fpr_kills)} conditions")
            for _, r in fpr_kills.iterrows():
                print(f"    N={int(r['n_per_group'])}, δ={r['delta']}: FPR={r['fpr']:.3f}")

        core = sub[(sub["delta"] >= 1.0) & (sub["n_per_group"] >= 50)]
        if len(core) > 0:
            n_pass = (core["verdict"] == "PASS").sum()
            n_gray = (core["verdict"] == "GRAY").sum()
            n_fail = (core["verdict"] == "FAIL").sum()
            print(f"  Core region (δ≥1.0, N≥50): {n_pass} PASS, {n_gray} GRAY, {n_fail} FAIL out of {len(core)}")
            min_tpr = core["tpr"].min()
            max_fpr = core["fpr"].max()
            print(f"    min TPR = {min_tpr:.3f}, max FPR = {max_fpr:.3f}")

        plus = sub[(sub["delta"] >= 1.4) & (sub["n_per_group"] >= 50)]
        if len(plus) > 0:
            plus_min_tpr = plus["tpr"].min()
            plus_ok = plus_min_tpr >= 0.80
            tag = "PASS+" if plus_ok else "NOT MET"
            print(f"  Pass+ (δ≥1.4, N≥50): min TPR = {plus_min_tpr:.3f} {'≥' if plus_ok else '<'} 0.80 → {tag}")

        verdicts = core["verdict"].values if len(core) > 0 else []
        if "FAIL" in verdicts:
            overall = "FAIL"
            fail_rows = core[core["verdict"] == "FAIL"]
            trigger = f"N={int(fail_rows.iloc[0]['n_per_group'])},δ={fail_rows.iloc[0]['delta']}"
        elif "GRAY" in verdicts:
            overall = "GRAY"
            gray_rows = core[core["verdict"] == "GRAY"]
            trigger = f"N={int(gray_rows.iloc[0]['n_per_group'])},δ={gray_rows.iloc[0]['delta']}"
        elif len(verdicts) > 0 and all(v == "PASS" for v in verdicts):
            overall = "PASS"
            trigger = "all core conditions met"
        else:
            overall = "N/A"
            trigger = "no core conditions"

        print(f"\n  *** {method} OVERALL: {overall} (trigger: {trigger}) ***")

    print("\n" + "=" * 70)


# ─── Main Simulation ────────────────────────────────────────────────────────

def run_simulation(n_reps=1000):
    ns = [30, 50, 80, 100]
    deltas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

    rng = np.random.default_rng(SEED)
    a, b = generate_item_params(N_ITEMS, rng)

    contaminated_idx = rng.choice(N_ITEMS, N_CONTAMINATED, replace=False)
    contaminated_mask = np.zeros(N_ITEMS, dtype=bool)
    contaminated_mask[contaminated_idx] = True

    all_rows = []
    total_conditions = len(ns) * len(deltas)
    cond_idx = 0

    for n_per_group, delta in product(ns, deltas):
        cond_idx += 1
        t_cond = time.time()
        print(f"[{cond_idx}/{total_conditions}] N={n_per_group}, δ={delta}, reps={n_reps}")

        rep_results = []
        for rep in range(n_reps):
            res = run_single_replication(n_per_group, delta, a, b, contaminated_mask, rng)
            rep_results.append(res)

        metrics = compute_metrics(rep_results, contaminated_mask)
        dt = time.time() - t_cond

        for method_name, method_key in [("MH", "mh"), ("LR", "lr")]:
            m = metrics[method_key]
            row = {
                "n_per_group": n_per_group,
                "delta": delta,
                "method": method_name,
                "tpr": round(m["tpr"], 4),
                "fpr": round(m["fpr"], 4),
                "auc": round(m["auc"], 4),
                "ets_c_tpr": round(m["ets_c_tpr"], 4) if not np.isnan(m["ets_c_tpr"]) else "",
                "ets_c_fpr": round(m["ets_c_fpr"], 4) if not np.isnan(m["ets_c_fpr"]) else "",
                "n_reps": n_reps,
                "fpr_exceeds_015": m["fpr"] > 0.15,
            }
            row["verdict"] = assign_row_verdict(row)
            all_rows.append(row)
            v_tag = f" [{row['verdict']}]" if row['verdict'] != "N/A" else ""
            print(f"  {method_name}: TPR={m['tpr']:.3f}  FPR={m['fpr']:.3f}  AUC={m['auc']:.3f}{v_tag}")

        print(f"  ({dt:.1f}s)")

    df = pd.DataFrame(all_rows)
    return df


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_power_curves(df, outpath):
    methods = df["method"].unique()
    ns = sorted(df["n_per_group"].unique())
    colors = {30: "#1f77b4", 50: "#ff7f0e", 80: "#2ca02c", 100: "#d62728"}

    fig, axes = plt.subplots(2, len(methods), figsize=(6 * len(methods), 8),
                             sharex=True, sharey="row")
    if len(methods) == 1:
        axes = axes.reshape(-1, 1)

    for col, method in enumerate(methods):
        sub = df[df["method"] == method]
        for n in ns:
            cond = sub[sub["n_per_group"] == n]
            axes[0, col].plot(cond["delta"], cond["tpr"], "o-",
                              color=colors[n], label=f"N={n}")
            axes[1, col].plot(cond["delta"], cond["fpr"], "s--",
                              color=colors[n], label=f"N={n}")

        axes[0, col].set_title(f"{method} — Power (TPR)")
        axes[0, col].set_ylabel("TPR")
        axes[0, col].axhline(0.70, color="gray", ls=":", lw=0.8)
        axes[0, col].axhline(0.80, color="gray", ls=":", lw=0.8)
        axes[0, col].axhline(0.50, color="orange", ls=":", lw=0.8)
        axes[0, col].legend(fontsize=7, loc="lower right")
        axes[0, col].set_ylim(-0.02, 1.02)

        axes[1, col].set_title(f"{method} — False Positive Rate (BH-corrected)")
        axes[1, col].set_xlabel("δ (contamination effect, logits)")
        axes[1, col].set_ylabel("FPR")
        axes[1, col].axhline(0.05, color="gray", ls=":", lw=0.8)
        axes[1, col].axhline(0.10, color="orange", ls=":", lw=0.8)
        axes[1, col].axhline(0.15, color="red", ls=":", lw=0.8)
        axes[1, col].legend(fontsize=7)
        axes[1, col].set_ylim(-0.02, 0.50)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {outpath}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print(f"Starting DIF power simulation with {n_reps} replications per condition")
    print(f"Items: {N_ITEMS}, Contaminated: {N_CONTAMINATED}, Strata: {N_STRATA}")
    print(f"Seed: {SEED}")
    print(f"Multiple testing correction: Benjamini-Hochberg (FDR)")
    print()

    t0 = time.time()
    df = run_simulation(n_reps=n_reps)
    elapsed = time.time() - t0
    print(f"\nSimulation completed in {elapsed/60:.1f} minutes")

    csv_path = os.path.join(ARTIFACTS_DIR, "dif_power_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    pdf_path = os.path.join(ARTIFACTS_DIR, "dif_power_curves.pdf")
    plot_power_curves(df, pdf_path)

    print("\n--- Full Results Table ---")
    print(df.to_string(index=False))

    evaluate_pass_fail(df)
