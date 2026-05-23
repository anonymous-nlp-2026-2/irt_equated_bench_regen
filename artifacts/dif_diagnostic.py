"""
DIF Power Diagnostic Analysis.

Three analyses:
1. Raw vs BH vs Bonferroni TPR/FPR comparison across all conditions
2. P-value distribution histograms for selected conditions
3. Effect size parameterization literature comparison (printed)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Import functions from main simulation to ensure consistency
sys.path.insert(0, ARTIFACTS_DIR)
from dif_power_simulation import (
    generate_item_params,
    generate_responses,
    mantel_haenszel_dif,
    logistic_regression_dif,
    N_ITEMS, N_CONTAMINATED, SEED, ALPHA,
)

NS = [30, 50, 80, 100]
DELTAS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]


def setup_simulation():
    rng = np.random.default_rng(SEED)
    a, b = generate_item_params(N_ITEMS, rng)
    contaminated_idx = rng.choice(N_ITEMS, N_CONTAMINATED, replace=False)
    contaminated_mask = np.zeros(N_ITEMS, dtype=bool)
    contaminated_mask[contaminated_idx] = True
    return a, b, contaminated_mask, rng


def run_replication_with_pvalues(n_per_group, delta, a, b, contaminated_mask, rng):
    """Run one replication, return raw p-values for both methods."""
    theta_ref = rng.normal(0, 1, n_per_group)
    theta_foc = rng.normal(0, 1, n_per_group)

    np.random.seed(rng.integers(0, 2**31))
    resp_ref = generate_responses(theta_ref, a, b)
    resp_foc = generate_responses(theta_foc, a, b, delta=delta, contaminated_mask=contaminated_mask)

    responses = np.vstack([resp_ref, resp_foc])
    group = np.array([0]*n_per_group + [1]*n_per_group, dtype=float)

    _, pval_mh, _ = mantel_haenszel_dif(responses, group)
    _, pval_lr = logistic_regression_dif(responses, group)

    return pval_mh, pval_lr


# ─── Analysis 1: Raw vs BH vs Bonferroni TPR/FPR ────────────────────────────

def analysis1_tpr_comparison(n_reps, a, b, contaminated_mask, rng):
    print("=" * 70)
    print(f"ANALYSIS 1: Raw vs BH vs Bonferroni TPR/FPR ({n_reps} reps)")
    print("=" * 70)

    pos = contaminated_mask
    neg = ~contaminated_mask
    n_pos = pos.sum()
    n_neg = neg.sum()

    rows = []
    total = len(NS) * len(DELTAS)
    idx = 0

    # Store p-values for Analysis 2 reuse
    pval_cache = {}

    for n_per_group in NS:
        for delta in DELTAS:
            idx += 1
            t0 = time.time()

            tpr_raw_mh = np.zeros(n_reps)
            fpr_raw_mh = np.zeros(n_reps)
            tpr_bh_mh = np.zeros(n_reps)
            fpr_bh_mh = np.zeros(n_reps)
            tpr_bonf_mh = np.zeros(n_reps)
            fpr_bonf_mh = np.zeros(n_reps)

            tpr_raw_lr = np.zeros(n_reps)
            fpr_raw_lr = np.zeros(n_reps)
            tpr_bh_lr = np.zeros(n_reps)
            fpr_bh_lr = np.zeros(n_reps)
            tpr_bonf_lr = np.zeros(n_reps)
            fpr_bonf_lr = np.zeros(n_reps)

            # Collect per-item p-values for Analysis 2
            cache_key = (n_per_group, delta)
            need_cache = cache_key in [(80, 1.4), (50, 1.0)]
            if need_cache:
                pval_mh_all = np.zeros((n_reps, N_ITEMS))
                pval_lr_all = np.zeros((n_reps, N_ITEMS))

            for rep in range(n_reps):
                pval_mh, pval_lr = run_replication_with_pvalues(
                    n_per_group, delta, a, b, contaminated_mask, rng
                )

                if need_cache:
                    pval_mh_all[rep] = pval_mh
                    pval_lr_all[rep] = pval_lr

                # MH corrections
                reject_raw = pval_mh < ALPHA
                reject_bh, _, _, _ = multipletests(pval_mh, method='fdr_bh', alpha=ALPHA)
                reject_bonf, _, _, _ = multipletests(pval_mh, method='bonferroni', alpha=ALPHA)

                tpr_raw_mh[rep] = reject_raw[pos].sum() / n_pos
                fpr_raw_mh[rep] = reject_raw[neg].sum() / n_neg
                tpr_bh_mh[rep] = reject_bh[pos].sum() / n_pos
                fpr_bh_mh[rep] = reject_bh[neg].sum() / n_neg
                tpr_bonf_mh[rep] = reject_bonf[pos].sum() / n_pos
                fpr_bonf_mh[rep] = reject_bonf[neg].sum() / n_neg

                # LR corrections
                reject_raw = pval_lr < ALPHA
                reject_bh, _, _, _ = multipletests(pval_lr, method='fdr_bh', alpha=ALPHA)
                reject_bonf, _, _, _ = multipletests(pval_lr, method='bonferroni', alpha=ALPHA)

                tpr_raw_lr[rep] = reject_raw[pos].sum() / n_pos
                fpr_raw_lr[rep] = reject_raw[neg].sum() / n_neg
                tpr_bh_lr[rep] = reject_bh[pos].sum() / n_pos
                fpr_bh_lr[rep] = reject_bh[neg].sum() / n_neg
                tpr_bonf_lr[rep] = reject_bonf[pos].sum() / n_pos
                fpr_bonf_lr[rep] = reject_bonf[neg].sum() / n_neg

            if need_cache:
                pval_cache[cache_key] = {
                    'mh': pval_mh_all,
                    'lr': pval_lr_all,
                }

            dt = time.time() - t0

            for method, arrs in [
                ("MH", (tpr_raw_mh, fpr_raw_mh, tpr_bh_mh, fpr_bh_mh, tpr_bonf_mh, fpr_bonf_mh)),
                ("LR", (tpr_raw_lr, fpr_raw_lr, tpr_bh_lr, fpr_bh_lr, tpr_bonf_lr, fpr_bonf_lr)),
            ]:
                rows.append({
                    "n_per_group": n_per_group,
                    "delta": delta,
                    "method": method,
                    "tpr_raw": round(arrs[0].mean(), 4),
                    "fpr_raw": round(arrs[1].mean(), 4),
                    "tpr_bh": round(arrs[2].mean(), 4),
                    "fpr_bh": round(arrs[3].mean(), 4),
                    "tpr_bonf": round(arrs[4].mean(), 4),
                    "fpr_bonf": round(arrs[5].mean(), 4),
                })

            print(f"[{idx}/{total}] N={n_per_group}, δ={delta} ({dt:.1f}s)")
            r_mh = rows[-2]
            r_lr = rows[-1]
            print(f"  MH: raw={r_mh['tpr_raw']:.3f} bh={r_mh['tpr_bh']:.3f} bonf={r_mh['tpr_bonf']:.3f}  "
                  f"(FPR raw={r_mh['fpr_raw']:.3f} bh={r_mh['fpr_bh']:.3f} bonf={r_mh['fpr_bonf']:.3f})")
            print(f"  LR: raw={r_lr['tpr_raw']:.3f} bh={r_lr['tpr_bh']:.3f} bonf={r_lr['tpr_bonf']:.3f}  "
                  f"(FPR raw={r_lr['fpr_raw']:.3f} bh={r_lr['fpr_bh']:.3f} bonf={r_lr['fpr_bonf']:.3f})")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ARTIFACTS_DIR, "dif_diagnostic_tpr_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    return df, pval_cache


# ─── Analysis 2: P-value Distribution Histograms ────────────────────────────

def analysis2_pvalue_distribution(pval_cache, contaminated_mask):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: P-value Distribution Histograms")
    print("=" * 70)

    pos = contaminated_mask
    neg = ~contaminated_mask
    conditions = [(50, 1.0), (80, 1.4)]
    condition_labels = ["N=50, δ=1.0 (Pass threshold)", "N=80, δ=1.4 (Best case)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bonf_threshold = ALPHA / N_ITEMS

    for row_idx, ((n, d), label) in enumerate(zip(conditions, condition_labels)):
        cache = pval_cache[(n, d)]

        for col_idx, (method, method_label) in enumerate([('mh', 'MH'), ('lr', 'LR')]):
            ax = axes[row_idx, col_idx]
            pvals = cache[method]  # (n_reps, n_items)

            # Median p-value per item across replications
            median_pvals = np.median(pvals, axis=0)

            median_contaminated = median_pvals[pos]
            median_clean = median_pvals[neg]

            bins = np.linspace(0, 1, 41)
            ax.hist(median_clean, bins=bins, alpha=0.6, color='steelblue',
                    label=f'Clean items (n={neg.sum()})', density=True)
            ax.hist(median_contaminated, bins=bins, alpha=0.7, color='firebrick',
                    label=f'DIF items (n={pos.sum()})', density=True)

            ax.axvline(ALPHA, color='orange', ls='--', lw=1.5, label=f'p=0.05')
            ax.axvline(bonf_threshold, color='red', ls=':', lw=1.5,
                       label=f'Bonf={bonf_threshold:.1e}')

            ax.set_title(f"{label}\n{method_label}", fontsize=10)
            ax.set_xlabel("Median p-value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7, loc='upper right')

            med_contam = np.median(median_contaminated)
            pct_below_05 = (median_contaminated < 0.05).mean() * 100
            print(f"  {label} | {method_label}: median(DIF items' median p) = {med_contam:.4f}, "
                  f"{pct_below_05:.0f}% below 0.05")

    fig.suptitle("Diagnostic: Per-Item Median P-value Distributions\n"
                 "(median across 1000 replications per item)", fontsize=12, y=1.02)
    fig.tight_layout()
    pdf_path = os.path.join(ARTIFACTS_DIR, "dif_diagnostic_pvalue_dist.pdf")
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")


# ─── Analysis 3: Effect Size Parameterization ───────────────────────────────

def analysis3_effect_size_report():
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Effect Size Parameterization vs Literature")
    print("=" * 70)

    report = """
Our parameterization:
  P_focal(θ) = 1 / (1 + exp(-a * (θ - b + δ)))
  Equivalent to: b_focal = b - δ, so Δb = b_ref - b_focal = δ
  The logit-scale effect at any θ is: a * δ (amplified by item discrimination)

Literature consensus (DIF simulation studies):
  Standard convention: DIF magnitude = b-parameter difference (Δb)
  The logit-scale impact IS a*Δb — this is expected, not a bug.

  ┌──────────────────────────────────────────────────────────────────────┐
  │ Study                          │ Small   │ Medium  │ Large          │
  ├──────────────────────────────────────────────────────────────────────┤
  │ PMC5978487 (3PL, area metric)  │ Δb≈0.50 │ Δb≈0.75 │ Δb≈1.00       │
  │ PMC5965523 (2PL Wald test)     │ 0.3     │ 0.5     │ 0.7           │
  │ PMC5978601 (MH power, Rasch)   │ —       │ 0.426   │ 0.638         │
  │ ETS A/B/C (delta→logit)        │ < 0.43  │ 0.43-0.64│ > 0.64       │
  │ PMC3552735 (Bayesian, 2PL)     │ —       │ 0.42    │ 0.63          │
  └──────────────────────────────────────────────────────────────────────┘

Our δ range vs literature:
  δ = 0.4  →  Small DIF (borderline medium)
  δ = 0.6  →  Medium DIF
  δ = 0.8  →  Medium-to-large DIF
  δ = 1.0  →  Large DIF (exceeds ETS C threshold)
  δ = 1.2+ →  Very large DIF

Conclusion on parameterization:
  Our δ is a b-parameter shift — consistent with literature standard.
  δ=1.0 already exceeds the ETS "C" (large) threshold (0.64 logits).
  With a~LogNormal(0,0.5), mean a≈1.13, so the average logit effect
  at δ=1.0 is ~1.13 logits — a substantial effect.

  Parameterization is NOT the problem. If TPR is low at δ=1.0,
  the cause is statistical (sample size, multiple testing), not
  definitional.
"""
    print(report)
    return report


# ─── Diagnostic Conclusion ───────────────────────────────────────────────────

def print_diagnostic_conclusion(df):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC CONCLUSION")
    print("=" * 70)

    # Focus on the core condition: N=50, δ=1.0
    core = df[(df["n_per_group"] == 50) & (df["delta"] == 1.0)]
    print("\n--- Core condition: N=50, δ=1.0 ---")
    for _, row in core.iterrows():
        method = row["method"]
        ratio_bh = row["tpr_bh"] / row["tpr_raw"] if row["tpr_raw"] > 0 else 0
        ratio_bonf = row["tpr_bonf"] / row["tpr_raw"] if row["tpr_raw"] > 0 else 0
        print(f"  {method}: raw TPR={row['tpr_raw']:.3f} → BH TPR={row['tpr_bh']:.3f} "
              f"({ratio_bh:.1%} retained) → Bonf TPR={row['tpr_bonf']:.3f} ({ratio_bonf:.1%} retained)")
        print(f"        raw FPR={row['fpr_raw']:.3f} → BH FPR={row['fpr_bh']:.3f} "
              f"→ Bonf FPR={row['fpr_bonf']:.3f}")

    # Summary table for δ≥1.0
    print("\n--- TPR comparison for δ ≥ 1.0 ---")
    high_delta = df[df["delta"] >= 1.0].copy()
    print(f"{'N':>4} {'δ':>4} {'Method':>4} {'raw':>6} {'BH':>6} {'Bonf':>6} {'BH/raw':>7}")
    for _, row in high_delta.iterrows():
        ratio = row["tpr_bh"] / row["tpr_raw"] if row["tpr_raw"] > 0 else 0
        print(f"{int(row['n_per_group']):>4} {row['delta']:>4.1f} {row['method']:>4} "
              f"{row['tpr_raw']:>6.3f} {row['tpr_bh']:>6.3f} {row['tpr_bonf']:>6.3f} {ratio:>7.1%}")

    # Diagnosis
    print("\n--- Diagnosis ---")
    mh_50_10 = core[core["method"] == "MH"].iloc[0]
    lr_50_10 = core[core["method"] == "LR"].iloc[0]

    if mh_50_10["tpr_raw"] < 0.20:
        print("  PRIMARY CAUSE: Raw statistical power is insufficient.")
        print("  BH correction is a secondary factor — even without correction, TPR is too low.")
        print("  The per-item tests lack power at N=50 per group with 200 items.")
    elif mh_50_10["tpr_raw"] >= 0.50 and mh_50_10["tpr_bh"] < 0.20:
        print("  PRIMARY CAUSE: BH multiple testing correction.")
        print("  Raw power is adequate, but BH correction with 200 simultaneous tests")
        print("  is too conservative, especially when only 10% of items have true DIF.")
    else:
        raw_ok = mh_50_10["tpr_raw"] >= 0.20
        bh_drop = mh_50_10["tpr_raw"] - mh_50_10["tpr_bh"]
        print(f"  Raw TPR (MH) = {mh_50_10['tpr_raw']:.3f} — {'marginal' if raw_ok else 'low'} base power")
        print(f"  BH drops TPR by {bh_drop:.3f} — {'substantial' if bh_drop > 0.1 else 'modest'} correction penalty")
        print("  COMBINED CAUSE: Both insufficient base power AND BH correction contribute.")

    print("\n  Effect size parameterization: CONSISTENT with literature (δ = Δb, standard convention).")
    print("  δ=1.0 is a 'large' effect by ETS standards — the low TPR is not due to weak effects.")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print(f"DIF Diagnostic Analysis — {n_reps} replications per condition")
    print(f"Items: {N_ITEMS}, DIF items: {N_CONTAMINATED}, Seed: {SEED}")
    print()

    a, b, contaminated_mask, rng = setup_simulation()

    t0 = time.time()

    # Analysis 1 + collect p-values for Analysis 2
    df, pval_cache = analysis1_tpr_comparison(n_reps, a, b, contaminated_mask, rng)

    # Analysis 2
    if (50, 1.0) in pval_cache and (80, 1.4) in pval_cache:
        analysis2_pvalue_distribution(pval_cache, contaminated_mask)
    else:
        print("\nSkipping Analysis 2: required conditions not in cache (need full run)")

    # Analysis 3
    effect_report = analysis3_effect_size_report()

    # Diagnostic conclusion
    print_diagnostic_conclusion(df)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
