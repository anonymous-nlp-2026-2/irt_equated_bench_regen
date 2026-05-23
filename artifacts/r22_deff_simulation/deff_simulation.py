#!/usr/bin/env python3
"""
R22: DEFF simple-mean vs weighted-mean simulation.

Methodology (matching the paper):
  1. Generate binary responses with within-family ICC structure
  2. Inject DIF for a known fraction of items (additive boost for focal group)
  3. Compute per-item chi2 via BOTH:
     - Unstratified z-test (no family-effect control)
     - Stratified MH test (partial family-effect control via score strata)
  4. Estimate per-item ICC via ANOVA
  5. Rao-Scott correction: chi2_adj = chi2 / (1 + (m_bar - 1) * ICC_i)
  6. Compare which m_bar (simple vs weighted) recovers GT C% better
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from scipy import sparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path(__file__).parent

# ── Constants ─────────────────────────────────────────────────
N_ITEMS = 2000
GT_DIF_RATE = 0.30
N_SEEDS = 50
BASE_RATE = 0.55
DIF_DELTA = 0.30
SIGMA_W = 0.05
ALPHA = 0.05
CHI2_CRIT = chi2_dist.ppf(1 - ALPHA, 1)
HOMOG_ICC = 0.15

TIER_ICC = {10: 0.25, 50: 0.22, 200: 0.14, 99999: 0.08}

SCENARIOS = {
    "A_uniform": {
        "families": [(30, 35)],
        "icc_by_size": True,
        "desc": "Uniform: 30×35",
    },
    "B_skewed": {
        "families": [(2, 300), (8, 50), (20, 10)],
        "icc_by_size": True,
        "desc": "Skewed (METABENCH-like)",
    },
    "C_extreme": {
        "families": [(1, 500), (5, 50), (24, 10)],
        "icc_by_size": True,
        "desc": "Extreme skew",
    },
    "D_homog_icc": {
        "families": [(2, 300), (8, 50), (20, 10)],
        "icc_by_size": False,
        "desc": "Skewed, homogeneous ICC",
    },
}


def get_tier_icc(size, by_size=True):
    if not by_size:
        return HOMOG_ICC
    for bound, icc in sorted(TIER_ICC.items()):
        if size <= bound:
            return icc
    return 0.08


def sigma_b_for_icc(target_icc):
    residual = SIGMA_W**2 + BASE_RATE * (1 - BASE_RATE)
    return np.sqrt(target_icc * residual / (1 - target_icc))


# ── Response generation ───────────────────────────────────────

def generate_responses(families_spec, icc_by_size, rng, dif_mask, dif_deltas):
    responses, fam_labels = [], []
    fam_id = 0
    for n_fam, size in families_spec:
        sb = sigma_b_for_icc(get_tier_icc(size, icc_by_size))
        for _ in range(n_fam):
            is_focal = (fam_id % 2 == 1)
            fe = rng.normal(0, sb, size=N_ITEMS)
            for _ in range(size):
                noise = rng.normal(0, SIGMA_W, size=N_ITEMS)
                p = BASE_RATE + fe + noise
                if is_focal:
                    p[dif_mask] += dif_deltas[dif_mask]
                p = np.clip(p, 0.01, 0.99)
                responses.append((rng.random(N_ITEMS) < p).astype(np.int8))
            fam_labels.extend([fam_id] * size)
            fam_id += 1
    return np.array(responses), np.array(fam_labels)


# ── Z-test ────────────────────────────────────────────────────

def chi2_ztest(responses, fam_labels):
    focal = (fam_labels % 2 == 1)
    n_f, n_r = focal.sum(), (~focal).sum()
    mf = responses[focal].mean(axis=0)
    mr = responses[~focal].mean(axis=0)
    pp = responses.mean(axis=0)
    se = np.sqrt(np.clip(pp * (1 - pp), 1e-10, None) * (1/n_f + 1/n_r))
    z = (mf - mr) / np.where(se > 0, se, 1)
    return z**2


# ── MH test (vectorized) ─────────────────────────────────────

def chi2_mh(responses, fam_labels, n_strata=5):
    """Stratified MH chi-square, vectorized across items."""
    n_models, n_items = responses.shape
    focal = (fam_labels % 2 == 1)
    scores = responses.sum(axis=1)

    try:
        strata = pd.qcut(scores, q=n_strata, labels=False, duplicates="drop")
    except ValueError:
        strata = pd.qcut(scores, q=3, labels=False, duplicates="drop")

    numer = np.zeros(n_items, dtype=np.float64)
    denom = np.zeros(n_items, dtype=np.float64)

    for s in np.unique(strata):
        sm = (strata == s)
        fm = focal & sm
        rm = (~focal) & sm
        n_f, n_r = fm.sum(), rm.sum()
        if n_f < 2 or n_r < 2:
            continue

        a = responses[rm].sum(axis=0).astype(np.float64)
        c = responses[fm].sum(axis=0).astype(np.float64)
        b = float(n_r) - a
        d = float(n_f) - c
        t = float(n_f + n_r)

        numer += a - (a + b) * (a + c) / t
        ev = (a + b) * (c + d) * (a + c) * (b + d) / (t**2 * (t - 1))
        denom += ev

    chi2 = np.where(denom > 0, numer**2 / denom, 0.0)
    return chi2


# ── ANOVA ICC ─────────────────────────────────────────────────

def anova_icc(responses, fam_labels):
    N, n_items = responses.shape
    uniq = np.unique(fam_labels)
    K = len(uniq)
    f2i = {f: i for i, f in enumerate(uniq)}
    fi = np.array([f2i[f] for f in fam_labels])

    F = sparse.csc_matrix((np.ones(N), (np.arange(N), fi)), shape=(N, K))
    fs = np.array(F.sum(axis=0)).ravel()

    X = responses.astype(np.float64)
    gm = X.mean(axis=0)
    fm_sums = F.T @ X
    fm_means = fm_sums / fs[:, None]

    SSB = (fs[:, None] * (fm_means - gm[None, :]) ** 2).sum(axis=0)
    SSW = (X ** 2).sum(axis=0) - (fs[:, None] * fm_means ** 2).sum(axis=0)

    MSB = SSB / max(K - 1, 1)
    MSW = SSW / max(N - K, 1)

    m_w = (fs ** 2).sum() / fs.sum()
    m_s = fs.mean()

    d = MSB + (m_w - 1) * MSW
    icc = np.where(d != 0, (MSB - MSW) / d, 0.0)
    return np.clip(icc, 0, 1), m_s, m_w


# ── Analytical DEFF ───────────────────────────────────────────

def deff_true_analytical(families_spec, icc_by_size):
    total_n = sum(n * m for n, m in families_spec)
    s = sum(n * m * (m - 1) * get_tier_icc(m, icc_by_size)
            for n, m in families_spec)
    return 1 + s / total_n


# ── Run one seed ──────────────────────────────────────────────

def run_one(sc_name, sc_cfg, seed):
    rng = np.random.default_rng(seed)

    n_dif = int(N_ITEMS * GT_DIF_RATE)
    dif_mask = np.zeros(N_ITEMS, dtype=bool)
    dif_mask[rng.choice(N_ITEMS, n_dif, replace=False)] = True
    dif_deltas = np.zeros(N_ITEMS)
    dif_deltas[dif_mask] = rng.uniform(DIF_DELTA * 0.8, DIF_DELTA * 1.2, n_dif)

    resp, fl = generate_responses(
        sc_cfg["families"], sc_cfg["icc_by_size"], rng, dif_mask, dif_deltas)

    # Chi2 — both methods
    c2_z = chi2_ztest(resp, fl)
    c2_mh = chi2_mh(resp, fl)

    # ICC
    icc, m_s, m_w = anova_icc(resp, fl)
    mi = icc.mean()

    # DEFF scalars
    ds = 1 + (m_s - 1) * mi
    dw = 1 + (m_w - 1) * mi
    dt = deff_true_analytical(sc_cfg["families"], sc_cfg["icc_by_size"])

    # Empirical DEFF (mean chi2 for null items)
    emp_z = c2_z[~dif_mask].mean()
    emp_mh = c2_mh[~dif_mask].mean()

    # Per-item DEFF for Rao-Scott
    dpi_s = np.clip(1 + (m_s - 1) * icc, 1, None)
    dpi_w = np.clip(1 + (m_w - 1) * icc, 1, None)

    gt = GT_DIF_RATE * 100

    def cpct(chi2_arr):
        return (chi2_arr > CHI2_CRIT).mean() * 100

    naive_z = cpct(c2_z)
    naive_mh = cpct(c2_mh)
    adj_z_s = cpct(c2_z / dpi_s)
    adj_z_w = cpct(c2_z / dpi_w)
    adj_mh_s = cpct(c2_mh / dpi_s)
    adj_mh_w = cpct(c2_mh / dpi_w)

    return {
        "scenario": sc_name, "seed": seed,
        "n_models": len(fl), "n_families": len(np.unique(fl)),
        "m_s": m_s, "m_w": m_w, "icc_anova": mi,
        "deff_simple": ds, "deff_weighted": dw,
        "deff_true": dt,
        "deff_emp_z": emp_z, "deff_emp_mh": emp_mh,
        "naive_z": naive_z, "naive_mh": naive_mh,
        "adj_z_s": adj_z_s, "adj_z_w": adj_z_w,
        "adj_mh_s": adj_mh_s, "adj_mh_w": adj_mh_w,
        "bias_z_s": adj_z_s - gt, "bias_z_w": adj_z_w - gt,
        "bias_mh_s": adj_mh_s - gt, "bias_mh_w": adj_mh_w - gt,
        "gt": gt,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R22: DEFF Simple-Mean vs Weighted-Mean Simulation")
    print("=" * 70)

    rows = []
    for sc, cfg in SCENARIOS.items():
        print(f"\n{sc}: {cfg['desc']}")
        for s in range(N_SEEDS):
            rows.append(run_one(sc, cfg, s))
            if (s + 1) % 10 == 0:
                print(f"  {s+1}/{N_SEEDS}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "deff_simulation_results.csv", index=False)

    agg = df.groupby("scenario").agg(
        n_models=("n_models", "first"),
        n_fam=("n_families", "first"),
        m_s=("m_s", "mean"), m_w=("m_w", "mean"),
        icc=("icc_anova", "mean"),
        ds=("deff_simple", "mean"), dw=("deff_weighted", "mean"),
        dt=("deff_true", "first"),
        de_z=("deff_emp_z", "mean"), de_mh=("deff_emp_mh", "mean"),
        naive_z=("naive_z", "mean"), naive_mh=("naive_mh", "mean"),
        az_s=("adj_z_s", "mean"), az_w=("adj_z_w", "mean"),
        am_s=("adj_mh_s", "mean"), am_w=("adj_mh_w", "mean"),
        am_s_std=("adj_mh_s", "std"), am_w_std=("adj_mh_w", "std"),
        bm_s=("bias_mh_s", "mean"), bm_w=("bias_mh_w", "mean"),
        rmse_m_s=("bias_mh_s", lambda x: np.sqrt((x**2).mean())),
        rmse_m_w=("bias_mh_w", lambda x: np.sqrt((x**2).mean())),
    ).reset_index()

    print("\n" + "=" * 70)
    print("DEFF ESTIMATES")
    print("=" * 70)
    print(f"{'Scenario':<14} {'DEFF_s':>7} {'DEFF_w':>7} {'DEFF_t':>7} "
          f"{'EmpZ':>7} {'EmpMH':>7}  {'|s-MH|':>7} {'|w-MH|':>7}")
    for _, r in agg.iterrows():
        ds, dw, dmh = r['ds'], r['dw'], r['de_mh']
        print(f"{r['scenario']:<14} {ds:7.2f} {dw:7.2f} {r['dt']:7.2f} "
              f"{r['de_z']:7.2f} {dmh:7.2f}  {abs(ds-dmh):7.2f} {abs(dw-dmh):7.2f}")

    print("\n" + "=" * 70)
    print("C% RECOVERY (MH-based, Rao-Scott)")
    print("=" * 70)
    print(f"{'Scenario':<14} {'Naive':>7} {'Adj(s)':>7} {'Adj(w)':>7} "
          f"{'Bias(s)':>8} {'Bias(w)':>8} {'Winner':>8}")
    for _, r in agg.iterrows():
        w = "SIMPLE" if abs(r['bm_s']) < abs(r['bm_w']) else "WEIGHTED"
        print(f"{r['scenario']:<14} {r['naive_mh']:7.1f} {r['am_s']:7.1f} {r['am_w']:7.1f} "
              f"{r['bm_s']:+8.1f} {r['bm_w']:+8.1f} {w:>8}")

    make_figure(df, agg)
    write_report(agg)
    print(f"\nDone → {OUT}")


def make_figure(df, agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = agg["scenario"].values
    x = np.arange(len(scenarios))
    w = 0.25
    lbl = {"A": "Uniform\n30×35", "B": "Skewed\n(METABENCH)",
           "C": "Extreme\nskew", "D": "Skewed,\nhomog ICC"}
    xlbl = [lbl.get(s.split("_")[0], s) for s in scenarios]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) DEFF
    ax = axes[0]
    ax.bar(x - w, agg["ds"].values, w, label="Simple", color="#4878CF")
    ax.bar(x, agg["dw"].values, w, label="Weighted", color="#D65F5F")
    ax.bar(x + w, agg["de_mh"].values, w, label="Empirical (MH)", color="#6ACC65")
    ax.set_ylabel("DEFF"); ax.set_title("(a) DEFF estimates")
    ax.set_xticks(x); ax.set_xticklabels(xlbl, fontsize=7)
    ax.legend(fontsize=7, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (b) Bias
    ax = axes[1]
    ax.bar(x - w/2, agg["bm_s"].values, w, label="Simple", color="#4878CF")
    ax.bar(x + w/2, agg["bm_w"].values, w, label="Weighted", color="#D65F5F")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Mean Bias (Adj − GT 30%)"); ax.set_title("(b) MH Rao-Scott bias")
    ax.set_xticks(x); ax.set_xticklabels(xlbl, fontsize=7)
    ax.legend(fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (c) Absolute bias
    ax = axes[2]
    ax.bar(x - w/2, np.abs(agg["bm_s"].values), w, label="Simple", color="#4878CF")
    ax.bar(x + w/2, np.abs(agg["bm_w"].values), w, label="Weighted", color="#D65F5F")
    ax.set_ylabel("|Bias|"); ax.set_title("(c) |Bias| comparison")
    ax.set_xticks(x); ax.set_xticklabels(xlbl, fontsize=7)
    ax.legend(fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("DEFF: Simple-Mean vs Weighted-Mean  (GT = 30%)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "deff_simulation_figure.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT / "deff_simulation_figure.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved figure.")


def write_report(agg):
    lines = [
        "# R22: DEFF Simple-Mean vs Weighted-Mean Simulation",
        "",
        "## Setup",
        f"- Items: {N_ITEMS}, GT DIF rate: {GT_DIF_RATE*100:.0f}%",
        f"- DIF boost: Δp ∈ [{DIF_DELTA*0.8:.2f}, {DIF_DELTA*1.2:.2f}]",
        f"- σ_w={SIGMA_W}, α={ALPHA}, Seeds={N_SEEDS}",
        "",
        "## DEFF Comparison",
        "",
        "| Scenario | m̄(s) | m̄(w) | ICC | DEFF(s) | DEFF(w) | DEFF(true) | Emp(z) | Emp(MH) | |s−MH| | |w−MH| |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['scenario']} | {r['m_s']:.1f} | {r['m_w']:.1f} "
            f"| {r['icc']:.4f} | {r['ds']:.2f} | {r['dw']:.2f} "
            f"| {r['dt']:.2f} | {r['de_z']:.2f} | {r['de_mh']:.2f} "
            f"| {abs(r['ds']-r['de_mh']):.2f} | {abs(r['dw']-r['de_mh']):.2f} |"
        )

    lines += [
        "",
        "## C% Recovery (MH Rao-Scott)",
        "",
        "| Scenario | Naive | Adj(s) | Adj(w) | Bias(s) | Bias(w) | RMSE(s) | RMSE(w) | Winner |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        w = "Simple" if abs(r['bm_s']) < abs(r['bm_w']) else "Weighted"
        lines.append(
            f"| {r['scenario']} | {r['naive_mh']:.1f} "
            f"| {r['am_s']:.1f}±{r['am_s_std']:.1f} "
            f"| {r['am_w']:.1f}±{r['am_w_std']:.1f} "
            f"| {r['bm_s']:+.1f} | {r['bm_w']:+.1f} "
            f"| {r['rmse_m_s']:.2f} | {r['rmse_m_w']:.2f} | **{w}** |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The MH test stratifies by total score, partially controlling for family",
        "effects. This reduces the effective DEFF relative to unstratified tests.",
        "",
        "The ANOVA ICC (computed from all families) is much lower than tier-specific",
        "ICCs because the ANOVA formula uses m̄_weighted in the denominator, which",
        "deflates the estimate for unbalanced designs.",
        "",
        "### When ICC decreases with family size (A–C):",
        "The ANOVA ICC reflects a weighted average pulled down by large families.",
        "Simple m̄ × low ICC gives modest DEFF that better matches the MH effective",
        "DEFF, since stratification already removes much of the large-family clustering.",
        "",
        "### When ICC is constant (D):",
        "Weighted m̄ is theoretically correct and should give better recovery.",
    ]

    (OUT / "deff_simulation_report.md").write_text("\n".join(lines) + "\n")
    print("Saved report.")


if __name__ == "__main__":
    main()
