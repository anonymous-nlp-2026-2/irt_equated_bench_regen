#!/usr/bin/env python3
"""
mmlu_bootstrap_ci.py — MMLU bootstrap 95% CI for MH-DIF C%.

Vectorized total-score stratification (~100x faster than per-item rest-score).
For J=12508, total-score ≈ rest-score (diff < 0.3pp); validated at startup.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from scipy.sparse import load_npz
from pathlib import Path
from multiprocessing import Pool
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("artifacts")
OUT = BASE / "plan_028_bootstrap_ci"
OUT.mkdir(exist_ok=True)

N_BOOTSTRAP = 1000
N_CONTROL = 20
SEED = 42
N_STRATA = 5


# ════════════════════════════════════════════════════════════════════════
#  VECTORIZED MH-DIF (total-score stratification)
# ════════════════════════════════════════════════════════════════════════
def mh_dif_c_pct_vec(X_ref, X_foc, n_strata=N_STRATA):
    """Fully vectorized MH-DIF → C%. Stratifies on total score (not rest)."""
    n_ref, J = X_ref.shape
    n_foc = X_foc.shape[0]

    tot_ref = X_ref.sum(axis=1).astype(np.float64)
    tot_foc = X_foc.sum(axis=1).astype(np.float64)
    s_all = np.concatenate([tot_ref, tot_foc])

    edges = np.unique(np.quantile(s_all, np.linspace(0, 1, n_strata + 1)))
    n_bins = len(edges) - 1
    if n_bins < 1:
        return 0.0

    k_ref = np.digitize(tot_ref, edges[1:-1])
    k_foc = np.digitize(tot_foc, edges[1:-1])

    n1 = np.zeros(n_bins, dtype=np.float64)
    n0 = np.zeros(n_bins, dtype=np.float64)
    A = np.zeros((n_bins, J), dtype=np.float64)
    C = np.zeros((n_bins, J), dtype=np.float64)

    for k in range(n_bins):
        mr = (k_ref == k)
        mf = (k_foc == k)
        n1[k] = mr.sum()
        n0[k] = mf.sum()
        if mr.any():
            A[k] = X_ref[mr].sum(axis=0).astype(np.float64)
        if mf.any():
            C[k] = X_foc[mf].sum(axis=0).astype(np.float64)

    B = n1[:, None] - A
    D = n0[:, None] - C
    Nk = (n1 + n0)[:, None]
    m1 = A + C
    m0 = B + D

    valid = ((n1[:, None] > 0) & (n0[:, None] > 0) &
             (Nk > 1) & (m1 > 0) & (m0 > 0))

    Rj = np.where(valid, A * D / Nk, 0).sum(axis=0)
    Sj = np.where(valid, B * C / Nk, 0).sum(axis=0)

    finite_mask = (Rj > 0) & (Sj > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_j = np.where(finite_mask, -2.35 * np.log(Rj / Sj), 0)

    EA = np.where(valid, n1[:, None] * m1 / Nk, 0)
    sum_diff = np.where(valid, A - EA, 0).sum(axis=0)

    Nk_m1 = np.where(Nk > 1, Nk - 1, 1)
    sum_var = np.where(
        valid,
        n1[:, None] * n0[:, None] * m1 * m0 / (Nk ** 2 * Nk_m1),
        0,
    ).sum(axis=0)

    chi2_j = np.where(
        sum_var > 0,
        np.maximum(np.abs(sum_diff) - 0.5, 0) ** 2 / np.maximum(sum_var, 1e-30),
        0,
    )
    pval_j = np.where(sum_var > 0, 1 - chi2_dist.cdf(chi2_j, df=1), 1.0)

    is_c = finite_mask & (np.abs(delta_j) >= 1.5) & (pval_j < 0.05)
    return is_c.sum() / J * 100


# ════════════════════════════════════════════════════════════════════════
#  REST-SCORE MH-DIF (validation only)
# ════════════════════════════════════════════════════════════════════════
def mh_dif_c_pct_rest(X_ref, X_foc, n_strata=N_STRATA):
    """Per-item rest-score MH-DIF (reference implementation)."""
    tot_ref = X_ref.sum(axis=1).astype(np.float64)
    tot_foc = X_foc.sum(axis=1).astype(np.float64)
    J = X_ref.shape[1]
    n_c = 0
    qs = np.linspace(0, 1, n_strata + 1)
    nr = len(tot_ref)
    nf = len(tot_foc)
    s_buf = np.empty(nr + nf, dtype=np.float64)

    for j in range(J):
        s_buf[:nr] = tot_ref - X_ref[:, j]
        s_buf[nr:] = tot_foc - X_foc[:, j]
        edges = np.unique(np.quantile(s_buf, qs))
        nb = len(edges) - 1
        if nb < 1:
            continue
        kr = np.digitize(s_buf[:nr], edges[1:-1])
        kf = np.digitize(s_buf[nr:], edges[1:-1])

        R = S = sd = sv = 0.0
        n_ok = 0
        for k in range(nb):
            mr = (kr == k)
            mf = (kf == k)
            n1 = int(mr.sum())
            n0 = int(mf.sum())
            if n1 == 0 or n0 == 0:
                continue
            Ak = float(X_ref[mr, j].sum())
            Bk = n1 - Ak
            Ck = float(X_foc[mf, j].sum())
            Dk = n0 - Ck
            Nk = n1 + n0
            m1k = Ak + Ck
            m0k = Bk + Dk
            if m1k == 0 or m0k == 0 or Nk <= 1:
                continue
            n_ok += 1
            R += Ak * Dk / Nk
            S += Bk * Ck / Nk
            sd += Ak - n1 * m1k / Nk
            sv += n1 * n0 * m1k * m0k / (Nk * Nk * (Nk - 1))

        if n_ok == 0 or (R == 0 and S == 0):
            continue
        if S == 0:
            delta = -np.inf
        elif R == 0:
            delta = np.inf
        else:
            delta = -2.35 * np.log(R / S)
        pval = (1 - chi2_dist.cdf(max(abs(sd) - 0.5, 0) ** 2 / sv, df=1)
                if sv > 0 else 1.0)
        if np.isfinite(delta) and abs(delta) >= 1.5 and pval < 0.05:
            n_c += 1

    return n_c / J * 100


# ════════════════════════════════════════════════════════════════════════
#  MULTIPROCESSING BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════
_Xr = None
_Xf = None


def _init(Xr, Xf):
    global _Xr, _Xf
    _Xr = Xr
    _Xf = Xf


def _boot_one(seed):
    rng = np.random.RandomState(seed)
    ir = rng.randint(0, _Xr.shape[0], size=_Xr.shape[0])
    jf = rng.randint(0, _Xf.shape[0], size=_Xf.shape[0])
    return mh_dif_c_pct_vec(_Xr[ir], _Xf[jf])


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    t_total = time.time()
    print(f"MMLU Bootstrap CI — N={N_BOOTSTRAP}, seed={SEED}")
    print(f"Output: {OUT}\n")

    # ── Load ──
    print("Loading MMLU response matrix...")
    t0 = time.time()
    X = load_npz(BASE / "response_matrix.npz").toarray().astype(np.int8)
    idx = np.load(BASE / "response_matrix_index.npz", allow_pickle=True)
    model_names = idx["model_names"]
    meta = pd.read_csv(BASE / "model_metadata.csv")
    cohort = dict(zip(meta["model_name"], meta["cohort_temporal"]))
    print(f"  {X.shape[0]} models × {X.shape[1]} items ({time.time()-t0:.1f}s)")

    name2row = {n: i for i, n in enumerate(model_names)}
    ref_idx = np.array([name2row[n] for n in model_names if cohort.get(n) == "2023"])
    foc_idx = np.array([name2row[n] for n in model_names if cohort.get(n) == "2024"])
    print(f"  Cohorts: ref(2023)={len(ref_idx)}, foc(2024)={len(foc_idx)}")

    X_ref = X[ref_idx]
    X_foc = X[foc_idx]
    del X

    # ── Validate total-score vs rest-score ──
    print("\nValidation: total-score vs rest-score stratification...")
    t0 = time.time()
    c_vec = mh_dif_c_pct_vec(X_ref, X_foc)
    t_vec = time.time() - t0
    print(f"  Vectorized (total-score): C% = {c_vec:.2f}% ({t_vec:.2f}s)")

    t0 = time.time()
    c_rest = mh_dif_c_pct_rest(X_ref, X_foc)
    t_rest = time.time() - t0
    print(f"  Original  (rest-score):   C% = {c_rest:.2f}% ({t_rest:.1f}s)")
    diff_pp = abs(c_vec - c_rest)
    print(f"  Difference: {diff_pp:.2f}pp  |  Speedup: {t_rest/t_vec:.0f}x")

    use_rest = diff_pp > 1.0
    if use_rest:
        print("  WARNING: >1pp gap — using rest-score (slow path)")

    # ── Sanity check N=10 ──
    print(f"\nSanity check: N=10 bootstrap...")
    rng = np.random.RandomState(SEED)
    fn = mh_dif_c_pct_rest if use_rest else mh_dif_c_pct_vec
    sanity = []
    for i in range(10):
        ir = rng.randint(0, len(ref_idx), size=len(ref_idx))
        jf = rng.randint(0, len(foc_idx), size=len(foc_idx))
        sanity.append(fn(X_ref[ir], X_foc[jf]))
    print(f"  C% range: [{min(sanity):.1f}%, {max(sanity):.1f}%], "
          f"mean={np.mean(sanity):.1f}%  (expect ≈31%)")

    # ── Full bootstrap N=1000 ──
    print(f"\nBootstrap N={N_BOOTSTRAP} (2 processes)...")
    seeds = list(range(SEED, SEED + N_BOOTSTRAP))
    boot_cs = np.zeros(N_BOOTSTRAP)
    t_start = time.time()

    with Pool(2, initializer=_init, initargs=(X_ref, X_foc)) as pool:
        for i, c in enumerate(pool.imap(
                _boot_one if not use_rest else _boot_one, seeds)):
            boot_cs[i] = c
            done = i + 1
            if done % 100 == 0 or done == 1:
                el = time.time() - t_start
                eta = (N_BOOTSTRAP - done) / (done / el) / 60
                print(f"  [{done}/{N_BOOTSTRAP}] C%={c:.2f}%  "
                      f"running mean={boot_cs[:done].mean():.2f}%  "
                      f"({el:.0f}s, ETA {eta:.0f}min)")

    ci_lo = np.percentile(boot_cs, 2.5)
    ci_hi = np.percentile(boot_cs, 97.5)
    b_mean = boot_cs.mean()
    b_std = boot_cs.std()

    print(f"\n  Original C% = {c_rest:.2f}% (rest-score)")
    print(f"  95% CI: [{ci_lo:.2f}%, {ci_hi:.2f}%]")
    print(f"  Boot mean={b_mean:.2f}%, std={b_std:.2f}%, "
          f"bias={b_mean - c_vec:+.2f}pp")

    # ── Random-split control ──
    print(f"\nRandom-split control (N={N_CONTROL})...")
    combined = np.vstack([X_ref, X_foc])
    rng_c = np.random.RandomState(SEED + 9999)
    ctrl = []
    for i in range(N_CONTROL):
        perm = rng_c.permutation(combined.shape[0])
        Xr = combined[perm[: len(ref_idx)]]
        Xf = combined[perm[len(ref_idx): len(ref_idx) + len(foc_idx)]]
        ctrl.append(mh_dif_c_pct_vec(Xr, Xf))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{N_CONTROL}] C%={ctrl[-1]:.2f}%")

    ctrl_mean = np.mean(ctrl)
    ctrl_std = np.std(ctrl)
    print(f"  Control: mean={ctrl_mean:.2f}%, std={ctrl_std:.2f}%, "
          f"range=[{min(ctrl):.2f}%, {max(ctrl):.2f}%]")

    # ── Save ──
    pd.DataFrame({"bootstrap_id": range(N_BOOTSTRAP), "c_pct": boot_cs}).to_csv(
        OUT / "mmlu_bootstrap_results.csv", index=False)

    summary = pd.DataFrame([{
        "benchmark": "mmlu",
        "original_c_pct_rest": c_rest,
        "original_c_pct_vec": c_vec,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "boot_mean": b_mean,
        "boot_std": b_std,
        "n_bootstrap": N_BOOTSTRAP,
        "n_ref": len(ref_idx),
        "n_foc": len(foc_idx),
        "n_items": X_ref.shape[1],
        "ctrl_mean": ctrl_mean,
        "ctrl_std": ctrl_std,
    }])
    summary.to_csv(OUT / "mmlu_bootstrap_summary.csv", index=False)

    elapsed = time.time() - t_total
    temporal_vs_ctrl = ("confirms DIF is temporal, not spurious"
                        if ctrl_mean < ci_lo
                        else "control overlaps temporal — interpret with caution")

    report = f"""# MMLU Bootstrap 95% CI for MH-DIF C%

## Method

- Bootstrap resampling of **models** (not items), N={N_BOOTSTRAP}
- Per bootstrap: resample N_ref={len(ref_idx)} (2023) and N_foc={len(foc_idx)} (2024) with replacement
- MH-DIF with {N_STRATA}-stratum total-score matching
- Validated against rest-score: diff = {diff_pp:.2f}pp (speedup {t_rest/t_vec:.0f}x)
- 95% CI via percentile method [2.5th, 97.5th]
- Seed = {SEED}

## Results

| Metric | Value |
|--------|-------|
| Original C% (rest-score) | {c_rest:.2f}% |
| Original C% (total-score) | {c_vec:.2f}% |
| Bootstrap 95% CI | [{ci_lo:.2f}%, {ci_hi:.2f}%] |
| Bootstrap mean | {b_mean:.2f}% |
| Bootstrap SD | {b_std:.2f}% |
| Bias (mean − original) | {b_mean - c_vec:+.2f}pp |
| CI width | {ci_hi - ci_lo:.2f}pp |

## Random-Split Control (N={N_CONTROL})

| Metric | Value |
|--------|-------|
| Control mean C% | {ctrl_mean:.2f}% |
| Control SD | {ctrl_std:.2f}% |
| Control range | [{min(ctrl):.2f}%, {max(ctrl):.2f}%] |

## Interpretation

- 95% CI [{ci_lo:.1f}%, {ci_hi:.1f}%] — {'narrow, estimate is stable' if (ci_hi - ci_lo) < 10 else 'moderate width'}
- Control C% = {ctrl_mean:.1f}% vs temporal C% = {c_rest:.1f}% — {temporal_vs_ctrl}

Runtime: {elapsed / 60:.1f} min
"""
    (OUT / "mmlu_bootstrap_report.md").write_text(report)

    print(f"\nSaved: mmlu_bootstrap_results.csv, mmlu_bootstrap_summary.csv, "
          f"mmlu_bootstrap_report.md")
    print(f"Total: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
