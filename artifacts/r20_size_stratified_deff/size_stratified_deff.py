"""
R20: Size-Stratified DEFF + Family Definition Sensitivity
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(".")
ARTIFACTS = ROOT / "artifacts"
OUT = ARTIFACTS / "r20_size_stratified_deff"
OUT.mkdir(parents=True, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────

def load_data():
    f = np.load(ARTIFACTS / "response_matrix.npz")
    X = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape']))
    idx = np.load(ARTIFACTS / "response_matrix_index.npz", allow_pickle=True)
    model_names = idx['model_names']
    item_ids = idx['item_ids']
    meta = pd.read_csv(ARTIFACTS / "plan_032_dif_cleaned_mmlu" / "cleaned_mmlu_analysis.csv")
    dif = pd.read_csv(ARTIFACTS / "plan_001" / "dif_results_temporal.csv")
    return X, model_names, item_ids, meta, dif


def filter_models(meta, model_names, family_col='family'):
    """Filter to cohort 2023/2024, known family, family size >= 2."""
    cohort_mask = meta['cohort_temporal'].isin(['2023', '2024'])
    meta_c = meta.loc[cohort_mask].copy()
    fc = meta_c[family_col].value_counts()
    valid_fams = [f for f in fc[fc >= 2].index if f != 'unknown' and pd.notna(f) and f != '']
    mask = meta_c[family_col].isin(valid_fams)
    valid_meta = meta_c.loc[mask]

    m2i = {name: i for i, name in enumerate(model_names)}
    indices, families, names = [], [], []
    for _, row in valid_meta.iterrows():
        m = row['model_name']
        if m in m2i:
            indices.append(m2i[m])
            families.append(row[family_col])
            names.append(m)
    return np.array(indices), np.array(families), names, valid_fams


# ── ICC computation ───────────────────────────────────────────

def compute_icc(X_valid, families_valid):
    """Per-item ANOVA-based ICC. Returns icc, m_bar_weighted, m_bar_simple, K, family_sizes."""
    N, n_items = X_valid.shape
    unique_fams = np.unique(families_valid)
    K = len(unique_fams)
    fam2idx = {f: i for i, f in enumerate(unique_fams)}
    fi = np.array([fam2idx[f] for f in families_valid])

    F = sparse.csc_matrix((np.ones(N), (np.arange(N), fi)), shape=(N, K))
    family_sizes = np.array(F.sum(axis=0)).ravel()

    X_d = X_valid.toarray().astype(np.float64)
    grand_mean = X_d.mean(axis=0)

    fam_sums = F.T @ X_d
    fam_means = fam_sums / family_sizes[:, None]

    dev = fam_means - grand_mean[None, :]
    SSB = (family_sizes[:, None] * dev ** 2).sum(axis=0)
    SSW = (X_d ** 2).sum(axis=0) - (family_sizes[:, None] * fam_means ** 2).sum(axis=0)

    MSB = SSB / (K - 1)
    MSW = SSW / (N - K)

    m_bar_w = (family_sizes ** 2).sum() / family_sizes.sum()
    m_bar_s = family_sizes.mean()

    num = MSB - MSW
    den = MSB + (m_bar_w - 1) * MSW
    icc = np.where(den != 0, num / den, 0.0)
    icc = np.clip(icc, 0, 1)

    return icc, m_bar_w, m_bar_s, K, family_sizes, unique_fams


def compute_icc_for_tier(X_valid, families_valid, tier_fams):
    """Compute ICC using only families in tier_fams."""
    mask = np.isin(families_valid, tier_fams)
    if mask.sum() < 2:
        return None, 0, 0, 0, np.array([])
    X_sub = X_valid[mask]
    fam_sub = families_valid[mask]
    unique = np.unique(fam_sub)
    if len(unique) < 2:
        return None, 0, 0, 0, np.array([])
    return compute_icc(X_sub, fam_sub)


# ── DEFF adjustment ──────────────────────────────────────────

def compute_c_pct(dif, icc, item_ids, m_bar):
    """Given per-item ICC and a single m_bar, compute adjusted C%."""
    item2icc = dict(zip(item_ids, icc))
    icc_vals = dif['item_id'].map(item2icc)
    deff = 1 + (m_bar - 1) * icc_vals
    chi2_orig = (dif['delta_mh'] / dif['se']) ** 2
    chi2_adj = chi2_orig / deff
    p_adj = 1 - chi2.cdf(chi2_adj, df=1)
    abs_d = dif['delta_mh'].abs()
    is_c = (abs_d >= 1.5) & (p_adj < 0.05)
    return is_c.mean() * 100


def full_ets_reclassify(dif, icc, item_ids, m_bar):
    """Full ETS reclassification at given m_bar."""
    item2icc = dict(zip(item_ids, icc))
    icc_vals = dif['item_id'].map(item2icc)
    deff = 1 + (m_bar - 1) * icc_vals
    chi2_orig = (dif['delta_mh'] / dif['se']) ** 2
    chi2_adj = chi2_orig / deff
    p_adj = 1 - chi2.cdf(chi2_adj, df=1)
    abs_d = dif['delta_mh'].abs()
    ets = pd.Series('A', index=dif.index)
    ets[(abs_d >= 1.0) & (p_adj < 0.05)] = 'B'
    ets[(abs_d >= 1.5) & (p_adj < 0.05)] = 'C'
    counts = ets.value_counts()
    return {
        'C': counts.get('C', 0),
        'B': counts.get('B', 0),
        'A': counts.get('A', 0),
        'C_pct': counts.get('C', 0) / len(dif) * 100,
        'B_pct': counts.get('B', 0) / len(dif) * 100,
        'deff_mean': deff.mean(),
    }


# ── Coarse family mapping ────────────────────────────────────

COARSE_MAP = {
    'llama-1': 'llama', 'llama-2': 'llama', 'llama-3': 'llama', 'llama-3.2': 'llama',
    'codellama': 'llama',
    'mistral': 'mistral', 'mistral-0.1': 'mistral', 'mixtral': 'mistral',
    'mixtral-8x7B': 'mistral',
    'gemma': 'gemma', 'gemma-1': 'gemma', 'gemma-2': 'gemma',
    'qwen': 'qwen', 'qwen-1': 'qwen', 'qwen-1.5': 'qwen', 'qwen-2': 'qwen',
    'phi-2': 'phi', 'phi-3': 'phi',
    'yi': 'yi', 'yi-1': 'yi', 'yi-1.5': 'yi',
    'wizardlm': 'wizardlm', 'wizardlm-2': 'wizardlm',
    'openchat': 'openchat', 'openchat-3.5': 'openchat',
    'starcoder': 'starcoder', 'starcoder-2': 'starcoder',
    'tulu': 'tulu', 'tulu-2': 'tulu',
    'falcon': 'falcon', 'falcon-2': 'falcon',
    'orca': 'orca', 'orca-2': 'orca',
}


def add_coarse_family(meta):
    meta = meta.copy()
    meta['family_coarse'] = meta['family'].map(lambda f: COARSE_MAP.get(f, f))
    return meta


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("R20: Size-Stratified DEFF + Family Definition Sensitivity")
    print("=" * 60)

    # ── Load ──
    print("\n[1/6] Loading data...")
    X, model_names, item_ids, meta, dif = load_data()
    print(f"  Response matrix: {X.shape}")

    # ── Part A1: Size-Stratified DEFF ──
    print("\n[2/6] Computing overall ICC (fine-grained families)...")
    vi, fv, _, vf = filter_models(meta, model_names)
    X_v = X[vi]
    icc, m_w, m_s, K, fsizes, ufams = compute_icc(X_v, fv)
    print(f"  {len(vi)} models, {K} families")
    print(f"  m̄_weighted={m_w:.1f}, m̄_simple={m_s:.1f}")
    print(f"  Mean ICC={icc.mean():.4f}")

    orig_c_pct = (dif['ets_class'] == 'C').mean() * 100
    print(f"  Original C% = {orig_c_pct:.1f}%")

    # Build family-size lookup
    fam_size_map = dict(zip(ufams, fsizes.astype(int)))

    # Define tiers based on actual distribution
    tier_defs = [
        ('Tier 1: 2–10',   2,  10),
        ('Tier 2: 11–50',  11, 50),
        ('Tier 3: 51–200', 51, 200),
        ('Tier 4: >200',   201, 99999),
    ]

    print("\n[3/6] Computing tier-stratified ICC...")
    tier_rows = []
    for tname, lo, hi in tier_defs:
        tier_fams = [f for f in ufams if lo <= fam_size_map[f] <= hi]
        n_fams = len(tier_fams)
        if n_fams < 2:
            print(f"  {tname}: {n_fams} families — skipping (need ≥2)")
            tier_fams_sizes = np.array([fam_size_map[f] for f in tier_fams])
            tier_rows.append({
                'tier': tname, 'n_families': n_fams,
                'mean_m': tier_fams_sizes.mean() if len(tier_fams_sizes) > 0 else 0,
                'mean_icc': np.nan, 'deff': np.nan, 'adj_c_pct': np.nan,
            })
            continue

        result = compute_icc_for_tier(X_v, fv, tier_fams)
        icc_tier, mw_t, ms_t, K_t, fs_t = result[:5]
        if icc_tier is None:
            continue

        tier_fam_sizes = np.array([fam_size_map[f] for f in tier_fams])
        mean_m = tier_fam_sizes.mean()
        mean_icc = icc_tier.mean()
        deff_tier = 1 + (mean_m - 1) * mean_icc
        c_pct_tier = compute_c_pct(dif, icc_tier, item_ids, mean_m)

        print(f"  {tname}: {n_fams} fams, mean_m={mean_m:.1f}, "
              f"ICC={mean_icc:.4f}, DEFF={deff_tier:.2f}, C%={c_pct_tier:.1f}%")
        tier_rows.append({
            'tier': tname, 'n_families': n_fams, 'mean_m': mean_m,
            'mean_icc': mean_icc, 'deff': deff_tier, 'adj_c_pct': c_pct_tier,
        })

    # Add overall rows
    deff_simple = 1 + (m_s - 1) * icc.mean()
    deff_weighted = 1 + (m_w - 1) * icc.mean()
    c_simple = compute_c_pct(dif, icc, item_ids, m_s)
    c_weighted = compute_c_pct(dif, icc, item_ids, m_w)

    tier_rows.append({
        'tier': 'Overall (simple m̄)', 'n_families': K, 'mean_m': m_s,
        'mean_icc': icc.mean(), 'deff': deff_simple, 'adj_c_pct': c_simple,
    })
    tier_rows.append({
        'tier': 'Overall (weighted m̄)', 'n_families': K, 'mean_m': m_w,
        'mean_icc': icc.mean(), 'deff': deff_weighted, 'adj_c_pct': c_weighted,
    })

    df_tiers = pd.DataFrame(tier_rows)
    df_tiers.to_csv(OUT / "stratified_deff_results.csv", index=False)
    print(f"\n  Saved stratified_deff_results.csv")

    # ── Parameterization curve ──
    print("\n[4/6] Computing parameterization curve...")
    m_range = np.linspace(2, 350, 500)
    c_curve = np.array([compute_c_pct(dif, icc, item_ids, m) for m in m_range])

    # Also compute per-tier ICC curves
    tier_icc_map = {}
    for tname, lo, hi in tier_defs:
        tier_fams = [f for f in ufams if lo <= fam_size_map[f] <= hi]
        if len(tier_fams) < 2:
            continue
        result = compute_icc_for_tier(X_v, fv, tier_fams)
        if result[0] is not None:
            tier_icc_map[tname] = result[0]

    # ── Figure ──
    print("\n[5/6] Generating figure...")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(m_range, c_curve, 'k-', lw=2, label='Overall ICC', zorder=5)

    # Per-tier ICC curves (lighter)
    tier_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for i, (tname, lo, hi) in enumerate(tier_defs):
        if tname in tier_icc_map:
            c_tier_curve = np.array([
                compute_c_pct(dif, tier_icc_map[tname], item_ids, m) for m in m_range
            ])
            ax.plot(m_range, c_tier_curve, '--', color=tier_colors[i], lw=1.2,
                    alpha=0.7, label=f'{tname} ICC')

    # Mark key points on overall curve
    key_points = [
        (m_s, c_simple, f'simple m̄={m_s:.0f}\nC%={c_simple:.1f}%', 'left'),
        (m_w, c_weighted, f'weighted m̄={m_w:.0f}\nC%={c_weighted:.1f}%', 'right'),
    ]
    for mx, cy, label, ha in key_points:
        ax.plot(mx, cy, 'ko', ms=7, zorder=10)
        offset = (-12, 10) if ha == 'right' else (12, 10)
        ax.annotate(label, (mx, cy), textcoords='offset points', xytext=offset,
                    ha=ha, fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3',
                    fc='white', ec='gray', alpha=0.9))

    # Mark tier means on overall curve
    for i, row in df_tiers.iterrows():
        if row['tier'].startswith('Tier') and not np.isnan(row['adj_c_pct']):
            c_at = compute_c_pct(dif, icc, item_ids, row['mean_m'])
            ax.plot(row['mean_m'], c_at, 's', color=tier_colors[i], ms=6, zorder=8)

    # Original C% reference line
    ax.axhline(orig_c_pct, color='red', ls=':', lw=1, alpha=0.6)
    ax.text(5, orig_c_pct + 0.5, f'Original C%={orig_c_pct:.1f}%', fontsize=8, color='red')

    ax.set_xlabel('Mean family size (m̄)', fontsize=11)
    ax.set_ylabel('Adjusted C-class items (%)', fontsize=11)
    ax.set_title('DEFF Parameterization: C% vs Mean Family Size', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "deff_parameterization_curve.png", dpi=200)
    fig.savefig(OUT / "deff_parameterization_curve.pdf")
    plt.close(fig)
    print("  Saved deff_parameterization_curve.{png,pdf}")

    # ── Part A2: Family Definition Sensitivity ──
    print("\n[6/6] Family definition sensitivity...")
    meta_c = add_coarse_family(meta)

    vi_c, fv_c, _, vf_c = filter_models(meta_c, model_names, family_col='family_coarse')
    X_vc = X[vi_c]
    icc_c, mw_c, ms_c, K_c, fs_c, ufams_c = compute_icc(X_vc, fv_c)

    deff_s_fine = 1 + (m_s - 1) * icc.mean()
    deff_w_fine = 1 + (m_w - 1) * icc.mean()
    c_s_fine = compute_c_pct(dif, icc, item_ids, m_s)
    c_w_fine = compute_c_pct(dif, icc, item_ids, m_w)

    deff_s_coarse = 1 + (ms_c - 1) * icc_c.mean()
    deff_w_coarse = 1 + (mw_c - 1) * icc_c.mean()
    c_s_coarse = compute_c_pct(dif, icc_c, item_ids, ms_c)
    c_w_coarse = compute_c_pct(dif, icc_c, item_ids, mw_c)

    sens_rows = [
        {
            'definition': 'Fine-grained',
            'n_families': K, 'n_models': len(vi),
            'm_bar_simple': m_s, 'm_bar_weighted': m_w,
            'mean_icc': icc.mean(),
            'deff_simple': deff_s_fine, 'deff_weighted': deff_w_fine,
            'c_pct_simple': c_s_fine, 'c_pct_weighted': c_w_fine,
        },
        {
            'definition': 'Coarse',
            'n_families': K_c, 'n_models': len(vi_c),
            'm_bar_simple': ms_c, 'm_bar_weighted': mw_c,
            'mean_icc': icc_c.mean(),
            'deff_simple': deff_s_coarse, 'deff_weighted': deff_w_coarse,
            'c_pct_simple': c_s_coarse, 'c_pct_weighted': c_w_coarse,
        },
    ]
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(OUT / "family_sensitivity_results.csv", index=False)

    print(f"\n  Fine-grained: {K} families, ICC={icc.mean():.4f}, "
          f"DEFF(s)={deff_s_fine:.2f}, C%(s)={c_s_fine:.1f}%")
    print(f"  Coarse:       {K_c} families, ICC={icc_c.mean():.4f}, "
          f"DEFF(s)={deff_s_coarse:.2f}, C%(s)={c_s_coarse:.1f}%")

    # Coarse family size distribution
    coarse_fam_sizes = dict(zip(ufams_c, fs_c.astype(int)))
    print(f"\n  Coarse family sizes (top-10):")
    for f, s in sorted(coarse_fam_sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"    {f}: {s}")

    # ── Report ──
    print("\n  Generating report...")

    # Fine-grained family sizes for report
    fine_fam_table = sorted(fam_size_map.items(), key=lambda x: -x[1])
    fine_fam_str = '\n'.join(f"| {f} | {s} |" for f, s in fine_fam_table)

    coarse_fam_table = sorted(coarse_fam_sizes.items(), key=lambda x: -x[1])
    coarse_fam_str = '\n'.join(f"| {f} | {s} |" for f, s in coarse_fam_table)

    tier_table_str = ""
    for _, row in df_tiers.iterrows():
        m_icc = f"{row['mean_icc']:.4f}" if not np.isnan(row['mean_icc']) else "—"
        m_deff = f"{row['deff']:.2f}" if not np.isnan(row['deff']) else "—"
        m_c = f"{row['adj_c_pct']:.1f}%" if not np.isnan(row['adj_c_pct']) else "—"
        tier_table_str += f"| {row['tier']} | {row['n_families']} | {row['mean_m']:.1f} | {m_icc} | {m_deff} | {m_c} |\n"

    report = f"""# R20: Size-Stratified DEFF + Family Definition Sensitivity

## Part A1: Size-Stratified DEFF

### Analysis Scope
- Models: {len(vi)} (cohort 2023/2024, known family, family≥2)
- Families: {K} (fine-grained)
- Items: {X.shape[1]}
- Original C%: {orig_c_pct:.1f}%

### Tier-Stratified Results

| Tier | N families | Mean m̄ | Mean ICC | DEFF | Adj C% |
|------|-----------|--------|----------|------|--------|
{tier_table_str}

### Key Finding

The DEFF parameterization curve shows that C% is a monotonically decreasing function of m̄.
- At m̄_simple = {m_s:.0f} (treating all families equally): DEFF = {deff_simple:.2f}, C% = {c_simple:.1f}%
- At m̄_weighted = {m_w:.0f} (proportional to family size): DEFF = {deff_weighted:.2f}, C% = {c_weighted:.1f}%
- The range [{c_weighted:.1f}%, {c_simple:.1f}%] brackets the plausible DEFF-adjusted C%.

### Fine-Grained Family Sizes

| Family | Size |
|--------|------|
{fine_fam_str}

---

## Part A2: Family Definition Sensitivity

### Coarse Mapping
Merged related families: llama-1/2/3/3.2/codellama → llama, mistral/mistral-0.1/mixtral/mixtral-8x7B → mistral,
gemma/gemma-1/gemma-2 → gemma, qwen/qwen-1/qwen-1.5/qwen-2 → qwen, phi-2/phi-3 → phi,
yi/yi-1/yi-1.5 → yi, wizardlm/wizardlm-2 → wizardlm, openchat/openchat-3.5 → openchat,
starcoder/starcoder-2 → starcoder, tulu/tulu-2 → tulu, falcon/falcon-2 → falcon, orca/orca-2 → orca.

### Comparison

| Metric | Fine-grained | Coarse |
|--------|-------------|--------|
| N families | {K} | {K_c} |
| N models | {len(vi)} | {len(vi_c)} |
| m̄ (simple) | {m_s:.1f} | {ms_c:.1f} |
| m̄ (weighted) | {m_w:.1f} | {mw_c:.1f} |
| Mean ICC | {icc.mean():.4f} | {icc_c.mean():.4f} |
| DEFF (simple) | {deff_s_fine:.2f} | {deff_s_coarse:.2f} |
| DEFF (weighted) | {deff_w_fine:.2f} | {deff_w_coarse:.2f} |
| C% (simple) | {c_s_fine:.1f}% | {c_s_coarse:.1f}% |
| C% (weighted) | {c_w_fine:.1f}% | {c_w_coarse:.1f}% |

### Coarse Family Sizes

| Family | Size |
|--------|------|
{coarse_fam_str}

### Interpretation

Coarse grouping merges {K} → {K_c} families. Larger families increase m̄, which increases DEFF,
which reduces C%. The ICC may increase (more within-family homogeneity when related sub-families
are merged) or decrease (if sub-families are actually heterogeneous). The net effect on C%
depends on the balance between these forces.
"""
    (OUT / "size_stratified_deff_report.md").write_text(report)
    print("  Saved size_stratified_deff_report.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
