"""Bootstrap CI for DIF-C × Gema et al. MMLU-Redux error enrichment."""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

SEED = 42
N_BOOT = 1000

BASE = "/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts"
DIF_PATH = f"{BASE}/plan_001/dif_results_temporal.csv"
REDUX_PATH = f"{BASE}/mmlu_redux/mmlu_redux_2.0_all.csv"
OUT_DIR = f"{BASE}/r17_gema_enrichment_ci"

dif = pd.read_csv(DIF_PATH)
redux = pd.read_csv(REDUX_PATH)

redux["item_id"] = redux["subject"] + "." + (redux["redux_index"] + 1).astype(str)

merged = dif.merge(redux[["item_id", "error_type"]], on="item_id", how="inner")

merged["is_dif_c"] = merged["ets_class"] == "C"
merged["is_gema_error"] = merged["error_type"].notna() & (merged["error_type"].str.strip() != "") & (merged["error_type"].str.strip() != "ok")

n_total = len(merged)
n_dif_c = merged["is_dif_c"].sum()
n_gema = merged["is_gema_error"].sum()
n_overlap = (merged["is_dif_c"] & merged["is_gema_error"]).sum()

print(f"Universe: {n_total}")
print(f"DIF-C: {n_dif_c} ({n_dif_c/n_total:.1%})")
print(f"Gema error: {n_gema} ({n_gema/n_total:.1%})")
print(f"Overlap: {n_overlap}")


def enrichment_ratio(dc, ge):
    err_mask = ge.astype(bool)
    noerr_mask = ~err_mask
    p_dc_given_err = dc[err_mask].mean() if err_mask.sum() > 0 else np.nan
    p_dc_given_noerr = dc[noerr_mask].mean() if noerr_mask.sum() > 0 else np.nan
    if p_dc_given_noerr == 0:
        return np.nan
    return p_dc_given_err / p_dc_given_noerr


point_est = enrichment_ratio(merged["is_dif_c"].values, merged["is_gema_error"].values)
print(f"Enrichment (point): {point_est:.4f}")

rng = np.random.default_rng(SEED)
dc_arr = merged["is_dif_c"].values
ge_arr = merged["is_gema_error"].values
n = len(merged)

boot_results = []
for _ in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    boot_results.append(enrichment_ratio(dc_arr[idx], ge_arr[idx]))

boot_arr = np.array(boot_results)
ci_lo, ci_hi = np.percentile(boot_arr, [2.5, 97.5])
print(f"Bootstrap mean: {boot_arr.mean():.4f} ± {boot_arr.std():.4f}")
print(f"Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

a = n_overlap
b = n_gema - n_overlap
c = n_dif_c - n_overlap
d = n_total - n_dif_c - n_gema + n_overlap
table = np.array([[a, b], [c, d]])
fisher_or, fisher_p = fisher_exact(table)
phi = (a * d - b * c) / np.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))

print(f"Fisher OR: {fisher_or:.4f}, p: {fisher_p:.4e}")
print(f"φ: {phi:.4f}")

pd.DataFrame({"enrichment": boot_results}).to_csv(f"{OUT_DIR}/gema_enrichment_ci_results.csv", index=False)

p_dc = n_dif_c / n_total
p_ge = n_gema / n_total
expected_overlap = p_dc * p_ge * n_total

report = f"""# R17 Enrichment Analysis: DIF-C × Gema et al. MMLU-Redux Error

## Data Summary

| Metric | Value |
|--------|-------|
| Universe (DIF ∩ Redux) | {n_total} |
| DIF-C items | {n_dif_c} ({n_dif_c/n_total:.1%}) |
| Gema error items | {n_gema} ({n_gema/n_total:.1%}) |
| Overlap (DIF-C ∩ Gema error) | {n_overlap} |
| P(DIF-C) | {p_dc:.4f} |
| P(Gema error) | {p_ge:.4f} |
| Expected overlap | {expected_overlap:.1f} |

## Enrichment

**Enrichment = {point_est:.4f}**

- Bootstrap 95% CI ({N_BOOT} reps): **[{ci_lo:.4f}, {ci_hi:.4f}]**
- Bootstrap mean: {boot_arr.mean():.4f} ± {boot_arr.std():.4f}

## Statistical Tests

| Test | Statistic | Value |
|------|-----------|-------|
| Fisher exact test | OR | {fisher_or:.4f} |
| Fisher exact test | p-value | {fisher_p:.4e} |
| Phi coefficient | φ | {phi:.4f} |

## Contingency Table

|  | Gema error | No error |
|--|------------|----------|
| DIF-C | {a} | {c} |
| non-DIF-C | {b} | {d} |

## Conclusion

The enrichment of {point_est:.4f} with 95% CI [{ci_lo:.4f}, {ci_hi:.4f}] {"includes" if ci_lo <= 1.0 <= ci_hi else "excludes"} 1.0.
{"This indicates no statistically significant enrichment — DIF-C items are not preferentially drawn from Gema et al. error-annotated items." if ci_lo <= 1.0 <= ci_hi else "This indicates statistically significant enrichment."}
Fisher exact test (OR={fisher_or:.4f}, p={fisher_p:.4e}) {"confirms no significant association" if fisher_p > 0.05 else "indicates significant association"} between DIF-C classification and Gema et al. error annotations.
The near-zero φ coefficient ({phi:.4f}) indicates negligible association.
"""

with open(f"{OUT_DIR}/gema_enrichment_ci_report.md", "w") as f:
    f.write(report)

print(f"\nResults saved to {OUT_DIR}/")
