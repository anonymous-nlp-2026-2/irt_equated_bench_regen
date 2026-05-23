"""Phase 2: Item-Property Predictive Model — leak-free feature set."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings("ignore")

OUT = Path("artifacts/predictive_model")

# ── Load data ──────────────────────────────────────────────────────────
feat = pd.read_csv(OUT / "features.csv")
target = pd.read_csv(OUT / "target.csv")
df = feat.merge(target, on="item_id", how="inner")
print(f"Merged: {len(df)} items, positive rate: {df['dif_c'].mean():.3f}")

# ── Banned features (leaked) ──────────────────────────────────────────
BANNED = {
    "alpha_mh", "delta_mh", "se", "p_value",
    "difficulty", "difficulty_foc", "difficulty_other",
    "difficulty_change", "difficulty_change_ratio",
    "item_variance",
    "discrimination_ptbis", "discrimination_foc", "discrimination_change",
    "subject_c_pct", "subject_b_pct", "subject_mean_diff_change",
    "subject_mean_difficulty",
    "difficulty_vs_subject",
}

# ── Recalculate 3 features from ref cohort only ──────────────────────
df["item_variance_ref"] = df["difficulty_ref"] * (1 - df["difficulty_ref"])
df["subject_mean_difficulty_ref"] = df.groupby("subject")["difficulty_ref"].transform("mean")
df["difficulty_vs_subject_ref"] = df["difficulty_ref"] - df["subject_mean_difficulty_ref"]

# ── Handle li_label / li_contamination_type missingness ──────────────
df["li_label_missing"] = df["li_label"].isna().astype(int)
# li_label: True→1, False→0, NaN→0
# The CSV stores booleans as strings sometimes
li_map = {True: 1, "True": 1, False: 0, "False": 0}
df["li_label_clean"] = df["li_label"].map(li_map).fillna(0).astype(int)

# li_contamination_type: fill NaN with "unknown", then one-hot
df["li_contamination_type"] = df["li_contamination_type"].fillna("unknown")
contam_dummies = pd.get_dummies(df["li_contamination_type"], prefix="contam")
print(f"Contamination type dummies: {list(contam_dummies.columns)}")
df = pd.concat([df, contam_dummies], axis=1)

# ── Build safe feature list ──────────────────────────────────────────
SAFE_NUMERIC = [
    "difficulty_ref", "discrimination_ref", "item_variance_ref",
    "subject_n_items", "subject_mean_difficulty_ref", "difficulty_vs_subject_ref",
    "question_length", "question_word_count", "option_count",
    "option_length_mean", "option_length_std", "has_negation", "prompt_total_length",
    "qtype_factual", "qtype_other", "qtype_reasoning", "qtype_statement_eval",
    "li_label_clean", "li_label_missing",
]
SAFE_CONTAM = sorted(contam_dummies.columns.tolist())
FEATURE_COLS = SAFE_NUMERIC + SAFE_CONTAM

# ── Assert no leakage ───────────────────────────────────────────────
for col in FEATURE_COLS:
    assert col not in BANNED, f"LEAKED FEATURE: {col}"
print(f"\n✓ {len(FEATURE_COLS)} safe features: {FEATURE_COLS}")

# ── Prepare X, y, groups ────────────────────────────────────────────
X = df[FEATURE_COLS].copy()
for c in X.columns:
    if X[c].dtype == "bool" or X[c].dtype == object:
        X[c] = X[c].astype(int)
X = X.astype(float)
y = df["dif_c"].values
groups = df["subject"].values

# ── Save safe_features.csv ──────────────────────────────────────────
safe_df = df[["item_id"] + FEATURE_COLS].copy()
for c in safe_df.columns:
    if safe_df[c].dtype == "bool":
        safe_df[c] = safe_df[c].astype(int)
safe_df.to_csv(OUT / "safe_features.csv", index=False)
print(f"Saved safe_features.csv: {safe_df.shape}")

# ── GroupKFold CV ────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
cv_rows = []
lr_coefs_all = []
gbm_shap_all = []

for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    print(f"\nFold {fold_i}: train={len(train_idx)}, test={len(test_idx)}, "
          f"test_pos_rate={y_te.mean():.3f}, "
          f"test_subjects={np.unique(groups[test_idx]).tolist()[:5]}...")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # ── Model 1: Full LR ────────────────────────────────────────────
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(X_tr_sc, y_tr)
    y_prob_lr = lr.predict_proba(X_te_sc)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)

    cv_rows.append({
        "fold": fold_i, "model": "lr_full",
        "auroc": roc_auc_score(y_te, y_prob_lr),
        "f1": f1_score(y_te, y_pred_lr),
        "precision": precision_score(y_te, y_pred_lr),
        "recall": recall_score(y_te, y_pred_lr),
    })
    lr_coefs_all.append(lr.coef_[0])

    # ── Model 2: Baseline (difficulty_ref only) ─────────────────────
    X_tr_base = X_tr[["difficulty_ref"]].values
    X_te_base = X_te[["difficulty_ref"]].values
    sc_base = StandardScaler()
    X_tr_base_sc = sc_base.fit_transform(X_tr_base)
    X_te_base_sc = sc_base.transform(X_te_base)

    lr_base = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr_base.fit(X_tr_base_sc, y_tr)
    y_prob_base = lr_base.predict_proba(X_te_base_sc)[:, 1]
    y_pred_base = (y_prob_base >= 0.5).astype(int)

    cv_rows.append({
        "fold": fold_i, "model": "lr_baseline",
        "auroc": roc_auc_score(y_te, y_prob_base),
        "f1": f1_score(y_te, y_pred_base),
        "precision": precision_score(y_te, y_pred_base),
        "recall": recall_score(y_te, y_pred_base),
    })

    # ── Model 3: LightGBM ──────────────────────────────────────────
    gbm = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    gbm.fit(X_tr, y_tr)
    y_prob_gbm = gbm.predict_proba(X_te)[:, 1]
    y_pred_gbm = (y_prob_gbm >= 0.5).astype(int)

    cv_rows.append({
        "fold": fold_i, "model": "gbm",
        "auroc": roc_auc_score(y_te, y_prob_gbm),
        "f1": f1_score(y_te, y_pred_gbm),
        "precision": precision_score(y_te, y_pred_gbm),
        "recall": recall_score(y_te, y_pred_gbm),
    })

    # SHAP for GBM
    explainer = shap.TreeExplainer(gbm)
    shap_vals = explainer.shap_values(X_te)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    gbm_shap_all.append(np.abs(shap_vals).mean(axis=0))

# ── Aggregate results ────────────────────────────────────────────────
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUT / "cv_results.csv", index=False)
print("\n" + "="*70)
print("CV RESULTS")
print("="*70)

for model in ["lr_full", "lr_baseline", "gbm"]:
    sub = cv_df[cv_df["model"] == model]
    print(f"\n{model}:")
    for m in ["auroc", "f1", "precision", "recall"]:
        print(f"  {m}: {sub[m].mean():.4f} ± {sub[m].std():.4f}")

# ── Feature importance ───────────────────────────────────────────────
mean_coefs = np.mean(lr_coefs_all, axis=0)
mean_shap = np.mean(gbm_shap_all, axis=0)

fi_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "lr_coef": mean_coefs,
    "lr_odds_ratio": np.exp(mean_coefs),
    "gbm_shap_mean": mean_shap,
})
fi_df = fi_df.sort_values("gbm_shap_mean", ascending=False)
fi_df.to_csv(OUT / "feature_importance.csv", index=False)

print("\n" + "="*70)
print("FEATURE IMPORTANCE (top 10 by SHAP)")
print("="*70)
print(fi_df.head(10).to_string(index=False))

# ── Generate report ──────────────────────────────────────────────────
lr_full_auroc = cv_df[cv_df["model"] == "lr_full"]["auroc"].mean()
lr_full_auroc_std = cv_df[cv_df["model"] == "lr_full"]["auroc"].std()
lr_base_auroc = cv_df[cv_df["model"] == "lr_baseline"]["auroc"].mean()
lr_base_auroc_std = cv_df[cv_df["model"] == "lr_baseline"]["auroc"].std()
gbm_auroc = cv_df[cv_df["model"] == "gbm"]["auroc"].mean()
gbm_auroc_std = cv_df[cv_df["model"] == "gbm"]["auroc"].std()

if gbm_auroc < 0.65:
    verdict = "**Abandon recommendation** — AUROC < 0.65, predictive model not viable for paper inclusion."
elif gbm_auroc <= 0.75:
    verdict = "**Appendix placement** — AUROC 0.65–0.75, modest predictive power; place in appendix as supplementary analysis."
else:
    verdict = "**Main text** — AUROC > 0.75, strong predictive power; include in main results."

def fmt_cv(model_name):
    sub = cv_df[cv_df["model"] == model_name]
    lines = []
    for _, row in sub.iterrows():
        lines.append(f"| {int(row['fold'])} | {row['auroc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |")
    lines.append(f"| **Mean** | **{sub['auroc'].mean():.4f} ± {sub['auroc'].std():.4f}** | **{sub['f1'].mean():.4f} ± {sub['f1'].std():.4f}** | **{sub['precision'].mean():.4f} ± {sub['precision'].std():.4f}** | **{sub['recall'].mean():.4f} ± {sub['recall'].std():.4f}** |")
    return "\n".join(lines)

shap_top10 = fi_df.head(10)
shap_lines = "\n".join(
    f"| {r['feature']} | {r['gbm_shap_mean']:.4f} | {r['lr_coef']:.4f} | {r['lr_odds_ratio']:.4f} |"
    for _, r in shap_top10.iterrows()
)

contam_cols_str = ", ".join(SAFE_CONTAM)

report = f"""# Phase 2: Item-Property Predictive Model — Leak-Free Results

## Safe Feature Set ({len(FEATURE_COLS)} features)

### Numeric features (19)
| Feature | Source | Notes |
|---------|--------|-------|
| difficulty_ref | ref cohort | proportion correct, ref 2023 |
| discrimination_ref | ref cohort | point-biserial, ref 2023 |
| item_variance_ref | **recalculated** | p_ref * (1 - p_ref) |
| subject_n_items | metadata | items per subject |
| subject_mean_difficulty_ref | **recalculated** | mean difficulty_ref per subject |
| difficulty_vs_subject_ref | **recalculated** | difficulty_ref - subject mean |
| question_length | text | char count |
| question_word_count | text | word count |
| option_count | text | number of options |
| option_length_mean | text | mean option char length |
| option_length_std | text | std of option lengths |
| has_negation | text | negation keyword present |
| prompt_total_length | text | total prompt char count |
| qtype_factual | text | question type indicator |
| qtype_other | text | question type indicator |
| qtype_reasoning | text | question type indicator |
| qtype_statement_eval | text | question type indicator |
| li_label_clean | Li et al. | contamination flag (binary, missing->0) |
| li_label_missing | Li et al. | missingness indicator |

### Contamination type dummies ({len(SAFE_CONTAM)})
{contam_cols_str}

Three features recalculated from ref cohort only (replacing leaked full-sample versions):
- `item_variance_ref` = difficulty_ref * (1 - difficulty_ref)
- `subject_mean_difficulty_ref` = mean(difficulty_ref) within each subject
- `difficulty_vs_subject_ref` = difficulty_ref - subject_mean_difficulty_ref

**GroupKFold**: 5 folds, grouped by `subject` (57 subjects). Same subject never split across train/test.

---

## CV Results

### Logistic Regression — Full Model

| Fold | AUROC | F1 | Precision | Recall |
|------|-------|-----|-----------|--------|
{fmt_cv("lr_full")}

### Logistic Regression — Baseline (difficulty_ref only)

| Fold | AUROC | F1 | Precision | Recall |
|------|-------|-----|-----------|--------|
{fmt_cv("lr_baseline")}

### LightGBM

| Fold | AUROC | F1 | Precision | Recall |
|------|-------|-----|-----------|--------|
{fmt_cv("gbm")}

---

## Baseline vs Full Model

| Model | Mean AUROC |
|-------|-----------|
| LR Baseline (difficulty_ref only) | {lr_base_auroc:.4f} +/- {lr_base_auroc_std:.4f} |
| LR Full ({len(FEATURE_COLS)} features) | {lr_full_auroc:.4f} +/- {lr_full_auroc_std:.4f} |
| LightGBM ({len(FEATURE_COLS)} features) | {gbm_auroc:.4f} +/- {gbm_auroc_std:.4f} |

AUROC improvement over baseline: LR +{lr_full_auroc - lr_base_auroc:.4f}, GBM +{gbm_auroc - lr_base_auroc:.4f}

---

## SHAP Feature Importance (GBM, top 10)

| Feature | SHAP mean |abs| | LR coef | LR odds ratio |
|---------|-----------|---------|---------------|
{shap_lines}

---

## Verdict

Best model AUROC (GBM): **{gbm_auroc:.4f}**

{verdict}
"""

with open(OUT / "phase2_results.md", "w") as f:
    f.write(report)

print(f"\n✓ All outputs saved to {OUT}")
print(f"  - phase2_results.md")
print(f"  - cv_results.csv ({len(cv_df)} rows)")
print(f"  - feature_importance.csv ({len(fi_df)} rows)")
print(f"  - safe_features.csv ({safe_df.shape})")
