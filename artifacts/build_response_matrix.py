#!/usr/bin/env python3
"""Build response matrix and metadata tables from METABENCH MMLU data.

Memory-safe: uses dense int8 matrix (~65MB), processes CSVs one at a time.
Previous worker OOM'd by pd.concat'ing all CSVs — this script never holds
more than one CSV in memory.

Outputs (all in ARTIFACTS dir):
  - response_matrix.npz       — scipy CSR, int8, shape (N_models, N_items)
  - response_matrix_index.npz — model_names and item_ids arrays
  - item_metadata.csv          — per-item IRT params + empirical_p
  - model_metadata.csv         — per-model family, cohort, accuracy, etc.
"""

import gc
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

ARTIFACTS = "/home/ubuntu/.agent-ml-research-idea_gen_0520_2/projects/irt_equated_bench_regen/artifacts"
DATA_DIR = os.path.join(ARTIFACTS, "metabench_data", "benchmark-data")


def get_mmlu_csvs():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, "mmlu_*.csv")))
    return [f for f in csvs if "_prompts" not in f]


def subject_from_path(csv_path):
    return os.path.basename(csv_path).replace("mmlu_", "").replace(".csv", "")


# ═══════════════════════════════════════════════════════════════════
# Load authoritative item list from IRT params (12,508 items)
# ═══════════════════════════════════════════════════════════════════

irt_params = pd.read_csv(os.path.join(ARTIFACTS, "metabench_irt_params.csv"))
irt_item_set = set(irt_params["item"])
print(f"IRT params: {len(irt_params)} items across {irt_params['subject'].nunique()} subjects")

# Build ordered item list (grouped by subject, sorted within subject)
item_list = irt_params.sort_values(["subject", "item"])["item"].tolist()
item_to_idx = {it: i for i, it in enumerate(item_list)}
N_ITEMS = len(item_list)

# ═══════════════════════════════════════════════════════════════════
# PASS 1: Discover all models
# ═══════════════════════════════════════════════════════════════════

print("=" * 60)
print("PASS 1: Discovering models")
print("=" * 60)

csv_files = get_mmlu_csvs()
print(f"Found {len(csv_files)} MMLU CSV files")

all_models = set()
items_in_csv_total = 0
items_matched_total = 0

for csv_path in csv_files:
    subject = subject_from_path(csv_path)
    df = pd.read_csv(csv_path, usecols=["source", "item"])
    all_models.update(df["source"].unique())

    csv_items = set(f"{subject}.{i}" for i in df["item"].unique())
    irt_items_for_subject = set(irt_params.loc[irt_params["subject"] == subject, "item"])
    matched = csv_items & irt_item_set
    items_in_csv_total += len(csv_items)
    items_matched_total += len(matched)

    print(f"  {subject}: {df['source'].nunique()} models, "
          f"{len(csv_items)} csv items, {len(irt_items_for_subject)} irt items, "
          f"{len(matched)} matched")
    del df
    gc.collect()

print(f"\nCSV items total: {items_in_csv_total}, matched to IRT: {items_matched_total}")

model_list = sorted(all_models)
del all_models
model_to_idx = {m: i for i, m in enumerate(model_list)}
N_MODELS = len(model_list)

print(f"Total: {N_MODELS} models, {N_ITEMS} items (IRT-filtered)")
print(f"Dense matrix: {N_MODELS * N_ITEMS / 1e6:.1f}M cells = {N_MODELS * N_ITEMS // (1024*1024)}MB int8")


# ═══════════════════════════════════════════════════════════════════
# PASS 2: Fill response matrix (dense int8, ~65MB)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PASS 2: Filling response matrix")
print("=" * 60)

# -1 = not observed, 0 = incorrect, 1 = correct
mat = np.full((N_MODELS, N_ITEMS), fill_value=-1, dtype=np.int8)
print(f"Allocated {mat.nbytes / (1024**2):.0f}MB dense matrix")

total_rows = 0
skipped_rows = 0
for csv_path in csv_files:
    subject = subject_from_path(csv_path)
    t0 = time.time()

    df = pd.read_csv(csv_path)
    item_ids = subject + "." + df["item"].astype(str)

    # Filter to only IRT items
    mask = item_ids.isin(irt_item_set)
    n_skip = (~mask).sum()
    skipped_rows += n_skip
    if n_skip > 0:
        df = df[mask.values]
        item_ids = item_ids[mask.values]

    row_idx = df["source"].map(model_to_idx).values
    col_idx = item_ids.map(item_to_idx).values
    vals = df["correct"].values.astype(np.int8)

    mat[row_idx, col_idx] = vals
    total_rows += len(df)
    print(f"  {subject}: {len(df):,} rows, {n_skip:,} skipped ({time.time()-t0:.1f}s)")

    del df, item_ids, row_idx, col_idx, vals, mask
    gc.collect()

print(f"\nTotal rows used: {total_rows:,}, skipped (non-IRT items): {skipped_rows:,}")

n_missing = int((mat == -1).sum())
n_zero = int((mat == 0).sum())
n_one = int((mat == 1).sum())
print(f"Missing: {n_missing:,} ({n_missing/mat.size*100:.3f}%)")
print(f"Incorrect: {n_zero:,} ({n_zero/mat.size*100:.1f}%)")
print(f"Correct: {n_one:,} ({n_one/mat.size*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# Compute per-item and per-model stats BEFORE converting to sparse
# ═══════════════════════════════════════════════════════════════════

print("\nComputing per-item and per-model statistics...")

observed_mask = (mat != -1)

# Per-item: empirical_p = mean(correct | observed)
item_n_observed = observed_mask.sum(axis=0)                   # (N_ITEMS,)
item_n_correct = ((mat == 1)).sum(axis=0)                     # (N_ITEMS,)
empirical_p = np.where(item_n_observed > 0,
                       item_n_correct / item_n_observed, np.nan)

# Per-model: accuracy = mean(correct | observed)
model_n_observed = observed_mask.sum(axis=1)                  # (N_MODELS,)
model_n_correct = ((mat == 1)).sum(axis=1)                    # (N_MODELS,)
model_accuracy = np.where(model_n_observed > 0,
                          model_n_correct / model_n_observed, np.nan)

del observed_mask
gc.collect()


# ═══════════════════════════════════════════════════════════════════
# Convert to sparse CSR and save
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3: Saving response_matrix.npz")
print("=" * 60)

mat[mat == -1] = 0  # missing → 0 for sparse (CSR drops explicit 0s anyway)
sparse_mat = csr_matrix(mat)
del mat
gc.collect()

print(f"CSR: shape={sparse_mat.shape}, nnz={sparse_mat.nnz:,}, "
      f"density={sparse_mat.nnz / (N_MODELS * N_ITEMS) * 100:.1f}%")

save_npz(os.path.join(ARTIFACTS, "response_matrix.npz"), sparse_mat)
np.savez_compressed(
    os.path.join(ARTIFACTS, "response_matrix_index.npz"),
    model_names=np.array(model_list, dtype=object),
    item_ids=np.array(item_list, dtype=object),
)
print("Saved response_matrix.npz + response_matrix_index.npz")


# ═══════════════════════════════════════════════════════════════════
# STEP 4: item_metadata.csv
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 4: Building item_metadata.csv")
print("=" * 60)

# irt_params already loaded at top; reuse it
irt_lookup = irt_params.set_index("item")

item_meta = pd.DataFrame({"item_id": item_list})
item_meta["subject"] = [it.rsplit(".", 1)[0] for it in item_list]
item_meta["difficulty"] = item_meta["item_id"].map(irt_lookup["diff"])
item_meta["discrimination"] = item_meta["item_id"].map(irt_lookup["disc"])
item_meta["sd"] = item_meta["item_id"].map(irt_lookup["sd"])
item_meta["is_negative_disc"] = item_meta["discrimination"] < 0
item_meta["empirical_p"] = empirical_p

irt_matched = item_meta["difficulty"].notna().sum()
print(f"IRT params matched: {irt_matched}/{len(item_meta)} (should be {N_ITEMS})")

item_meta.to_csv(os.path.join(ARTIFACTS, "item_metadata.csv"), index=False)
print(f"Saved item_metadata.csv ({len(item_meta)} items)")


# ═══════════════════════════════════════════════════════════════════
# STEP 5: model_metadata.csv
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 5: Building model_metadata.csv")
print("=" * 60)

# --- Family extraction ---

FAMILY_PATTERNS = [
    (r"(?i)llama[-_. ]?3\.?1",            "llama-3.1"),
    (r"(?i)llama[-_. ]?3\.?3",            "llama-3.3"),
    (r"(?i)(meta[-_])?llama[-_. ]?3",      "llama-3"),
    (r"(?i)(meta[-_])?llama[-_. ]?2",      "llama-2"),
    (r"(?i)code[-_]?llama",                "codellama"),
    (r"(?i)mixtral",                       "mixtral"),
    (r"(?i)mistral[-_]?nemo",              "mistral-nemo"),
    (r"(?i)mistral[-_]?small",             "mistral-small"),
    (r"(?i)mistral",                       "mistral"),
    (r"(?i)qwen[-_]?2\.5",                "qwen-2.5"),
    (r"(?i)qwen[-_]?2",                   "qwen-2"),
    (r"(?i)qwen[-_]?1\.5",                "qwen-1.5"),
    (r"(?i)qwen",                          "qwen"),
    (r"(?i)phi[-_]?4",                     "phi-4"),
    (r"(?i)phi[-_]?3\.?5",                "phi-3.5"),
    (r"(?i)phi[-_]?3",                     "phi-3"),
    (r"(?i)phi[-_]?2",                     "phi-2"),
    (r"(?i)gemma[-_]?2",                   "gemma-2"),
    (r"(?i)gemma",                         "gemma"),
    (r"(?i)(?:01[-_]ai[/])?yi[-_]?1\.5",  "yi-1.5"),
    (r"(?i)(?:01[-_]ai[/])?yi[-_]",       "yi"),
    (r"(?i)falcon[-_]?2",                 "falcon-2"),
    (r"(?i)falcon",                        "falcon"),
    (r"(?i)mpt[-_]",                       "mpt"),
    (r"(?i)internlm[-_]?2\.5",            "internlm-2.5"),
    (r"(?i)internlm[-_]?2",               "internlm-2"),
    (r"(?i)internlm",                      "internlm"),
    (r"(?i)baichuan[-_]?2",               "baichuan-2"),
    (r"(?i)baichuan",                      "baichuan"),
    (r"(?i)deep[-_]?seek[-_]?v3",          "deepseek-v3"),
    (r"(?i)deep[-_]?seek[-_]?v2\.5",       "deepseek-v2.5"),
    (r"(?i)deep[-_]?seek[-_]?v2",          "deepseek-v2"),
    (r"(?i)deep[-_]?seek",                 "deepseek"),
    (r"(?i)command[-_]?r",                 "command-r"),
    (r"(?i)solar",                         "solar"),
    (r"(?i)star[-_]?coder[-_]?2",          "starcoder-2"),
    (r"(?i)star[-_]?coder",                "starcoder"),
    (r"(?i)olmo[-_]?2",                    "olmo-2"),
    (r"(?i)olmo",                          "olmo"),
    (r"(?i)orca[-_]?2",                    "orca-2"),
    (r"(?i)orca",                          "orca"),
    (r"(?i)zephyr",                        "zephyr"),
    (r"(?i)openchat",                      "openchat"),
    (r"(?i)neural[-_]?chat",               "neuralchat"),
    (r"(?i)wizard[-_]?lm[-_]?2",           "wizardlm-2"),
    (r"(?i)wizard[-_]?lm",                 "wizardlm"),
    (r"(?i)wizard[-_]?math",               "wizardmath"),
    (r"(?i)wizard[-_]?coder",              "wizardcoder"),
    (r"(?i)nous[-_]?hermes[-_]?2",         "nous-hermes-2"),
    (r"(?i)nous[-_]?hermes",               "nous-hermes"),
    (r"(?i)open[-_]?hermes",               "openhermes"),
    (r"(?i)abel",                          "abel"),
    (r"(?i)tulu[-_]?2",                    "tulu-2"),
    (r"(?i)tulu[-_]?3",                    "tulu-3"),
    (r"(?i)tulu",                          "tulu"),
    (r"(?i)vicuna",                        "vicuna"),
    (r"(?i)stable[-_]?lm",                 "stablelm"),
    (r"(?i)map[-_]?neo",                   "map-neo"),
    (r"(?i)amber",                         "amber"),
    (r"(?i)c4ai",                          "c4ai"),
    (r"(?i)dbrx",                          "dbrx"),
    (r"(?i)jamba",                         "jamba"),
    (r"(?i)exaone",                        "exaone"),
    (r"(?i)glm[-_]?4",                     "glm-4"),
    (r"(?i)chatglm",                       "chatglm"),
    (r"(?i)smaug",                         "smaug"),
    (r"(?i)open[-_]?chat",                 "openchat"),
]

COMPILED_FAMILY = [(re.compile(p), name) for p, name in FAMILY_PATTERNS]


def extract_family(model_name):
    for pat, name in COMPILED_FAMILY:
        if pat.search(model_name):
            return name
    return "unknown"


# --- Training method extraction ---

TRAINING_PATTERNS = [
    (re.compile(r"(?i)[-_. ]dpo"),   "DPO"),
    (re.compile(r"(?i)[-_. ]orpo"),  "ORPO"),
    (re.compile(r"(?i)[-_. ]ppo"),   "PPO"),
    (re.compile(r"(?i)[-_. ]kto"),   "KTO"),
    (re.compile(r"(?i)[-_. ]rlhf"),  "RLHF"),
    (re.compile(r"(?i)[-_. ]grpo"),  "GRPO"),
    (re.compile(r"(?i)[-_. ]simpo"), "SimPO"),
    (re.compile(r"(?i)[-_. ]sft"),   "SFT"),
]


def extract_training_method(model_name):
    for pat, method in TRAINING_PATTERNS:
        if pat.search(model_name):
            return method
    return "unknown"


# --- Release year/half heuristic ---
# Based on known model family release timelines

FAMILY_RELEASE = {
    "llama-2":       ("2023", "H2"),
    "codellama":     ("2023", "H2"),
    "llama-3":       ("2024", "H1"),
    "llama-3.1":     ("2024", "H2"),
    "llama-3.3":     ("2024", "H2"),
    "mistral":       ("2023", "H2"),
    "mistral-nemo":  ("2024", "H2"),
    "mistral-small": ("2024", "H2"),
    "mixtral":       ("2024", "H1"),
    "qwen":          ("2023", "H2"),
    "qwen-1.5":      ("2024", "H1"),
    "qwen-2":        ("2024", "H1"),
    "qwen-2.5":      ("2024", "H2"),
    "phi-2":         ("2023", "H2"),
    "phi-3":         ("2024", "H1"),
    "phi-3.5":       ("2024", "H2"),
    "phi-4":         ("2024", "H2"),
    "gemma":         ("2024", "H1"),
    "gemma-2":       ("2024", "H1"),
    "yi":            ("2023", "H2"),
    "yi-1.5":        ("2024", "H1"),
    "falcon":        ("2023", "H1"),
    "falcon-2":      ("2024", "H1"),
    "mpt":           ("2023", "H1"),
    "internlm-2":    ("2024", "H1"),
    "internlm-2.5":  ("2024", "H2"),
    "deepseek":      ("2024", "H1"),
    "deepseek-v2":   ("2024", "H1"),
    "deepseek-v2.5": ("2024", "H2"),
    "deepseek-v3":   ("2024", "H2"),
    "command-r":     ("2024", "H1"),
    "solar":         ("2024", "H1"),
    "olmo":          ("2024", "H1"),
    "olmo-2":        ("2024", "H2"),
    "dbrx":          ("2024", "H1"),
    "jamba":         ("2024", "H1"),
    "glm-4":         ("2024", "H1"),
    "vicuna":        ("2023", "H1"),
    "zephyr":        ("2023", "H2"),
    "openchat":      ("2023", "H2"),
    "tulu-2":        ("2023", "H2"),
    "tulu-3":        ("2024", "H2"),
    "stablelm":      ("2023", "H2"),
    "nous-hermes-2": ("2024", "H1"),
    "openhermes":    ("2023", "H2"),
    "wizardlm":      ("2023", "H1"),
    "wizardlm-2":    ("2024", "H1"),
    "smaug":         ("2024", "H1"),
    "exaone":        ("2024", "H2"),
    "c4ai":          ("2024", "H1"),
}


def extract_release(model_name, family):
    if family in FAMILY_RELEASE:
        return FAMILY_RELEASE[family]
    return ("unknown", "unknown")


# --- Build the metadata DataFrame ---

# Load reference files
model_acc_df = pd.read_csv(os.path.join(ARTIFACTS, "metabench_model_accuracy.csv"))
acc_lookup = dict(zip(model_acc_df["model"], model_acc_df["accuracy"]))

rl_df = pd.read_csv(os.path.join(ARTIFACTS, "metabench_rl_models.csv"))
rl_family_lookup = dict(zip(rl_df["model"], rl_df["family"]))
rl_signals_lookup = dict(zip(rl_df["model"], rl_df["rl_signals"]))

rows = []
for i, name in enumerate(model_list):
    family = rl_family_lookup.get(name) or extract_family(name)
    method = extract_training_method(name)
    # If method unknown, check rl_signals
    if method == "unknown" and name in rl_signals_lookup:
        sig = rl_signals_lookup[name]
        for kw, label in [("dpo", "DPO"), ("orpo", "ORPO"), ("ppo", "PPO"),
                          ("kto", "KTO"), ("rlhf", "RLHF"), ("grpo", "GRPO"),
                          ("simpo", "SimPO")]:
            if kw in sig.lower():
                method = label
                break

    year, half = extract_release(name, family)
    acc = acc_lookup.get(name, model_accuracy[i])

    rows.append({
        "model_name": name,
        "family": family,
        "release_year": year,
        "release_half": half,
        "training_method": method,
        "accuracy": round(acc, 6),
    })

model_meta = pd.DataFrame(rows)

# Cohort temporal: 2023 / 2024 / other
def assign_cohort_temporal(year):
    if year in ("2023", "2024"):
        return year
    return "other"

model_meta["cohort_temporal"] = model_meta["release_year"].apply(assign_cohort_temporal)

# Accuracy quintile (1=lowest, 5=highest)
model_meta["cohort_accuracy_quintile"] = pd.qcut(
    model_meta["accuracy"], q=5, labels=[1, 2, 3, 4, 5]
).astype(int)

model_meta.to_csv(os.path.join(ARTIFACTS, "model_metadata.csv"), index=False)

family_counts = model_meta["family"].value_counts()
print(f"Saved model_metadata.csv ({len(model_meta)} models)")
print(f"  Top families: {dict(family_counts.head(10))}")
print(f"  Unknown family: {(model_meta['family'] == 'unknown').sum()}")
print(f"  Training methods: {dict(model_meta['training_method'].value_counts())}")
print(f"  Release years: {dict(model_meta['release_year'].value_counts())}")
print(f"  Cohort temporal: {dict(model_meta['cohort_temporal'].value_counts())}")


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

# 1. Shape check
print(f"\n1. Shape: {sparse_mat.shape} (expected ~5219 x 12508)")

# 2. Density check
density = sparse_mat.nnz / (N_MODELS * N_ITEMS)
print(f"2. Density (fraction of 1s): {density:.3f} (expected ~0.59)")

# 3. Cross-check: 5 random models, compare matrix row mean vs model_accuracy.csv
print("\n3. Model accuracy cross-check:")
rng = np.random.RandomState(42)
sample_model_idx = rng.choice(N_MODELS, 5, replace=False)
for idx in sample_model_idx:
    name = model_list[idx]
    row = sparse_mat.getrow(idx).toarray().flatten()
    mat_acc = row.sum() / N_ITEMS  # approximate (treats missing as 0)
    ref_acc = acc_lookup.get(name, float("nan"))
    diff = abs(mat_acc - ref_acc) if not np.isnan(ref_acc) else float("nan")
    print(f"  {name[:50]:50s}  matrix={mat_acc:.4f}  ref={ref_acc:.4f}  delta={diff:.4f}")

# 4. Cross-check: 5 random items, compare matrix column mean vs IRT difficulty
print("\n4. Item empirical_p cross-check:")
sample_item_idx = rng.choice(N_ITEMS, 5, replace=False)
for idx in sample_item_idx:
    item_id = item_list[idx]
    col = sparse_mat.getcol(idx).toarray().flatten()
    mat_p = col.sum() / N_MODELS
    irt_diff = irt_lookup.loc[item_id, "diff"] if item_id in irt_lookup.index else float("nan")
    print(f"  {item_id:40s}  empirical_p={mat_p:.4f}  irt_diff={irt_diff:.4f}")

# 5. File sizes
print("\n5. Output files:")
for fname in ["response_matrix.npz", "response_matrix_index.npz",
              "item_metadata.csv", "model_metadata.csv"]:
    fpath = os.path.join(ARTIFACTS, fname)
    sz = os.path.getsize(fpath) / (1024**2)
    print(f"  {fname}: {sz:.1f}MB")

print("\nDone.")
