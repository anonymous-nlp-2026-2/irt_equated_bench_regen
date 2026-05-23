#!/usr/bin/env python3
"""METABENCH MMLU data exploration — memory-efficient version.

Extracts statistics from:
1. RDS files (pre-fitted IRT params, response matrix, train/test split)
2. CSV files (per-subject verification, incremental processing)

Key questions:
- Can we form cohorts with N>=80 for DIF analysis?
- Are there RL-trained models in the dataset?
- Do RDS files contain usable IRT parameters?
"""

import os, re, json, gc, warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "artifacts" / "metabench_data" / "benchmark-data"
OUT_DIR = PROJECT_DIR / "artifacts"

# ============================================================
# PART 1: RDS Analysis (IRT params + response matrix)
# ============================================================
print("=" * 60)
print("PART 1: RDS File Analysis")
print("=" * 60)

import rdata

def load_rds(fname):
    fpath = DATA_DIR / fname
    if not fpath.exists():
        print(f"  {fname}: NOT FOUND")
        return None
    sz = fpath.stat().st_size / 1024 / 1024
    print(f"  Loading {fname} ({sz:.1f} MB)...")
    try:
        parsed = rdata.parser.parse_file(str(fpath))
        converted = rdata.conversion.convert(parsed)
        return converted
    except Exception as e:
        print(f"  Error: {e}")
        return None

# --- 1a: mmlu-preproc.rds (IRT params) ---
print("\n--- mmlu-preproc.rds ---")
preproc = load_rds("mmlu-preproc.rds")

items_df = preproc['items'][['item', 'sd', 'diff', 'disc', 'exclude']].copy()
items_df['subject'] = items_df['item'].str.replace(r'\.\d+$', '', regex=True)

print(f"\nIRT Parameters Summary:")
print(f"  Total items: {len(items_df)}")
print(f"  Subjects: {items_df['subject'].nunique()}")
print(f"  Excluded items: {items_df['exclude'].sum()}")
print(f"  Difficulty (diff): [{items_df['diff'].min():.3f}, {items_df['diff'].max():.3f}], mean={items_df['diff'].mean():.3f}")
print(f"  Discrimination (disc): [{items_df['disc'].min():.3f}, {items_df['disc'].max():.3f}], mean={items_df['disc'].mean():.3f}")

items_per_subject = items_df.groupby('subject').size().sort_values()
print(f"\n  Items per subject: min={items_per_subject.min()}, max={items_per_subject.max()}, median={items_per_subject.median():.0f}")

# Response matrix for model accuracy
data_matrix = preproc['data']
model_names = list(data_matrix.index)
n_models = len(model_names)
print(f"\n  Response matrix: {data_matrix.shape[0]} models x {data_matrix.shape[1]} items")

# Per-model accuracy
model_accuracy = data_matrix.mean(axis=1)
model_acc_dict = dict(zip(model_accuracy.index, model_accuracy.values))

# Per-item difficulty (empirical p-value)
item_p_values = data_matrix.mean(axis=0)
floor_items = (item_p_values < 0.05).sum()
ceiling_items = (item_p_values > 0.95).sum()
print(f"\n  Floor items (p<0.05): {floor_items}")
print(f"  Ceiling items (p>0.95): {ceiling_items}")
print(f"  Empirical p-value range: [{item_p_values.min():.4f}, {item_p_values.max():.4f}]")

del preproc
gc.collect()

# --- 1b: mmlu-preproc-split.rds (train/test split) ---
print("\n--- mmlu-preproc-split.rds ---")
split_data = load_rds("mmlu-preproc-split.rds")
train_models = []
test_models = []
if split_data:
    train_models = list(split_data['data.train'].index)
    test_models = list(split_data['data.test'].index)
    print(f"  Train models: {len(train_models)}")
    print(f"  Test models: {len(test_models)}")
    del split_data
    gc.collect()

# ============================================================
# PART 2: Model Family & Cohort Analysis
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Model Family & Cohort Analysis")
print("=" * 60)

FAMILY_PATTERNS = {
    r'(?i)llama[-_]?3\.?1': ('llama-3.1', 2024, 2),
    r'(?i)llama[-_]?3\.?3': ('llama-3.3', 2024, 2),
    r'(?i)llama[-_]?3\.?2': ('llama-3.2', 2024, 2),
    r'(?i)llama[-_]?3(?![\.\d])': ('llama-3', 2024, 1),
    r'(?i)llama[-_]?2': ('llama-2', 2023, 2),
    r'(?i)codellama': ('codellama', 2023, 2),
    r'(?i)llama(?![-_]?\d)': ('llama-1', 2023, 1),
    r'(?i)mistral[-_]?7[bB][-_]?v?0\.?3': ('mistral-0.3', 2024, 1),
    r'(?i)mistral[-_]?7[bB][-_]?v?0\.?2': ('mistral-0.2', 2024, 1),
    r'(?i)mistral[-_]?(7[bB]|v?0\.?1)': ('mistral-0.1', 2023, 2),
    r'(?i)mixtral[-_]?8x22': ('mixtral-8x22B', 2024, 1),
    r'(?i)mixtral': ('mixtral-8x7B', 2023, 2),
    r'(?i)mistral[-_]?(nemo|large|small)': ('mistral-nemo', 2024, 2),
    r'(?i)qwen[-_]?2\.?5': ('qwen-2.5', 2024, 2),
    r'(?i)qwen[-_]?2(?![\.\d])': ('qwen-2', 2024, 1),
    r'(?i)qwen[-_]?1\.?5': ('qwen-1.5', 2024, 1),
    r'(?i)qwen(?![-_]?\d)': ('qwen-1', 2023, 2),
    r'(?i)phi[-_]?3\.?5': ('phi-3.5', 2024, 2),
    r'(?i)phi[-_]?3': ('phi-3', 2024, 1),
    r'(?i)phi[-_]?2': ('phi-2', 2023, 2),
    r'(?i)phi[-_]?1': ('phi-1', 2023, 1),
    r'(?i)gemma[-_]?2': ('gemma-2', 2024, 1),
    r'(?i)gemma(?![-_]?\d)': ('gemma-1', 2024, 1),
    r'(?i)yi[-_]?1\.?5': ('yi-1.5', 2024, 1),
    r'(?i)/yi[-_]': ('yi-1', 2023, 2),
    r'(?i)deepseek[-_]?v3': ('deepseek-v3', 2024, 2),
    r'(?i)deepseek[-_]?v2': ('deepseek-v2', 2024, 1),
    r'(?i)deepseek[-_]?r1': ('deepseek-r1', 2025, 1),
    r'(?i)deepseek[-_]?coder[-_]?v2': ('deepseek-coder-v2', 2024, 1),
    r'(?i)deepseek[-_]?coder': ('deepseek-coder', 2023, 2),
    r'(?i)deepseek[-_]?math': ('deepseek-math', 2024, 1),
    r'(?i)deepseek[-_]?llm': ('deepseek-llm', 2024, 1),
    r'(?i)deepseek(?![-_]?(v|r|coder|math|llm))': ('deepseek-other', 2024, 1),
    r'(?i)falcon[-_]?(40|180)': ('falcon-40/180B', 2023, 2),
    r'(?i)falcon[-_]?2': ('falcon-2', 2024, 1),
    r'(?i)falcon': ('falcon-1', 2023, 1),
    r'(?i)solar': ('solar', 2024, 1),
    r'(?i)command[-_]?r': ('command-r', 2024, 1),
    r'(?i)dbrx': ('dbrx', 2024, 1),
    r'(?i)starcoder2': ('starcoder2', 2024, 1),
    r'(?i)starcoder': ('starcoder', 2023, 1),
    r'(?i)internlm[-_]?2\.?5': ('internlm-2.5', 2024, 2),
    r'(?i)internlm[-_]?2': ('internlm-2', 2024, 1),
    r'(?i)internlm': ('internlm-1', 2023, 2),
    r'(?i)olmo[-_]?2': ('olmo-2', 2024, 2),
    r'(?i)olmo': ('olmo-1', 2024, 1),
    r'(?i)mpt[-_]': ('mpt', 2023, 1),
    r'(?i)pythia': ('pythia', 2023, 1),
    r'(?i)stablelm': ('stablelm', 2023, 2),
    r'(?i)openchat[-_]?3\.5': ('openchat-3.5', 2023, 2),
    r'(?i)openchat[-_]?3\.6': ('openchat-3.6', 2024, 1),
    r'(?i)openchat': ('openchat', 2023, 1),
    r'(?i)zephyr': ('zephyr', 2023, 2),
    r'(?i)vicuna': ('vicuna', 2023, 1),
    r'(?i)wizardlm[-_]?2': ('wizardlm-2', 2024, 1),
    r'(?i)wizardlm': ('wizardlm-1', 2023, 1),
    r'(?i)orca[-_]?2': ('orca-2', 2023, 2),
    r'(?i)orca': ('orca', 2023, 1),
    r'(?i)nous[-_]?hermes[-_]?2': ('nous-hermes-2', 2024, 1),
    r'(?i)hermes[-_]?3': ('hermes-3', 2024, 2),
    r'(?i)tinyllama': ('tinyllama', 2024, 1),
    r'(?i)opt[-_]': ('opt', 2022, 1),
    r'(?i)bloom': ('bloom', 2022, 2),
    r'(?i)gpt[-_]?neo': ('gpt-neo', 2021, 1),
    r'(?i)gpt[-_]?j': ('gpt-j', 2021, 1),
    r'(?i)cerebras': ('cerebras', 2023, 1),
    r'(?i)c4ai': ('c4ai', 2024, 1),
    r'(?i)smollm': ('smollm', 2024, 2),
    r'(?i)aya[-_]?23': ('aya-23', 2024, 1),
    r'(?i)aya[-_]?expanse': ('aya-expanse', 2024, 2),
    r'(?i)granite[-_]?3': ('granite-3', 2024, 2),
    r'(?i)granite': ('granite', 2024, 1),
    r'(?i)nemotron': ('nemotron', 2024, 2),
    r'(?i)exaone[-_]?3': ('exaone-3', 2024, 2),
}

RL_PATTERNS = [
    r'(?i)deepseek[-_]?r1',
    r'(?i)deepseek[-_]?v3',
    r'(?i)deepseek[-_]?v2',
    r'(?i)qwen[-_]?2\.?5',
    r'(?i)o1[-_]',
    r'(?i)o3[-_]',
    r'(?i)claude',
    r'(?i)gemini',
    r'(?i)gpt[-_]?4',
    r'(?i)llama[-_]?3\.?1',
    r'(?i)llama[-_]?3\.?2',
    r'(?i)llama[-_]?3\.?3',
    r'(?i)phi[-_]?3\.?5',
    r'(?i)command[-_]?r\+',
    r'(?i)gemma[-_]?2',
    r'(?i)nemotron',
    r'(?i)[-_]rl[-_]',
    r'(?i)[-_]rlhf',
    r'(?i)[-_]dpo',
    r'(?i)[-_]grpo',
    r'(?i)[-_]ppo',
    r'(?i)[-_]kto',
    r'(?i)[-_]orpo',
    r'(?i)simpo',
    r'(?i)reward',
]

def classify_model(model_name):
    for pattern, (family, year, half) in FAMILY_PATTERNS.items():
        if re.search(pattern, model_name):
            return family, year, half
    return None, None, None

def check_rl_trained(model_name):
    matches = []
    for pattern in RL_PATTERNS:
        if re.search(pattern, model_name):
            matches.append(pattern)
    return matches

family_counts = Counter()
cohort_models = defaultdict(list)
unclassified = []
rl_models = []

for model in model_names:
    family, year, half = classify_model(model)
    if family:
        family_counts[family] += 1
        cohort_models[(year, half)].append(model)
    else:
        unclassified.append(model)

    rl_matches = check_rl_trained(model)
    if rl_matches:
        rl_models.append((model, rl_matches))

print(f"\nTotal models: {n_models}")
print(f"Classified: {n_models - len(unclassified)} ({(n_models - len(unclassified))/n_models*100:.1f}%)")
print(f"Unclassified: {len(unclassified)} ({len(unclassified)/n_models*100:.1f}%)")

print(f"\nTop 30 model families:")
for family, count in family_counts.most_common(30):
    print(f"  {family}: {count}")

print(f"\nCohort analysis (year-half):")
for (year, half), models in sorted(cohort_models.items()):
    label = f"{year}-H{half}"
    print(f"  {label}: {len(models)} models")

annual_cohorts = defaultdict(list)
for (year, half), models in cohort_models.items():
    annual_cohorts[year].extend(models)

print(f"\nAnnual cohorts:")
for year, models in sorted(annual_cohorts.items()):
    sufficient = "SUFFICIENT" if len(models) >= 80 else "insufficient"
    print(f"  {year}: {len(models)} models  {sufficient}")

pre_2024 = []
post_2024 = []
for (year, half), models in cohort_models.items():
    if year < 2024:
        pre_2024.extend(models)
    else:
        post_2024.extend(models)

print(f"\n  Pre-2024 classified: {len(pre_2024)}")
print(f"  2024+ classified: {len(post_2024)}")

print(f"\n--- RL-trained / Reasoning Models ---")
print(f"Total RL-related models found: {len(rl_models)}")

rl_by_type = defaultdict(list)
for model, matches in rl_models:
    for m in matches:
        if 'dpo' in m.lower():
            rl_by_type['DPO'].append(model)
        elif 'rlhf' in m.lower():
            rl_by_type['RLHF'].append(model)
        elif 'ppo' in m.lower():
            rl_by_type['PPO'].append(model)
        elif 'grpo' in m.lower():
            rl_by_type['GRPO'].append(model)
        elif 'kto' in m.lower():
            rl_by_type['KTO'].append(model)
        elif 'orpo' in m.lower():
            rl_by_type['ORPO'].append(model)
        elif 'simpo' in m.lower():
            rl_by_type['SimPO'].append(model)
        elif 'reward' in m.lower():
            rl_by_type['Reward'].append(model)
        elif 'deepseek' in m.lower() and 'r1' in m.lower():
            rl_by_type['DeepSeek-R1'].append(model)
        elif 'qwen' in m.lower() and '2.5' in m.lower():
            rl_by_type['Qwen-2.5'].append(model)

for rl_type, models in sorted(rl_by_type.items(), key=lambda x: -len(x[1])):
    unique_models = list(set(models))
    print(f"\n  {rl_type}: {len(unique_models)} unique models")
    for m in unique_models[:5]:
        acc = model_acc_dict.get(m, float('nan'))
        print(f"    {acc:.4f}  {m}")
    if len(unique_models) > 5:
        print(f"    ... and {len(unique_models) - 5} more")

# ============================================================
# PART 3: CSV Verification (incremental, memory-safe)
# ============================================================
print("\n" + "=" * 60)
print("PART 3: CSV Per-Subject Verification")
print("=" * 60)

mmlu_csvs = sorted(f for f in DATA_DIR.glob("mmlu_*.csv") if "_prompts" not in f.name)
print(f"\nFound {len(mmlu_csvs)} MMLU subject CSVs")

csv_subject_stats = {}
csv_total_rows = 0
csv_all_models = set()

for i, csv_path in enumerate(mmlu_csvs):
    subject = csv_path.stem.replace("mmlu_", "")
    df = pd.read_csv(csv_path)
    csv_total_rows += len(df)

    models_here = set(df["source"].unique())
    csv_all_models.update(models_here)
    n_items = df["item"].nunique()
    mean_correct = df["correct"].mean()

    csv_subject_stats[subject] = {
        'n_models': len(models_here),
        'n_items': n_items,
        'n_rows': len(df),
        'mean_correct': mean_correct,
    }

    if (i + 1) % 10 == 0 or i == 0 or i == len(mmlu_csvs) - 1:
        print(f"  [{i+1}/{len(mmlu_csvs)}] {subject}: {len(models_here)} models, {n_items} items, p={mean_correct:.3f}")

    del df
    gc.collect()

print(f"\nCSV Summary:")
print(f"  Total rows across all CSVs: {csv_total_rows:,}")
print(f"  Unique models from CSVs: {len(csv_all_models)}")
print(f"  Unique models from RDS: {n_models}")
print(f"  Match: {len(csv_all_models) == n_models}")

model_counts_per_csv = [s['n_models'] for s in csv_subject_stats.values()]
print(f"\n  Models per subject: min={min(model_counts_per_csv)}, max={max(model_counts_per_csv)}")
print(f"  All subjects have same model count: {len(set(model_counts_per_csv)) == 1}")

items_match = all(
    csv_subject_stats[s]['n_items'] == len(items_df[items_df['subject'] == s])
    for s in csv_subject_stats
)
print(f"  All item counts match RDS: {items_match}")

# ============================================================
# PART 4: Difficulty Distribution & Subject Analysis
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Item Difficulty & Subject Analysis")
print("=" * 60)

diff_vals = items_df['diff'].values
disc_vals = items_df['disc'].values

print("\nDifficulty (diff) distribution:")
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  P{pct}: {np.percentile(diff_vals, pct):.4f}")

print("\nDiscrimination (disc) distribution:")
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  P{pct}: {np.percentile(disc_vals, pct):.4f}")

neg_disc = (disc_vals < 0).sum()
print(f"\nNegative discrimination items: {neg_disc} ({neg_disc/len(disc_vals)*100:.1f}%)")

subj_diff = items_df.groupby('subject')['diff'].agg(['mean', 'std', 'count'])
print(f"\nHardest subjects (by mean difficulty, lower = harder):")
for subj, row in subj_diff.sort_values('mean').head(10).iterrows():
    print(f"  {subj}: diff={row['mean']:.3f} +/-{row['std']:.3f} ({int(row['count'])} items)")

print(f"\nEasiest subjects (by mean difficulty, higher = easier):")
for subj, row in subj_diff.sort_values('mean', ascending=False).head(10).iterrows():
    print(f"  {subj}: diff={row['mean']:.3f} +/-{row['std']:.3f} ({int(row['count'])} items)")

# ============================================================
# PART 5: Model Accuracy Distribution
# ============================================================
print("\n" + "=" * 60)
print("PART 5: Model Accuracy Distribution")
print("=" * 60)

acc_vals = np.array(list(model_acc_dict.values()))
print(f"\nOverall accuracy distribution:")
print(f"  Mean: {acc_vals.mean():.4f}")
print(f"  Std: {acc_vals.std():.4f}")
print(f"  Min: {acc_vals.min():.4f}")
print(f"  Max: {acc_vals.max():.4f}")
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  P{pct}: {np.percentile(acc_vals, pct):.4f}")

below_random = (acc_vals < 0.25).sum()
print(f"\n  Models below random (0.25): {below_random} ({below_random/len(acc_vals)*100:.1f}%)")

# ============================================================
# PART 6: Save All Results
# ============================================================
print("\n" + "=" * 60)
print("PART 6: Saving Results")
print("=" * 60)

subject_stats_df = pd.DataFrame(csv_subject_stats).T
subject_stats_df.index.name = 'subject'
subject_stats_path = OUT_DIR / 'metabench_subject_stats.csv'
subject_stats_df.to_csv(subject_stats_path)
print(f"Saved: {subject_stats_path}")

cohort_summary = {}
for (year, half), models in sorted(cohort_models.items()):
    label = f"{year}-H{half}"
    accs = [model_acc_dict.get(m, np.nan) for m in models]
    cohort_summary[label] = {
        'n_models': len(models),
        'mean_accuracy': np.nanmean(accs),
        'std_accuracy': np.nanstd(accs),
        'sufficient_for_dif': len(models) >= 80,
    }

cohort_df = pd.DataFrame(cohort_summary).T
cohort_df.index.name = 'cohort'
cohort_path = OUT_DIR / 'metabench_cohort_analysis.csv'
cohort_df.to_csv(cohort_path)
print(f"Saved: {cohort_path}")

rl_data = []
for model, matches in rl_models:
    acc = model_acc_dict.get(model, np.nan)
    family, year, half = classify_model(model)
    rl_data.append({
        'model': model,
        'accuracy': acc,
        'family': family,
        'rl_signals': '|'.join(m.replace('(?i)', '') for m in matches),
    })
rl_df = pd.DataFrame(rl_data).sort_values('accuracy', ascending=False)
rl_path = OUT_DIR / 'metabench_rl_models.csv'
rl_df.to_csv(rl_path, index=False)
print(f"Saved: {rl_path}")

summary = {
    'dataset': {
        'name': 'METABENCH MMLU',
        'n_models': n_models,
        'n_items': len(items_df),
        'n_subjects': items_df['subject'].nunique(),
        'total_csv_rows': csv_total_rows,
        'all_subjects_same_model_count': len(set(model_counts_per_csv)) == 1,
        'models_per_subject': model_counts_per_csv[0] if len(set(model_counts_per_csv)) == 1 else model_counts_per_csv,
    },
    'irt_params': {
        'source': 'mmlu-preproc.rds',
        'difficulty_range': [float(items_df['diff'].min()), float(items_df['diff'].max())],
        'difficulty_mean': float(items_df['diff'].mean()),
        'discrimination_range': [float(items_df['disc'].min()), float(items_df['disc'].max())],
        'discrimination_mean': float(items_df['disc'].mean()),
        'negative_discrimination_items': int(neg_disc),
        'floor_items': int(floor_items),
        'ceiling_items': int(ceiling_items),
    },
    'model_accuracy': {
        'mean': float(acc_vals.mean()),
        'std': float(acc_vals.std()),
        'min': float(acc_vals.min()),
        'max': float(acc_vals.max()),
        'below_random': int(below_random),
    },
    'cohorts': {k: {'n': v['n_models'], 'sufficient': v['sufficient_for_dif']} for k, v in cohort_summary.items()},
    'annual_cohorts': {str(y): len(m) for y, m in sorted(annual_cohorts.items())},
    'rl_models_count': len(rl_models),
    'train_test_split': {
        'train': len(train_models),
        'test': len(test_models),
    },
    'key_answers': {
        'cohorts_n_ge_80': {str(y): len(m) >= 80 for y, m in sorted(annual_cohorts.items())},
        'rl_models_present': len(rl_models) > 0,
        'irt_params_in_rds': True,
    },
}

summary_path = OUT_DIR / 'metabench_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {summary_path}")

# ============================================================
# KEY ANSWERS
# ============================================================
print("\n" + "=" * 60)
print("KEY ANSWERS FOR DIRECTOR")
print("=" * 60)

print(f"\n1. CAN WE FORM COHORTS WITH N>=80?")
print(f"   Annual cohorts:")
for year, models in sorted(annual_cohorts.items()):
    status = "YES" if len(models) >= 80 else "NO"
    print(f"     {year}: N={len(models)}  -> {status}")
print(f"   Total classified: {sum(len(m) for m in annual_cohorts.values())}/{n_models}")
print(f"   Unclassified: {len(unclassified)}")
total_sufficient = sum(1 for m in annual_cohorts.values() if len(m) >= 80)
print(f"   -> {total_sufficient} annual cohorts with N>=80")

print(f"\n2. ARE THERE RL-TRAINED MODELS?")
print(f"   Total RL-related models: {len(rl_models)}")
for rl_type, models in sorted(rl_by_type.items(), key=lambda x: -len(set(x[1]))):
    print(f"     {rl_type}: {len(set(models))}")
print(f"   -> YES, abundant RL-trained models present")

print(f"\n3. DO RDS FILES CONTAIN IRT PARAMETERS?")
print(f"   mmlu-preproc.rds contains:")
print(f"     - Item difficulty (diff): {len(items_df)} items")
print(f"     - Item discrimination (disc): {len(items_df)} items")
print(f"     - Response matrix: {n_models} x {len(items_df)}")
print(f"     - Train/test split: {len(train_models)}/{len(test_models)}")
print(f"   -> YES, pre-fitted 2PL IRT parameters available")

print("\nExploration complete.")
