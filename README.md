# Temporal Score Incomparability in LLM Benchmarks

Code and data for the paper: *Temporal Score Incomparability in LLM Benchmarks: A Large-Scale Psychometric Audit of 5,227 Models* (ARR / EMNLP 2026 submission).

## Abstract

Every leaderboard comparison assumes that benchmark items measure the same construct across model generations. In educational testing, this foundational property, measurement invariance, is routinely verified. In large language model (LLM) evaluation, it has never been tested at scale. We bring differential item functioning (DIF) analysis from psychometrics to the METABENCH response matrix (5,227 models on MMLU) and find that 31% of items function systematically differently between 2023 and 2024 model cohorts, 50x above random baselines. This instability is not driven by data contamination: DIF flags are near-independent of web-overlap labels (phi = 0.012), an overlooked diagnostic dimension. Nor does it threaten validity: aggregate rankings remain near-perfect (rho ~ 0.997). Instead, unstable items expose which capability dimensions shift across generations; base and instruct models exhibit opposite DIF directions. We develop a semi-synthetic validation framework with cross-subject matching that guarantees zero false-positive inflation, replicate findings across six benchmarks (18-31% instability), and propose an actionable audit protocol. LLM benchmarks need psychometric monitoring, not just score reporting.

## Overview

This repository contains analysis scripts for applying Mantel-Haenszel Differential Item Functioning (MH-DIF) analysis to LLM benchmark evaluation. The methodology treats benchmark items as psychometric test items and detects measurement instability (differential item functioning) across temporal cohorts of language models. Data is drawn from the [METABENCH](https://github.com/adkipnis/metabench) response matrix (Kipnis et al., ICLR 2025).

## Repository Structure

### `artifacts/` — Analysis Scripts

| Directory | Description |
|-----------|-------------|
| `build_response_matrix.py` | Builds response matrix and metadata from METABENCH MMLU data |
| `metabench_explore.py` | METABENCH data exploration: cohort sizes, IRT parameters, response matrix statistics |
| `plan_001/` | Purified MH-DIF detection on MMLU with temporal (2023 vs 2024) cohort comparisons |
| `plan_001_real_data_dif_mmlu/` | Matched-ability temporal DIF within accuracy quintiles |
| `plan_001b/` | Matched-ability DIF analysis using temporal comparisons within accuracy quintiles |
| `plan_004/` | GSM8K replication of MH-DIF methodology with standard and external matching |
| `plan_010/` | Architecture-stratified DIF analysis on decoder-only models |
| `plan_011/` | Semi-synthetic DIF validation: injects known contamination and tests detection power |
| `plan_012/` | IRT theta-based conditioning for MH-DIF stratification |
| `plan_013/` | Yen's Q3 local independence diagnostic for MMLU items |
| `plan_013_supplement/` | Cross-analysis of local dependence (Q3) vs DIF concentration |
| `plan_014/` | Within-subject temporal DIF and cross-subject matching |
| `plan_015/` | Bifactor-conditioned MH-DIF using per-subject IRT theta |
| `plan_016/` | Within-subject semi-synthetic injection validation |
| `plan_016_supplement/` | Bootstrap CIs for ΔFPR in cross-subject matching |
| `plan_017/` | Downstream impact: model ranking stability from DIF-C item removal |
| `plan_017_supplement/` | Rank shift distribution and top rank-changing model identification |
| `plan_018/` | Threshold sensitivity analysis with BH multiple testing correction |
| `plan_019/` | Instability decomposition into domain-level improvement vs residual instability |
| `plan_020/` | Circularity ablation comparing DIF-C removal against alternative selection strategies |
| `plan_021/` | Non-uniform DIF robustness check via logistic regression |
| `plan_022/` | Model independence sensitivity: three de-duplication levels |
| `plan_023/` | DIF-C item characterization across difficulty, discrimination, and domain |
| `plan_024/` | Within-family analysis (Llama-2 vs Llama-3) with paired ability-matched DIF |
| `plan_026/` | Empirical comparison of DIF-C against Li et al. web-overlap contamination labels |
| `plan_027_temporal_overlap/` | Temporal DIF signal robustness via cohort ability distribution overlap |
| `plan_028_bootstrap_ci/` | Bootstrap 95% CI for MH-DIF C% across 6 benchmarks |
| `plan_028_multi_bench_dif/` | Multi-benchmark temporal MH-DIF (ARC, HellaSwag, WinoGrande, TruthfulQA) |
| `plan_029_permutation_null/` | Permutation null test for temporal DIF C% (1000 permutations) |
| `plan_030_odds_ratio/` | Odds ratio and CI analysis for accuracy-DIF regression |
| `plan_031_temporal_granularity/` | Full-year vs 6-month vs within-year DIF signal comparison |
| `plan_032_dif_cleaned_mmlu/` | Model ranking changes after removing ETS C-flagged items |
| `plan_033_subject_dif_concentration/` | Subject-level DIF concentration analysis with domain group summaries |
| `plan_034_multi_subject_injection/` | Multi-subject simultaneous injection ΔFPR robustness test |
| `plan_035_q3_inflation_sim/` | Local dependence inflation of MH-DIF Type I error quantification |
| `deff_parametric_curve/` | Design effect (DEFF) parametric curve fitting vs family size |
| `difficulty_modeltype_interaction/` | Difficulty × model-type interaction via logistic regression |
| `k3_power_simulation/` | K=3 vs K=5 strata power simulation for MH-DIF |
| `lofo_robustness/` | Leave-one-family-out (LOFO) robustness check for temporal DIF C% |
| `mf1_lofo_power_calibration/` | Power vs composition effect separation in LOFO analysis |
| `mf2_deff_estimation/` | DEFF estimation with cohort-consistent reference/focal models |
| `mf2_practical_significance/` | Stable-item Spearman ρ and flip rate across 6 benchmarks |
| `mf2_rho_degradation/` | Spearman ρ degradation under progressive DIF-C item removal |
| `mixed_mechanism_semisynthetic/` | Mixed-mechanism semi-synthetic injection (difficulty × model-type) |
| `predictive_model/` | Item-property predictive model for DIF-C classification |
| `score_inflation_analysis/` | Score inflation quantification and DIF-corrected scoring |
| `sf1_cluster_bootstrap/` | Family-level cluster bootstrap CIs for temporal C% |
| `sf8_power_sensitivity/` | Power sensitivity: TPR as function of item discrimination |
| `temporal_accumulation/` | C% monotonicity across cumulative temporal distances |
| `dif_diagnostic.py` | DIF power diagnostic: raw vs BH vs Bonferroni TPR/FPR |
| `dif_phase2_simulation.py` | Semi-synthetic DIF power simulation using real MMLU IRT parameters |
| `dif_power_simulation.py` | DIF power Monte Carlo with 2PL IRT and Mantel-Haenszel/LR methods |

#### Rebuttal analyses (`r*` directories)

| Directory | Description |
|-----------|-------------|
| `r14_mf1_domain_profile/` | Cross-subject-unique C item concentration with Cohen's kappa |
| `r15_base_instruct_temporal_dif/` | Within-type temporal DIF (base-only and instruct-only) |
| `r15_random_split_control/` | Random split control within 2024 cohort as sanity check |
| `r15_subject_level_rho/` | Subject-level ranking correlation (instability masked by aggregate ρ≈0.997) |
| `r16_threshold_calibration/` | Random-split null distribution for ETS classification threshold calibration |
| `r17_evidence/` | Weighted-mean DEFF downstream analysis |
| `r17_gema_enrichment_ci/` | Bootstrap CI for DIF-C enrichment with MMLU-Redux annotations |
| `r17_gsm1k_jaccard_permutation/` | GSM1K model-level Jaccard permutation null test |
| `r17_k_sensitivity/` | K (strata count) sensitivity for MH-DIF (K∈{3,5,7,10,20}) |
| `r18_base_vs_instruct_dif/` | Base vs instruct temporal MH-DIF with within-type stratification |
| `r18_dedup_k_curve/` | Family de-duplication C% vs K curve |
| `r18_icc_by_family_size/` | Per-family homogeneity (mean pairwise Pearson r) vs family size |
| `r18_rank_bucket_reversal/` | Rank-bucket reversal distribution for closely-ranked models |
| `r18_restricted_range_rho/` | Restricted-range Spearman ρ for difficulty-DIF direction association |
| `r19_model_size_dif/` | Model-size DIF decomposition (Small/Medium/Large) |
| `r19_open_closed_dif/` | Open vs closed-source DIF analysis |
| `r19_semisynthetic_gradient/` | Difficulty-dependent semi-synthetic injection patterns |
| `r19_temporal_granularity/` | Half-year cohort pairwise temporal granularity analysis |
| `r20_size_stratified_deff/` | Size-stratified DEFF with family definition sensitivity |
| `r22_composition_control/` | Base/instruct composition control via matched subsampling |
| `r22_deff_simulation/` | DEFF mean vs weighted-mean simulation with Rao-Scott correction |
| `r22_within_arch_temporal/` | Within-architecture temporal DIF ruling out cross-arch confound |

### `figures/paper/` — Figure Scripts

| Script | Description |
|--------|-------------|
| `fig_1_global_vs_differential.py` | Figure 1: Global vs differential measurement comparison |
| `fig_2_dose_response.py` | Figure 2: Dose-response curve for semi-synthetic injection |
| `fig_3_within_subject_matching.py` | Figure 3: Within-subject matching results |
| `fig_4_downstream_impact.py` | Figure 4: Downstream ranking impact |
| `fig_A1_subject_heatmap.py` | Figure A1: Subject-level DIF heatmap |
| `fig_A2_q3_distribution.py` | Figure A2: Q3 local dependence distribution |
| `fig_A3_rho_degradation.py` | Figure A3: Spearman ρ degradation curves |

### `scripts/` — Utility Scripts

| Script | Description |
|--------|-------------|
| `r14_cooks_d_analysis.py` | Cook's distance analysis for DIF outlier detection |
| `r14_domain_profile.py` | Domain-level DIF profiling |
| `r16_dif_union.py` | Uniform + non-uniform DIF union/intersection sizes |

## Data

Raw data (response matrices, model metadata) is not included due to size.
To reproduce:
1. Download MMLU evaluation results from [METABENCH](https://github.com/adkipnis/metabench) (Kipnis et al., ICLR 2025)
2. Run `artifacts/build_response_matrix.py` to construct the response matrix
3. Run individual analysis scripts in `artifacts/plan_*/`

## Requirements

```
pip install -r requirements.txt
```

Python 3.9+ recommended.

## License

This code is released for academic use. See the paper for methodological details.
