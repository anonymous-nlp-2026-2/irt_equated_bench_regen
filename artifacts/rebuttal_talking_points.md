# Rebuttal Talking Points (prepared 2026-05-23)

## B1 (Q3): Difficulty-dependent contamination defense
**审稿人可能追问**: difficulty × model_type 交互效应下，direction reversal 仍可被 contamination 解释。
**Defense**: "We agree that a difficulty × model-type interaction remains plausible; the base-vs-instruct decomposition constrains but does not eliminate this account. Crucially, regardless of mechanism, the direction reversal itself constitutes measurement instability—items that favor one group in base models disfavor the same group in instruct models, undermining cross-generation comparability."
**E2 regression**: Logistic regression confirms difficulty×model-type interaction (β=1.56, OR=4.75, p<10⁻¹⁸⁸); controlling for difficulty, model type remains significant (OR=4.24, p<10⁻²²⁵). See §app:interaction.
**E3 mixed mechanism**: Semi-synthetic Scheme D (base=difficulty-dependent, instruct=uniform) reproduces overall ρ (−0.676 vs observed −0.597) and χ² reversal (634 vs 562.7), but cannot replicate instruct near-zero gradient: simulated ρ_instruct=−0.302±0.090 vs observed −0.079 (3.8× gap). No contamination scheme matches instruct DIF pattern → supports non-contamination mechanism (alignment-induced capability shifts). See §app:mixed_mechanism.

## B5 (Q6): ≥500 threshold
**审稿人可能追问**: K=3 的 power 是多少？最低需要多少模型？
**Defense**: 引用 Clauser & Mazor (1998) 和 Penfield & Camilli (2006) 关于 K=3 的已有结论。如果需要，可跑 K=3 power simulation（复用 §4.1 框架，半小时出结果）。
**E1 K=3 power**: K=3 vs K=5 TPR 差异 <0.02（所有条件下）。N≥80 时 K=3 达到 TPR=0.571 (|Δ_MH|≥1.4), AUC=0.905。详见 Tab k3_power + §5.2。

## B8: Size-stratified DEFF
**审稿人可能追问**: 为什么不用 tier-specific DEFF？哪个 tier 的 C% 才对？
**Defense**: 没有单一"正确"tier。所有 5,227 模型共同构成分析群体。Simple-mean (m̄=63) 和 weighted-mean (m̄=254) 分别代表 family-weighted 和 model-weighted 视角，真实情况介于两者之间。

## Negative results framing
| Result | Framing | Strength |
|--------|---------|----------|
| AUROC=0.60 | Emergent cohort-item interactions, not fixed defects | 4/5 |
| R²=0.0007 | Exclusion evidence (domain improvement ≠ DIF) | 5/5 |
| ρ=-0.597 diagnostic limit | DIF is diagnostic lens, not causal detector | 4/5 |
| Jaccard=0.206 | Orthogonal constructs (core contribution) | 5/5 |
