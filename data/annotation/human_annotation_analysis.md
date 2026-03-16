# Human vs GPT-OSS-120B Annotation Agreement Analysis

**Date:** 2026-03-01
**Notebook:** `analysis_human_vs_llm.ipynb`
**Annotators:** Annotator-4 (Annotator-4), Annotator-2 (Annotator-2), Annotator-3 (Annotator-3)
**LLM Baseline:** GPT-OSS-120B (Scoring LLM)
**Design:** 30 items per annotator per task (20 core shared + 10 individual unique)

---

## Task 1: Framework Attribution (Human vs LLM)

**Task:** Distribute 100 points across 5 ethical frameworks per reasoning step. Compare human distributions against GPT-OSS-120B automated scores.

### Per-Annotator Correlation

| Annotator | N | Pearson r | p | Spearman ρ | MAE |
|-----------|---|-----------|---|------------|-----|
| Annotator-4 | 150 | 0.636 | 2.07e-18 | 0.669 | 6.5 |
| Annotator-2 | 145 | 0.301 | 2.33e-04 | 0.321 | 10.4 |
| Annotator-3 | 150 | 0.515 | 1.54e-11 | 0.538 | 13.0 |
| **Overall** | **445** | **0.468** | **1.48e-25** | **0.493** | **10.0** |
| Mean-Human (core) | 100 | 0.568 | 6.90e-10 | 0.614 | 7.9 |

Annotator-4 shows the strongest alignment with the LLM scorer; Annotator-2 the weakest. Averaging across annotators (mean-human on core items) improves the correlation to r=0.568.

### Per-Framework Correlation

| Framework | N | Pearson r | p | MAE | Human Mean | LLM Mean | Bias (H−L) |
|-----------|---|-----------|---|-----|-----------|----------|------------|
| Deontology | 89 | 0.320 | 0.0022 | 11.1 | 24.1 | 26.3 | −2.2 |
| Utilitarianism | 89 | 0.537 | <0.0001 | 8.8 | 24.2 | 22.2 | +2.0 |
| Virtue Ethics | 89 | 0.268 | 0.0112 | 12.5 | 27.4 | 24.3 | +3.1 |
| Contractualism | 89 | 0.187 | 0.0801 | 10.9 | 15.8 | 18.2 | −2.4 |
| Contractarianism | 89 | 0.071 | 0.5078 | 6.6 | 8.5 | 9.0 | −0.5 |

Utilitarianism has the highest human-LLM agreement; Contractarianism the lowest (but scores are very low and compressed near zero). The LLM slightly overweights Deontology and Contractualism; humans slightly overweight Virtue Ethics.

### Cosine Similarity (Item-Level, 5-Dim Vectors)

| Annotator | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Annotator-4 | 0.926 | 0.061 | 0.743 | 0.992 |
| Annotator-2 | 0.836 | 0.126 | 0.353 | 0.979 |
| Annotator-3 | 0.815 | 0.136 | 0.482 | 0.995 |
| **Overall** | **0.859** | **0.122** | | |

High cosine similarity (mean=0.859) indicates that even when point-wise scores differ, the overall framework distribution shapes are well-aligned between human and LLM annotations.

### Task 1 Summary

The GPT-OSS-120B framework attributions show **moderate-to-good alignment** with human judgments. The overall correlation (r=0.468) is statistically significant, and the cosine similarity (0.859) indicates the LLM captures the relative framework weighting pattern well. This validates the automated framework classification pipeline as a reasonable proxy for human annotation.

---

## Task 2: Faithfulness (Human vs LLM)

**Task:** Judge whether framework transitions across 4 reasoning steps are logically justified (binary) + confidence (0–100). Compare human justified/unjustified judgments against LLM Framework Drift Rate (FDR).

### Justified vs Unjustified Summary

| | Justified | Unjustified |
|---|-----------|-------------|
| Count | 85 | 5 |
| Rate | 94.4% | 5.6% |
| Mean FDR | 0.400 | 0.600 |

### Statistical Tests

| Metric | Value | p |
|--------|-------|---|
| Point-biserial r (justified vs FDR) | −0.125 | 0.2389 |
| Mann-Whitney U | 150 | 0.2402 |
| Confidence vs FDR (Pearson r) | 0.041 | 0.6997 |
| Confidence vs FDR (Spearman ρ) | 0.054 | 0.6140 |

### Justified Rate by FDR Level

| FDR Bin | N | Justified Rate | Mean Confidence |
|---------|---|----------------|-----------------|
| 0 (stable) | 36 | 97.2% | 83.2 |
| 0.33 | 7 | 100% | 88.7 |
| 0.67 | 37 | 91.9% | 83.8 |
| 1.0 (max drift) | 10 | 90.0% | 85.7 |

### Per-Annotator Breakdown

| Annotator | Justified | Unjustified | Mean Justified FDR | Mean Unjustified FDR |
|-----------|-----------|-------------|--------------------|-----------------------|
| Annotator-4 | 29 | 1 | 0.356 | 0.667 |
| Annotator-2 | 28 | 2 | 0.452 | 0.667 |
| Annotator-3 | 28 | 2 | 0.393 | 0.500 |

### Task 2 Summary

No statistically significant relationship between human faithfulness judgments and LLM FDR. The direction is as expected (unjustified items have higher FDR: 0.600 vs 0.400), but the sample of unjustified items is too small (n=5) for statistical power. Even at maximum FDR (1.0), 90% of transitions are rated as justified by human annotators.

**Interpretation:** The overwhelmingly "justified" ratings (94.4%) suggest that framework transitions in LLM moral reasoning are generally logically motivated, even when the degree of framework switching is high. Human annotators appear to evaluate transitions based on argumentative coherence rather than framework consistency per se.

---

## Task 3: Coherence Rating (Human vs LLM)

**Task:** Rate overall coherence of 4-step moral reasoning trajectories on a 0–100 scale. Compare against GPT-OSS-120B coherence scores and MRC (Moral Reasoning Consistency) metric.

### Descriptive Statistics

| Measure | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| Human coherence | 81.2 | 8.8 | 60 | 95 |
| LLM coherence | 63.0 | 25.0 | 30 | 92 |
| MRC score | 0.462 | 0.203 | 0.089 | 0.833 |

### Human vs LLM Coherence Correlation

| Annotator | N | Pearson r | p | Spearman ρ | MAE | Bias (H−L) |
|-----------|---|-----------|---|------------|-----|------------|
| Annotator-4 | 30 | 0.184 | 0.3303 | 0.243 | 25.0 | +19.7 |
| Annotator-2 | 30 | 0.025 | 0.8977 | 0.050 | 26.0 | +19.1 |
| Annotator-3 | 30 | −0.108 | 0.5706 | −0.168 | 27.4 | +16.0 |
| **Overall** | **90** | **0.012** | **0.9118** | **0.026** | **26.1** | **+18.3** |
| Mean-Human (core) | 20 | 0.085 | 0.7214 | 0.064 | 25.4 | +20.0 |

**There is essentially zero correlation between human and LLM coherence ratings.** Humans rate coherence 18.3 points higher on average.

### Human vs MRC Correlation

| Annotator | N | Pearson r | p | Spearman ρ |
|-----------|---|-----------|---|------------|
| Annotator-4 | 30 | 0.009 | 0.9623 | 0.074 |
| Annotator-2 | 30 | −0.138 | 0.4678 | −0.048 |
| Annotator-3 | 30 | −0.061 | 0.7490 | −0.061 |
| **Overall** | **90** | **−0.065** | **0.5421** | **−0.014** |
| Mean-Human (core) | 20 | −0.236 | 0.3170 | −0.205 |

**MRC does not correlate with human coherence ratings.**

### The Key Finding: Category Discrimination

| Category | N | Human Mean (±SD) | LLM Mean (±SD) | MRC Mean (±SD) |
|----------|---|------------------|-----------------|----------------|
| Single Framework | 27 | 81.2 (±9.1) | 83.4 (±19.3) | 0.70 (±0.05) |
| Bounce | 39 | 81.3 (±8.8) | 58.3 (±20.6) | 0.45 (±0.10) |
| High Entropy | 24 | 81.2 (±8.9) | 47.5 (±22.9) | 0.21 (±0.07) |

**ANOVA: Human Coherence ~ Trajectory Category**
- F = 0.001, p = 0.9990, η² = 0.0000

**ANOVA: LLM Coherence ~ Trajectory Category**
- F = 20.620, p < 0.0001, η² = 0.3216

The LLM and MRC sharply distinguish trajectory categories (LLM range: 47.5–83.4; MRC range: 0.21–0.70). **Humans do not**: all three categories receive essentially identical ratings (~81 ± 9). Human ratings cluster in the 60–95 range with zero discrimination between stable and unstable trajectories (η² = 0.0000).

### Task 3 Summary

Human coherence ratings do not correlate with either the LLM coherence scorer (r=0.012) or the MRC metric (r=−0.065). The fundamental disconnect is that humans rate all trajectory types as equally coherent (~81), while the LLM and MRC metrics are sensitive to framework-level instability. This suggests humans evaluate **surface-level argumentative coherence** (which remains high regardless of framework switches), while the LLM scorer evaluates **framework-specific consistency**.

---

## Cross-Task Summary Table

| Task | Metric | Value | Interpretation |
|------|--------|-------|----------------|
| 1 (Framework Attribution) | Pearson r | 0.468 | Moderate agreement |
| 1 | Cosine similarity | 0.859 | Good distributional alignment |
| 1 | Mean-Human r (core) | 0.568 | Moderate-good (averaged) |
| 2 (Faithfulness) | Justified rate | 94.4% | Near-universal agreement |
| 2 | Point-biserial r | −0.125 | Not significant |
| 3 (Coherence) | Human-LLM r | 0.012 | No agreement |
| 3 | Human-MRC r | −0.065 | No agreement |
| 3 | η² (human, category) | 0.0000 | No category discrimination |
| 3 | η² (LLM, category) | 0.3216 | Strong category discrimination |

---

## Implications for the Manuscript

### Validated Claims

1. **Framework attribution pipeline is validated.** GPT-OSS-120B's 5-way framework distributions moderately correlate with human judgments (r=0.468, cosine=0.859). This supports using automated framework classification at scale.

2. **Framework transitions are logically motivated.** Human annotators rate 94.4% of transitions as justified, even at high FDR levels (90% justified at FDR=1.0). This supports the paper's characterization of LLM moral reasoning as following coherent (if shifting) ethical logic.

### Claims Requiring Revision

3. **MRC "human validation" must be reframed.** The paper's reported r=0.715 correlation is between MRC and *LLM* coherence scores, **not** human ratings. The actual human-MRC correlation is r=−0.065 (n.s.), and human-LLM coherence correlation is r=0.012 (n.s.). The manuscript should:
   - Clarify that the r=0.715 validates MRC against the LLM scorer, not human judgment
   - Report the human annotation results as showing that humans evaluate a different dimension of coherence (surface argumentative quality vs. framework consistency)
   - Frame MRC as capturing framework-level stability that is not spontaneously salient to human readers without explicit framework labels

### Possible Explanations for the Task 3 Disconnect

1. **Surface coherence vs. framework coherence:** LLM-generated moral reasoning maintains high surface-level argumentative quality regardless of how many framework switches occur. Humans respond to this surface quality.
2. **Framework switches are invisible to naive readers:** Without explicit framework labels, annotators cannot easily detect when the underlying ethical framework changes between steps.
3. **Ceiling effect:** Human ratings cluster at 75–90 with limited variance, reducing statistical power to detect correlations.
4. **Calibration failure:** Despite guidelines anchoring scores to framework switch counts, annotators did not follow these anchors.

---

## Generated Figures

| File | Description |
|------|-------------|
| `hlm_task1_scatter.png` | Human vs LLM framework scores (overall + per-annotator scatter) |
| `hlm_task1_bias.png` | Per-framework bias bars + cosine similarity distribution |
| `hlm_task2_analysis.png` | FDR by judgment, justified rate by FDR bin, confidence vs FDR |
| `hlm_task3_analysis.png` | Human vs LLM coherence (per-annotator), category comparison, MRC validation, score distributions |
