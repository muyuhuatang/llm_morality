# Human Annotation Analysis: Summary

**Date:** 2026-03-01
**Annotators:** Annotator-4 (Annotator-4), Annotator-2 (Annotator-2), Annotator-3 (Annotator-3)
**Design:** 30 items per annotator per task (20 core shared + 10 individual unique)
**LLM Baseline:** GPT-OSS-120B (Scoring LLM)

---

## Task 1: Step-level Framework Attribution

**Task:** Distribute 100 points across 5 ethical frameworks for each reasoning step.

### Inter-Annotator Agreement (20 core items x 5 frameworks = 95-100 comparisons)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ICC(3,1) | 0.504 | Moderate |
| ICC(3,k) | 0.753 | Good (averaged across 3 raters) |
| Krippendorff α (interval) | 0.505 | Moderate |

**Pairwise correlations:**

| Pair | Pearson r | p | MAE |
|------|-----------|---|-----|
| Annotator-4 vs Annotator-2 | 0.531 | <0.0001 | 7.8 |
| Annotator-4 vs Annotator-3 | 0.846 | <0.0001 | 8.5 |
| Annotator-2 vs Annotator-3 | 0.302 | 0.003 | 14.3 |

Annotator-4 and Annotator-3 show strong agreement (r=0.846); Annotator-2 diverges more.

**Per-framework agreement (Krippendorff α):**

| Framework | α | Mean Pairwise r | Mean Score |
|-----------|---|-----------------|------------|
| Deontology | 0.225 | 0.292 | 25.9 |
| Utilitarianism | 0.458 | 0.570 | 25.0 |
| Virtue Ethics | 0.375 | 0.443 | 25.9 |
| Contractualism | 0.299 | 0.526 | 16.0 |
| Contractarianism | 0.057 | 0.334 | 7.1 |

Utilitarianism has highest agreement; Contractarianism has lowest (near-zero α, but scores are very low and compressed).

### Human-LLM Agreement

| Annotator | N | Pearson r | p | MAE |
|-----------|---|-----------|---|-----|
| Annotator-4 | 150 | 0.636 | 2.1e-18 | 6.5 |
| Annotator-2 | 145 | 0.301 | 2.3e-04 | 10.4 |
| Annotator-3 | 150 | 0.515 | 1.5e-11 | 13.0 |
| **Overall** | **445** | **0.468** | **1.5e-25** | **10.0** |

**Cosine similarity (item-level, 5-dim vectors):**

| Annotator | Mean | Std | Min |
|-----------|------|-----|-----|
| Annotator-4 | 0.926 | 0.061 | 0.743 |
| Annotator-2 | 0.836 | 0.126 | 0.353 |
| Annotator-3 | 0.815 | 0.136 | 0.482 |
| **Overall** | **0.859** | **0.122** | |

**Mean framework distribution comparison:**

| Framework | Human Mean | LLM Mean | Diff |
|-----------|-----------|----------|------|
| Deontology | 24.1 | 26.3 | -2.2 |
| Utilitarianism | 24.2 | 22.2 | +2.0 |
| Virtue Ethics | 27.4 | 24.3 | +3.1 |
| Contractualism | 15.8 | 18.2 | -2.4 |
| Contractarianism | 8.5 | 9.0 | -0.5 |

Distributions are broadly aligned. The LLM tends to slightly overweight Contractualism and Deontology; humans slightly overweight Virtue Ethics.

### Data Quality Notes
- Annotator-2 had 6 items with sums not equal to 100 (range: 70-110); these were normalized to 100.
- 1 item (core_019) had null scores from Annotator-2, excluded from inter-annotator analysis.
- Annotator-3 had 1 item (core_002) summing to 95; normalized.

---

## Task 2: Trajectory-level Faithfulness

**Task:** Judge whether framework transitions across 4 reasoning steps are logically justified (binary) + confidence (0-100).

### Inter-Annotator Agreement (20 core items)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Krippendorff α (nominal) | -0.073 | No agreement beyond chance |
| Unanimous agreement | 15/20 (75%) | High raw agreement |
| Cohen's κ (all pairs) | -0.07 to -0.11 | Negative (but meaningless) |

**Why κ is negative:** Nearly all judgments are "justified" (85/90 = 94.4%), creating extreme class imbalance. With only 5 total "unjustified" judgments across all annotators, chance-corrected metrics are undefined/negative.

| Annotator | Justified | Unjustified |
|-----------|-----------|-------------|
| Annotator-4 | 29 | 1 |
| Annotator-2 | 28 | 2 |
| Annotator-3 | 28 | 2 |

**Confidence scores:**

| Metric | Value |
|--------|-------|
| ICC(3,1) | 0.121 |
| Krippendorff α (interval) | 0.130 |
| Mean confidence | 84.1 |

Confidence scores show low inter-annotator agreement, suggesting annotators apply different confidence calibrations even when agreeing on the binary judgment.

### Human-LLM Agreement

| Metric | Value | p |
|--------|-------|---|
| Point-biserial r (justified vs FDR) | -0.125 | 0.239 |
| Confidence vs FDR (Pearson) | 0.041 | 0.700 |

No significant relationship. The direction is as expected (unjustified items have higher FDR: 0.600 vs 0.400) but the sample of unjustified items is too small (n=5) for statistical power.

### Interpretation
The overwhelmingly "justified" ratings suggest either: (1) the sampled trajectories genuinely contain well-motivated framework shifts, (2) the task is too lenient (annotators default to "justified" when uncertain), or (3) human annotators have difficulty detecting unjustified framework transitions without explicit framework labels.

---

## Task 3: MRC Validation / Coherence Rating

**Task:** Rate overall coherence of 4-step moral reasoning trajectories on a 0-100 scale.

### Inter-Annotator Agreement (20 core items)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ICC(3,1) | 0.395 | Fair |
| ICC(3,k) | 0.662 | Moderate-Good (averaged) |
| Krippendorff α (interval) | 0.404 | Fair |

**Pairwise correlations:**

| Pair | Pearson r | p | MAE |
|------|-----------|---|-----|
| Annotator-4 vs Annotator-2 | 0.460 | 0.041 | 5.8 |
| Annotator-4 vs Annotator-3 | 0.290 | 0.214 | 8.5 |
| Annotator-2 vs Annotator-3 | 0.463 | 0.040 | 5.8 |

### Human-LLM Coherence Agreement

| Metric | Value | p |
|--------|-------|---|
| Overall Pearson r | 0.012 | 0.912 |
| Overall MAE | 26.1 points | |
| Mean Human vs LLM (core, averaged) | 0.085 | 0.721 |

**There is essentially zero correlation between human and LLM coherence ratings.**

### MRC Validation

| Metric | Value | p |
|--------|-------|---|
| Human vs MRC (all, pooled) | r = -0.065 | 0.542 |
| Mean Human vs MRC (core) | r = -0.236 | 0.317 |

**MRC does not correlate with human coherence ratings.**

### The Core Problem: Range Restriction

**Coherence ratings by trajectory category:**

| Category | Human Mean (±SD) | LLM Mean (±SD) | MRC Mean (±SD) |
|----------|-----------------|-----------------|----------------|
| Single Framework | 81.2 (±9.1) | 83.4 (±19.3) | 0.702 (±0.046) |
| Bounce | 81.3 (±8.8) | 58.3 (±20.6) | 0.450 (±0.104) |
| High Entropy | 81.2 (±8.9) | 47.5 (±22.9) | 0.210 (±0.073) |

The LLM and MRC sharply distinguish trajectory categories (LLM range: 47.5-83.4; MRC range: 0.21-0.70). **Humans do not**: all three categories receive essentially identical ratings (~81 ± 9). Human ratings cluster in the 60-95 range with almost no discrimination between stable and unstable trajectories.

---

## Implications for the Paper

### What the results support

1. **Task 1 (Framework Attribution):** GPT-OSS-120B framework attributions are moderately aligned with human judgments (r=0.468, cosine=0.859). The overall framework distributions match well. This validates the automated framework classification pipeline.

2. **Task 2 (Faithfulness):** Human annotators overwhelmingly rate framework transitions as justified (94.4%). This is consistent with the paper's finding that LLM moral reasoning, even when switching frameworks, tends to follow logically motivated transitions rather than random jumps.

### What the results challenge

3. **Task 3 (Coherence / MRC Validation):** The paper's claim that MRC correlates with "human-validated LLM coherence ratings" (r=0.715, p<0.0001) needs careful reframing:
   - The r=0.715 is between MRC and *LLM* coherence ratings (GPT-OSS-120B scores), **not** human ratings.
   - Human-LLM coherence correlation is near zero (r=0.012).
   - Human-MRC correlation is near zero (r=-0.065).
   - Humans do not distinguish between trajectory categories that the LLM and MRC clearly separate.

### Possible explanations for Task 3 disconnect

1. **Ceiling effect:** Humans rate most reasoning as "coherent" (mean ~81), lacking sensitivity to framework-level instability that is visible to the LLM scorer with explicit framework taxonomy training.
2. **Surface coherence vs. framework coherence:** Humans may evaluate natural language fluency and logical flow, which remains high even when underlying ethical frameworks shift. The LLM scorer specifically evaluates framework consistency, a more technical criterion.
3. **Calibration mismatch:** Despite calibration guidelines (0 switches → 85-100, 3+ switches → 20-50), human raters did not follow these anchors, suggesting framework switches are not salient to human readers without explicit framework labels.
4. **Sample size:** 20 core items with 3 annotators may be underpowered to detect moderate correlations (the 95% CI for r=0.085 spans roughly -0.37 to 0.50).

### Recommended revisions to the manuscript

1. **Reframe "human-validated":** Instead of claiming human validation of coherence ratings, report the actual finding: humans show moderate inter-annotator agreement on coherence (ICC=0.395) but do not discriminate between trajectory categories. LLM coherence ratings capture framework-level distinctions that humans do not spontaneously detect.

2. **Revise the abstract/intro claim:** The current text says coherence ratings are "confirmed high quality by human annotators." This should be revised to accurately reflect that human annotators agree moderately with each other but not with the LLM scorer, suggesting LLM coherence captures a distinct (framework-specific) dimension of coherence.

3. **Strengthen Task 1 reporting:** The framework attribution results (r=0.468, cosine=0.859) are a genuine validation success and should be prominently reported.

4. **Report Task 2 finding positively:** The near-universal "justified" rating suggests that framework transitions in LLM moral reasoning are logically motivated, which is itself an interesting finding.

---

## Generated Files

| File | Description |
|------|-------------|
| `analysis_task1_framework_attribution.ipynb` | Full Task 1 analysis with ICC, correlations, heatmaps |
| `analysis_task2_faithfulness.ipynb` | Full Task 2 analysis with kappa, ICC, FDR comparison |
| `analysis_task3_coherence.ipynb` | Full Task 3 analysis with ICC, MRC validation, category comparison |
| `task1_human_vs_llm_scatter.png` | Human vs LLM framework scores by annotator |
| `task1_heatmap_comparison.png` | Side-by-side heatmap of human mean vs LLM scores |
| `task1_inter_annotator_scatter.png` | Pairwise annotator agreement scatter plots |
| `task2_faithfulness_analysis.png` | FDR by justified/unjustified + confidence distribution |
| `task2_agreement_heatmap.png` | Per-item justified judgment heatmap |
| `task3_human_vs_llm_coherence.png` | Human vs LLM coherence scatter by annotator |
| `task3_mrc_validation.png` | Human coherence vs MRC score (key validation plot) |
| `task3_inter_annotator_scatter.png` | Pairwise annotator coherence agreement |
| `task3_category_comparison.png` | Coherence metrics by trajectory category (human vs LLM vs MRC) |
