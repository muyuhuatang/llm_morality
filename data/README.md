# Data Archive

Model responses, analysis results, and human annotations organized into three subfolders.

## Directory Structure

```
data/
├── raw_results/       # Sampled scenarios, model responses, attribution scores (48 files)
├── analysis/          # Computed metrics, statistical results, summaries (58 files)
└── annotation/        # Human annotation data and inter-annotator analysis (23 files)
```

## analysis/

Computed metrics, statistical test results, and experiment summaries.

### Trajectory Analysis
| File | Description |
|------|-------------|
| `trajectory_features.csv` | Per-trajectory metrics: FDR, entropy, faithfulness, archetype |
| `trajectory_metrics.csv` | Trajectory-level metrics for probing models |
| `trajectory_summary.json` | Aggregate trajectory statistics |
| `faithfulness_evaluations.csv` | LLM-evaluated transition faithfulness scores |
| `archetype_distribution.csv` | Trajectory archetype counts |
| `archetype_characteristics.csv` | Per-archetype metric summaries |
| `transition_matrix_global_{model}.csv` | Global framework transition matrices |
| `transition_matrix_step23_{model}.csv` | Step 2→3 transition matrices |
| `single_framework_composition.csv` | Composition of single-framework trajectories |
| `single_framework_rate_by_dataset.csv` | Single-framework rates per dataset |

### Step-Level Analysis
| File | Description |
|------|-------------|
| `all_reasoning_steps.csv` | All reasoning steps across models |
| `all_model_4step_descriptions.csv` | Step descriptions for 4-step trajectories |
| `all_model_4step_clusters.csv` | Cluster assignments for step descriptions |
| `step_counts_by_dataset_model.csv` | Step count statistics |
| `accuracy_by_endturn.csv` | Accuracy by final reasoning step |
| `morality_accuracy_correlation.csv` | Morality score vs accuracy correlation |
| `attribution_by_model.csv` | Per-model attribution summary |

### Probing Results
| File | Description |
|------|-------------|
| `probing_dataset.parquet` | Step-level probing data with 5D framework distributions |
| `probing_splits.json` | Train/valid/test split indices |
| `probe_results.json` | Layer-wise probe performance (KL, top-1, correlation) |
| `probing_metrics_table.csv` | Probe metrics in tabular format |
| `category_analysis.json` | Probe performance by trajectory category |
| `cross_model_transfer.json` | Cross-model transfer results |

### Steering and Persuasion
| File | Description |
|------|-------------|
| `stable_unstable_splits.json` | Stable/unstable trajectory definitions |
| `steering_evaluation.json` | Alpha sweep results (50 values) |
| `steering_evaluation_1k.json` | Extended alpha sweep (1,000 values) |
| `layer_specificity.json` | Layer specificity analysis |
| `persuasion_prompts.json` | Persuasive attack prompt templates |
| `persuasion_results.json` | Baseline persuasion flip rates |
| `steering_persuasion_{model}.json` | Per-model steering + persuasion results |
| `robustness_analysis.json` | Statistical tests (chi-square, z-test, Cohen's h) |

### MRC Validation
| File | Description |
|------|-------------|
| `mrc_scores.csv` | Per-trajectory MRC scores |
| `mrc_summary.json` | MRC statistics by category |
| `llm_annotations.json` | GPT-OSS-120B coherence ratings (180 trajectories) |
| `llm_ratings.csv` | LLM ratings in tabular format |
| `mrc_validation.json` | MRC-coherence correlation and component analysis |
| `mrc_persuasion_analysis.json` | MRC vs persuasion resistance |

### Model Summaries
| File | Description |
|------|-------------|
| `comprehensive_summary.json` | Overall experiment summary |
| `accuracy_comparison_summary.json` | Model accuracy comparison |
| `summary_{model}.json` | Per-model response summaries |

## annotation/

Human annotation data for validation study (3 annotators x 3 tasks).

| Pattern | Description |
|---------|-------------|
| `Anno-{1,2,3}_task-{1,2,3}.json` | Raw annotation data |
| `Anno-{1,2,3}_task-{1,2,3}-annotation.md` | Human-readable annotation forms |
| `Anno-{1,2,3}_README.md` | Per-annotator instructions |
| `human_annotation_summary.md` | Results summary |
| `human_annotation_analysis.md` | Inter-annotator agreement analysis |

Tasks: (1) step-level framework attribution, (2) trajectory-level faithfulness, (3) coherence rating.

## Pre-Release Filtering

Data files were filtered to include only those directly supporting claims in the paper:
- Removed intermediate clustering files, redundant pivot tables, and auto-generated LaTeX fragments.
- Removed files from abandoned experiments (non-Turbo Qwen duplicates, o4-mini 1200-sample run, 100%-error Qwen Turbo instructed-framework files).
- Sanitized API error messages that revealed provider identity from 3 Llama response/attribution files.
- Stripped execution timestamps from summary JSON files.
- Annotator names anonymized to Annotator-1/2/3. No API keys, file paths, or personally identifiable information remains.

## Excluded Files (Too Large)

Regenerable using the code archive:
- `activations_llama.h5` (~2.4 GB) — Llama-3.3-70B hidden state activations
- `activations_qwen.h5` (~2.4 GB) — Qwen2.5-72B hidden state activations
- `*.pkl` — Embedding pickle files
- `probe_weights/` (~26 MB) — Trained linear probe parameters

## Source Datasets

- [Moral Stories](https://github.com/demelin/moral_stories) (Emelin et al., 2021)
- [ETHICS](https://github.com/hendrycks/ethics) (Hendrycks et al., 2021)
- [Social Chemistry 101](https://github.com/mbforbes/social-chemistry-101) (Forbes et al., 2020)
