# Code (Software Archive)

Python scripts and prompt templates for reproducing all experiments.

## Scripts

### Data Preparation
| Script | Description |
|--------|-------------|
| `generate_pilot_samples.py` | Sample scenarios from source datasets |

### Response Collection
| Script | Description |
|--------|-------------|
| `collect_responses_batch.py` | Collect structured moral reasoning via OpenAI Batch API (gpt-4o, gpt-4o-mini). **Note:** uses the pilot study prompt (Appendix), not the refined theory-neutral prompt from the main experiments. |
| `collect_responses_parallel.py` | Async parallel collection from OpenAI (gpt-5, o3/o4-mini) and Together.ai (Llama-3.3-70B, Qwen2.5-72B). **Note:** uses the pilot study prompt (Appendix), not the refined theory-neutral prompt from the main experiments. |

### Framework Attribution
| Script | Description |
|--------|-------------|
| `score_attributions.py` | Score each reasoning step's 5-framework distribution via GPT-OSS-120B |
| `analyze_gptoss_robustness.py` | Scorer robustness analysis |
| `robustness_check.py` | Framework instructability test (can models use all 5 frameworks when instructed?) |

### Trajectory Analysis
| Script | Description |
|--------|-------------|
| `compute_trajectory_metrics.py` | Compute FDR, entropy, faithfulness, and classify trajectory archetypes |
| `bootstrap_confidence_intervals.py` | 10K-resample bootstrap CIs for framework distributions |

### Probing (requires GPU)
| Script | Description |
|--------|-------------|
| `extract_activations.py` | Extract hidden states from Llama-3.3-70B or Qwen2.5-72B at all layers |
| `train_probes.py` | Train linear probes (softmax(Wx+b)), cross-model transfer, permutation tests |

### Activation Steering (requires GPU)
| Script | Description |
|--------|-------------|
| `construct_steering_vectors.py` | Compute steering vectors from stable vs unstable trajectory activations |
| `evaluate_steering.py` | Simulated steering with 1000 alpha values, layer specificity analysis |
| `persuasion_attacks.py` | Real activation steering via forward hooks + 3 persuasive attack types |
| `robustness_analysis.py` | Statistical tests on stability-susceptibility relationship |

### MRC Validation
| Script | Description |
|--------|-------------|
| `compute_mrc.py` | Compute Moral Representation Consistency metric for all trajectories |
| `collect_llm_ratings.py` | Collect LLM coherence ratings (GPT-OSS-120B, 3 ratings, median) |
| `validate_mrc.py` | MRC-coherence correlation, component analysis, persuasion link |

## Pre-Release Filtering

These scripts were distilled from ~70 Jupyter notebooks into 17 clean Python files. Only scripts whose outputs directly support tables, figures, or statistics in the paper are included. Visualization code (matplotlib/seaborn), unused prompt templates, pilot-only utilities, and redundant scripts were removed. All API keys use `os.environ` references; no hardcoded credentials remain.

## Requirements

```
python >= 3.10
torch >= 2.0
transformers >= 4.40
h5py
openai
together
pandas
numpy
scipy
scikit-learn
statsmodels
tqdm
bitsandbytes          # for 4-bit quantization in steering
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
export HF_TOKEN="your-token"     # for downloading gated models
```

## Reproduction Pipeline

```bash
# 1. Data preparation (pre-computed data provided in data/ archive)
python generate_pilot_samples.py

# 2. Response collection
python collect_responses_batch.py submit --models gpt-4o gpt-4o-mini
python collect_responses_parallel.py collect \
    --openai-models gpt-5 gpt-5-mini o3-mini o4-mini \
    --together-models meta-llama/Llama-3.3-70B-Instruct-Turbo Qwen/Qwen2.5-72B-Instruct-Turbo

# 3. Framework attribution
python score_attributions.py --data-dir data/step_attribution/

# 4. Trajectory analysis
python compute_trajectory_metrics.py
python bootstrap_confidence_intervals.py

# 5. Probing (GPU required; probing dataset provided in data/analysis/)
python extract_activations.py --model llama --device cuda:0
python extract_activations.py --model qwen --device cuda:1
python train_probes.py

# 6. Activation steering (GPU required)
python construct_steering_vectors.py
python evaluate_steering.py
python persuasion_attacks.py --model llama --alphas 0.0 2.0 4.0 6.0 8.0 10.0
python persuasion_attacks.py --model qwen --alphas 0.0 2.0 3.0 4.0 5.0 6.0
python robustness_analysis.py

# 7. MRC validation
python compute_mrc.py
python collect_llm_ratings.py
python validate_mrc.py

# 8. Human annotation (annotation data provided in data/annotation/)
```
