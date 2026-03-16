# Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability

This repository contains the code and data for reproducing the experiments in our paper.

- **`code/`** — 17 Python scripts covering the full pipeline.
- **`data/`** — 129 data files organized into 2 subfolders: `analysis/` (58), `annotation/` (23).

## Overview

We introduce *moral reasoning trajectories* — sequences of ethical framework invocations across intermediate reasoning steps — and analyze their dynamics across six models (GPT-5, GPT-5-mini, GPT-4o, GPT-4o-mini, o3-mini, o4-mini) and three benchmarks (Moral Stories, ETHICS, Social Chemistry 101). The paper investigates:

1. How LLMs organize multi-framework moral reasoning within structured deliberation
2. Whether framework-specific patterns are grounded in identifiable internal representations (linear probing)
3. Whether those representations can be leveraged to modulate integration patterns (activation steering)

## Ethical Frameworks

The five canonical ethical frameworks analyzed:
1. Kantian Deontology
2. Benthamite Act Utilitarianism
3. Aristotelian Virtue Ethics
4. Scanlonian Contractualism
5. Gauthierian Contractarianism

## Requirements

See `requirements.txt`. Key dependencies:
- Python 3.10+
- PyTorch, Transformers (HuggingFace), bitsandbytes
- OpenAI and Together.ai API access
- numpy, pandas, scipy, scikit-learn, statsmodels, tqdm, h5py

## Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
export HF_TOKEN="your-token"
```

## Pre-Release Filtering

Before public release, the following filtering was applied:

- **Code**: Converted from ~70 Jupyter notebooks to 17 clean Python scripts retaining only core execution logic used in the paper. Removed visualization code, unused prompt templates, pilot-only utilities, and redundant scripts.
- **Data**: Removed intermediate clustering files, redundant analysis outputs, auto-generated LaTeX fragments, and files from abandoned experiments (e.g., non-Turbo Qwen duplicates, o4-mini 1200-sample run). Removed records containing API error messages that revealed provider identity. Stripped timestamps from summary files.
- **Sensitive data**: All API keys replaced with environment variable references. Annotator names anonymized. Absolute file paths removed. OpenAI batch IDs redacted. Two rounds of automated scanning confirmed zero remaining sensitive information.

## Notes

- The response collection scripts (`collect_responses_batch.py`, `collect_responses_parallel.py`) use the pilot study prompt described in the Appendix, not the refined theory-neutral prompt from the main experiments. The main experiment prompt is documented in the paper (Section 3.2 and Appendix).
- Raw activation files (`.h5`, ~5 GB) are excluded due to size but can be regenerated using `code/extract_activations.py`.
- Embedding pickle files (`.pkl`) and trained probe weights are excluded for the same reason.
- API keys are not included; users must supply their own via environment variables.
