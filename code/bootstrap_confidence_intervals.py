"""
Compute bootstrapped 95% confidence intervals for ethical framework distributions.

Loads LLM-classified framework labels from a JSONL file, then computes
non-parametric bootstrap CIs (percentile method, 10,000 resamples by default)
for overall and per-model framework proportions.

Framework label mapping (classification -> display name):
  CARE_ETHICS       -> Contractualism
  DEONTOLOGY        -> Deontology
  CONSEQUENTIALISM  -> Act Utilitarianism
  SOCIAL_CONTRACT   -> Contractarianism
  VIRTUE_ETHICS     -> Virtue Ethics

Inputs:
  results/framework_classification/llm_classification_raw.jsonl

Outputs:
  results/framework_classification/bootstrap_ci_summary.json

Computes non-parametric bootstrap CIs (percentile method) for overall
and per-model framework proportions.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

FRAMEWORKS = [
    "CARE_ETHICS",
    "DEONTOLOGY",
    "CONSEQUENTIALISM",
    "SOCIAL_CONTRACT",
    "VIRTUE_ETHICS",
]

FRAMEWORK_DISPLAY_NAMES = {
    "CARE_ETHICS": "Contractualism",
    "DEONTOLOGY": "Deontology",
    "CONSEQUENTIALISM": "Act Utilitarianism",
    "SOCIAL_CONTRACT": "Contractarianism",
    "VIRTUE_ETHICS": "Virtue Ethics",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_classification_data(filepath: Path) -> pd.DataFrame:
    """Load LLM classification JSONL into a DataFrame."""
    records: List[Dict] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["framework"] = df["final_classification"].map(FRAMEWORK_DISPLAY_NAMES)
    return df


# =============================================================================
# Bootstrap Computation
# =============================================================================


def bootstrap_proportions(
    labels: np.ndarray,
    frameworks: List[str],
    n_boot: int = 10_000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute bootstrapped 95% CIs (percentile method) for each framework.

    Returns a dict keyed by framework label with obs, ci_lo, ci_hi, and
    the full bootstrap distribution array.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    results: Dict[str, Dict[str, Any]] = {}

    for fw in frameworks:
        boot_props = np.zeros(n_boot)
        for b in range(n_boot):
            sample = rng.choice(labels, size=n, replace=True)
            boot_props[b] = np.mean(sample == fw)

        obs = np.mean(labels == fw) * 100
        ci_lo = float(np.percentile(boot_props, 2.5) * 100)
        ci_hi = float(np.percentile(boot_props, 97.5) * 100)
        results[fw] = {
            "obs": float(obs),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "dist": boot_props,
        }

    return results


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute bootstrapped 95% confidence intervals for ethical "
            "framework distributions (overall and per-model)."
        )
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="results/framework_classification/llm_classification_raw.jsonl",
        help="Path to classification JSONL file.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/framework_classification",
        help="Directory for bootstrap summary JSON.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    data_file = Path(args.data_file)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print(f"Loading data from: {data_file}")
    df = load_classification_data(data_file)
    print(f"Total samples: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    labels = df["final_classification"].values
    n = len(labels)

    # ---- Overall bootstrap ----
    print(f"\n=== Overall Bootstrap (n={n}, B={args.n_boot:,}) ===")
    print(f"{'Framework':<20} {'Observed':>10} {'95% CI':>18} {'CI Width':>10}")
    print("-" * 62)

    boot_results = bootstrap_proportions(
        labels, FRAMEWORKS, n_boot=args.n_boot, seed=args.seed
    )

    for fw in FRAMEWORKS:
        r = boot_results[fw]
        name = FRAMEWORK_DISPLAY_NAMES[fw]
        print(
            f"{name:<20} {r['obs']:>9.1f}% "
            f"[{r['ci_lo']:>6.1f}--{r['ci_hi']:.1f}%] "
            f"{r['ci_hi'] - r['ci_lo']:>9.1f}pp"
        )

    # ---- Per-model bootstrap ----
    models = sorted(df["model"].unique())
    per_model: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model in models:
        model_labels = df[df["model"] == model]["final_classification"].values
        n_m = len(model_labels)
        print(f"\n=== {model} (n={n_m}) ===")
        print(f"{'Framework':<20} {'Observed':>10} {'95% CI':>18}")
        print("-" * 52)

        per_model[model] = {}
        for fw in FRAMEWORKS:
            boot_props = np.zeros(args.n_boot)
            rng = np.random.RandomState(args.seed)
            for b in range(args.n_boot):
                sample = rng.choice(model_labels, size=n_m, replace=True)
                boot_props[b] = np.mean(sample == fw)

            obs = float(np.mean(model_labels == fw) * 100)
            ci_lo = float(np.percentile(boot_props, 2.5) * 100)
            ci_hi = float(np.percentile(boot_props, 97.5) * 100)
            per_model[model][fw] = {"obs": obs, "ci_lo": ci_lo, "ci_hi": ci_hi}

            name = FRAMEWORK_DISPLAY_NAMES[fw]
            print(f"{name:<20} {obs:>9.1f}% [{ci_lo:>6.1f}--{ci_hi:.1f}%]")

    # ---- Save summary JSON ----
    summary: Dict[str, Any] = {
        "description": "Bootstrapped 95% CIs for framework distributions",
        "n_samples": n,
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
        "overall": {
            fw: {
                "display_name": FRAMEWORK_DISPLAY_NAMES[fw],
                "observed_pct": boot_results[fw]["obs"],
                "ci_lo_pct": boot_results[fw]["ci_lo"],
                "ci_hi_pct": boot_results[fw]["ci_hi"],
            }
            for fw in FRAMEWORKS
        },
        "per_model": {
            model: {
                fw: {
                    "display_name": FRAMEWORK_DISPLAY_NAMES[fw],
                    "observed_pct": per_model[model][fw]["obs"],
                    "ci_lo_pct": per_model[model][fw]["ci_lo"],
                    "ci_hi_pct": per_model[model][fw]["ci_hi"],
                }
                for fw in FRAMEWORKS
            }
            for model in models
        },
    }
    json_path = results_dir / "bootstrap_ci_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
