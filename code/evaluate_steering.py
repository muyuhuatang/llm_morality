"""
Evaluate steering effectiveness with high-resolution alpha sweep.

Simulates activation-level steering on saved activations using 1000
log-spaced alpha values (0.01 -- 10.0). For each alpha, modifies
activations at the optimal probe layer and measures FDR reduction via
the trained linear probe.

Inputs:
    - data/analysis/steering_vectors_{model}.pt
    - data/analysis/stable_unstable_splits.json
    - data/analysis/activations_{model}.h5
    - data/analysis/probe_weights/{model}/layer_{l}.pt

Outputs:
    - data/analysis/steering_evaluation.json
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPTIMAL_LAYERS = {"llama": 63, "qwen": 17}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_steering_vectors(path: Path):
    """Load steering vectors from a .pt file."""
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def load_probe_weights(analysis_dir: Path, model_name: str, layer: int):
    """Load probe weights from either results or results_pilot100."""
    for results_dir_name in ["results", "results_pilot100"]:
        probe_path = analysis_dir / results_dir_name / "probe_weights" / model_name / f"layer_{layer}.pt"
        if probe_path.exists():
            return torch.load(probe_path, weights_only=False)
    return None


def compute_framework_consistency(probs_sequence: list[torch.Tensor]) -> dict:
    """Compute FDR, entropy, and consistency from a probability sequence."""
    if len(probs_sequence) < 2:
        return {"fdr": 0.0, "entropy": 0.0, "consistency": 1.0}
    dominant = [p.argmax().item() for p in probs_sequence]
    n_transitions = sum(1 for i in range(1, len(dominant)) if dominant[i] != dominant[i - 1])
    fdr = n_transitions / (len(dominant) - 1)
    entropies = [-(p * torch.log(p + 1e-10)).sum().item() for p in probs_sequence]
    max_probs = [p.max().item() for p in probs_sequence]
    return {"fdr": fdr, "entropy": float(np.mean(entropies)), "consistency": float(np.mean(max_probs))}


def evaluate_steering(
    model_name: str,
    steering_data: dict,
    splits_data: dict,
    h5_path: Path,
    alpha_values: list[float],
    analysis_dir: Path,
    n_test_samples: int = 50,
) -> dict | None:
    """Evaluate steering for all alpha values on unstable test samples."""
    if steering_data is None:
        return None

    optimal_layer = steering_data["metadata"]["optimal_layer"]
    probe_weights = load_probe_weights(analysis_dir, model_name, optimal_layer)
    if probe_weights is None:
        print(f"Probe weights not found for {model_name}")
        return None

    # Extract probe parameters
    if "linear.weight" in probe_weights:
        W = probe_weights["linear.weight"].cpu()
        b = probe_weights["linear.bias"].cpu()
    else:
        W = probe_weights["weight"].cpu()
        b = probe_weights["bias"].cpu()

    results: dict = {
        "model": model_name,
        "optimal_layer": optimal_layer,
        "alpha_results": {},
    }
    unstable_samples = splits_data["unstable_samples"]

    # Pre-load activations
    with h5py.File(h5_path, "r") as f:
        all_activations = f["activations"][:]
        all_sample_ids = [s.decode() if isinstance(s, bytes) else s for s in f["sample_ids"][:]]
        all_step_ids = f["step_ids"][:]

        sample_to_indices: dict[str, list] = defaultdict(list)
        for idx, (sid, step) in enumerate(zip(all_sample_ids, all_step_ids)):
            sample_to_indices[sid].append((idx, step))

    print(f"{model_name}: Evaluating {len(alpha_values)} alpha values...")

    for alpha in tqdm(alpha_values, desc=model_name):
        alpha_metrics: dict[str, list] = defaultdict(list)

        for sample_id in unstable_samples[:n_test_samples]:
            if sample_id not in sample_to_indices:
                continue

            indices_steps = sorted(sample_to_indices[sample_id], key=lambda x: x[1])[:4]
            if len(indices_steps) < 2:
                continue

            for fw_name, fw_data in steering_data["vectors"].items():
                steering_vector = fw_data["steering_vector"].cpu()
                probs_baseline, probs_steered = [], []

                for idx, step in indices_steps:
                    act = torch.tensor(all_activations[idx, optimal_layer, :], dtype=torch.float32)

                    logits_base = act @ W.T + b
                    probs_baseline.append(torch.softmax(logits_base, dim=-1))

                    act_steered = act + alpha * steering_vector
                    logits_steered = act_steered @ W.T + b
                    probs_steered.append(torch.softmax(logits_steered, dim=-1))

                if len(probs_baseline) >= 2:
                    m_base = compute_framework_consistency(probs_baseline)
                    m_steered = compute_framework_consistency(probs_steered)
                    alpha_metrics[fw_name].append(
                        {
                            "baseline_fdr": m_base["fdr"],
                            "steered_fdr": m_steered["fdr"],
                            "fdr_reduction": m_base["fdr"] - m_steered["fdr"],
                            "consistency_gain": m_steered["consistency"] - m_base["consistency"],
                        }
                    )

        # Aggregate
        results["alpha_results"][str(alpha)] = {}
        for fw_name, metrics_list in alpha_metrics.items():
            if metrics_list:
                results["alpha_results"][str(alpha)][fw_name] = {
                    "n_samples": len(metrics_list),
                    "mean_baseline_fdr": float(np.mean([m["baseline_fdr"] for m in metrics_list])),
                    "mean_steered_fdr": float(np.mean([m["steered_fdr"] for m in metrics_list])),
                    "mean_fdr_reduction": float(np.mean([m["fdr_reduction"] for m in metrics_list])),
                    "mean_consistency_gain": float(np.mean([m["consistency_gain"] for m in metrics_list])),
                }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate steering with high-resolution alpha sweep.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama", "qwen"],
        choices=["llama", "qwen"],
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--n-alpha",
        type=int,
        default=1000,
        help="Number of alpha values to test (default: 1000).",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.01,
        help="Minimum alpha value (default: 0.01).",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=10.0,
        help="Maximum alpha value (default: 10.0).",
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=50,
        help="Number of unstable samples to test per alpha (default: 50).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    analysis_dir = base_dir / "data" / "analysis"

    alpha_values = np.logspace(np.log10(args.alpha_min), np.log10(args.alpha_max), args.n_alpha).tolist()
    print(f"Steering Evaluation: {len(alpha_values)} alpha values from {alpha_values[0]:.4f} to {alpha_values[-1]:.2f}")

    # Load splits
    with open(analysis_dir / "stable_unstable_splits.json", "r") as f:
        splits = json.load(f)

    # Evaluate each model
    eval_results_all: dict = {}
    for model_key in args.models:
        steering_data = load_steering_vectors(analysis_dir / f"steering_vectors_{model_key}.pt")
        if steering_data is None:
            print(f"Steering vectors not found for {model_key}")
            continue

        h5_path = analysis_dir / f"activations_{model_key}.h5"
        if not h5_path.exists():
            print(f"Activation file not found: {h5_path}")
            continue

        eval_results = evaluate_steering(
            model_key,
            steering_data,
            splits[model_key],
            h5_path,
            alpha_values,
            analysis_dir,
            n_test_samples=args.n_test_samples,
        )
        eval_results_all[model_key] = eval_results

    # Save combined results
    output_data = {
        "config": {
            "alpha_values": alpha_values,
            "n_alpha": len(alpha_values),
            "optimal_layers": OPTIMAL_LAYERS,
        },
    }
    output_data.update(eval_results_all)

    output_path = analysis_dir / "steering_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=float)
    print(f"\nSaved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("STEERING EVALUATION SUMMARY")
    print("=" * 70)
    for model_name, eval_results in eval_results_all.items():
        if eval_results is None:
            continue
        best_alpha, best_reduction = None, -float("inf")
        for alpha_str, fw_results in eval_results["alpha_results"].items():
            if fw_results:
                avg = float(np.mean([m["mean_fdr_reduction"] for m in fw_results.values()]))
                if avg > best_reduction:
                    best_reduction = avg
                    best_alpha = float(alpha_str)
        print(f"\n{model_name.upper()}")
        print(f"  Best alpha: {best_alpha:.4f}")
        print(f"  Best avg FDR reduction: {best_reduction:.4f} ({best_reduction * 100:.2f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
