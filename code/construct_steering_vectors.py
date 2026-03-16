"""
Construct probe-guided steering vectors from stable vs unstable moral reasoning trajectories.

Steering vector formula:
    v_f = E[h^(l*) | stable, dominant=f] - E[h^(l*) | unstable]

Identifies stable (FDR < 0.05) and unstable (FDR > 0.15) trajectories from
trajectory metrics, loads activations at optimal probe layers (L63 Llama,
L17 Qwen), computes per-framework steering vectors, analyzes pairwise cosine
similarities, and saves vectors as .pt files.

Inputs:
    - data/analysis/activations_llama.h5
    - data/analysis/activations_qwen.h5
    - data/analysis/trajectory_metrics.csv
    - data/analysis/probing_dataset.parquet

Outputs:
    - data/analysis/stable_unstable_splits.json
    - data/analysis/steering_vectors_llama.pt
    - data/analysis/steering_vectors_qwen.pt
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STABLE_FDR_THRESHOLD = 0.05
UNSTABLE_FDR_THRESHOLD = 0.15

OPTIMAL_LAYERS = {"llama": 63, "qwen": 17}

FRAMEWORK_NAMES = [
    "benthamite_act_utilitarianism",
    "kantian_deontology",
    "aristotelian_virtue_ethics",
    "scanlonian_contractualism",
    "gauthierian_contractarianism",
]

FRAMEWORK_SHORT = {
    "benthamite_act_utilitarianism": "util",
    "kantian_deontology": "kant",
    "aristotelian_virtue_ethics": "virt",
    "scanlonian_contractualism": "scan",
    "gauthierian_contractarianism": "gaut",
}

MODEL_NAME_MAP = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "qwen": "Qwen/Qwen2.5-72B-Instruct-Turbo",
}

MIN_SAMPLES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_stable_unstable_splits(
    probing_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    model_key: str,
) -> dict:
    """Identify stable and unstable trajectories based on FDR thresholds.

    Filters by both sample_id AND model to avoid mixing FDR values from
    different models.
    """
    model_probing = probing_df[probing_df["model_id"] == model_key].copy()
    unique_samples = model_probing["sample_id"].unique()
    print(f"\n{model_key}: {len(unique_samples)} unique samples in probing data")

    model_full_name = MODEL_NAME_MAP.get(model_key)
    if model_full_name:
        model_metrics = trajectory_df[
            (trajectory_df["sample_id"].isin(unique_samples))
            & (trajectory_df["model"] == model_full_name)
        ].copy()
    else:
        model_metrics = trajectory_df[
            (trajectory_df["sample_id"].isin(unique_samples))
            & (trajectory_df["model"].str.lower().str.contains(model_key.lower()))
        ].copy()

    print(f"  Matched {len(model_metrics)} trajectories with metrics")

    stable_mask = model_metrics["fdr"] < STABLE_FDR_THRESHOLD
    unstable_mask = model_metrics["fdr"] > UNSTABLE_FDR_THRESHOLD

    stable_samples = model_metrics[stable_mask]["sample_id"].tolist()
    unstable_samples = model_metrics[unstable_mask]["sample_id"].tolist()

    print(f"  Stable (FDR < {STABLE_FDR_THRESHOLD}): {len(stable_samples)} trajectories")
    print(f"  Unstable (FDR > {UNSTABLE_FDR_THRESHOLD}): {len(unstable_samples)} trajectories")

    # Dominant framework breakdown for stable trajectories
    stable_by_framework: dict[str, list] = defaultdict(list)
    for _, row in model_metrics[stable_mask].iterrows():
        try:
            seq = eval(row["dominant_sequence"])  # noqa: S307
            dominant = seq[0] if seq else None
            if dominant:
                stable_by_framework[dominant].append(row["sample_id"])
        except Exception:
            pass

    print("  Stable by framework:")
    for fw, samples in stable_by_framework.items():
        print(f"    {FRAMEWORK_SHORT.get(fw, fw)}: {len(samples)}")

    return {
        "stable_samples": stable_samples,
        "unstable_samples": unstable_samples,
        "stable_by_framework": dict(stable_by_framework),
        "metrics": {
            "stable_mean_fdr": float(
                model_metrics[stable_mask]["fdr"].mean() if len(stable_samples) > 0 else 0
            ),
            "unstable_mean_fdr": float(
                model_metrics[unstable_mask]["fdr"].mean() if len(unstable_samples) > 0 else 0
            ),
            "stable_mean_entropy": float(
                model_metrics[stable_mask]["entropy"].mean() if len(stable_samples) > 0 else 0
            ),
            "unstable_mean_entropy": float(
                model_metrics[unstable_mask]["entropy"].mean() if len(unstable_samples) > 0 else 0
            ),
        },
    }


def load_activations_for_samples(
    h5_path: Path,
    sample_ids: list,
    layer: int,
    step: int | None = None,
) -> tuple:
    """Load activations for specific samples at a given layer.

    Supports both flat (activations, sample_ids, step_ids) and hierarchical
    H5 structures.

    Returns:
        (activations_tensor, valid_sample_ids)
    """
    activations = []
    valid_ids = []
    sample_ids_set = set(sample_ids)

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        if "activations" in keys and "sample_ids" in keys:
            # Flat structure
            all_activations = f["activations"][:]
            all_sample_ids = [
                s.decode() if isinstance(s, bytes) else s for s in f["sample_ids"][:]
            ]
            all_step_ids = f["step_ids"][:] if "step_ids" in keys else None

            for idx, sid in enumerate(
                tqdm(all_sample_ids, desc="Filtering activations", leave=False)
            ):
                if sid not in sample_ids_set:
                    continue
                if step is not None and all_step_ids is not None:
                    if all_step_ids[idx] != step:
                        continue
                act = all_activations[idx, layer, :]
                activations.append(act)
                valid_ids.append(sid)
        else:
            # Hierarchical structure
            available_keys = set(keys)
            for sample_id in tqdm(sample_ids, desc="Loading activations", leave=False):
                key = sample_id
                if key not in available_keys:
                    for s in range(1, 5):
                        step_key = f"{sample_id}_step{s}"
                        if step_key in available_keys:
                            key = step_key
                            break
                if key not in available_keys:
                    continue
                sample_data = f[key]
                if isinstance(sample_data, h5py.Group) and "activations" in sample_data:
                    act = sample_data["activations"][:]
                    if len(act.shape) == 2:
                        activations.append(act[layer])
                    elif len(act.shape) == 3:
                        if step is not None:
                            activations.append(act[step - 1, layer])
                        else:
                            activations.append(act[:, layer].mean(axis=0))
                    valid_ids.append(sample_id)

    if len(activations) == 0:
        return None, []

    activations_np = np.stack(activations)
    return torch.tensor(activations_np, dtype=torch.float32), valid_ids


def construct_steering_vectors(
    h5_path: Path,
    splits: dict,
    optimal_layer: int,
    model_name: str,
) -> tuple:
    """Construct per-framework steering vectors.

    v_f = E[h | stable, framework=f] - E[h | unstable]
    """
    print(f"\nConstructing steering vectors for {model_name}")
    print(f"  Optimal layer: {optimal_layer}")

    steering_vectors = {}
    metadata: dict = {
        "model": model_name,
        "optimal_layer": optimal_layer,
        "frameworks": {},
    }

    # Load unstable activations (shared across all frameworks)
    unstable_samples = splits["unstable_samples"]
    print(f"  Loading unstable activations ({len(unstable_samples)} samples)...")
    unstable_acts, unstable_valid = load_activations_for_samples(
        h5_path, unstable_samples, optimal_layer
    )

    if unstable_acts is None or len(unstable_valid) == 0:
        print("  WARNING: No unstable activations found!")
        unstable_mean = None
    else:
        unstable_mean = unstable_acts.mean(dim=0)
        print(f"  Loaded {len(unstable_valid)} unstable activations")
        metadata["unstable"] = {
            "n_samples": len(unstable_valid),
            "mean_norm": float(torch.norm(unstable_mean)),
        }

    # Per-framework steering vectors
    stable_by_fw = splits["stable_by_framework"]
    for framework_full, sample_ids in stable_by_fw.items():
        framework = FRAMEWORK_SHORT.get(framework_full, framework_full)
        print(f"\n  Framework: {framework} ({len(sample_ids)} stable samples)")

        if len(sample_ids) < MIN_SAMPLES:
            print(f"    Skipping: too few samples (need >= {MIN_SAMPLES})")
            continue

        stable_acts, stable_valid = load_activations_for_samples(
            h5_path, sample_ids, optimal_layer
        )
        if stable_acts is None or len(stable_valid) == 0:
            print("    WARNING: No stable activations found")
            continue

        stable_mean = stable_acts.mean(dim=0)
        print(f"    Loaded {len(stable_valid)} stable activations")

        if unstable_mean is not None:
            steering_vector = stable_mean - unstable_mean
        else:
            steering_vector = stable_mean

        steering_vector_norm = steering_vector / torch.norm(steering_vector)

        steering_vectors[framework] = {
            "vector": steering_vector,
            "vector_normalized": steering_vector_norm,
            "stable_mean": stable_mean,
            "magnitude": float(torch.norm(steering_vector)),
        }

        metadata["frameworks"][framework] = {
            "n_stable_samples": len(stable_valid),
            "steering_magnitude": float(torch.norm(steering_vector)),
            "stable_mean_norm": float(torch.norm(stable_mean)),
            "cosine_with_unstable": float(
                F.cosine_similarity(
                    stable_mean.unsqueeze(0),
                    (unstable_mean if unstable_mean is not None else stable_mean).unsqueeze(0),
                )[0]
            ),
        }

        print(f"    Steering vector magnitude: {metadata['frameworks'][framework]['steering_magnitude']:.4f}")
        print(f"    Cosine(stable, unstable): {metadata['frameworks'][framework]['cosine_with_unstable']:.4f}")

    return steering_vectors, metadata


def analyze_steering_vectors(steering_vectors: dict, model_name: str) -> dict | None:
    """Compute pairwise cosine similarities between steering vectors."""
    if not steering_vectors:
        print(f"No steering vectors for {model_name}")
        return None

    print(f"\n{'=' * 60}")
    print(f"Steering Vector Analysis: {model_name.upper()}")
    print(f"{'=' * 60}")

    frameworks = list(steering_vectors.keys())
    n_fw = len(frameworks)

    similarity_matrix = np.zeros((n_fw, n_fw))
    for i, fw1 in enumerate(frameworks):
        for j, fw2 in enumerate(frameworks):
            v1 = steering_vectors[fw1]["vector_normalized"]
            v2 = steering_vectors[fw2]["vector_normalized"]
            sim = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))
            similarity_matrix[i, j] = sim

    # Print similarity matrix
    print(f"\nPairwise Cosine Similarities:")
    header = f"{'':>6}" + "".join(f"{fw:>8}" for fw in frameworks)
    print(header)
    for i, fw1 in enumerate(frameworks):
        row = f"{fw1:<6}" + "".join(f"{similarity_matrix[i, j]:>8.3f}" for j in range(n_fw))
        print(row)

    print(f"\nSteering Vector Magnitudes:")
    for fw in frameworks:
        print(f"  {fw}: {steering_vectors[fw]['magnitude']:.4f}")

    return {"similarity_matrix": similarity_matrix.tolist(), "frameworks": frameworks}


def save_steering_vectors(steering_vectors: dict, metadata: dict, output_path: Path) -> None:
    """Save steering vectors to a PyTorch file."""
    if not steering_vectors:
        print("No vectors to save")
        return

    save_dict: dict = {"metadata": metadata, "vectors": {}}
    for fw, data in steering_vectors.items():
        save_dict["vectors"][fw] = {
            "steering_vector": data["vector"],
            "steering_vector_normalized": data["vector_normalized"],
            "stable_mean": data["stable_mean"],
            "magnitude": data["magnitude"],
        }

    torch.save(save_dict, output_path)
    print(f"Saved steering vectors to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct probe-guided steering vectors from stable/unstable trajectories."
    )
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
        help="Models to process.",
    )
    parser.add_argument(
        "--stable-fdr",
        type=float,
        default=STABLE_FDR_THRESHOLD,
        help="FDR threshold for stable trajectories (default: 0.05).",
    )
    parser.add_argument(
        "--unstable-fdr",
        type=float,
        default=UNSTABLE_FDR_THRESHOLD,
        help="FDR threshold for unstable trajectories (default: 0.15).",
    )
    args = parser.parse_args()

    global STABLE_FDR_THRESHOLD, UNSTABLE_FDR_THRESHOLD  # noqa: PLW0603
    STABLE_FDR_THRESHOLD = args.stable_fdr
    UNSTABLE_FDR_THRESHOLD = args.unstable_fdr

    base_dir = Path(args.base_dir)
    analysis_dir = base_dir / "data" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load shared data
    # ------------------------------------------------------------------
    trajectory_df = pd.read_csv(analysis_dir / "trajectory_metrics.csv")
    print(f"Loaded {len(trajectory_df)} trajectories")

    probing_df = pd.read_parquet(analysis_dir / "probing_dataset.parquet")
    print(f"Probing dataset: {len(probing_df)} rows")

    # ------------------------------------------------------------------
    # Create stable / unstable splits
    # ------------------------------------------------------------------
    splits: dict = {}
    for model_key in args.models:
        splits[model_key] = create_stable_unstable_splits(probing_df, trajectory_df, model_key)

    # Save splits
    splits_for_json = {
        model: {
            "stable_samples": data["stable_samples"],
            "unstable_samples": data["unstable_samples"],
            "stable_by_framework": data["stable_by_framework"],
            "metrics": data["metrics"],
            "thresholds": {
                "stable_fdr": STABLE_FDR_THRESHOLD,
                "unstable_fdr": UNSTABLE_FDR_THRESHOLD,
            },
        }
        for model, data in splits.items()
    }
    splits_path = analysis_dir / "stable_unstable_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits_for_json, f, indent=2)
    print(f"\nSaved splits to {splits_path}")

    # ------------------------------------------------------------------
    # Construct and save steering vectors per model
    # ------------------------------------------------------------------
    all_analyses: dict = {}
    for model_key in args.models:
        h5_path = analysis_dir / f"activations_{model_key}.h5"
        if not h5_path.exists():
            print(f"Activation file not found: {h5_path}")
            continue

        steering_vectors, metadata = construct_steering_vectors(
            h5_path,
            splits[model_key],
            OPTIMAL_LAYERS[model_key],
            model_key,
        )

        analysis = analyze_steering_vectors(steering_vectors, model_key)
        if analysis is not None:
            all_analyses[model_key] = analysis

        save_steering_vectors(
            steering_vectors,
            metadata,
            analysis_dir / f"steering_vectors_{model_key}.pt",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEERING VECTOR CONSTRUCTION COMPLETE")
    print("=" * 70)
    for model_key in args.models:
        pt_path = analysis_dir / f"steering_vectors_{model_key}.pt"
        if pt_path.exists():
            size_kb = pt_path.stat().st_size / 1024
            print(f"  {pt_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
