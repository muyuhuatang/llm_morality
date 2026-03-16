"""Extract hidden-state activations from Llama-3.3-70B or Qwen2.5-72B.

For each step instance in the probing dataset, constructs the prompt the
model would have seen (scenario + prior steps + current step description),
performs a forward pass, and saves the last-token hidden state at every
layer to an HDF5 file.

Supports both models via --model flag.  Uses os.environ["HF_TOKEN"] for
HuggingFace authentication.

Inputs (relative to --data-dir):
    probing_dataset.parquet
    probing_splits.json

Outputs (relative to --data-dir):
    activations_{llama,qwen}.h5
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "llama": {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "num_layers": 80,
        "d_model": 8192,
        "output_file": "activations_llama.h5",
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "num_layers": 80,
        "d_model": 8192,
        "output_file": "activations_qwen.h5",
    },
}

STEP_DESCRIPTIONS = {
    1: "Identify the key moral issue in the scenario",
    2: "Consider the intentions and context of the action",
    3: "Analyze from multiple stakeholder perspectives",
    4: "Integrate analysis and provide final moral judgment",
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def construct_prompt_for_step(row: pd.Series, all_steps_df: pd.DataFrame) -> str:
    """Build the prompt the model sees *before* generating the step response.

    Includes the scenario, all completed previous steps, and the current
    step description.  Activations are extracted at the last token.
    """
    sample_id = row["sample_id"]
    model_id = row["model_id"]
    current_step = row["step_id"]
    scenario_text = row["scenario_text"]

    prev_steps = all_steps_df[
        (all_steps_df["sample_id"] == sample_id)
        & (all_steps_df["model_id"] == model_id)
        & (all_steps_df["step_id"] < current_step)
    ].sort_values("step_id")

    parts = [
        f"Scenario: {scenario_text}\n\n",
        "Analyze this scenario step by step:\n\n",
    ]
    for _, prev_row in prev_steps.iterrows():
        sn = prev_row["step_id"]
        parts.append(f"Step {sn}: {STEP_DESCRIPTIONS[sn]}\n{prev_row['step_text']}\n\n")
    parts.append(f"Step {current_step}: {STEP_DESCRIPTIONS[current_step]}\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_name: str, device: str, cache_dir: str | None):
    """Load model (bfloat16, output_hidden_states=True) and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading tokenizer for {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {model_name} ...")
    kwargs: dict = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
        token=hf_token,
    )
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    print(f"  Loaded on {next(model.parameters()).device}")
    print(f"  Layers: {model.config.num_hidden_layers}, hidden: {model.config.hidden_size}")
    return model, tokenizer


def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_activations(
    model, tokenizer, prompt: str, device: str = "cuda", max_length: int = 4096
) -> np.ndarray:
    """Return hidden states at the last-token position for all layers.

    Returns:
        np.ndarray of shape (num_layers + 1, d_model).
        Index 0 = embedding layer, 1..N = transformer layer outputs.
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

    last_token_acts = []
    for layer_hidden in hidden_states:
        last_token_acts.append(layer_hidden[0, -1, :].float().cpu().numpy())
    return np.stack(last_token_acts)


def validate_activation_shape(
    activations: np.ndarray, expected_layers: int, expected_dim: int
) -> None:
    expected = (expected_layers + 1, expected_dim)
    if activations.shape != expected:
        raise ValueError(f"Shape {activations.shape} != expected {expected}")
    if np.isnan(activations).any():
        raise ValueError("NaN detected in activations")
    if np.isinf(activations).any():
        raise ValueError("Inf detected in activations")


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def extract_model_activations(
    model_id: str,
    model_config: dict,
    probing_df: pd.DataFrame,
    output_path: Path,
    device: str,
    cache_dir: str | None,
    checkpoint_interval: int = 100,
) -> Path:
    """Extract and save activations for all step instances of *model_id*."""
    model_df = probing_df[probing_df["model_id"] == model_id].copy()
    n_instances = len(model_df)
    print(f"\nExtracting activations for {model_id}: {n_instances} instances")

    model, tokenizer = load_model_and_tokenizer(
        model_config["name"], device, cache_dir
    )
    num_layers = model_config["num_layers"]
    d_model = model_config["d_model"]

    with h5py.File(output_path, "w") as f:
        act_ds = f.create_dataset(
            "activations",
            shape=(n_instances, num_layers + 1, d_model),
            dtype="float32",
            chunks=(1, num_layers + 1, d_model),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "sample_ids", data=model_df["sample_id"].values.astype("S")
        )
        f.create_dataset("step_ids", data=model_df["step_id"].values)
        f.attrs["model_id"] = model_id
        f.attrs["model_name"] = model_config["name"]
        f.attrs["num_layers"] = num_layers
        f.attrs["d_model"] = d_model
        f.attrs["n_instances"] = n_instances

        scenarios_done: set = set()
        for idx, (_, row) in enumerate(
            tqdm(model_df.iterrows(), total=n_instances, desc=f"Extracting {model_id}")
        ):
            try:
                prompt = construct_prompt_for_step(row, probing_df)
                acts = extract_activations(model, tokenizer, prompt, device=device)
                validate_activation_shape(acts, num_layers, d_model)
                act_ds[idx] = acts
            except Exception as exc:
                print(
                    f"  Error idx={idx} ({row['sample_id']}, step {row['step_id']}): {exc}"
                )
                act_ds[idx] = np.zeros((num_layers + 1, d_model), dtype=np.float32)

            prev = len(scenarios_done)
            scenarios_done.add(row["sample_id"])
            if len(scenarios_done) > prev and len(scenarios_done) % checkpoint_interval == 0:
                print(f"  Checkpoint: {len(scenarios_done)} scenarios")
                f.flush()

        f.flush()
        print(f"  Done: {len(scenarios_done)} unique scenarios")

    del model, tokenizer
    clear_gpu_memory()
    print(f"  Saved to {output_path}")
    return output_path


def verify_activations(filepath: Path) -> None:
    """Print summary statistics and flag issues in the HDF5 file."""
    print(f"\nVerifying {filepath} ...")
    with h5py.File(filepath, "r") as f:
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        activations = f["activations"]
        n = activations.shape[0]
        zero_count = sum(1 for i in range(n) if np.allclose(activations[i], 0))
        nan_count = sum(1 for i in range(n) if np.isnan(activations[i]).any())
        inf_count = sum(1 for i in range(n) if np.isinf(activations[i]).any())
        print(f"  Zeros: {zero_count}/{n}, NaN: {nan_count}/{n}, Inf: {inf_count}/{n}")
        if zero_count == 0 and nan_count == 0 and inf_count == 0:
            print("  Verification passed.")
        else:
            print("  WARNING: issues detected.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hidden-state activations for probing."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llama", "qwen"],
        help="Which model to extract activations from.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Relative path to data directory (default: data).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (default: cuda).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace model cache directory.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Print checkpoint every N scenarios (default: 100).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip extraction; just verify an existing HDF5 file.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = MODEL_CONFIGS[args.model]
    output_path = data_dir / config["output_file"]

    if args.verify_only:
        verify_activations(output_path)
        return

    # Ensure HF_TOKEN is available
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set. Private/gated models may fail to load.")

    # Load probing dataset
    print("Loading probing dataset ...")
    probing_df = pd.read_parquet(data_dir / "probing_dataset.parquet")
    print(f"  {len(probing_df)} step instances")

    # Set CUDA device if specified in env
    if args.device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"  CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # Run extraction
    extract_model_activations(
        model_id=args.model,
        model_config=config,
        probing_df=probing_df,
        output_path=output_path,
        device=args.device,
        cache_dir=args.cache_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Verify
    verify_activations(output_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
