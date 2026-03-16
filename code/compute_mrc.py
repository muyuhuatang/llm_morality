"""
Compute MRC (Moral Representation Consistency) composite metric for all trajectories.

MRC captures the consistency and coherence of moral reasoning as a single scalar:

    MRC = (1/3) * (1 - H_norm) + (1/3) * (1 - FDR) + (1/3) * Stability

Components:
    - Variance component:  1 - normalised entropy  (lower entropy -> higher MRC)
    - Drift component:     1 - FDR                 (fewer framework switches -> higher MRC)
    - Stability component: probe-prediction consistency across steps

The script also produces category-wise statistics (single_framework, bounce,
high_entropy) and component correlation tables.

Inputs:
    - data/analysis/trajectory_metrics.csv
    - data/analysis/probing_dataset.parquet
    - data/analysis/probe_results.json

Outputs:
    - data/analysis/mrc_scores.csv           (per-trajectory MRC)
    - data/analysis/mrc_summary.json         (aggregate statistics)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# MRC weights -- equal contribution from each component
# ---------------------------------------------------------------------------
MRC_WEIGHTS = {"variance": 1 / 3, "drift": 1 / 3, "stability": 1 / 3}


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------
def compute_moral_state_variance(entropy: float) -> float:
    """Return variance component: 1 - normalised entropy (max entropy = log 5)."""
    max_entropy = np.log(5)
    return 1.0 - entropy / max_entropy


def compute_drift_magnitude(fdr: float) -> float:
    """Return drift component: 1 - FDR."""
    return 1.0 - fdr


def compute_subspace_stability(
    sample_id: str,
    model_key: str,
    probing_df: pd.DataFrame,
) -> float:
    """
    Return stability component based on dominant-framework consistency across
    steps (mode fraction).
    """
    step_data = probing_df[
        (probing_df["sample_id"] == sample_id)
        & (probing_df["model_id"] == model_key)
    ].sort_values("step_id")

    if len(step_data) < 2:
        return 1.0  # single-step -> perfectly stable

    if "dominant_framework" in step_data.columns:
        frameworks = step_data["dominant_framework"].values
        unique_vals, counts = np.unique(frameworks, return_counts=True)
        return float((frameworks == unique_vals[np.argmax(counts)]).mean())

    return 0.5  # fallback


def compute_mrc_score(
    var_component: float,
    drift_component: float,
    stability_component: float,
) -> float:
    """Composite MRC = weighted sum, clipped to [0, 1]."""
    mrc = (
        MRC_WEIGHTS["variance"] * var_component
        + MRC_WEIGHTS["drift"] * drift_component
        + MRC_WEIGHTS["stability"] * stability_component
    )
    return float(np.clip(mrc, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------
def infer_category(row: pd.Series) -> str:
    """Infer trajectory category from existing label or FDR."""
    if "trajectory_category" in row.index and pd.notna(row.get("trajectory_category")):
        return row["trajectory_category"]
    fdr = row["fdr"]
    if fdr == 0:
        return "single_framework"
    if fdr >= 0.67:
        return "high_entropy"
    return "bounce"


# ---------------------------------------------------------------------------
# Model-key resolution
# ---------------------------------------------------------------------------
def resolve_model_key(
    row: pd.Series,
    llama_samples: set,
    qwen_samples: set,
) -> str:
    """Map a trajectory row to 'llama', 'qwen', or 'unknown'."""
    model_name = str(row.get("model", row.get("model_id", "unknown")))
    if "llama" in model_name.lower():
        return "llama"
    if "qwen" in model_name.lower():
        return "qwen"
    sample_id = row["sample_id"]
    if sample_id in llama_samples:
        return "llama"
    if sample_id in qwen_samples:
        return "qwen"
    return "unknown"


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def compute_all_mrc_scores(
    trajectory_df: pd.DataFrame,
    probing_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute MRC for every trajectory and return a DataFrame."""
    llama_samples = set(
        probing_df.loc[probing_df["model_id"] == "llama", "sample_id"]
    )
    qwen_samples = set(
        probing_df.loc[probing_df["model_id"] == "qwen", "sample_id"]
    )

    records = []
    for _, row in tqdm(
        trajectory_df.iterrows(), total=len(trajectory_df), desc="Computing MRC"
    ):
        model_key = resolve_model_key(row, llama_samples, qwen_samples)

        var_comp = compute_moral_state_variance(row["entropy"])
        drift_comp = compute_drift_magnitude(row["fdr"])
        stab_comp = compute_subspace_stability(row["sample_id"], model_key, probing_df)
        mrc = compute_mrc_score(var_comp, drift_comp, stab_comp)

        records.append(
            {
                "sample_id": row["sample_id"],
                "model": model_key,
                "dataset": row.get("dataset", "unknown"),
                "fdr": row["fdr"],
                "entropy": row["entropy"],
                "faithfulness": row.get("faithfulness", np.nan),
                "mrc_variance_component": var_comp,
                "mrc_drift_component": drift_comp,
                "mrc_stability_component": stab_comp,
                "mrc_score": mrc,
                "trajectory_category": infer_category(row),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MRC scores for all trajectories."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root (contains data/ and results/ sub-trees).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)

    # Resolve input paths
    analysis_dir = base / "data" / "analysis"
    trajectory_csv = analysis_dir / "trajectory_metrics.csv"
    probing_parquet = analysis_dir / "probing_dataset.parquet"
    probe_json = analysis_dir / "probe_results.json"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading inputs ...")
    trajectory_df = pd.read_csv(trajectory_csv)
    print(f"  Trajectories: {len(trajectory_df)}")
    probing_df = pd.read_parquet(probing_parquet)
    print(f"  Probing steps: {len(probing_df)}")
    with open(probe_json) as f:
        _ = json.load(f)
    print("  Probe results: loaded")

    # Compute
    mrc_df = compute_all_mrc_scores(trajectory_df, probing_df)
    print(f"\nComputed MRC for {len(mrc_df)} trajectories")

    # Summary statistics
    print("\nMRC Score Statistics:")
    print(mrc_df["mrc_score"].describe().round(4))
    print("\nMRC by Trajectory Category:")
    print(
        mrc_df.groupby("trajectory_category")["mrc_score"]
        .agg(["mean", "std", "count"])
        .round(4)
    )

    # Component correlations
    print("\nComponent correlations with MRC:")
    for metric in ["fdr", "entropy", "faithfulness"]:
        if metric in mrc_df.columns:
            valid = mrc_df[[metric, "mrc_score"]].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[metric], valid["mrc_score"])
                print(f"  MRC vs {metric}: r = {r:.4f} (p = {p:.2e})")

    # Save CSV
    out_csv = analysis_dir / "mrc_scores.csv"
    mrc_df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    # Save summary JSON
    summary = {
        "weights": MRC_WEIGHTS,
        "overall_stats": {
            k: float(v)
            for k, v in mrc_df["mrc_score"]
            .describe()
            .to_dict()
            .items()
        },
        "by_category": {
            cat: {
                "mean": float(g["mrc_score"].mean()),
                "std": float(g["mrc_score"].std()),
                "count": len(g),
            }
            for cat, g in mrc_df.groupby("trajectory_category")
        },
        "by_model": {
            model: {
                "mean": float(g["mrc_score"].mean()),
                "std": float(g["mrc_score"].std()),
                "count": len(g),
            }
            for model, g in mrc_df.groupby("model")
        },
        "correlations": {
            "fdr_vs_mrc": float(mrc_df["fdr"].corr(mrc_df["mrc_score"])),
            "entropy_vs_mrc": float(mrc_df["entropy"].corr(mrc_df["mrc_score"])),
        },
    }
    summary_path = analysis_dir / "mrc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
