"""
Generate LLM coherence ratings (0-100) for stratified trajectory samples.

Sends 180 trajectories (60 per category: single_framework, bounce, high_entropy)
to GPT-OSS-120B via Together.ai and collects three ratings per trajectory at
temperature 0 for maximum consistency.  The final score is the **median** of the
three ratings (more robust to outliers than the mean).

The prompt uses few-shot calibration examples so the LLM rates framework
consistency on the same criteria captured by MRC.

Environment:
    TOGETHER_API_KEY  -- Together.ai API key (required)

Inputs:
    - data/analysis/mrc_scores.csv
    - data/analysis/probing_dataset.parquet

Outputs:
    - data/annotation/llm_annotations.json   (full annotations + config + summary)
    - data/analysis/llm_ratings.csv           (flat table for downstream analysis)
"""

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "openai/gpt-oss-120b"
N_SAMPLES = 180
N_RATINGS_PER_SAMPLE = 3
MAX_RETRIES = 3
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert in moral philosophy evaluating the coherence of ethical reasoning.\n\n"
    "Your task is to rate how coherent and consistent the ethical reasoning is across a "
    "multi-step moral reasoning trajectory.\n\n"
    "Key evaluation criteria (in order of importance):\n"
    "1. FRAMEWORK CONSISTENCY (most important): Does the reasoning maintain the SAME ethical "
    "framework throughout? Switching between frameworks indicates LOW coherence.\n"
    "2. LOGICAL PROGRESSION: Do later steps build on earlier reasoning coherently?\n"
    "3. INTERNAL CONSISTENCY: Is the reasoning free from contradictions?\n\n"
    "A trajectory that stays in ONE framework throughout = HIGH score (80-100)\n"
    "A trajectory that switches frameworks multiple times = LOW score (20-50)"
)

FEW_SHOT_EXAMPLES = """
EXAMPLE 1 (High coherence - score 90):
Step 1: Framework: utilitarianism - "We should maximize overall happiness"
Step 2: Framework: utilitarianism - "The action that produces most good consequences"
Step 3: Framework: utilitarianism - "Calculating net benefit for all affected"
Step 4: Framework: utilitarianism - "The utilitarian conclusion is..."
-> RATING: 90 (Consistent framework throughout, logical progression)

EXAMPLE 2 (Low coherence - score 35):
Step 1: Framework: virtue_ethics - "A virtuous person would..."
Step 2: Framework: utilitarianism - "We need to maximize utility"
Step 3: Framework: deontology - "There's a duty to follow rules"
Step 4: Framework: virtue_ethics - "Character is what matters"
-> RATING: 35 (Multiple framework switches, inconsistent reasoning)

EXAMPLE 3 (Medium coherence - score 60):
Step 1: Framework: deontology - "Following moral rules"
Step 2: Framework: deontology - "The categorical imperative requires"
Step 3: Framework: utilitarianism - "But considering consequences..."
Step 4: Framework: deontology - "Ultimately duty takes precedence"
-> RATING: 60 (Mostly consistent with one drift)
"""


# ---------------------------------------------------------------------------
# Trajectory text reconstruction
# ---------------------------------------------------------------------------
def reconstruct_trajectory_text(
    sample_id: str,
    probing_df: pd.DataFrame,
) -> str | None:
    """Build a readable multi-step trajectory from the probing dataset."""
    steps = probing_df[probing_df["sample_id"] == sample_id].sort_values("step_id")
    if len(steps) == 0:
        return None

    parts = [f"Moral Reasoning Trajectory (Sample: {sample_id})\n"]

    first = steps.iloc[0]
    if "scenario_text" in first.index and pd.notna(first["scenario_text"]):
        scenario = str(first["scenario_text"])
        if len(scenario) > 300:
            scenario = scenario[:300] + "..."
        parts.append(f"Scenario: {scenario}\n")

    for _, step in steps.iterrows():
        step_num = step.get("step_id", "?")
        framework = step.get("dominant_framework", "unknown")
        parts.append(f"Step {step_num}:")
        parts.append(f"  Framework: {framework}")

        if "step_text" in step.index and pd.notna(step["step_text"]):
            text = str(step["step_text"])
            if len(text) > 200:
                text = text[:200] + "..."
            parts.append(f"  Reasoning: {text}")
        elif "step_description" in step.index and pd.notna(step["step_description"]):
            desc = str(step["step_description"])
            if len(desc) > 200:
                desc = desc[:200] + "..."
            parts.append(f"  Description: {desc}")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Annotation prompt (no MRC metrics -- avoids information leakage)
# ---------------------------------------------------------------------------
def create_annotation_prompt(trajectory_text: str) -> str:
    return (
        "Rate the coherence and consistency of ethical reasoning in the following "
        "moral reasoning trajectory.\n\n"
        "RATING SCALE (0-100):\n"
        "- 0-20: Incoherent - Chaotic framework switching, contradictory reasoning\n"
        "- 21-40: Poor - Multiple framework switches, weak logical connection\n"
        "- 41-60: Moderate - Some framework drift but partial coherence\n"
        "- 61-80: Good - Mostly consistent framework with minor deviations\n"
        "- 81-100: Excellent - Same framework throughout, strong logical progression\n\n"
        f"CALIBRATION EXAMPLES:\n{FEW_SHOT_EXAMPLES}\n"
        f"NOW EVALUATE THIS TRAJECTORY:\n{trajectory_text}\n\n"
        "Provide your rating as a number from 0 to 100.\n"
        "Format: RATING: [number]\n"
        "JUSTIFICATION: [brief explanation focusing on framework consistency]"
    )


# ---------------------------------------------------------------------------
# Rating parser
# ---------------------------------------------------------------------------
def parse_rating(response_text: str | None) -> tuple[int | None, str | None]:
    """Extract a 0-100 integer rating from the LLM response."""
    if response_text is None:
        return None, None

    # Primary pattern
    m = re.search(r"RATING:\s*(\d{1,3})", response_text)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            jm = re.search(r"JUSTIFICATION:\s*(.+)", response_text, re.DOTALL)
            return val, (jm.group(1).strip() if jm else None)

    # Fallbacks
    for pat in [
        r"score[:\s]+(\d{1,3})",
        r"rating[:\s]+(\d{1,3})",
        r"\b(\d{1,3})\s*/\s*100",
        r"\b(\d{1,3})\s*out of\s*100",
    ]:
        m = re.search(pat, response_text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val, response_text

    # Last resort: first 2-digit number
    for num_str in re.findall(r"\b(\d{2})\b", response_text):
        num = int(num_str)
        if 0 <= num <= 100:
            return num, response_text

    return None, response_text


# ---------------------------------------------------------------------------
# Single-trajectory annotation with retries
# ---------------------------------------------------------------------------
def annotate_trajectory(
    client,
    trajectory_text: str,
    model: str,
    temperature: float,
) -> str | None:
    """Call the Together.ai chat API, retrying on failure / invalid output."""
    prompt = create_annotation_prompt(trajectory_text)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=300,
            )
            text = response.choices[0].message.content
            rating, _ = parse_rating(text)
            if rating is not None:
                return text
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
        except Exception as exc:  # noqa: BLE001
            if attempt < MAX_RETRIES - 1:
                print(f"  Error (attempt {attempt + 1}/{MAX_RETRIES}): {exc}, retrying...")
                time.sleep(2)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts: {exc}")
                return None
    return None


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------
def stratified_sample(
    df: pd.DataFrame,
    n_samples: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Return *n_samples* rows balanced across trajectory categories."""
    categories = df["trajectory_category"].unique()
    per_cat = n_samples // len(categories)

    parts = []
    for cat in categories:
        cat_df = df[df["trajectory_category"] == cat]
        parts.append(cat_df.sample(n=min(per_cat, len(cat_df)), random_state=seed))

    result = pd.concat(parts, ignore_index=True)

    # Top up if needed
    if len(result) < n_samples:
        remaining = df[~df["sample_id"].isin(result["sample_id"])]
        extra = min(n_samples - len(result), len(remaining))
        if extra > 0:
            result = pd.concat(
                [result, remaining.sample(n=extra, random_state=seed)],
                ignore_index=True,
            )
    return result


# ---------------------------------------------------------------------------
# Full annotation loop
# ---------------------------------------------------------------------------
def run_annotations(
    client,
    sampled_df: pd.DataFrame,
    probing_df: pd.DataFrame,
    model: str,
    temperature: float,
    n_ratings: int,
) -> list[dict]:
    """Collect *n_ratings* per trajectory and compute median / mean / std."""
    annotations: list[dict] = []

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Annotating"):
        traj_text = reconstruct_trajectory_text(row["sample_id"], probing_df)
        if traj_text is None:
            continue

        record: dict = {
            "sample_id": row["sample_id"],
            "model": row.get("model", "unknown"),
            "trajectory_category": row["trajectory_category"],
            "mrc_score": float(row["mrc_score"]),
            "fdr": float(row["fdr"]),
            "entropy": float(row.get("entropy", np.nan)),
            "mrc_variance_component": float(row.get("mrc_variance_component", np.nan)),
            "mrc_drift_component": float(row.get("mrc_drift_component", np.nan)),
            "mrc_stability_component": float(row.get("mrc_stability_component", np.nan)),
            "ratings": [],
            "justifications": [],
            "raw_responses": [],
        }

        for _ in range(n_ratings):
            resp = annotate_trajectory(client, traj_text, model, temperature)
            record["raw_responses"].append(resp)
            rating, justification = parse_rating(resp)
            if rating is not None:
                record["ratings"].append(rating)
                record["justifications"].append(justification)
            time.sleep(0.5)

        if record["ratings"]:
            record["median_rating"] = float(np.median(record["ratings"]))
            record["mean_rating"] = float(np.mean(record["ratings"]))
            record["std_rating"] = float(np.std(record["ratings"]))
        else:
            record["median_rating"] = None
            record["mean_rating"] = None
            record["std_rating"] = None

        annotations.append(record)

    return annotations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect LLM coherence ratings for MRC validation.",
    )
    ap.add_argument(
        "--base-dir", type=str, default=".",
        help="Project root (contains data/ and results/ sub-trees).",
    )
    ap.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Together.ai model ID (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--n-samples", type=int, default=N_SAMPLES,
        help="Total trajectories to annotate (default: 180).",
    )
    ap.add_argument(
        "--n-ratings", type=int, default=N_RATINGS_PER_SAMPLE,
        help="Ratings per trajectory (default: 3).",
    )
    ap.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help="Sampling temperature (default: 0.0).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified sampling.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise SystemExit("Set the TOGETHER_API_KEY environment variable.")

    from together import Together  # late import to allow --help without the SDK

    client = Together(api_key=api_key)

    analysis_dir = base / "data" / "analysis"
    annotation_dir = base / "data" / "annotation"
    mrc_csv = analysis_dir / "mrc_scores.csv"
    probing_parquet = analysis_dir / "probing_dataset.parquet"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading inputs ...")
    mrc_df = pd.read_csv(mrc_csv)
    probing_df = pd.read_parquet(probing_parquet)
    print(f"  MRC scores: {len(mrc_df)}")
    print(f"  Probing steps: {len(probing_df)}")

    # Filter to trajectories with text
    available_ids = set(probing_df["sample_id"].unique())
    mrc_with_text = mrc_df[mrc_df["sample_id"].isin(available_ids)]
    print(f"  With text: {len(mrc_with_text)} / {len(mrc_df)}")

    # Sample
    sampled = stratified_sample(mrc_with_text, args.n_samples, args.seed)
    print(f"\nSampled {len(sampled)} trajectories:")
    print(sampled["trajectory_category"].value_counts().to_string())

    # Annotate
    print(f"\nAnnotating with {args.model} (T={args.temperature}, "
          f"{args.n_ratings} ratings/sample) ...")
    annotations = run_annotations(
        client, sampled, probing_df, args.model, args.temperature, args.n_ratings,
    )

    # Build flat DataFrame
    ann_df = pd.DataFrame(
        [
            {
                "sample_id": a["sample_id"],
                "model": a["model"],
                "trajectory_category": a["trajectory_category"],
                "mrc_score": a["mrc_score"],
                "mrc_variance_component": a.get("mrc_variance_component", np.nan),
                "mrc_drift_component": a.get("mrc_drift_component", np.nan),
                "mrc_stability_component": a.get("mrc_stability_component", np.nan),
                "fdr": a["fdr"],
                "entropy": a.get("entropy", np.nan),
                "median_rating": a.get("median_rating"),
                "mean_rating": a.get("mean_rating"),
                "std_rating": a.get("std_rating"),
                "n_ratings": len(a["ratings"]),
            }
            for a in annotations
        ]
    )

    # Quick correlation check
    valid = ann_df.dropna(subset=["median_rating", "mrc_score"])
    if len(valid) >= 10:
        r, p = stats.pearsonr(valid["mrc_score"], valid["median_rating"])
        rho, sp = stats.spearmanr(valid["mrc_score"], valid["median_rating"])
        print(f"\nQuick correlation (n={len(valid)}):")
        print(f"  Pearson r  = {r:.4f} (p={p:.4e})")
        print(f"  Spearman rho = {rho:.4f} (p={sp:.4e})")

    # Save JSON
    payload = {
        "config": {
            "model": args.model,
            "n_samples": len(annotations),
            "n_ratings_per_sample": args.n_ratings,
            "temperature": args.temperature,
            "max_retries": MAX_RETRIES,
            "rating_scale": "0-100",
            "aggregation": "median",
        },
        "annotations": annotations,
        "summary": {
            "total_annotated": len(ann_df),
            "valid_ratings": int(ann_df["median_rating"].notna().sum()),
            "median_rating_mean": (
                float(ann_df["median_rating"].mean())
                if ann_df["median_rating"].notna().any()
                else None
            ),
            "median_rating_std": (
                float(ann_df["median_rating"].std())
                if ann_df["median_rating"].notna().any()
                else None
            ),
            "inter_rating_consistency": (
                float(np.nanmean(ann_df["std_rating"]))
                if ann_df["std_rating"].notna().any()
                else None
            ),
        },
    }

    json_out = annotation_dir / "llm_annotations.json"
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved -> {json_out}")

    csv_out = analysis_dir / "llm_ratings.csv"
    ann_df.to_csv(csv_out, index=False)
    print(f"Saved -> {csv_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
