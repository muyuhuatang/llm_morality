#!/usr/bin/env python3
"""Moral Representation Consistency (MRC).

    MRC = (1/3) * Stability + (1/3) * (1 - FDR) + (1/3) * (1 - H_norm)

  Stability   fraction of steps whose dominant ethical framework equals the
              trajectory's modal framework (a mode fraction over the Scoring LLM's
              step-level framework attributions). No probe output is used, so MRC
              is independent of the probing analysis rather than circular with it.
  1 - FDR     framework drift rate, the share of consecutive step pairs whose
              dominant framework changes.
  1 - H_norm  normalised Shannon entropy of the trajectory's framework distribution.

Reads  data/analysis/trajectory_metrics.csv
Writes data/analysis/moral_representation_consistency.csv

Usage:  python3 compute_moral_representation_consistency.py [--data ../data]
"""
import argparse, json, os
from collections import Counter

import numpy as np
import pandas as pd

N_FRAMEWORKS = 5
MAX_ENTROPY = float(np.log(N_FRAMEWORKS))
WEIGHTS = {"stability": 1 / 3, "drift": 1 / 3, "variance": 1 / 3}


def stability(dominant_sequence: str) -> float:
    seq = json.loads(dominant_sequence)
    if not seq:
        return float("nan")
    return Counter(seq).most_common(1)[0][1] / len(seq)


def variance_component(entropy: float) -> float:
    return float(np.clip(1.0 - entropy / MAX_ENTROPY, 0.0, 1.0))


def trajectory_category(dominant_sequence: str, fdr: float) -> str:
    """single_framework | bounce | high_entropy."""
    seq = json.loads(dominant_sequence)
    if len(set(seq)) == 1:
        return "single_framework"
    return "high_entropy" if fdr >= 1.0 else "bounce"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mrc_stability_component"] = out["dominant_sequence"].map(stability)
    out["mrc_drift_component"] = 1.0 - out["fdr"]
    out["mrc_variance_component"] = out["entropy"].map(variance_component)
    out["mrc_score"] = (
        WEIGHTS["stability"] * out["mrc_stability_component"]
        + WEIGHTS["drift"] * out["mrc_drift_component"]
        + WEIGHTS["variance"] * out["mrc_variance_component"]
    ).clip(0.0, 1.0)
    if "trajectory_category" not in out.columns:
        out["trajectory_category"] = [
            trajectory_category(s, f) for s, f in zip(out["dominant_sequence"], out["fdr"])
        ]
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "..", "data"))
    a = ap.parse_args()

    src = os.path.join(a.data, "analysis", "trajectory_metrics.csv")
    df = pd.read_csv(src)
    print(f"loaded {len(df)} trajectories from {src}")

    mrc = compute(df)

    # Stability is a mode fraction, so a trajectory that changes framework at least
    # once cannot be perfectly stable. Assert it: an earlier implementation resolved
    # this component through a partial probing table and violated the invariant on
    # 1,987 trajectories.
    violations = int(((mrc.fdr > 0) & np.isclose(mrc.mrc_stability_component, 1.0)).sum())
    assert violations == 0, f"{violations} trajectories have FDR>0 but Stability==1.0"
    print("invariant OK: no trajectory with FDR>0 has Stability = 1.0")

    print("\nStability by drift level (must decrease monotonically):")
    print(mrc.groupby(mrc.fdr.round(2)).mrc_stability_component.mean().round(4).to_string())
    print("\nMRC by trajectory category:")
    print(mrc.groupby("trajectory_category").mrc_score.agg(["mean", "std", "count"]).round(3).to_string())
    print(f"\noverall MRC {mrc.mrc_score.mean():.3f} +/- {mrc.mrc_score.std():.3f} (n={len(mrc)})")

    cols = [c for c in ["sample_id", "model", "dataset", "fdr", "entropy", "faithfulness",
                        "mrc_variance_component", "mrc_drift_component",
                        "mrc_stability_component", "mrc_score", "trajectory_category"]
            if c in mrc.columns]
    dest = os.path.join(a.data, "analysis", "moral_representation_consistency.csv")
    mrc[cols].to_csv(dest, index=False)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
