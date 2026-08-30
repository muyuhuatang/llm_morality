"""
Statistical robustness analysis of steering persuasion experiments.

Loads all steering persuasion result files (llama, qwen, and their -more
variants), unifies them into a single DataFrame, and runs:

    - Chi-square test for independence (stability vs flip)
    - Two-proportion z-test (stable < unstable flip rate)
    - Point-biserial correlation (stability vs flip)
    - Cohen's h effect size
    - Mann-Whitney U test (steering helps unstable > stable)
    - Per-attack-type chi-square tests

Inputs:
    - data/analysis/steering_persuasion_llama.json
    - data/analysis/steering_persuasion_qwen.json
    - data/analysis/steering_persuasion_llama_more.json  (optional)
    - data/analysis/steering_persuasion_qwen_more.json   (optional)
    - data/analysis/trajectory_metrics.csv               (optional)

Outputs:
    - data/analysis/robustness_analysis.json
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_persuasion_results(filepath: Path) -> dict | None:
    """Load a single steering-persuasion JSON file."""
    if not filepath.exists():
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def extract_scenario_results(persuasion_data: dict[str, dict]) -> pd.DataFrame:
    """Flatten per-scenario, per-alpha, per-attack results into a DataFrame."""
    records: list[dict] = []
    for source_name, data in persuasion_data.items():
        model = data["model"]
        for alpha_str, scenarios in data["results_by_alpha"].items():
            alpha = float(alpha_str)
            for scenario in scenarios:
                for attack in scenario["attacks"]:
                    records.append(
                        {
                            "source": source_name,
                            "model": model,
                            "scenario_id": scenario["scenario_id"],
                            "dataset": scenario.get("dataset", "unknown"),
                            "stability": scenario["stability"],
                            "alpha": alpha,
                            "attack_type": attack["attack_type"],
                            "attack_name": attack["attack_name"],
                            "judgment_changed": attack["judgment_changed"],
                        }
                    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def run_statistical_tests(baseline_df: pd.DataFrame) -> dict:
    """Run all statistical tests on baseline (alpha=0) data.

    Returns a dictionary of test results.
    """
    results: dict = {}

    # ------------------------------------------------------------------
    # Chi-square test for independence
    # ------------------------------------------------------------------
    contingency = pd.crosstab(baseline_df["stability"], baseline_df["judgment_changed"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
    results["chi_square"] = {
        "statistic": float(chi2),
        "p_value": float(p_chi2),
        "dof": int(dof),
        "significant": bool(p_chi2 < 0.05),
    }
    print(f"Chi-square: chi2={chi2:.4f}, p={p_chi2:.2e}, sig={p_chi2 < 0.05}")

    # ------------------------------------------------------------------
    # Two-proportion z-test (H1: stable < unstable)
    # ------------------------------------------------------------------
    try:
        from statsmodels.stats.proportion import proportions_ztest

        stable_data = baseline_df[baseline_df["stability"] == "stable"]
        unstable_data = baseline_df[baseline_df["stability"] == "unstable"]
        count = np.array([stable_data["judgment_changed"].sum(), unstable_data["judgment_changed"].sum()])
        nobs = np.array([len(stable_data), len(unstable_data)])
        z_stat, p_ztest = proportions_ztest(count, nobs, alternative="smaller")
        results["two_proportion_ztest"] = {
            "z_statistic": float(z_stat),
            "p_value": float(p_ztest),
            "significant": bool(p_ztest < 0.05),
        }
        print(f"Z-test: z={z_stat:.4f}, p={p_ztest:.2e}, sig={p_ztest < 0.05}")
    except ImportError:
        print("statsmodels not installed -- skipping z-test")
        results["two_proportion_ztest"] = None

    # ------------------------------------------------------------------
    # Point-biserial correlation
    # ------------------------------------------------------------------
    stability_numeric = (baseline_df["stability"] == "stable").astype(int)
    corr_pb, p_pb = stats.pointbiserialr(stability_numeric, baseline_df["judgment_changed"].astype(int))
    results["point_biserial"] = {
        "correlation": float(corr_pb),
        "p_value": float(p_pb),
        "significant": bool(p_pb < 0.05),
    }
    print(f"Point-biserial: r={corr_pb:.4f}, p={p_pb:.2e}")

    # ------------------------------------------------------------------
    # Cohen's h
    # ------------------------------------------------------------------
    p_stable = baseline_df[baseline_df["stability"] == "stable"]["judgment_changed"].mean()
    p_unstable = baseline_df[baseline_df["stability"] == "unstable"]["judgment_changed"].mean()
    h = cohens_h(p_unstable, p_stable)
    effect_label = "large" if abs(h) >= 0.8 else "medium" if abs(h) >= 0.5 else "small"
    results["cohens_h"] = float(h)
    results["effect_size"] = effect_label
    print(f"Cohen's h: {h:.4f} ({effect_label})")

    return results


def per_attack_chi_square(baseline_df: pd.DataFrame) -> dict:
    """Chi-square test per attack type."""
    attack_tests: dict = {}
    for attack in baseline_df["attack_type"].unique():
        adf = baseline_df[baseline_df["attack_type"] == attack]
        contingency = pd.crosstab(adf["stability"], adf["judgment_changed"])
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        stable_flip = float(adf[adf["stability"] == "stable"]["judgment_changed"].mean())
        unstable_flip = float(adf[adf["stability"] == "unstable"]["judgment_changed"].mean())
        attack_tests[attack] = {
            "chi2": float(chi2),
            "p_value": float(p),
            "stable_flip": stable_flip,
            "unstable_flip": unstable_flip,
        }
        sig = "*" if p < 0.05 else ""
        print(f"  {attack}: chi2={chi2:.2f}, p={p:.3f} {sig}  (stable={stable_flip:.1%}, unstable={unstable_flip:.1%})")
    return attack_tests


def compute_steering_effect(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute flip-rate reduction from baseline for each (model, stability, alpha)."""
    rows: list[dict] = []
    for model in results_df["model"].unique():
        for stability in results_df["stability"].unique():
            subset = results_df[(results_df["model"] == model) & (results_df["stability"] == stability)]
            baseline = subset[subset["alpha"] == 0.0]["judgment_changed"].mean()
            if pd.isna(baseline):
                continue
            for alpha in sorted(subset["alpha"].unique()):
                if alpha == 0.0:
                    continue
                steered = subset[subset["alpha"] == alpha]["judgment_changed"].mean()
                if pd.isna(steered):
                    continue
                abs_red = baseline - steered
                rel_red = abs_red / baseline if baseline > 0 else 0
                rows.append(
                    {
                        "model": model,
                        "stability": stability,
                        "alpha": alpha,
                        "baseline_flip_rate": float(baseline),
                        "steered_flip_rate": float(steered),
                        "absolute_reduction": float(abs_red),
                        "relative_reduction": float(rel_red),
                    }
                )
    return pd.DataFrame(rows)


def mann_whitney_steering(steering_effect_df: pd.DataFrame) -> dict | None:
    """Mann-Whitney U: does steering help unstable more than stable?"""
    stable_red = steering_effect_df[steering_effect_df["stability"] == "stable"]["absolute_reduction"]
    unstable_red = steering_effect_df[steering_effect_df["stability"] == "unstable"]["absolute_reduction"]
    if len(stable_red) == 0 or len(unstable_red) == 0:
        return None
    u_stat, p_mw = stats.mannwhitneyu(unstable_red, stable_red, alternative="greater")
    print(f"Mann-Whitney U: U={u_stat:.2f}, p={p_mw:.4f}, sig={p_mw < 0.05}")
    return {"U_statistic": float(u_stat), "p_value": float(p_mw), "significant": bool(p_mw < 0.05)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness analysis of steering persuasion experiments.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        help="Repository root directory.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    analysis_dir = base_dir / "data" / "analysis"

    # ------------------------------------------------------------------
    # 1. Load all persuasion result files
    # ------------------------------------------------------------------
    file_map = {
        "llama": analysis_dir / "steering_persuasion_llama.json",
        "qwen": analysis_dir / "steering_persuasion_qwen.json",
        "llama_more": analysis_dir / "steering_persuasion_llama_more.json",
        "qwen_more": analysis_dir / "steering_persuasion_qwen_more.json",
    }

    persuasion_data: dict = {}
    for name, path in file_map.items():
        data = load_persuasion_results(path)
        if data is not None:
            persuasion_data[name] = data
            print(f"Loaded {name}: alpha values = {data['alpha_values']}")
        else:
            print(f"Missing: {name}")

    if not persuasion_data:
        print("No persuasion result files found. Nothing to analyze.")
        return

    results_df = extract_scenario_results(persuasion_data)
    print(f"\nTotal observations: {len(results_df)}")
    print(f"Models: {results_df['model'].unique().tolist()}")
    print(f"Alpha values: {sorted(results_df['alpha'].unique())}")

    # ------------------------------------------------------------------
    # 2. Baseline (alpha=0) analysis
    # ------------------------------------------------------------------
    baseline_df = results_df[results_df["alpha"] == 0.0].copy()
    print(f"\nBaseline (alpha=0) observations: {len(baseline_df)}")
    print(f"Overall baseline flip rate: {baseline_df['judgment_changed'].mean():.3f}")

    # Flip rates by stability
    flip_by_stability = baseline_df.groupby("stability").agg({"judgment_changed": ["mean", "sum", "count"]}).round(4)
    flip_by_stability.columns = ["flip_rate", "n_flips", "n_total"]

    print("\nFlip rates by stability:")
    print(flip_by_stability)

    stable_rate = float(flip_by_stability.loc["stable", "flip_rate"])
    unstable_rate = float(flip_by_stability.loc["unstable", "flip_rate"])
    susceptibility_ratio = unstable_rate / stable_rate if stable_rate > 0 else float("inf")
    print(f"Susceptibility ratio: {susceptibility_ratio:.2f}x")

    # By stability + attack type
    flip_by_stability_attack = baseline_df.groupby(["stability", "attack_type"]).agg({"judgment_changed": "mean"}).unstack()
    flip_by_stability_attack.columns = flip_by_stability_attack.columns.droplevel(0)

    # By model + stability
    flip_by_model_stability = baseline_df.groupby(["model", "stability"]).agg({"judgment_changed": ["mean", "count"]}).round(4)
    flip_by_model_stability.columns = ["flip_rate", "n_observations"]

    # Attack effectiveness ranking
    attack_effectiveness = (
        baseline_df.groupby("attack_type")
        .agg({"judgment_changed": ["mean", "sum", "count"]})
        .round(4)
    )
    attack_effectiveness.columns = ["flip_rate", "n_flips", "n_total"]
    attack_effectiveness = attack_effectiveness.sort_values("flip_rate", ascending=False)

    # ------------------------------------------------------------------
    # 3. Statistical tests
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS (baseline alpha=0)")
    print("=" * 60)
    stat_results = run_statistical_tests(baseline_df)

    print("\nPer-attack chi-square tests:")
    attack_tests = per_attack_chi_square(baseline_df)

    # ------------------------------------------------------------------
    # 4. Steering effect analysis
    # ------------------------------------------------------------------
    flip_by_alpha = (
        results_df.groupby(["model", "stability", "alpha"])
        .agg({"judgment_changed": "mean"})
        .round(4)
        .reset_index()
    )
    flip_by_alpha.columns = ["model", "stability", "alpha", "flip_rate"]

    steering_effect_df = compute_steering_effect(results_df)

    mw_result = None
    if len(steering_effect_df) > 0:
        print("\n" + "=" * 60)
        print("STEERING EFFECT ANALYSIS")
        print("=" * 60)
        stable_red = steering_effect_df[steering_effect_df["stability"] == "stable"]["absolute_reduction"]
        unstable_red = steering_effect_df[steering_effect_df["stability"] == "unstable"]["absolute_reduction"]
        print(f"Mean absolute reduction -- stable: {stable_red.mean():.4f}, unstable: {unstable_red.mean():.4f}")
        mw_result = mann_whitney_steering(steering_effect_df)

    # ------------------------------------------------------------------
    # 5. Save JSON
    # ------------------------------------------------------------------
    analysis_output: dict = {
        "data_source": "real_persuasion_experiments",
        "version": 2,
        "n_observations": len(baseline_df),
        "models": results_df["model"].unique().tolist(),
        "alpha_values": sorted(results_df["alpha"].unique().tolist()),
        "flip_rates_by_stability": {
            stab: {
                "flip_rate": float(row["flip_rate"]),
                "n_flips": int(row["n_flips"]),
                "n_total": int(row["n_total"]),
            }
            for stab, row in flip_by_stability.iterrows()
        },
        "susceptibility_ratio": float(susceptibility_ratio),
        "statistical_tests": stat_results,
        "attack_effectiveness": {
            attack: {
                "flip_rate": float(row["flip_rate"]),
                "n_flips": int(row["n_flips"]),
                "n_total": int(row["n_total"]),
            }
            for attack, row in attack_effectiveness.iterrows()
        },
        "attack_tests": attack_tests,
        "steering_effects": steering_effect_df.to_dict("records") if len(steering_effect_df) > 0 else [],
    }
    if mw_result is not None:
        analysis_output["mann_whitney_u"] = mw_result

    output_path = analysis_dir / "robustness_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\nSaved {output_path}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nStable flip rate:   {stable_rate:.1%}")
    print(f"Unstable flip rate: {unstable_rate:.1%}")
    print(f"Susceptibility ratio: {susceptibility_ratio:.2f}x")
    print(f"Chi-square p-value: {stat_results['chi_square']['p_value']:.2e}")
    if stat_results.get("two_proportion_ztest"):
        print(f"Z-test p-value: {stat_results['two_proportion_ztest']['p_value']:.2e}")
    print(f"Cohen's h: {stat_results['cohens_h']:.3f} ({stat_results['effect_size']})")

    most_effective = attack_effectiveness.index[0]
    print(f"\nMost effective attack: {most_effective} ({attack_effectiveness.loc[most_effective, 'flip_rate']:.1%})")

    if len(steering_effect_df) > 0:
        avg_red = steering_effect_df["absolute_reduction"].mean()
        print(f"Avg flip-rate reduction with steering: {avg_red:.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
