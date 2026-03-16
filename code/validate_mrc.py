"""
Validate MRC against LLM coherence ratings and analyse MRC vs persuasion resistance.

Validates MRC scores against LLM coherence ratings (correlation analysis,
category-wise breakdown, component analysis) and examines how MRC relates
to persuasion resistance (flip rates, steering effectiveness, logistic regression).

Analyses performed:
    1. Primary validation  -- Pearson / Spearman correlation of MRC with LLM ratings
    2. Category-wise        -- MRC and rating means for single_framework / bounce / high_entropy
    3. Component analysis   -- which MRC component (variance, drift, stability) correlates best
    4. Baseline comparison  -- MRC vs simpler metrics (1-FDR, 1-entropy)
    5. MRC vs persuasion    -- flip rates by MRC level, steering effectiveness
    6. Logistic regression  -- MRC (stability proxy) predicting flip rate

Inputs:
    - data/analysis/mrc_scores.csv
    - data/annotation/llm_annotations.json
    - data/analysis/steering_persuasion_{llama,qwen}.json
    - data/analysis/steering_persuasion_{llama,qwen}_more.json

Outputs:
    - data/analysis/mrc_validation.json
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ===================================================================
#  PART A -- MRC vs LLM-rating validation
# ===================================================================

def load_llm_annotations(json_path: Path) -> pd.DataFrame:
    """Parse llm_annotations.json into a flat DataFrame."""
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for ann in data["annotations"]:
        rating = ann.get("median_rating") or ann.get("mean_rating")
        rows.append({
            "sample_id": ann["sample_id"],
            "llm_rating": rating,
            "rating_std": ann.get("std_rating"),
            "n_ratings": len(ann.get("ratings", [])),
            "ann_mrc_variance": ann.get("mrc_variance_component"),
            "ann_mrc_drift": ann.get("mrc_drift_component"),
            "ann_mrc_stability": ann.get("mrc_stability_component"),
        })
    return pd.DataFrame(rows)


def run_correlation_validation(
    validation_df: pd.DataFrame,
    results_dir: Path,
) -> dict:
    """Correlate MRC with LLM ratings, run component / baseline analysis."""

    out: dict = {}

    # -- Primary correlation ---------------------------------------------------
    pearson_r, pearson_p = stats.pearsonr(
        validation_df["mrc_score"], validation_df["llm_rating"]
    )
    spearman_r, spearman_p = stats.spearmanr(
        validation_df["mrc_score"], validation_df["llm_rating"]
    )
    print(f"\nPrimary validation (n={len(validation_df)}):")
    print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.4e})")
    print(f"  Spearman rho = {spearman_r:.4f}  (p = {spearman_p:.4e})")

    out["primary_validation"] = {
        "n_samples": len(validation_df),
        "pearson_correlation": {"r": float(pearson_r), "p": float(pearson_p),
                                "significant": bool(pearson_p < 0.05)},
        "spearman_correlation": {"rho": float(spearman_r), "p": float(spearman_p),
                                 "significant": bool(spearman_p < 0.05)},
    }

    # -- Category-wise ---------------------------------------------------------
    cat_stats = (
        validation_df.groupby("trajectory_category")
        .agg(
            mrc_mean=("mrc_score", "mean"),
            mrc_std=("mrc_score", "std"),
            rating_mean=("llm_rating", "mean"),
            rating_std=("llm_rating", "std"),
            n_samples=("sample_id", "count"),
        )
        .round(4)
    )
    print("\nCategory-wise statistics:")
    print(cat_stats)

    out["category_validation"] = {
        cat: {k: float(v) if k != "n_samples" else int(v) for k, v in row.items()}
        for cat, row in cat_stats.iterrows()
    }

    # t-test: single_framework vs high_entropy
    if {"single_framework", "high_entropy"}.issubset(
        validation_df["trajectory_category"].unique()
    ):
        sf = validation_df.loc[
            validation_df["trajectory_category"] == "single_framework", "llm_rating"
        ]
        he = validation_df.loc[
            validation_df["trajectory_category"] == "high_entropy", "llm_rating"
        ]
        t_stat, t_p = stats.ttest_ind(sf, he)
        print(f"\n  T-test single_framework vs high_entropy: t={t_stat:.4f}, p={t_p:.4e}")

    # -- Component analysis ----------------------------------------------------
    components = [
        ("mrc_variance_component", "Variance (1 - entropy)"),
        ("mrc_drift_component", "Drift (1 - FDR)"),
        ("mrc_stability_component", "Stability"),
    ]
    comp_results: dict = {}
    print("\nComponent correlations with LLM rating:")
    for col, name in components:
        if col not in validation_df.columns:
            continue
        v = validation_df[[col, "llm_rating"]].dropna()
        if len(v) < 10:
            continue
        pr, pp = stats.pearsonr(v[col], v["llm_rating"])
        sr, sp = stats.spearmanr(v[col], v["llm_rating"])
        sig = "*" if pp < 0.05 else ""
        print(f"  {name:<30}  r={pr:.4f} (p={pp:.4e}) rho={sr:.4f}{sig}")
        comp_results[col] = {
            "name": name, "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
        }
    out["component_analysis"] = comp_results

    # -- Baseline comparison ---------------------------------------------------
    baseline_results: dict = {"mrc_score": {"r": float(pearson_r), "p": float(pearson_p)}}
    for metric in ["fdr", "entropy"]:
        if metric not in validation_df.columns:
            continue
        v = validation_df[[metric, "llm_rating"]].dropna()
        if len(v) < 10:
            continue
        r, p = stats.pearsonr(-v[metric], v["llm_rating"])
        baseline_results[metric] = {"r": float(r), "p": float(p)}
    out["baseline_comparison"] = baseline_results
    print("\nBaseline comparison (corr with LLM rating):")
    for m, res in baseline_results.items():
        label = f"1 - {m}" if m in ("fdr", "entropy") else "MRC Score"
        print(f"  {label:<20}  r={res['r']:.4f} (p={res['p']:.4e})")

    # -- Expected vs actual ----------------------------------------------------
    out["expected_vs_actual"] = {}
    for cat, expected_mrc in [("single_framework", "high (~0.7)"),
                               ("high_entropy", "low (~0.4)")]:
        if cat in cat_stats.index:
            out["expected_vs_actual"][cat] = {
                "expected_mrc": expected_mrc,
                "actual_mrc": float(cat_stats.loc[cat, "mrc_mean"]),
            }

    return out


# ===================================================================
#  PART B -- MRC vs Persuasion Resistance
# ===================================================================

def load_persuasion_results(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def extract_scenario_results(persuasion_data: dict[str, dict]) -> pd.DataFrame:
    """Flatten per-scenario, per-alpha, per-attack results."""
    rows = []
    for source_name, data in persuasion_data.items():
        model = data["model"]
        for alpha_str, scenarios in data["results_by_alpha"].items():
            alpha = float(alpha_str)
            for sc in scenarios:
                for attack in sc["attacks"]:
                    rows.append({
                        "source": source_name,
                        "model": model,
                        "scenario_id": sc["scenario_id"],
                        "dataset": sc.get("dataset", "unknown"),
                        "stability": sc["stability"],
                        "alpha": alpha,
                        "attack_type": attack["attack_type"],
                        "attack_name": attack["attack_name"],
                        "judgment_changed": attack["judgment_changed"],
                    })
    return pd.DataFrame(rows)


def compute_steering_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Compute flip-rate reduction from alpha=0 baseline per model x stability."""
    rows = []
    for model in df["model"].unique():
        for stab in df["stability"].unique():
            sub = df[(df["model"] == model) & (df["stability"] == stab)]
            baseline = sub.loc[sub["alpha"] == 0.0, "judgment_changed"].mean()
            if pd.isna(baseline):
                continue
            for alpha in sorted(sub["alpha"].unique()):
                if alpha == 0.0:
                    continue
                steered = sub.loc[sub["alpha"] == alpha, "judgment_changed"].mean()
                if pd.isna(steered):
                    continue
                rows.append({
                    "model": model,
                    "stability": stab,
                    "alpha": alpha,
                    "baseline_flip_rate": baseline,
                    "steered_flip_rate": steered,
                    "absolute_reduction": baseline - steered,
                    "relative_reduction": (baseline - steered) / baseline if baseline > 0 else 0,
                })
    return pd.DataFrame(rows)


def run_persuasion_analysis(
    results_df: pd.DataFrame,
    mrc_df: pd.DataFrame | None,
    results_dir: Path,
) -> dict:
    """Full persuasion-resistance analysis; returns JSON-serialisable dict."""

    # Merge MRC if available
    if mrc_df is not None:
        results_df = results_df.merge(
            mrc_df[["sample_id", "mrc_score", "fdr", "entropy", "trajectory_category"]].rename(
                columns={"sample_id": "scenario_id"}
            ),
            on="scenario_id",
            how="left",
        )

    results_df["mrc_numeric"] = results_df["stability"].map({"stable": 1, "unstable": 0})

    # Flip rates by stability + alpha
    flip_by = (
        results_df.groupby(["model", "stability", "alpha"])
        .agg(n_flips=("judgment_changed", "sum"),
             n_total=("judgment_changed", "count"),
             flip_rate=("judgment_changed", "mean"))
        .reset_index()
        .round(4)
    )
    print("\nFlip rates by model / stability / alpha:")
    print(flip_by.to_string(index=False))

    # Baseline comparison
    baseline = results_df[results_df["alpha"] == 0.0]
    corr_pb, p_pb = stats.pointbiserialr(
        baseline["mrc_numeric"], baseline["judgment_changed"].astype(int)
    )
    print(f"\nPoint-biserial (MRC vs flip @alpha=0): r={corr_pb:.4f}, p={p_pb:.4e}")

    # Chi-square
    ct = pd.crosstab(baseline["stability"], baseline["judgment_changed"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)
    print(f"Chi-square: chi2={chi2:.4f}, df={dof}, p={p_chi2:.4e}")

    # Two-proportion z-test
    stable_bl = baseline[baseline["stability"] == "stable"]
    unstable_bl = baseline[baseline["stability"] == "unstable"]
    p_stable = stable_bl["judgment_changed"].mean()
    p_unstable = unstable_bl["judgment_changed"].mean()

    # Steering effect
    steering_df = compute_steering_effect(results_df)

    # Logistic regression: flip ~ mrc_high * alpha
    logit_results: dict = {}
    try:
        from statsmodels.formula.api import logit as logit_model

        reg_data = results_df.copy()
        reg_data["mrc_high"] = (reg_data["stability"] == "stable").astype(int)
        reg_data["flip"] = reg_data["judgment_changed"].astype(int)
        model_fit = logit_model("flip ~ mrc_high * alpha", data=reg_data).fit(disp=0)
        print(f"\nLogistic regression (Pseudo R2={model_fit.prsquared:.4f}):")
        for var in model_fit.params.index:
            coef = model_fit.params[var]
            pval = model_fit.pvalues[var]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {var:20s}: {coef:8.4f} (p={pval:.4f}) {sig}")
            logit_results[var] = {"coef": float(coef), "p": float(pval)}
    except ImportError:
        print("  (statsmodels not installed -- skipping logistic regression)")

    # Cohen's h
    h = 2 * (np.arcsin(np.sqrt(p_unstable)) - np.arcsin(np.sqrt(p_stable)))

    # Build output dict
    out: dict = {
        "data_summary": {
            "n_observations": len(results_df),
            "models": results_df["model"].unique().tolist(),
            "alpha_values": sorted(results_df["alpha"].unique().tolist()),
            "attack_types": results_df["attack_type"].unique().tolist(),
        },
        "flip_rates_by_stability": {
            f"{r['model']}_{r['stability']}_alpha{r['alpha']}": {
                "model": r["model"], "stability": r["stability"],
                "alpha": float(r["alpha"]), "flip_rate": float(r["flip_rate"]),
                "n_flips": int(r["n_flips"]), "n_total": int(r["n_total"]),
            }
            for _, r in flip_by.iterrows()
        },
        "statistical_tests": {
            "point_biserial": {"r": float(corr_pb), "p": float(p_pb),
                               "significant": bool(p_pb < 0.05)},
            "chi_square": {"chi2": float(chi2), "df": int(dof),
                           "p": float(p_chi2), "significant": bool(p_chi2 < 0.05)},
            "stable_flip_rate": float(p_stable),
            "unstable_flip_rate": float(p_unstable),
            "cohens_h": float(h),
            "logistic_regression": logit_results,
        },
        "steering_effects": {
            f"{r['model']}_{r['stability']}_alpha{r['alpha']}": {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in r.items()
            }
            for _, r in steering_df.iterrows()
        },
    }

    # Key findings
    ratio = p_unstable / p_stable if p_stable > 0 else float("inf")
    avg_st_red = steering_df[steering_df["stability"] == "stable"]["absolute_reduction"].mean()
    avg_un_red = steering_df[steering_df["stability"] == "unstable"]["absolute_reduction"].mean()
    out["key_findings"] = [
        f"High-MRC scenarios show {(1 - p_stable / p_unstable) * 100:.1f}% lower flip rates",
        f"Susceptibility ratio (Low/High MRC): {ratio:.2f}x",
        f"Chi-square significant: {p_chi2 < 0.05} (chi2={chi2:.2f})",
        f"Steering avg reduction -- High MRC: {avg_st_red:.1%}, Low MRC: {avg_un_red:.1%}",
        f"Cohen's h = {h:.3f}",
    ]

    return out



# ===================================================================
#  CLI + main
# ===================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate MRC against LLM ratings and analyse persuasion resistance.",
    )
    ap.add_argument(
        "--base-dir", type=str, default=".",
        help="Project root (contains data/ and results/ sub-trees).",
    )
    ap.add_argument(
        "--skip-persuasion", action="store_true",
        help="Skip persuasion-resistance analysis (Part B).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    analysis_dir = base / "data" / "analysis"
    annotation_dir = base / "data" / "annotation"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ── Part A: MRC vs LLM rating validation ──────────────────────
    mrc_csv = analysis_dir / "mrc_scores.csv"
    ann_json = annotation_dir / "llm_annotations.json"

    print("=" * 70)
    print("PART A: MRC vs LLM Rating Validation")
    print("=" * 70)

    mrc_df = pd.read_csv(mrc_csv)
    print(f"Loaded MRC scores: {len(mrc_df)}")
    ann_df = load_llm_annotations(ann_json)
    print(f"Loaded LLM annotations: {ann_df['llm_rating'].notna().sum()} valid")

    validation_df = mrc_df.merge(ann_df, on="sample_id", how="inner")
    validation_df = validation_df.dropna(subset=["mrc_score", "llm_rating"])
    print(f"Validation set: {len(validation_df)}")

    validation_out = run_correlation_validation(validation_df, analysis_dir)

    # ── Part B: MRC vs Persuasion ──────────────────────────────────
    persuasion_out: dict = {}
    if not args.skip_persuasion:
        print("\n" + "=" * 70)
        print("PART B: MRC vs Persuasion Resistance")
        print("=" * 70)

        persuasion_files = {
            "llama": analysis_dir / "steering_persuasion_llama.json",
            "qwen": analysis_dir / "steering_persuasion_qwen.json",
            "llama_more": analysis_dir / "steering_persuasion_llama_more.json",
            "qwen_more": analysis_dir / "steering_persuasion_qwen_more.json",
        }
        persuasion_data: dict = {}
        for name, path in persuasion_files.items():
            if path.exists():
                persuasion_data[name] = load_persuasion_results(path)
                print(f"  Loaded {name}: {len(persuasion_data[name]['alpha_values'])} alphas")
            else:
                print(f"  Missing: {name} -- skipping")

        if persuasion_data:
            results_df = extract_scenario_results(persuasion_data)
            print(f"\nTotal persuasion observations: {len(results_df)}")
            persuasion_out = run_persuasion_analysis(
                results_df, mrc_df, analysis_dir,
            )

    # ── Save combined results ──────────────────────────────────────
    combined = {**validation_out, "persuasion_analysis": persuasion_out}
    out_json = analysis_dir / "mrc_validation.json"
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved -> {out_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
