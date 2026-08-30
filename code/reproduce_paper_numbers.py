#!/usr/bin/env python3
"""Recompute every statistic the paper reports, from the released files only.

Each line prints  KEY | recomputed value | the file it came from, so any claim in
the paper can be traced in one pass. Nothing is hardcoded from the manuscript.

Usage:  python3 reproduce_paper_numbers.py [--data ../data]
"""
import argparse, json, os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats


def head(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


def rec(key, val, src):
    print(f"{key:46s} {str(val):<40s} <- {src}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "..", "data"))
    a = ap.parse_args()
    A = lambda n: os.path.join(a.data, "analysis", n)
    N = lambda n: os.path.join(a.data, "annotation", n)

    head("Trajectory dynamics (Section 5)")
    tm = pd.read_csv(A("trajectory_metrics.csv"))
    rec("trajectories", len(tm), "trajectory_metrics.csv")
    fdr = tm.groupby("model").fdr.mean()
    rec("framework drift rate, range over models", f"{fdr.min():.3f}-{fdr.max():.3f}", "trajectory_metrics.csv")
    cons = tm.groupby("model").fdr.apply(lambda s: 100 * (s == 0).mean())
    rec("framework-consistent trajectories (%)", f"{cons.min():.1f}-{cons.max():.1f}", "trajectory_metrics.csv")
    ent = tm.groupby("model").entropy.mean()
    rec("mean entropy, range over models", f"{ent.min():.3f}-{ent.max():.3f}", "trajectory_metrics.csv")

    head("Probing (Section 6)")
    P = json.load(open(A("probe_results.json")))
    for m in ("llama", "qwen"):
        d = P["probe_results"][m]
        layers = np.array([l["layer"] for l in d["layer_results"]])
        kl = np.array([l["metrics"]["kl_mean"] for l in d["layer_results"]])
        prior = d["baselines"]["step_prior"]["kl_mean"]
        i = int(np.argmin(kl))
        rec(f"{m}: best layer / KL", f"layer {layers[i]} of {len(layers)-1}, KL {kl[i]:.3f}", "probe_results.json")
        rec(f"{m}: % below step-prior baseline", f"{100*(prior-kl[i])/prior:.2f}", "probe_results.json")
        rec(f"{m}: layers within 0.005 KL of best", int((kl <= kl[i] + 0.005).sum()), "probe_results.json")
        rec(f"{m}: layers with divergent probes (KL>1)", int((kl > 1).sum()), "probe_results.json")
    T = json.load(open(A("cross_model_transfer.json")))
    for k, v in T.items():
        rec(f"transfer {k}: best layer / KL", f"layer {v['best_layer']}, KL {v['best_kl']:.4f}", "cross_model_transfer.json")

    head("Steering (Section 7.1)")
    L = json.load(open(A("steering_layer_sweep.json")))
    for m in ("llama", "qwen"):
        per = [(r["layer"], 100 * r["mean_fdr_reduction"]) for g in L[m]["layer_groups"].values() for r in g["per_layer"]]
        best = max(per, key=lambda x: x[1])
        rec(f"{m}: best probe-space drift reduction", f"layer {best[0]}, {best[1]:.1f}%", "steering_layer_sweep.json")
        rec(f"{m}: sweep config", f"alpha={L[m]['alpha']}, {len(per)} layers, one direction from layer {L[m]['optimal_layer']}", "steering_layer_sweep.json")
    S = json.load(open(A("consistency_splits.json")))
    for m in ("llama", "qwen"):
        rec(f"{m}: consistent / drifting arm sizes", f"{len(S[m]['stable_samples'])} / {len(S[m]['unstable_samples'])}", "consistency_splits.json")

    head("Steered accuracy (Section 7.1, generation time)")
    pr = pd.read_csv(A("model_predictions.csv"))
    pr["ok"] = pr.is_correct.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    MODEL = {"llama": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "qwen": "Qwen/Qwen2.5-72B-Instruct-Turbo"}
    base = {}
    for m in ("llama", "qwen"):
        d = pr[pr.model == MODEL[m]].set_index("sample_id")
        c = d.reindex([x for x in S[m]["stable_samples"] if x in d.index]).ok
        g = d.reindex([x for x in S[m]["unstable_samples"] if x in d.index]).ok
        base[m] = 100 * (c.mean() - g.mean())
        rec(f"{m}: alpha=0 consistent / drifting / gap",
            f"{100*c.mean():.1f}% / {100*g.mean():.1f}% / {base[m]:+.1f} pp",
            "model_predictions.csv + consistency_splits.json")
    for fn in ("steered_accuracy_alpha4_llama_alpha5_qwen.json",
               "steered_accuracy_alpha10_llama.json",
               "steered_accuracy_alpha4_qwen.json"):
        if not os.path.exists(A(fn)):
            continue
        J = json.load(open(A(fn)))
        blocks = J if "by_stability" not in J else {J.get("model", "?"): J}
        for k, v in blocks.items():
            if not isinstance(v, dict) or "by_stability" not in v:
                continue
            bs = v["by_stability"]
            gap = 100 * (bs["stable"]["accuracy"] - bs["unstable"]["accuracy"])
            mk = "llama" if "llama" in str(k).lower() else "qwen"
            rec(f"{k}: gap / change vs baseline", f"{gap:+.1f} pp / {gap-base[mk]:+.1f} pp", fn)

    head("Persuasion (Section 7.2)")
    R = json.load(open(A("persuasion_robustness.json")))
    f = R["flip_rates_by_stability"]
    rec("consistent flip rate", f"{100*f['stable']['flip_rate']:.1f}% ({f['stable']['n_flips']}/{f['stable']['n_total']} pairs)", "persuasion_robustness.json")
    rec("drifting flip rate", f"{100*f['unstable']['flip_rate']:.1f}% ({f['unstable']['n_flips']}/{f['unstable']['n_total']} pairs)", "persuasion_robustness.json")
    rec("susceptibility ratio", f"{R['susceptibility_ratio']:.2f}x", "persuasion_robustness.json")
    st = R["statistical_tests"]
    rec("chi-square", f"{st['chi_square']['statistic']:.2f}, p={st['chi_square']['p_value']:.4f}", "persuasion_robustness.json")
    rec("Cohen's h", f"{st['cohens_h']:.3f} ({st['effect_size']})", "persuasion_robustness.json")
    for k, v in R["attack_tests"].items():
        rec(f"attack {k}", f"{v['stable_flip']:.2f} vs {v['unstable_flip']:.2f} ({v['unstable_flip']/v['stable_flip']:.2f}x, p={v['p_value']:.3f})", "persuasion_robustness.json")

    head("Moral Representation Consistency (Section 7.3)")
    mrc = pd.read_csv(A("moral_representation_consistency.csv"))
    rec("overall MRC", f"{mrc.mrc_score.mean():.3f} +/- {mrc.mrc_score.std():.3f}", "moral_representation_consistency.csv")
    for c, g in mrc.groupby("trajectory_category"):
        rec(f"MRC {c}", f"{g.mrc_score.mean():.3f} +/- {g.mrc_score.std():.3f} (n={len(g)})", "moral_representation_consistency.csv")
    r, _ = stats.pearsonr(mrc.fdr, mrc.mrc_score)
    rec("MRC vs drift rate", f"r={r:.3f}", "moral_representation_consistency.csv")

    rat = pd.read_csv(A("scoring_llm_coherence_ratings.csv"))
    # join on (sample_id, model): sample_id alone is a SCENARIO key shared by all three
    # models, so joining on it would silently mismatch most of the rows
    v = rat.merge(mrc, on=["sample_id", "model"], suffixes=("", "_m"))
    assert len(v) == len(rat), f"join lost rows: {len(v)} of {len(rat)}"
    rec("validation sample", f"{len(v)} of 180 rated items", "scoring_llm_coherence_ratings.csv")
    rec("  rated text was a single trajectory", f"{len(v)}; 61 interleaved + 6 duplicates dropped",
        "scoring_llm_coherence_ratings.csv")
    rec("  source label needed correcting", int((~v.source_label_was_correct).sum()),
        "scoring_llm_coherence_ratings.csv")
    for lab, col in [("MRC", "mrc_score"), ("Stability", "mrc_stability_component"),
                     ("Drift (1-FDR)", "mrc_drift_component"), ("Variance (1-entropy)", "mrc_variance_component")]:
        r, p = stats.pearsonr(v[col], v.median_rating)
        rec(f"validation r, {lab}", f"{r:.3f} (p={p:.2g})", "scoring_llm_coherence_ratings.csv")
    for c, g in v.groupby("trajectory_category"):
        rec(f"{c}: MRC / coherence", f"{g.mrc_score.mean():.3f} / {g.median_rating.mean():.1f}", "(joined)")
    print("\n  NOTE: this correlation is not independent validation. The Scoring LLM was shown each")
    print("  step's framework label and instructed that keeping one framework throughout scores")
    print("  80-100 while switching scores 20-50, which is the rule MRC's stability and drift")
    print("  components implement. Read it as a consistency check on the taxonomy, not as evidence")
    print("  that MRC captures human-perceived coherence. The human study, whose raters saw no")
    print("  framework labels, did not separate the categories at all.")

    head("Judgment accuracy (Section 3.1 and Appendix B)")
    models = sorted(pr.model.unique())
    rec("model order in the rows below", " / ".join(models), "model_predictions.csv")
    for ds, g in pr.groupby("dataset"):
        rec(f"accuracy {ds}", " / ".join(f"{100*g[g.model==m].ok.mean():.1f}" for m in models), "model_predictions.csv")
    sub = pd.read_csv(A("ethics_subtask_map.csv"))
    e = pr[pr.dataset == "ethics"].merge(sub, on="sample_id", how="left")
    rec("ETHICS sub-task composition", dict(Counter(e.drop_duplicates("sample_id").ethics_subtask.dropna())), "ethics_subtask_map.csv")
    for stk, gg in e.groupby("ethics_subtask"):
        rec(f"ETHICS {stk} accuracy", " / ".join(f"{100*gg[gg.model==m].ok.mean():.1f}" for m in models), "(joined)")
    print()

    head("Human annotation study (Appendix E)")
    if not os.path.exists(N("human_task1_framework_attribution.csv")):
        print("  The human annotation responses are not distributed with this repository.")
        print("  They were collected from identifiable individuals under Indiana University")
        print("  Bloomington IRB protocol 18813, whose consent statement commits us to keeping")
        print("  the response records private. They are shared on request for research use; see")
        print("  'Requesting the human annotation data' in the repository README.")
        print("  Statistics reported in the paper from this study:")
        print("    task 1: human-LLM cosine similarity          0.859 (n=89)")
        print("    task 2: transitions judged justified         94.4% (85/90)")
        print("    task 3: coherence by category                81.2 / 81.3 / 81.2")
        print("    task 3: ANOVA over categories                F=0.001, p=0.999")
        print("    task 3: ICC(3,1) on shared items             0.395 (n=20)")
        print("  Everything above this section is reproduced from the released files.")
        print()
        return
    t1 = pd.read_csv(N("human_task1_framework_attribution.csv"))
    H = [c for c in ["kantian_deontology", "benthamite_act_utilitarianism", "aristotelian_virtue_ethics",
                     "scanlonian_contractualism", "gauthierian_contractarianism"] if c in t1.columns]
    Lc = ["llm_" + c for c in H]
    d = t1.dropna(subset=Lc + H)
    hv, lv = d[H].values.astype(float), d[Lc].values.astype(float)
    cos = (hv * lv).sum(1) / (np.linalg.norm(hv, axis=1) * np.linalg.norm(lv, axis=1))
    rec("task 1: human-LLM cosine similarity", f"{cos.mean():.3f} (n={len(d)})", "human_task1_framework_attribution.csv")
    t2 = pd.read_csv(N("human_task2_transition_faithfulness.csv"))
    rec("task 2: transitions judged justified", f"{100*t2.transition_justified.mean():.1f}% ({int(t2.transition_justified.sum())}/{len(t2)})", "human_task2_transition_faithfulness.csv")
    t3 = pd.read_csv(N("human_task3_coherence.csv"))
    g3 = t3.groupby("trajectory_category").coherence
    rec("task 3: coherence by category", " / ".join(f"{k} {vv:.1f}" for k, vv in g3.mean().items()), "human_task3_coherence.csv")
    F, p = stats.f_oneway(*[t3[t3.trajectory_category == c].coherence for c in t3.trajectory_category.unique()])
    rec("task 3: ANOVA over categories", f"F={F:.3f}, p={p:.3f}", "human_task3_coherence.csv")
    core = t3[t3.item_type == "core"].pivot_table(index="item_id", columns="annotator", values="coherence").dropna()
    k, n = core.shape[1], core.shape[0]
    gm = core.values.mean()
    MSR = ((core.mean(axis=1) - gm) ** 2).sum() * k / (n - 1)
    MSE = (((core.values - core.mean(axis=1).values[:, None] - core.mean(axis=0).values[None, :] + gm) ** 2).sum()) / ((n - 1) * (k - 1))
    rec("task 3: ICC(3,1) on shared items", f"{(MSR-MSE)/(MSR+(k-1)*MSE):.3f} (n={n})", "human_task3_coherence.csv")
    rr = t3.merge(tm[["sample_id", "model", "num_transitions"]], on=["sample_id", "model"], how="left").dropna(subset=["num_transitions"])
    if len(rr) > 10:
        r, p = stats.pearsonr(rr.num_transitions, rr.coherence)
        rec("task 3: rating vs switch count", f"r={r:.3f} (p={p:.3f})", "(joined)")


if __name__ == "__main__":
    main()
