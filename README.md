# Understanding Moral Reasoning Trajectories in Large Language Models

Code and data for *Understanding Moral Reasoning Trajectories in Large Language Models: Toward
Probing-Based Explainability* (Findings of EMNLP 2026).

Fan Huang, Haewoon Kwak, Jisun An — Indiana University Bloomington

We model moral reasoning as a *trajectory*: the sequence of ethical frameworks a model invokes
across the steps of a structured deliberation. The paper measures how those trajectories behave for
three models across three benchmarks, with a six-model pilot and factorial comparison; shows that
framework identity is linearly decodable from the hidden states of the two open-weight models; tests
whether activation steering can move it; and proposes Moral Representation Consistency (MRC) as a
trajectory-level diagnostic.

## Reproduce every number in the paper

```bash
pip install -r requirements.txt
cd code
python3 reproduce_paper_numbers.py
```

That prints, next to the file it came from, every statistic this repository covers, so any of those
claims can be traced in one pass. Nothing is hardcoded from the manuscript.

### What this repository does and does not reproduce

**Reproduced from the released files.** The trajectory metrics of Section 5 (drift rate, entropy,
faithfulness); the per-layer probing results and cross-model transfer of Section 6; both steering
configurations of Section 7.1, the probe-space layer sweep and the generation-time steered accuracy;
the persuasion results of Section 7.2; MRC and its components in Section 7.3, and its comparison against the Scoring LLM (see the caveat in `data/README.md`); the
per-dataset and per-ETHICS-sub-task accuracy tables; the per-step framework attribution means; and
the pilot step-count distribution.

**Not reproduced here.** The 2x2 factorial experiment of Section 4; the distribution of the dominant
framework's attribution score over the 14,384 step-level allocations (the per-step *means* are in
`framework_attribution_by_step.csv`, but the distribution needs the raw per-step allocations); the
framework transition matrices and trajectory archetypes; the alternative-taxonomy robustness check;
and the bootstrap confidence intervals for the consistency-accuracy relationship. These rest on
intermediate artifacts that are either raw Scoring LLM output or large enough that shipping them
would defeat the point of a minimal release. The human annotation study is withheld for the separate reason given below. If you need any
of these to check a specific claim, ask and we will send the derived table. `compute_moral_representation_consistency.py`
regenerates `data/analysis/moral_representation_consistency.csv` from the trajectory metrics.

## Layout

```
code/
  compute_moral_representation_consistency.py   MRC from the framework sequences
  reproduce_paper_numbers.py                    every reported statistic, with provenance
data/
  analysis/     derived statistics: trajectory metrics, probe results, steering sweeps,
                persuasion outcomes, MRC, model predictions
```

The human annotation responses are **not** in this repository. They are available on request; see
*Requesting the human annotation data* below.

## What is here, and what is deliberately not

This is a minimal release: a file is included only if a claim in the paper depends on it.

**Included.** Derived, tabular data. Trajectory-level metrics for all 3,596 trajectories; per-layer
probe results for both open-weight models; the probe-space steering sweep and the generation-time
steered-accuracy summaries; persuasion-attack outcomes; MRC scores; and parsed model predictions
with correctness.

**Not included, by choice.**

- **Raw provider output.** No API responses, no generated reasoning text, no Scoring LLM
  justifications. Only the numbers derived from them.
- **The benchmark text.** Moral Stories, ETHICS and Social Chemistry 101 are not redistributed.
  Rows are keyed by our own `sample_id`, so the release joins to those datasets without copying
  them.
- **Credentials and account data.** No API keys, no `.env`, no usage or spend records.
- **The human annotation responses.** Withheld by default and shared on request; see below.
- **Model weights, activations, and probe checkpoints.** Large binaries; the derived probe metrics
  are here instead.

## Human annotation study

Conducted under Indiana University Bloomington IRB protocol 18813 (initial
approval 25 April 2023, PI Jisun An). Each annotator gave informed consent under that protocol and
could withdraw at any time.

Three graduate research assistants each completed 30 items per task across three tasks: step-level
framework attribution, transition faithfulness, and whole-trajectory coherence. Twenty items per
task are shared across all three annotators, so inter-annotator agreement is measurable. The
statistics the paper reports from this study are the human-LLM cosine similarity of 0.859 on
framework attribution, the 94.4% of framework transitions judged logically justified, and the
coherence ratings of 81.2, 81.3 and 81.2 across the three trajectory categories with
ICC(3,1) = 0.395.

### Requesting the human annotation data

**The annotation responses are not included in this repository.** They were collected from
identifiable individuals under an IRB protocol whose informed consent statement commits us to
keeping the response records within our private research documents. Releasing them openly is not
something that consent covers, so we share them on request instead, for research use, after a
case-by-case check.

To request access, email **huangfan@acm.org** with the subject line
`[Moral Reasoning Trajectories] human annotation data request`, and include:

1. **Who you are.** Your full name, your institutional affiliation, your role (faculty, postdoc,
   PhD student, industry researcher), and an institutional email address we can reply to.
2. **Why you want it.** What you intend to do with the data, concretely enough that we can judge
   whether the request is research use. If it supports a specific paper or project, say which.
3. **What you intend to publish.** Whether any part of the data, or figures derived from it, would
   appear in a public artifact.
4. **How you will handle it.** Confirmation that you will not attempt to re-identify the
   annotators, will not redistribute the data, and will delete it when the stated purpose is
   complete.

We will reply either way. If we can share, you receive the responses in the pseudonymised form the
paper analyses: annotators appear as `annotator_1` to `annotator_3`, and no demographic or other
identifying information about them exists in the files, because none was ever collected.

Everything in `data/analysis/` is released without restriction, and the request process applies only
to the human responses. See *What this repository does and does not reproduce* above for exactly
which results those files cover.

## Citation

Repository: <https://github.com/muyuhuatang/llm_morality>

```bibtex
@inproceedings{huang2026moral,
  title     = {Understanding Moral Reasoning Trajectories in Large Language Models:
               Toward Probing-Based Explainability},
  author    = {Huang, Fan and Kwak, Haewoon and An, Jisun},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

- **`code/`** — Apache License 2.0.
- **`data/`** — Creative Commons Attribution 4.0 International (CC BY 4.0).

The three source benchmarks (Moral Stories, ETHICS, Social Chemistry 101) are **not** redistributed
here and remain under their own licenses. This repository contains only derived statistics keyed by
identifiers we assigned, so it can be used alongside those datasets without relicensing them.
