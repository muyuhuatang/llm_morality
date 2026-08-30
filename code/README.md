# Code

Nineteen scripts. Two of them reproduce the paper's numbers from the data in this repository and run
on a laptop. The other seventeen are the full experiment pipeline, preserved from the original
release because the methodology is useful on its own: they document how the trajectories,
activations, probes and steered generations were produced. Most need provider API credentials,
multi-GPU inference, or intermediate artifacts this repository does not redistribute.

## Runnable here

Plain Python, needing only `numpy`, `pandas` and `scipy`. They read only from `../data`, write only
into `../data`, and call no network service and load no model.

| Script | What it does |
|---|---|
| `compute_moral_representation_consistency.py` | Computes MRC and its three components for all 3,596 trajectories from the framework sequences, and writes `data/analysis/moral_representation_consistency.csv`. Asserts the invariant that a trajectory which changes framework cannot have a perfect stability component. |
| `reproduce_paper_numbers.py` | Recomputes every statistic the paper reports and prints each next to the file it came from. Nothing is hardcoded from the manuscript. The human-annotation section reports what the paper found and how to request the underlying responses, since those are not distributed here; if you have been granted them, drop the three CSVs into `data/annotation/` and the same script recomputes that section too. |

```bash
pip install -r ../requirements.txt
python3 compute_moral_representation_consistency.py
python3 reproduce_paper_numbers.py
```

## The experiment pipeline

Included for provenance. These are **not runnable end to end from this repository**, because several
of their inputs — raw provider responses, generated reasoning text, cached activations and
`probing_dataset.parquet` — are deliberately not redistributed (see the repository `README.md`).
Read them as methodology.

| Stage | Scripts |
|---|---|
| Sampling | `generate_pilot_samples.py` |
| Response collection *(provider API)* | `collect_responses_batch.py`, `collect_responses_parallel.py` |
| Framework attribution *(provider API)* | `score_attributions.py`, `analyze_gptoss_robustness.py`, `robustness_check.py` |
| Trajectory analysis | `compute_trajectory_metrics.py`, `bootstrap_confidence_intervals.py` |
| Probing *(GPU)* | `extract_activations.py`, `train_probes.py` |
| Steering *(GPU)* | `construct_steering_vectors.py`, `evaluate_steering.py`, `persuasion_attacks.py`, `robustness_analysis.py` |
| Coherence ratings *(provider API)* | `collect_llm_ratings.py` |
| MRC | `compute_mrc.py` *(superseded — see below)*, `validate_mrc.py` |

The two collection scripts use the **pilot** prompt described in the appendix, not the refined
theory-neutral prompt used for the main experiments.

Credentials are read from the environment (`OPENAI_API_KEY`, `TOGETHER_API_KEY`, `HF_TOKEN`); no key
is hardcoded in any script here.

## A note on the two MRC scripts

Both are kept, but they are not equivalent.

`compute_mrc.py` is the **superseded** implementation, retained so the published correction can be
checked against the code that caused it. It carries a warning header saying so. It resolved the
stability component through `probing_dataset.parquet`, which covers two of the three models on 500
of the 1,200 scenarios, and gave every trajectory absent from that file a hardcoded perfect score.
That left 1,987 of 3,596 trajectories with a nonzero drift rate and a stability of exactly 1.0,
which the metric's own definition forbids. Its numbers do not match the published paper.
**Use `compute_moral_representation_consistency.py` instead** — it computes stability from the
attributed framework sequence for all 3,596 trajectories, needs no probing data, and asserts the
invariant the old implementation violated.

`validate_mrc.py` is unaffected by that bug: it reads an MRC table rather than computing one, so
given the corrected file it does correct analysis. Only its original *outputs* were superseded, and
those (`mrc_validation.json` and `mrc_summary.json`, reporting r = 0.645 and overall MRC 0.460) are
not shipped, because no camera-ready claim rests on them.

## Also not here

- Visualisation code; the figure sources live with the manuscript.
- The human annotation responses. See *Requesting the human annotation data* in the repository
  `README.md`.
