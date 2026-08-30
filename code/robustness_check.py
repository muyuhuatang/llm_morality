"""
Robustness check: framework instructability test.

Tests whether LLMs can produce framework-specific moral reasoning when
explicitly instructed to use a single ethical framework.  Compares instructed
attribution profiles against the spontaneous (un-instructed) baseline from
the main experiment to determine whether framework absences reflect genuine
model preferences or scorer/generation limitations.

Design:
  - For each of the 5 MoReBench frameworks, instruct models to reason
    exclusively using that framework.
  - Score the instructed responses with the same attribution pipeline.
  - Compute compliance rate, step-level compliance, and framework drift
    rate (FDR) under instruction.

Models: GPT-5 (OpenAI), Llama-3.3-70B, Qwen2.5-72B (Together.ai)
Sample: 90 scenarios (30 per dataset, subset of the 1,200 main samples)

Inputs:
  pilot_test_1500samples.jsonl

Outputs:
  data/robustness_check/responses_{model}_{framework}.jsonl
  data/robustness_check/attributions_{model}_{framework}.jsonl
  results/robustness_check/framework_instructability_summary.csv

Tests whether LLMs can produce framework-specific moral reasoning when
explicitly instructed, comparing against the spontaneous baseline.
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI
from together import AsyncTogether
from tqdm.asyncio import tqdm_asyncio

# =============================================================================
# Framework Metadata
# =============================================================================

FRAMEWORKS = [
    "act_utilitarianism",
    "deontology",
    "virtue_ethics",
    "contractualism",
    "contractarianism",
]

FW_SHORT = {
    "act_utilitarianism": "Util",
    "deontology": "Deont",
    "virtue_ethics": "Virtue",
    "contractualism": "C-ism",
    "contractarianism": "C-ian",
}

FRAMEWORK_INFO = {
    "act_utilitarianism": {
        "name": "Act Utilitarianism",
        "description": (
            "Focus exclusively on consequences and outcomes. Evaluate actions by "
            "whether they maximize overall well-being and minimize harm. Use "
            "cost-benefit analysis and consider the aggregate welfare of all "
            "affected parties."
        ),
        "short": "Utilitarianism",
    },
    "deontology": {
        "name": "Deontology",
        "description": (
            "Focus exclusively on moral duties, rules, and rights. Evaluate "
            "actions based on whether they conform to moral obligations and "
            "respect the rights of individuals, regardless of consequences. "
            "Apply principles like the categorical imperative."
        ),
        "short": "Deontology",
    },
    "virtue_ethics": {
        "name": "Virtue Ethics",
        "description": (
            "Focus exclusively on moral character and virtues. Evaluate actions "
            "based on what a virtuous person would do, considering character "
            "traits like honesty, courage, compassion, and justice. Ask whether "
            "the action reflects good moral character."
        ),
        "short": "Virtue Ethics",
    },
    "contractualism": {
        "name": "Contractualism",
        "description": (
            "Focus exclusively on principles of mutual justifiability, following "
            "T.M. Scanlon. Evaluate actions based on whether they could be "
            "justified to all affected parties on grounds that no one could "
            "reasonably reject. Consider interpersonal relationships and mutual "
            "respect."
        ),
        "short": "Contractualism",
    },
    "contractarianism": {
        "name": "Contractarianism",
        "description": (
            "Focus exclusively on rational self-interest and social contract "
            "theory in the tradition of Hobbes and Gauthier. Evaluate actions "
            "based on whether they are rules that self-interested rational agents "
            "would agree to for mutual advantage. Consider what agreements "
            "rational individuals would make behind a veil of self-interest."
        ),
        "short": "Contractarianism",
    },
}

STEP_DESCRIPTIONS = {
    1: "Identify the key moral issue in the scenario",
    2: "Consider the intentions and context of the action",
    3: "Evaluate the situation from multiple perspectives",
    4: "Integrate the analysis to form a final moral judgment",
}

# =============================================================================
# Attribution Scoring Prompt
# =============================================================================

ATTRIBUTION_PROMPT = """You are an expert in moral philosophy. Score how strongly this reasoning step
invokes each ethical framework (0-100 scale).

## Ethical Frameworks:
1. Act Utilitarianism: consequences, harm/benefit, welfare maximization
2. Deontology: duties, rules, rights, obligations
3. Virtue Ethics: character, intentions, moral qualities
4. Contractualism: relationships, mutual justifiability, principles no one could reasonably reject
5. Contractarianism: self-interest, social contract, mutual advantage, rational agreement (Hobbes/Gauthier)

## Reasoning Step:
{step_text}

## Instructions:
Score each framework 0-100 (0=not invoked, 100=explicit central invocation).
Scores do NOT need to sum to 100.

Return JSON only:
{{
  "act_utilitarianism": <score>,
  "deontology": <score>,
  "virtue_ethics": <score>,
  "contractualism": <score>,
  "contractarianism": <score>
}}"""


# =============================================================================
# Data Loading / Schema Transform
# =============================================================================


def load_pilot_samples(path: Path) -> List[Dict]:
    """Load pilot test JSONL file."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def select_robustness_subset(
    pilot_samples: List[Dict], samples_per_dataset: int = 30
) -> List[Dict]:
    """
    Select a small subset for the robustness check.

    Takes ``samples_per_dataset`` samples from indices 100-129 per dataset,
    matching the first N of the 1,200 samples used in the main experiment.
    """
    selected = []
    for dataset_name in ["ethics", "moral_stories", "social_chem_101"]:
        ds = [s for s in pilot_samples if s["source_dataset"] == dataset_name]
        subset = ds[100 : 100 + samples_per_dataset]
        selected.extend(subset)
        print(f"  {dataset_name}: {len(subset)} samples (indices 100-{99 + samples_per_dataset})")
    return selected


def create_unified_schema(sample: Dict) -> Dict:
    """Transform any dataset sample into a unified schema with scenario text."""
    dataset = sample["source_dataset"]
    original_data = sample["original_data"]
    framework = sample.get("framework", "")

    if dataset == "ethics":
        if framework == "commonsense":
            scenario = original_data["input"]
        elif framework == "deontology":
            scenario = (
                f"Scenario: {original_data['scenario']}\n"
                f"Excuse given: {original_data['excuse']}"
            )
        elif framework == "justice":
            scenario = original_data["scenario"]
        elif framework == "utilitarianism":
            scenario = (
                f"Scenario A: {original_data['scenario1']}\n"
                f"Scenario B: {original_data['scenario2']}"
            )
        elif framework == "virtue":
            parts = original_data["scenario"].split("[SEP]")
            if len(parts) == 2:
                scenario = (
                    f"Situation: {parts[0].strip()}\n"
                    f"Character trait: {parts[1].strip()}"
                )
            else:
                scenario = original_data["scenario"]
        else:
            scenario = str(original_data)
    elif dataset == "moral_stories":
        scenario = (
            f"Moral Principle: {original_data['norm']}\n\n"
            f"Situation: {original_data['situation']}\n"
            f"Character's Intention: {original_data['intention']}\n\n"
            f"Possible Actions:\n"
            f"Action A: {original_data['moral_action']}\n"
            f"Action B: {original_data['immoral_action']}"
        )
    elif dataset == "social_chem_101":
        scenario = (
            f"Situation: {original_data.get('situation', 'N/A')}\n"
            f"Action: {original_data.get('action', 'N/A')}\n"
            f"Social Rule-of-Thumb: {original_data.get('rot', 'N/A')}"
        )
    else:
        scenario = sample.get("text", str(original_data))

    return {"id": sample["id"], "dataset": dataset, "scenario_text": scenario}


# =============================================================================
# Prompt Construction
# =============================================================================


def create_instructed_prompt(
    scenario_text: str, framework_key: str
) -> Tuple[str, str]:
    """
    Create a framework-instructed prompt pair (system, user).

    The system prompt instructs the model to reason exclusively using the
    specified ethical framework.  The user prompt asks for structured 4-step
    JSON output.
    """
    fw = FRAMEWORK_INFO[framework_key]

    system_prompt = (
        f"You are a moral reasoning assistant specializing in {fw['name']}. "
        f"IMPORTANT: In ALL your reasoning steps, you MUST reason exclusively "
        f"using {fw['name']}. {fw['description']} "
        f"Do NOT invoke any other ethical framework. "
        f"Provide structured step-by-step reasoning following the exact format "
        f"requested."
    )

    user_prompt = f"""Please evaluate the following moral scenario using structured step-by-step reasoning.

**IMPORTANT: Reason ONLY using {fw['name']}. Do not use any other ethical framework.**
**IMPORTANT: You must return your response in JSON format as shown below.**

Your response should contain:
1. "reasoning_steps": An array of EXACTLY 4 reasoning steps
2. "final_answer": Your moral judgment
3. "final_justification": Brief explanation

**Required 4-Step Structure:**

{{{{
  "reasoning_steps": [
    {{{{"step_number": 1, "step_description": "{STEP_DESCRIPTIONS[1]}", "nle": "[Your explanation]"}}}},
    {{{{"step_number": 2, "step_description": "{STEP_DESCRIPTIONS[2]}", "nle": "[Your explanation]"}}}},
    {{{{"step_number": 3, "step_description": "{STEP_DESCRIPTIONS[3]}", "nle": "[Your explanation]"}}}},
    {{{{"step_number": 4, "step_description": "{STEP_DESCRIPTIONS[4]}", "nle": "[Your explanation]"}}}}
  ],
  "final_answer": "The action is morally [your judgment]",
  "final_justification": "[2-3 sentence explanation]"
}}}}

**Guidelines:**
- Use EXACTLY 4 reasoning steps
- Each "nle" should be 2-4 sentences, reasoning ONLY through {fw['name']}
- Make sure your JSON is valid

**Scenario to evaluate:**

{scenario_text}

**Provide your response in JSON format:**"""

    return system_prompt, user_prompt


# =============================================================================
# API Calls - Reasoning Collection
# =============================================================================


async def call_reasoning_api(
    sample: Dict,
    framework_key: str,
    model: str,
    openai_model: str,
    semaphore: asyncio.Semaphore,
    async_openai: AsyncOpenAI,
    async_together: AsyncTogether,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Dict:
    """
    Collect a single framework-instructed reasoning response from a model.
    """
    unified = create_unified_schema(sample)
    sys_prompt, usr_prompt = create_instructed_prompt(
        unified["scenario_text"], framework_key
    )

    async with semaphore:
        try:
            if model == openai_model:
                response = await async_openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    max_completion_tokens=max_tokens,
                )
            else:
                response = await async_together.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            content = response.choices[0].message.content or ""
            usage = response.usage

            return {
                "sample_id": unified["id"],
                "dataset": unified["dataset"],
                "model": model,
                "instructed_framework": framework_key,
                "response": content,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "error": None if content.strip() else "Empty response",
            }
        except Exception as e:
            return {
                "sample_id": unified["id"],
                "dataset": unified["dataset"],
                "model": model,
                "instructed_framework": framework_key,
                "response": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e)[:200],
            }


# =============================================================================
# API Calls - Attribution Scoring
# =============================================================================


def parse_reasoning_response(response_text: str) -> Optional[List[str]]:
    """Extract 4 step NLEs from a structured JSON response."""
    try:
        text = response_text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        data = json.loads(text)
        steps = data.get("reasoning_steps", [])
        if len(steps) >= 4:
            return [s.get("nle", "") for s in steps[:4]]
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


async def score_step(
    step_text: str,
    semaphore: asyncio.Semaphore,
    async_openai: AsyncOpenAI,
    attribution_model: str,
    temperature: float = 0.1,
) -> Optional[Dict]:
    """Score a single reasoning step using the attribution LLM."""
    async with semaphore:
        try:
            response = await async_openai.chat.completions.create(
                model=attribution_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in moral philosophy. Return only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": ATTRIBUTION_PROMPT.format(step_text=step_text),
                    },
                ],
                max_tokens=300,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
                if match:
                    text = match.group(1).strip()
            data = json.loads(text)
            if all(f in data for f in FRAMEWORKS):
                return {f: float(data[f]) for f in FRAMEWORKS}
        except Exception:
            pass
    return None


# =============================================================================
# Collection and Scoring Orchestration
# =============================================================================


async def collect_all_reasoning(
    sampled_scenarios: List[Dict],
    all_models: List[str],
    openai_model: str,
    frameworks_to_test: List[str],
    async_openai: AsyncOpenAI,
    async_together: AsyncTogether,
    data_dir: Path,
    openai_concurrency: int = 30,
    together_concurrency: int = 20,
    temperature: float = 0.3,
) -> List[Dict]:
    """Collect instructed reasoning responses for all model x framework conditions."""
    all_results = []

    for model in all_models:
        is_openai = model == openai_model
        semaphore = asyncio.Semaphore(
            openai_concurrency if is_openai else together_concurrency
        )
        model_short = model.split("/")[-1] if "/" in model else model

        for fw_key in frameworks_to_test:
            fw_short = FRAMEWORK_INFO[fw_key]["short"]
            safe_name = model.replace("/", "_").replace("-", "_")
            outfile = data_dir / f"responses_{safe_name}_{fw_key}.jsonl"

            # Skip if already collected
            if outfile.exists():
                with open(outfile) as f:
                    existing = [json.loads(l) for l in f if l.strip()]
                if len(existing) == len(sampled_scenarios):
                    print(
                        f"  {model_short} x {fw_short}: already collected "
                        f"({len(existing)}), skipping"
                    )
                    all_results.extend(existing)
                    continue

            print(
                f"  Collecting: {model_short} x {fw_short} "
                f"({len(sampled_scenarios)} samples)..."
            )
            tasks = [
                call_reasoning_api(
                    s,
                    fw_key,
                    model,
                    openai_model,
                    semaphore,
                    async_openai,
                    async_together,
                    temperature,
                )
                for s in sampled_scenarios
            ]
            results = await tqdm_asyncio.gather(
                *tasks, desc=f"{model_short[:15]} x {fw_short[:12]}"
            )

            with open(outfile, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

            success = sum(1 for r in results if not r["error"])
            print(f"    Saved: {success}/{len(results)} successful")
            all_results.extend(results)
            await asyncio.sleep(2)

    return all_results


async def score_all_responses(
    data_dir: Path,
    async_openai: AsyncOpenAI,
    attribution_model: str,
    attribution_concurrency: int = 50,
    attribution_temperature: float = 0.1,
) -> List[Dict]:
    """Score all collected instructed responses with the attribution pipeline."""
    semaphore = asyncio.Semaphore(attribution_concurrency)
    all_scored = []

    response_files = sorted(data_dir.glob("responses_*.jsonl"))
    print(f"Found {len(response_files)} response files")

    for rfile in response_files:
        score_file = data_dir / rfile.name.replace("responses_", "attributions_")
        if score_file.exists():
            with open(score_file) as f:
                existing = [json.loads(l) for l in f if l.strip()]
            print(f"  {rfile.name}: already scored ({len(existing)}), skipping")
            all_scored.extend(existing)
            continue

        with open(rfile) as f:
            responses = [json.loads(l) for l in f if l.strip()]

        print(f"  Scoring {rfile.name} ({len(responses)} responses)...")

        # Collect all steps to score
        steps_to_score = []  # (response_idx, step_idx, step_text)
        for i, resp in enumerate(responses):
            if resp["error"]:
                continue
            steps = parse_reasoning_response(resp["response"])
            if steps:
                for j, step_text in enumerate(steps):
                    if step_text.strip():
                        steps_to_score.append((i, j, step_text))

        # Score all steps in parallel
        score_tasks = [
            score_step(
                s[2], semaphore, async_openai, attribution_model, attribution_temperature
            )
            for s in steps_to_score
        ]
        scores = await tqdm_asyncio.gather(
            *score_tasks, desc=f"Scoring {rfile.stem[:30]}"
        )

        # Assemble results
        score_map = {}
        for (ri, si, _), sc in zip(steps_to_score, scores):
            if sc:
                score_map[(ri, si)] = sc

        scored_records = []
        for i, resp in enumerate(responses):
            step_scores = []
            for j in range(4):
                if (i, j) in score_map:
                    step_scores.append(score_map[(i, j)])

            scored_records.append(
                {
                    "sample_id": resp["sample_id"],
                    "dataset": resp["dataset"],
                    "model": resp["model"],
                    "instructed_framework": resp["instructed_framework"],
                    "step_attributions": step_scores,
                    "n_steps_scored": len(step_scores),
                    "error": resp["error"],
                }
            )

        with open(score_file, "w") as f:
            for r in scored_records:
                f.write(json.dumps(r) + "\n")

        valid = sum(1 for r in scored_records if r["n_steps_scored"] == 4)
        print(f"    Scored: {valid}/{len(scored_records)} with all 4 steps")
        all_scored.extend(scored_records)

    return all_scored


# =============================================================================
# Analysis
# =============================================================================


def build_analysis_dataframe(
    all_scored: List[Dict], frameworks_to_test: List[str]
) -> List[Dict]:
    """
    Build a list of analysis rows from scored records.

    Each row captures compliance, step-level compliance, FDR, and per-framework
    mean scores for one (sample, model, instructed_framework) combination.
    """
    rows = []
    for record in all_scored:
        if record["error"] or record["n_steps_scored"] < 4:
            continue

        instructed = record["instructed_framework"]
        model = (
            record["model"].split("/")[-1]
            if "/" in record["model"]
            else record["model"]
        )

        # Mean attribution score per framework across 4 steps
        mean_scores = {}
        for fw in FRAMEWORKS:
            scores = [step[fw] for step in record["step_attributions"]]
            mean_scores[fw] = float(np.mean(scores))

        dominant = max(mean_scores, key=mean_scores.get)
        compliant = dominant == instructed
        instructed_score = mean_scores[instructed]

        # Per-step dominance
        step_dominant = []
        for step in record["step_attributions"]:
            step_dom = max(FRAMEWORKS, key=lambda f: step[f])
            step_dominant.append(step_dom)
        step_compliance = sum(1 for d in step_dominant if d == instructed) / 4

        # FDR under instruction
        transitions = sum(
            1
            for i in range(1, len(step_dominant))
            if step_dominant[i] != step_dominant[i - 1]
        )
        fdr = transitions / (len(step_dominant) - 1) if len(step_dominant) > 1 else 0.0

        rows.append(
            {
                "sample_id": record["sample_id"],
                "dataset": record["dataset"],
                "model": model,
                "instructed_framework": instructed,
                "instructed_short": FW_SHORT.get(instructed, instructed),
                "dominant_framework": dominant,
                "compliant": compliant,
                "instructed_mean_score": instructed_score,
                "step_compliance": step_compliance,
                "fdr": fdr,
                **{f"mean_{fw}": mean_scores[fw] for fw in FRAMEWORKS},
            }
        )

    return rows


def print_analysis(rows: List[Dict], frameworks_to_test: List[str]) -> None:
    """Print key metrics from the analysis rows."""
    if not rows:
        print("No valid scored records to analyze.")
        return

    print(f"\nAnalysis: {len(rows)} valid records")

    # Group by (instructed_framework, model)
    from collections import defaultdict

    groups = defaultdict(list)
    fw_groups = defaultdict(list)
    for r in rows:
        groups[(r["instructed_short"], r["model"])].append(r)
        fw_groups[r["instructed_short"]].append(r)

    # Compliance rate
    print("\n" + "=" * 70)
    print("FRAMEWORK COMPLIANCE RATE")
    print("(% of responses where instructed framework has highest mean score)")
    print("=" * 70)
    models = sorted({r["model"] for r in rows})
    fw_shorts = [FW_SHORT[f] for f in frameworks_to_test]
    header = f"{'Framework':<12}" + "".join(f"{m:>20}" for m in models) + f"{'Overall':>12}"
    print(header)
    for fws in fw_shorts:
        line = f"{fws:<12}"
        for m in models:
            recs = groups.get((fws, m), [])
            rate = (
                sum(1 for r in recs if r["compliant"]) / len(recs) * 100
                if recs
                else 0
            )
            line += f"{rate:>20.1f}"
        all_recs = fw_groups.get(fws, [])
        overall = (
            sum(1 for r in all_recs if r["compliant"]) / len(all_recs) * 100
            if all_recs
            else 0
        )
        line += f"{overall:>12.1f}"
        print(line)

    # Mean instructed framework score
    print("\n" + "=" * 70)
    print("MEAN INSTRUCTED FRAMEWORK SCORE (0-100)")
    print("=" * 70)
    print(header)
    for fws in fw_shorts:
        line = f"{fws:<12}"
        for m in models:
            recs = groups.get((fws, m), [])
            avg = (
                sum(r["instructed_mean_score"] for r in recs) / len(recs)
                if recs
                else 0
            )
            line += f"{avg:>20.1f}"
        all_recs = fw_groups.get(fws, [])
        overall = (
            sum(r["instructed_mean_score"] for r in all_recs) / len(all_recs)
            if all_recs
            else 0
        )
        line += f"{overall:>12.1f}"
        print(line)

    # Step-level compliance
    print("\n" + "=" * 70)
    print("STEP-LEVEL COMPLIANCE (%)")
    print("=" * 70)
    print(header)
    for fws in fw_shorts:
        line = f"{fws:<12}"
        for m in models:
            recs = groups.get((fws, m), [])
            avg = (
                sum(r["step_compliance"] for r in recs) / len(recs) * 100
                if recs
                else 0
            )
            line += f"{avg:>20.1f}"
        all_recs = fw_groups.get(fws, [])
        overall = (
            sum(r["step_compliance"] for r in all_recs) / len(all_recs) * 100
            if all_recs
            else 0
        )
        line += f"{overall:>12.1f}"
        print(line)

    # FDR
    print("\n" + "=" * 70)
    print("FDR UNDER INSTRUCTION")
    print("=" * 70)
    print(header)
    for fws in fw_shorts:
        line = f"{fws:<12}"
        for m in models:
            recs = groups.get((fws, m), [])
            avg = sum(r["fdr"] for r in recs) / len(recs) if recs else 0
            line += f"{avg:>20.3f}"
        all_recs = fw_groups.get(fws, [])
        overall = (
            sum(r["fdr"] for r in all_recs) / len(all_recs) if all_recs else 0
        )
        line += f"{overall:>12.3f}"
        print(line)


def save_summary_csv(
    rows: List[Dict],
    frameworks_to_test: List[str],
    results_dir: Path,
) -> None:
    """Export summary table as CSV."""
    import csv

    models = sorted({r["model"] for r in rows})
    summary_rows = []

    for fw_key in frameworks_to_test:
        fw_recs = [r for r in rows if r["instructed_framework"] == fw_key]
        for model in models:
            m_recs = [r for r in fw_recs if r["model"] == model]
            if not m_recs:
                continue
            summary_rows.append(
                {
                    "Framework": FRAMEWORK_INFO[fw_key]["short"],
                    "Model": model,
                    "N": len(m_recs),
                    "Compliance_pct": round(
                        sum(1 for r in m_recs if r["compliant"])
                        / len(m_recs)
                        * 100,
                        1,
                    ),
                    "Mean_Instructed_Score": round(
                        sum(r["instructed_mean_score"] for r in m_recs)
                        / len(m_recs),
                        1,
                    ),
                    "Step_Compliance_pct": round(
                        sum(r["step_compliance"] for r in m_recs)
                        / len(m_recs)
                        * 100,
                        1,
                    ),
                    "Mean_FDR": round(
                        sum(r["fdr"] for r in m_recs) / len(m_recs), 3
                    ),
                }
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "framework_instructability_summary.csv"
    fieldnames = [
        "Framework",
        "Model",
        "N",
        "Compliance_pct",
        "Mean_Instructed_Score",
        "Step_Compliance_pct",
        "Mean_FDR",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary CSV: {csv_path}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    for fw_key in frameworks_to_test:
        fw_recs = [r for r in rows if r["instructed_framework"] == fw_key]
        if fw_recs:
            rate = sum(1 for r in fw_recs if r["compliant"]) / len(fw_recs) * 100
            print(f"  {FRAMEWORK_INFO[fw_key]['short']:20s}: {rate:.1f}% compliance")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness check: test framework instructability of LLMs."
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5",
        help="OpenAI model for reasoning generation.",
    )
    parser.add_argument(
        "--together-models",
        nargs="+",
        default=[
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ],
        help="Together.ai models for reasoning generation.",
    )
    parser.add_argument(
        "--attribution-model",
        default="gpt-4o-mini",
        help="Model used for attribution scoring.",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=FRAMEWORKS,
        help="Frameworks to test instructability for.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=30,
        help="Number of scenarios per dataset (default 30 = 90 total).",
    )
    parser.add_argument(
        "--pilot-data",
        type=str,
        default="pilot_test_1500samples.jsonl",
        help="Path to pilot test JSONL file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/robustness_check",
        help="Directory for intermediate data files.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/robustness_check",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for reasoning generation.",
    )
    parser.add_argument(
        "--attribution-temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for attribution scoring.",
    )
    parser.add_argument(
        "--openai-concurrency",
        type=int,
        default=30,
        help="Max concurrent OpenAI API calls.",
    )
    parser.add_argument(
        "--together-concurrency",
        type=int,
        default=20,
        help="Max concurrent Together.ai API calls.",
    )
    parser.add_argument(
        "--attribution-concurrency",
        type=int,
        default=50,
        help="Max concurrent attribution scoring calls.",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        default=False,
        help="Skip reasoning collection, only run scoring and analysis.",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        default=False,
        help="Skip attribution scoring, only run analysis on existing data.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # API clients
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    together_key = os.environ.get("TOGETHER_API_KEY", "")

    if not args.skip_collection:
        if not openai_key:
            raise RuntimeError("Set OPENAI_API_KEY environment variable.")
        if not together_key:
            raise RuntimeError("Set TOGETHER_API_KEY environment variable.")

    async_openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    async_together_client = AsyncTogether(api_key=together_key) if together_key else None

    all_models = [args.openai_model] + args.together_models

    print("=" * 72)
    print("ROBUSTNESS CHECK: FRAMEWORK INSTRUCTABILITY")
    print("=" * 72)
    print(f"  Models:        {all_models}")
    print(f"  Frameworks:    {args.frameworks}")
    print(f"  Samples/ds:    {args.samples_per_dataset}")
    print(f"  Attribution:   {args.attribution_model}")
    print(f"  Data dir:      {data_dir}")
    print(f"  Results dir:   {results_dir}")
    print()

    # Load and select samples
    pilot_path = Path(args.pilot_data)
    if not pilot_path.exists():
        print(f"Pilot data not found at {pilot_path}, skipping collection.")
        args.skip_collection = True
        sampled_scenarios = []
    else:
        pilot_samples = load_pilot_samples(pilot_path)
        print(f"Loaded {len(pilot_samples)} pilot samples")
        sampled_scenarios = select_robustness_subset(
            pilot_samples, args.samples_per_dataset
        )
        print(f"Total: {len(sampled_scenarios)} scenarios")

    # Step 1: Collect instructed reasoning responses
    if not args.skip_collection:
        print("\n" + "=" * 60)
        print("STEP 1: REASONING COLLECTION")
        print("=" * 60)
        reasoning_results = asyncio.run(
            collect_all_reasoning(
                sampled_scenarios,
                all_models,
                args.openai_model,
                args.frameworks,
                async_openai_client,
                async_together_client,
                data_dir,
                args.openai_concurrency,
                args.together_concurrency,
                args.temperature,
            )
        )
        print(f"\nTotal collected: {len(reasoning_results)}")
        print(f"Successful: {sum(1 for r in reasoning_results if not r['error'])}")
    else:
        print("\nSkipping reasoning collection.")

    # Step 2: Score all responses
    if not args.skip_scoring:
        if not openai_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY for attribution scoring with "
                f"{args.attribution_model}."
            )
        print("\n" + "=" * 60)
        print("STEP 2: ATTRIBUTION SCORING")
        print("=" * 60)
        scored_results = asyncio.run(
            score_all_responses(
                data_dir,
                async_openai_client,
                args.attribution_model,
                args.attribution_concurrency,
                args.attribution_temperature,
            )
        )
        print(f"\nTotal scored: {len(scored_results)}")
    else:
        print("\nSkipping attribution scoring.")

    # Step 3: Analysis
    print("\n" + "=" * 60)
    print("STEP 3: ANALYSIS")
    print("=" * 60)

    # Load all scored records
    all_scored = []
    for f in sorted(data_dir.glob("attributions_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_scored.append(json.loads(line))

    print(f"Loaded {len(all_scored)} scored records")

    if all_scored:
        analysis_rows = build_analysis_dataframe(all_scored, args.frameworks)
        print_analysis(analysis_rows, args.frameworks)
        save_summary_csv(analysis_rows, args.frameworks, results_dir)
    else:
        print("No scored records found. Run collection and scoring first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
