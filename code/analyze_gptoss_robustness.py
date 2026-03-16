"""
Analyze GPT-OSS-120B re-scored robustness check attributions.
Computes: compliance rate, mean instructed score, step-level compliance, FDR.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('data/robustness_check')
FRAMEWORKS = ['act_utilitarianism', 'deontology', 'virtue_ethics', 'contractualism', 'contractarianism']
FW_SHORT = {'act_utilitarianism': 'Util', 'deontology': 'Deont', 'virtue_ethics': 'Virtue', 'contractualism': 'C-ism', 'contractarianism': 'C-ian'}

MODEL_MAP = {
    'gpt-5': 'GPT-5',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo': 'Llama',
    'Qwen/Qwen2.5-72B-Instruct': 'Qwen'
}
ALL_MODELS = ['GPT-5', 'Llama', 'Qwen']

def load_attributions():
    """Load all GPT-OSS attribution files."""
    records = []
    for fpath in sorted(DATA_DIR.glob('attributions_gptoss_*.jsonl')):
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                records.append(r)
    return records

def get_model_short(model_str):
    for key, val in MODEL_MAP.items():
        if key in model_str:
            return val
    return model_str

def analyze():
    records = load_attributions()
    print(f"Total records: {len(records)}")

    # Group by model × instructed_framework
    groups = defaultdict(list)
    for r in records:
        if r['n_steps_scored'] < 4:
            continue
        model = get_model_short(r['model'])
        fw = r['instructed_framework']
        groups[(model, fw)].append(r)

    print(f"\nRecords with all 4 steps scored:")
    for (model, fw), recs in sorted(groups.items()):
        print(f"  {model} × {FW_SHORT.get(fw, fw)}: {len(recs)}")

    # === 1. Mean Instructed Framework Score ===
    print("\n" + "="*70)
    print("1. MEAN INSTRUCTED FRAMEWORK SCORE (0-100)")
    print("="*70)
    header = f"{'Framework':<20} {'GPT-5':>8} {'Llama':>8} {'Qwen':>8} {'Overall':>8}"
    print(header)
    print("-" * len(header))

    overall_scores = {}
    for fw in FRAMEWORKS:
        scores_by_model = {}
        all_scores = []
        for model in ALL_MODELS:
            recs = groups.get((model, fw), [])
            fw_scores = []
            for r in recs:
                step_means = [s[fw] for s in r['step_attributions']]
                fw_scores.append(np.mean(step_means))
            scores_by_model[model] = np.mean(fw_scores) if fw_scores else float('nan')
            all_scores.extend(fw_scores)
        overall = np.mean(all_scores) if all_scores else float('nan')
        overall_scores[fw] = overall
        print(f"{FW_SHORT[fw]:<20} {scores_by_model.get('GPT-5', float('nan')):>8.1f} {scores_by_model.get('Llama', float('nan')):>8.1f} {scores_by_model.get('Qwen', float('nan')):>8.1f} {overall:>8.1f}")

    # === 2. Compliance Rate ===
    print("\n" + "="*70)
    print("2. COMPLIANCE RATE (instructed fw has highest mean score)")
    print("="*70)
    header = f"{'Framework':<20} {'GPT-5':>8} {'Llama':>8} {'Qwen':>8} {'Overall':>8}"
    print(header)
    print("-" * len(header))

    for fw in FRAMEWORKS:
        compliance_by_model = {}
        all_compliant = []
        for model in ALL_MODELS:
            recs = groups.get((model, fw), [])
            compliant = 0
            for r in recs:
                mean_scores = {}
                for f in FRAMEWORKS:
                    mean_scores[f] = np.mean([s[f] for s in r['step_attributions']])
                if mean_scores[fw] == max(mean_scores.values()):
                    compliant += 1
                    all_compliant.append(1)
                else:
                    all_compliant.append(0)
            compliance_by_model[model] = (compliant / len(recs) * 100) if recs else float('nan')
        overall = (sum(all_compliant) / len(all_compliant) * 100) if all_compliant else float('nan')
        print(f"{FW_SHORT[fw]:<20} {compliance_by_model.get('GPT-5', float('nan')):>7.1f}% {compliance_by_model.get('Llama', float('nan')):>7.1f}% {compliance_by_model.get('Qwen', float('nan')):>7.1f}% {overall:>7.1f}%")

    # === 3. Step-Level Compliance ===
    print("\n" + "="*70)
    print("3. STEP-LEVEL COMPLIANCE (% of steps where instructed fw is dominant)")
    print("="*70)
    header = f"{'Framework':<20} {'GPT-5':>8} {'Llama':>8} {'Qwen':>8} {'Overall':>8}"
    print(header)
    print("-" * len(header))

    for fw in FRAMEWORKS:
        step_compliance_by_model = {}
        all_steps = []
        for model in ALL_MODELS:
            recs = groups.get((model, fw), [])
            compliant_steps = 0
            total_steps = 0
            for r in recs:
                for step in r['step_attributions']:
                    total_steps += 1
                    if step[fw] == max(step[f] for f in FRAMEWORKS):
                        compliant_steps += 1
                        all_steps.append(1)
                    else:
                        all_steps.append(0)
            step_compliance_by_model[model] = (compliant_steps / total_steps * 100) if total_steps > 0 else float('nan')
        overall = (sum(all_steps) / len(all_steps) * 100) if all_steps else float('nan')
        print(f"{FW_SHORT[fw]:<20} {step_compliance_by_model.get('GPT-5', float('nan')):>7.1f}% {step_compliance_by_model.get('Llama', float('nan')):>7.1f}% {step_compliance_by_model.get('Qwen', float('nan')):>7.1f}% {overall:>7.1f}%")

    # === 4. Framework Drift Rate (FDR) ===
    print("\n" + "="*70)
    print("4. FRAMEWORK DRIFT RATE (FDR)")
    print("="*70)
    header = f"{'Framework':<20} {'GPT-5':>8} {'Llama':>8} {'Qwen':>8} {'Overall':>8}"
    print(header)
    print("-" * len(header))

    for fw in FRAMEWORKS:
        fdr_by_model = {}
        all_fdrs = []
        for model in ALL_MODELS:
            recs = groups.get((model, fw), [])
            sample_fdrs = []
            for r in recs:
                transitions = 0
                steps = r['step_attributions']
                for i in range(len(steps) - 1):
                    dom_i = max(FRAMEWORKS, key=lambda f: steps[i][f])
                    dom_j = max(FRAMEWORKS, key=lambda f: steps[i+1][f])
                    if dom_i != dom_j:
                        transitions += 1
                fdr = transitions / (len(steps) - 1) if len(steps) > 1 else 0
                sample_fdrs.append(fdr)
                all_fdrs.append(fdr)
            fdr_by_model[model] = np.mean(sample_fdrs) if sample_fdrs else float('nan')
        overall = np.mean(all_fdrs) if all_fdrs else float('nan')
        print(f"{FW_SHORT[fw]:<20} {fdr_by_model.get('GPT-5', float('nan')):>8.3f} {fdr_by_model.get('Llama', float('nan')):>8.3f} {fdr_by_model.get('Qwen', float('nan')):>8.3f} {overall:>8.3f}")

    # === 5. Cross-Framework Leakage for Contractarianism ===
    print("\n" + "="*70)
    print("5. CROSS-FRAMEWORK LEAKAGE (when instructed Contractarianism)")
    print("="*70)
    for model in ALL_MODELS:
        recs = groups.get((model, 'contractarianism'), [])
        if not recs:
            print(f"  {model}: no data")
            continue
        print(f"\n  {model} (n={len(recs)}):")
        for fw in FRAMEWORKS:
            scores = []
            for r in recs:
                scores.extend([s[fw] for s in r['step_attributions']])
            print(f"    {FW_SHORT[fw]:<15} {np.mean(scores):>6.1f}")

    # === 6. Sample counts ===
    print("\n" + "="*70)
    print("6. SAMPLE COUNTS (with all 4 steps scored)")
    print("="*70)
    for fw in FRAMEWORKS:
        counts = {}
        total = 0
        for model in ALL_MODELS:
            n = len(groups.get((model, fw), []))
            counts[model] = n
            total += n
        print(f"  {FW_SHORT[fw]:<15} GPT-5: {counts['GPT-5']:>3}  Llama: {counts['Llama']:>3}  Qwen: {counts['Qwen']:>3}  Total: {total:>3}")

if __name__ == '__main__':
    analyze()
