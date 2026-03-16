#!/usr/bin/env python3
"""
Generate 1,500 pilot test samples from three moral reasoning datasets (500 per dataset).
"""

import pandas as pd
import json
import random
from pathlib import Path
from collections import Counter

# Set random seed for reproducibility
random.seed(42)

print("=" * 80)
print("PILOT SAMPLING SCRIPT")
print("=" * 80)

# ============================================================================
# 1. Load Ethics Dataset
# ============================================================================
print("\n[1/7] Loading Ethics dataset...")
ethics_path = Path('datasets/ethics')

# Load Commonsense
cm_train = pd.read_csv(ethics_path / 'commonsense/cm_train.csv')
cm_test = pd.read_csv(ethics_path / 'commonsense/cm_test.csv')
cm_test_hard = pd.read_csv(ethics_path / 'commonsense/cm_test_hard.csv')

# Load Deontology
deont_train = pd.read_csv(ethics_path / 'deontology/deontology_train.csv')
deont_test = pd.read_csv(ethics_path / 'deontology/deontology_test.csv')
deont_test_hard = pd.read_csv(ethics_path / 'deontology/deontology_test_hard.csv')

# Load Justice
justice_train = pd.read_csv(ethics_path / 'justice/justice_train.csv')
justice_test = pd.read_csv(ethics_path / 'justice/justice_test.csv')
justice_test_hard = pd.read_csv(ethics_path / 'justice/justice_test_hard.csv')

# Load Utilitarianism
util_train = pd.read_csv(ethics_path / 'utilitarianism/util_train.csv', header=None, names=['scenario1', 'scenario2'])
util_test = pd.read_csv(ethics_path / 'utilitarianism/util_test.csv', header=None, names=['scenario1', 'scenario2'])
util_test_hard = pd.read_csv(ethics_path / 'utilitarianism/util_test_hard.csv', header=None, names=['scenario1', 'scenario2'])

# Load Virtue
virtue_train = pd.read_csv(ethics_path / 'virtue/virtue_train.csv')
virtue_test = pd.read_csv(ethics_path / 'virtue/virtue_test.csv')
virtue_test_hard = pd.read_csv(ethics_path / 'virtue/virtue_test_hard.csv')

print(f"✓ Ethics dataset loaded")

# ============================================================================
# 2. Load Moral Stories Dataset
# ============================================================================
print("[2/7] Loading Moral Stories dataset...")
moral_stories_path = Path('datasets/moral_stories_dataset/data/moral_stories_full.jsonl')

moral_stories = []
with open(moral_stories_path, 'r') as f:
    for line in f:
        moral_stories.append(json.loads(line))

print(f"✓ Moral Stories loaded: {len(moral_stories):,} stories")

# ============================================================================
# 3. Load Social-Chem-101 Dataset
# ============================================================================
print("[3/7] Loading Social-Chem-101 dataset...")
social_chem_path = Path('datasets/social-chem-101/social-chem-101.v1.0.tsv')

social_chem = pd.read_csv(social_chem_path, sep='\t', low_memory=False)
print(f"✓ Social-Chem-101 loaded: {len(social_chem):,} instances")

# ============================================================================
# 4. Sample Ethics Dataset (500 instances)
# ============================================================================
print("\n[4/7] Sampling Ethics dataset (500 instances)...")

def sample_ethics_framework(train_df, test_df, test_hard_df, framework_name, train_n, test_n, test_hard_n):
    """Sample from train, test, and test_hard splits for a given framework."""
    samples = []

    # Sample from train
    train_samples = train_df.sample(n=train_n, random_state=42)
    for idx, row in train_samples.iterrows():
        samples.append({
            'source_dataset': 'ethics',
            'framework': framework_name,
            'split': 'train',
            'original_data': row.to_dict(),
            'metadata': {'difficulty': 'standard'}
        })

    # Sample from test
    test_samples = test_df.sample(n=test_n, random_state=42)
    for idx, row in test_samples.iterrows():
        samples.append({
            'source_dataset': 'ethics',
            'framework': framework_name,
            'split': 'test',
            'original_data': row.to_dict(),
            'metadata': {'difficulty': 'standard'}
        })

    # Sample from test_hard
    test_hard_samples = test_hard_df.sample(n=test_hard_n, random_state=42)
    for idx, row in test_hard_samples.iterrows():
        samples.append({
            'source_dataset': 'ethics',
            'framework': framework_name,
            'split': 'test_hard',
            'original_data': row.to_dict(),
            'metadata': {'difficulty': 'hard'}
        })

    return samples

ethics_samples = []

# Each framework: 100 instances (70 train, 20 test, 10 test_hard)
# Commonsense: 100
ethics_samples.extend(sample_ethics_framework(cm_train, cm_test, cm_test_hard, 'commonsense', 70, 20, 10))

# Deontology: 100
ethics_samples.extend(sample_ethics_framework(deont_train, deont_test, deont_test_hard, 'deontology', 70, 20, 10))

# Justice: 100
ethics_samples.extend(sample_ethics_framework(justice_train, justice_test, justice_test_hard, 'justice', 70, 20, 10))

# Utilitarianism: 100
ethics_samples.extend(sample_ethics_framework(util_train, util_test, util_test_hard, 'utilitarianism', 70, 20, 10))

# Virtue: 100
ethics_samples.extend(sample_ethics_framework(virtue_train, virtue_test, virtue_test_hard, 'virtue', 70, 20, 10))

print(f"✓ Ethics sampling complete: {len(ethics_samples)} instances")

# ============================================================================
# 5. Sample Moral Stories Dataset (500 instances)
# ============================================================================
print("[5/7] Sampling Moral Stories dataset (500 instances)...")

moral_stories_sampled = random.sample(moral_stories, 500)

moral_stories_samples = []
for story in moral_stories_sampled:
    moral_stories_samples.append({
        'source_dataset': 'moral_stories',
        'framework': 'social_norms',
        'split': 'full',
        'original_data': story,
        'metadata': {
            'difficulty': 'standard',
            'story_id': story['ID'],
            'norm': story['norm']
        }
    })

unique_norms = len(set(s['metadata']['norm'] for s in moral_stories_samples))
print(f"✓ Moral Stories sampling complete: {len(moral_stories_samples)} instances")
print(f"  Norm diversity: {unique_norms} unique norms")

# ============================================================================
# 6. Sample Social-Chem-101 Dataset (500 instances)
# ============================================================================
print("[6/7] Sampling Social-Chem-101 dataset (500 instances)...")

# Filter for quality
social_chem_filtered = social_chem[
    (social_chem['rot-bad'] == 0) &
    (social_chem['split'] == 'train') &
    (social_chem['rot-moral-foundations'].notna()) &
    (social_chem['rot-moral-foundations'] != '')
].copy()

# Extract primary moral foundation
social_chem_filtered['primary_foundation'] = social_chem_filtered['rot-moral-foundations'].apply(
    lambda x: x.split('|')[0] if pd.notna(x) and x != '' else None
)

# Sample by moral foundation (100 each for 500 total)
foundation_targets = {
    'care-harm': 100,
    'fairness-cheating': 100,
    'loyalty-betrayal': 100,
    'authority-subversion': 100,
    'sanctity-degradation': 100
}

social_chem_samples = []

for foundation, target_n in foundation_targets.items():
    foundation_data = social_chem_filtered[social_chem_filtered['primary_foundation'] == foundation]

    if len(foundation_data) >= target_n:
        sampled = foundation_data.sample(n=target_n, random_state=42)
    else:
        print(f"  Warning: Only {len(foundation_data)} instances for {foundation}, using all")
        sampled = foundation_data

    for idx, row in sampled.iterrows():
        social_chem_samples.append({
            'source_dataset': 'social_chem_101',
            'framework': 'moral_foundations',
            'split': 'train',
            'original_data': row.to_dict(),
            'metadata': {
                'difficulty': 'standard',
                'moral_foundation': foundation,
                'rot_agree': row['rot-agree'],
                'area': row['area']
            }
        })

print(f"✓ Social-Chem-101 sampling complete: {len(social_chem_samples)} instances")

# ============================================================================
# 7. Combine and Export
# ============================================================================
print("\n[7/7] Combining and exporting samples...")

# Combine all samples
all_samples = ethics_samples + moral_stories_samples + social_chem_samples

# Add unique IDs and text representations
for i, sample in enumerate(all_samples):
    sample['id'] = f"pilot_{i:04d}"

    # Add text representation
    dataset = sample['source_dataset']
    data = sample['original_data']

    if dataset == 'ethics':
        framework = sample['framework']

        if framework == 'commonsense':
            label = "RIGHT" if data['label'] == 1 else "WRONG"
            sample['text'] = f"[{label}] {data['input']}"

        elif framework == 'deontology':
            label = "REASONABLE" if data['label'] == 1 else "UNREASONABLE"
            sample['text'] = f"Scenario: {data['scenario']}\nExcuse: {data['excuse']}\nJudgment: {label}"

        elif framework == 'justice':
            label = "JUST" if data['label'] == 1 else "UNJUST"
            sample['text'] = f"[{label}] {data['scenario']}"

        elif framework == 'utilitarianism':
            sample['text'] = f"Scenario 1: {data['scenario1']}\nScenario 2 (worse): {data['scenario2']}"

        elif framework == 'virtue':
            label = "APPLIES" if data['label'] == 1 else "DOES NOT APPLY"
            sample['text'] = f"[{label}] {data['scenario']}"

    elif dataset == 'moral_stories':
        sample['text'] = f"""Norm: {data['norm']}
Situation: {data['situation']}
Intention: {data['intention']}

MORAL PATH:
  Action: {data['moral_action']}
  Consequence: {data['moral_consequence']}

IMMORAL PATH:
  Action: {data['immoral_action']}
  Consequence: {data['immoral_consequence']}"""

    elif dataset == 'social_chem_101':
        rot = data.get('rot', 'N/A')
        situation = data.get('situation', 'N/A')
        action = data.get('action', 'N/A')
        judgment = data.get('rot-judgment', 'N/A')
        sample['text'] = f"""Rule-of-Thumb: {rot}
Situation: {situation}
Action: {action}
Judgment: {judgment}
Moral Foundation: {sample['metadata'].get('moral_foundation', 'N/A')}"""

# Export to JSONL
output_file = 'pilot_test_1500samples.jsonl'
with open(output_file, 'w') as f:
    for sample in all_samples:
        f.write(json.dumps(sample) + '\n')

print(f"✓ Exported {len(all_samples)} samples to {output_file}")

# Validation
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print(f"\nTotal samples: {len(all_samples)}")
print(f"Target: 1,500")
print(f"Match: {'✓ YES' if len(all_samples) == 1500 else '✗ NO'}")

# Distribution by dataset
print(f"\nDistribution by dataset:")
dataset_counts = Counter(s['source_dataset'] for s in all_samples)
for dataset, count in sorted(dataset_counts.items()):
    percentage = (count / len(all_samples)) * 100
    print(f"  {dataset}: {count} ({percentage:.1f}%)")

# Distribution by framework
print(f"\nDistribution by framework:")
framework_counts = Counter(s['framework'] for s in all_samples)
for framework, count in sorted(framework_counts.items()):
    print(f"  {framework}: {count}")

print("\n" + "=" * 80)
print("SAMPLING COMPLETE!")
print("=" * 80)
