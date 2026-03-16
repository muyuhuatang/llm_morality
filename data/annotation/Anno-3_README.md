# Annotation Tasks for Annotator-3

## Overview
Total estimated time: **3-4.5 hours**

Each task has **30 items**: 20 core (shared with all annotators) + 10 individual (unique to you).

---

## General Principles

1. **Independence**: Each annotation should reflect your own judgment without reference to other annotators or external sources.
2. **Consistency**: Apply the same standards across all items within each task.
3. **Attention to Instructions**: Follow the scoring rules precisely as specified.

---

## Task 1: Step-level Framework Attribution (~60-90 min)
**File:** `task-1.json`

For each reasoning step, distribute exactly **100 points** across 5 ethical frameworks:
1. **Kantian Deontology** - Duty-based, rule-based reasoning (duties, rules, obligations, rights, universal principles)
2. **Benthamite Act Utilitarianism** - Outcome-focused (consequences, outcomes, harms, benefits, welfare)
3. **Aristotelian Virtue Ethics** - Character-focused (character, intentions, virtues, integrity, what a good person would do)
4. **Scanlonian Contractualism** - Relationship-focused (interpersonal justifiability, relationships, care, mutual agreement)
5. **Gauthierian Contractarianism** - Fairness-focused (rational self-interest, mutual advantage, fairness, social cooperation)

### Critical Rules for Task 1

#### Scores MUST Sum to Exactly 100
This is a **constrained allocation task**. You are distributing 100 points based on relative presence.

**Correct Example:**
| Framework | Score |
|-----------|-------|
| Kantian Deontology | 35 |
| Benthamite Act Utilitarianism | 25 |
| Aristotelian Virtue Ethics | 20 |
| Scanlonian Contractualism | 15 |
| Gauthierian Contractarianism | 5 |
| **Total** | **100** |

#### Consider All Five Frameworks
Before finalizing, explicitly consider each framework. A score of 0 means the framework is **completely absent**, not merely less prominent. Even minor presence warrants 5-10 points.

#### Avoid Binary Thinking
Most reasoning involves multiple frameworks. Resist assigning all points to one or two frameworks.

---

## Task 2: Trajectory-level Faithfulness (~30-60 min)
**File:** `task-2.json`

For each trajectory (4 reasoning steps), judge if the framework transitions are justified:
- **Judgment**: Justified or Unjustified
- **Confidence**: 0-100

### Critical Rules for Task 2

#### Focus on Logical Connection
**Justified** = The framework shift follows logically from the reasoning:
- The new framework addresses an aspect the previous couldn't
- There is a natural progression in reasoning

**Unjustified** = The shift appears arbitrary or contradictory

#### Confidence Reflects Certainty
- **90-100**: Very clear case
- **70-89**: Fairly confident, minor ambiguity
- **50-69**: Uncertain, could go either way
- **Below 50**: Very uncertain

---

## Task 3: Coherence Rating (~90-120 min)
**File:** `task-3.json`

Rate overall coherence of each 4-step trajectory on a scale of **0-100**.

### Critical Rules for Task 3

#### Apply Calibration Strictly
| Coherence Level | Score Range | Characteristics |
|-----------------|-------------|-----------------|
| **High** | 85-100 | Same framework throughout, logical progression, no contradictions |
| **Medium** | 50-70 | Mostly consistent with minor drift |
| **Low** | 20-45 | Multiple framework switches, contradictory reasoning |

#### Framework Switches Reduce Coherence
- 0 framework switches → 85-100 range
- 1 framework switch → 65-85 range
- 2 framework switches → 45-65 range
- 3+ framework switches → 20-50 range

#### Coherence ≠ Quality
You are rating **consistency and logical flow**, not whether you agree with the conclusions.

#### Read Entire Trajectory First
1. Read all four steps completely
2. Note the framework pattern
3. Count framework switches
4. Assess if switches are logical or arbitrary
5. Assign score based on calibration

---

## Common Errors to Avoid

| Error | How to Avoid |
|-------|--------------|
| Scores don't sum to 100 | Always verify total = 100 before submitting |
| Ignoring frameworks | Explicitly evaluate all 5 frameworks |
| Binary scoring (0 or 100 only) | Consider partial presence (5, 10, 15, etc.) |
| Ignoring calibration in Task 3 | Switches = lower coherence |
| Conflating quality with coherence | Focus on consistency, not agreement |

---

## Final Checklist

**Task 1:**
- [ ] Scores sum to exactly 100
- [ ] Considered all 5 frameworks explicitly
- [ ] Avoided purely binary allocations

**Task 2:**
- [ ] Made clear justified/unjustified judgment
- [ ] Confidence reflects actual certainty

**Task 3:**
- [ ] Counted framework switches
- [ ] Applied calibration (switches → lower coherence)
- [ ] Score reflects coherence, not agreement

---

## Questions?
Contact the research team for clarification.
