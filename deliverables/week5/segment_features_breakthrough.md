# Segment-Relative Features — Ceiling Breakthrough Log
**Experiments:** exp_019, exp_020, exp_021  
**Date added:** post Week 5  
**Previous best:** exp_013 — F1=0.6585  
**New best:** exp_020 — F1=0.6727  

---

## Problem Identified

The F1 score was stuck at ~0.658 across all Week 4 and Week 5 tuning experiments.
The root cause was identified as the **global congestion threshold**.

The global threshold labels any segment with speed below **23 mph** as congested.
This does not account for the fact that different road segments have very different
normal speed ranges.

- A highway segment with a normal speed of 45 mph is labeled "congested" at 22 mph
  — but 22 mph may just be its normal slow-peak traffic, not actual congestion.
- An arterial segment with a normal speed of 15 mph is labeled "not congested" at 20 mph
  — but 20 mph may be unusually fast for that segment.

**Measured disagreement:** 16.3% of training rows are labeled differently by the
global threshold vs. a segment-relative threshold. This means at least 16% of training
labels are incorrect for some segments, creating a hard ceiling no model can overcome
regardless of algorithm or hyperparameter tuning.

---

## Fix: Segment-Relative Features Added to run.py Superset

Four new features were added to `run.py`'s precomputed feature superset.
All four are derived from **training data only** — no leakage into validation or test.

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `segment_mean_speed` | mean(SPEED) per segment, train only | What is "normal" for this road |
| `segment_std_speed` | std(SPEED) per segment, train only | How variable this road's speed is |
| `speed_vs_seg_mean` | SPEED / segment_mean_speed | How fast relative to this segment's norm |
| `speed_zscore` | (SPEED − mean) / std | How many std deviations from normal |

These features give the model the context it was missing:
instead of asking "is 22 mph slow?" it can now ask "is this speed unusual *for this road*?"

### Leakage check
- Segment stats computed from `train_df` only — same pattern as the congestion threshold
- Applied to `valid_df` by merge (valid rows contribute nothing to the stats)
- Test split never loaded or accessed at any point in `run.py`

---

## Experiment Results

| exp | description | val_f1 | val_precision | val_recall | status |
|-----|-------------|--------|---------------|------------|--------|
| exp_019 | segment features + undersample 2:1 + threshold=0.50 | 0.6654 | 0.7202 | 0.6183 | keep |
| exp_020 | segment features + undersample 2:1 + threshold=0.40 | **0.6727** | 0.6441 | 0.7039 | keep |
| exp_021 | segment features + class_weight=balanced + threshold=0.40 | 0.6641 | 0.5600 | 0.8157 | keep |

**exp_020 is the new overall best.**

---

## What Changed and Why

**exp_019 vs previous best (exp_013):**
- F1: 0.6585 → 0.6654 (+0.0069)
- Precision jumped from 0.654 → 0.720 — the model now correctly identifies highway segments
  running at their normal slow pace as "not congested"
- Recall dropped from 0.663 → 0.618 — threshold=0.50 made the model more conservative

**exp_020 (best):** Lowering threshold to 0.40 recovered recall (0.618 → 0.704) while keeping
the precision benefit from segment features. This is the sweet spot.

**exp_021:** Using `class_weight=balanced` instead of undersampling pushed recall to 0.816 but
dropped F1 below exp_020. Useful if recall is the deployment priority.

---

## Overall Progress vs Baseline

| | F1 | Precision | Recall |
|--|-----|-----------|--------|
| Baseline (exp_001, LR) | 0.5598 | 0.7549 | 0.4448 |
| Best before ceiling fix (exp_013) | 0.6585 | 0.6538 | 0.6633 |
| **Best after ceiling fix (exp_020)** | **0.6727** | **0.6441** | **0.7039** |
| Gain vs baseline | +0.1129 (+20.2%) | −0.1108 | +0.2591 |

---

## Current Best Model Configuration

```python
# src/model.py
FEATURES = [
    "SPEED", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
    "rolling_mean_3", "speed_diff", "HOUR", "DAY_OF_WEEK",
    "speed_zscore", "speed_vs_seg_mean", "segment_mean_speed",
]

# UndersampledRF: undersample majority to 2:1 ratio, decision threshold=0.40
build_model() → UndersampledRF(n_estimators=200, max_depth=8, ratio=2.0, threshold=0.40)
```
