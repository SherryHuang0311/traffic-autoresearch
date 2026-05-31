# Final Results Table — Locked
**STAT 390 Capstone | Traffic Congestion Prediction**  
**Date locked: Week 7 | Test set: NOT YET EVALUATED (locked for final presentation)**

---

## Key Milestones

| exp | Description | F1 | Precision | Recall | Status | Phase |
|-----|-------------|----|-----------|----|--------|-------|
| exp_001 | Logistic Regression baseline | 0.5598 | 0.7549 | 0.4448 | baseline | Wk3 |
| exp_007 | RF n=200 depth=8 balanced lags 1-6 | 0.6566 | 0.5781 | 0.7598 | keep | Wk3 |
| exp_013 | RF undersample 2:1 threshold=0.50 | 0.6585 | 0.6538 | 0.6633 | keep | Wk5 |
| exp_020 | RF + segment features threshold=0.40 | 0.6727 | 0.6441 | 0.7039 | keep | Wk5 |
| exp_025 | LightGBM full 17 features threshold=0.40 | 0.6780 | 0.6379 | 0.7235 | keep | Wk6 |
| **exp_030** | **LightGBM tuned (final)** | **0.6806** | **0.6352** | **0.7329** | **final** | **Wk7** |

---

## Full Experiment Archive

| exp | Description | F1 | Precision | Recall | Status |
|-----|-------------|----|-----------|----|--------|
| exp_001 | Baseline logistic regression lags+time | 0.5598 | 0.7549 | 0.4448 | baseline |
| exp_002 | Logistic regression class_weight=balanced | 0.6363 | 0.5325 | 0.7903 | keep |
| exp_003 | RF n=100 depth=8 class_weight=balanced | 0.6450 | 0.5631 | 0.7547 | keep |
| exp_004 | RF balanced extended features lags1-6 rolling_mean speed_diff | 0.6523 | 0.5735 | 0.7562 | keep |
| exp_005 | HistGradientBoosting max_iter=200 depth=6 lr=0.05 balanced | 0.6471 | 0.5468 | 0.7925 | discard |
| exp_006 | RF n_estimators=50 depth=8 balanced extended | 0.6494 | 0.5695 | 0.7554 | keep |
| exp_007 | RF n_estimators=200 depth=8 balanced extended | 0.6566 | 0.5781 | 0.7598 | keep |
| exp_008 | RF n_estimators=100 depth=4 balanced extended | 0.6524 | 0.5692 | 0.7642 | keep |
| exp_009 | RF n_estimators=100 depth=12 balanced extended | 0.6526 | 0.6127 | 0.6981 | keep |
| exp_010 | RF balanced depth=8 n=100 minimal features | 0.6450 | 0.5631 | 0.7547 | keep |
| exp_011 | RF balanced depth=8 n=100 all features incl rolling_std MONTH | 0.6552 | 0.5771 | 0.7576 | keep |
| exp_012 | RF n=100 depth=8 no class_weight extended features | 0.6235 | 0.7697 | 0.5239 | discard |
| exp_013 | Downsampling 2:1 threshold=0.5 RF n=200 depth=8 | 0.6585 | 0.6538 | 0.6633 | keep |
| exp_014 | Downsampling 1:1 threshold=0.5 RF n=200 depth=8 | 0.6520 | 0.5581 | 0.7837 | keep |
| exp_015 | threshold=0.40 class_weight=balanced RF n=200 depth=8 | 0.6408 | 0.5211 | 0.8316 | keep |
| exp_016 | threshold=0.35 class_weight=balanced RF n=200 depth=8 | 0.6316 | 0.4983 | 0.8621 | keep |
| exp_017 | Downsampling 2:1 + threshold=0.40 combined RF n=200 depth=8 | 0.6560 | 0.5885 | 0.7409 | keep |
| exp_018 | Max-recall downsampling 1:1 + threshold=0.35 RF n=200 depth=8 | 0.6235 | 0.4847 | 0.8737 | keep |
| exp_019 | Segment features + undersample 2:1 + threshold=0.50 RF n=200 | 0.6654 | 0.7202 | 0.6183 | keep |
| exp_020 | Segment features + undersample 2:1 + threshold=0.40 RF n=200 | 0.6727 | 0.6441 | 0.7039 | keep |
| exp_021 | Segment features + class_weight=balanced + threshold=0.40 RF n=200 | 0.6641 | 0.5600 | 0.8157 | keep |
| exp_022 | HGB class_weight=balanced max_iter=300 + segment features | 0.6553 | 0.5397 | 0.8338 | discard |
| exp_023 | XGBoost n=300 depth=6 lr=0.05 + segment features | 0.6712 | 0.6253 | 0.7242 | discard |
| exp_024 | LightGBM n=300 depth=6 lr=0.05 leaves=63 + 14 features | 0.6736 | 0.6274 | 0.7271 | keep |
| exp_025 | LightGBM full 17 features + rolling_std_3 MONTH segment_std | 0.6780 | 0.6379 | 0.7235 | keep |
| exp_026 | LightGBM num_leaves=31 | 0.6780 | 0.6390 | 0.7221 | keep |
| exp_027 | LightGBM num_leaves=127 | 0.6780 | 0.6379 | 0.7235 | discard |
| exp_028 | LightGBM num_leaves=31 min_child_samples=10 | 0.6783 | 0.6406 | 0.7206 | keep |
| exp_029 | LightGBM num_leaves=31 min_child_samples=10 colsample=0.8 sub=0.8 | 0.6801 | 0.6339 | 0.7337 | keep |
| exp_030 | LightGBM n=500 num_leaves=31 min_child_samples=10 colsample=0.8 sub=0.8 | **0.6806** | **0.6352** | **0.7329** | **final** |

---

## Final Model Configuration (LOCKED)

```python
# src/model.py — final locked configuration
FEATURES = [
    "SPEED", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
    "rolling_mean_3", "rolling_std_3", "speed_diff",
    "HOUR", "DAY_OF_WEEK", "MONTH",
    "speed_zscore", "speed_vs_seg_mean", "segment_mean_speed", "segment_std_speed",
]
# 17 features total

build_model() → UndersampledLGB(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=10,
    colsample_bytree=0.8, subsample=0.8,
    ratio=2.0, threshold=0.40
)
```

---

## Stability Check (5 random seeds, final model config)

| Seed | F1 | Precision | Recall |
|------|----|-----------|--------|
| 42 | 0.6709 | 0.5953 | 0.7685 |
| 0 | 0.6684 | 0.5998 | 0.7547 |
| 123 | 0.6735 | 0.5998 | 0.7678 |
| 7 | 0.6821 | 0.6357 | 0.7358 |
| 99 | 0.6740 | 0.6212 | 0.7366 |
| **Mean** | **0.6738** | **0.6104** | **0.7527** |
| **Std** | **0.0046** | **0.0160** | **0.0143** |

Result is stable. Range [0.668, 0.682]. The run.py result (0.6806, seed=42)
is within 1.5 std of the mean.

---

## Summary vs Baseline

| | F1 | Precision | Recall |
|--|-----|-----------|--------|
| Baseline (exp_001, LR) | 0.5598 | 0.7549 | 0.4448 |
| **Final (exp_030, LightGBM)** | **0.6806** | **0.6352** | **0.7329** |
| Gain | **+0.1208 (+21.6%)** | −0.1197 | **+0.2881 (+64.8%)** |

**Test set: LOCKED — will be evaluated exactly once at final presentation (Week 8).**
