# Ablation & Model Comparison Table — Week 6

**Current best:** exp_025 — LightGBM, full 17 features, undersample 2:1, threshold=0.40
**Previous best:** exp_020 — Random Forest, 14 features, undersample 2:1, threshold=0.40

---

## Model Comparison (all using same segment features + undersample 2:1 + threshold=0.40)

| exp | Model | Features | F1 | Precision | Recall | vs. best RF |
|-----|-------|----------|----|-----------|--------|-------------|
| exp_020 | Random Forest n=200 depth=8 | 14 | **0.6727** | 0.6441 | 0.7039 | baseline |
| exp_022 | HistGradientBoosting max_iter=300 depth=6 lr=0.05 | 14 | 0.6553 | 0.5397 | 0.8338 | −0.0174 ❌ |
| exp_023 | XGBoost n=300 depth=6 lr=0.05 | 14 | 0.6712 | 0.6253 | 0.7242 | −0.0015 ❌ |
| exp_024 | LightGBM n=300 depth=6 lr=0.05 leaves=63 | 14 | 0.6736 | 0.6274 | 0.7271 | +0.0009 ✓ |
| exp_025 | LightGBM n=300 depth=6 lr=0.05 leaves=63 | **17** | **0.6780** | 0.6379 | 0.7235 | +0.0053 ✓✓ |

**Winner: LightGBM + full 17 features**

---

## Feature Ablation (what the 3 added features in exp_025 contribute)

| Configuration | F1 | Delta |
|--------------|-----|-------|
| LightGBM, 14 features (exp_024) | 0.6736 | — |
| + rolling_std_3 + MONTH + segment_std_speed (exp_025) | 0.6780 | +0.0044 |

The three previously unused features add meaningful signal:
- `rolling_std_3` — speed volatility over the last 3 steps (spiky traffic vs. steady)
- `segment_std_speed` — how variable this road normally is (noisy vs. consistent segment)
- `MONTH` — seasonal pattern (Chicago traffic differs meaningfully by month)

---

## Full Experiment History — What Was Tested and What It Showed

| exp | Direction | F1 | Decision | What it showed |
|-----|-----------|-----|----------|----------------|
| exp_001 | Baseline LR | 0.5598 | baseline | Starting point |
| exp_002–004 | LR balanced → RF with features | 0.636→0.652 | keep | RF + features >> LR |
| exp_005 | HGB (no seg features) | 0.6471 | discard | Boosting < RF without seg features |
| exp_006–011 | RF hyperparameter grid | 0.649–0.657 | keep | Tuning ceiling ~0.658 |
| exp_012 | RF no class_weight | 0.6235 | discard | Class balance is essential |
| exp_013–018 | Downsampling + threshold sweep | 0.623–0.659 | keep/discard | Can't break 0.659 via threshold |
| exp_019–021 | Segment-relative features added | 0.664–0.673 | keep | **Ceiling break — seg features work** |
| exp_022 | HGB retry with seg features | 0.6553 | discard | HGB still worse than RF even with seg features |
| exp_023 | XGBoost with seg features | 0.6712 | discard | XGB competitive but < RF |
| exp_024 | LightGBM with seg features | 0.6736 | keep | LightGBM edges out RF |
| exp_025 | LightGBM + all 17 features | **0.6780** | **keep** | **New best — full feature set helps** |

---

## Officially Dropped Directions

| Direction | Reason |
|-----------|--------|
| Pure hyperparameter tuning of RF | Confirmed ceiling at ~0.658 regardless of n_estimators, depth, ratio |
| Threshold below 0.40 | Hurts F1 even as it raises recall; not the right lever |
| HistGradientBoosting | Tried twice (exp_005, exp_022) — consistently below RF/LightGBM |
| Redefining the congestion label | Requires changing frozen run.py — out of scope |
| External data sources | Not permitted by project rules |
