# AutoResearch Agent Instructions — Traffic Congestion Prediction

## Objective

Maximize **validation F1 score** on the Chicago traffic congestion classification task.

The task: predict whether a given traffic segment will be congested **30 minutes ahead**,
using only historical speed readings and time-based features.

---

## Rules

1. You may **ONLY** modify `src/model.py`
2. `src/run.py` is **FROZEN** — do not touch it (evaluation logic, data split, logging)
3. `src/train_model.py` is the original baseline — do not modify it
4. `build_model()` must return an sklearn-compatible estimator
5. `FEATURES` must be a subset of the columns listed below (all are pre-computed by `run.py`)
6. Training + evaluation must complete in **under 60 seconds** on CPU
7. Do not access `test_times` — the test split is **locked** and never used during development

### Available feature columns (pre-computed by run.py)
```
SPEED               — current observed speed for the segment
lag_1..lag_6        — speed at previous 1–6 time steps (per segment)
rolling_mean_3      — rolling 3-step mean of lag speeds (per segment)
rolling_std_3       — rolling 3-step std of lag speeds (per segment)
speed_diff          — SPEED minus lag_1 (instantaneous change)
HOUR                — hour of day (0–23)
DAY_OF_WEEK         — day of week (0=Monday … 6=Sunday)
MONTH               — month of year
segment_mean_speed  — per-segment historical mean speed (computed from train only)
segment_std_speed   — per-segment historical speed std (computed from train only)
speed_vs_seg_mean   — SPEED / segment_mean_speed (how fast relative to this segment's norm)
speed_zscore        — (SPEED − segment_mean) / segment_std (standard deviations from normal)
```

---

## Workflow

```
1. Read current src/model.py
2. Propose one change (model type, hyperparameter, or feature set)
3. Edit src/model.py
4. Run:  python src/run.py "description of change"
5. Check val_f1 in output
6. If improved:  git add src/model.py experiments/results.csv
                 git commit -m "feat: <description>"
7. If worse:     update status to "discard" in results.csv
                 git checkout src/model.py   (revert to previous best)
8. If crash:     log status="crash" with error description, revert
9. Repeat from step 1
```

---

## Keep / Discard / Crash Decision Rule

| Outcome | Condition | Action |
|---------|-----------|--------|
| **keep** | val_f1 > previous best val_f1 | Commit model.py + results.csv |
| **discard** | val_f1 ≤ previous best val_f1 | Revert model.py; update status in results.csv |
| **crash** | Script raises an exception | Revert model.py; log with status=crash and error message |

---

## Evaluation Metric

**Validation F1 score** (binary, positive class = congested).

- Computed on the validation split only (timestamps 70%–85% of the dataset)
- The test split (final 15%) is **never used** during development
- The congestion threshold is derived from training data only (30th percentile of SPEED)
- The data split is deterministic and time-based — it does not change between runs

---

## Logging

Every run is appended to `experiments/results.csv` with these fields:

```
experiment_id   — sequential ID (exp_001, exp_002, …)
description     — plain-text description of the change
val_f1          — validation F1 score (6 decimal places)
val_precision   — validation precision
val_recall      — validation recall
status          — baseline | keep | discard | crash
runtime_seconds — wall-clock training + eval time
```

---

## Evaluation Protection Rules

- The data split fractions (`TRAIN_FRAC=0.70`, `VALID_FRAC=0.85`) are constants in `run.py` — never modify
- The congestion threshold is always recomputed from the training split — never hardcoded
- The metric is always `sklearn.metrics.f1_score` with default `average='binary'`
- `run.py` imports `build_model` and `FEATURES` fresh on each run — no caching

---

## Week 6 Scope Lock (updated after 25 experiments)

**Story:** Segment-relative features broke an F1 ceiling at 0.658 that hyperparameter tuning
alone could not cross. LightGBM with full 17-feature set is the current best (F1=0.6780).

**Current best model:** LightGBM, n_estimators=300, max_depth=6, lr=0.05, num_leaves=63,
undersample 2:1, threshold=0.40. F1=0.6780, P=0.638, R=0.724.

**Locked search space (Week 7 only):**
- LightGBM hyperparameters: num_leaves, min_child_samples, colsample_bytree, subsample
- Budget: 5 experiments max
- Decision threshold stays at 0.40 — no further threshold tuning

**Officially dropped directions:**
- HistGradientBoosting (tried exp_005, exp_022 — consistently worse)
- XGBoost (exp_023 — competitive but loses to LightGBM)
- Threshold below 0.40 (raises recall but drops F1)
- New model families (scope closed)
- Redefining congestion label (requires frozen run.py)

## What NOT to Do

- Do not modify `src/run.py` (data loading, split, evaluation, logging)
- Do not modify the congestion threshold definition
- Do not use test data during development
- Do not add external data sources or downloads
- Do not hard-code validation labels into the model
- Do not change the `build_model()` or `FEATURES` signature
- Do not open new model families or feature engineering directions (scope is locked)
