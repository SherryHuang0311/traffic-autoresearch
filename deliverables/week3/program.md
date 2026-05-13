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
SPEED          — current observed speed for the segment
lag_1..lag_6   — speed at previous 1–6 time steps (per segment)
rolling_mean_3 — rolling 3-step mean of lag speeds (per segment)
rolling_std_3  — rolling 3-step std of lag speeds (per segment)
speed_diff     — SPEED minus lag_1 (instantaneous change)
HOUR           — hour of day (0–23)
DAY_OF_WEEK    — day of week (0=Monday … 6=Sunday)
MONTH          — month of year
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

## Ideas to Explore

- **Model class**: LogisticRegression, RandomForest, GradientBoosting, HistGradientBoosting, SVM
- **Class imbalance**: `class_weight="balanced"`, threshold tuning on predict_proba
- **Feature engineering**: more lags (4–6), rolling statistics, speed_diff, segment-relative features
- **Hyperparameters**: n_estimators, max_depth, learning_rate, regularization strength

## What NOT to Do

- Do not modify `src/run.py` (data loading, split, evaluation, logging)
- Do not modify the congestion threshold definition
- Do not use test data during development
- Do not add external data sources or downloads
- Do not hard-code validation labels into the model
- Do not change the `build_model()` or `FEATURES` signature
