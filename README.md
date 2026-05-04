# Traffic Congestion Prediction in Chicago

## Overview
This project builds a reproducible AutoResearch pipeline to predict whether a Chicago traffic segment will be congested 30 minutes into the future using historical traffic data.

The pipeline supports automated experimentation: an agent iteratively modifies `src/model.py`, runs `src/run.py`, and logs results to `experiments/results.csv` — keeping improvements and discarding regressions.

## Research Question
Can we predict whether a Chicago traffic segment will be congested 30 minutes ahead using historical speed and time-based features?

## Dataset
This project uses a subset of the Chicago Traffic Tracker historical congestion dataset (September 2023).

The dataset contains:
- timestamp (`TIME`)
- traffic segment ID (`SEGMENT_ID`)
- estimated speed (`SPEED`)
- hour of day (`HOUR`)
- day of week (`DAY_OF_WEEK`)

Stored locally as: `data/raw/traffic.csv`

## Project Structure

```
├── src/
│   ├── model.py          # EDITABLE — agent modifies this file only
│   ├── run.py            # FROZEN — data loading, evaluation, logging
│   └── train_model.py    # Original Week 2 baseline (untouched)
├── experiments/
│   └── results.csv       # Logged results from all experiments
├── program.md            # AutoResearch loop specification
├── reflection.md         # Agent reflection and failure mode analysis
├── baseline_work.ipynb   # Exploratory notebook from Week 2
└── requirements.txt
```

## AutoResearch Loop

The agent may only modify `src/model.py`. Everything else is frozen.

```bash
# Run one experiment
python src/run.py "description of change"

# Mark as baseline
python src/run.py "description" --baseline

# If worse than current best, mark as discard
python src/run.py "description" --discard
```

Results are automatically appended to `experiments/results.csv`.

## Baseline Model
Logistic regression using current speed, 3 lag speeds, hour of day, and day of week.
- Validation F1: **0.5598** | Precision: 0.755 | Recall: 0.444

## Best Model So Far (Week 3)
Random Forest (`n=100`, `max_depth=8`, `class_weight="balanced"`) with extended features.
- Validation F1: **0.6523** | Precision: 0.573 | Recall: 0.756

## Features

The full feature set available to the agent (pre-computed by `run.py`):

| Feature | Description |
|---------|-------------|
| `SPEED` | Current observed speed |
| `lag_1` … `lag_6` | Speed at previous 1–6 time steps (per segment) |
| `rolling_mean_3` | 3-step rolling mean of lag speeds |
| `rolling_std_3` | 3-step rolling std of lag speeds |
| `speed_diff` | SPEED minus lag_1 (instantaneous change) |
| `HOUR` | Hour of day (0–23) |
| `DAY_OF_WEEK` | Day of week (0=Monday … 6=Sunday) |
| `MONTH` | Month of year |

## Target
Binary classification: will the segment be congested 30 minutes from now?

Congestion is defined as speed below the 30th percentile of training-set speeds (no leakage).

## Data Split
Deterministic and time-based:
- **Train**: earliest 70% of timestamps
- **Validation**: next 15% (used for all experiment evaluation)
- **Test**: final 15% — locked, never used during development

## Evaluation Metric
Fixed validation F1 score (binary, positive class = congested).

## Experiment Log

| exp | description | val_f1 | status |
|-----|-------------|--------|--------|
| exp_001 | Logistic Regression baseline | 0.5598 | baseline |
| exp_002 | LR + class_weight=balanced | 0.6363 | keep |
| exp_003 | Random Forest balanced | 0.6450 | keep |
| exp_004 | RF balanced + extended features (lags 1–6, rolling_mean, speed_diff) | **0.6523** | keep |
| exp_005 | HistGradientBoosting balanced + extended features | 0.6471 | discard |

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline:

```bash
python src/run.py "baseline" --baseline
```

Run any experiment:

```bash
python src/run.py "description of your change"
```

View results:

```bash
cat experiments/results.csv
```
