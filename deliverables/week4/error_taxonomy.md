# Error Taxonomy — Traffic Congestion AutoResearch Loop

## Overview

Errors observed across 12 dry-run experiments are classified into four categories,
following the Week 4 framework. Each category has a distinct cause, signature, and fix.

---

## Category 1 — Signal Failure

**Definition:** The model genuinely cannot learn the pattern, regardless of implementation quality.

| Observed instance | Evidence |
|-------------------|----------|
| Baseline LR (exp_001) had recall=0.44, missing more than half of congestion events | High precision (0.75) but low recall (0.44) — model is systematically biased toward predicting non-congestion |
| Removing class_weight (exp_012) drops F1 from 0.652 to 0.624 | Without rebalancing, the model learns to predict the majority class (non-congested), ignoring real congestion |
| Plateau effect across Axis A, B, C experiments (F1 range: 0.645–0.657) | Further tuning yields diminishing returns — the signal ceiling is being approached with current features |

**Root cause:** The global congestion threshold (30th percentile of all speeds) mixes slow arterial
roads with fast highways. A segment-relative threshold would likely expose a cleaner signal.

**Distinguishing signature:** Precision and recall move in predictable directions when the model
changes — this is a feature/labeling problem, not a code problem.

---

## Category 2 — Code Instability

**Definition:** The loop breaks due to implementation bugs, environment issues, or process conflicts.

| Observed instance | Evidence |
|-------------------|----------|
| Duplicate log entry for exp_002 | A background shell process and a foreground process both ran simultaneously, appending two identical rows to results.csv |
| `ModuleNotFoundError: No module named 'sklearn'` when switching Python executables | The experiments previously ran under a different Python environment (3.11 from pyenv); the system Python 3.9 had no packages installed |
| `python` command not found (exit code 127) | macOS does not alias `python` to Python 3; must use `python3` or full path |

**Root cause:** No pinned Python interpreter in the project; `run.py` does not validate the
environment before importing. Process isolation was not enforced during early runs.

**Distinguishing signature:** Error appears before any training happens — traceback points to
import statements or shell environment, not model logic.

---

## Category 3 — Evaluation Leakage

**Definition:** The evaluation setup changes between runs, making results incomparable.

| Observed instance | Evidence |
|-------------------|----------|
| No confirmed leakage observed in this project | The congestion threshold is always derived from `train_df` only; the split fractions are frozen constants in `run.py` |
| *Potential* leakage risk: `rolling_std_3` drops more rows via `dropna()` than other features | When `rolling_std_3` is included (exp_011), training set size could differ slightly — not currently tracked in the log |

**Root cause (potential):** Feature engineering that produces different amounts of NaN across
experiments changes the effective training set size without being logged, making cross-run
comparisons slightly imprecise.

**Distinguishing signature:** F1 changes even when the model and features are identical.
Adding a `train_size` and `val_size` column to results.csv would catch this.

---

## Category 4 — Agent Misbehavior

**Definition:** The agent takes an action outside its permitted scope or makes a decision
inconsistent with the loop rules.

| Observed instance | Evidence |
|-------------------|----------|
| exp_005 (HGB) logged with status=keep before the comparison was made | The agent ran the experiment and the runner logged it as "keep" (default); the status was corrected to "discard" manually after comparing with exp_004 |
| Agent ran a background process simultaneously with a foreground run | Caused duplicate log entry (exp_002); violates the one-run-at-a-time rule |

**Root cause:** The current loop has no automatic keep/discard enforcement — the agent reads
the output and decides manually. If the agent is interrupted mid-decision, the log can be
left in an inconsistent state.

**Distinguishing signature:** The results.csv has an entry with incorrect status, or multiple
entries for what should have been one experiment.

---

## Summary Table

| Category | Count observed | Severity | Fix |
|----------|---------------|----------|-----|
| Signal failure | 3 instances | Medium — model ceiling, not a bug | Segment-relative congestion threshold |
| Code instability | 3 instances | High — breaks the loop | Pin Python path; validate environment at startup |
| Evaluation leakage | 0 confirmed / 1 potential | Low | Log train/val row counts per experiment |
| Agent misbehavior | 2 instances | Medium — corrupts the log | Auto-compare F1 and enforce status before logging |
