# Week 3 Reflection — AutoResearch Loop Dry Runs

## Experiment Summary

| exp_id | description | val_f1 | status |
|--------|-------------|--------|--------|
| exp_001 | baseline logistic regression lags+time | 0.5598 | baseline |
| exp_002 | logistic regression class_weight=balanced | 0.6363 | keep |
| exp_003 | random forest n=100 depth=8 class_weight=balanced | 0.6450 | keep |
| exp_004 | random forest balanced + extended features (lags 1–6, rolling_mean, speed_diff) | 0.6523 | keep |
| exp_005 | HistGradientBoosting max_iter=200 depth=6 lr=0.05 balanced + extended features | 0.6471 | discard |

Best result so far: **exp_004**, F1 = 0.6523  
Baseline: F1 = 0.5598 (+16.5% improvement)

---

## What the Agent Did Well

**1. Executed the loop cleanly end-to-end.**
Every experiment ran without manual intervention. The pipeline loaded data, trained, evaluated on validation only, logged results, and exited cleanly within 4 seconds per run.

**2. Correctly identified and targeted the core weakness.**
The baseline had high precision (0.75) but low recall (0.44) — it was systematically missing real congestion events. The first modification (`class_weight="balanced"`) directly addressed this, boosting recall from 0.44 → 0.79 and F1 from 0.56 → 0.64 in one step.

**3. Kept experiments isolated and comparable.**
Each run used the identical data split, congestion threshold, and metric. Results are directly comparable because `run.py` is frozen and deterministic.

**4. Correctly reverted a failed experiment.**
exp_005 (HistGradientBoosting) scored 0.647 < 0.652 (current best). The agent marked it as `discard` and reverted `model.py` to exp_004. The log still shows the run for auditability.

**5. Made one change per experiment.**
Each commit touches `src/model.py` only, making it easy to trace exactly what changed between experiments.

---

## What the Agent Did Badly

**1. Caused a duplicate log entry in exp_002.**
A background process and a foreground process both ran the same experiment simultaneously, resulting in two logged rows for the same change. This had to be manually cleaned up. Root cause: running background tasks when foreground is sufficient.

**2. No automatic revert mechanism.**
The keep/discard decision is made by the agent reading the output, then manually reverting. If the agent were interrupted mid-decision, `model.py` could be left in an inconsistent state. A proper loop would automate the revert via git stash or a checkpoint file.

**3. Blind to segment-level heterogeneity.**
All five experiments used a global congestion threshold (bottom 30% of all speeds). This conflates slow arterial roads with fast highways. A segment-relative threshold would likely improve F1 meaningfully, but this requires restructuring the feature engineering in `run.py` — which is currently frozen.

**4. No hyperparameter search.**
Experiments explored model families (LR → RF → HGB) but did not systematically tune within a family. Grid search or even a few targeted variations within the best model class (RF) could yield further gains.

---

## Common Failure Modes Encountered

**1. Duplicate experiment logging (critical)**  
*Cause:* Starting a background shell command while also running foreground, both appending to the same CSV.  
*Fix:* Always run `python src/run.py` in foreground; never use background jobs for experiment runs.

**2. Experiment ID counter out of sync (minor)**  
*Cause:* `get_next_exp_id()` counts CSV lines — if the file is manually edited between runs, the counter drifts.  
*Fix:* Parse the last `experiment_id` value from the CSV rather than counting lines.

**3. Stale module import cache (potential)**  
*Cause:* If `model.py` is modified and `run.py` is imported inside the same Python process, the old version of `model.py` may be cached. In the current design each run is a fresh subprocess, so this is not yet a problem — but it would be if the runner were ever embedded in a loop within Python.  
*Fix:* Always invoke `run.py` as a subprocess; never import it as a module.

**4. Feature name mismatch crash (potential)**  
*Cause:* `FEATURES` in `model.py` can list any string. If it references a column that `run.py` does not produce, the script crashes with a `KeyError`.  
*Fix:* Add a validation check at the top of `run.py` that confirms every name in `FEATURES` exists in the prepared DataFrame before training.

**5. Test data exposure risk (structural)**  
*Cause:* `run.py` computes test-split rows internally (they are simply never passed to the model), but nothing prevents a future agent modification from accessing them via `load_and_prepare()`.  
*Fix:* `load_and_prepare()` should return only `(train_df, valid_df)` — the test split should not be computed at all during the loop, or should be split into a completely separate locked file.

---

## Open Questions in program.md

1. Should the agent be allowed to change the congestion threshold quantile (currently 0.30)? Changing it changes the label definition and makes experiments incomparable — this should probably be frozen.
2. Is `rolling_std_3` safe to use as a feature? It produces NaN for the first two rows per segment; `run.py` drops these via `dropna()`, which reduces training set size slightly.
3. Should the metric be macro-F1 or binary-F1? Currently binary (positive = congested). If the class balance shifts with threshold changes, the metric interpretation changes too.
