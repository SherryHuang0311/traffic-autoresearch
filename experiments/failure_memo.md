# Failure Analysis Memo
**Project:** Traffic Congestion Prediction — Chicago AutoResearch Loop  
**Week:** 4  
**Experiments analyzed:** exp_001 through exp_012  

---

## What Changed and Why

Across 12 experiments, validation F1 improved from **0.5598** (baseline) to **0.6566** (exp_007),
a gain of +17.3%. The improvement did not come from a single breakthrough — it came from
two systematic decisions: fixing the class imbalance and expanding the feature set.

**Most impactful change:** Adding `class_weight="balanced"` (exp_002) — F1 jumped +13.6% in
one step. The baseline model was biased toward predicting non-congestion because congested
samples were underrepresented. Balanced weighting forced the model to treat both classes
equally, raising recall from 0.44 to 0.79. This was the dominant signal.

**Secondary gain:** Extending features from lag_1-3 to lag_1-6 plus `rolling_mean_3` and
`speed_diff` (exp_004) added +1.6% F1. These features capture slower-moving trends in speed
history that short lags miss.

**Diminishing returns on hyperparameters:** Axis A (n_estimators: 50→200) and Axis B
(max_depth: 4→12) each produced less than 0.5% F1 variation. This suggests the model has
reached a plateau under the current feature engineering and labeling setup.

---

## What Failed and What It Tells Us

**exp_005 (HGB, discarded):** HistGradientBoosting underperformed Random Forest with the
same features (0.6471 vs 0.6523). This is a signal failure — not a bug. Both models are
well-implemented; the RF's ensemble variance reduction appears better suited to this
feature space.

**exp_012 (no class_weight, discarded):** Removing balanced weighting dropped F1 to 0.624
while raising precision to 0.77. The model became conservative — it only flagged congestion
when highly confident, missing 48% of actual events. This confirms that class imbalance
is a structural feature of the dataset, not noise.

**Code failures (environment):** The loop broke when switching from the original Python
environment (3.11) to system Python (3.9). This is the most actionable failure — the
project must pin a specific Python interpreter path to prevent silent environment drift.

---

## What the Loop Still Cannot Explain

The F1 plateau (0.645–0.657 across six experiments) suggests the current features and
labeling strategy have reached their ceiling. The global congestion threshold (30th percentile
of all training speeds) treats a slow arterial road the same as a normally fast highway.
A segment-relative threshold would likely define congestion more precisely and unlock the
next layer of signal. This is the most promising direction for Week 5.

---

## Is the Loop Meaningful?

Yes. Three pieces of evidence support this:

1. **Results are reproducible:** The same model configuration (exp_003 = exp_010) produces
   identical F1 scores (0.6450) across runs, confirming the evaluation pipeline is deterministic.

2. **Results respond to real changes:** exp_012 (no class_weight) shows F1 drops 4.2% and
   precision jumps 19.7% — a coherent and expected trade-off, not random noise.

3. **Discard decisions are meaningful:** Both discarded experiments (exp_005, exp_012) had
   lower F1 than the previous best, and the log correctly captures why each was rejected.
