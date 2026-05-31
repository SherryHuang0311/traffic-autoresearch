01 — Experiment Axis

Four axes were varied this week, each in isolation:

Axis A: n_estimators (50, 100, 200)
Axis B: max_depth (4, 8, 12)
Axis C: feature sets (minimal → extended → all features)
Axis D: class_weight (balanced vs. none)
02 — Most Important Result

Removing class_weight="balanced" (exp_012) caused F1 to drop from 0.652 to 0.624 while precision jumped to 0.77. This confirms that class imbalance is the dominant structural challenge in the dataset — the model naturally defaults to predicting non-congestion because it's the majority class. Every meaningful F1 gain in this project traces back to forcing the model to take congestion predictions seriously.

03 — Dominant Error Type

Signal failure. The F1 scores across all six controlled experiments (exp_006–011) ranged only from 0.645 to 0.657 — less than 1.2% spread despite changing model size, depth, and features. This plateau means the current features and labeling strategy have hit a ceiling. The global congestion threshold (same speed cutoff for all road segments regardless of road type) is likely the bottleneck — a fast highway and a slow side street get judged by the same threshold, which muddies the signal.

04 — Open Uncertainty

Whether the F1 plateau is caused by the labeling definition (global threshold) or by missing features (no segment-level context like road type or historical average speed per segment). These two explanations predict different fixes — one requires changing how congestion is defined, the other requires adding new features — and the current experiments can't distinguish between them.

Across all 12 experiments, three model types were used:

1. Logistic Regression (exp_001, exp_002)

Simple linear classifier
Used for the baseline and the first improvement (adding class_weight=balanced)
2. Random Forest (exp_003, exp_004, exp_006–012)

The main model for all controlled experiments
Best result: exp_007, RF with n_estimators=200, F1=0.6566
3. HistGradientBoosting (exp_005)

Tried as an alternative to Random Forest
Discarded — scored F1=0.6471, slightly below RF's 0.6523 at the time
Random Forest ended up being the best-performing model and was used for all controlled experiments in Week 4 because it gave the highest F1 and was fast enough (~3 seconds per run).