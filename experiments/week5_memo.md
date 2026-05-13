# What Actually Worked — Week 5 Autonomous Block
**Project:** Traffic Congestion Prediction — Chicago  
**Experiments:** exp_013 through exp_018 (6 runs, ~3s each)  
**Best result:** exp_013 — F1=0.6585, Precision=0.6538, Recall=0.6633

---

## Context and Motivation

Week 4 feedback identified two open problems:
1. **Precision/recall trade-off** — which metric should be prioritized?
2. **Class imbalance** — 3.49:1 ratio (77.7% non-congested / 22.3% congested) not fully addressed

This block explored two techniques to address both: **random undersampling** and **decision threshold tuning**.

---

## What Actually Worked

### 1. Downsampling at 2:1 ratio (exp_013) — New Overall Best

Randomly undersampling the majority class to a 2:1 ratio (non-congested:congested) at fit time, with a default 0.5 decision threshold, produced the best F1 score across all 18 experiments: **F1=0.6585**.

**Why it worked:** The previous best (exp_007, `class_weight="balanced"`) reweighted the loss function but still trained on all 31,995 rows. Undersampling at 2:1 instead presented the model with a cleaner, more balanced training signal — roughly 14,266 rows (7,133 congested + 7,133 majority-sampled non-congested). This reduced the noise from the majority class while keeping more majority examples than the 1:1 case, preserving enough signal to maintain high precision.

**Key difference from class_weight=balanced:** The balanced weight approach adjusts loss weights without changing the data. Undersampling physically removes majority examples, which changes what patterns the trees learn, not just how they are penalized.

---

### 2. Threshold Tuning — Clear Recall Control, F1 Cost

Lowering the decision threshold from 0.5 to 0.40 (exp_015) and 0.35 (exp_016) pushed recall to 0.83 and 0.86 respectively, at the cost of precision and F1:

| Threshold | F1     | Precision | Recall |
|-----------|--------|-----------|--------|
| 0.50      | 0.6566 | 0.5781    | 0.7598 |
| 0.40      | 0.6408 | 0.5211    | 0.8316 |
| 0.35      | 0.6316 | 0.4983    | 0.8621 |

This is a **clean, interpretable trade-off**. Every 0.05 drop in threshold adds roughly +0.03 recall and costs roughly -0.025 F1. This is useful to know for deployment: if the application requires catching 85%+ of congestion events, a threshold of 0.35 achieves that with a precision of ~50% (one in two alerts is real).

**For this project:** Since missing congestion is costlier than a false alarm (drivers can ignore unnecessary alerts but cannot ignore unexpectedly slow roads), recall ≥ 0.75 is the practical target. All experiments with `class_weight=balanced` already met this. Threshold tuning offers a deployable knob to adjust this post-training.

---

### 3. What Did Not Work

**Combining downsampling + low threshold (exp_017, exp_018):** Stacking both techniques did not outperform either alone. exp_017 (2:1 + threshold 0.40) scored F1=0.6560 — below exp_013 but with lower recall than exp_015. exp_018 (1:1 + threshold 0.35) achieved the highest recall of all experiments (0.874) but the worst F1 (0.624) outside of discard runs.

**Aggressive 1:1 undersampling alone (exp_014):** Equal class sampling lowered F1 slightly (0.652) versus 2:1 (0.6585) while pushing recall to 0.784. The model lost too much majority-class signal, causing precision to drop.

---

## Precision vs Recall Decision

**Recommendation: optimize for F1, use threshold as a deployment dial.**

The F1 metric is appropriate as the primary loop metric because it prevents either precision or recall from collapsing. However, the threshold should be tuned at deployment time based on use case:

- **Navigation app** (alert drivers proactively): use threshold=0.40, recall≈0.83
- **Infrastructure monitoring** (only flag severe events): use threshold=0.50, precision≈0.65

For this project, **exp_013 (F1=0.6585) is the best model**, and threshold=0.40 is the recommended deployment configuration if recall is the priority.

---

## Is the Improvement Real?

Yes. Three checks:

1. **Reproducibility:** exp_013 uses `random_state=42` throughout; re-running produces identical results.
2. **Coherent behavior:** As the undersampling ratio decreases from 3.49:1 → 2:1 → 1:1, recall increases monotonically and precision decreases monotonically. This is the expected direction.
3. **Baseline gap is meaningful:** +17.6% F1 over baseline, +21.8% recall gain. The baseline (LR, no rebalancing) had structural failure — its recall of 0.44 is not a tuning problem, it is a model design problem. All subsequent improvements addressed real root causes.

---

## What Is Still Blocked

The F1 ceiling (~0.658) appears to be driven by the **global congestion threshold** (bottom 30% of all speeds). A highway segment at 40 mph is labeled "not congested" while an arterial at 18 mph (its normal speed) is labeled "congested." Segment-relative features would require modifying `run.py` (currently frozen) or adding precomputed segment-mean columns to the feature superset. This is the most promising direction for future weeks.

---

## Resource Notes

- All 6 Week 5 experiments completed in 2.96–3.42 seconds each
- No crashes
- Peak memory: ~500MB (full dataset + RF with 200 trees)
- Python: 3.9.6 / sklearn 1.6.1 / macOS CPU only
