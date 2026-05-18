# Locked Final Two-Week Plan — Week 6

**Story locked:** LightGBM + segment-relative features broke the global-threshold ceiling.
The remaining two weeks refine this story and produce the final deliverable.

---

## What Is Officially Locked

- **Model family:** LightGBM with 2:1 undersampling
- **Feature set:** 17 features including all segment-relative features
- **Decision threshold:** 0.40 (fixed — no further threshold fishing)
- **Evaluation:** validation F1 only, frozen run.py, test data still locked
- **Claim:** segment-relative features are the single most impactful contribution

## What Is Officially Dropped

- HGB — tried twice, consistently loses
- XGBoost — competitive but LightGBM wins; not worth further parallel tracking
- Threshold tuning below 0.40 — ruled out
- New model families — scope is closed
- Any modification to run.py or the evaluation protocol

---

## Week 7 — Refine Within the Locked Model

**Goal:** Squeeze the last F1 points from LightGBM through disciplined hyperparameter search.
This is the one remaining open area — LightGBM hyperparameters (num_leaves, min_child_samples,
colsample_bytree, subsample) have not been tuned at all yet.

| Task | Experiments | What to vary |
|------|------------|--------------|
| LightGBM num_leaves sweep | 2 runs | 31 vs. 127 (current=63) |
| LightGBM regularization | 2 runs | min_child_samples=20, lambda_l1=0.1 |
| LightGBM subsampling | 1 run | colsample_bytree=0.8, subsample=0.8 |

**Budget:** 5 experiments max. If none beat F1=0.6780, the model is final.
**Decision rule:** keep if F1 > 0.6780, discard otherwise. No exceptions.

---

## Week 8 — Final Presentation and Submission

**Goal:** Produce presentation and write-up. No new experiments.

| Task | Output |
|------|--------|
| Final results summary | Updated results.csv + results plot |
| Feature importance analysis | Bar chart of LightGBM feature importances |
| Error analysis | What types of congestion events the model still misses |
| Final presentation slides | Story: problem → ceiling discovery → fix → results |
| Week 8 deliverable PDF | All required sections |

**Presentation narrative (locked):**
1. The prediction task and why it is hard
2. What the AutoResearch loop is and how it ran
3. The ceiling discovery: global threshold mislabels 16% of rows
4. The fix: segment-relative features (what they are, why no leakage)
5. Model comparison: RF vs LightGBM, what each tried
6. Final result: F1=0.678, +21% over baseline, +63% recall
7. Remaining limitation: label definition still global

---

## The One Open Question for Week 7

> Can LightGBM hyperparameter tuning push F1 above 0.680?
> If yes: report the specific hyperparameter and the gain.
> If no: the segment-relative feature story stands as-is — that is a complete result.
