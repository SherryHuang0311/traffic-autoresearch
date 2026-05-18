# Revised Project Statement — Week 6

## What This Project Does

This project builds and iteratively improves a machine learning system that predicts
whether a specific Chicago road segment will be **congested 30 minutes from now**,
using only speed history and time-of-day features — no external data, no real-time feeds.

The system runs as an AutoResearch loop: it proposes a model change, runs it, checks
the validation F1 score, keeps it if it improves, and reverts if it doesn't.

---

## What the Project Has Actually Demonstrated

**The main contribution is identifying and partially fixing a structural labeling problem.**

The Chicago traffic data uses a global congestion threshold (30th percentile of all training
speeds, ~23 mph). This threshold does not account for the fact that different road segments
have very different normal speeds. As a result, ~16% of training labels are wrong:
highways at their normal slow pace are labeled "congested," and fast side streets are not.

The fix — adding segment-relative features (per-segment mean speed, std, z-score, ratio)
computed from training data only — broke through the F1 ceiling at 0.658 that no amount of
model or hyperparameter tuning had been able to crack.

**Best result to date: F1 = 0.6780** (exp_025, LightGBM, 17 features, threshold=0.40)
vs. **baseline F1 = 0.5598** (exp_001, Logistic Regression)
= **+0.1182 F1 (+21.1%), +0.278 recall (+62.6%)**

---

## What This Project Is NOT Doing

- Not redefining the congestion label itself (run.py is frozen — label stays global)
- Not using external data sources, real-time feeds, or test data
- Not claiming to solve Chicago traffic; predicting congestion 30 minutes ahead
  on held-out validation data with a reproducible, time-based split

---

## The One-Sentence Claim

> Segment-relative speed features, combined with LightGBM and 2:1 undersampling,
> raise 30-minute-ahead congestion prediction F1 from 0.560 to 0.678 on
> Chicago Traffic Tracker data — breaking a labeling-driven ceiling that
> hyperparameter tuning alone could not cross.
