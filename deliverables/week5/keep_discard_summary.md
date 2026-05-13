# Keep / Discard / Crash Summary — All Experiments
Total experiments: 18

## BASELINE (1 runs)
  exp_001  F1=0.5598  P=0.7549  R=0.4448  — baseline logistic regression lags+time

## DISCARD (2 runs)
  exp_005  F1=0.6471  P=0.5468  R=0.7925  — HistGradientBoosting max_iter=200 depth=6 lr=0.05 balanced extended features
  exp_012  F1=0.6235  P=0.7697  R=0.5239  — axis-D RF n=100 depth=8 no class_weight extended features

## KEEP (15 runs)
  exp_002  F1=0.6363  P=0.5325  R=0.7903  — logistic regression class_weight=balanced
  exp_003  F1=0.6450  P=0.5631  R=0.7547  — random forest n=100 depth=8 class_weight=balanced
  exp_004  F1=0.6523  P=0.5735  R=0.7562  — random forest balanced extended features lags1-6 rolling_mean speed_diff
  exp_006  F1=0.6494  P=0.5695  R=0.7554  — axis-A RF n_estimators=50 depth=8 balanced extended
  exp_007  F1=0.6566  P=0.5781  R=0.7598  — axis-A RF n_estimators=200 depth=8 balanced extended
  exp_008  F1=0.6524  P=0.5692  R=0.7642  — axis-B RF n_estimators=100 depth=4 balanced extended
  exp_009  F1=0.6526  P=0.6127  R=0.6981  — axis-B RF n_estimators=100 depth=12 balanced extended
  exp_010  F1=0.6450  P=0.5631  R=0.7547  — axis-C RF balanced depth=8 n=100 minimal features speed+lag1-3+time
  exp_011  F1=0.6552  P=0.5771  R=0.7576  — axis-C RF balanced depth=8 n=100 all features incl rolling_std MONTH
  exp_013  F1=0.6585  P=0.6538  R=0.6633  — week5 downsampling ratio=2:1 threshold=0.5 RF n=200 depth=8
  exp_014  F1=0.6520  P=0.5581  R=0.7837  — week5 downsampling ratio=1:1 threshold=0.5 RF n=200 depth=8
  exp_015  F1=0.6408  P=0.5211  R=0.8316  — week5 threshold=0.40 class_weight=balanced RF n=200 depth=8
  exp_016  F1=0.6316  P=0.4983  R=0.8621  — week5 threshold=0.35 class_weight=balanced RF n=200 depth=8
  exp_017  F1=0.6560  P=0.5885  R=0.7409  — week5 downsampling 2:1 + threshold=0.40 combined RF n=200 depth=8
  exp_018  F1=0.6235  P=0.4847  R=0.8737  — week5 max-recall downsampling 1:1 + threshold=0.35 RF n=200 depth=8

## BEST vs BASELINE
Baseline (exp_001): F1=0.5598  P=0.7549  R=0.4448
Best     (exp_013): F1=0.6585  P=0.6538  R=0.6633
F1 gain: +0.0987 (17.6%)
