"""
EDITABLE — The agent modifies this file only.
Define the feature set and model for congestion prediction.

FEATURES must be a subset of columns produced by run.py:
  SPEED, lag_1..lag_6, rolling_mean_3, rolling_std_3,
  speed_diff, HOUR, DAY_OF_WEEK, MONTH

build_model() must return an sklearn-compatible estimator.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

FEATURES = [
    "SPEED", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
    "rolling_mean_3", "rolling_std_3", "speed_diff",
    "HOUR", "DAY_OF_WEEK", "MONTH",
    # segment-relative
    "speed_zscore", "speed_vs_seg_mean", "segment_mean_speed", "segment_std_speed",
]


class UndersampledRF(BaseEstimator, ClassifierMixin):
    """RF with random undersampling of the majority class at fit time.
    ratio = target majority:minority ratio after sampling.
    threshold = decision threshold on predict_proba (default 0.5).
    """
    def __init__(self, n_estimators=200, max_depth=8,
                 ratio=2.0, threshold=0.5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.ratio = ratio
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        rng = np.random.RandomState(self.random_state)
        minority_idx = np.where(y == 1)[0]
        majority_idx = np.where(y == 0)[0]
        n_keep = min(int(len(minority_idx) * self.ratio), len(majority_idx))
        sampled_maj = rng.choice(majority_idx, n_keep, replace=False)
        idx = np.concatenate([minority_idx, sampled_maj])
        rng.shuffle(idx)
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf_.fit(X[idx], y[idx])
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        proba = self.rf_.predict_proba(np.array(X))[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.rf_.predict_proba(np.array(X))


class ThresholdRF(BaseEstimator, ClassifierMixin):
    """Standard RF (class_weight=balanced) with tunable decision threshold."""
    def __init__(self, n_estimators=200, max_depth=8,
                 threshold=0.5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        proba = self.rf_.predict_proba(np.array(X))[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.rf_.predict_proba(np.array(X))


class UndersampledLGB(BaseEstimator, ClassifierMixin):
    """LightGBM with random undersampling 2:1 + tunable decision threshold."""
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05,
                 num_leaves=63, min_child_samples=20,
                 colsample_bytree=1.0, subsample=1.0,
                 ratio=2.0, threshold=0.40, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.ratio = ratio
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        rng = np.random.RandomState(self.random_state)
        minority_idx = np.where(y == 1)[0]
        majority_idx = np.where(y == 0)[0]
        n_keep = min(int(len(minority_idx) * self.ratio), len(majority_idx))
        sampled_maj = rng.choice(majority_idx, n_keep, replace=False)
        idx = np.concatenate([minority_idx, sampled_maj])
        rng.shuffle(idx)
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        self.model_.fit(X[idx], y[idx])
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        proba = self.model_.predict_proba(np.array(X))[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model_.predict_proba(np.array(X))


def build_model():
    # FINAL BEST MODEL (exp_030): LightGBM fully tuned + full 17 features
    # F1=0.6806, P=0.635, R=0.733
    # n=500, depth=6, lr=0.05, num_leaves=31, min_child_samples=10,
    # colsample_bytree=0.8, subsample=0.8, undersample 2:1, threshold=0.40
    return UndersampledLGB(n_estimators=500, max_depth=6, learning_rate=0.05,
                           num_leaves=31, min_child_samples=10,
                           colsample_bytree=0.8, subsample=0.8,
                           ratio=2.0, threshold=0.40)
