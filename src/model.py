"""
EDITABLE — The agent modifies this file only.
Define the feature set and model for congestion prediction.

FEATURES must be a subset of columns produced by run.py:
  SPEED, lag_1..lag_6, rolling_mean_3, rolling_std_3,
  speed_diff, HOUR, DAY_OF_WEEK, MONTH

build_model() must return an sklearn-compatible estimator.
"""
from sklearn.linear_model import LogisticRegression

FEATURES = ["SPEED", "lag_1", "lag_2", "lag_3", "HOUR", "DAY_OF_WEEK"]


def build_model():
    return LogisticRegression(max_iter=1000, class_weight="balanced")
