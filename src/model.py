"""
EDITABLE — The agent modifies this file only.
Define the feature set and model for congestion prediction.

FEATURES must be a subset of columns produced by run.py:
  SPEED, lag_1..lag_6, rolling_mean_3, rolling_std_3,
  speed_diff, HOUR, DAY_OF_WEEK, MONTH

build_model() must return an sklearn-compatible estimator.
"""
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "SPEED", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
    "rolling_mean_3", "speed_diff",
    "HOUR", "DAY_OF_WEEK",
]


def build_model():
    return RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
