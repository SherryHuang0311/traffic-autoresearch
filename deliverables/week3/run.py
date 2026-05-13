"""
FROZEN — Do not modify this file.
Handles data loading, preprocessing, evaluation, and logging.

Usage:
    python src/run.py "description of change"             # status=keep
    python src/run.py "description" --baseline            # status=baseline
    python src/run.py "description" --discard             # status=discard
    python src/run.py "description" --crash               # status=crash
"""
import sys
import os
import time
import csv

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Constants (frozen) ─────────────────────────────────────
DATA_PATH = "data/raw/traffic.csv"
RESULTS_FILE = "experiments/results.csv"
TRAIN_FRAC = 0.70
VALID_FRAC = 0.85
CONGESTION_QUANTILE = 0.30
FORECAST_STEPS = 3  # 30-min-ahead prediction


# ── Data loading (frozen) ──────────────────────────────────
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    df["SPEED"] = pd.to_numeric(df["SPEED"], errors="coerce")
    df = df.dropna(subset=["TIME", "SEGMENT_ID", "SPEED"])
    df = df[df["SPEED"] >= 0]
    df = df.sort_values("TIME").reset_index(drop=True)

    unique_times = df["TIME"].unique()
    n = len(unique_times)
    train_end = int(n * TRAIN_FRAC)
    valid_end = int(n * VALID_FRAC)

    train_times = set(unique_times[:train_end])
    valid_times = set(unique_times[train_end:valid_end])
    # test_times = unique_times[valid_end:]  # LOCKED — never used here

    train_df = df[df["TIME"].isin(train_times)].copy()
    valid_df = df[df["TIME"].isin(valid_times)].copy()

    # Congestion threshold derived from train only (no leakage)
    threshold = train_df["SPEED"].quantile(CONGESTION_QUANTILE)

    for split in [train_df, valid_df]:
        split["congested"] = (split["SPEED"] < threshold).astype(int)

    # Compute feature superset — model.py selects which to use
    for split in [train_df, valid_df]:
        grp = split.groupby("SEGMENT_ID")["SPEED"]
        for lag in range(1, 7):
            split[f"lag_{lag}"] = grp.shift(lag)
        split["rolling_mean_3"] = grp.transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
        split["rolling_std_3"] = grp.transform(
            lambda x: x.shift(1).rolling(3).std()
        )
        split["speed_diff"] = split["SPEED"] - split["lag_1"]
        split["target"] = (
            split.groupby("SEGMENT_ID")["congested"].shift(-FORECAST_STEPS)
        )

    train_df = train_df.dropna()
    valid_df = valid_df.dropna()
    return train_df, valid_df


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return (
        float(f1_score(y_val, y_pred)),
        float(precision_score(y_val, y_pred)),
        float(recall_score(y_val, y_pred)),
    )


# ── Logging ────────────────────────────────────────────────
def get_next_exp_id():
    if not os.path.exists(RESULTS_FILE):
        return "exp_001"
    with open(RESULTS_FILE) as f:
        n = sum(1 for line in f) - 1  # subtract header row
    return f"exp_{n + 1:03d}"


def log_result(exp_id, val_f1, val_precision, val_recall, status, description, runtime):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "experiment_id", "description", "val_f1",
                "val_precision", "val_recall", "status", "runtime_seconds",
            ])
        writer.writerow([
            exp_id, description,
            f"{val_f1:.6f}", f"{val_precision:.6f}", f"{val_recall:.6f}",
            status, f"{runtime:.2f}",
        ])


# ── Main ───────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a in ("--baseline", "--discard", "--crash"):
            status = a.lstrip("-")
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    t0 = time.time()

    train_df, valid_df = load_and_prepare()
    print(f"Data: {len(train_df)} train rows, {len(valid_df)} val rows")

    sys.path.insert(0, os.path.dirname(__file__))
    from model import build_model, FEATURES

    X_train = train_df[FEATURES]
    y_train = train_df["target"].astype(int)
    X_valid = valid_df[FEATURES]
    y_valid = valid_df["target"].astype(int)

    model = build_model()
    print(f"Model:    {model.__class__.__name__}")
    print(f"Features: {FEATURES}")

    model.fit(X_train, y_train)

    val_f1, val_precision, val_recall = evaluate(model, X_valid, y_valid)
    runtime = time.time() - t0

    print(f"val_f1:        {val_f1:.6f}")
    print(f"val_precision: {val_precision:.6f}")
    print(f"val_recall:    {val_recall:.6f}")
    print(f"runtime:       {runtime:.2f}s")

    exp_id = get_next_exp_id()
    log_result(exp_id, val_f1, val_precision, val_recall, status, description, runtime)
    print(f"Logged as {exp_id} (status={status})")


if __name__ == "__main__":
    main()
