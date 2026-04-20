import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

# =========================
# Start timer
# =========================
start = time.time()

# =========================
# Load data
# =========================
df = pd.read_csv("data/raw/traffic.csv")

# =========================
# Clean data
# =========================
df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
df["SPEED"] = pd.to_numeric(df["SPEED"], errors="coerce")

# Remove missing and invalid values
df = df.dropna(subset=["TIME", "SEGMENT_ID", "SPEED"])
df = df[df["SPEED"] >= 0]

# Sort by time (IMPORTANT)
df = df.sort_values("TIME")

# =========================
# Deterministic time split
# =========================
unique_times = df["TIME"].unique()
n = len(unique_times)

train_end = int(n * 0.7)
valid_end = int(n * 0.85)

train_times = unique_times[:train_end]
valid_times = unique_times[train_end:valid_end]
test_times = unique_times[valid_end:]

train_df = df[df["TIME"].isin(train_times)].copy()
valid_df = df[df["TIME"].isin(valid_times)].copy()
test_df = df[df["TIME"].isin(test_times)].copy()

# =========================
# Define congestion (TRAIN ONLY)
# =========================
threshold = train_df["SPEED"].quantile(0.3)

train_df["congested"] = (train_df["SPEED"] < threshold).astype(int)
valid_df["congested"] = (valid_df["SPEED"] < threshold).astype(int)
test_df["congested"] = (test_df["SPEED"] < threshold).astype(int)

# =========================
# Create lag features
# =========================
for lag in [1, 2, 3]:
    train_df[f"lag_{lag}"] = train_df.groupby("SEGMENT_ID")["SPEED"].shift(lag)
    valid_df[f"lag_{lag}"] = valid_df.groupby("SEGMENT_ID")["SPEED"].shift(lag)
    test_df[f"lag_{lag}"] = test_df.groupby("SEGMENT_ID")["SPEED"].shift(lag)

# =========================
# Create prediction target (30 min ahead)
# =========================
train_df["target"] = train_df.groupby("SEGMENT_ID")["congested"].shift(-3)
valid_df["target"] = valid_df.groupby("SEGMENT_ID")["congested"].shift(-3)
test_df["target"] = test_df.groupby("SEGMENT_ID")["congested"].shift(-3)

# =========================
# Drop missing values
# =========================
train_df = train_df.dropna()
valid_df = valid_df.dropna()
test_df = test_df.dropna()

# =========================
# Select features
# =========================
features = ["SPEED", "lag_1", "lag_2", "lag_3", "HOUR", "DAY_OF_WEEK"]

X_train = train_df[features]
y_train = train_df["target"].astype(int)

X_valid = valid_df[features]
y_valid = valid_df["target"].astype(int)

# =========================
# Train model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# Evaluate
# =========================
y_pred = model.predict(X_valid)

f1 = f1_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)

# =========================
# End timer
# =========================
end = time.time()
runtime = end - start

# =========================
# Print results
# =========================
print("Validation F1:", round(f1, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("Runtime:", round(runtime, 2), "seconds")

# =========================
# Save experiment log
# =========================
log = pd.DataFrame([{
    "experiment_id": "exp_001",
    "model": "logistic_regression",
    "features": "lags + time",
    "validation_f1": f1,
    "runtime_seconds": runtime
}])

log.to_csv("experiments/results.csv", index=False)