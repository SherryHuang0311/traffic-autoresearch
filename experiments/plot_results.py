"""
Generate metric-over-time plot and experiment-result matrix from results.csv.
Run from project root: python3 experiments/plot_results.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

RESULTS_FILE = "experiments/results.csv"
df = pd.read_csv(RESULTS_FILE)
df["val_f1"] = df["val_f1"].astype(float)
df["val_precision"] = df["val_precision"].astype(float)
df["val_recall"] = df["val_recall"].astype(float)
df["index"] = range(len(df))

# ── Axis labels (experiment group) ─────────────────────────
def get_group(desc):
    if "baseline" in desc:
        return "baseline"
    if "axis-A" in desc:
        return "Axis A: n_estimators"
    if "axis-B" in desc:
        return "Axis B: max_depth"
    if "axis-C" in desc:
        return "Axis C: features"
    if "axis-D" in desc:
        return "Axis D: class_weight"
    return "Week 3"

df["group"] = df["description"].apply(get_group)

# ── Colors by status ────────────────────────────────────────
color_map = {
    "baseline": "#3498db",
    "keep":     "#2ecc71",
    "discard":  "#e74c3c",
    "crash":    "#8e44ad",
}
colors = [color_map.get(s, "#95a5a6") for s in df["status"]]

# ── Best-so-far envelope ────────────────────────────────────
best_so_far = []
current_best = 0.0
for f1, status in zip(df["val_f1"], df["status"]):
    if status != "discard":
        current_best = max(current_best, f1)
    best_so_far.append(current_best)

# ── Figure ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle("Traffic Congestion Prediction — AutoResearch Loop\nMetric Over Time", fontsize=14, fontweight="bold")

# Top: F1 + precision + recall
ax = axes[0]
ax.plot(df["index"], df["val_f1"], "o-", color="#2c3e50", linewidth=1.2, markersize=6, label="F1", zorder=3)
ax.plot(df["index"], df["val_precision"], "s--", color="#e67e22", linewidth=1, markersize=5, alpha=0.7, label="Precision")
ax.plot(df["index"], df["val_recall"], "^--", color="#9b59b6", linewidth=1, markersize=5, alpha=0.7, label="Recall")
ax.plot(df["index"], best_so_far, "-", color="#2ecc71", linewidth=2.5, label="Best F1 so far", zorder=2)

# Scatter dots colored by status
for i, (idx, row) in enumerate(df.iterrows()):
    ax.scatter(i, row["val_f1"], color=color_map.get(row["status"], "gray"), s=80, zorder=4, edgecolors="white", linewidth=0.8)

ax.axhline(df[df["status"] == "baseline"]["val_f1"].values[0], color="#3498db", linestyle=":", linewidth=1.2, alpha=0.6, label="Baseline")
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0.38, 0.85)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower right")

# Add experiment group shading
group_colors = {
    "baseline": "#ecf0f1",
    "Week 3":   "#fef9e7",
    "Axis A: n_estimators": "#eafaf1",
    "Axis B: max_depth": "#eaf4fb",
    "Axis C: features": "#fdf2f8",
    "Axis D: class_weight": "#fef5e7",
}
prev_group = None
start_idx = 0
for i, g in enumerate(df["group"].tolist() + ["END"]):
    if g != prev_group and prev_group is not None:
        ax.axvspan(start_idx - 0.5, i - 0.5, alpha=0.15,
                   color=group_colors.get(prev_group, "#ecf0f1"), label="_nolegend_")
        axes[1].axvspan(start_idx - 0.5, i - 0.5, alpha=0.15,
                        color=group_colors.get(prev_group, "#ecf0f1"), label="_nolegend_")
        start_idx = i
    prev_group = g

# Bottom: bar chart of F1 by experiment
ax2 = axes[1]
bar_colors = [color_map.get(s, "#95a5a6") for s in df["status"]]
bars = ax2.bar(df["index"], df["val_f1"], color=bar_colors, edgecolor="white", linewidth=0.5, alpha=0.85)
ax2.plot(df["index"], best_so_far, "-", color="#2ecc71", linewidth=2.5, zorder=3)
ax2.axhline(df[df["status"] == "baseline"]["val_f1"].values[0], color="#3498db", linestyle=":", linewidth=1.2, alpha=0.6)
ax2.set_ylabel("Validation F1", fontsize=11)
ax2.set_xlabel("Experiment", fontsize=11)
ax2.set_ylim(0.38, 0.72)
ax2.grid(True, alpha=0.3, axis="y")

short_labels = [f"{row['experiment_id']}\n{row['description'][:18]}.." if len(row['description']) > 18 else f"{row['experiment_id']}\n{row['description']}"
                for _, row in df.iterrows()]
ax2.set_xticks(df["index"])
ax2.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=7)

# ── Legend ──────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(color="#3498db", label="baseline"),
    mpatches.Patch(color="#2ecc71", label="keep"),
    mpatches.Patch(color="#e74c3c", label="discard"),
]
ax2.legend(handles=legend_elements, fontsize=9, loc="lower right")

# Group annotations on top
ax_top = axes[0]
groups_seen = {}
for i, g in enumerate(df["group"]):
    if g not in groups_seen:
        groups_seen[g] = i
for g, start in groups_seen.items():
    if g not in ("baseline",):
        ax_top.annotate(g, xy=(start, 0.84), xycoords=("data", "axes fraction"),
                        fontsize=7.5, color="#555", style="italic",
                        ha="left", va="top")

plt.tight_layout()
out_path = "experiments/metric_over_time.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# ── Print experiment-result matrix ──────────────────────────
print("\n=== EXPERIMENT-RESULT MATRIX ===")
matrix = df[["experiment_id", "description", "val_f1", "val_precision", "val_recall", "status"]].copy()
matrix["val_f1"] = matrix["val_f1"].round(4)
matrix["val_precision"] = matrix["val_precision"].round(4)
matrix["val_recall"] = matrix["val_recall"].round(4)
print(matrix.to_string(index=False))
