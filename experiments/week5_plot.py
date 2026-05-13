"""
Week 5: metric trajectory plot + keep/discard/crash summary.
Run from project root: python3 experiments/week5_plot.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_FILE = "experiments/results.csv"
df = pd.read_csv(RESULTS_FILE)
df["val_f1"]        = df["val_f1"].astype(float)
df["val_precision"] = df["val_precision"].astype(float)
df["val_recall"]    = df["val_recall"].astype(float)
df["idx"]           = range(len(df))

# ── Week labels ──────────────────────────────────────────────
def week_label(desc, i):
    if i == 0: return "W2 Baseline"
    if i <= 4:  return "Week 3"
    if i <= 11: return "Week 4"
    return "Week 5"

df["week"] = [week_label(r["description"], i) for i, r in df.iterrows()]

# ── Best-so-far (F1, ignoring discard) ──────────────────────
best, best_so_far = 0.0, []
for f1, status in zip(df["val_f1"], df["status"]):
    if status not in ("discard", "crash"):
        best = max(best, f1)
    best_so_far.append(best)

# ── Colors ──────────────────────────────────────────────────
cmap = {"baseline": "#3498db", "keep": "#2ecc71",
        "discard": "#e74c3c", "crash": "#8e44ad"}
colors = [cmap.get(s, "#95a5a6") for s in df["status"]]

# ── Figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
fig.suptitle("Traffic Congestion Prediction — AutoResearch Metric Trajectory\n"
             "18 Experiments across Weeks 3–5", fontsize=13, fontweight="bold")

# Shade week bands
week_bands = {"W2 Baseline": ("#ecf0f1", 0, 1),
              "Week 3": ("#fef9e7", 1, 5),
              "Week 4": ("#eafaf1", 5, 12),
              "Week 5": ("#eaf4fb", 12, 18)}
for ax in axes:
    for label, (color, start, end) in week_bands.items():
        ax.axvspan(start - 0.5, end - 0.5, alpha=0.18, color=color)

# ── Top: F1 trajectory ────────────────────────────────────────
ax0 = axes[0]
ax0.plot(df["idx"], df["val_f1"], "o-", color="#2c3e50",
         linewidth=1.2, markersize=5, label="F1", zorder=3)
ax0.plot(df["idx"], best_so_far, "-", color="#2ecc71",
         linewidth=2.5, label="Best F1 so far", zorder=2)
ax0.axhline(df.loc[0, "val_f1"], color="#3498db",
            linestyle=":", linewidth=1.2, alpha=0.7, label="Baseline F1")
for i, (_, row) in enumerate(df.iterrows()):
    ax0.scatter(i, row["val_f1"], color=cmap.get(row["status"], "gray"),
                s=70, zorder=4, edgecolors="white", linewidth=0.7)
ax0.set_ylabel("Validation F1", fontsize=10)
ax0.set_ylim(0.42, 0.72)
ax0.grid(True, alpha=0.3)
ax0.legend(fontsize=8, loc="lower right")

# Annotate best
best_idx = df["val_f1"].idxmax()
ax0.annotate(f"Best: {df.loc[best_idx,'val_f1']:.4f}\n({df.loc[best_idx,'experiment_id']})",
             xy=(best_idx, df.loc[best_idx,"val_f1"]),
             xytext=(best_idx + 0.5, df.loc[best_idx,"val_f1"] - 0.025),
             fontsize=7.5, color="#27ae60",
             arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1))

# Week labels at top
for label, (color, start, end) in week_bands.items():
    mid = (start + end - 1) / 2
    ax0.text(mid, 0.715, label, ha="center", va="top", fontsize=8,
             color="#555", style="italic", fontweight="bold")

# ── Middle: Precision / Recall ────────────────────────────────
ax1 = axes[1]
ax1.plot(df["idx"], df["val_precision"], "s-", color="#e67e22",
         linewidth=1.2, markersize=5, label="Precision")
ax1.plot(df["idx"], df["val_recall"], "^-", color="#9b59b6",
         linewidth=1.2, markersize=5, label="Recall")
ax1.axhline(df.loc[0,"val_precision"], color="#e67e22",
            linestyle=":", linewidth=1, alpha=0.5)
ax1.axhline(df.loc[0,"val_recall"], color="#9b59b6",
            linestyle=":", linewidth=1, alpha=0.5)
ax1.set_ylabel("Precision / Recall", fontsize=10)
ax1.set_ylim(0.38, 0.95)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc="upper right")

# ── Bottom: Bar chart colored by status ───────────────────────
ax2 = axes[2]
ax2.bar(df["idx"], df["val_f1"], color=colors,
        edgecolor="white", linewidth=0.5, alpha=0.85)
ax2.plot(df["idx"], best_so_far, "-", color="#2ecc71", linewidth=2, zorder=3)
ax2.axhline(df.loc[0,"val_f1"], color="#3498db",
            linestyle=":", linewidth=1.2, alpha=0.7)
ax2.set_ylabel("Validation F1", fontsize=10)
ax2.set_xlabel("Experiment", fontsize=10)
ax2.set_ylim(0.42, 0.70)
ax2.grid(True, alpha=0.3, axis="y")

labels = [f"{r['experiment_id']}\n{r['description'][:16]}.."
          if len(r["description"]) > 16 else f"{r['experiment_id']}\n{r['description']}"
          for _, r in df.iterrows()]
ax2.set_xticks(df["idx"])
ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)

legend_els = [mpatches.Patch(color=c, label=l)
              for l, c in cmap.items()]
ax2.legend(handles=legend_els, fontsize=8, loc="lower right")

plt.tight_layout()
out = "experiments/week5_metric_trajectory.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# ── Keep / Discard / Crash summary ───────────────────────────
print("\n=== KEEP / DISCARD / CRASH SUMMARY ===")
summary = df.groupby("status").agg(
    count=("experiment_id", "count"),
    avg_f1=("val_f1", "mean"),
    best_f1=("val_f1", "max"),
    worst_f1=("val_f1", "min"),
).round(4)
print(summary.to_string())

print("\n=== BEST vs BASELINE ===")
baseline = df[df["status"] == "baseline"].iloc[0]
best_row = df.loc[df["val_f1"].idxmax()]
print(f"Baseline  ({baseline['experiment_id']}): F1={baseline['val_f1']:.4f}  "
      f"P={baseline['val_precision']:.4f}  R={baseline['val_recall']:.4f}")
print(f"Best      ({best_row['experiment_id']}): F1={best_row['val_f1']:.4f}  "
      f"P={best_row['val_precision']:.4f}  R={best_row['val_recall']:.4f}")
print(f"F1 gain:  +{(best_row['val_f1']-baseline['val_f1']):.4f}  "
      f"({(best_row['val_f1']-baseline['val_f1'])/baseline['val_f1']*100:.1f}%)")
print(f"Recall gain: +{(best_row['val_recall']-baseline['val_recall']):.4f}")

print("\n=== FULL RESULT MATRIX (Week 5 only) ===")
w5 = df[df["idx"] >= 12][["experiment_id","description","val_f1",
                            "val_precision","val_recall","status"]]
w5 = w5.copy()
w5["val_f1"] = w5["val_f1"].round(4)
w5["val_precision"] = w5["val_precision"].round(4)
w5["val_recall"] = w5["val_recall"].round(4)
print(w5.to_string(index=False))
