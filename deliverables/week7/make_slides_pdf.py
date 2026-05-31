"""
Generate final_presentation.pdf — 8 slides in landscape widescreen (13.33 × 7.5 in).
Run: python3 deliverables/week7/make_slides_pdf.py
"""
from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase.pdfmetrics import stringWidth

OUTPUT = "deliverables/week7/final_presentation.pdf"

# ── Page size: 13.33 × 7.5 inches ──────────────────────────────────────────
W = 13.33 * inch
H = 7.5  * inch

# ── Colours ─────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1a3a5c")
BLUE   = colors.HexColor("#2e6da4")
WHITE  = colors.white
LGREY  = colors.HexColor("#f2f4f8")
DGREY  = colors.HexColor("#333333")
GREEN  = colors.HexColor("#1d7a3a")
LGREEN = colors.HexColor("#d4edda")
RED    = colors.HexColor("#a31621")
LRED   = colors.HexColor("#f8d7da")
AMBER  = colors.HexColor("#856404")
LAMBER = colors.HexColor("#fff3cd")
LBLUE  = colors.HexColor("#d0e4f5")
DBLUE  = colors.HexColor("#1a5276")

def new_canvas():
    c = canvas.Canvas(OUTPUT, pagesize=(W, H))
    c.setTitle("AutoResearch: Traffic Congestion Prediction — Final Presentation")
    c.setAuthor("Sherry Huang")
    return c

# ── Drawing helpers ──────────────────────────────────────────────────────────

def filled_rect(c, x, y, w, h, fill, stroke=None):
    """x,y = bottom-left in reportlab coords."""
    c.saveState()
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(0.5)
        c.rect(x, y, w, h, fill=1, stroke=1)
    else:
        c.rect(x, y, w, h, fill=1, stroke=0)
    c.restoreState()

def text(c, txt, x, y, size=12, color=WHITE, bold=False, align="left", max_width=None):
    """Draw a single line of text. y = baseline in reportlab coords (bottom-up)."""
    c.saveState()
    c.setFillColor(color)
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    if align == "center" and max_width:
        c.drawCentredString(x + max_width / 2, y, txt)
    elif align == "right" and max_width:
        c.drawRightString(x + max_width, y, txt)
    else:
        c.drawString(x, y, txt)
    c.restoreState()

def wrap_text(c, txt, x, y, max_w, size=12, color=DGREY, bold=False,
              line_gap=4, align="left"):
    """
    Draw wrapped text. y = top of first line (reportlab bottom-up).
    Returns y of last line bottom.
    """
    font = "Helvetica-Bold" if bold else "Helvetica"
    line_h = size * 1.25 + line_gap
    words = txt.split()
    lines, current = [], ""
    for w in words:
        test = (current + " " + w).strip()
        if stringWidth(test, font, size) <= max_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    c.saveState()
    c.setFillColor(color)
    c.setFont(font, size)
    for line in lines:
        if align == "center":
            c.drawCentredString(x + max_w / 2, y - size, line)
        else:
            c.drawString(x, y - size, line)
        y -= line_h
    c.restoreState()
    return y  # y after last line

def bullet_list(c, items, x, y, max_w, size=12, color=DGREY, line_gap=3):
    """Draw bulleted items with bold prefix support (**bold** rest)."""
    font_n = "Helvetica"
    font_b = "Helvetica-Bold"
    line_h = size * 1.3 + line_gap

    for item in items:
        # Parse **bold** prefix
        bold_part, rest = "", item
        if item.startswith("**") and "**" in item[2:]:
            end = item.index("**", 2)
            bold_part = item[2:end]
            rest = item[end+2:]

        bullet = "• "
        bw = stringWidth(bullet, font_n, size)

        # Build full text for wrapping
        if bold_part:
            prefix = bullet + bold_part
            full = prefix + rest
        else:
            full = bullet + item
            prefix = ""

        # Wrap
        avail = max_w - bw
        words = full.split()
        lines, current = [], ""
        for w in words:
            test = (current + " " + w).strip()
            sw = stringWidth(test, font_n, size)
            if sw <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = "   " + w  # indent continuation
        if current:
            lines.append(current)

        c.saveState()
        c.setFillColor(color)
        for li, line in enumerate(lines):
            cy = y - size
            if li == 0 and bold_part:
                # Draw bold prefix
                c.setFont(font_b, size)
                bw2 = stringWidth(bullet + bold_part, font_b, size)
                c.drawString(x, cy, bullet + bold_part)
                c.setFont(font_n, size)
                c.drawString(x + bw2, cy, rest)
            else:
                c.setFont(font_n, size)
                c.drawString(x, cy, line)
            y -= line_h
        c.restoreState()
    return y

def header_bar(c, title, subtitle=None):
    """Navy bar at top, title + optional subtitle."""
    BAR_H = 1.05 * inch
    filled_rect(c, 0, H - BAR_H, W, BAR_H, NAVY)
    text(c, title, 0.35*inch, H - 0.55*inch, size=26, color=WHITE, bold=True)
    if subtitle:
        text(c, subtitle, 0.35*inch, H - 0.88*inch, size=12, color=LBLUE)

def slide_num(c, n, total=8):
    text(c, f"{n} / {total}", W - 0.9*inch, 0.18*inch, size=10, color=DGREY)

def table(c, rows, col_widths, x, y, row_h=0.42*inch,
          header_fill=NAVY, alt_fill=LGREY, text_size=11):
    """
    Draw table. x,y = top-left corner (y in top-down drawing coords,
    converted internally to reportlab bottom-up).
    """
    total_rows = len(rows)
    for ri, row in enumerate(rows):
        rx = x
        # Colours
        if ri == 0:
            bg = header_fill
            tc = WHITE
            fb = True
        elif ri % 2 == 1:
            bg = WHITE
            tc = DGREY
            fb = False
        else:
            bg = alt_fill
            tc = DGREY
            fb = False

        ry_bottom = H - y - (ri + 1) * row_h   # reportlab bottom-up
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            filled_rect(c, rx, ry_bottom, cw, row_h, bg)
            # Thin grid line
            c.saveState()
            c.setStrokeColor(colors.HexColor("#cccccc"))
            c.setLineWidth(0.3)
            c.rect(rx, ry_bottom, cw, row_h, fill=0, stroke=1)
            c.restoreState()
            # Text centred vertically & horizontally
            ty = ry_bottom + row_h/2 - text_size*0.35
            c.saveState()
            c.setFillColor(tc)
            c.setFont("Helvetica-Bold" if fb else "Helvetica", text_size)
            # Wrap if needed
            pad = 0.08*inch
            avail = cw - 2*pad
            words = str(cell).split()
            line, tw_lines = "", []
            for w in words:
                test = (line + " " + w).strip()
                if stringWidth(test, "Helvetica-Bold" if fb else "Helvetica", text_size) <= avail:
                    line = test
                else:
                    if line: tw_lines.append(line)
                    line = w
            if line: tw_lines.append(line)
            n_lines = len(tw_lines)
            start_y = ry_bottom + row_h/2 + (n_lines-1)*text_size*0.7
            for tl in tw_lines:
                c.drawCentredString(rx + cw/2, start_y, tl)
                start_y -= text_size * 1.3
            c.restoreState()
            rx += cw

def stat_box(c, label, val, sub, x, y, w=3.0*inch, h=1.65*inch, bg=NAVY):
    """Big number stat box. x,y = top-left."""
    rl_y = H - y - h
    filled_rect(c, x, rl_y, w, h, bg)
    text(c, label, x, H - y - 0.42*inch, size=11, color=LBLUE, bold=True,
         align="center", max_width=w)
    text(c, val,   x, H - y - 0.85*inch, size=24, color=WHITE, bold=True,
         align="center", max_width=w)
    wrap_text(c, sub, x + 0.1*inch, H - y - 1.05*inch,
              w - 0.2*inch, size=10, color=LGREY, align="center")


# ════════════════════════════════════════════════════════════════════════════
# BUILD SLIDES
# ════════════════════════════════════════════════════════════════════════════

c = new_canvas()

# ── SLIDE 1 — Title ──────────────────────────────────────────────────────────
filled_rect(c, 0, 0, W, H, NAVY)
filled_rect(c, 0, H*0.30, W, H*0.40, BLUE)

text(c, "AutoResearch for Traffic Congestion Prediction",
     0, H - 1.15*inch, size=32, color=WHITE, bold=True,
     align="center", max_width=W)
text(c, "Segment-relative features break a labeling-driven F1 ceiling",
     0, H - 1.75*inch, size=18, color=LBLUE,
     align="center", max_width=W)

text(c, "STAT 390 Capstone  •  Sherry Huang  •  Northwestern University",
     0, H - 2.5*inch, size=14, color=WHITE,
     align="center", max_width=W)

text(c, "One-sentence claim:", 1.2*inch, H - 3.15*inch, size=13, color=LAMBER, bold=True)
claim = ("Adding 4 road-specific speed features to a LightGBM model raises 30-min-ahead "
         "congestion F1 from 0.560 → 0.681 on validation (+21.6%) and 0.618 on held-out "
         "test, breaking a ceiling that 18 hyperparameter experiments could not cross.")
wrap_text(c, claim, 1.2*inch, H - 3.35*inch, W - 2.4*inch,
          size=16, color=WHITE, line_gap=3)

text(c, "30 experiments  •  0 crashes  •  Test set opened exactly once",
     0, 0.35*inch, size=12, color=LBLUE, align="center", max_width=W)
slide_num(c, 1)
c.showPage()

# ── SLIDE 2 — Contract ───────────────────────────────────────────────────────
filled_rect(c, 0, 0, W, H, LGREY)
header_bar(c, "From Topic to AutoResearch Contract",
           "How the research question was formalized")

# Left box
filled_rect(c, 0.3*inch, H - 6.55*inch, 5.8*inch, 5.3*inch, WHITE)
text(c, "The Research Question", 0.5*inch, H - 1.45*inch,
     size=14, color=NAVY, bold=True)
bullet_list(c, [
    "Can speed history alone predict congestion 30 min ahead?",
    "No real-time feeds, no external map data",
    "Binary classification per road segment",
    "Chicago Traffic Tracker: 786K rows, Sept 2023",
    "Metric: validation binary F1 (positive = congested)",
], 0.5*inch, H - 1.75*inch, 5.4*inch, size=14, color=DGREY, line_gap=5)

# Right box
filled_rect(c, 6.5*inch, H - 6.55*inch, 6.5*inch, 5.3*inch, NAVY)
text(c, "AutoResearch Contract", 6.7*inch, H - 1.45*inch,
     size=14, color=WHITE, bold=True)
bullet_list(c, [
    "Modify ONLY src/model.py",
    "src/run.py FROZEN — split, label, metric immutable",
    "Keep if val F1 improves, revert if not",
    "Git commit every improvement; git checkout every discard",
    "Test set locked — opened exactly once at the end",
    "All 30 runs logged to experiments/results.csv",
], 6.7*inch, H - 1.75*inch, 6.1*inch, size=14, color=WHITE, line_gap=5)

slide_num(c, 2)
c.showPage()

# ── SLIDE 3 — Data & Baseline ────────────────────────────────────────────────
filled_rect(c, 0, 0, W, H, LGREY)
header_bar(c, "Data, Label, and Baseline",
           "What was measured and where we started")

stats = [
    ("786,420", "rows", "Chicago Traffic Tracker, Sept 2023"),
    ("3.49 : 1", "class ratio", "77.7% not-congested vs 22.3% congested"),
    ("~23 mph", "threshold", "30th percentile of training speeds"),
    ("30 min", "forecast horizon", "predict 3 steps ahead per segment"),
]
for i, (val, lbl, sub) in enumerate(stats):
    stat_box(c, lbl, val, sub, x=(0.3 + i*3.26)*inch, y=1.2*inch)

text(c, "Key results — baseline vs. final model:",
     0.3*inch, H - 3.05*inch, size=13, color=NAVY, bold=True)
table(c, [
    ["", "F1", "Precision", "Recall", "Notes"],
    ["exp_001  Logistic Regression baseline", "0.5598", "0.7549", "0.4448", "Starting point"],
    ["exp_030  Final LightGBM (validation)",  "0.6806", "0.6352", "0.7329", "+21.6% gain"],
    ["exp_030  Final LightGBM (TEST SET)",    "0.6183", "0.5639", "0.6843", "Held-out, once"],
], [4.6*inch, 0.9*inch, 1.05*inch, 0.95*inch, 2.3*inch],
   x=0.3*inch, y=3.2*inch, row_h=0.52*inch, text_size=12)

slide_num(c, 3)
c.showPage()

# ── SLIDE 4 — Loop Design + Trace ────────────────────────────────────────────
filled_rect(c, 0, 0, W, H, LGREY)
header_bar(c, "Loop Design and Experiment Trace",
           "How the AutoResearch loop was structured and what it explored")

# Flow steps
steps = ["Read\nmodel.py", "Propose\none change", "Edit &\nrun",
         "F1\nimproved?", "Commit\n✓ keep", "Revert +\nlog discard"]
step_colors = [NAVY, BLUE, BLUE, AMBER, GREEN, RED]
bw, bh = 1.85*inch, 1.0*inch
for i, (step, sc) in enumerate(zip(steps, step_colors)):
    bx = (0.3 + i * 2.17) * inch
    by = H - 1.2*inch - bh
    filled_rect(c, bx, by, bw, bh, sc)
    # Two-line step text
    lines = step.split("\n")
    ty = by + bh/2 + (len(lines)-1)*7
    for ln in lines:
        text(c, ln, bx, ty, size=13, color=WHITE, bold=True,
             align="center", max_width=bw)
        ty -= 16
    if i < 5:
        text(c, "→", bx + bw + 0.05*inch, by + bh/2 - 6,
             size=18, color=NAVY)

text(c, "30 experiments across 7 phases:",
     0.3*inch, H - 2.45*inch, size=13, color=NAVY, bold=True)
table(c, [
    ["Phase", "Experiments", "Peak F1", "Gain", "Verdict"],
    ["Model selection (LR → RF + balance + lags)", "exp_001–007", "0.6566", "+0.097", "RF wins"],
    ["RF hyperparameter grid (depth, trees, ratio)", "exp_008–013", "0.6585", "+0.002", "Ceiling hit"],
    ["Threshold sweep (0.35–0.50)",                 "exp_014–018", "0.6585", "+0.000", "Can't break it"],
    ["Segment-relative features added",             "exp_019–021", "0.6727", "+0.014", "CEILING BROKEN"],
    ["Model comparison (RF/HGB/XGB/LightGBM)",      "exp_022–025", "0.6780", "+0.005", "LightGBM wins"],
    ["LightGBM HP tuning (leaves, subsample, n)",   "exp_026–030", "0.6806", "+0.003", "Final locked"],
], [4.1*inch, 1.75*inch, 0.9*inch, 0.78*inch, 3.5*inch],
   x=0.3*inch, y=2.6*inch, row_h=0.48*inch, text_size=11)

slide_num(c, 4)
c.showPage()

# ── SLIDE 5 — Ceiling Discovery ──────────────────────────────────────────────
filled_rect(c, 0, 0, W, H, LGREY)
header_bar(c, "The Key Finding: A Labeling-Driven Ceiling",
           "Why hyperparameter tuning hit a wall — and how we fixed it")

# Problem box (left)
filled_rect(c, 0.3*inch, H - 6.55*inch, 6.1*inch, 5.3*inch, LRED)
text(c, "The Problem", 0.5*inch, H - 1.45*inch, size=16, color=RED, bold=True)
bullet_list(c, [
    "Global threshold: speed < 23 mph → 'congested'",
    "But different roads have very different normal speeds",
    "Highway at 22 mph = normal rush-hour pace → labeled congested",
    "Side street at 20 mph = unusually fast → labeled not congested",
    "~16.3% of training rows are mislabeled",
    "No model can learn from systematically wrong labels",
    "Hard ceiling at F1 ≈ 0.658 regardless of model or tuning",
], 0.5*inch, H - 1.75*inch, 5.7*inch, size=13, color=RED, line_gap=4)

# Fix box (right)
filled_rect(c, 6.7*inch, H - 6.55*inch, 6.2*inch, 5.3*inch, LGREEN)
text(c, "The Fix (no leakage)", 6.9*inch, H - 1.45*inch, size=16, color=GREEN, bold=True)
bullet_list(c, [
    "Compute per-segment stats from training data only:",
    "  segment_mean_speed = mean(SPEED) per SEGMENT_ID",
    "  segment_std_speed  = std(SPEED)  per SEGMENT_ID",
    "  speed_zscore = (SPEED − mean) / std",
    "  speed_vs_seg_mean = SPEED / mean",
    "Merge onto train+val+test by SEGMENT_ID (no leakage)",
    "Model now asks: 'slow for THIS road?'",
    "F1: 0.6585 → 0.6727 in one step  (+0.014)",
], 6.9*inch, H - 1.75*inch, 5.8*inch, size=13, color=GREEN, line_gap=4)

slide_num(c, 5)
c.showPage()

# ── SLIDE 6 — Final Result + Stability ──────────────────────────────────────
filled_rect(c, 0, 0, W, H, LGREY)
header_bar(c, "Final Result and Stability Check",
           "exp_030: LightGBM tuned — test set opened exactly once")

# Big stat boxes
big_stats = [
    ("Validation F1", "0.6806", "model selection metric", NAVY),
    ("TEST SET F1",   "0.6183", "held-out, opened once",  GREEN),
    ("vs Baseline",   "+21.6%", "F1 gain, validation",    BLUE),
    ("Stability std", "±0.005", "across 5 random seeds",  DGREY),
]
for i, (lbl, val, sub, bg) in enumerate(big_stats):
    stat_box(c, lbl, val, sub, x=(0.3 + i*3.26)*inch, y=1.15*inch, bg=bg)

# Final model config (left)
filled_rect(c, 0.3*inch, H - 6.55*inch, 5.8*inch, 3.6*inch, NAVY)
text(c, "Final Model (exp_030)", 0.5*inch, H - 3.2*inch, size=13, color=WHITE, bold=True)
bullet_list(c, [
    "LightGBM  n_estimators=500, max_depth=6, lr=0.05",
    "num_leaves=31, min_child_samples=10",
    "colsample_bytree=0.8, subsample=0.8",
    "Undersample majority class 2:1, threshold=0.40",
    "17 features incl. all 4 segment-relative features",
], 0.5*inch, H - 3.5*inch, 5.4*inch, size=12, color=WHITE, line_gap=4)

# Stability table (right)
text(c, "Stability across 5 random seeds:",
     6.4*inch, H - 3.2*inch, size=13, color=NAVY, bold=True)
table(c, [
    ["Seed", "F1", "Precision", "Recall"],
    ["42",  "0.671", "0.595", "0.769"],
    ["0",   "0.668", "0.600", "0.755"],
    ["123", "0.674", "0.600", "0.768"],
    ["7",   "0.682", "0.636", "0.736"],
    ["99",  "0.674", "0.621", "0.737"],
    ["Mean ± Std", "0.674 ± 0.005", "0.610 ± 0.016", "0.753"],
], [1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch],
   x=6.4*inch, y=3.35*inch, row_h=0.42*inch, text_size=12)

wrap_text(c, "Val→Test gap (−0.062): test period has 24.4% congestion vs 29.7% in validation — distribution shift in the later time window. No additional tuning was done.",
          0.3*inch, H - 6.65*inch, W - 0.6*inch, size=10, color=DGREY, line_gap=2)

slide_num(c, 6)
c.showPage()

# ── SLIDE 7 — Worked vs Failed ───────────────────────────────────────────────
filled_rect(c, 0, 0, W, H, LGREY)
header_bar(c, "What Worked vs. What Failed",
           "Honest breakdown of all 30 experiments")

# Worked (left)
filled_rect(c, 0.3*inch, H - 6.55*inch, 6.1*inch, 5.3*inch, LGREEN)
text(c, "✓  Worked", 0.5*inch, H - 1.45*inch, size=16, color=GREEN, bold=True)
bullet_list(c, [
    "**Segment-relative features** → +0.014 F1 (single biggest gain)",
    "**Undersample 2:1** → cleaner precision than class_weight=balanced",
    "**LightGBM over RF** → consistent +0.003 on identical setup",
    "**Subsampling regularization** → colsample+subsample=0.8 → +0.003",
    "**Frozen evaluator** → all 30 comparisons valid and comparable",
    "**Git discipline** → zero lost experiments, zero accidental overrides",
], 0.5*inch, H - 1.78*inch, 5.7*inch, size=13, color=GREEN, line_gap=4)

# Failed (right)
filled_rect(c, 6.7*inch, H - 6.55*inch, 6.2*inch, 5.3*inch, LRED)
text(c, "✗  Did Not Work", 6.9*inch, H - 1.45*inch, size=16, color=RED, bold=True)
bullet_list(c, [
    "**HistGradientBoosting** → failed both attempts (exp_005, exp_022)",
    "**Threshold below 0.40** → raises recall, always drops F1",
    "**RF hyperparameter tuning** → 6 experiments, total gain < 0.003",
    "**XGBoost** → competitive but LightGBM wins every comparison",
    "**Fixing the label definition** → requires unfreezing run.py",
    "**Single-seed evaluation** → lucky runs kept without knowing it",
], 6.9*inch, H - 1.78*inch, 5.8*inch, size=13, color=RED, line_gap=4)

slide_num(c, 7)
c.showPage()

# ── SLIDE 8 — Reflection ─────────────────────────────────────────────────────
filled_rect(c, 0, 0, W, H, NAVY)
header_bar(c, "Reflection: Limits of the AutoResearch Loop", "")

text(c, "What did I learn about doing research with AI agents?",
     0.45*inch, H - 1.3*inch, size=16, color=LBLUE, bold=True)

boxes_data = [
    ("Agent excelled at", [
        "Systematic bookkeeping — 30 exps, 0 errors",
        "Execution speed — hours, not weeks",
        "Identifying the labeling problem (16.3% disagreement)",
        "Clean model comparison (all variables held constant)",
    ], BLUE, WHITE),
    ("Agent struggled with", [
        "No lookahead — committed to RF for 20 exps before LightGBM",
        "No repeat-on-change — HGB retry delayed 17 experiments",
        "Single-seed decisions — lucky runs not flagged",
        "Cannot diagnose why a ceiling exists — only measure it",
    ], colors.HexColor("#6c1f1f"), LRED),
    ("Human judgment irreplaceable for", [
        "Framing 'why is the ceiling there?' — not just measuring it",
        "Deciding when diminishing returns justify stopping",
        "Keeping run.py frozen — protects all comparisons retroactively",
        "Choosing which result is the story worth telling",
    ], colors.HexColor("#1d5c2e"), LGREEN),
    ("Fundamental limits", [
        "Congestion label still globally defined — residual label noise",
        "Val→Test gap (−0.062): later time window, different rate",
        "Single dataset, one month — generalizability unknown",
        "Greedy loop misses global optima by design",
    ], colors.HexColor("#5c4504"), LAMBER),
]

box_w = 6.2*inch
box_h = 2.25*inch
for i, (title, items, fill, tc) in enumerate(boxes_data):
    bx = (0.3 + (i % 2) * 6.73) * inch
    by_top = 1.65*inch + (i // 2) * 2.45*inch
    by_rl = H - by_top - box_h
    filled_rect(c, bx, by_rl, box_w, box_h, fill)
    text(c, title, bx + 0.15*inch, H - by_top - 0.3*inch, size=12, color=tc, bold=True)
    bullet_list(c, items, bx + 0.15*inch, H - by_top - 0.5*inch,
                box_w - 0.3*inch, size=11, color=tc, line_gap=3)

wrap_text(c, "Key insight: the loop shifts the bottleneck from execution to diagnosis — "
            "asking the right question matters more than running more experiments.",
          0.45*inch, 0.55*inch, W - 0.9*inch, size=12,
          color=LBLUE, bold=True, align="center")

slide_num(c, 8)
c.showPage()

# ── SAVE ─────────────────────────────────────────────────────────────────────
c.save()
print(f"PDF written to {OUTPUT}  (8 slides)")
