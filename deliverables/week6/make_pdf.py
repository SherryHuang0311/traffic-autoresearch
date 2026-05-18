"""
Generate week6_deliverables.pdf using reportlab.
Run: python3 deliverables/week6/make_pdf.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUTPUT = "deliverables/week6/week6_deliverables.pdf"

DARK_BLUE   = colors.HexColor("#1a3a5c")
MID_BLUE    = colors.HexColor("#2e6da4")
LIGHT_BLUE  = colors.HexColor("#d0e4f5")
GREEN       = colors.HexColor("#1d7a3a")
LIGHT_GREEN = colors.HexColor("#d4edda")
RED         = colors.HexColor("#a31621")
LIGHT_RED   = colors.HexColor("#f8d7da")
GREY_BG     = colors.HexColor("#f5f5f5")
BORDER      = colors.HexColor("#cccccc")
AMBER_BG    = colors.HexColor("#fff3cd")
AMBER_BD    = colors.HexColor("#856404")

styles = getSampleStyleSheet()

title_style = ParagraphStyle("T", parent=styles["Normal"], fontSize=20, leading=26,
    textColor=DARK_BLUE, spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold")
subtitle_style = ParagraphStyle("ST", parent=styles["Normal"], fontSize=11, leading=14,
    textColor=MID_BLUE, spaceAfter=2, alignment=TA_CENTER, fontName="Helvetica")
section_style = ParagraphStyle("S", parent=styles["Normal"], fontSize=13, leading=16,
    textColor=DARK_BLUE, spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
subsection_style = ParagraphStyle("SS", parent=styles["Normal"], fontSize=11, leading=14,
    textColor=MID_BLUE, spaceBefore=8, spaceAfter=4, fontName="Helvetica-Bold")
body_style = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=14,
    textColor=colors.black, spaceAfter=5, fontName="Helvetica")
body_bold = ParagraphStyle("BB", parent=styles["Normal"], fontSize=10, leading=14,
    textColor=colors.black, spaceAfter=5, fontName="Helvetica-Bold")
small_style = ParagraphStyle("SM", parent=styles["Normal"], fontSize=8.5, leading=12,
    textColor=colors.HexColor("#222222"), spaceAfter=3, fontName="Helvetica")
small_bold = ParagraphStyle("SMB", parent=styles["Normal"], fontSize=8.5, leading=12,
    textColor=colors.HexColor("#222222"), spaceAfter=3, fontName="Helvetica-Bold")
italic_style = ParagraphStyle("IT", parent=styles["Normal"], fontSize=10, leading=14,
    textColor=colors.HexColor("#333333"), spaceAfter=5, fontName="Helvetica-Oblique")

def HR(): return HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=6, spaceBefore=2)
def SP(n=8): return Spacer(1, n)

def box(content_para, bg=LIGHT_GREEN, border=GREEN):
    t = Table([[content_para]], colWidths=[6.3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("BOX", (0,0), (-1,-1), 1.2, border),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    return t

def two_col(rows, w1=1.5*inch, w2=4.8*inch, header_bg=DARK_BLUE):
    t = Table(rows, colWidths=[w1, w2])
    cmds = [
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY_BG]),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]
    t.setStyle(TableStyle(cmds))
    return t

def build():
    doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.85*inch, bottomMargin=0.85*inch)
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(SP(16))
    story.append(Paragraph("Week 6 Deliverables", title_style))
    story.append(Paragraph("Traffic Congestion Prediction — AutoResearch", subtitle_style))
    story.append(Paragraph("STAT 390 Capstone &nbsp;|&nbsp; Sherry Huang &nbsp;|&nbsp; 2026", subtitle_style))
    story.append(SP(8))
    story.append(HR())
    story.append(SP(4))
    story.append(box(Paragraph(
        "<b>New best result (exp_025):</b> &nbsp; F1 = 0.6780 &nbsp;|&nbsp; "
        "Precision = 0.638 &nbsp;|&nbsp; Recall = 0.724<br/>"
        "Model: LightGBM, n=300, depth=6, lr=0.05, num_leaves=63, "
        "undersample 2:1, threshold=0.40, 17 features",
        body_style)))
    story.append(SP(4))
    story.append(Paragraph(
        "<b>vs baseline (exp_001):</b> &nbsp; F1 +0.1182 (+21.1%) &nbsp;|&nbsp; "
        "Recall +0.2793 (+62.8%) &nbsp;|&nbsp; 25 experiments, 0 crashes",
        small_style))
    story.append(SP(6))
    story.append(Paragraph(
        "Contents: &nbsp; 1 — Revised Project Statement &nbsp;|&nbsp; "
        "2 — Updated Agent Strategy &nbsp;|&nbsp; "
        "3 — Ablation / Comparison Table &nbsp;|&nbsp; "
        "4 — Locked Final Two-Week Plan", small_style))

    story.append(PageBreak())

    # ── SECTION 1: REVISED PROJECT STATEMENT ──────────────────────────────
    story.append(Paragraph("1 &nbsp; Revised Project Statement", section_style))
    story.append(HR())

    story.append(Paragraph("What This Project Does", subsection_style))
    story.append(Paragraph(
        "This project builds and iteratively improves a machine learning system that predicts "
        "whether a specific Chicago road segment will be <b>congested 30 minutes from now</b>, "
        "using only speed history and time-of-day features. No external data, no real-time feeds. "
        "The system runs as an AutoResearch loop: propose a change, run it, keep if F1 improves, "
        "revert if not.", body_style))

    story.append(Paragraph("What the Project Has Actually Demonstrated", subsection_style))
    story.append(Paragraph(
        "The main contribution is <b>identifying and partially fixing a structural labeling problem.</b>",
        body_bold))
    story.append(Paragraph(
        "The Chicago traffic data uses a global congestion threshold (~23 mph, 30th percentile "
        "of all training speeds). This threshold does not account for the fact that different "
        "road segments have very different normal speeds. As a result, <b>~16% of training labels "
        "are incorrect</b>: highways at their normal slow pace are labeled congested, and fast "
        "side streets are not.", body_style))
    story.append(Paragraph(
        "The fix — adding segment-relative features (per-segment mean speed, std, z-score, ratio) "
        "computed from training data only — broke through the F1 ceiling at 0.658 that no amount "
        "of model or hyperparameter tuning had been able to cross.", body_style))

    prog_rows = [
        [Paragraph("<b>Stage</b>", small_bold), Paragraph("<b>Best F1</b>", small_bold), Paragraph("<b>What drove it</b>", small_bold)],
        [Paragraph("Baseline (exp_001)", small_style), Paragraph("0.5598", small_style), Paragraph("Logistic regression, lags + time only", small_style)],
        [Paragraph("Model selection (exp_003–007)", small_style), Paragraph("0.6566", small_style), Paragraph("Random Forest + extended lag features", small_style)],
        [Paragraph("Imbalance handling (exp_013–018)", small_style), Paragraph("0.6585", small_style), Paragraph("Undersampling 2:1 — ceiling reached", small_style)],
        [Paragraph("Segment features (exp_019–021)", small_style), Paragraph("0.6727", small_style), Paragraph("speed_zscore, speed_vs_seg_mean, segment_mean — ceiling broken", small_style)],
        [Paragraph("<b>LightGBM + all features (exp_025)</b>", small_bold), Paragraph("<b>0.6780</b>", small_bold), Paragraph("<b>LightGBM edges RF; full 17-feature set</b>", small_bold)],
    ]
    pt = Table(prog_rows, colWidths=[1.7*inch, 0.75*inch, 3.85*inch])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY_BG]),
        ("BACKGROUND", (0,5), (-1,5), LIGHT_GREEN),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(pt)
    story.append(SP(8))

    story.append(Paragraph("What This Project Is NOT Doing", subsection_style))
    not_doing = [
        "Not redefining the congestion label (run.py is frozen — label stays global)",
        "Not using external data, real-time feeds, or test data",
        "Not claiming to solve Chicago traffic — predicting congestion 30 min ahead on held-out validation data with a reproducible time-based split",
    ]
    for nd in not_doing:
        story.append(Paragraph(f"&#8226; &nbsp; {nd}", body_style))

    story.append(SP(6))
    story.append(box(Paragraph(
        "<b>The one-sentence claim:</b><br/>"
        "<i>Segment-relative speed features, combined with LightGBM and 2:1 undersampling, "
        "raise 30-minute-ahead congestion prediction F1 from 0.560 to 0.678 on Chicago Traffic "
        "Tracker data — breaking a labeling-driven ceiling that hyperparameter tuning alone "
        "could not cross.</i>", body_style), bg=LIGHT_BLUE, border=MID_BLUE))

    story.append(PageBreak())

    # ── SECTION 2: UPDATED AGENT STRATEGY ─────────────────────────────────
    story.append(Paragraph("2 &nbsp; Updated Agent Strategy (program.md)", section_style))
    story.append(HR())

    story.append(Paragraph("Scope Lock — Week 6", subsection_style))
    story.append(box(Paragraph(
        "<b>Current best:</b> LightGBM, n_estimators=300, max_depth=6, lr=0.05, "
        "num_leaves=63, undersample 2:1, threshold=0.40, 17 features. "
        "F1=0.6780, P=0.638, R=0.724.", body_style), bg=LIGHT_GREEN, border=GREEN))
    story.append(SP(6))

    story.append(Paragraph("Locked Search Space (Week 7 only)", subsection_style))
    story.append(Paragraph(
        "Only LightGBM hyperparameters are eligible for further tuning. "
        "Budget is 5 experiments maximum. Decision threshold stays at 0.40.", body_style))

    hp_rows = [
        [Paragraph("<b>Parameter</b>", small_bold), Paragraph("<b>Current</b>", small_bold), Paragraph("<b>Values to try</b>", small_bold)],
        [Paragraph("num_leaves", small_style), Paragraph("63", small_style), Paragraph("31, 127", small_style)],
        [Paragraph("min_child_samples", small_style), Paragraph("20 (default)", small_style), Paragraph("10, 50", small_style)],
        [Paragraph("colsample_bytree", small_style), Paragraph("1.0 (default)", small_style), Paragraph("0.8", small_style)],
        [Paragraph("subsample", small_style), Paragraph("1.0 (default)", small_style), Paragraph("0.8", small_style)],
    ]
    ht = Table(hp_rows, colWidths=[1.8*inch, 1.4*inch, 3.1*inch])
    ht.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY_BG]),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(ht)
    story.append(SP(8))

    story.append(Paragraph("Officially Dropped Directions", subsection_style))
    dropped = [
        ("HistGradientBoosting", "Tried exp_005 (no seg features) and exp_022 (with seg features) — consistently below RF and LightGBM"),
        ("XGBoost", "exp_023 scored F1=0.6712 — competitive but loses to LightGBM on same setup"),
        ("Threshold below 0.40", "Raises recall but always drops F1; tried exp_015, 016, 018"),
        ("New model families", "Scope is closed per Week 6 lock"),
        ("Redefining congestion label", "Requires changing frozen run.py — out of scope"),
        ("New features beyond current 17", "Full available superset is now used"),
    ]
    drop_rows = [[Paragraph("<b>Direction</b>", small_bold), Paragraph("<b>Reason dropped</b>", small_bold)]]
    for d in dropped:
        drop_rows.append([Paragraph(d[0], small_style), Paragraph(d[1], small_style)])
    dt = Table(drop_rows, colWidths=[1.8*inch, 4.5*inch])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6c1f1f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_RED]),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(dt)

    story.append(PageBreak())

    # ── SECTION 3: ABLATION / COMPARISON TABLE ─────────────────────────────
    story.append(Paragraph("3 &nbsp; Ablation and Comparison Table", section_style))
    story.append(HR())

    story.append(Paragraph("3a &nbsp; Model Comparison (same segment features + undersample 2:1 + threshold=0.40)", subsection_style))
    story.append(Paragraph(
        "All four model families tested head-to-head with identical setup. "
        "LightGBM wins on F1 and trains fastest.", body_style))

    model_rows = [
        [Paragraph("<b>exp</b>", small_bold), Paragraph("<b>Model</b>", small_bold),
         Paragraph("<b>F1</b>", small_bold), Paragraph("<b>Prec</b>", small_bold),
         Paragraph("<b>Recall</b>", small_bold), Paragraph("<b>vs RF</b>", small_bold), Paragraph("<b>Status</b>", small_bold)],
        [Paragraph("exp_020", small_style), Paragraph("Random Forest n=200 depth=8", small_style),
         Paragraph("0.6727", small_style), Paragraph("0.644", small_style), Paragraph("0.704", small_style),
         Paragraph("—", small_style), Paragraph("prev best", small_style)],
        [Paragraph("exp_022", small_style), Paragraph("HistGradientBoosting max_iter=300 depth=6 lr=0.05", small_style),
         Paragraph("0.6553", small_style), Paragraph("0.540", small_style), Paragraph("0.834", small_style),
         Paragraph("−0.017", small_style), Paragraph("discard", small_style)],
        [Paragraph("exp_023", small_style), Paragraph("XGBoost n=300 depth=6 lr=0.05", small_style),
         Paragraph("0.6712", small_style), Paragraph("0.625", small_style), Paragraph("0.724", small_style),
         Paragraph("−0.002", small_style), Paragraph("discard", small_style)],
        [Paragraph("exp_024", small_style), Paragraph("LightGBM n=300 depth=6 lr=0.05 leaves=63", small_style),
         Paragraph("0.6736", small_style), Paragraph("0.627", small_style), Paragraph("0.727", small_style),
         Paragraph("+0.001", small_style), Paragraph("keep", small_style)],
        [Paragraph("<b>exp_025</b>", small_bold), Paragraph("<b>LightGBM — same + 3 extra features</b>", small_bold),
         Paragraph("<b>0.6780</b>", small_bold), Paragraph("<b>0.638</b>", small_bold), Paragraph("<b>0.724</b>", small_bold),
         Paragraph("<b>+0.005</b>", small_bold), Paragraph("<b>best</b>", small_bold)],
    ]
    mt = Table(model_rows, colWidths=[0.6*inch, 2.3*inch, 0.55*inch, 0.52*inch, 0.6*inch, 0.6*inch, 0.6*inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY_BG]),
        ("BACKGROUND", (0,5), (-1,5), LIGHT_GREEN),
        ("BACKGROUND", (5,2), (5,3), LIGHT_RED),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,0), (-1,-1), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(mt)
    story.append(SP(10))

    story.append(Paragraph("3b &nbsp; Feature Ablation (on LightGBM)", subsection_style))
    feat_rows = [
        [Paragraph("<b>Configuration</b>", small_bold), Paragraph("<b>Features</b>", small_bold),
         Paragraph("<b>F1</b>", small_bold), Paragraph("<b>Delta</b>", small_bold)],
        [Paragraph("LightGBM, core 14 features (exp_024)", small_style),
         Paragraph("SPEED, lag_1-6, rolling_mean_3, speed_diff, HOUR, DAY_OF_WEEK, speed_zscore, speed_vs_seg_mean, segment_mean_speed", small_style),
         Paragraph("0.6736", small_style), Paragraph("—", small_style)],
        [Paragraph("+ rolling_std_3 + MONTH + segment_std_speed (exp_025)", small_style),
         Paragraph("All 17 available features", small_style),
         Paragraph("0.6780", small_style), Paragraph("+0.0044", small_style)],
    ]
    ft = Table(feat_rows, colWidths=[2.0*inch, 2.85*inch, 0.7*inch, 0.75*inch])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_GREEN]),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(ft)
    story.append(SP(6))
    story.append(Paragraph(
        "The three added features each capture a different kind of road-specific variability: "
        "<b>rolling_std_3</b> (speed volatility over last 3 steps), "
        "<b>segment_std_speed</b> (how inherently noisy this road is), "
        "<b>MONTH</b> (Chicago seasonal patterns).", body_style))

    story.append(SP(8))
    story.append(Paragraph("3c &nbsp; Full Experiment History — Summary by Direction", subsection_style))

    hist_rows = [
        [Paragraph("<b>Experiments</b>", small_bold), Paragraph("<b>Direction</b>", small_bold),
         Paragraph("<b>F1 range</b>", small_bold), Paragraph("<b>Verdict</b>", small_bold)],
        [Paragraph("exp_001–002", small_style), Paragraph("Logistic Regression baseline + balanced", small_style),
         Paragraph("0.560–0.636", small_style), Paragraph("LR ceiling found", small_style)],
        [Paragraph("exp_003–007", small_style), Paragraph("Random Forest model selection + n_estimators", small_style),
         Paragraph("0.645–0.657", small_style), Paragraph("RF wins over LR", small_style)],
        [Paragraph("exp_008–011", small_style), Paragraph("RF depth + feature set sweep", small_style),
         Paragraph("0.645–0.656", small_style), Paragraph("Tuning ceiling ~0.658", small_style)],
        [Paragraph("exp_012", small_style), Paragraph("RF no class weighting", small_style),
         Paragraph("0.623", small_style), Paragraph("Class balance essential — discard", small_style)],
        [Paragraph("exp_013–018", small_style), Paragraph("Undersampling ratios + threshold sweep", small_style),
         Paragraph("0.623–0.659", small_style), Paragraph("Cannot break 0.659 via threshold", small_style)],
        [Paragraph("exp_019–021", small_style), Paragraph("Segment-relative features added", small_style),
         Paragraph("0.664–0.673", small_style), Paragraph("Ceiling broken — key contribution", small_style)],
        [Paragraph("exp_022", small_style), Paragraph("HGB retry with segment features", small_style),
         Paragraph("0.655", small_style), Paragraph("Still loses to RF — discard", small_style)],
        [Paragraph("exp_023", small_style), Paragraph("XGBoost with segment features", small_style),
         Paragraph("0.671", small_style), Paragraph("Close but loses — discard", small_style)],
        [Paragraph("exp_024–025", small_style), Paragraph("LightGBM (14 then 17 features)", small_style),
         Paragraph("0.674–0.678", small_style), Paragraph("New best — locked", small_style)],
    ]
    htab = Table(hist_rows, colWidths=[1.0*inch, 2.5*inch, 0.9*inch, 1.9*inch])
    htab.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY_BG]),
        ("BACKGROUND", (0,7), (-1,9), LIGHT_GREEN),
        ("BACKGROUND", (0,4), (-1,4), LIGHT_RED),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(htab)

    story.append(PageBreak())

    # ── SECTION 4: LOCKED TWO-WEEK PLAN ───────────────────────────────────
    story.append(Paragraph("4 &nbsp; Locked Final Two-Week Plan", section_style))
    story.append(HR())

    story.append(box(Paragraph(
        "<b>Story locked:</b> LightGBM + segment-relative features broke the global-threshold ceiling. "
        "Weeks 7–8 refine and present this story — no new directions, no new model families.",
        body_style), bg=AMBER_BG, border=AMBER_BD))
    story.append(SP(8))

    story.append(Paragraph("The One Open Question", subsection_style))
    story.append(box(Paragraph(
        "Can LightGBM hyperparameter tuning push F1 above 0.680? "
        "If yes: report the specific parameter and the gain. "
        "If no: F1=0.678 is the final result — that is still a complete and defensible story.",
        body_style), bg=LIGHT_BLUE, border=MID_BLUE))
    story.append(SP(8))

    story.append(Paragraph("Week 7 — Hyperparameter Refinement Within LightGBM", subsection_style))
    story.append(Paragraph(
        "Budget: 5 experiments max. Decision threshold stays at 0.40. "
        "Keep if F1 &gt; 0.6780, discard otherwise. No exceptions.", body_style))

    w7_rows = [
        [Paragraph("<b>exp</b>", small_bold), Paragraph("<b>What to try</b>", small_bold), Paragraph("<b>Rationale</b>", small_bold)],
        [Paragraph("exp_026", small_style), Paragraph("num_leaves=31 (smaller, less overfit)", small_style),
         Paragraph("Current 63 may be overfitting on 32k rows", small_style)],
        [Paragraph("exp_027", small_style), Paragraph("num_leaves=127 (more expressive)", small_style),
         Paragraph("More leaves = finer splits on segment patterns", small_style)],
        [Paragraph("exp_028", small_style), Paragraph("min_child_samples=10 (allow smaller leaves)", small_style),
         Paragraph("Segments with few samples may be underfit", small_style)],
        [Paragraph("exp_029", small_style), Paragraph("colsample_bytree=0.8 + subsample=0.8", small_style),
         Paragraph("Regularization via feature/row subsampling", small_style)],
        [Paragraph("exp_030", small_style), Paragraph("Best winner from above, n_estimators=500", small_style),
         Paragraph("More trees if best HP shows further room", small_style)],
    ]
    w7t = Table(w7_rows, colWidths=[0.6*inch, 2.5*inch, 3.2*inch])
    w7t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY_BG]),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(w7t)
    story.append(SP(10))

    story.append(Paragraph("Week 8 — Final Presentation and Submission", subsection_style))
    story.append(Paragraph("No new experiments. All effort goes to producing the final deliverable.", body_style))

    w8_rows = [
        [Paragraph("<b>Task</b>", small_bold), Paragraph("<b>Output</b>", small_bold)],
        [Paragraph("Final results summary", small_style), Paragraph("Updated results.csv + F1 trajectory plot across all runs", small_style)],
        [Paragraph("Feature importance analysis", small_style), Paragraph("LightGBM feature importance bar chart — which features matter most", small_style)],
        [Paragraph("Error analysis", small_style), Paragraph("What types of congestion events the model still misses (time of day, segment type)", small_style)],
        [Paragraph("Presentation slides", small_style), Paragraph("Narrative: problem → ceiling discovery → fix → model comparison → results", small_style)],
        [Paragraph("Week 8 deliverable PDF", small_style), Paragraph("All required sections including test set evaluation (first time test is used)", small_style)],
    ]
    w8t = Table(w8_rows, colWidths=[1.8*inch, 4.5*inch])
    w8t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY_BG]),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(w8t)
    story.append(SP(8))

    story.append(Paragraph("Locked Presentation Narrative", subsection_style))
    narrative = [
        ("1", "The prediction task and why it is hard", "30-min-ahead congestion, class imbalance 3.5:1, global threshold issue"),
        ("2", "The AutoResearch loop", "What it does, how it keeps experiments honest, why frozen run.py matters"),
        ("3", "The ceiling discovery", "Global threshold mislabels 16% of rows — no model can fix wrong labels"),
        ("4", "The fix", "Segment-relative features: what they are, how computed, why no leakage"),
        ("5", "Model comparison", "RF vs HGB vs XGB vs LightGBM — why LightGBM wins"),
        ("6", "Final result", "F1=0.678, +21% over baseline, +63% recall — with test set number"),
        ("7", "Remaining limitation", "Label definition still global — what a future project would change"),
    ]
    nar_rows = [[Paragraph(f"<b>{n[0]}</b>", small_bold), Paragraph(f"<b>{n[1]}</b><br/>{n[2]}", small_style)] for n in narrative]
    nart = Table(nar_rows, colWidths=[0.25*inch, 6.05*inch])
    nart.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, GREY_BG]),
        ("BOX", (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID", (0,0), (-1,-1), 0.3, BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (0,0), (0,-1), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(nart)

    doc.build(story)
    print(f"PDF written to {OUTPUT}")

if __name__ == "__main__":
    build()
