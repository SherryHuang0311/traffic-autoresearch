"""
Generate week7_deliverables.pdf — a 4-page research paper + reflection memo.
Run: python3 deliverables/week7/make_pdf.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "deliverables/week7/week7_deliverables.pdf"

DARK   = colors.HexColor("#111111")
BLUE   = colors.HexColor("#1a3a5c")
MBLUE  = colors.HexColor("#2e6da4")
LGREY  = colors.HexColor("#f5f5f5")
BORDER = colors.HexColor("#cccccc")
GREEN  = colors.HexColor("#1d7a3a")
LGREEN = colors.HexColor("#d4edda")
LRED   = colors.HexColor("#f8d7da")
LBLUE  = colors.HexColor("#d0e4f5")

styles = getSampleStyleSheet()
body = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=13,
    fontName="Times-Roman", spaceAfter=5, alignment=TA_JUSTIFY)
body_bold = ParagraphStyle("BB", parent=styles["Normal"], fontSize=10, leading=13,
    fontName="Times-Bold", spaceAfter=5)
h1 = ParagraphStyle("H1", parent=styles["Normal"], fontSize=12, leading=15,
    fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4, textColor=BLUE)
h2 = ParagraphStyle("H2", parent=styles["Normal"], fontSize=10.5, leading=13,
    fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3, textColor=BLUE)
para_head = ParagraphStyle("PH", parent=styles["Normal"], fontSize=10, leading=13,
    fontName="Times-Bold", spaceAfter=2)
title_s = ParagraphStyle("TT", parent=styles["Normal"], fontSize=15, leading=19,
    fontName="Helvetica-Bold", textColor=BLUE, alignment=TA_CENTER, spaceAfter=4)
author_s = ParagraphStyle("AU", parent=styles["Normal"], fontSize=10, leading=13,
    fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2)
abstract_s = ParagraphStyle("AB", parent=styles["Normal"], fontSize=9.5, leading=13,
    fontName="Times-Roman", alignment=TA_JUSTIFY, leftIndent=30, rightIndent=30, spaceAfter=6)
caption_s = ParagraphStyle("CAP", parent=styles["Normal"], fontSize=9, leading=11,
    fontName="Times-Roman", textColor=colors.HexColor("#444444"), spaceAfter=4,
    alignment=TA_CENTER)
small = ParagraphStyle("SM", parent=styles["Normal"], fontSize=8.5, leading=11,
    fontName="Helvetica", textColor=DARK, spaceAfter=2)
small_bold = ParagraphStyle("SMB", parent=styles["Normal"], fontSize=8.5, leading=11,
    fontName="Helvetica-Bold", textColor=DARK, spaceAfter=2)
memo_h = ParagraphStyle("MH", parent=styles["Normal"], fontSize=11, leading=14,
    fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4, textColor=BLUE)

def HR(): return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=4, spaceBefore=2)
def SP(n=6): return Spacer(1, n)

def tab(rows, widths, header_bg=BLUE):
    t = Table(rows, colWidths=widths)
    cmds = [
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LGREY]),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ]
    t.setStyle(TableStyle(cmds))
    return t

def build():
    doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
        leftMargin=1.0*inch, rightMargin=1.0*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch)
    s = []

    # ── PAPER TITLE ─────────────────────────────────────────────────────────
    s.append(HRFlowable(width="100%", thickness=3, color=BLUE, spaceAfter=6))
    s.append(Paragraph(
        "AutoResearch for Traffic Congestion Prediction:<br/>"
        "Segment-Relative Features Break a Labeling-Driven F1 Ceiling",
        title_s))
    s.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=4))
    s.append(Paragraph("Sherry Huang &nbsp;|&nbsp; STAT 390 Capstone &nbsp;|&nbsp; "
        "University of Illinois Urbana-Champaign &nbsp;|&nbsp; xiyuh5@illinois.edu", author_s))
    s.append(Paragraph("Code: github.com/SherryHuang0311/traffic-autoresearch", author_s))
    s.append(SP(6))

    s.append(Paragraph("<b>Abstract.</b> We present an AutoResearch loop applied to "
        "30-minute-ahead congestion prediction on Chicago road segments "
        "(786K observations, Sept. 2023). Over 30 experiments we identify and partially "
        "fix a structural labeling problem: the global congestion threshold (~23 mph) "
        "mislabels ~16% of training rows because different road segments have very different "
        "normal speed ranges. Adding four segment-relative features breaks an F1 ceiling "
        "at 0.658 that no amount of hyperparameter tuning could cross. LightGBM with "
        "full feature subsampling achieves the final validation F1 = <b>0.6806</b> "
        "(+21.6% over logistic regression baseline), stable across random seeds "
        "(mean 0.674, std 0.005). A one-time evaluation on the held-out test set yields "
        "<b>F1 = 0.6183</b> (+10.5% above the baseline), with the val→test gap "
        "explained by a distribution shift in positive rate (29.7% → 24.4%).", abstract_s))
    s.append(HR())

    # ── SECTION 1 ────────────────────────────────────────────────────────────
    s.append(Paragraph("1  Introduction", h1))
    s.append(Paragraph(
        "Traffic congestion prediction from speed history alone — without real-time feeds or "
        "external map data — is a constrained but practically relevant problem. This project "
        "formalizes it as an AutoResearch contract: predict whether a given Chicago road segment "
        "will be congested exactly 30 minutes from now, evaluated by validation binary F1 on a "
        "fixed time-based split. The AutoResearch loop modifies only <b>src/model.py</b> "
        "(model and features) while <b>src/run.py</b> (data split, label definition, metric) "
        "is frozen, guaranteeing all 30 experiments are evaluated identically.", body))

    # ── SECTION 2 ────────────────────────────────────────────────────────────
    s.append(Paragraph("2  Data, Metric, and Baseline", h1))
    s.append(Paragraph("<b>Dataset.</b> Chicago Traffic Tracker (City of Chicago Open Data, "
        "Sept. 2023). 786,420 observations. Each row is one speed reading for one road segment "
        "at a 10-minute timestamp. <b>SEGMENT_ID</b> identifies individual road segments.", body))
    s.append(Paragraph("<b>Label.</b> A segment is congested if speed &lt; 30th percentile "
        "of all training speeds (~23 mph). Class ratio 3.49:1 (77.7% non-congested). "
        "<b>Target:</b> congestion status 3 steps (30 min) ahead via shift(-3) per segment.", body))
    s.append(Paragraph("<b>Split.</b> Deterministic, time-based: train 70%, validation 15%, "
        "test 15% (locked — never accessed during development).", body))
    s.append(Paragraph("<b>Features (17 total).</b> SPEED, lags 1–6, rolling_mean_3, "
        "rolling_std_3, speed_diff, HOUR, DAY_OF_WEEK, MONTH, and four segment-relative "
        "features (segment_mean_speed, segment_std_speed, speed_vs_seg_mean, speed_zscore) "
        "computed from training data only.", body))
    s.append(Paragraph("<b>Baseline.</b> Logistic regression, lags 1–3, time features: "
        "F1 = 0.5598, P = 0.755, R = 0.445 (exp_001).", body))

    # ── SECTION 3 ────────────────────────────────────────────────────────────
    s.append(Paragraph("3  AutoResearch Loop Design", h1))
    s.append(Paragraph(
        "Each iteration: (1) read model.py; (2) propose one change; (3) edit model.py; "
        "(4) run python src/run.py \"description\"; (5) if F1 improves → commit; "
        "else → git checkout src/model.py and log status=discard. Results append to "
        "experiments/results.csv. The loop is greedy (single-step revert, no lookahead). "
        "The frozen evaluator prevents metric gaming — changing the split or threshold "
        "after seeing a good result is structurally impossible.", body))

    # ── SECTION 4 ────────────────────────────────────────────────────────────
    s.append(Paragraph("4  Experiments and Results", h1))
    s.append(Paragraph("4.1  F1 Progression by Phase", h2))

    phase_rows = [
        [Paragraph("<b>Phase</b>", small_bold), Paragraph("<b>Key change</b>", small_bold),
         Paragraph("<b>Best F1</b>", small_bold), Paragraph("<b>Gain</b>", small_bold)],
        [Paragraph("Baseline (exp_001)", small), Paragraph("Logistic regression", small),
         Paragraph("0.5598", small), Paragraph("—", small)],
        [Paragraph("Model selection (exp_002–007)", small), Paragraph("RF + class balance + lags 1–6", small),
         Paragraph("0.6566", small), Paragraph("+0.097", small)],
        [Paragraph("HP grid (exp_008–013)", small), Paragraph("Depth, trees, undersampling ratio", small),
         Paragraph("0.6585", small), Paragraph("+0.002", small)],
        [Paragraph("Threshold sweep (exp_014–018)", small), Paragraph("Decision threshold 0.35–0.50", small),
         Paragraph("0.6585", small), Paragraph("+0.000", small)],
        [Paragraph("Segment features (exp_019–021)", small), Paragraph("speed_zscore, speed_vs_seg_mean, seg. mean/std", small),
         Paragraph("0.6727", small), Paragraph("+0.014", small)],
        [Paragraph("Model comparison (exp_022–025)", small), Paragraph("LightGBM vs RF vs XGBoost vs HGB", small),
         Paragraph("0.6780", small), Paragraph("+0.005", small)],
        [Paragraph("<b>LightGBM tuning (exp_026–030)</b>", small_bold), Paragraph("<b>num_leaves, subsampling, n_estimators</b>", small_bold),
         Paragraph("<b>0.6806</b>", small_bold), Paragraph("<b>+0.003</b>", small_bold)],
    ]
    pt = tab(phase_rows, [1.5*inch, 2.5*inch, 0.75*inch, 0.65*inch])
    # Highlight last row
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LGREY]),
        ("BACKGROUND", (0,7), (-1,7), LGREEN),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]))
    s.append(pt)
    s.append(Paragraph("Table 1: F1 progression across 30 experiments (0 crashes). Each phase "
        "builds on the previous best configuration.", caption_s))
    s.append(SP(4))

    s.append(Paragraph("4.2  The Ceiling Discovery", h2))
    s.append(Paragraph(
        "Experiments 3–13 confirmed a hard ceiling at F1 ≈ 0.658. Varying n_estimators "
        "(50–200), max_depth (4–12), and undersampling ratio (1:1–2:1) moved F1 by under "
        "0.003 total. The root cause: the global 23 mph threshold mislabels ~16.3% of "
        "training rows. A highway segment averaging 45 mph is labeled 'congested' at 22 mph "
        "even if that is its normal rush-hour pace. The fix — adding segment-relative features "
        "computed from training data only via groupby('SEGMENT_ID')['SPEED'].agg(...) — "
        "gives the model road-specific context. F1 jumped 0.014 in one step.", body))

    s.append(Paragraph("4.3  Model Comparison and Final Tuning", h2))
    model_rows = [
        [Paragraph("<b>Model (exp)</b>", small_bold), Paragraph("<b>F1</b>", small_bold),
         Paragraph("<b>Precision</b>", small_bold), Paragraph("<b>Recall</b>", small_bold)],
        [Paragraph("Random Forest (exp_020)", small), Paragraph("0.6727", small),
         Paragraph("0.644", small), Paragraph("0.704", small)],
        [Paragraph("HistGradientBoosting (exp_022)", small), Paragraph("0.6553", small),
         Paragraph("0.540", small), Paragraph("0.834", small)],
        [Paragraph("XGBoost (exp_023)", small), Paragraph("0.6712", small),
         Paragraph("0.625", small), Paragraph("0.724", small)],
        [Paragraph("LightGBM 17 features (exp_025)", small), Paragraph("0.6780", small),
         Paragraph("0.638", small), Paragraph("0.724", small)],
        [Paragraph("<b>LightGBM tuned (exp_030, final)</b>", small_bold),
         Paragraph("<b>0.6806</b>", small_bold), Paragraph("<b>0.635</b>", small_bold),
         Paragraph("<b>0.733</b>", small_bold)],
    ]
    mt = tab(model_rows, [2.5*inch, 0.75*inch, 0.9*inch, 0.85*inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LGREY]),
        ("BACKGROUND", (0,2), (-1,2), LRED),
        ("BACKGROUND", (0,5), (-1,5), LGREEN),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]))
    s.append(mt)
    s.append(Paragraph("Table 2: Model comparison (identical setup: 17 features, 2:1 undersample, "
        "threshold=0.40). Red = discard. Green = final best.", caption_s))
    s.append(Paragraph(
        "LightGBM consistently edged Random Forest. Reducing num_leaves (63→31), "
        "setting min_child_samples=10, and adding 20% feature and row subsampling "
        "(colsample_bytree=0.8, subsample=0.8) gave +0.003 F1. "
        "Increasing n_estimators to 500 gave a further marginal gain.", body))

    s.append(Paragraph("4.4  Stability", h2))
    stab_rows = [
        [Paragraph("<b>Seed</b>", small_bold), Paragraph("<b>F1</b>", small_bold),
         Paragraph("<b>Precision</b>", small_bold), Paragraph("<b>Recall</b>", small_bold)],
        [Paragraph("42", small), Paragraph("0.6709", small), Paragraph("0.5953", small), Paragraph("0.7685", small)],
        [Paragraph("0", small),  Paragraph("0.6684", small), Paragraph("0.5998", small), Paragraph("0.7547", small)],
        [Paragraph("123", small),Paragraph("0.6735", small), Paragraph("0.5998", small), Paragraph("0.7678", small)],
        [Paragraph("7", small),  Paragraph("0.6821", small), Paragraph("0.6357", small), Paragraph("0.7358", small)],
        [Paragraph("99", small), Paragraph("0.6740", small), Paragraph("0.6212", small), Paragraph("0.7366", small)],
        [Paragraph("<b>Mean ± Std</b>", small_bold),
         Paragraph("<b>0.674 ± 0.005</b>", small_bold),
         Paragraph("0.610 ± 0.016", small_bold),
         Paragraph("0.753 ± 0.014", small_bold)],
    ]
    st = tab(stab_rows, [1.0*inch, 0.9*inch, 1.0*inch, 0.9*inch])
    s.append(st)
    s.append(Paragraph("Table 3: Stability across 5 random seeds. Run.py result (0.6806, seed=42) "
        "is within 1.5 std of the mean.", caption_s))

    # ── SECTION 5 ────────────────────────────────────────────────────────────
    s.append(Paragraph("5  What Worked and What Did Not", h1))
    s.append(Paragraph("<b>Worked.</b> The single most impactful step was adding segment-relative "
        "features (+0.014 F1). LightGBM marginally outperformed Random Forest (+0.003). "
        "Undersampling 2:1 achieved better F1 than class_weight='balanced' while maintaining "
        "higher precision. The frozen evaluator made all 30 comparisons valid.", body))
    s.append(Paragraph("<b>Did not work.</b> HistGradientBoosting underperformed in both attempts. "
        "Threshold tuning below 0.40 reliably raised recall but dropped F1. "
        "Six hyperparameter experiments on RF moved F1 by under 0.003 total.", body))
    s.append(Paragraph("<b>Fundamental limit.</b> The congestion label is still globally defined. "
        "Fixing this requires changing the frozen run.py — the residual label noise is "
        "irreducible within the current evaluation framework.", body))

    # ── SECTION 6 ────────────────────────────────────────────────────────────
    s.append(Paragraph("6  Conclusion", h1))
    compare_rows = [
        [Paragraph("<b>Split</b>", small_bold), Paragraph("<b>F1</b>", small_bold),
         Paragraph("<b>Precision</b>", small_bold), Paragraph("<b>Recall</b>", small_bold)],
        [Paragraph("Baseline validation (exp_001, LR)", small), Paragraph("0.5598", small),
         Paragraph("0.7549", small), Paragraph("0.4448", small)],
        [Paragraph("Validation (exp_030, LightGBM)", small_bold),
         Paragraph("<b>0.6806</b>", small_bold), Paragraph("<b>0.6352</b>", small_bold),
         Paragraph("<b>0.7329</b>", small_bold)],
        [Paragraph("<b>Test (final, one-time)</b>", small_bold),
         Paragraph("<b>0.6183</b>", small_bold), Paragraph("<b>0.5639</b>", small_bold),
         Paragraph("<b>0.6843</b>", small_bold)],
        [Paragraph("Val → Test gap", small), Paragraph("−0.0623", small),
         Paragraph("−0.0713", small), Paragraph("−0.0486", small)],
    ]
    ct = tab(compare_rows, [2.2*inch, 1.0*inch, 1.0*inch, 1.0*inch])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,2), (-1,2), LGREEN),
        ("BACKGROUND", (0,3), (-1,3), LBLUE),
        ("ROWBACKGROUNDS", (0,1), (0,1), [colors.white]),
        ("GRID", (0,0), (-1,-1), 0.3, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]))
    s.append(ct)
    s.append(Paragraph(
        "Table 4: Final results. Green = best validation. Blue = test (one-time, never seen during development). "
        "Val→Test gap explained by distribution shift: test period positive rate 24.4% vs 29.7% in validation.", caption_s))
    s.append(Paragraph(
        "The clearest lesson: for structured tabular data, feature engineering that encodes "
        "domain knowledge (road-specific speed norms) is more impactful than model selection "
        "or hyperparameter search. The loop arrived at this insight through systematic "
        "elimination — every failed tuning experiment narrowed the search space. "
        "Test F1 = <b>0.6183</b> confirms generalization: +10.5% above the baseline, "
        "with no tuning performed after the test set was opened.",
        body))

    s.append(PageBreak())

    # ── REFLECTION MEMO ───────────────────────────────────────────────────────
    s.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=6))
    s.append(Paragraph("Reflection Memo — Agent Performance and Human-Agent Collaboration", title_s))
    s.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=6))
    s.append(SP(4))

    s.append(Paragraph("What the Agent Did Well", memo_h))
    s.append(Paragraph(
        "<b>Systematic elimination with zero bookkeeping errors.</b> Over 30 experiments "
        "the agent never conflated results, never ran a worse model forward, and never "
        "lost track of the current best. Every discard was caught and reverted correctly. "
        "This kind of precise, patient bookkeeping is where the agent genuinely outperformed "
        "informal human iteration.", body))
    s.append(Paragraph(
        "<b>Identifying the labeling problem.</b> The agent measured the 16.3% label "
        "disagreement between the global threshold and a segment-relative threshold, "
        "connected it to the ceiling at F1=0.658, and designed the fix — segment features "
        "from training data only with explicit leakage analysis. A human researcher might "
        "have spent weeks tuning before arriving at this diagnosis.", body))
    s.append(Paragraph(
        "<b>Execution speed.</b> 30 experiments with full git discipline ran in hours "
        "without fatigue or attention lapses.", body))

    s.append(Paragraph("What the Agent Did Poorly", memo_h))
    s.append(Paragraph(
        "<b>Greedy search with no lookahead.</b> The loop committed to Random Forest for "
        "20 experiments before trying LightGBM — which turned out to be marginally better. "
        "A simple early model tournament would have found LightGBM sooner.", body))
    s.append(Paragraph(
        "<b>No repeat-on-change logic.</b> HistGradientBoosting was dropped after exp_005 "
        "(without segment features) and not retried until exp_022 — a 17-experiment delay. "
        "The loop had no mechanism to flag experiments for retry under changed conditions.", body))
    s.append(Paragraph(
        "<b>Single-seed evaluation.</b> Every experiment was a single run (seed=42). "
        "Keep/discard decisions were made on single-point estimates, risking lucky-run "
        "false positives. Multi-seed evaluation should have been standard.", body))

    s.append(Paragraph("What Required Human Judgment", memo_h))
    s.append(Paragraph(
        "<b>Ceiling diagnosis.</b> Measuring the 16.3% label disagreement was automatic; "
        "understanding that this was the <i>fundamental limiting factor</i> — that no model "
        "could overcome systematically wrong labels — required framing of the learning problem.", body))
    s.append(Paragraph(
        "<b>Scope lock.</b> Deciding to stop exploring new model families and commit to "
        "LightGBM tuning was a judgment about diminishing returns that the agent cannot make. "
        "The agent measures what improved; humans judge when further improvement is not worth the cost.", body))
    s.append(Paragraph(
        "<b>Presentation narrative.</b> Which experiments to highlight — not just 'highest F1' "
        "but 'most interesting finding' — is a human judgment about what constitutes "
        "a contribution worth communicating.", body))

    s.append(Paragraph("How Would You Redesign the Loop", memo_h))
    redesign = [
        "<b>Early model tournament:</b> in the first 10 experiments, try every major model family on the baseline feature set. Lock the winner, then tune.",
        "<b>Repeat-on-change rule:</b> when a structural change is made (new features), automatically re-run previously discarded configurations under the new conditions.",
        "<b>Multi-seed evaluation:</b> run each experiment with 3 seeds, report the median. Costs 3× compute but eliminates lucky-run false positives.",
        "<b>Automatic ceiling diagnosis:</b> after every 5 non-improving experiments, compute label agreement between the current strategy and alternatives and surface it as a diagnostic.",
    ]
    for r in redesign:
        s.append(Paragraph(f"&#8226; &nbsp; {r}", body))

    s.append(Paragraph("Final Reflection: Research With AI Agents", memo_h))
    s.append(Paragraph(
        "The agent is excellent at <i>execution within a well-defined protocol</i> and poor "
        "at <i>knowing when to change the protocol</i>. The most valuable human contribution "
        "was not running experiments — the agent did that reliably. It was asking the right "
        "diagnostic question: <i>why is the ceiling there?</i>", body))
    s.append(Paragraph(
        "Agent-assisted research shifts the bottleneck from execution to diagnosis. The "
        "agent can run 100 experiments in the time it would take a human to run 5 — but "
        "if those 100 experiments are all asking the wrong question, the agent produces "
        "100 wrong answers very efficiently. The human's job becomes: read results fast, "
        "spot the pattern the agent is missing, and redirect.", body))
    s.append(Paragraph(
        "In this project that happened once — Week 5, when the ceiling pattern was identified "
        "and the labeling problem diagnosed. That one redirection was worth more than all "
        "20 hyperparameter-tuning experiments combined.", body))

    doc.build(s)
    print(f"PDF written to {OUTPUT}")

if __name__ == "__main__":
    build()
