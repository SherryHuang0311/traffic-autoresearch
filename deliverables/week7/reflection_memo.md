# Reflection Memo — AutoResearch Loop
**STAT 390 Capstone | Sherry Huang | Week 7**

---

## What Did the Agent Do Well?

**1. Systematic elimination with zero bookkeeping errors.**
Over 30 experiments the agent never conflated results, never ran a worse model
forward, and never lost track of what the current best was. Every discard was
caught and reverted correctly. The results.csv is a faithful record — no
post-hoc edits. This kind of precise, patient bookkeeping is where the agent
genuinely outperformed what a human researcher does informally.

**2. Identifying the labeling problem.**
The agent measured the disagreement between the global threshold and a hypothetical
segment-relative threshold (16.3% of rows labeled differently), connected this to
the observed ceiling at F1=0.658, and designed the right fix — segment features
computed from training data only, with explicit leakage analysis. A human researcher
might have spent weeks tuning models before arriving at this diagnosis.

**3. Execution speed.**
30 experiments with full git discipline (commit, revert, log) ran in what would
have taken a human days of careful manual iteration. The agent ran them in hours
without fatigue or attention lapses.

**4. Model comparison discipline.**
When switching model families in Week 6, the agent held all other variables
constant (same features, same undersampling, same threshold) so the comparison
was clean. This is basic experimental hygiene that is easy to violate when
iterating quickly under pressure.

---

## What Did the Agent Do Poorly?

**1. Greedy search with no lookahead.**
The loop keeps the first improvement and builds on it. This means it can get
stuck on a local optimum. For example, the agent committed to Random Forest
for 20 experiments before trying LightGBM — which turned out to be marginally
better. A smarter search strategy (even a simple grid on model type early on)
would have found LightGBM sooner.

**2. Logging mistakes when discarding.**
On two occasions, running `python src/run.py ... --discard` appended a *new*
duplicate row instead of updating the existing one, because the logging logic
always appends. The agent had to manually edit results.csv to clean these up.
This is a design flaw in the loop that required human intervention to fix.

**3. No uncertainty quantification.**
Every experiment was a single run (random_state=42). The agent never ran the
same configuration twice with different seeds until the final stability check
at the very end. The "keep" decisions were all made on single-point estimates,
which could have led to keeping a lucky run and discarding an unlucky one.

**4. The HGB retry was too late.**
HistGradientBoosting was first tried in exp_005 (Week 3) *before* segment
features existed, failed, and was dropped. When segment features were added in
Week 5, HGB should have been retried immediately. Instead it was retried in
Week 6 (exp_022), a 17-experiment delay. The agent lacked any mechanism to
flag "this experiment should be repeated under changed conditions."

---

## What Required Human Judgment?

**1. Recognizing the ceiling as structural, not statistical.**
The agent could measure the 16.3% label disagreement, but the insight that
this was the *fundamental limiting factor* — that no model or tuning strategy
could overcome systematically wrong labels — required understanding of the
learning problem, not just measurement. This was a human-framed diagnosis.

**2. Deciding to freeze the evaluator.**
The decision to keep run.py frozen (and therefore keep the global label definition)
was a research design choice that protected experiment comparability. The agent
operated within this constraint but did not set it. Without this constraint, the
agent might have tried to change the evaluation to get better numbers — invalidating
all prior comparisons.

**3. Scope lock in Week 6.**
The decision to stop exploring new model families and commit to LightGBM
hyperparameter tuning was a judgment call about diminishing returns. The agent
can measure what improved, but it cannot judge when further improvement is
unlikely to be worth the experimental cost relative to other uses of time.

**4. Presentation narrative.**
Deciding which experiments to highlight in the final story — not just "what got
the highest F1" but "what was the most interesting finding" (the labeling problem)
— is a human judgment about what constitutes a contribution worth communicating.

---

## How Would You Redesign the Loop?

**1. Early model tournament.**
In the first 10 experiments, try every major model family (LR, RF, HGB, XGB,
LightGBM) on the baseline feature set. Lock the winning family, then tune it.
This avoids committing to RF for 20 experiments before discovering LightGBM.

**2. Repeat-on-change rule.**
Any time a structural change is made (new features, new model family), automatically
re-run all previously discarded configurations under the new conditions. This would
have caught the HGB improvement 15 experiments sooner.

**3. Multi-seed evaluation.**
Run each experiment with 3 seeds and report the median F1. Keep if median improves.
This costs 3x compute but eliminates lucky-run false positives. At 3-4 seconds per
run, this is still entirely tractable.

**4. Principled threshold selection.**
Instead of manually trying thresholds, compute the optimal threshold on the
validation set directly from the predicted probability distribution. This takes
one extra line of code and eliminates an entire experimental axis.

**5. Automatic ceiling diagnosis.**
After every 5 experiments with no improvement, compute the agreement between the
current labeling strategy and alternative strategies (segment-relative, time-of-day
relative, etc.) and surface this as a diagnostic. The loop would have identified
the ceiling issue at Week 3 instead of Week 5.

---

## Final Question: What Did You Learn About Doing Research With AI Agents?

The agent is excellent at *execution within a well-defined protocol* and poor
at *knowing when to change the protocol*. The most valuable human contribution
was not running the experiments — the agent did that reliably. It was asking
the right diagnostic question: *why is the ceiling there?* 

Agent-assisted research shifts the bottleneck from execution to diagnosis. The
agent can run 100 experiments in the time it would take a human to run 5. But
if those 100 experiments are all asking the wrong question, the agent just
produces 100 wrong answers very efficiently. The human's job becomes: read the
results fast, spot the pattern the agent is missing, and redirect.

In this project that happened once — Week 5, when the ceiling pattern was
identified and the labeling problem diagnosed. That one redirection was worth
more than all 20 hyperparameter-tuning experiments combined.
