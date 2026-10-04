# FR-8 Spending Spikes — Feature Design

Oct 4, 2026 · @Sidd · Status: **Accepted with conditions** (owner, Oct 4, 2026, on #58; see [Decisions](#decisions-and-open-questions)) · **Milestones 1–4 (serving) and 5 built; test scoring and promotion held by the owner** (#59–#63) · Branch: `docs/fr-8-design`

## Summary

This feature flags months in which a user's spending in one category runs well above their normal level. Each flag says how big the deviation is and which charges are behind it, and is measured against FR-2's planted spikes.

- **Requirement:** FR-8 (P0): *"Flag spending spikes: monthly spend in a category significantly above the user's normal level, with the size of the deviation and the transactions driving it."* Weekly spikes are out of v1 (FR-2, option F). PRD success metric: precision ≥ 0.70 per monthly period, recall above a simple rule-based alternative. NFR-7 requires a reason on every flag.
- **Starting point:** FR-1 plants spikes. FR-2 scores them (`Truth.score_periods`, `score_drivers`, the top-5-by-amount driver baseline), and its count oracle reaches precision 1.00 at recall 0.5, so the target is reachable. Nothing detects spikes yet: `detect_anomalies` returns `spending_spikes: not_available`, and "Worth a look" (1f) has a placeholder card.
- **Feasibility, measured on train users only** ([evidence](#feasibility)):
  - **The Technical Design's baseline fails.** Per-user mean ± k·std on monthly spend reaches precision 0.065 and recall 0.044 at the flag volume below. Spend is dominated by lumpy one-offs: its top scores are fees, subscriptions and health bills, and the top 240 of the other spend-based detectors are almost all Travel.
  - **Purchase counts carry the signal.** A spike multiplies how often the user buys, not what they pay. A Poisson tail on the month's purchase count, against the user's trailing 12-month average, reaches **recall 0.47 at precision 0.80**.
  - **Seasonality, income and a minimum volume bring it close to the ceiling.** The model adds a seasonal index (the user's own previous year, shrunk toward a cross-user profile for the category), a pooled income elasticity, and a rule that a usual month needs at least 2 purchases. It reaches **precision 0.81 at recall 0.55** at 0.035 flags per user-month (about one flag per user every 2.4 years). The oracle reaches 0.85 and 0.57 at the same rate. A rule that spend must be at least 1.3× usual stays as a product rule; it costs half a point of recall rather than earning any.
  - **The main risk: one mechanism.** The generator plants count spikes, the labels mark count spikes, and the detector detects count spikes. Months that are up because of bigger baskets or higher prices are missed by construction. On simulated ones, recall is 0.3% ([Main risk](#main-risk-one-mechanism)).
- **Approach:**
  1. **Score each (user, category, month) only from the user's earlier months** (point in time), once the month is complete.
  2. **Score the purchase count, and require the spend to be up too.** A month is a spike only if its count is improbably high for the user and its spend is at least 1.3× their usual level. The usual month must also have at least 2 purchases. Both are fixed product rules, set before the round and never tuned. A "spike" whose spend isn't up would contradict its own reason, and a jump from 1 to 4 purchases isn't a meaningful spike.
  3. **Seasonality from the user's own history, shrunk toward category season profiles built across users.** Profiles are a nightly feature table with the scored user left out, as FR-7's merchant profiles are. No persona label is used (FR-11, decision 11).
  4. **A round through the FR-3 framework:** the baseline, a report-only simple count rule, and two count models (Poisson and negative binomial). They're ranked on out-of-fold recall at a common flag rate, as FR-7's round was.
  5. **Tune to precision 0.80 and gate at 0.70,** as FR-7 does.
  6. **Reasons are templates that quote only the user's own numbers:** this month's spend and purchases against their average over the previous 12 months. The drivers are the 5 largest charges, which is optimal under FR-2's driver metric.
- **Owner decisions (Oct 4, 2026, on #58):** all eleven decisions were accepted. Decision 1 comes with three conditions: the spend floor is a product rule only, the minimum volume is fixed rather than tuned, and the basket-size gap is stated as the main risk and measured in the round. FR-8 aims for the Oct 6 demo on FR-11's terms. See [Decisions and open questions](#decisions-and-open-questions).
- **Principle (carried from FR-2 and FR-7):** the modeled behavior is not tuned to make the target pass, and test users are scored once, for finalists.

## Context

### What is planted

FR-1 plants about one spending spike per user per year after a 3-month warm-up, half monthly and half weekly. On the default dataset (main at `acfa1e9`, content hash `b4d43bf4`, the same data FR-7 used) that's 601 monthly spikes: 411 for train users and 190 for test users. 52 (8.7%; the generator's report, `spikes_month_weak`) are `weak`, meaning the realized spend stayed below 1.3× the expected spend. FR-2's implementation note says 49, which was counted on an earlier dataset.

| Property | Value | Consequence for a detector |
| --- | --- | --- |
| Effect | The category's purchase **rate** × 1.8–3.0 for the whole month; extra purchases come from the same merchants at the same prices (FR-2 §2) | Counts move cleanly; prices add heavy-tailed noise on top |
| Categories | Only where the user buys at least once a week: Dining (170), Shopping (153), Transportation (161), Groceries (117) | A flag in any other category is a false positive |
| Timing | Never in a month where that category's seasonality exceeds 1.1 (`peak_threshold`) | Seasonal peaks (a family's August and December shopping) are hard negatives |
| Freelancers | Spending follows trailing two-month income (elasticity 0.5) | A month after a big payment is busy without being a spike |

### What the label contract already decides (FR-2)

- **A period** is (user, category, month) on **true** categories, and its spend counts every transaction in the category.
- **Ignored:** months with a planted weekly spike in the category, months with a planted unusual charge in the category (FR-7 scores that charge), and the first 3 months.
- **False positive:** any other flagged period that isn't a label. **Recall** is reported on all labels and on `clear` labels.
- **Driving transactions:** at most 5, scored by **excess coverage** (their spend over the period's excess above expected spend, capped at 1), next to a top-5-by-amount baseline.
- **Thresholds** are tuned on train users and reported on test users.

### What the product asks for

- **The PRD's key scenario 3:** "Why was August so high?" → the coach points to the spike and the charges behind it.
- **Screen 1f:** a "Spending spikes" card that says "A month where a category runs well above your normal level, with the charges behind it", today a placeholder.
- **Web App UI, open question:** what "usual" means ("$150 more than your usual month"). It says that may come from FR-8's expected spend.

## Scope

| | FR-7 | FR-8 (this design) | FR-9 (P1) |
| --- | --- | --- | --- |
| Unit | One charge | One (user, category, complete month) | Either |
| Gates on | Precision ≥ 0.70 and recall above its baseline | The same, per monthly period | — |
| Delivers | `score_transactions`, unusual charges in `detect_anomalies` | `score_periods`, the `spending_spikes` half of `detect_anomalies`, the spikes card on 1f | A sensitivity control and flag actions |

A month that is high because of **one** large charge isn't a spike under this design: the count doesn't move, and FR-7 judges the charge. The contract agrees and ignores months that contain a planted unusual charge (decision 3). Months that are up from several larger charges, rather than more of them, aren't caught by either feature. That's the design's main risk ([below](#main-risk-one-mechanism)).

## Feasibility

**Evidence:** POC branch `poc/fr-8-spending-spikes`, commit `bd5ae63` (first version `20806f9`; the revisions for review on #58 add the ablations, the look-alikes, the basket-size simulation and the user-bootstrap interval at 0.035, and change no earlier number): `experiments/fr8_spikes/feasibility.py` and its output, `results/feasibility.md`. One command reproduces every number below.

Setup: the default dataset as above (`validate` passes). **Train users only:** 240 users, 7,920 post-warm-up user-months, 89,760 scored (user, category, month) periods (1,426 ignored by the contract), 411 labels (373 `clear`), so 0.052 labels per user-month. The file is loaded whole, and every truth table is cut to train users before anything is computed, so **no test user's rows or labels reach any number**. The first version scored all labels and filtered to train users afterwards; the numbers didn't change.

Every detector is point in time: month *m* is scored from months before *m* only (a 12-month trailing window, at least 3 months). Spend and counts use true categories. "Count" is the number of outflows in the category, because a model can't see the generator's processes.

What the POC simplifies, so its numbers are optimistic:

- **Thresholds were chosen and measured on the same users.** The round tunes on folds and measures out of fold (§6).
- **The population seasonal index pools all train users and all months,** the scored user and later months included. The round builds it as of each month, with the scored user left out (§2).
- **The income elasticity is fitted once on all train users.** It reads no labels, but the round fits it within folds.

### Detectors

At a fixed flag rate (flags per post-warm-up user-month), and at the best recall that holds a precision target:

| Detector | Recall at p ≥ 0.80 (flags) | Precision at 0.035 | Recall at 0.035 | Recall `clear` at 0.03 |
| --- | --- | --- | --- | --- |
| Baseline: mean ± k·std of monthly spend (Technical Design) | 0 (0) | 0.065 | 0.044 | 0.035 |
| Robust: log spend against median and MAD | 0 (0) | 0.000 | 0.000 | 0.000 |
| Spend ratio, seasonal | 0 (0) | 0.000 | 0.000 | 0.000 |
| Count, Poisson, trailing mean | 0.474 (240) | 0.726 | 0.489 | 0.517 |
| + seasonal index | 0.487 (250) | 0.755 | 0.509 | 0.520 |
| + income coupling | 0.518 (264) | 0.787 | 0.530 | 0.544 |
| **+ spend ≥ 1.3× usual and ≥ 2 purchases in a usual month (the leader)** | **0.545 (275)** | **0.809** | **0.545** | **0.547** |
| *Ablation:* + spend ≥ 1.3× usual only | 0.513 (262) | 0.783 | 0.528 | 0.542 |
| *Ablation:* + ≥ 2 purchases in a usual month only | 0.550 (277) | 0.816 | 0.550 | 0.552 |
| *Diagnostic:* the four planted categories only | 0.550 (276) | 0.816 | 0.550 | 0.552 |
| Count, negative binomial, seasonal and income | 0.431 (217) | 0.711 | 0.479 | 0.491 |
| *Oracle: Poisson tail on the generator's true expected count* | *0.623 (311)* | *0.845* | *0.569* | *0.590* |

- **Spend is the wrong signal.** The robust and ratio scores are topped by Travel (238 and 240 of their top 240): one flight in a category the user rarely uses is an enormous ratio. The baseline's top 240 are fees (89), subscriptions (30) and health bills (27), whose month-to-month spread is small, so one larger bill makes a big z. Restricted to categories with at least 4 purchases in a usual month, spend scores still reach only 65–103 true spikes in their top 240 (precision ≤ 0.43). This matches FR-2's oracle, where the spend ratio reached 0.29.
- **What each piece buys depends on the operating point** (change in recall, in points):

  | | p ≥ 0.70 | p ≥ 0.80 | 0.02 | 0.03 | 0.035 | 0.05 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Seasonal index | +4.4 | +1.2 | −1.0 | +0.2 | +1.9 | +1.7 |
  | Income coupling | +1.7 | +3.2 | +1.0 | +2.2 | +2.2 | +1.0 |
  | Spend floor only | −0.5 | −0.5 | 0.0 | −0.2 | −0.2 | −0.5 |
  | Minimum volume only | +1.9 | +3.2 | +0.5 | +0.7 | +1.9 | +1.0 |
  | Both rules | +1.5 | +2.7 | +0.5 | +0.2 | +1.5 | +1.0 |

  Income coupling helps everywhere. Seasonality helps at every rate but the strictest. The leader recovers 96% of the oracle's recall at 0.035.
- **The spend floor earns no recall.** On its own it costs up to half a point. It stays as a product rule (decision 1): a flagged month's spend is visibly up, so the reason never describes ordinary spending. It behaves the same whether "usual" is the seasonal expectation or the user's plain 12-month average (precision 0.801 against 0.809 at 0.035).
- **The minimum volume earns the gain, and it rediscovers where the generator plants.** "At least 2 purchases in a usual month" gives the same result as restricting to the four planted categories (0.550 with 277 flags against 0.550 with 276). What it removes is Poisson-tail false positives in low-volume categories. A month that goes from 1 purchase to 4 is improbable but not a meaningful spike, so the rule stands as a product rule fixed at 2 before the round (decision 1). It isn't tuned on labels: tuning it would only search for the generator's eligibility floor.
- **Negative binomial loses here,** probably for a structural reason: the generator's counts are Poisson by construction, so estimating over-dispersion from 12 months only adds noise. Real counts are over-dispersed, which is why it stays in the round (§3).
- **The pooled income elasticity is 0.25** (persona-free; the freelancers' true value is 0.5 and the others' is 0). A per-user elasticity is a later candidate.
- **Uncertainty (user bootstrap, 5–95%):** at 0.03, leader precision 0.81–0.91 and recall 0.47–0.53, and baseline precision 0.03–0.08. At the common rate of 0.035, leader precision 0.74–0.86 and recall 0.51–0.58. That interval is what milestone 1's check of the season-profile statistic is held to (§2).

### What the leader gets wrong (at 0.03 flags per user-month)

- **False positives (33 of 237):** spread over the months of the year (1–4 each), so seasonality isn't what's left. By category: Dining 10, Transportation 9, Shopping 7, Entertainment 3, Groceries 2, Health & Fitness 2. Five are in categories where nothing is ever planted: natural Poisson extremes, which the contract calls false positives.
- **Misses (207):** all 38 `weak` labels and 169 `clear` ones. Shopping 63, Groceries 57, Transportation 52, Dining 35. The oracle misses 176 of them too at this rate: low-volume categories (Groceries at about 1.2 purchases a week) and spikes at the low end of 1.8–3.0.

### Size of the deviation and drivers

- **Purchases:** a flagged spike month has a median of 48 purchases, against 19 in the user's average month.
- **"Usual" from the user's own numbers is as accurate as the model's.** Against the generator's true expected spend, the user's plain average over the previous 12 months is off by a median 9.0% (90th percentile 25.5%). The seasonal and income-adjusted expectation is off by 9.8% (27.2%). The median excess is $1,074 by the plain average and $1,087 in truth.
- **Drivers:** the 5 largest charges cover a mean 0.69 of a labeled spike's excess and all of it 31% of the time. By construction, no 5 charges in the period cover more, so this is the ceiling of FR-2's metric. FR-2 also rejected trying to pick out the "extra" purchases, which no detector can tell apart (FR-2 §1).

### Main risk: one mechanism

**The planted spikes, the labels and the detector all share one mechanism: more purchases.** The generator plants a spike by raising the purchase rate, the contract labels exactly those months, and the leader scores the count. So the feasibility numbers measure how well the detector finds the kind of spike the generator makes. They don't measure spikes made any other way. Real overspending also comes from bigger baskets, higher prices, holiday shopping and repairs paid in several charges, and those leave the count flat.

Measured on train users, in the four planted categories:

- **Look-alikes:** 3,685 non-label months have spend at least 1.5× the user's 12-month average, 9.0 for every label.
  - In 64.4% of them, the 3 largest charges cover the whole excess.
  - In 26.5% (28.2% against the seasonal expectation), the count rose less than 1.3×, so the rise is purely price or basket.
  - The leader flags 14 of them at its 0.035 cutoff.
- **Simulated basket-size spikes:** on 400 non-label months, spend is multiplied by U(1.8, 3.0) with counts unchanged. The leader's recall at its 0.035 cutoff is **0.3%**, against 54.5% on planted spikes.

What follows (decision 1):

- **Stated as a known limit,** here and in the round's report: v1 detects months with more purchasing than usual, not months with bigger purchases.
- **Measured in the round:** recall on simulated basket-size spikes is a reported diagnostic for every candidate (§6), so the gap has a number on validation and test.
- **Not patched with spend:** spend scores fail at precision ≤ 0.43 even on frequent categories (above). Adding them back would trade the precision gate for a gap the labels can't score.
- **Revisited in v2** with real data, where spikes of every kind occur and a spend-side signal can be judged against real outcomes.

The review on #58 measured the same quantities slightly differently: 3,684 look-alikes, the count up less than 1.3× in 22–27%, 37 of them flagged, and 0.2% recall on simulated spikes. The numbers above are the POC's output; the conclusion is the same.

## Goals and non-goals

**Goals**

1. A promoted spike scorer that passes precision ≥ 0.70 on test users and beats the baseline's recall at the same flag volume.
2. Every flag carries its size (actual, usual, excess, purchases), a plain-language reason and up to 5 driving charges, all checked at runtime (NFR-7).
3. Point-in-time scoring: a month is judged only on earlier months, and only once it's complete.
4. Spikes reach "Worth a look", the coach and outside assistants through `detect_anomalies`, scoped to the session's user, on the categories that user sees.
5. One command reproduces the round and its report (NFR-8).

**Non-goals**

- Weekly spikes (FR-2, option F) and month-to-date pacing ("you're on track to overspend Dining"): v1.1 or later.
- Gradual drifts, FR-2's open "silent spike": no label shape exists.
- Single large charges: FR-7.
- The sensitivity control and flag actions (FR-9, P1). v1 fixes the flag id they'll key on (§8).
- Labels for natural extremes the generator didn't plant (FR-2, option A).

## Design

### 1. Contract and point-in-time scoring

The contract is batch-first, like FR-3's and FR-7's, so one job can score every user:

```
score_periods(rows) → one row per (user_id, category, period_start) in `rows`:
  score, is_flagged, evidence, model_version
```

- **Input:** monthly aggregate rows from the feature pipeline (§2). Per (user, spending category, month) they hold `spend` (net outflow), `count` (outflows), the user's `income` that month, and the category season profile columns. Each user's rows run from their first month to the scoring month. A category comes from a column the caller fills: true categories in evaluation, the user's effective categories in serving (§8). Income is never a spending category.
- **Point in time:** a period's score uses only the same user's earlier months and season profiles as of the first of the month. Later months never change it.
- **Complete months only.** Only months that are complete at the scoring date are scored. v1 doesn't score a month in progress.
- **Minimum history:** a period is scored once the user has 3 earlier months. That's the same length as the label contract's warm-up, fixed, not tuned. Before that, the tool says the history is too short (§8).
- **Runtime checks** (FR-3's `Checked` wrapper):
  - one output row per input period, in order;
  - when `is_flagged`: `evidence.actual ≥ 1.3 × evidence.usual` (the spend floor of §3), `evidence.usual_count ≥ 2` (the minimum volume of §3), and `evidence` holds every field the reason template needs (§7);
  - no persona or truth column in the input.

  A flag without a reason or a size breaks the contract; it isn't a quality issue.
- **The Technical Design's signature** `detect_spikes(user_id, period, granularity)` becomes a thin per-user call over this one plus the drivers (§7). The Technical Design changes in milestone 5.

### 2. Features and category season profiles

All features come from model-visible columns. Nothing reads `truth_*`, and the existing isolation test covers new code under `intelligence/`.

| Feature | Source | Used for |
| --- | --- | --- |
| Monthly aggregates | `spend` and `count` per (user, category, month), zero-filled from the user's first purchase in the category | The score; the reason |
| Usual level | The mean `spend` and `count` over the previous 12 months (at least 3) | The Poisson rate; the reason's "usual"; the spend floor |
| The user's own season | That category's count in the same month a year earlier, over the user's trailing mean then | Seasonality, once a year of history exists |
| **Category season profile** | Per (category, month of year, as-of month): across other users, the average of each user's log ratio of count to their trailing mean, clipped to ±log 3, from months before the as-of month | Seasonality before the user has a year, and shrinkage after |
| Income ratio | The user's income over the previous two months, over their 12-month average (`Income` rows) | Freelancer-style coupling |

**Season profiles are a feature table, built like FR-7's merchant profiles:**

- built nightly by the feature pipeline from every user's model-visible rows, **as of the first of each month**, so a period never sees later months. The table is keyed by (category, month of year, as-of month);
- **the scored user's own months are left out** of the profile used to score them. Each cell stores the sum of users' clipped log ratios and the number of users, so leaving one user out is exact: subtract that user's own terms, which serving recomputes from their ledger. The POC used a median over all users, which can't be left out cheaply. Milestone 1 checks that the POC's result holds with the clipped mean before the round runs, against a pass condition fixed now (owner, Oct 4, 2026, on #58): at 0.035 flags per user-month, the leader with the clipped mean has recall within the POC's user-bootstrap interval, **0.506–0.578**, and precision within it, **0.737–0.860** (both printed by the script at `bd5ae63`). If either falls outside, the statistic goes back to the owner before the round; it isn't swapped silently;
- a cell needs at least **20 distinct other users**, otherwise the season is 1. It's an aggregate over many people's buying, never a single merchant or a price. FR-7's 3-user minimum is a privacy floor; 20 is about stability.
- **Which categories the pool uses:**
  - **in evaluation**, true categories, like every FR-8 metric. Validation profiles come from train users only, and test profiles from all users: FR-7's two-pool rule, so test users never shape a validation score;
  - **in serving**, the promoted categorizer's predictions for every user. They're shared and include nobody's corrections, so one user's corrections never reach another user's profile (user story 5). The served user's own terms, subtracted for the leave-out, are computed on the same predicted categories, while their own aggregates use their effective categories (§8).
- **The demo bundle** (decision 8): the build computes the table from the **full `--data` pool**, not just the 3 demo accounts, and ships it as a bundle file, as FR-7's `flag_dataset(pool=data)` does for merchant profiles. The demo accounts are all test users, so a table built from the bundle alone would have too few users in every cell and fall back to a season of 1. That needs predicted categories for every user, so the build either categorizes the pool with the promoted categorizer or reads a full predictions file it's given. Milestone 4 measures that build time.

**Shrinkage:** the season used is `w · own + (1 − w) · profile`, with `w = years / (years + κ)`. `years` counts the user's earlier same-month observations, and κ is tuned within folds (grid {0.5, 1, 2}, owner, Oct 4, 2026, on #58; the POC used κ = 1). Two noisy Augusts don't become the user's season. This is FR-11's idea, without the persona.

**No persona.** `users.persona` is a label real users don't have (FR-11, decision 11). The only cross-user input is the category profile, which mixes everyone. The contract check rejects a persona column. Folds are still stratified by persona, on the evaluation side only.

**Categories stay as given.** The scorer doesn't know or care whether categories are true, predicted or corrected. Evaluation fixes them to the truth so categorizer errors can't leak into FR-8's metrics (Technical Design controls). Serving uses what the user sees (§8).

### 3. Candidates

Each candidate outputs one score per period on a common scale and is a class plus a config in the FR-3 registry.

| Candidate | What it is | Why |
| --- | --- | --- |
| **Baseline:** mean ± k·std | z of the month's spend against the user's earlier months in that category (all of them) | Required: the Technical Design's "simple alternative". Gated |
| **Simple count rule** (reported only) | Poisson upper tail of the count against the user's trailing 12-month mean count. No season, income or floor | A stronger simple rule, reported beside the gated baseline, as FR-7 reported "baseline plus duplicates" |
| **Count, Poisson** | Poisson upper tail of the count against `usual count × season × income_ratio^β`; flagged only where spend ≥ 1.3× usual and the usual month has at least 2 purchases | The POC leader. One score is the tail probability, so the FR-9 sensitivity becomes one knob |
| **Count, negative binomial** | The same expectation, with over-dispersion from the user's trailing count variance shrunk toward a pooled value | Real counts are over-dispersed and synthetic ones aren't (§Feasibility), so this is the variant built for v2. The round measures what it costs here |

- **Fitted without labels:** β (pooled, by least squares of the log count ratio on the log income ratio), the pooled dispersion and the season profiles.
- **Tuned with labels, within folds (§5):** κ only, by recall at the common flag rate.
- **Fixed before the round, never tuned (decision 1):** both are product rules and apply to every count candidate.
  - **The spend floor of 1.3×** reuses FR-2's `weak_lift`, the line between a visible spike and noise. Below it, the reason would describe ordinary spending. It earns no recall (it costs up to half a point in the POC).
  - **The minimum volume:** the user's usual month has at least 2 purchases in the category. Fewer than that, and a "spike" is a handful of purchases (1 becoming 4), improbable but not meaningful. The value is the POC's, chosen before this review. It removes the same periods as restricting to the generator's planted categories (Feasibility), which is why it must not be tuned: on labels, tuning would only rediscover the generator's floor.
- **Ablations, reported, not ranked:** the Poisson candidate without the season, without income, and without the spend floor. The Technical Design asks theory to narrow the choice to seasonality-aware options, and these show what each piece buys on folds.
- **Not carried forward:**
  - Spend-based robust statistics (median and MAD, seasonal spend ratios): ruled out by the POC, where they reach precision ≤ 0.43 even on frequent categories.
  - Seasonal decomposition residuals (STL and similar): ruled out by theory. They need two full cycles per series, and users have at most 36 months, so the first two years of every user would go unscored.
  - Isolation forests over period features: no hard kind to separate and one reason kind, so a forest adds only opacity.

### 4. Decision rule

Fixed before the round runs. It follows FR-7 §4 where it can.

- **A common operating point:** **0.035 flags per post-warm-up user-month**, the POC leader's 0.80 point, fixed now. In each held-out fold, a run flags its highest-scoring periods up to that rate times the fold's user-months. That needs no labels, so every run is compared on the same number of alerts.
- **Eligibility:**
  - out-of-fold precision ≥ 0.70 at the run's own cutoff, tuned to 0.80 on the other folds;
  - recall at the common rate above the baseline's at the same rate.
- **Ranking:** out-of-fold recall at the common rate. Ties against the leader are judged by a paired **user** bootstrap, as in FR-7.
- **Reported beside the ranking:** each run's out-of-fold precision and flag rate at its own cutoff, read before `finalize`, since that precision predicts the test gate.
- **Tie-breaks, in order:**
  1. out-of-fold precision at the run's own cutoff;
  2. **the negative binomial before the Poisson** (decision 6): in a tie on this data, take the variant that doesn't assume Poisson counts, since real counts aren't;
  3. batch cost;
  4. operational simplicity.

  There's no reason-accuracy tie-break: FR-8 has one reason kind.
- **No shipping twins:** scorers don't train on labels, so `shipping_params` is empty (FR-7 §4).
- **Test users are scored once,** for at most three finalists plus the baseline, through `finalize`.

### 5. The operating point and the gates

- **The cutoff:** FR-7's `Thresholded(base, precision=0.80)`, generalized from transaction ids to period keys. Its `fit` passes no labels to the base scorer and places the cutoff at the target precision on the training users. Within validation, each fold's cutoff comes from the other folds.
- **Every label-tuned parameter is fitted within the folds** (κ), as FR-7's review required. The two product rules aren't tuned (§3).
- **Tune to 0.80, gate at 0.70** (decision 4). That's FR-7's reasoning:
  - test users are 120, with 190 monthly labels; at 0.035 flags per user-month that's about 140 flags, so measured precision swings by about ±0.07;
  - a cutoff tuned to 0.70 would fail the gate about half the time;
  - in the POC, tuning to 0.80 instead of 0.70 costs 3 points of recall (0.577 to 0.545).
- **Promotion gates (test users):**
  1. precision ≥ 0.70, with its user-bootstrap interval;
  2. recall at the common rate above the baseline's at the same rate (decision 5);
  3. every flag has a reason, a size and valid drivers: a contract check, so it can't fail silently.
- **Reported, not gated:**
  - recall on `clear` labels;
  - excess coverage of the drivers, next to the top-5 baseline (equal by construction, §7);
  - the error of the quoted "usual" against the generator's expected spend;
  - **recall on simulated basket-size spikes** (decision 1, [Main risk](#main-risk-one-mechanism)): on a seeded sample of non-label months in the planted categories, spend × U(1.8, 3.0) with counts unchanged, recall at the run's cutoff. On validation for every run and on test for finalists. It isn't gated: the design misses them by construction, and the number is there so the gap is measured, not assumed;
  - precision and recall when the categories are the promoted categorizer's predictions rather than the truth. That's what serving sees, and it's measured on test users only, because train users' predictions are in-sample for the categorizer.
- **If rank 1 fails a gate,** it is investigated, not resolved by promoting rank 2 (FR-3's rule).
- **For real data (v2):** promotion records the flag rate at the cutoff, as FR-7's does, so `Thresholded(rate=…)` can place a cutoff without labels.

### 6. Evaluation task

A new task, `spending_spikes`, built from the FR-3 framework's parts, beside `unusual_transactions`.

- **Examples:** every (user, spending category, month) after the warm-up, on true categories, with the aggregate and season-profile columns. The label (`spike`, `ignored`, `normal`) and the `tier` sit on the evaluation side and come from the label contract.
- **Splits:** `by_user_split`: train users for validation (5 folds of whole users, stratified by persona), test users for test.
- **Leak checks:**
  - no user in two sets;
  - season profiles leave out the scored user;
  - validation profiles contain no test user's rows;
  - each period's profile is as of the first of its month;
  - each period's features use only earlier months (checked by appending later months to a user and comparing scores);
  - no truth or persona column reaches the model.
- **Metrics** through `Truth.score_periods` and `score_drivers`, so the contract decides every outcome:
  - precision, recall, recall on `clear` labels;
  - recall at the common rate (the ranking measure);
  - precision and flag rate at each run's own cutoff;
  - PR-AUC, flags per user-month, user-bootstrap intervals;
  - excess coverage, and the median absolute error of the quoted usual level;
  - recall on simulated basket-size spikes (§5), and the look-alike counts of [Main risk](#main-risk-one-mechanism).
- **Diagnostics for the report:**
  - false positives by category, month of year and usual purchase count (under 4, 4–15, 16+ a month);
  - misses by tier, category, usual count and the user's history length;
  - recall per persona, for reading only, never gated: the family slice tests seasonality, the freelancer slice income coupling.
- **Baselines:** mean ± k·std, through the same `Thresholded` wrapper, gated; the simple count rule, reported only.

### 7. Reasons, size and drivers

A spike's reason is rendered by a deterministic template from its `evidence`, as FR-7's reasons are. Neither the model nor the LLM writes it.

| Evidence | Meaning |
| --- | --- |
| `category`, `period_start` | The flagged period |
| `actual` | The month's net spend in the category |
| `usual`, `usual_months` | The user's average monthly spend there over the previous 12 months (or as many as exist, at least 3) |
| `excess`, `ratio` | `actual − usual`, `actual / usual` |
| `count`, `usual_count` | Purchases this month; the average over the same months |

Illustrative, with the POC's median flagged spike:

> "You spent $1,853 on Dining in August, $1,105 more than your average month over the past year ($748). That came from 48 purchases, against about 19 in an average month."

- **Only the user's own numbers appear** (decision 7). The season profile and the income adjustment change the **score**, never the quoted usual, which matches FR-7's rule for merchant profiles (NFR-2). The POC shows the plain average is as accurate a "usual" as the model's expectation (9.0% against 9.8% median error), and every number in the sentence is one the coach can recompute from `get_spending_summary` (FR-14).
- **"Usual" is defined once, here.** "Your average month over the past year" closes the Web App UI's open question about what "usual" means. The Overview can use the same definition later.
- **The spend floor** guarantees the sentence never describes a month that's barely up: a flag always has `ratio ≥ 1.3`.
- **Drivers:** the 5 largest outflows in the period and category. Under FR-2's excess coverage, no other 5 charges in the period score higher, so drivers need no model, and the round doesn't compare driver methods. They're labelled **"Largest charges"**, not "what caused it". They cover about 69% of the excess on average, and a spike is mostly many ordinary purchases. The rest is summarized: "and 43 more purchases".
- **The kind label** in the UI is "Spending spike".

### 8. Serving

**Computed on request from the session's ledger** (decision 8). The tool builds that user's monthly aggregates from the ledger the session sees, with FR-6's effective categories. It then scores them with the promoted model and the nightly season profiles. A user's spikes therefore always agree with their Spending totals, and a correction that moves charges out of Dining moves the spike with them.

- **Cost:** one user is about 12 categories × 36 months. Scoring is closed-form (a rate and a tail probability), so a request takes milliseconds. Milestone 4 adds a timing test well inside NFR-5's 2 s.
- **Serving state** comes from the promotion, as FR-11's `forecasts.json` does for goals:
  - the model (β, the dispersion, κ, the cutoff) and the two product rules;
  - the season profile table (§2), built from all users' predicted categories by the nightly job, or from the full `--data` pool at image build time for the demo bundle.
- **Why not flag files like FR-7:** a nightly file is computed on model categories. The demo applies corrections per browser session on read (FR-5/FR-6 §3), so a stored spike could quote a Dining total the user's own Spending page no longer shows. FR-7's flags don't have this problem, because a charge's flag doesn't depend on its category.
- **The tool:** `detect_anomalies(start_date, end_date)` fills `spending_spikes` with the flagged complete months that overlap the range:

  ```json
  "spending_spikes": [{"category": "Dining", "period_start": "2026-08-01", "period_end": "2026-08-31",
    "actual": 1853.00, "usual": 748.00, "excess": 1105.00, "ratio": 2.48,
    "count": 48, "usual_count": 19, "usual_months": 12,
    "reason": "You spent $1,853 on Dining in August, …",
    "largest_charges": [{"transaction_id": "…", "date": "2026-08-14", "merchant": "…", "amount": 212.40}],
    "other_purchases": 43}]
  ```

  When the range holds no complete month, or the user has fewer than 3 earlier months, the tool says so (`spikes_status: "too_short"` or `"month_in_progress"`). The coach must then say it can't judge yet, not that nothing was unusual (PRD risk: short histories).
- **The coach** answers "Why was August so high?" with `detect_anomalies` for August. With no spike, it says no category ran well above usual and uses `get_spending_summary` to name the biggest categories or charges. It doesn't invent a spike (FR-14).
- **"Worth a look":** the spikes card lists flagged months that ended in the last 60 days, FR-7's window (decision 9). For the demo, dated Sep 30, 2026, that's August and September 2026. Each shows category, month, actual against usual, the reason and the largest charges. The Overview's "Worth a look" card counts them with the unusual charges.
- **Without a promoted scorer** (open question 1, FR-11's terms): the pipeline ships either way. The spikes card falls back to a clearly labelled simple rule, never "not available". The fallback is the simple count rule of §3 with the two product rules (spend ≥ 1.3× usual, at least 2 purchases in a usual month), cut at the common rate of 0.035 flags per user-month, which needs no labels (owner, Oct 4, 2026, on #58). The card and the tool label it "simple rule" (`"method": "simple_rule"`), and the reason wording is unchanged, since it quotes only the user's own numbers. When a scorer is promoted, the PR that promotes it switches the card to the model.
- **Flag actions (v1.1, FR-9).** The flag id is the model version plus `user_id`, category and `period_start`, so it stays stable for actions to key on. Actions will be stored per user in FR-5/FR-6's feedback store. "Expected, all good" will suppress that category's flag for that month only.

## Metrics and why

1. **Precision at the tuned cutoff** is the PRD's metric and the gate. False alarms are the PRD's named risk, alert fatigue.
2. **Recall at a common flag rate** ranks candidates and compares against the baseline, so every run is judged on the same number of alerts, as in FR-7. The baseline never reaches precision 0.70 at any cutoff, so "recall at precision 0.70" would compare against nothing.
3. **Recall on `clear` labels:** weak labels are, by definition, months the noise cancelled. They are reported, not hidden, and don't dominate selection.
4. **Flags per user-month** is the alert burden in the user's terms. At 0.035 it's about one spike per user every 2.4 years, on top of FR-7's 1.3 unusual charges a year.
5. **Excess coverage and the error of "usual"** measure the two halves of FR-8's explanation: which charges, and how big. Both are reported, not gated; the PRD sets no target for them.
6. **Recall on simulated basket-size spikes** measures the main risk: what the labels can't. Reported, not gated.
7. **User-bootstrap intervals,** because a user's months aren't independent.

## Options considered

### A. What the score measures

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Monthly spend (Technical Design baseline) | Literally the requirement's words | Precision 0.065 at the common rate: one-offs dominate |
| **(b) Purchase count, with spend required to be up ≥ 1.3× (chosen)** | 0.81 precision, 96% of the oracle's recall; the spend rule keeps the reason honest | Misses basket- and price-driven spikes by construction (0.3% recall on simulated ones; the main risk). Doesn't call a single huge charge a spike (FR-7's job) |
| (c) Count and spend combined into one score | Might catch spikes with few, large purchases | No such spikes are planted, so the labels can't score the gain; adds spend's noise back (precision ≤ 0.43 on frequent categories) |

### B. Where seasonality comes from

| Option | Pros | Cons |
| --- | --- | --- |
| (a) None | Simplest | Family Augusts and Decembers become false positives |
| (b) The user's own previous year only | Personal | Nothing in year one; one noisy year becomes the season |
| **(c) Own, shrunk toward a cross-user category profile (chosen)** | Works from month 4; personal as history grows | A nightly table and a leave-user-out rule |
| (d) The persona's profile | Strong prior on synthetic data | Real users have no persona label (FR-11, decision 11) |

### C. The cutoff

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Tune to 0.70 | Most recall | Passes the test gate about half the time |
| **(b) Tune to 0.80, gate at 0.70 (chosen; FR-7's choice)** | Passes reliably | About 3 points of recall in the POC |
| (c) A fixed flag rate | No labels; carries to real data | Ignores the precision target where labels exist; recorded for v2 instead |

### D. "Recall above a simple rule-based alternative"

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) The Technical Design's mean ± k·std at equal flag volume, gated; the simple count rule reported (chosen)** | The bar was set before seeing results, as in FR-7 | The gated bar is low (recall 0.044) |
| (b) The simple count rule as the gated bar | A stronger, more meaningful comparison | Moves the bar after the POC showed counts work; FR-7 declined the same move |

### E. Serving

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Nightly spike files, like FR-7's flags | Matches FR-7; no compute on request | Computed on model categories: disagrees with the user's corrected Spending totals |
| **(b) On request from the session's ledger, with nightly season profiles (chosen)** | Always matches what the user sees; corrections apply at once | Compute on the request path (milliseconds) |
| (c) Nightly files, recomputed on request only for users with corrections | Least request-time compute | Two paths that must agree, and a test for each |

### F. What "usual" means in a reason

| Option | Pros | Cons |
| --- | --- | --- |
| (a) The model's expectation (season and income adjusted) | The number the score used | Contains other users' aggregate; not recomputable by the user or the coach |
| **(b) The user's own average over the previous 12 months (chosen)** | Own numbers only; recomputable; as accurate in the POC | Can differ from the expectation that drove the score in seasonal months, where spikes are rare |
| (c) The same month last year | Seasonal by nature | Missing in year one; one noisy month |

### G. Which months are scored

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Complete months only (chosen)** | Matches the labels; nothing to project | A spike shows only after the month ends |
| (b) Month to date, projected | Earlier warning | Needs a pacing model and its own labels; v1.1 or later |

## Testing

- **Unit:**
  - point in time: appending later months never changes an earlier period's score;
  - the 3-month minimum; zero-filled months; Income never scored;
  - season profiles: under 20 users means a season of 1; leaving a user out by subtracting their terms equals rebuilding the table without them; as of the month; serving profiles use predicted categories, so one user's corrections never change another user's season;
  - the contract rejects a flag below the spend floor or the minimum volume, without evidence, or with a persona column;
  - the reason renders from evidence alone, and every number in it equals a sum over the user's own ledger;
  - `Thresholded` on period keys passes no labels to the base scorer.
- **Isolation:** nothing under `intelligence/` reads truth (existing test). `detect_anomalies` returns user A's spikes to A and nothing to B. A's corrections change A's spikes and never B's.
- **Framework (toy task):** out-of-fold cutoffs and grid choices; recall at the common rate; validation profiles without test users; no twins; finalize scores test users once.
- **Serving:** a spike's `actual` equals `get_spending_summary` for the same category and month, with and without corrections. A request for one user scores within the timing budget.
- **Demo bundle:** the season profile file is built from the full `--data` pool (it has cells with 20+ users), and the fallback card says "simple rule" when no scorer is promoted.
- **Slow (default data):** the promoted scorer passes the gates on test users.

## Milestones

One PR per milestone, stacked as FR-7's were.

1. **Contract and features:** the `SpikeScorer` contract and runtime checks; monthly aggregates over a given category column; season profiles in the feature pipeline (as of the month, exact leave-one-user-out, at least 20 users), with the check that the POC's result holds with the clipped mean, against the fixed pass condition of §2 (if it fails, back to the owner before milestone 3); the income ratio; the reason template.
2. **Task:** `spending_spikes`, with user-grouped folds, two-pool profiles, `Thresholded` on period keys, metrics through the contract, recall at the common rate, the basket-size diagnostic, the decision rule of §4, and the baseline.
3. **The round:** the simple count rule, both count models and the ablations on validation, with a round results report that states the main risk and its measured size.
4. **Finalize, promote and serve:** finalists scored once on test users; promote if the gates pass. A failing gate means no promotion and a root-cause analysis (open question 1). Then the serving state, the on-request scoring, `detect_anomalies`' spikes half, the 1f card (with the simple-rule fallback), the coach and the demo bundle with its season profile file. This milestone ships whether or not a scorer is promoted.
5. **Docs:** the Technical Design (contract, the feature-pipeline row, model selection, the evaluation row), the PRD's first-measurement note, and the Web App UI (1f, the "usual" question).

## Implementation notes

Departures and results from milestone 1 (#59, revised after its review):

- **The season-profile check holds** (decision 13; `scripts/fr8_season_check.py`, train users only). At 0.035 flags per user-month, the POC leader with milestone 1's season profile has recall **0.518** and precision **0.769**, inside the fixed intervals (0.506–0.578, 0.737–0.860). The margin on recall is about 3 true positives. Step by step, on milestone 1's grid:
  - the POC's median profile, rebuilt: 0.543 and 0.805. On the POC's own grid the script reproduced the POC exactly (0.545, 0.809); categories now start at their first purchase (below);
  - the clipped mean, still pooled like the POC's (every user and month, nobody left out): 0.533 and 0.791;
  - then as of each month, the scored user left out and at least 20 others: 0.518 and 0.769.

  Most of the cost is the as-of rule, which is the honest part. The POC's pooled profile saw later years. As of the month, no earlier same-month observation exists anywhere in 2024 (observations start once a user has 3 months, in January 2024), so 35.4% of scored periods, all in 2024, get a season of 1. Floor on the plain usual or on the seasonal one: the same result.
- **The user's own season is the same month a year earlier,** as §2 decided. The first version averaged every earlier year; the review on #59 measured it as slightly worse (0.513 and 0.762) and asked for the decided design.
- **A category starts at the user's first purchase in it** (review on #59). Zero-filling from the user's first month put pre-adoption zeros in 5.1% of scored periods' windows and a +log 3 season observation at adoption. The grid shrinks from 89,760 to 88,232 periods.
- **Only complete months are kept** (decision 10, review on #59). `scoring_periods` takes a required scoring date, `as_of`, and keeps months that have ended by then. Evaluation passes the dataset's last day, which ends a month. A user's partial first month still counts towards their usual level; synthetic users all start on the 1st. Real accounts linked mid-month need an account start date (v2).
- **Season observations need 3 earlier months,** as in the POC, so a user's first months never shape a season.
- **A period's id** is `"<user_id>|<category>|<period_start>"`, since the framework's contracts key on one column. The tool and the label contract still use the three columns. A flag's evidence must name its own period, and its ratio must equal actual over usual.
- **Models supply only a score and a flag decision.** The base class applies both product rules on top and builds the evidence from the input row, so no model can flag a month the rules exclude or quote a number that isn't the user's own. The scorer passes models only the contract's input columns, and rejects persona and the truth tables' columns by name.
- **Reason wording:** the month carries its year ("in August 2026"), since three years of history make "August" ambiguous. Amounts are whole dollars, rounded half up, and the excess shown is the rounded actual minus the rounded usual, so the sentence adds up. Evidence keeps the cents. "About" is dropped when the usual count is a whole number.

From milestone 2:

- **The product rules apply to the baseline too.** The contract checks both rules on every flag, from any model, so the gated mean ± k·std baseline also flags only months with spend ≥ 1.3× usual and at least 2 purchases in a usual month. That makes it a stronger bar than the POC's raw baseline. On validation at 0.035 flags per user-month, its recall is **0.163** (precision 0.289), against the POC's 0.044 (0.065). It still never reaches precision 0.80, so its own cutoff flags nothing. **Confirmed by the owner (Oct 4, 2026), before the round:** the round gates against the baseline with the rules, not the raw one, since no servable model could produce the raw baseline's flags.
- **Two §3 items run with the product rules too** (owner, Oct 4, 2026, after review on #60). The reported simple count rule, defined in §3 with no spend floor, runs with both rules like every flag. The ablation without the spend floor can't run under the contract, so it's dropped, and the POC's measurement stands for it (the floor alone costs 0.5 points). Both are report-only; nothing gated changes.
- **Gate 3 checks the drivers too** (review on #60): every flag needs evidence and valid drivers, the 5 largest charges in its period and category, scored by FR-2's rules (`flags_with_invalid_drivers`).
- **History length counts from the user's first transaction,** warm-up included (review on #60).
- **Test scoring also reports precision and recall on predicted categories** (§5), from the promoted categorizer's predictions for every user, scored on test users only. It isn't gated.
- **The diagnostics count the look-alikes** (the main risk): normal months in the planted categories with spend ≥ 1.5× usual, and how many a run flags.
- **`SpikeThresholded` is its own class,** not FR-7's `Thresholded` generalized. FR-7's class is inside a promoted model, and changing it would risk that model's loading. It reuses `top_k` and keeps the same `precision`, `rate` and `search` parameters.
- **A rate cutoff is fixed when it's fitted.** `rate=` places the cutoff at the score that flags that many periods per user-month on the rows it was fitted on, and keeps it. FR-7's re-ranks every batch. Serving scores one user's months at a time (§8), and re-ranking within one user would flag their top months whatever they were. This is the simple-rule fallback's path (decision 12).
- **Ignored periods stay in the rate budget,** as FR-7's duplicate originals do (review on #34): a deployed cutoff can't know them. The POC left them out of its ranking, so its rates are slightly optimistic.
- **Simulated basket-size spikes are examples labelled `basket`:** 400 for train users and 200 for test users, seeded. They copy normal months in the planted categories that have some spend and at least 2 purchases in a usual month, with spend × U(1.8, 3.0). They're never trained on and never count towards a rate or precision. On validation, the spend-based baseline flags **25%** of them at the common rate, against about 0.3% for the count leader in the POC. That's the main risk, measured.
- **The negative-binomial tie-break** (decision 6) is a scorer attribute, `assumes_poisson` (1 for the Poisson, 0 otherwise), logged as `fit.assumes_poisson` and read as the second tie-break.
- **Drivers in evaluation** are the 5 largest outflows in the period on true categories, so excess coverage is reported for true positives at the run's own cutoff. The error of "usual" is measured on the same periods.

From milestone 3 (the round on validation, FR-8 Spending Spikes — Round Results):

- **Rank 1: `count_negbin`, 0.521** recall at 0.035 flags per user-month, tied with `count_poisson` (0.516) on recall and on own-cutoff precision. Decision 6's second tie-break decided it. They're near-equivalent, not the same (review on #61):
  - the pooled over-dispersion is 0 under its estimator, a median of rows' own estimates clipped at 0, because fewer than half the rows (46%) are over-dispersed. The ratio-of-sums moment estimate would be 0.034;
  - each row's own estimate still counts, so the two differ by more than 1% on a quarter of rows (own-cutoff recall 0.513 against 0.523);
  - the estimator is documented as run, not changed after the results.
- **Ablations are report-only runs:** a config tag `report_only: "true"` keeps a run out of the ranking and out of `finalize`. It's a tag rather than a config field, so no existing run's config hash changes. The ablation without the spend floor isn't possible, since the contract enforces the floor on every flag; the POC's measurement stands for it.
- **Seasonality earns nothing out of fold** (0.518 without, 0.516 with), and κ landed on the top of the decided grid (2). Both are reported, not acted on.
- **The simple count rule is a baseline run** (reported, never gated), so `finalize` scores it on test users with the gated baseline, as FR-7's report-only baseline was. Since decision 15 it's cut at the fallback's rate, not at precision 0.80. Its departures from §3: it runs with both product rules (an owner decision), and its expected count is floored at 0.3, an implementation detail shared by every count model.

From milestone 4, test scoring and promotion (FR-8 Spending Spikes — Round Results):

- **Test users were scored once** (owner's go-ahead on #61, after decisions 15 and 16). Rank 1, `count_negbin`, passed every gate:
  - precision **0.705** (0.634–0.770) against 0.70;
  - recall at the rate **0.521** against the baseline's 0.158;
  - every flag with evidence and valid drivers.
- **Promoted** as `8c428c54-d85b4650-64917ea6` under milestone 4's rule (promote if the gates pass), and accepted by the owner after seeing the test results (Oct 4, 2026, on #62). The go-ahead on #61 covered test scoring, not promotion in advance. **Known limit:** the precision margin is thin, 0.799 out of fold and 0.705 on test, with the flag rate rising from 0.035 to 0.039. Reviewing the cutoff on real data is part of v2's re-validation.
- On predicted categories, precision is 0.777 and recall at the rate 0.526. The fallback simple rule would also have passed precision (0.705).

From milestone 4, serving (built before test scoring, so the simple rule served until the promotion):

- **`spikes.json`** holds the scorer (its spec and what it fitted) and the season table's sums and counts per (category, month of year, as-of month): no row of any user (a test checks it). It's 21 KB for the default data.
- **The simple rule's cutoff is fitted on the whole pool at 0.035 flags per user-month,** without labels (decision 12). On the default data's 360 users it's a tail score of 8.93. A promoted model replaces it in the same file, and nothing else changes.
- **The demo bundle categorizes the full `--data` pool** with the promoted categorizer for the season table: 51 s for 1.1 million transactions. The whole `build-demo` takes about 1 minute 45 seconds.
- **Per request:** one user's spikes take about 75 ms (`detect_anomalies` over their whole history), well inside NFR-5. A test holds it under 1 s.
- **`detect_anomalies`' two halves are independent:** unusual charges are "not available" without an FR-7 model, and spikes without a spikes file. Each spike says `simple_rule` when the simple rule found it. The card shows a "Simple rule" badge and the coach says so.
- **`/healthz` scores every demo account's spikes,** so a deploy's smoke test reads user data (and reports which scorer serves).
- **Review on #62:**
  - the card and the coach say "simple rule" whenever that scorer serves, even with no spike;
  - the deploy's smoke test requires a spikes scorer in `/healthz`;
  - `spikes.json` records the categorizer its table was built on, and loading it fails against other predictions;
  - the source line says when unusual charges weren't checked.
- **The table's cells are cumulative by as-of month,** so differencing two consecutive cells shows the total change of the users who joined or changed that month. Every cell has at least 20 other users (65+ on the default data), within the decided format (review on #62).
- **What the simple rule got wrong in the demo:** it has no season, so the family account's August 2024 Shopping, a seasonal peak, reads as a spike. The candidates' season term exists to explain such months. It's outside "Worth a look"'s 60-day window, but the coach can find it when asked about 2024.

## Decisions and open questions

**Decisions** (owner, Oct 4, 2026, confirmed on #58). Any later decision that affects evaluation or test scoring goes to the owner before it runs.

1. [x] **Score purchase counts for v1** (§3, option A-b), with three conditions:
   - **the spend floor (≥ 1.3× usual) is a product rule only.** It keeps the reason honest; the design doesn't credit it with recall, since ablated it lowers recall at p ≥ 0.80 (0.550 to 0.545);
   - **the minimum volume is fixed before the round from the product argument** (at least 2 purchases in a usual month; 1 becoming 4 isn't a meaningful spike) **and not tuned on labels.** Tuned, it rediscovers where the generator plants;
   - **the basket-size gap is the main stated risk, and it's measured.** The round reports recall on simulated basket-size spikes, and v2 revisits it with real data ([Main risk](#main-risk-one-mechanism)).
2. [x] **Seasonality from the user's own history, shrunk toward cross-user category profiles:** at least 20 users, the scored user left out, as of the month; no persona label (§2, option B-c).
3. [x] **A month that's high because of one large charge is not a spike;** FR-7 judges the charge (Scope).
4. [x] **Tune to precision 0.80, gate at 0.70** on test users (§5, option C-b).
5. [x] **The baseline:** the Technical Design's mean ± k·std at equal flag volume, gated; the simple count rule reported, not gated (§5, option D-a). Both product rules apply to the baseline's flags too, as the contract requires (owner, Oct 4, 2026, confirmed before the round; implementation notes).
6. [x] **The decision rule of §4,** fixed before the round runs: recall at 0.035 flags per user-month, user-bootstrap ties, then own-cutoff precision, negative binomial before Poisson, cost and simplicity; no shipping twins.
7. [x] **Reasons quote only the user's own numbers;** "usual" is the average of the previous 12 months (§7, option F-b).
8. [x] **Serve on request from the session's ledger with effective categories** (§8, option E-b). For the demo bundle, the season profiles are built from the full `--data` pool at build time and shipped as a bundle file (§2).
9. [x] **"Worth a look" shows spikes in months that ended in the last 60 days** (§8).
10. [x] **Complete months only** (§1, option G-a).
11. [x] **Drivers are the 5 largest charges, shown as "Largest charges",** with a count of the rest (§7).

**Open questions**

1. [x] **The Oct 6 demo: aim for it, on FR-11's terms** (decisions 10 and 13 on #54) (owner, Oct 4, 2026, on #58):
   - the pipeline ships either way; without a promoted scorer, the spikes card falls back to a clearly labelled simple rule, not "not available" (§8);
   - test users are scored once, for finalists;
   - a failing gate means no promotion and a root-cause analysis on train and validation data (data, model or criteria); whether test is scored again is decided after it. No gate is loosened.
2. [x] **A per-user income elasticity: deferred** (owner, Oct 4, 2026, on #58). The pooled value stays for v1.

**Decided after the revision for #58's review** (owner, Oct 4, 2026, on #58)

12. [x] **The fallback rule:** if no scorer is promoted, the spikes card uses the simple count rule with both product rules (spend ≥ 1.3× usual, at least 2 purchases in a usual month) at 0.035 flags per user-month, which needs no labels, labelled "simple rule" (§8).
13. [x] **Season profiles use a clipped mean of log ratios,** with the pass condition fixed before milestone 1 checks it. The check passes if, at 0.035 flags per user-month, the leader with the clipped mean has recall within the POC's user-bootstrap interval, **0.506–0.578**, and precision within it, **0.737–0.860**. Both intervals were printed by the script at `bd5ae63`. If either falls outside, the statistic comes back to the owner before the round, not swapped silently (§2). The first wording, "at precision ≥ 0.80", was replaced by the precision interval (owner, option (b) on #58), since the POC's own point precision was 0.809.
14. [x] **κ's grid is {0.5, 1, 2},** tuned within folds (§2).

**Decided before test scoring** (owner, Oct 4, 2026, on #60 and #61, after the round on validation)

15. [x] **The simple count rule that `finalize` scores is decision 12's fallback exactly:** both product rules, cut at 0.035 flags per user-month with no labels (`rate=0.035`), not at precision 0.80. Its test result says whether the fallback would be good enough to ship if no model is promoted. Like the gated baseline, it's scored on test users with the finalists.
16. [x] **κ's grid stays {0.5, 1, 2},** although κ = 2 was chosen in every fold. Widening it after seeing validation results would be tuning on the outcome. A wider grid is a v2 item.
