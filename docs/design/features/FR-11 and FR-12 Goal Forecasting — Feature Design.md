# FR-11 and FR-12 Goal Forecasting — Feature Design

Oct 3, 2026 · @Sidd · Status: **Accepted** (owner, Oct 4, 2026, on #50) · Branch: `docs/fr-11-12-design`

## Summary

This feature tells a user whether they're on track for a savings goal: where it will likely land, the range, the chance of making it, and what would close the gap. It accounts for seasonality and each user's own income and spending.

- **Requirements:**
  - **FR-11** (P0): *"Tell the user whether they are on track, the projected amount, the gap, and how certain that is."*
  - **FR-12** (P0): *"Account for seasonal patterns and each user's own income and spending behavior."*
  - **PRD success metric:** forecast error (RMSE) at least 15% lower than a naive forecast, and an on-track call better calibrated than naive.
  - **NFR-7:** every forecast has a range.
- **One design for both.** FR-12 isn't a separate feature; it's a requirement on how FR-11's forecast is made. So it's written as acceptance criteria per persona: the family slice tests seasonality, the freelancer slice tests irregular income.
- **Starting point:**
  - FR-1 generates goals whose balance is a hidden share of the user's monthly net savings, and 248 of them (153 train, 95 test) have a known outcome.
  - FR-10 built goals: the store, the tools and the pages.
  - `forecast_goal` returns "not available yet". `check_goal` gives facts but no verdict, and the setup screen has no fit badge.
- **Feasibility, measured on train users only** ([evidence](#feasibility)). Goals take two paths, measured separately:
  - **Goals with a track record** (the generated ones). Inferring each goal's share from its history and simulating future months scores a Brier of **0.171**, against **0.320** for the naive pace call and **0.250** for a flat 50%.
  - **New goals:** every goal created through FR-10 and every draft on the setup screen, which start from a prior share. Scoring each evaluation goal as if it were new gives **0.208**. This is the number behind the fit badge.
  - **Both paths are calibrated.** In each status band, the share of goals actually met falls inside that band, and the 80% range covers 83% of outcomes.
  - **Freelancers are the hard case:** 0.239 with a track record, 0.251 as new, right at the flat 50%. That's FR-12's question.
  - **The PRD's RMSE target can't be met against seasonal-naive.** Sensible models beat last month's value by about 70%, but seasonal-naive (the Technical Design's baseline) by at most 2% at 6 months. Even the true level in hindsight is only 19% better. Net savings are dominated by one-off purchases no model can see coming. [Decision 1](#decisions-and-open-questions) proposes recording a target change before anything is scored on test.
- **Approach:**
  1. **Forecast each user's monthly net savings as simulated paths:** a level, a seasonal profile shrunk toward the persona's, and the user's own month-to-month variation, widened by a factor tuned only on how often the range covers the realized balance.
  2. **A share per goal:**
     - generated goals: inferred from their history;
     - FR-10 goals: inferred from changes between the user's own saved entries, once two are far enough apart;
     - any other goal: a prior, which is the typical total allocation divided by the number of the user's active goals.
  3. **Any goal is arithmetic over the paths,** saved goals and drafts alike:
     - the chance of making it;
     - the median and the 80% range;
     - the gap;
     - the monthly top-up that would put it on track.
  4. **Nightly, store each user's forecast state, not the paths:** level, profile and residuals, a few hundred numbers. Simulate per request with a fixed seed, so answers are reproducible and fast enough for the live setup check.
  5. **A round of candidates through the FR-3 framework,** gated on Brier and calibration for both paths, overall and per persona.
- **Owner decisions** (Oct 4, 2026, on #50) settled all nine questions; see [Decisions and open questions](#decisions-and-open-questions).
- **Principles:**
  - Carried from FR-2: the modeled behavior isn't tuned to make the target pass, and test users are scored once, for finalists.
  - Specific to goals ([why](#why-nothing-is-tuned-on-outcomes)): nothing is tuned on whether goals were met, because those outcomes are planted relative to the future.

## Context

### How goals and their outcomes are generated (FR-1, stage 9)

- A goal's balance grows by a fixed, hidden **share** of the user's monthly net savings from `created_date`. The balance is floored at zero each month, so a bad month can empty it, and it stays empty until net savings turn positive.
- Each user's shares add up to 0.5–0.9 of net savings, split across their 1–2 goals.
- **The outcome is planted by the target.** For goals that end inside the history, the generator computes the balance actually reached by the target date. It then sets the target at 0.6–0.85× that (`on_track`, always met), 0.95–1.05× (`borderline`, met about half the time) or 1.3–1.8× (`off_track`, never met).
- `as_of_date` is 3–12 months before the target, and only transactions up to it may be used (Technical Design, evaluation controls).
- The default dataset has **36 months** of history (Oct 2023 – Sep 2026), so up to three seasonal cycles.

| Split | Goals | Outcome known | Per user | Of which `borderline` |
| --- | --- | --- | --- | --- |
| Train users (240) | 358 | 153 | 0.64 | 44 |
| Test users (120) | 187 | 95 | 0.79 | not opened |

### Why nothing is tuned on outcomes

The met rate is planted relative to the realized future: the generator picks a class mix (here 35% on track, 29% borderline, 37% off track) and sets each target from the actual outcome. Anything tuned on "was it met?" learns that mix, not how real people's targets relate to their outcomes. That covers band thresholds, a calibration map, or a spread chosen to minimize Brier.

So the only tuned step reads realized balances, never targets: the spread factor, chosen so the 80% range covers 80% of realized balances. The band thresholds are fixed in advance ([decision 5](#decisions-and-open-questions)). Calibration is checked, not fitted.

### What the personas put into net savings

| Persona | Income | Seasonality that reaches net savings |
| --- | --- | --- |
| Young professional | Salaried biweekly, an annual raise, sometimes a bonus | Holidays and summer travel; mild |
| Family budgeter | Two salaries, sometimes a bonus | School year: August back-to-school, summer camps, December |
| Freelancer | 2–6 clients, lognormal months, 1–2 slow months a year at 0.3×, seasonal billing (Jan 0.6×, Dec 1.35×) | Income drives it; spending partly follows income (elasticity 0.5) |

On top of these come heavy-tailed one-off purchases (travel, repairs, medical, electronics) and the planted unusual charges. In the feasibility results they dominate the month-to-month noise.

### What the mockups and FR-10 ask for

- **1g, 1l Goal detail:** saved of target, the likely range by the target date ("We're fairly confident you'll land between …"), a status, and what would close the gap ("Set aside $75 more a month").
- **1h Goal setup:** a fit badge ("Within reach", "A stretch") for a goal that isn't saved yet. FR-10 deliberately shipped the setup check without it (FR-10 decision 3).
- **FR-10 hand-offs:**
  - `check_goal` gains `p_goal_met` and the range;
  - the 80% interval goes into the forecast contract (Web App UI, gap 4);
  - FR-11 decides how savings split across goals. FR-10 decision 2's starting point was "in proportion to what each needs per month". This design recommends an equal split instead, on product grounds (§3, [decision 2](#decisions-and-open-questions)).

## Scope

**In:**
- the net-savings forecast and the goal forecast built on it;
- `forecast_goal` and the probability in `check_goal`;
- the fit badge;
- the goal detail screen (1g, 1l) and ranges on the goal cards;
- the coach's use of all of the above;
- the evaluation task, candidates, gates and promotion.

**Out:**
- what-if scenarios (Web App UI, gap 5; [decision 7](#decisions-and-open-questions));
- forecasting categories or spending spikes (FR-8);
- investment advice of any kind (NFR-4);
- weekly forecasts (the PRD assumes monthly is enough).

## Feasibility

All numbers are from train users only, on the default dataset generated at `main` `9bd5097` (content hash `b4d43bf4`).
- **Script:** `poc/fr11_12_feasibility.py` on branch `poc/fr-11-12-feasibility`.
- **Output:** `poc/fr11_12_feasibility_train_review.txt`, from `--point=profile --win=24 --prior --review --spread=1.1 --split=equal`.
- **Standard errors** are bootstrapped by user, since one user's goals share one future.
- Test users haven't been scored.

### Forecasting monthly net savings

A rolling-origin backtest: from every month with 12 or more months of history, forecast total net savings over the next 3, 6 and 12 months.

| Model | RMSE, 3 months | 6 months | 12 months | Scaled RMSE, 6 months, vs seasonal-naive |
| --- | --- | --- | --- | --- |
| Last month repeated (naive) | $12,261 | $23,196 | $41,475 | 3.04 |
| Same month last year (seasonal-naive, the Technical Design's baseline) | $5,280 | $6,685 | $9,268 | 1.00 |
| Mean of the last 12 months | $4,950 | $6,734 | $9,268 | 1.01 |
| Mean of the last 24 months | $4,878 | $6,573 | $8,392 | 1.03 |
| Level + seasonal profile, shrunk toward the persona's | $4,827 | $6,878 | $11,155 | 1.01 |
| *Hindsight: the user's true 36-month level (not a model)* | *$4,451* | *$5,428* | *$5,260* | *0.83* |

**Scaled RMSE** divides each error by the user's mean absolute monthly net savings before the origin, so high earners don't dominate. In dollars, they do: the dollar RMSE is mostly the highest-income users.

- **Any sensible level beats last month's value by about 70%** (6 months: $6,573–6,878 against $23,196).
- **Nothing beats seasonal-naive by much,** in dollars or scaled. The best is 2% better at 6 months (dollars) and 9% at 12. Seasonal profiles help only at 3 months, and only for families.
- **The ceiling is low.** Even the true level, known in hindsight, is only 17% better than seasonal-naive at 6 months, scaled (19% in dollars). The rest is one-off purchases and income noise.

By persona, scaled RMSE at 6 months relative to seasonal-naive:

| Persona | Naive | 12-month mean | 24-month mean | Seasonal profile | Hindsight level |
| --- | --- | --- | --- | --- | --- |
| Family budgeter | 2.78 | 1.03 | 1.09 | 1.00 | 0.87 |
| Freelancer | 3.39 | 1.02 | 0.97 | 1.03 | 0.81 |
| Young professional | 3.00 | 0.99 | 1.00 | 1.01 | 0.81 |

### The on-track call: goals with a track record

For the 153 train goals with a known outcome, at each goal's `as_of_date`:

- **Infer the goal's share:** the share at which FR-1's rule reproduces the balance from the net savings since `created_date` (a bisection; the rule is monotone in the share). A goal with a $0 balance has no usable track record, so it gets the median inferred share of goals with a balance: **0.476** (137 goals).
- **Simulate:** 400 paths of the coming months. Each path is a point forecast (a 24-month level, plus the persona-shrunk seasonal profile once 12 months exist), with the user's own monthly deviations resampled. The deviations are scaled by a spread of 1.1, which makes the 80% range cover 83% of realized balances. The balance is applied month by month with the floor, and P(met) is the share of paths that reach the target.

| Call | Brier (lower is better) | Accuracy |
| --- | --- | --- |
| Naive pace: balance ÷ months so far, extended to the target date | 0.320 | 0.68 |
| A flat 50% for every goal | 0.250 | — |
| Inferred share, 12-month mean, no simulation | 0.301 | 0.70 |
| Inferred share, the same point forecast as the paths, no simulation | 0.268 | 0.73 |
| **Inferred share, simulated paths** | **0.171** (user bootstrap 90%: 0.139–0.203) | **0.74** |
| *Oracle: inferred share with the actual future months* | *0.020* | *0.98* |

- **Inferring the share works:** with the actual future, it reproduces 98% of outcomes. The remaining error is all in forecasting the future months.
- **Simulation is what makes the call useful:** 0.171 against 0.268 for the same point forecast without paths. The paths give a probability that reflects how volatile the user is.
- **By persona** (FR-12): family 0.154, young professional 0.127, **freelancer 0.239**.

### The on-track call: new goals

Every FR-10 goal and every draft starts without a track record, so it takes the prior share. To measure that path, each evaluation goal is also scored **as if it were new**: created at its `as_of_date`, its balance entered by hand, its share from the prior.

The prior is the typical total allocation: the median over train users of their goals' summed inferred shares, **0.662**. It's split across the user's goals. Both splits below use the same spread (1.1). The commands are the one above with `--split=equal`, and with `--split=need` (output: `poc/fr11_12_feasibility_train_review_need.txt`).

| Prior split | Brier | Per persona (family, freelancer, young professional) |
| --- | --- | --- |
| In proportion to what each goal needs per month (FR-10 decision 2's starting point) | 0.228 (user bootstrap 90%: 0.190–0.269) | 0.220, 0.274, 0.194 |
| **Equally: 0.662 ÷ the number of the user's goals** | **0.208** (user bootstrap 90%: 0.173–0.243) | 0.198, **0.251**, 0.178 |

- **This comparison can't choose between the rules for real users.** Stage 9 draws each goal's share independently of its target, and plants targets as multiples of the realized outcome, so off-track goals are the ones that need the most per month. Any rule that gives a bigger share to a bigger need loses on generated goals by construction. The two intervals also overlap. The equal split is recommended on product grounds instead: it doesn't make a stretch target look more achievable, and it's easy to explain ([decision 2](#decisions-and-open-questions)).
- **New goals are less certain than goals with a track record** (0.208 against 0.171), as they should be. They're still better than a flat 50% overall. Freelancers are at the flat 50%.

### Calibration by status band

With the bands fixed at 0.3 and 0.7 ([decision 5](#decisions-and-open-questions)), this is the share of goals actually met in each band:

| Band | Track record: goals | Met | New: goals | Met |
| --- | --- | --- | --- | --- |
| Off track / A stretch (p < 0.3) | 64 | 0.20 | 65 | 0.28 |
| Could go either way (0.3–0.7) | 48 | 0.60 | 44 | 0.59 |
| On track / Within reach (p ≥ 0.7) | 41 | 0.85 | 44 | 0.75 |

- Every band's met rate falls inside the band on both paths.
- In finer bins, without the wider spread, p in (0.1, 0.3] was met 48% of the time (27 goals). The spread of 1.1 is what moves those coin flips toward the middle band. Finer bins aren't gated, since 15–30 goals per bin are too few.

### How precise the evaluation can be

- Bootstrapped by user, the Brier's 90% interval is about ±0.03 on 153 train goals. Test users have 95 goals, about 30 per persona, too few to gate FR-12 per persona.
- [Decision 4](#decisions-and-open-questions) proposes more evaluation goals. Draws for one user share one net-savings series and one future, so their errors are correlated, and the effective sample is closer to the number of users than the number of goals. Every standard error and gate tolerance is therefore bootstrapped by user.

## Goals and non-goals

**Goals**

1. For every active goal, saved or a draft: the chance of making it, the likely balance by the target date with an 80% range, the gap, and the monthly top-up that would put it on track. Every number comes from the forecast service, none from the LLM (FR-14).
2. Better calibrated than naive on both paths (track record and new), overall and for each persona (FR-12), on test users.
3. Honest uncertainty: wide ranges for irregular earners, and a plain statement when a short history limits the forecast (PRD risk).
4. The dashboard and the setup check work without the LLM (NFR-6), and the setup check stays fast enough to run as the person types.
5. Reproducible: the same data, model version and goal always give the same numbers (NFR-8).

**Non-goals**

- Beating seasonal-naive on point accuracy by a wide margin. Feasibility shows the ceiling is low; the value is in the calibrated probability.
- Advice on where to keep the money.
- Forecasting a balance the user hasn't told us about. Goals stay notional in v1 (FR-10 §1).

## What it will look like

### Flows

| Flow | Where | What happens |
| --- | --- | --- |
| **Am I on track?** | Goal card, goal detail (1g, 1l), overview card | A status ("On track", "Could go either way", "Off track"), "likely $2,780 by Dec 31, between $2,310 and $3,190", and when short of on track, "setting aside $75 more a month would put you on track" |
| **Reached** | Same places | "Reached", with no status or probability. If the paths show a real chance of dropping back below the target by the date (§3), a note: "You've reached it. Months where you spend more than you earn could draw it down before Dec 31." |
| **Set up a goal** | 1h "How it fits" | FR-10's facts, plus the fit badge from the draft's probability: "Within reach", "Could go either way", "A stretch" |
| **Ask the coach** | Chat | "Am I going to hit my vacation goal?" Wren calls `forecast_goal` and states the band, the range and the top-up, quoting the numbers |
| **Ended goals** | Goals page "Ended" | Unchanged from FR-10. The outcome isn't known, so there's no forecast |
| **Short history** | Everywhere a forecast shows | With fewer than 6 months of history: "Based on only N months of your history, so this is a rough guide", and the range widens |

**The assumption, said once:** next to the range, "This assumes a month where you spend more than you earn draws on what you've set aside", so a family's December dip in the forecast isn't a surprise (owner, Oct 4, 2026).

### Tools

| Tool | Change |
| --- | --- |
| `forecast_goal` (`goal_id`) | Returns:<br>• `status`: `on_track`, `either_way`, `off_track` or `reached`<br>• `p_goal_met`<br>• `projected_balance` (the median)<br>• `range` (10th and 90th percentiles; an 80% interval)<br>• `gap` (target − median, never below 0)<br>• `extra_per_month`<br>• `monthly` (median and range by month, for the chart)<br>• `share_source` (`track_record`, `your_entries` or `typical`)<br>• `months_of_history`, `short_history`, `model_version`<br><br>For a reached goal: `status: reached`, `p_goal_met`, `range`, `gap` and `extra_per_month` are null, and `may_draw_down` is true when 10% or more of paths end below the target |
| `check_goal` | Adds the same forecast fields for a valid draft, and the fit badge's `fit` (`within_reach`, `either_way`, `stretch`) |
| `list_goals` | Each active goal gains `status` and `p_goal_met`. `forecast` becomes `available` |

## Design

### 1. Contracts

- **Net-savings forecaster** (`intelligence/forecasting`, the "Forecasting service" in the Technical Design):
  - `fit_state(user_id, monthly_net, as_of) → ForecastState`: a level, a seasonal profile, a residual pool, the spread, the number of months of history and the model version. It's a small record, computed nightly.
  - `simulate(state, months, n_paths, seed) → paths[n_paths × months]`: deterministic for a given seed.
- **Goal forecast** (`forecast_goal(user_id, goal_id)` in the Technical Design, unchanged in shape): built from the paths, the goal's balance and its share (§3). Checked against its contract at runtime, like every service (FR-3).
- **Interval: 80%,** the 10th to 90th percentile, fixed in the contract (Web App UI, gap 4).

### 2. The net-savings forecast (FR-12 lives here)

- **Input:** monthly net savings, the sum of a month's amounts. Categories never matter, so FR-6 corrections can't change a forecast. Months up to `as_of`; the first month counts only if it starts on the 1st (FR-10's rule).
- **Level:** the mean of the last 24 months, or of the history available if shorter.
- **Seasonality:** a month-of-year profile, used once 12 months exist. Each user's own deviation is shrunk toward their persona's median profile, by the number of years observed, so two noisy Augusts don't become the forecast. The round decides whether seasonality earns its place at each horizon (§4).
- **The user's own behavior:** the residual pool is that user's own monthly deviations from the point forecast. A freelancer's paths swing as much as their past did; a salaried user's are tight. With fewer than 6 months of residuals, the pool is pooled with the persona's, scaled to the user's level.
- **Spread:** residuals are scaled by one factor, tuned on train users so the 80% range covers 80% of realized balances. The tuning goals are training goals whose target month lies inside their user's visible history (another of the user's rows reaches it), each run on its own share over the months that followed (#52). It reads realized balances, never targets ([why](#why-nothing-is-tuned-on-outcomes)).
- **Paths:** the point forecast plus scaled, resampled residuals, 1,000 paths by default.

### 3. From paths to a goal

**The balance recursion** is FR-1's rule: balance = max(0, balance + share × net), month by month from `saved_as_of` to the target month.
- **Owner decision (Oct 4, 2026), a v1 assumption tied to notional goals:** a bad month draws a goal down in proportion to its share. Goals are earmarked money in the same pool, so a deficit comes out of it.
- It also matches how the evaluation goals are generated.
- When v2 links real accounts, money in a separate account doesn't move unless the user moves it, and this rule should be revisited.

**The goal's share**, by where it comes from (`share_source`):

| Source | When | How |
| --- | --- | --- |
| `track_record` | A generated goal with a balance above $0 | Inferred as in feasibility: the recursion from $0 at `created_date` reproduces the balance at `saved_as_of`. Capped at 1 |
| `your_entries` | An FR-10 goal with two saved entries at least 3 months apart in its event log | The recursion runs **from the first entry's amount and date**, not from $0, and the share is the one that reproduces the latest entry. A goal whose saved amount never changed has no information here and stays on the prior |
| `typical` | Everything else: new goals, drafts, $0 balances, entries too close together | The typical total allocation (0.662 on train) **divided by the number of the user's active goals**, a draft included. Goals with their own share keep it; nothing is subtracted, so a new goal never starts at about 0. Ended goals don't count, because they no longer draw on future savings. This is the rule feasibility measured |

- A user's shares never total more than 1.
- **If a user's shares would total more than 1,** the `typical` shares are scaled down until they total 1. Shares from a track record or the user's entries are evidence, so they stay as measured.
- A draft counts as one more active goal, so `check_goal`'s numbers for a draft match `forecast_goal`'s for the same goal once it's saved. For example, a demo user with one active generated goal gets 0.662 ÷ 2 = 0.331 for a draft.

**Results from the paths:**
- `p_goal_met`: the share of paths at or above the target.
- The median and the 10th and 90th percentiles of the final balance; `gap` = max(0, target − median).
- `extra_per_month`: the smallest whole-dollar monthly **deposit** that brings `p_goal_met` to the on-track threshold (0.7). It's added straight to the balance each month, not scaled by the share, since it's money the user sets aside on purpose. It's found by bisection over the same paths, so it's consistent with the forecast and isn't LLM arithmetic. It's null when the goal is already on track or reached.
- **Status and fit,** from `p_goal_met`, on fixed bands:
  - "On track" / "Within reach" at 0.7 or above;
  - "Off track" / "A stretch" below 0.3;
  - "Could go either way" in between.
- **Reached** (saved ≥ target, date ahead): `status: reached`, no probability, range or top-up. The paths still run, but only to set `may_draw_down`, when 10% or more of them end below the target (owner, Oct 4, 2026).

### 4. Candidates

| Candidate | Level | Seasonality | Paths |
| --- | --- | --- | --- |
| **Naive pace** (baseline) | Balance ÷ months so far, extended | — | None: a 0/1 call |
| **Flat 50%** (reference) | — | — | — |
| **Flat level** | 24-month mean | None | Own residuals, scaled |
| **Seasonal, persona-shrunk** | 24-month mean | User profile shrunk toward the persona's | Own residuals, scaled |
| **Exponential smoothing** | ETS with additive seasonality | ETS | Simulated from the fitted model |

- All candidates share §3. Only the net-savings forecast differs.
- Exponential smoothing needs `statsmodels`, a new dependency ([decision 3](#decisions-and-open-questions)), and only users with 24+ months have the two full cycles it needs.
- **Ranking,** written down before the round: the mean of the two paths' validation Brier (track record and new), cross-fitted over train users. Tie-breakers, in order: the worst band's calibration error, then scaled RMSE at 6 months, then cost. Tie sets and shipping twins as FR-4 §5.

### 5. Gates (on test users, once, for finalists)

| Gate | Target | Why |
| --- | --- | --- |
| Brier, each path | Below the naive pace call and below a flat 50% | PRD "better calibrated than naive"; a flat 50% is the honest floor |
| Brier per persona, each path (FR-12) | Below a flat 50%, within the user-bootstrap tolerance | Seasonality and irregular income must each be handled, not averaged away |
| Calibration per band, each path | Each band's met rate inside its band (below 0.3, 0.3–0.7, 0.7 or above), within the user-bootstrap tolerance | FR-12's own question: does the probability say so honestly |
| 80% range coverage | 70–90% of realized balances inside | NFR-7: the range must mean what it says |
| RMSE, 6-month net savings | [Decision 1](#decisions-and-open-questions) | PRD metric |

- The gates are checked in that order by the FR-3 promotion code.
- Tolerances come from the user bootstrap on validation and are recorded before test.

### 6. Evaluation task

- **Examples:** each goal with a known outcome, at its `as_of_date`, scored on both paths: once with its own share source, and once as if new. Only transactions up to that date are used, enforced by the splitter's leak check (Technical Design, evaluation controls).
- **More goals for evaluation** ([decision 4](#decisions-and-open-questions)):
  - FR-1's stage 9 is refactored into a pure sampler over a user's monthly net savings, with the same code and rule. A test checks that stage 9 still produces exactly today's goals, so the generated dataset doesn't change.
  - The sampler is run with more draws per user, under its own seed, into an evaluation-only goal set.
  - **Each draw is scored as its own goal set,** the 1–2 goals stage 9 makes together, with shares summing as the generator's do. Draws never coexist with each other.
  - At today's rate of known-outcome goals (0.64 per train user, 0.79 per test user), 10 draws per user give about 1,500 for train and 950 for test. Their errors are correlated within a user, so precision is reported from the user bootstrap, not from the goal count.
- **Net-savings RMSE, on goal origins** (amended in #52): once per user and `as_of` with at least 12 months of history, the next 6 months' total, against last-month naive and seasonal-naive from the same history. It's in dollars (for the PRD and decision 1's gate) and scaled (for ranking). This replaces the rolling-origin backtest at 3, 6 and 12 months first planned here. The gate needs only the 6-month horizon, and a rolling-origin check in review on #52 gave similar ratios (0.86 against 0.82 of seasonal-naive in dollars), so goal origins don't flatter it.
- **Splits:** train users for fitting and tuning (the persona profiles, the typical allocation, the spread), test users once. This is the same separation as FR-2.

### 7. Serving

- **Nightly batch** (the Technical Design's "batch precompute nightly" for forecasts): fit each user's `ForecastState` and write it to a forecasts file, one per model version and dataset, like FR-3's predictions files. In the demo, it's built into the bundle by `sfc-web build-demo`.
- **Per request:** load the state, simulate the user's paths once with a seed derived from (user, `as_of` month, model version), and run each goal over them. Every goal of the user shares one future, and a draft gets the same numbers as the same goal once saved (#52). 1,000 paths × 120 months is a few milliseconds with NumPy, fast enough for the live setup check. Every answer for the same goal and model version is identical (NFR-8).
- **Saved goals and drafts go through the same function,** so a goal's numbers in `check_goal` before saving match `forecast_goal` after.
- **FR-10's event log feeds `your_entries`:** the goal store's revisions already hold every saved amount with its date. A new read gives the forecast the first and latest entries.

### 8. The coach and the pages

- **Coach:** the prompt says to:
  - quote `forecast_goal`'s band, range and top-up, never compute them;
  - name the uncertainty ("could go either way");
  - say when a short history limits the forecast, or when the share is a typical one because the goal is new;
  - for a reached goal, say it's reached, and mention the drawdown note only when `may_draw_down` is true.
- **Pages:**
  - goal detail (1g, 1l): a chart of saved-so-far plus the median and the 80% band by month, the status, the range, the top-up, and the assumption line;
  - goal cards and the overview card: the status replaces "Forecast soon";
  - the setup box gains the fit badge.

  All of it is rendered server-side from the tools, as FR-10's pages are.

## Metrics and why

- **Brier score** is the primary metric because the product's answer is a probability ("am I on track?"), and Brier rewards both calibration and sharpness. It's measured on both paths, because the new-goal path is what every user-created goal and the fit badge use.
- **Calibration per band** is gated separately, because the label the user sees is the band.
- **Coverage of the 80% range** is checked against realized balances, because NFR-7's range has to be trustworthy on its own.
- **RMSE of net savings** is kept because the PRD names it, and scaled RMSE because dollar RMSE is mostly the highest earners. Feasibility shows it's a weak lever, so it gates for non-inferiority, not for selection.

## Options considered

### A. What to forecast

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Each goal's balance directly | Simple | One series per goal with 1–3 observations; can't serve drafts |
| **(b) The user's net savings, then each goal through its share (recommended)** | One forecast per user serves every goal and draft; the seasonality and volatility live in the right place | Depends on the share model (§3) |
| (c) Income and spending separately | Explains why | Twice the models; the goal only needs the difference |

### B. Point forecast or paths

| Option | Pros | Cons |
| --- | --- | --- |
| (a) A point forecast with a normal interval | Cheap | The balance floor makes outcomes path-dependent; feasibility: 0.268 for the same point forecast against 0.171 with paths |
| **(b) Simulated paths (recommended)** | Handles the floor, skewed incomes and the probability in one mechanism | Needs a seed and care to stay reproducible |

### C. What's precomputed

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Nothing: fit per request | Always fresh | Slow for the live setup check |
| (b) Paths per user | Fastest reads | Hundreds of KB per user; a draft with a new horizon needs new paths anyway |
| **(c) A small forecast state per user, simulated per request (recommended)** | Small; fast; reproducible with a fixed seed; serves drafts | Simulation cost per request (milliseconds) |

### D. The share for a goal without a track record

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) The typical total allocation ÷ the number of active goals (recommended)** | No new field; doesn't make a stretch target look more achievable; easy to explain | An assumption the user can't see or change (the copy says "typical") |
| (b) The typical total, split by what each goal needs per month (FR-10 decision 2's starting point) | Intuitive | Gives the biggest share to the most ambitious goal, so a stretch target looks more achievable than it is |
| (c) Ask the user for a monthly contribution | Exact | A field FR-10 decided against; most people don't know it |
| (d) Assume all net savings go to the goal | Simple | Badly optimistic with several goals |

## Testing

- **Unit:**
  - the balance recursion matches FR-1's `_saved` on random inputs;
  - share inference round-trips (simulate with a share, infer it back), from $0 and from a first entry;
  - an FR-10 goal with one entry, or entries under 3 months apart, stays on the prior;
  - every model falls back correctly on short histories (1, 5, 11 and 13 months);
  - a `typical` share is 0.662 ÷ active goals, a draft included, ended goals not;
  - when shares would total more than 1, only the `typical` ones are scaled down;
  - `extra_per_month` brings `p_goal_met` to 0.7 and is null when already on track;
  - reached goals return `reached` with `may_draw_down` set from the paths;
  - paths are identical for the same seed.
- **Leak check:** an evaluation example can't see a transaction after its `as_of_date`. The splitter refuses it, and a test proves the refusal.
- **Contract:** `forecast_goal` and `check_goal` shapes; the same numbers in-process and over MCP; a draft and the same goal after saving give identical numbers.
- **Stage 9 unchanged:** the refactored sampler reproduces today's goals exactly for the dataset's seed.
- **Pages:** goal detail with each status, reached with and without the drawdown note, and a short history; the badge on the setup check; the overview card's status.
- **Coach suite** (when it exists): "am I on track?" quotes the band, range and top-up from `forecast_goal`.

## Milestones

One PR each.

1. **Contracts and the evaluation task:**
   - the forecaster and goal-forecast contracts;
   - monthly net savings with the cutoff;
   - the balance recursion and share inference from $0 and from a first entry;
   - the stage-9 sampler and evaluation goals;
   - the task with both paths and its leak check;
   - the user bootstrap;
   - the baselines (naive pace, flat 50%) scored on validation.
2. **Candidates and the round:**
   - flat level, seasonal persona-shrunk and (if decided) exponential smoothing;
   - the spread tuned on coverage;
   - scored on validation per path and per persona, with a round results report.
3. **Promotion and serving:**
   - test scoring for finalists, the gates, promotion;
   - the forecasts file in the bundle and the nightly batch;
   - `forecast_goal` and `check_goal` live, with `your_entries` from FR-10's event log;
   - `list_goals` status;
   - the coach prompt.
4. **Pages:** goal detail (1g, 1l), statuses on cards and the overview, reached goals, the fit badge, the short-history notice and the assumption line; a browser check at desktop and phone width.
5. **Docs:** the Technical Design (the forecasting decision, the contracts), the PRD (the measured metric and any target change), the Web App UI (1g/1l built, gap 4 closed); status to Implemented.

## Status and handoff (Oct 4, 2026)

### Where it stands

| Milestone | PR | State |
| --- | --- | --- |
| Design | #50 | Merged, Accepted. Amended in #52 (§2, §6, §7) |
| 1. Contracts, the task, the sampler, baselines | #51 | Merged |
| 2. Candidates and the validation round | #52 | Approved. Rebased on `main` with #51; merges next |
| 3. Test scoring, promotion, serving | — | Not started |
| 4. Pages | — | Not started |
| 5. Docs | — | Not started |

**The validation round's result** (round 4, `docs/reports/FR-11 and FR-12 Goal Forecasting — Round Results.md`):
- **Rank 1:** `paths_seasonal_persona`, Brier 0.196 (track) and 0.229 (new). It's ahead of `paths_ets` and `paths_flat_level` on the paired user bootstrap.
- **Fitted values:** typical total 0.686, spread 1.039.
- **Gates:** it passes every gate on validation; freelancers on new goals score 0.248.
- **No test user has been scored.**

### Reproducing the round

- **Data:** `uv run sfc-data generate --spec configs/data/default.yaml --out <path>/default.sqlite --force`, about 80 s. It should give content hash `b4d43bf4…`, the same as `main`'s generator before #51; a test pins the stage-9 goals.
- **Data and split hashes:** the task's are `a650268fd2a0` and `6f2fc4ed8b7c`, at 2,004 validation goals per path.
- **Tracking:** MLflow needs a database URI, e.g. `--tracking-uri sqlite:///<path>/mlflow.db`, because the file store is refused.
- **Rerun the round from `main` once #52 has merged:** `uv run sfc-experiment run configs/experiments/goal_forecasting --data <path>/default.sqlite --tracking-uri sqlite:///<path>/mlflow.db`, about 2.5 minutes. `finalize` refuses runs from mixed code versions, so all five must come from one commit.
- **`statsmodels`** (the ETS candidate) is in the `train` dependency group, so a plain `uv sync` includes it.

### Milestone 3, step by step

1. **Test scoring,** once: `uv run sfc-experiment finalize --task goal_forecasting --data … --tracking-uri …` scores the top three and the baselines on test users. Read the gate output in §5's order. The freelancer Brier gate on new goals is non-blocking (decision 6), and the CLI marks it so.
2. **Promotion:** run `promote` for `goal_forecasting` (see `evaluation/promote.py` and how FR-7's #39 did it). It records the gates, with `blocking`, in the promotion log. Record the test results in the round report's "test results" section, as FR-7 did.
3. **Serving:**
   - Fit the promoted `PathsModel` on all users' rows and serialize its learned values: persona priors, typical total, spread.
   - Build each user's `ForecastState` at the dataset's `as_of` into the demo bundle (`sfc-web build-demo`), like FR-3's predictions file and FR-7's flag file.
   - Paths are simulated per request with `paths_for(user_id, as_of, state)`.
4. **Tools** (`access/tools.py`):
   - `forecast_goal` returns the §"Tools" fields. **For a reached goal it hides p, the range and the top-up,** and adds the drawdown note when `may_draw_down`; test this at the tool level (review on #51).
   - `check_goal` adds the forecast and the `fit` badge for a valid draft. A draft must give the saved goal's numbers: one future per user, `active_goals` counting the draft.
   - `list_goals` adds `status` and `p_goal_met`.
   - Goals from FR-10's store need their input row built from `goal_revisions`: `first_saved`/`first_saved_as_of` from the earliest live entry, `origin`, `active_goals`, `set_goals`.
5. **Coach:** the §8 prompt rules.
6. **Tests:** contract tests in-process and over MCP; a draft equal to the saved goal; reached-goal hiding; a short-history notice under 6 months.

### Practical notes

- **Reviews:** before saying a PR is up to date, list every review and inline comment with no time filter. Re-reviews arrive often, and filtering by time hid two of them in this session.
- **Stacked PRs:** keep each branch current by merging `main` (or the branch below), not by rebasing. GitHub retargets the next PR when its base branch merges.
- **Tolerances:** the gates read the run's own validation intervals (`val_*_lo/_hi`) as tolerances. Don't recompute them from test.
- **The CLI:** the experiment runner prints a run's metrics. `sfc-experiment report --out` writes the comparison table, and `fit.*` metrics carry the spread and typical total.

## Decisions and open questions

All decided by the owner on Oct 4, 2026, on #50.

1. [x] **The RMSE target: a change, recorded before test.** The Technical Design defines the baseline as seasonal-naive, and nothing beats it by 15%: the best is 2% at 6 months, and the true level in hindsight is 17–19%. So:
   - RMSE gates at 15% below last-month naive (the PRD's word "naive"), and no worse than seasonal-naive;
   - Brier and calibration are the primary gates;
   - the ranking uses scaled RMSE;
   - the PRD keeps the seasonal-naive numbers next to the new target.

   Recorded now, before any test scoring, as FR-4 did on #27 (0.70 to 0.66).
2. [x] **A new goal's share: option D(a), on product grounds.** It's the typical total allocation divided by the number of active goals, capped so a user's shares never total more than 1, which is the rule feasibility measured (§3). It switches to `your_entries` after two saved entries at least 3 months apart. It doesn't make a stretch target look more achievable, and it's easy to explain. The feasibility numbers (0.208 against 0.228) can't choose between the rules for real users, because the generated data favors the equal split by construction. This replaces FR-10 decision 2's starting point (split by need).
3. [x] **Exponential smoothing joins the round** (adding `statsmodels`), if it fits in milestone 2's time, and only for users with 24 or more months of history.
4. [x] **More evaluation goals** from the refactored stage-9 sampler (§6), without changing the dataset.
5. [x] **Status bands, fixed:** "On track" at 0.7 or more, "Off track" below 0.3, "Could go either way" in between, with the setup badge's "Within reach" / "Could go either way" / "A stretch" on the same bands. Validation checks each band; nothing tunes them ([why](#why-nothing-is-tuned-on-outcomes)).
6. [x] **Gates as proposed:** Brier below naive and below a flat 50%, and calibration inside each band, for both paths, overall and per persona; 80% coverage within 70–90%. If the freelancer gate on the new-goal path fails (0.251 in feasibility), it's **reported, not loosened**, and it doesn't block FR-11's promotion on its own.
7. [x] **What-if forecasts are deferred** to a follow-up (Web App UI, gap 5).
8. [x] **A bad month draws a goal down** in proportion to its share (FR-1's rule), as a v1 assumption tied to notional goals; revisit when v2 links real accounts.
9. [x] **Reached goals show "Reached",** with no on-track status or probability, and a drawdown note only when the paths show a real chance of dropping below the target by the date.
