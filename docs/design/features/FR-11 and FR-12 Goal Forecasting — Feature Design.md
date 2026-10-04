# FR-11 and FR-12 Goal Forecasting — Feature Design

Oct 3, 2026 · @Sidd · Status: **Proposed** · Branch: `docs/fr-11-12-design`

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
- **Feasibility, measured on train users only** ([evidence](#feasibility)):
  - **The on-track call works.** Inferring each goal's pace from its history and simulating future months scores a Brier of **0.173**, against **0.320** for the naive pace call and **0.250** for a flat 50% guess. The 80% range covers the actual outcome 76% of the time.
  - **Freelancers are the hard case:** 0.244, barely better than the flat guess. That's FR-12's question.
  - **The PRD's RMSE target depends entirely on what "naive" means.** Against last month's value, a 12-month mean is 71% better. Against the Technical Design's seasonal-naive baseline, the best model is 2% better at 6 months. Even knowing the true level in hindsight would only make it 19% better. Net savings are dominated by one-off purchases that no model can see coming. [Decision 1](#decisions-and-open-questions).
- **Approach:**
  1. **Forecast each user's monthly net savings as simulated paths.** Each user gets a level, a seasonal profile shrunk toward their persona's, and their own month-to-month variation.
  2. **Infer each goal's pace** from its balance and the net savings since it was created. A goal with no track record gets a typical share.
  3. **Any goal is arithmetic over the paths.** Saved goals and unsaved drafts (`check_goal`) get the same treatment: the chance of making it, the median and the 80% range, the gap, and the extra per month that would close it.
  4. **Nightly, store each user's forecast state, not the paths:** level, seasonal profile and residuals, a few hundred numbers. Simulate per request with a fixed seed, so every answer is reproducible and fast enough for the live setup check.
  5. **A round of candidates through the FR-3 framework,** ranked on validation Brier, with per-persona gates for FR-12.
- **Seven decisions for the owner,** listed in [Decisions and open questions](#decisions-and-open-questions). The two biggest are the RMSE baseline and how a new goal's share of savings is set.
- **Principle (carried from FR-2):** the modeled behavior isn't tuned to make the target pass, and test users are scored once, for finalists.

## Context

### How goals and their outcomes are generated (FR-1, stage 9)

- A goal's balance grows by a fixed, hidden **share** of the user's monthly net savings from `created_date`. The balance is floored at zero each month, so a bad month can empty it, and it stays empty until net savings turn positive.
- Each user's shares add up to 0.5–0.9 of net savings, split across their 1–2 goals.
- **The outcome is planted by the target.** For goals that end inside the history, the generator computes the balance actually reached by the target date. It then sets the target at 0.6–0.85× that (`on_track`, always met), 0.95–1.05× (`borderline`, met about half the time) or 1.3–1.8× (`off_track`, never met).
- `as_of_date` is 3–12 months before the target, and only transactions up to it may be used (Technical Design, evaluation controls).
- The default dataset has **36 months** of history (Oct 2023 – Sep 2026), so up to three seasonal cycles.

| Split | Goals | Outcome known | Of which `borderline` |
| --- | --- | --- | --- |
| Train users (240) | 358 | 153 | 44 |
| Test users (120) | 187 | 95 | not opened |

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
  - FR-11 decides how savings split across goals (FR-10 decision 2).

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

All numbers are from train users only, on the default dataset generated at `main` `9bd5097` (content hash `b4d43bf4`). The script is on branch `poc/fr-11-12-feasibility` (`poc/fr11_12_feasibility.py`). Test users haven't been scored.

### Forecasting monthly net savings

A rolling-origin backtest: from every month with 12 or more months of history, forecast total net savings over the next 3, 6 and 12 months.

| Model | RMSE, 3 months | 6 months | 12 months |
| --- | --- | --- | --- |
| Last month repeated (naive) | $12,261 | $23,196 | $41,475 |
| Same month last year (seasonal-naive, the Technical Design's baseline) | $5,280 | $6,685 | $9,268 |
| Mean of the last 12 months | $4,950 | $6,734 | $9,268 |
| Mean of the last 24 months | $4,878 | $6,573 | $8,392 |
| Level + seasonal profile, shrunk toward the persona's | $4,827 | $6,878 | $11,155 |
| *Hindsight: the user's true 36-month level (not a model)* | *$4,451* | *$5,428* | *$5,260* |

- **Any sensible level beats last month's value by about 70%** (6 months: $6,573–6,878 against $23,196). One month is a poor estimate of anything.
- **Nothing beats seasonal-naive or a trailing mean by much.** The best model is 2% better at 6 months and 9% better at 12. Seasonal profiles help only at 3 months, and only for families (7% better than the 12-month mean); further out, their estimation noise costs more than they gain.
- **The ceiling is low.** Even the true level, known in hindsight, is only 19% better than the 12-month mean at 6 months. The rest is one-off purchases and income noise that no forecast of the level can remove.

By persona, RMSE relative to the 12-month mean (lower is better):

| Persona | Best at 3 months | Best at 12 months | Hindsight level at 6 months |
| --- | --- | --- | --- |
| Family budgeter | 0.93 (seasonal profile, persona-shrunk) | 1.00 (seasonal-naive) | 0.85 |
| Freelancer | 0.97 (24-month mean) | 0.84 (24-month mean) | 0.80 |
| Young professional | 0.91 (seasonal-naive) | 1.00 (seasonal-naive) | 0.82 |

### The on-track call

For the 153 train goals with a known outcome, at each goal's `as_of_date`:

- **Infer the goal's share:** the share at which the generator's rule reproduces the balance from the net savings since `created_date` (a bisection; the rule is monotone in the share). A goal with a $0 balance has no usable track record (its share would come out as 0), so it gets the median inferred share instead, 0.42.
- **Simulate:** 400 paths of the coming months. Each path is a point forecast (a 24-month level, plus the persona-shrunk seasonal profile once 12 months exist), with the user's own monthly deviations resampled. The balance is applied month by month with the floor, and P(met) is the share of paths that reach the target.

| Call | Brier (lower is better) | Accuracy |
| --- | --- | --- |
| Naive pace: balance ÷ months so far, extended to the target date | 0.320 | 0.68 |
| A flat 50% for every goal | 0.250 | — |
| Inferred share, 12-month mean, no simulation | 0.301 | 0.70 |
| **Inferred share, simulated paths** | **0.173** (bootstrap 90%: 0.143–0.206) | **0.75** |
| *Oracle: inferred share with the actual future months* | *0.020* | *0.98* |

- **Inferring the share works:** with the actual future, it reproduces 98% of outcomes. The remaining error is all in forecasting the future months.
- **Simulation is what makes the call useful.** A single point forecast is barely better than the naive pace, but the distribution of paths gives a probability that reflects how volatile the user is.
- **By class:** `on_track` 0.183, `off_track` 0.086, `borderline` 0.273. Borderline goals are coin flips by construction, so the best possible Brier on them is about 0.25.
- **By persona** (FR-12): family 0.151, young professional 0.129, **freelancer 0.244**. Freelancer income swings make their 6–12-month outcomes genuinely uncertain. The question is whether the probability says so honestly (calibration), not whether it's sharp.
- **The 80% range** contains the outcome for 76% of goals, close to nominal; it can be widened in tuning.
- **Two bugs the POC found that the design must handle:**
  - *Short histories:* goals created early in a user's history have only a few months at their origin, so every model must fall back to the history it has.
  - *$0 balances:* a goal with no balance yet has no track record, so it needs a prior share, not 0.

### How precise the evaluation can be

- 153 train goals give the Brier a standard error of about 0.020. Test users have 95 goals, about 30 per persona, so a per-persona Brier on test would carry a standard error of about 0.04.
- That's too noisy to gate FR-12 per persona, or to tell candidates 0.02 apart. [Decision 4](#decisions-and-open-questions) proposes more evaluation goals.

## Goals and non-goals

**Goals**

1. For every active goal, saved or a draft: the chance of making it, the likely balance by the target date with an 80% range, the gap, and the extra per month that would close it. Every number comes from the forecast service, none from the LLM (FR-14).
2. Better calibrated than naive, overall and for each persona (FR-12), on test users.
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
| **Am I on track?** | Goal card, goal detail (1g, 1l), overview card | A status ("On track", "Could go either way", "Off track"), "likely $2,780 by Dec 31, between $2,310 and $3,190", and when short, "about $220 short; setting aside $75 more a month would close it" |
| **Set up a goal** | 1h "How it fits" | FR-10's facts, plus the fit badge from the draft's probability: "Within reach", "Could go either way", "A stretch" |
| **Ask the coach** | Chat | "Am I going to hit my vacation goal?" Wren calls `forecast_goal` and states the probability band, the range and the gap, quoting the numbers |
| **Ended goals** | Goals page "Ended" | Unchanged from FR-10. The outcome isn't known, so there's no forecast |
| **Short history** | Everywhere a forecast shows | With fewer than 6 months of history: "Based on only N months of your history, so this is a rough guide", and the range widens |

### Tools

| Tool | Change |
| --- | --- |
| `forecast_goal` (`goal_id`) | Returns `status` (`on_track`, `either_way`, `off_track`), `p_goal_met`, `projected_balance` (the median), `range` (10th and 90th percentiles; an 80% interval), `gap` (target − median, never below 0), `extra_per_month`, `monthly` (median and range by month for the chart), `months_of_history`, `short_history`, `model_version` |
| `check_goal` | Adds the same forecast fields for a valid draft, and the fit badge's `fit` (`within_reach`, `either_way`, `stretch`) |
| `list_goals` | Each active goal gains `status` and `p_goal_met`. `forecast` becomes `available` |

## Design

### 1. Contracts

- **Net-savings forecaster** (`intelligence/forecasting`, the "Forecasting service" in the Technical Design):
  - `fit_state(user_id, monthly_net, as_of) → ForecastState`: a level, a seasonal profile, a residual pool, the number of months of history and the model version. It's a small record, computed nightly.
  - `simulate(state, months, n_paths, seed) → paths[n_paths × months]`: deterministic for a given seed.
- **Goal forecast** (`forecast_goal(user_id, goal_id)` in the Technical Design, unchanged in shape): built from the paths, the goal's balance and its share (§3). Checked against its contract at runtime, like every service (FR-3).
- **Interval: 80%,** the 10th to 90th percentile, fixed in the contract (Web App UI, gap 4).

### 2. The net-savings forecast (FR-12 lives here)

- **Input:** monthly net savings, the sum of a month's amounts. Categories never matter, so FR-6 corrections can't change a forecast (FR-10 §"The setup check"). Months up to `as_of`; the first month counts only if it starts on the 1st (FR-10's rule).
- **Level:** the mean of the last 24 months, or of the history available if shorter. The 24-month mean was the best single level in feasibility, and it's the freelancer's best model by a wide margin (0.84 of the 12-month mean at 12 months).
- **Seasonality:** a month-of-year profile, used once 12 months exist. Each user's own deviation is shrunk toward their persona's median profile, by the number of years observed. That way two noisy Augusts don't become the forecast, and families keep the school-year shape. The round decides whether seasonality earns its place at each horizon (§4).
- **The user's own behavior:** the residual pool is that user's own monthly deviations from the point forecast. A freelancer's paths swing as much as their past did; a salaried user's are tight. With fewer than 6 months of residuals, the pool is pooled with the persona's, scaled to the user's level.
- **Paths:** the point forecast plus resampled residuals, 1,000 paths by default.

### 3. From paths to a goal

- **The balance recursion** (FR-1's rule): balance = max(0, balance + share × net), month by month from `saved_as_of` to the target month.
- **The goal's share:**
  - *A goal with a track record* (a balance above $0 and at least 3 months since creation): inferred as in feasibility, capped at 1.
  - *A goal without one* (just created, or $0 saved), which is every goal created through FR-10: a **prior** ([decision 2](#decisions-and-open-questions)). The user's other goals with a track record keep their shares. Whatever is left of a typical total allocation is split across the new goals in proportion to what each needs per month (FR-10 decision 2's default).
  - *Bounds:* a user's shares never total more than 1.
- **Results from the paths:**
  - `p_goal_met`: the share of paths at or above the target;
  - the median and the 10th and 90th percentiles of the final balance;
  - `gap` = max(0, target − median);
  - `extra_per_month`: the smallest whole-dollar monthly top-up, added to every path, that brings the median to the target. It's found by bisection over the same paths, so it's consistent with the forecast and isn't LLM arithmetic.
- **Status and fit,** from `p_goal_met`: at 0.7 or above, "On track" / "Within reach"; below 0.3, "Off track" / "A stretch"; in between, "Could go either way". The thresholds are tuned on validation so each band is calibrated ([decision 5](#decisions-and-open-questions)).

### 4. Candidates

| Candidate | Level | Seasonality | Paths |
| --- | --- | --- | --- |
| **Naive pace** (baseline) | Balance ÷ months so far, extended | — | None: a 0/1 call |
| **Flat 50%** (reference) | — | — | — |
| **Flat level** | 24-month mean | None | Own residuals |
| **Seasonal, persona-shrunk** | 24-month mean | User profile shrunk toward the persona's | Own residuals |
| **Exponential smoothing** | ETS with additive seasonality | ETS | Simulated from the fitted model |

- All candidates share §3. Only the net-savings forecast differs.
- Exponential smoothing needs `statsmodels`, a new dependency. It joins the round only if it's cheap to add; the Technical Design lists it, and theory warns it needs two full cycles, which only users with 24+ months have.
- **Ranking:** validation Brier on goals, cross-fitted over train users. Tie-breakers, in order: calibration (expected calibration error over the status bands), then RMSE of 6-month net savings, then cost. This is the same tie set and twin logic as FR-4 §5. It's written down before the round.

### 5. Gates (on test users, once, for finalists)

| Gate | Target | Why |
| --- | --- | --- |
| Brier on goals | Below the naive pace call and below a flat 50% | PRD "better calibrated than naive"; a flat 50% is the honest floor |
| Brier per persona (FR-12) | Below a flat 50% for each persona | Seasonality and irregular income must each be handled, not averaged away |
| 80% range coverage | 70–90% of outcomes inside | NFR-7: the range must mean what it says |
| RMSE, 6-month net savings | [Decision 1](#decisions-and-open-questions) | PRD metric |

The gates are checked in that order by the FR-3 promotion code.

### 6. Evaluation task

- **Examples:** each goal with a known outcome, at its `as_of_date`. Only transactions up to that date are used, enforced by the splitter's leak check (Technical Design, evaluation controls).
- **More goals for evaluation** ([decision 4](#decisions-and-open-questions)): FR-1's stage 9 is refactored into a pure sampler over a user's monthly net savings (the same code and rule). It's run with more draws per user, under its own seed, into an evaluation-only goal set. The generated dataset doesn't change; a test checks that stage 9 still produces exactly today's goals. Today's goals give about 0.64 known-outcome goals per user, so 10 draws per user give about 1,500 for train users and 750 for test users, about 250 per persona on test: a standard error near 0.02, enough for per-persona gates.
- **Net-savings backtest:** rolling origins with at least 12 months of history, horizons of 3, 6 and 12 months, scored by RMSE, overall and per persona.
- **Splits:** train users for fitting and tuning (the persona profiles, the share prior, the thresholds), test users once. This is the same separation as FR-2.

### 7. Serving

- **Nightly batch** (the Technical Design's "batch precompute nightly" for forecasts): fit each user's `ForecastState` and write it to a forecasts file, one per model version and dataset, like FR-3's predictions files. In the demo, it's built into the bundle by `sfc-web build-demo`.
- **Per request:** load the state, simulate with a seed derived from (user, model version), and compute the goal numbers. 1,000 paths × 120 months is a few milliseconds with NumPy, fast enough for the live setup check. Every answer for the same goal and model version is identical (NFR-8).
- **Saved goals and drafts go through the same function,** so a goal's numbers in `check_goal` before saving match `forecast_goal` after.

### 8. The coach and the pages

- **Coach:** the prompt says to quote `forecast_goal`'s band, range and gap, never to compute them; to name the uncertainty ("could go either way"); and to say when a short history limits the forecast. Asked about a goal's chances, Wren gives the band and range, not just the median.
- **Pages:**
  - goal detail (1g, 1l): a chart of saved-so-far plus the median and the 80% band by month, the status, the range, and what would close the gap;
  - goal cards and the overview card: the status replaces "Forecast soon";
  - the setup box gains the fit badge.

  All of it is rendered server-side from the tools, as FR-10's pages are.

## Metrics and why

- **Brier score** is the primary metric because the product's answer is a probability ("am I on track?"), and Brier rewards both calibration and sharpness.
- **Coverage of the 80% range** is checked separately, because NFR-7's range has to be trustworthy on its own.
- **RMSE of net savings** is kept because the PRD names it. Feasibility shows it's a weak lever, so it gates for non-inferiority, not for selection.

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
| (a) A point forecast with a normal interval | Cheap | The balance floor makes outcomes path-dependent; feasibility: Brier 0.301 against 0.173 |
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
| **(a) A prior: what's left of a typical total allocation, split by what each goal needs per month (recommended)** | No new field; consistent with FR-10 decision 2 | An assumption the user can't see or change |
| (b) Ask the user for a monthly contribution | Exact | A field FR-10 decided against; most people don't know it |
| (c) Assume all net savings go to the goal | Simple | Badly optimistic with several goals |

## Testing

- **Unit:**
  - the balance recursion matches FR-1's `_saved` on random inputs;
  - share inference round-trips (simulate with a share, infer it back);
  - every model falls back correctly on short histories (1, 5, 11 and 13 months);
  - a goal with a $0 balance gets the prior;
  - shares never total more than 1;
  - `extra_per_month` brings the median to the target;
  - paths are identical for the same seed.
- **Leak check:** an evaluation example can't see a transaction after its `as_of_date`. The splitter refuses it, and a test proves the refusal.
- **Contract:** `forecast_goal` and `check_goal` shapes; the same numbers in-process and over MCP; a draft and the same goal after saving give identical numbers.
- **Stage 9 unchanged:** the refactored sampler reproduces today's goals exactly for the dataset's seed.
- **Pages:** goal detail with each status and with a short history; the badge on the setup check; the overview card's status.
- **Coach suite** (when it exists): "am I on track?" quotes the band, range and gap from `forecast_goal`.

## Milestones

One PR each.

1. **Contracts and the evaluation task:**
   - the forecaster and goal-forecast contracts;
   - monthly net savings with the cutoff;
   - share inference and the balance recursion;
   - the stage-9 sampler and evaluation goals;
   - the task with its leak check;
   - the baselines (naive pace, flat 50%) scored on validation.
2. **Candidates and the round:** flat level, seasonal persona-shrunk and (if decided) exponential smoothing, scored on validation by persona, with a round results report.
3. **Promotion and serving:**
   - test scoring for finalists, the gates, promotion;
   - the forecasts file in the bundle and the nightly batch;
   - `forecast_goal` and `check_goal` live;
   - `list_goals` status;
   - the coach prompt.
4. **Pages:** goal detail (1g, 1l), statuses on cards and the overview, the fit badge, the short-history notice; a browser check at desktop and phone width.
5. **Docs:** the Technical Design (the forecasting decision, the contracts), the PRD (the measured metric), the Web App UI (1g/1l built, gap 4 closed); status to Implemented.

## Decisions and open questions

1. [ ] **The RMSE baseline.** The PRD asks for RMSE "at least 15% lower than a naive forecast". Against last month's value, every sensible model passes by about 70%. Against seasonal-naive (the Technical Design's baseline), nothing reaches 15%: the best is 2% at 6 months, and even the true level in hindsight gets only 19%. **Recommend:** gate RMSE at 15% below last-month naive (the PRD's wording), require no worse than seasonal-naive, and make the goal Brier the primary gate. Update the PRD's metric text with the measured numbers, as FR-4 did.
2. [ ] **A new goal's share of savings.** Recommend option D(a): what's left of a typical total allocation (fitted on train users' goals with a track record), split by what each goal needs per month. A goal switches to its own inferred share once it has 3 months of history and a balance above $0.
3. [ ] **Exponential smoothing in the round,** which adds `statsmodels` as a dependency. Recommend yes, but only if it fits in milestone 2's time, and only for users with 24+ months.
4. [ ] **More evaluation goals** from the refactored stage-9 sampler (§6), without changing the dataset. The alternatives are living with 95 test goals (a standard error of about 0.04 per persona) or regenerating the dataset with more goals per user (every model's predictions rebuilt). Recommend the sampler.
5. [ ] **Status bands:** "On track" at 70% or more, "Off track" below 30%, "Could go either way" in between, with the setup badge's "Within reach" / "Could go either way" / "A stretch" on the same bands. The thresholds stay fixed; validation checks each band is calibrated.
6. [ ] **Gates:** Brier below naive and below a flat 50% overall and per persona, and 80% coverage within 70–90%. The per-persona gate is FR-12's acceptance. Freelancers sit close to the flat 50% in feasibility (0.244 against 0.250), so this gate may fail. If it does, recommend reporting it, as FR-4 did with its unseen-merchant target, rather than loosening it silently.
7. [ ] **What-if forecasts** (Web App UI, gap 5: "a December like last year's"). Recommend deferring them to a follow-up. The paths make them cheap later (replace one month's draws), but they need their own tool and copy.
