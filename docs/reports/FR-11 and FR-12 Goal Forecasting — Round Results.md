# FR-11 and FR-12 Goal Forecasting — Round Results

Oct 4, 2026 · @Sidd · Milestone 2 of FR-11 and FR-12 Goal Forecasting — Feature Design (test results and promotion follow in milestone 3)

## Summary

- **This is the round as rerun after review on #51 and #52,** with the fixes that bring the code to the design:
  - one future per user, the same paths for all of a user's goals (§7);
  - exponential smoothing simulated from its fitted model (§4);
  - the spread tuned on realized goal balances (§2);
  - `active_goals` counting only goals that existed at `as_of`. It had counted siblings created later, which inflated the typical allocation (0.655 to 0.552).

  [History](#round-history) has the earlier rounds.
- **Five runs on validation:** the two baselines (`naive_pace`, `flat_50`) and the three §4 candidates, all built on simulated net-savings paths:
  - `paths_flat_level`: a flat 24-month level;
  - `paths_seasonal_persona`: the level plus a month-of-year profile shrunk toward the persona's;
  - `paths_ets`: exponential smoothing for users with 24+ months (owner decision 3).
- **Rank 1: `paths_seasonal_persona`, Brier 0.201 on goals with a track record and 0.236 on new goals** (mean 0.218, user bootstrap 95% 0.21–0.23).
  - **`paths_ets` is tied with it** (paired difference −0.010 to +0.000). Both have zero calibration error, so scaled RMSE decides, where the seasonal model leads (0.913 of seasonal-naive against 0.959).
  - **`paths_flat_level` is behind** (−0.012 to −0.002).
- **Every candidate beats the baselines:** naive pace scores 0.344 on the track path (it isn't a floor for new goals; review on #51), and a flat 50% scores 0.250.
- **On validation, rank 1 passes every gate but one,** with these values:
  - every status band's met rate falls inside its band, on both paths;
  - coverage is 0.77–0.78 (gate: 70–90%);
  - RMSE is 0.24 of last-month naive (gate ≤ 0.85) and 0.82 of seasonal-naive (gate ≤ 1.0).
  - The exception is **freelancers on the new-goal path, at 0.251**, just over a flat 50%. Feasibility predicted this, and decision 6 makes it reported and non-blocking.
- **No test user was scored by this milestone.** Test results come with milestone 3.

## Setup

- **Data:** the default dataset generated from `main` (360 users, 36 months, schema 4).
- **Splits:** 240 train users in 5 user-grouped folds, stratified by persona. Data hash `9955683d1553`: the examples changed with `active_goals` and the `monthly_net` histories. Split hash `6f2fc4ed8b7c`.
- **Code:** commit `11ceb21` (code version `8a5ed3e8`).
- **Examples:** every known-outcome goal at its `as_of_date`, from the dataset (153 goals) plus 10 stage-9 sampler draws per user, **2,004 goals per path.** Each goal is scored as `track` (its own history) and as `new` (created at `as_of`, on the prior share).
- **Labels:** fits get none; `training_rows` withholds them.
- **Fitted inside each fold from its training users,** and logged as `fit.*` for the final model on all train users:
  - each persona's seasonal profile and pooled deviations;
  - the typical allocation: **0.552**;
  - the spread, tuned by bisection so the 80% range covers 80% of **realized goal balances**: 973 training goals whose target month lies inside their user's visible history, each run on its own share. The fitted spreads were 1.039 (seasonal), 1.109 (exponential smoothing) and 0.943 (flat level).
- **Paths:** 1,000 per user and `as_of`, over 120 months, with a seed from (user, `as_of` month, model version). Every goal of the user runs over the same paths.
- **RMSE** is scored on goal origins: once per user and `as_of`, the next 6 months, against last-month naive and seasonal-naive from the same history.
- **Validation intervals:** each run logs the 95% user-bootstrap intervals of the per-persona Brier and of every band's met rate, overall and per persona. These are the gates' tolerances, fixed before test (review on #51).
- **Time:** about 2.5 minutes for all five runs on a laptop CPU.

## Results

Ranked by `val_neg_brier`, minus the mean of the two paths' Brier. Ties are judged on the 95% paired user bootstrap.

| Rank | Run | Brier, track | Brier, new | Mean (95% CI) | vs leader | Band error | Coverage | RMSE vs naive | vs seasonal-naive | Scaled vs seasonal-naive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `paths_seasonal_persona` | **0.201** | **0.236** | 0.218 (0.21–0.23) | – | 0.000 | 0.77 | 0.24 | 0.82 | 0.91 |
| 2 | `paths_ets` (tied) | 0.206 | 0.242 | 0.224 (0.21–0.24) | −0.010 to +0.000 | 0.000 | 0.76 | 0.26 | 0.88 | 0.96 |
| 3 | `paths_flat_level` | 0.211 | 0.239 | 0.225 (0.21–0.24) | −0.012 to −0.002 | 0.004 | 0.78 | 0.28 | 0.93 | 0.99 |
| — | `naive_pace` (baseline) | 0.344 | 0.467 | 0.405 | | 0.190 | – | – | – | – |
| — | `flat_50` (baseline) | 0.250 | 0.250 | 0.250 | | 0.000 | – | – | – | – |

By persona (FR-12):

| Run | Track: family | freelancer | young professional | New: family | freelancer | young professional |
| --- | --- | --- | --- | --- | --- | --- |
| `paths_seasonal_persona` | 0.205 | 0.235 | 0.162 | 0.238 | **0.251** | 0.220 |
| `paths_ets` | 0.204 | 0.251 | 0.161 | 0.241 | 0.267 | 0.217 |
| `paths_flat_level` | 0.218 | 0.245 | 0.169 | 0.238 | 0.260 | 0.220 |

Calibration of rank 1 by status band:

| Band | Track: goals | Met | New: goals | Met |
| --- | --- | --- | --- | --- |
| Off track (p < 0.3) | 927 | 0.23 | 857 | 0.29 |
| Could go either way (0.3–0.7) | 573 | 0.60 | 555 | 0.53 |
| On track (p ≥ 0.7) | 504 | 0.80 | 592 | 0.71 |

## Reading it

- **Seasonality helps every persona a little on goals with a track record.** Against the flat level: family 0.205 against 0.218, freelancer 0.235 against 0.245, young professional 0.162 against 0.169. The freelancer gain fits their billing seasonality (a slow January, a strong December). On new goals the candidates are close, since the prior share dominates.
- **Exponential smoothing doesn't beat the simpler seasonal model.** It's level for families and young professionals, but worse for freelancers (0.251 and 0.267), whose 2–3 years of lumpy months give it little to fit. It also takes twice the time.
- **Fixing `active_goals` cost the new-goal path 0.006** (0.230 to 0.236), most of it among young professionals (0.203 to 0.220). Counting siblings created later had raised the typical allocation from 0.552 to 0.655, which happened to suit this data. The fixed number is the honest one: a forecast can't know about goals the user hasn't set up yet.
- **New goals sit nearer their bands' edges** than goals with a track record (on-track goals met 0.71 against 0.80). Both are inside the bands.
- **RMSE on goal origins** puts the seasonal model 18% ahead of seasonal-naive in dollars (9% scaled). Only the 6-month horizon is measured, which is what decision 1's gate uses (design §6, amended in #52). A rolling-origin check in review gave 0.86 (dollars) and 0.93 (scaled), so goal origins don't flatter the gate.

## Round history

| Round | Commit | What changed | Seasonal: track / new | ETS | Flat level |
| --- | --- | --- | --- | --- | --- |
| 1 | `797e0d3` | First round: paths seeded per goal, ETS resampling around a fixed forecast, the spread tuned on net-savings totals | 0.203 / 0.231 | 0.207 / 0.236 | 0.208 / 0.231 |
| 2 | `1fee571` | Review on #52: one future per user, ETS from its model, the spread on goal balances | 0.201 / 0.230 | 0.206 / 0.234 | 0.212 / 0.235 |
| 3 | `11ceb21` | Review on #51: `active_goals` without the future, `monthly_net` histories, no labels in fits | 0.201 / 0.236 | 0.206 / 0.242 | 0.211 / 0.239 |

The rank-1 run and the gate outcomes on validation were the same in every round. The flat level lost its tie with the leader in round 2.

## Next (milestone 3)

- `finalize` scores the top three on test users once.
- `promote` checks the gates in the design's order. The freelancer new-goal gate is reported and non-blocking (decision 6).
- Serving: the nightly forecast states in the bundle, `forecast_goal` and `check_goal` live, and the coach prompt.
