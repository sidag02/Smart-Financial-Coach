# FR-11 and FR-12 Goal Forecasting — Round Results

Oct 4, 2026 · @Sidd · Milestone 2 of FR-11 and FR-12 Goal Forecasting — Feature Design (test results and promotion follow in milestone 3)

## Summary

- **This is the round as rerun after reviews on #51 and #52,** with the code matching the design:
  - one future per user (§7);
  - exponential smoothing simulated from its fitted model (§4);
  - the spread tuned on realized goal balances (§2);
  - `active_goals` counting only goals that existed at `as_of`;
  - the typical total allocation estimated as the design defines it: each goal set's shares summed over all its goals, then the median.

  [History](#round-history) has the earlier rounds.
- **Five runs on validation:** the two baselines (`naive_pace`, `flat_50`) and the three §4 candidates, all built on simulated net-savings paths:
  - `paths_flat_level`: a flat 24-month level;
  - `paths_seasonal_persona`: the level plus a month-of-year profile shrunk toward the persona's;
  - `paths_ets`: exponential smoothing for users with 24+ months (owner decision 3).
- **Rank 1: `paths_seasonal_persona`, Brier 0.196 on goals with a track record and 0.229 on new goals** (mean 0.213, user bootstrap 95% 0.20–0.23). It's ahead of both other candidates: the paired intervals are −0.011 to −0.000 against `paths_ets` and −0.013 to −0.002 against `paths_flat_level`.
- **Every candidate beats the baselines:** naive pace scores 0.344 on the track path (it isn't a floor for new goals; review on #51), and a flat 50% scores 0.250.
- **On validation, rank 1 passes every gate,** with these values:
  - the band met rates are inside their bands within tolerance (new goals at "on track" are met 0.70);
  - coverage is 0.77–0.78 (gate: 70–90%);
  - RMSE is 0.24 of last-month naive (gate ≤ 0.85) and 0.82 of seasonal-naive (gate ≤ 1.0);
  - **freelancers on new goals score 0.248**, just under a flat 50%. Decision 6's non-blocking gate is passed here, though feasibility had them at 0.251.
- **No test user was scored by this milestone.** Test results come with milestone 3.

## Setup

- **Data:** the default dataset generated from `main` (360 users, 36 months, schema 4).
- **Splits:** 240 train users in 5 user-grouped folds, stratified by persona. Data hash `a650268fd2a0` (the examples gained `set_goals`), split hash `6f2fc4ed8b7c`.
- **Code:** commit `1dfdab8` (code version `b829cca3`).
- **Examples:** every known-outcome goal at its `as_of_date`, from the dataset (153 goals) plus 10 stage-9 sampler draws per user, **2,004 goals per path.** Each goal is scored as `track` (its own history) and as `new` (created at `as_of`, on the prior share).
- **Labels:** fits get none; `training_rows` withholds them.
- **Fitted inside each fold from its training users,** and logged as `fit.*` for the final model on all train users:
  - each persona's seasonal profile and pooled deviations;
  - **the typical allocation: 0.686.** Per goal set, the measured goals' mean share times every goal in the set (`set_goals`, read only in `fit`), then the median over sets. Feasibility measured 0.662;
  - the spread, tuned by bisection so the 80% range covers 80% of **realized goal balances**: 973 training goals whose target month lies inside their user's visible history, each run on its own share. The fitted spreads were 1.039 (seasonal), 1.109 (exponential smoothing) and 0.943 (flat level).
- **Paths:** 1,000 per user and `as_of`, over 120 months, with a seed from (user, `as_of` month, model version). Every goal of the user runs over the same paths.
- **RMSE** is scored on goal origins: once per user and `as_of`, the next 6 months, against last-month naive and seasonal-naive from the same history.
- **Validation intervals:** each run logs the 95% user-bootstrap intervals of the per-persona Brier and of every band's met rate, overall and per persona. These are the gates' tolerances, fixed before test (review on #51). For example, rank 1's freelancer Brier on new goals has an interval of 0.227–0.271.
- **Time:** about 2.5 minutes for all five runs on a laptop CPU.

## Results

Ranked by `val_neg_brier`, minus the mean of the two paths' Brier. Ties are judged on the 95% paired user bootstrap.

| Rank | Run | Brier, track | Brier, new | Mean (95% CI) | vs leader | Band error | Coverage | RMSE vs naive | vs seasonal-naive | Scaled vs seasonal-naive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `paths_seasonal_persona` | **0.196** | **0.229** | 0.213 (0.20–0.23) | – | 0.004 | 0.77 | 0.24 | 0.82 | 0.91 |
| 2 | `paths_ets` | 0.202 | 0.235 | 0.218 (0.21–0.23) | −0.011 to −0.000 | 0.021 | 0.76 | 0.26 | 0.88 | 0.96 |
| 3 | `paths_flat_level` | 0.207 | 0.232 | 0.220 (0.21–0.23) | −0.013 to −0.002 | 0.025 | 0.78 | 0.28 | 0.93 | 0.99 |
| — | `naive_pace` (baseline) | 0.344 | 0.467 | 0.405 | | 0.190 | – | – | – | – |
| — | `flat_50` (baseline) | 0.250 | 0.250 | 0.250 | | 0.000 | – | – | – | – |

By persona (FR-12):

| Run | Track: family | freelancer | young professional | New: family | freelancer | young professional |
| --- | --- | --- | --- | --- | --- | --- |
| `paths_seasonal_persona` | 0.200 | 0.232 | 0.157 | 0.228 | **0.248** | 0.210 |
| `paths_ets` | 0.200 | 0.249 | 0.156 | 0.233 | 0.266 | 0.206 |
| `paths_flat_level` | 0.212 | 0.244 | 0.165 | 0.225 | 0.260 | 0.212 |

Calibration of rank 1 by status band:

| Band | Track: goals | Met | New: goals | Met |
| --- | --- | --- | --- | --- |
| Off track (p < 0.3) | 892 | 0.23 | 692 | 0.24 |
| Could go either way (0.3–0.7) | 586 | 0.59 | 612 | 0.51 |
| On track (p ≥ 0.7) | 526 | 0.80 | 700 | 0.70 |

## Reading it

- **Seasonality helps every persona on goals with a track record.** Against the flat level: family 0.200 against 0.212, freelancer 0.232 against 0.244, young professional 0.157 against 0.165. The freelancer gain fits their billing seasonality (a slow January, a strong December). On new goals the candidates are closer, since the prior share dominates.
- **Exponential smoothing doesn't beat the simpler seasonal model.** It's level for families and young professionals, but worse for freelancers (0.249 and 0.266), whose 2–3 years of lumpy months give it little to fit. It also takes twice the time.
- **How the typical allocation is estimated mattered more than any model choice for new goals.** Round 3's estimator multiplied each set's mean share by the goals active at `as_of`. Once #51 stopped counting siblings created later, that halved 2-goal sets, and the estimate fell to 0.552 (new goals 0.236). The `active_goals` fix itself is right where it belongs: a new goal's share is divided by the goals that exist at forecast time. It doesn't belong in *estimating* the total from training sets, where every goal's share is measured. Summing over the whole set, as the design says, gives 0.686 and new goals 0.229.
- **New goals at "on track" are met 0.70 of the time,** at the band's edge (band error 0.004), against 0.80 for goals with a track record. A typical share is an assumption, and the band says so honestly.
- **RMSE on goal origins** puts the seasonal model 18% ahead of seasonal-naive in dollars (9% scaled). Only the 6-month horizon is measured, which is what decision 1's gate uses (design §6, amended in #52). A rolling-origin check in review gave 0.86 (dollars) and 0.93 (scaled), so goal origins don't flatter the gate.

## Round history

| Round | Commit | What changed | Seasonal: track / new | ETS | Flat level | Typical total |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `797e0d3` | First round: paths seeded per goal, ETS resampling around a fixed forecast, the spread tuned on net-savings totals | 0.203 / 0.231 | 0.207 / 0.236 | 0.208 / 0.231 | 0.656 |
| 2 | `1fee571` | Review on #52: one future per user, ETS from its model, the spread on goal balances | 0.201 / 0.230 | 0.206 / 0.234 | 0.212 / 0.235 | 0.655 |
| 3 | `11ceb21` | Review on #51: `active_goals` without later siblings, `monthly_net` histories, no labels in fits | 0.201 / 0.236 | 0.206 / 0.242 | 0.211 / 0.239 | 0.552 |
| 4 | `1dfdab8` | Review on #52: the typical total summed over each set's goals (the design's estimator) | **0.196 / 0.229** | 0.202 / 0.235 | 0.207 / 0.232 | 0.686 |

- The rank-1 run was the same in every round.
- The flat level lost its tie with the leader in round 2, and exponential smoothing lost its tie in round 4.
- In rounds 1–3 the freelancer new-goal Brier sat at 0.250–0.253. Round 4 puts it at 0.248.

## Next (milestone 3)

- `finalize` scores the top three on test users once.
- `promote` checks the gates in the design's order. The freelancer new-goal gate is reported and non-blocking (decision 6).
- Serving: the nightly forecast states in the bundle, `forecast_goal` and `check_goal` live, and the coach prompt.
