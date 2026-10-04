# FR-11 and FR-12 Goal Forecasting — Round Results

Oct 4, 2026 · @Sidd · Milestone 2 of FR-11 and FR-12 Goal Forecasting — Feature Design (test results and promotion follow in milestone 3)

## Summary

- **This is the round as rerun after review on #52,** with the three fixes that bring the code to the design:
  - one future per user, the same paths for all of a user's goals (§7);
  - exponential smoothing simulated from its fitted model (§4);
  - the spread tuned on realized goal balances (§2).

  The first round's numbers moved by at most 0.004 in Brier. [History](#round-history) has them.
- **Five runs on validation:** the two baselines (`naive_pace`, `flat_50`) and the three §4 candidates, all built on simulated net-savings paths:
  - `paths_flat_level`: a flat 24-month level;
  - `paths_seasonal_persona`: the level plus a month-of-year profile shrunk toward the persona's;
  - `paths_ets`: exponential smoothing for users with 24+ months (owner decision 3).
- **Rank 1: `paths_seasonal_persona`, Brier 0.201 on goals with a track record and 0.230 on new goals** (mean 0.215, user bootstrap 95% 0.20–0.23).
  - **`paths_ets` is tied with it** (paired difference −0.010 to +0.001). Both have zero calibration error, so the next tie-breaker decides: scaled RMSE, where the seasonal model leads (0.913 of seasonal-naive against 0.959).
  - **`paths_flat_level` is behind** (−0.013 to −0.003).
- **Every candidate beats both baselines on both paths:** naive pace scores 0.344 and 0.467, a flat 50% scores 0.250.
- **On validation, rank 1 passes every gate but one,** with these values:
  - every status band's met rate falls inside its band, on both paths;
  - coverage is 0.78 on both paths (gate: 70–90%);
  - RMSE is 0.24 of last-month naive (gate ≤ 0.85) and 0.82 of seasonal-naive (gate ≤ 1.0).
  - The exception is **freelancers on the new-goal path, at 0.250**, level with a flat 50%. Feasibility predicted this, and decision 6 makes it reported and non-blocking.
- **No test user was scored by this milestone.** Test results come with milestone 3.

## Setup

- **Data:** the default dataset generated from `main` (360 users, 36 months, schema 4).
- **Splits:** 240 train users in 5 user-grouped folds, stratified by persona. Data hash `7261111db5a6`, split hash `6f2fc4ed8b7c`.
- **Code:** commit `1fee571` (code version `7ed3dad9`).
- **Examples:** every known-outcome goal at its `as_of_date`, from the dataset (153 goals) plus 10 stage-9 sampler draws per user, **2,004 goals per path.** Each goal is scored as `track` (its own history) and as `new` (created at `as_of`, on the prior share).
- **Fitted inside each fold from its training users, never from outcomes,** and logged as `fit.*` for the final model on all train users:
  - each persona's seasonal profile and pooled deviations;
  - the typical allocation: 0.655, against the POC's 0.662;
  - the spread, tuned by bisection so the 80% range covers 80% of **realized goal balances**: 973 training goals whose target month lies inside their user's visible history, each run on its own share. The fitted spreads were 1.039 (seasonal), 1.109 (exponential smoothing) and 0.943 (flat level).
- **Paths:** 1,000 per user and `as_of`, over 120 months, with a seed from (user, `as_of` month, model version). Every goal of the user runs over the same paths.
- **RMSE** is scored on goal origins: once per user and `as_of`, the next 6 months, against last-month naive and seasonal-naive from the same history.
- **Time:** about 2.5 minutes for all five runs on a laptop CPU. The paths models take about 4.5 s per 2,000-goal batch, and exponential smoothing 10 s.

## Results

Ranked by `val_neg_brier`, minus the mean of the two paths' Brier. Ties are judged on the 95% paired user bootstrap.

| Rank | Run | Brier, track | Brier, new | Mean (95% CI) | vs leader | Band error | Coverage | RMSE vs naive | vs seasonal-naive | Scaled vs seasonal-naive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `paths_seasonal_persona` | **0.201** | **0.230** | 0.215 (0.20–0.23) | – | 0.000 | 0.78 | 0.24 | 0.82 | 0.91 |
| 2 | `paths_ets` (tied) | 0.206 | 0.234 | 0.220 (0.21–0.23) | −0.010 to +0.001 | 0.000 | 0.76 | 0.26 | 0.88 | 0.96 |
| 3 | `paths_flat_level` | 0.212 | 0.235 | 0.223 (0.21–0.24) | −0.013 to −0.003 | 0.000 | 0.78 | 0.28 | 0.93 | 0.99 |
| — | `naive_pace` (baseline) | 0.344 | 0.467 | 0.405 | | 0.190 | – | – | – | – |
| — | `flat_50` (baseline) | 0.250 | 0.250 | 0.250 | | 0.000 | – | – | – | – |

By persona (FR-12):

| Run | Track: family | freelancer | young professional | New: family | freelancer | young professional |
| --- | --- | --- | --- | --- | --- | --- |
| `paths_seasonal_persona` | 0.208 | 0.236 | 0.161 | 0.236 | **0.250** | 0.203 |
| `paths_ets` | 0.207 | 0.253 | 0.160 | 0.238 | 0.266 | 0.198 |
| `paths_flat_level` | 0.220 | 0.247 | 0.169 | 0.239 | 0.259 | 0.206 |

## Reading it

- **Seasonality helps every persona a little.** Against the flat level, the seasonal model is better for families (0.208 against 0.220 on the track path), for freelancers (0.236 against 0.247) and for young professionals (0.161 against 0.169). The freelancer gain fits their billing seasonality (a slow January, a strong December).
- **Exponential smoothing doesn't beat the simpler seasonal model.** It's level for families and young professionals but worse for freelancers (0.253 and 0.266), whose 2–3 years of lumpy months give it little to fit. It also takes twice the time.
- **Tuning the spread on realized balances raised coverage** from 0.76 to 0.78 for the seasonal model, and from 0.75 to 0.76 for exponential smoothing, closer to the 0.80 aim. The remaining shortfall is out-of-fold: the spread is tuned on the training folds' realized goals and scored on other users.
- **RMSE on goal origins** puts the seasonal model 18% ahead of seasonal-naive in dollars (9% scaled). This round measures only the 6-month horizon, which is what decision 1's gate uses. The design is updated (#50) to say the gate runs on goal origins rather than the §6 rolling backtest. A rolling-origin check in review gave 0.86 (dollars) and 0.93 (scaled), so goal origins don't flatter the gate.

## Round history

**First round** (commit `797e0d3`), before the review fixes. Paths were seeded per goal, exponential smoothing resampled residuals around a fixed forecast, and the spread was tuned on net-savings totals inside the histories.

| Run | Brier, track | Brier, new | Coverage |
| --- | --- | --- | --- |
| `paths_seasonal_persona` | 0.203 | 0.231 | 0.76 |
| `paths_ets` | 0.207 | 0.236 | 0.75 |
| `paths_flat_level` | 0.208 | 0.231 | 0.81 |

The ranking and the gate outcomes didn't change. The flat level lost its tie with the leader, because its spread tuned on goal balances (0.94) came out lower than on net-savings totals.

## Next (milestone 3)

- `finalize` scores the top three on test users once.
- `promote` checks the gates in the design's order. The freelancer new-goal gate is reported and non-blocking (decision 6).
- Serving: the nightly forecast states in the bundle, `forecast_goal` and `check_goal` live, and the coach prompt.
