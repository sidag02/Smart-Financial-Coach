# FR-11 and FR-12 Goal Forecasting — Round Results

Oct 4, 2026 · @Sidd · Milestone 2 of FR-11 and FR-12 Goal Forecasting — Feature Design (test results and promotion follow in milestone 3)

## Summary

- **Five runs on validation:** the two baselines (`naive_pace`, `flat_50`) and the three §4 candidates, all built on simulated net-savings paths:
  - `paths_flat_level`: a flat 24-month level;
  - `paths_seasonal_persona`: the level plus a month-of-year profile shrunk toward the persona's;
  - `paths_ets`: exponential smoothing for users with 24+ months (owner decision 3).
- **Rank 1: `paths_seasonal_persona`, Brier 0.203 on goals with a track record and 0.231 on new goals** (mean 0.217, user bootstrap 95% 0.20–0.23).
  - **All three candidates are tied** with it on the paired user bootstrap.
  - **Every candidate's calibration error is 0,** so the next tie-breaker decides: scaled RMSE of the next 6 months' net savings. The seasonal model leads there (0.913 of seasonal-naive), ahead of exponential smoothing (0.959) and the flat level (0.989).
- **Every candidate beats both baselines on both paths,** by a wide margin: naive pace scores 0.344 and 0.467, a flat 50% scores 0.250.
- **On validation, rank 1 passes every gate but one,** with these values:
  - every status band's met rate falls inside its band, on both paths;
  - coverage is 0.76 on both paths (gate: 70–90%);
  - RMSE is 0.24 of last-month naive (gate ≤ 0.85) and 0.82 of seasonal-naive (gate ≤ 1.0).
  - The exception is **freelancers on the new-goal path, at 0.253** against the flat 50%'s 0.250. Feasibility predicted this, and decision 6 makes it reported and non-blocking.
- **No test user was scored by this milestone.** Test results come with milestone 3.

## Setup

- **Data:** the default dataset generated from `main` (360 users, 36 months, schema 4).
- **Splits:** 240 train users in 5 user-grouped folds, stratified by persona. Data hash `7261111db5a6`, split hash `6f2fc4ed8b7c`.
- **Code:** commit `797e0d3` (code version `2f30e120`).
- **Examples:** every known-outcome goal at its `as_of_date`, from the dataset (153 goals) plus 10 stage-9 sampler draws per user, **2,004 goals per path.** Each goal is scored as `track` (its own history) and as `new` (created at `as_of`, on the prior share).
- **Fitted inside each fold from its training users, never from outcomes:**
  - each persona's seasonal profile and pooled deviations;
  - the typical allocation (0.656 in fold 0; the POC's was 0.662);
  - the spread: 1.025 in fold 0 for the seasonal model, tuned so 80% ranges of the next 3–12 months' net savings cover 80% inside the training histories.
- **Time:** about 7 minutes for all five runs on a laptop CPU. The paths models take about 3.5 s per 2,000-goal batch, and exponential smoothing 8.9 s.

## Results

Ranked by `val_neg_brier`, minus the mean of the two paths' Brier. Ties are judged on the 95% paired user bootstrap.

| Rank | Run | Brier, track | Brier, new | Mean (95% CI) | vs leader | Band error | Coverage | RMSE vs naive | vs seasonal-naive | Scaled vs seasonal-naive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `paths_seasonal_persona` | **0.203** | **0.231** | 0.217 (0.20–0.23) | – | 0.000 | 0.76 | 0.24 | 0.82 | 0.91 |
| 2 | `paths_ets` (tied) | 0.207 | 0.236 | 0.222 (0.21–0.24) | −0.010 to +0.001 | 0.000 | 0.75 | 0.26 | 0.88 | 0.96 |
| 3 | `paths_flat_level` (tied) | 0.208 | 0.231 | 0.220 (0.21–0.23) | −0.008 to +0.002 | 0.000 | 0.81 | 0.28 | 0.93 | 0.99 |
| — | `naive_pace` (baseline) | 0.344 | 0.467 | 0.405 | | 0.190 | – | – | – | – |
| — | `flat_50` (baseline) | 0.250 | 0.250 | 0.250 | | 0.000 | – | – | – | – |

By persona (FR-12):

| Run | Track: family | freelancer | young professional | New: family | freelancer | young professional |
| --- | --- | --- | --- | --- | --- | --- |
| `paths_seasonal_persona` | 0.209 | 0.238 | 0.161 | 0.237 | **0.253** | 0.203 |
| `paths_ets` | 0.207 | 0.255 | 0.160 | 0.240 | 0.268 | 0.199 |
| `paths_flat_level` | 0.214 | 0.243 | 0.167 | 0.234 | 0.256 | 0.203 |

Calibration of rank 1 by status band:

| Band | Track: goals | Met | New: goals | Met |
| --- | --- | --- | --- | --- |
| Off track (p < 0.3) | 962 | 0.25 | 913 | 0.29 |
| Could go either way (0.3–0.7) | 542 | 0.59 | 532 | 0.54 |
| On track (p ≥ 0.7) | 500 | 0.81 | 559 | 0.74 |

## Reading it

- **Seasonality earns its place, narrowly, and only for families.** Family Brier is 0.209 against the flat level's 0.214, and scaled RMSE improves most at short horizons, as in feasibility. For freelancers the seasonal profile helps a little (0.238 against 0.243): their billing seasonality (a slow January, a strong December) is real.
- **Exponential smoothing doesn't beat the simpler models.** It's slightly better for young professionals, worse for freelancers (0.255 and 0.268), whose 2–3 years of lumpy months give it too little to fit. It also costs 2.5× the time.
- **Track-record goals versus the dataset's own goals.** On the dataset's 153 goals alone, rank 1 scores 0.187 on the track path, close to the POC's 0.171 (its interval was ±0.03). The sampled goals are a little harder (0.204), mostly because about a quarter of them have no usable track record (a $0 balance or under 3 months), and those score 0.271 on the typical share.
- **Coverage is 0.76 for the seasonal model and 0.81 for the flat level.** The spread is tuned on net-savings totals inside the histories, not on goal balances, because tuning on goals would read the realized future. The floor on the balance narrows the realized range a little. Both are inside the 70–90% gate.
- **RMSE:** on goal origins, the seasonal model is 18% better than seasonal-naive in dollars (9% scaled). That's better than feasibility's backtest, which used every origin with 12+ months. Either way it clears decision 1's gates.

## Next (milestone 3)

- `finalize` scores the top three on test users once.
- `promote` checks the gates in the design's order. The freelancer new-goal gate is reported and non-blocking (decision 6).
- Serving: the nightly forecast states in the bundle, `forecast_goal` and `check_goal` live, and the coach prompt.
