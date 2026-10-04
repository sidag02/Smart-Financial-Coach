# FR-8 feasibility (train users only)

- Dataset `default.sqlite`; train users: 240; post-warm-up user-months: 7920
- Scored periods: 89760 (ignored: 1426); monthly spike labels: 411 (clear 373)
- Labels per user-month: 0.0519
- Pooled income elasticity of counts (fitted, persona-free): 0.246

## Detectors

| detector | recall@p0.70 | flags@p0.70 | recall@p0.80 | flags@p0.80 | recall@0.02 | precision@0.02 | recall@0.03 | precision@0.03 | recall@0.035 | precision@0.035 | recall@0.05 | precision@0.05 | recall_clear@0.03 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_mean_k_std | 0.000 | 0 | 0.000 | 0 | 0.007 | 0.019 | 0.032 | 0.055 | 0.044 | 0.065 | 0.107 | 0.111 | 0.035 |
| robust_log_mad | 0.000 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spend_ratio_seasonal | 0.000 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| count_poisson | 0.501 | 291 | 0.474 | 240 | 0.367 | 0.956 | 0.470 | 0.811 | 0.489 | 0.726 | 0.562 | 0.583 | 0.517 |
| count_poisson_seasonal | 0.545 | 315 | 0.487 | 250 | 0.358 | 0.930 | 0.472 | 0.815 | 0.509 | 0.755 | 0.579 | 0.601 | 0.520 |
| count_poisson_seasonal_income | 0.562 | 323 | 0.518 | 264 | 0.367 | 0.956 | 0.494 | 0.853 | 0.530 | 0.787 | 0.589 | 0.611 | 0.544 |
| count_poisson_seasonal_income_floor | 0.577 | 337 | 0.545 | 275 | 0.372 | 0.968 | 0.496 | 0.857 | 0.545 | 0.809 | 0.599 | 0.621 | 0.547 |
| ablation: spend floor only | 0.557 | 321 | 0.513 | 262 | 0.367 | 0.956 | 0.491 | 0.849 | 0.528 | 0.783 | 0.584 | 0.606 | 0.542 |
| ablation: min usual count only | 0.582 | 341 | 0.550 | 277 | 0.372 | 0.968 | 0.501 | 0.866 | 0.550 | 0.816 | 0.599 | 0.621 | 0.552 |
| diagnostic: planted categories only | 0.586 | 344 | 0.550 | 276 | 0.375 | 0.975 | 0.501 | 0.866 | 0.550 | 0.816 | 0.608 | 0.631 | 0.552 |
| count_negbin_seasonal_income | 0.489 | 284 | 0.431 | 217 | 0.363 | 0.943 | 0.445 | 0.769 | 0.479 | 0.711 | 0.545 | 0.566 | 0.491 |
| oracle_true_expected_count | 0.623 | 311 | 0.623 | 311 | 0.384 | 1.000 | 0.535 | 0.924 | 0.569 | 0.845 | 0.642 | 0.667 | 0.590 |

## Ablations

| change in recall (points) | recall@p0.70 | recall@p0.80 | recall@0.02 | recall@0.03 | recall@0.035 | recall@0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| seasonal index | 4.380 | 1.217 | -0.973 | 0.243 | 1.946 | 1.703 |
| income coupling | 1.703 | 3.163 | 0.973 | 2.190 | 2.190 | 0.973 |
| spend floor only | -0.487 | -0.487 | 0.000 | -0.243 | -0.243 | -0.487 |
| min usual only | 1.946 | 3.163 | 0.487 | 0.730 | 1.946 | 0.973 |
| both rules | 1.460 | 2.676 | 0.487 | 0.243 | 1.460 | 0.973 |

## What tops the spend-based scores (top 240, non-ignored)

- baseline_mean_k_std: Insurance & Fees 89, Subscriptions 30, Health & Fitness 27; true spikes 13. With at least 4 purchases in a usual month: true spikes 103 of 240
- robust_log_mad: Travel 238, Health & Fitness 1, Entertainment 1; true spikes 0. With at least 4 purchases in a usual month: true spikes 65 of 240
- spend_ratio_seasonal: Travel 240; true spikes 0. With at least 4 purchases in a usual month: true spikes 68 of 240

## User bootstrap at 0.03 flags per user-month (5-95%: precision, recall)

- baseline_mean_k_std: precision 0.033-0.079, recall 0.018-0.044
- count_poisson: precision 0.739-0.874, recall 0.436-0.501
- count_poisson_seasonal_income_floor: precision 0.807-0.908, recall 0.466-0.534
- count_negbin_seasonal_income: precision 0.714-0.832, recall 0.411-0.481

At 0.035 flags per user-month (the common rate):

- count_poisson_seasonal_income_floor: precision 0.737-0.860, recall 0.506-0.578

## count_poisson_seasonal_income_floor at 0.03: false positives

- flags 237, false positives 33; in categories FR-1 never plants in: 5
- by category: Dining 10, Transportation 9, Shopping 7, Entertainment 3, Groceries 2, Health & Fitness 2
- by month of year: 1 4, 2 4, 3 1, 4 4, 5 3, 6 3, 7 3, 8 4, 9 3, 10 1, 11 1, 12 2
- by persona (analysis only): freelancer 13, family_budgeter 13, young_professional 7

- of the leader's misses, the oracle misses at the same rate: 176
- true positives by tier: clear 204
- missed by tier: clear 169, weak 38
- missed by category: Shopping 63, Groceries 57, Transportation 52, Dining 35

## Size of the deviation (flagged true spikes, against the generator's)

- trailing mean x season: median absolute error of the usual level 0.098, 90th percentile 0.272; median excess $1,157 (true $1,087)
- own 12-month average: median absolute error of the usual level 0.090, 90th percentile 0.255; median excess $1,074 (true $1,087)
- the leader with the floor on the plain 12-month average: precision at 0.035 0.801
- purchases in a flagged spike month: median 48 against 19.0 in the user's average month

## Driving transactions (labeled spikes, train users)

- top 5 by amount: mean excess coverage 0.694, median 0.693, share at 1.0 0.312

## Look-alikes and basket-size spikes (planted categories)

- non-label months with spend >= 1.5x the user's 12-month average: 3685 (9.0 per label)
- the 3 largest charges cover the whole excess: 0.644
- count up less than 1.3x: 0.265 against the plain average, 0.282 against the seasonal one
- flagged by the leader at its 0.035 cutoff: 14
- simulated basket-size spikes (spend x U(1.8, 3.0), counts unchanged, 400 non-label months): recall at the 0.035 cutoff 0.003
