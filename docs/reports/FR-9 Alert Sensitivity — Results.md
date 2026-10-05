# FR-9 Alert Sensitivity — Results

Oct 4, 2026 · @Sidd · Milestone 1 of FR-9 Alert Sensitivity and Flag Actions — Feature Design

## Summary

- **Presets hold their rates on users they weren't placed on.** Out of fold, Less often flags 0.49× and More often 1.98× as many unusual charges per post-warm-up user-month as Balanced; for spending spikes, 0.50× and 2.01×.
- **Precision per level, out of fold:**

  | | Less often | Balanced | More often |
  | --- | --- | --- | --- |
  | Unusual charges | 0.939 (0.914–0.961) | 0.792 (0.767–0.816) | 0.455 (0.430–0.481) |
  | Spending spikes | 0.963 (0.927–0.992) | 0.799 (0.752–0.849) | 0.491 (0.449–0.532) |

- **Balanced reproduces each round's own out-of-fold precision** (0.792 and 0.799 in the FR-7 and FR-8 round results), which checks the setup.
- **More often costs 0.31–0.34 of precision for 0.10–0.11 of recall,** close to the design's in-sample estimate. About half of what it shows is ordinary, and the page says so (decision 5).
- **In the warm-up,** More often flags no more than Balanced (decision 17): 65 unusual charges on the held-out users at both levels, against 2 at Less often. Spike periods are never in the warm-up.
- **Reported, not gated.** Balanced keeps its gates. No test user was scored (decision 4).

## Setup

- **Data:** the default dataset (spec hash `26e86f3b`, content hash `b4d43bf4`), the data both promoted models were built on.
- **Models:** the promoted configs, `31_isolation_forest_one_sided` (`e0b67433-8b9632e6-2e033606`) and `20_count_negbin` (`8c428c54-d85b4650-64917ea6`). The script checks each config's hash against the promoted version.
- **Folds:** each round's own 5 user-grouped folds of the 240 train users, built by the task with the round's seed and leak checks.
- **The method** (decision 12), in each fold:
  - fit the config on the other folds' users, as the round did; Balanced is its fitted cutoff;
  - place Less and More on those same users with `sfc-model presets`' rule (`presets.place`: post-warm-up rows, `top_k`'s tie-break by id);
  - apply all three to the held-out users. In a user's first 90 days the cutoff is the stricter of the level and Balanced (decision 17).
- **Metrics:** precision and recall through the label contract on the pooled held-out flags, with 95% user-bootstrap intervals (1,000 draws). Flags per post-warm-up user-month: FR-7's span-based user-months, and FR-8's (user, month) count. The contract scores nothing in the warm-up, so warm-up flags are counted apart. Spike rows simulating basket-size spikes are left out.

**To reproduce:**

```
uv run python scripts/fr9_presets_oof.py data/synthetic/default.sqlite
```

## Results

### Unusual charges (FR-7)

240 train users, 7,930 post-warm-up user-months.

| Preset | Flags after warm-up | Per post-warm-up user-month | vs Balanced | Precision (95%) | Recall (95%) | Flags in warm-up |
| --- | --- | --- | --- | --- | --- | --- |
| Less often | 505 | 0.0637 | 0.49× | 0.939 (0.914–0.961) | 0.435 (0.407–0.463) | 2 |
| **Balanced** | 1,024 | 0.1291 | 1.00× | **0.792** (0.767–0.816) | 0.755 (0.725–0.783) | 65 |
| More often | 2,030 | 0.2560 | 1.98× | 0.455 (0.430–0.481) | 0.866 (0.845–0.885) | 65 |

Each fold's cutoffs barely move: Less 0.771–0.784, Balanced 0.728–0.734, More 0.682–0.687.

### Spending spikes (FR-8)

240 train users, 7,920 post-warm-up user-months.

| Preset | Flags after warm-up | Per post-warm-up user-month | vs Balanced | Precision (95%) | Recall (95%) | Flags in warm-up |
| --- | --- | --- | --- | --- | --- | --- |
| Less often | 136 | 0.0172 | 0.50× | 0.963 (0.927–0.992) | 0.316 (0.266–0.364) | 0 |
| **Balanced** | 274 | 0.0346 | 1.00× | **0.799** (0.752–0.849) | 0.513 (0.462–0.562) | 0 |
| More often | 552 | 0.0697 | 2.01× | 0.491 (0.449–0.532) | 0.611 (0.558–0.660) | 0 |

Cutoffs per fold: Less 12.3–12.4, Balanced 7.63–7.81, More 5.36–5.50.

## The committed presets

`sfc-model presets` placed each promoted model's presets on all 360 users of the same dataset and wrote them beside the manifest (`artifacts/<task>/<version>/presets.json`):

| Model | Less | Balanced | More | Post-warm-up flags (Less / Balanced / More) | In the warm-up |
| --- | --- | --- | --- | --- | --- |
| Unusual charges | 0.7783 | 0.7300 | 0.6855 | 768 / 1,536 / 3,072 | 3 / 66 / 66 |
| Spending spikes | 12.64 | 7.682 | 5.553 | 200 / 400 / 800 | 0 |

Unusual charges' pool cutoffs fall inside each fold's range above. Spikes' Less and More fall just outside it (12.64 against 12.30–12.44, and 5.55 against 5.36–5.50), because the pool is scored on the promoted categorizer's predicted categories, as serving is, and the folds on true ones, over fewer users. On the pool, 0.5× and 2× hold by construction (200 / 400 / 800). The simple-rule fallback has no promoted artifact, so it fits its own Less and More at build time, at 0.5× and 2× its rate (decision 13).

## Reading the results

- **The rates carry.** Cutoffs placed on four folds' users give 0.49–0.50× and 1.98–2.01× on the fifth's. A cutoff placed on the pool should therefore give users close to the alert rate their setting promises.
- **More often is noisy by design.** Its precision is under half for both halves. That's the trade decision 5 accepted, with copy that says so.
- **Less often keeps almost only planted alerts** (0.94 and 0.96). It finds 44% of planted unusual charges and 32% of planted spikes.
- **Synthetic data is easier than real data** (PRD caveat). On real data these cutoffs come from a rate, without labels; the precision at each level has to be measured again (v2).
