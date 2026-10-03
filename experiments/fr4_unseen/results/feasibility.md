# FR-4 feasibility: unseen-merchant categorization

Oct 2, 2026. Validation data only: the launch round's 3 merchant-grouped folds over train users (169 held-out merchants), default dataset, data hash `2f0e60a671c7`, split hash `ce93ef873e6c`, one code version (`5570dd0f929d`). No test set was scored.

## Candidates

`configs/experiments/categorization/fr4/`, run into a separate MLflow store so nothing joins the FR-3 leaderboard:

```sh
uv run sfc-experiment run configs/experiments/categorization/fr4/ --data data/synthetic/default.sqlite --tracking-uri sqlite:///mlruns/poc-fr4.db
uv run sfc-experiment report --task categorization --data data/synthetic/default.sqlite --tracking-uri sqlite:///mlruns/poc-fr4.db
```

All candidates are calibrated per familiarity group, C = 1.0. "Noise" is the injected 2% uniform label noise (FR-3 §5); "clean" sets `label_noise: 0`.

| Run | Unseen macro F1 (95% CI) | Paired vs leader | Known macro F1 | Unseen Brier | Unseen misallocated spend | Batch ms (10k rows) |
| --- | --- | --- | --- | --- | --- | --- |
| `21_base_noise0_unweighted`: bge-base, clean, no class weights | **0.745** (0.68–0.80) | – | 0.988 | 0.126 | 11.3% | 4,538 |
| `22_small_noise0`: bge-small, clean | 0.715 (0.65–0.77) | −0.058 to +0.005 (tied) | 0.988 | 0.134 | 12.4% | 1,562 |
| `20_base_noise0`: bge-base, clean | 0.714 (0.65–0.78) | −0.051 to −0.001 | 0.988 | 0.130 | 11.9% | 4,579 |
| `41_emb_only_noise0`: embeddings only, clean | 0.704 (0.64–0.77) | −0.077 to +0.005 (tied) | 0.980 | 0.130 | 14.1% | 4,384 |
| `30_base_unweighted`: bge-base, noise, no class weights | 0.666 (0.59–0.72) | −0.121 to −0.043 | 0.988 | 0.148 | 16.7% | 4,691 |
| `40_emb_only`: embeddings only, noise | 0.536 (0.47–0.60) | −0.258 to −0.141 | 0.972 | 0.184 | 25.4% | 4,577 |
| `10_ref_base`: the shipped configuration (bge-base, noise) | 0.512 (0.45–0.58) | −0.280 to −0.173 | 0.969 | 0.216 | 26.8% | 4,386 |
| `51_knn_noise0`: nearest neighbours, clean | 0.468 (0.38–0.55) | −0.352 to −0.201 | 0.972 | 0.212 | 51.7% | 4,450 |
| `50_knn`: nearest neighbours, noise | 0.468 (0.38–0.55) | −0.352 to −0.201 | 0.971 | 0.210 | 51.7% | 4,376 |
| `keyword` (baseline) | 0.425 | | 0.474 | 0.197 | 45.3% | 51 |

The reference reproduces the launch round's bge-base exactly (0.512), so the comparison is on the same footing. The full generated report is `leaderboard.md`.

## Per-category F1 on unseen merchants

`uv run python experiments/fr4_unseen/unseen_errors.py <tracking-uri or -> <run_id>`. Each merchant is counted once, under its majority category; the ambiguous merchants (warehouse clubs, marketplaces) have rows in both Groceries and Shopping. *Corrected in review (PR #17): an earlier version counted them in both, so the column summed to 172.*

| Category | Merchants | Shipped (`10_ref_base`) | Clean (`20`) | Clean, no weights (`21`) |
| --- | --- | --- | --- | --- |
| Housing | 8 | 0.689 | 0.985 | 0.981 |
| Utilities | 12 | 0.916 | 0.905 | 0.892 |
| Groceries | 17 | 0.588 | 0.815 | 0.801 |
| Dining | 33 | 0.670 | 0.893 | 0.886 |
| Transportation | 14 | 0.696 | 0.731 | 0.719 |
| Shopping | 26 | 0.580 | 0.758 | 0.801 |
| Entertainment | 8 | 0.289 | 0.358 | 0.463 |
| Subscriptions | 15 | 0.879 | 0.883 | 0.890 |
| Health & Fitness | 10 | 0.075 | 0.142 | 0.156 |
| Travel | 8 | **0.025** (63,545 predicted vs 848 true) | 0.683 | 0.886 |
| Childcare & Education | 10 | 0.396 | 0.652 | 0.706 |
| Insurance & Fees | 8 | 0.342 | 0.770 | 0.760 |
| Accuracy (rows) | | 0.581 | 0.817 | 0.825 |

## How precise is a holdout of N merchants?

`uv run python experiments/fr4_unseen/holdout_size.py`: category-stratified subsets of the 169 validation merchants, bge-base's pooled validation predictions (launch run `000ef7d3`, unseen macro F1 0.512).

| Merchants | Bootstrap SD | 95% bootstrap interval width | P(score ≥ 0.70) if true score is 0.745 | … if 0.73 |
| --- | --- | --- | --- | --- |
| 58 (today's test holdout) | 0.048 | 0.186 | 83% | 73% |
| 87 | 0.042 | 0.163 | 86% | 76% |
| 115 | 0.038 | 0.148 | 88% | 78% |

- The bootstrap resamples merchants with replacement, so it treats them as independent draws, as a fresh holdout's merchants are. Going from 58 to 115 merchants narrows the interval by about a fifth (√(58/115) predicts about 29% for independent merchants).
- At 58 merchants, subsets of the 169 put the same model anywhere from 0.425 to 0.624.
- 0.73 allows for 0.745 being the best of nine runs on the same folds. The pass rates use a normal approximation, and the SD measured on this model (unseen F1 0.51), not on the leader.

*Corrected in review (PR #17):* an earlier version reported the spread of subset scores at 87, 115 and 140 merchants. Subsets drawn without replacement from the same 169 merchants share most of them (finite-population factor 0.57 at 115), so that spread understated a fresh holdout's: SD 0.029 at 115, against 0.038 from the bootstrap.

## A larger holdout

`experiments/fr4_unseen/holdout40.yaml` extends the default spec: holdout share 0.4, a new holdout seed (8), `test_user_bias` 0.45.

```sh
uv run sfc-data generate --spec experiments/fr4_unseen/holdout40.yaml --out data/synthetic/holdout40.sqlite
```

- 115 holdout merchants (Travel, Housing and Entertainment 6 each; Dining 19), against 58.
- 23.5% of test users' spending at holdout merchants; 85,051 transactions. All data quality checks pass.
- With the default `test_user_bias` (1.25), 43% of test spending lands at holdout merchants, outside the 15–25% check.
