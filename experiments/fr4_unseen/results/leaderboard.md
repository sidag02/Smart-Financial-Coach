- Data hash `2f0e60a671c7`, split hash `ce93ef873e6c`, code versions: `5570dd0f929d`
- Ranked by `val_unseen_macro_f1` (95% merchant-bootstrap interval); the difference column is the paired interval against the leader. *(tied)* marks runs in the leader's tie set, which are ordered by `val_unseen_brier`, then `latency_batch_ms`, then `complexity`.

| Rank | Run | `val_unseen_macro_f1` (95% CI) | vs leader | `val_known_macro_f1` | `val_unseen_brier` | `val_unseen_ece` | `val_unseen_acc_at_90` | `val_unseen_coverage_at_90` | `val_unseen_misallocated_spend` | `latency_batch_ms` | `latency_p95_ms` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `21_base_noise0_unweighted` (tied) | 0.745 (0.68–0.80) | – | 0.988 | 0.126 | 0.050 | 0.918 | 0.612 | 0.113 | 4537.63 | 5.81 |
| 2 | `41_emb_only_noise0` (tied) | 0.704 (0.64–0.77) | -0.077 to +0.005 | 0.980 | 0.130 | 0.051 | 0.932 | 0.525 | 0.141 | 4383.88 | 4.83 |
| 3 | `22_small_noise0` (tied) | 0.715 (0.65–0.77) | -0.058 to +0.005 | 0.988 | 0.134 | 0.056 | 0.913 | 0.554 | 0.124 | 1562.45 | 2.99 |
| 4 | `20_base_noise0` | 0.714 (0.65–0.78) | -0.051 to -0.001 | 0.988 | 0.130 | 0.057 | 0.917 | 0.586 | 0.119 | 4579.40 | 6.27 |
| 5 | `30_base_unweighted` | 0.666 (0.59–0.72) | -0.121 to -0.043 | 0.988 | 0.148 | 0.048 | 0.929 | 0.447 | 0.167 | 4690.55 | 6.44 |
| 6 | `40_emb_only` | 0.536 (0.47–0.60) | -0.258 to -0.141 | 0.972 | 0.184 | 0.064 | 0.962 | 0.271 | 0.254 | 4577.01 | 4.90 |
| 7 | `10_ref_base` | 0.512 (0.45–0.58) | -0.280 to -0.173 | 0.969 | 0.216 | 0.062 | 0.973 | 0.146 | 0.268 | 4386.36 | 4.99 |
| 8 | `51_knn_noise0` | 0.468 (0.38–0.55) | -0.352 to -0.201 | 0.972 | 0.212 | 0.105 | 0.777 | 0.069 | 0.517 | 4450.12 | 6.49 |
| 9 | `50_knn` | 0.468 (0.38–0.55) | -0.352 to -0.201 | 0.971 | 0.210 | 0.092 | 0.764 | 0.065 | 0.517 | 4375.81 | 7.70 |

**Baselines** (the floor, not candidates):

| Run | `val_unseen_macro_f1` | `val_known_macro_f1` | `val_unseen_brier` | `val_unseen_ece` | `val_unseen_acc_at_90` | `val_unseen_coverage_at_90` | `val_unseen_misallocated_spend` | `latency_batch_ms` | `latency_p95_ms` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `keyword` | 0.425 | 0.474 | 0.197 | 0.051 | 0.934 | 0.241 | 0.453 | 51.13 | 0.93 |
