# FR-5 feasibility on FR-4's promoted model (Oct 3, 2026)

`review_thresholds.py` on the promoted model's twin (`21_small_unweighted.ship`, run `6bc58706`, FR-4 round), validation only, data hash `44781bc4e4a5`:

```sh
uv run python experiments/fr5_review/review_thresholds.py --run 6bc58706a55343b8abd03927a870621d \
    --tracking-uri sqlite:///<the FR-4 round's mlruns>/mlflow.db
```

Model `20eea4fb-44781bc4-c0274576`. Test users' rows at strings it hasn't seen: 23.7%. Error rate: familiar **2.8%**, unfamiliar **18.6%** (the old model `3f0ccc82`: 2.0% and 41.9%).

| Threshold | Familiar flagged | Familiar errors caught | Familiar flags wrong | Unfamiliar flagged | Unfamiliar errors caught | Unfamiliar flags wrong | Mix flagged | Mix errors caught |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.1% | 1% | 45% | 7.2% | 22% | 57% | 1.8% | 15% |
| 0.60 | 3.3% | 58% | 50% | 15.8% | 39% | 46% | 6.2% | 45% |
| 0.70 | 4.4% | 77% | 49% | 25.8% | 53% | 38% | 9.5% | 61% |
| 0.80 | 5.8% | 98% | 48% | 33.1% | 61% | 34% | 12.3% | 73% |
| 0.90 | 6.0% | 100% | 47% | 43.1% | 67% | 29% | 14.8% | 78% |
| 0.95 | 6.2% | 100% | 46% | 53.5% | 71% | 25% | 17.4% | 81% |

- **The design's threshold rule (#15 §1) can't be met for unfamiliar strings:** no threshold catches 80% of unfamiliar errors (71% at 0.95). The clean model's remaining unfamiliar errors are often confident ones.
- **For familiar strings,** flags are at least 25% errors at every threshold, so the rule's "highest" goes to 0.95 or more: about 6% of familiar rows, catching nearly all familiar errors.
- **Review burden** (distinct merchant strings per test user): first month 27.4 familiar and 8.7 unfamiliar; from month 4, 1.6 and 0.8 new strings a month. At familiar < 0.95 and unfamiliar < 0.8 that is roughly 4–5 items in the first month, then under 0.4 a month.
- **Automation bias:** at unfamiliar < 0.8, 34% of unfamiliar flags are real errors (the old model: about half). Confirmations are more often right, and the correction requirement (#15 §4) still guards the rest.
