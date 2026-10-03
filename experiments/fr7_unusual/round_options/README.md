# FR-7 round: exploratory variants (validation only)

Scripts behind the validation-only comparisons in `docs/reports/FR-7 Unusual Transactions — Round Results.md`. They monkeypatch the round's code, so run them from a checkout of the round's training commit, `afeed9c` (branch `feature/fr-7-m3` at the first round). Each writes to its own MLflow store and never touches the round's leaderboard.

```sh
git checkout afeed9c
uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite
uv run python <this folder>/profile_options.py A data/synthetic/default.sqlite /tmp/options-A.db
uv run python <this folder>/profile_options.py B data/synthetic/default.sqlite /tmp/options-B.db
uv run python <this folder>/one_sided_rank.py data/synthetic/default.sqlite /tmp/one-sided.db
```

- `profile_options.py A`: a merchant's typical price counts only users with 2+ charges there.
- `profile_options.py B`: profiles need 6+ other users.
- `one_sided_rank.py`: the forest with `max(rank, 0.5)`. It has since become a real candidate, `31_isolation_forest_one_sided` (owner decision on #39), and this script is kept for the record.
