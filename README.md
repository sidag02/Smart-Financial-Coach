# Smart Financial Coach

Statistical models compute every number (categorization, anomaly detection, goal forecasting); an LLM coach only explains results it receives through an MCP tool server.

- Product requirements: [Smart Financial Coach — PRD.md](<docs/product/Smart Financial Coach — PRD.md>)
- Architecture: [Smart Financial Coach — Technical Design.md](<docs/design/Smart Financial Coach — Technical Design.md>)

## Layout

```
src/smart_financial_coach/
  config.py        settings from env vars (SFC_*) / .env
  data/            generator (FR-1), data store (SQLite), feature pipeline
  intelligence/    categorization, anomaly and forecasting services
  access/          MCP tool server with session-scoped identity
  experience/      coach agent, web app
  evaluation/      evaluation harness and reports
artifacts/         versioned model artifacts
configs/data/      generator specs, persona files and merchant catalog
data/              generated data (git-ignored)
docs/product/      product requirements (PRD)
docs/design/       technical design and architecture decisions
tests/
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in `.python-version`.

```sh
uv sync                        # create .venv and install pinned deps from uv.lock
cp .env.example .env           # then fill in values
uv run pre-commit install      # optional: run checks on commit
```

## Development

```sh
uv run pytest                  # tests
uv run pytest -m slow          # long-running checks, e.g. full dataset runtime (CI: slow.yml)
uv run ruff check . && uv run ruff format .
uv run mypy                    # strict type checking
```

CI (`.github/workflows/ci.yml`) runs lint, format check, type check and tests on every push.

## Synthetic data

```sh
uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite
uv run sfc-data validate --spec configs/data/default.yaml data/synthetic/default.sqlite
uv run sfc-data hash data/synthetic/default.sqlite
uv run sfc-data labels data/synthetic/default.sqlite   # label counts, tiers and oracle ceilings
```

`small.yaml` (30 users, a few seconds) is for tests; `clean.yaml` has canonical merchant names and no planted events. Design: [FR-1 Synthetic Data Generator — Feature Design.md](<docs/design/features/FR-1 Synthetic Data Generator — Feature Design.md>).

Ground truth lives in `truth_*` tables. Score flags against it with `smart_financial_coach.data.labels` (`load_truth`), which model code under `intelligence/` must never import. Design: [FR-2 Ground Truth Labels — Feature Design.md](<docs/design/features/FR-2 Ground Truth Labels — Feature Design.md>).
