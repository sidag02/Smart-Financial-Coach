# Smart Financial Coach

Statistical models compute every number (categorization, anomaly detection, goal forecasting); an LLM coach only explains results it receives through an MCP tool server.

- Product requirements: [Smart Financial Coach — PRD.md](<docs/product/Smart Financial Coach — PRD.md>)
- Architecture: [Smart Financial Coach — Technical Design.md](<docs/design/Smart Financial Coach — Technical Design.md>)

## Layout

```
src/smart_financial_coach/
  config.py        settings from env vars (SFC_*) / .env
  data/            generator, data store (SQLite + Parquet), feature pipeline
  intelligence/    categorization, anomaly and forecasting services
  access/          MCP tool server with session-scoped identity
  experience/      coach agent, web app
  evaluation/      evaluation harness and reports
artifacts/         versioned model artifacts
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
uv run ruff check . && uv run ruff format .
uv run mypy                    # strict type checking
```

CI (`.github/workflows/ci.yml`) runs lint, format check, type check and tests on every push.
