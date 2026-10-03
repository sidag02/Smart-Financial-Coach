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

## Models and experiments

Models are trained, compared and promoted through the evaluation framework; callers only ever load the promoted model for a service (`intelligence.service.load_service`).

```sh
uv run sfc-experiment run configs/experiments/categorization/ --data data/synthetic/default.sqlite
uv run sfc-experiment leaderboard --task categorization --data data/synthetic/default.sqlite
uv run sfc-experiment report --task categorization --data data/synthetic/default.sqlite   # Markdown, with intervals
uv run sfc-experiment finalize --task categorization --data data/synthetic/default.sqlite   # top three, once
uv run sfc-model promote --task categorization --run <rank-1 id> --note "<explainability and operations>"
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # compare runs
```

Runs are tracked in MLflow: a local store in `mlruns/` (git-ignored) by default, or a shared server via `SFC_MLFLOW_TRACKING_URI`. MLflow is in the `train` dependency group; a serving install can leave it out with `uv sync --no-default-groups`. Design: [FR-3 Transaction Categorization — Feature Design.md](<docs/design/features/FR-3 Transaction Categorization — Feature Design.md>).

## Web app

Overview, transactions and the coach chat, from a read-only demo bundle: the demo accounts' data (`configs/web/demo_accounts.yaml`), categorized by the promoted model.

```sh
uv run sfc-web build-demo --data data/synthetic/default.sqlite   # -> build/demo
uv run sfc-web serve --dev        # http://127.0.0.1:8000, password demo-password, one-click sign-in
```

`--dev` makes throwaway secrets for plain http. A hosted run needs `SFC_DEMO_PASSWORD` and `SFC_SESSION_SECRET`, and `SFC_LLM_API_KEY` (or `ANTHROPIC_API_KEY`) for chat; without a key, chat says it's unavailable and the rest works. The `Dockerfile` packages the app with `build/demo`. Design: [Smart Financial Coach — Web App UI.md](<docs/design/Smart Financial Coach — Web App UI.md>) and the mockups in `docs/design/mockups/`.

### Demo deployment (Azure)

`deploy/azure/provision.sh` creates the demo's resource group, registry, Container Apps environment and app, builds the first image in Azure, and sets up the GitHub `demo` environment with an OIDC identity scoped to that resource group. After that, `.github/workflows/deploy.yml` rebuilds the bundle and image and rolls them out on every merge to `main` that touches the app. `deploy/azure/set-llm-key.sh` passes the Anthropic key from `.env` to the app as a secret; `deploy/azure/teardown.sh` deletes everything.
