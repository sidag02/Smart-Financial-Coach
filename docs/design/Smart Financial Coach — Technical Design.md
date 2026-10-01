# Smart Financial Coach — Technical Design

Sep 29, 2026 · @Sidd

## Summary

This design describes how v1 of Smart Financial Coach meets the requirements in Smart Financial Coach — PRD: the modules, their interfaces, the infrastructure, and the high-level architecture choices with alternatives.

- **Core idea:** statistical models compute every number; an LLM assistant only explains results it gets back through a standard tool interface.
- **Model choices are deliberately open.** This doc fixes each model's interface and features; the specific algorithm is chosen afterwards, by theory or by experiment against the evaluation framework.
- **Out of scope here:** product scope and success targets, which live in the PRD.

## Problem formulation

The PRD's capabilities reduce to five learning problems, each with ground truth available from the synthetic data.

| Capability | Formulation | Input → output | Primary metric |
| --- | --- | --- | --- |
| Transaction categorization | Supervised multi-class text classification | Raw merchant string + amount → category (\~12 classes) + confidence | Macro F1 |
| Unusual transactions | Unsupervised, per-user point anomaly detection | Transaction features relative to the user's history → anomaly score + flag + reason | Precision, recall, PR-AUC |
| Spending spikes | Per-user anomaly detection on aggregated time series | Weekly / monthly spend per category → flagged period + deviation size + driving transactions | Precision, recall at period level |
| Goal forecasting | Univariate time-series forecasting with seasonality, turned into a probability | Monthly net savings history → forecast with interval → P(goal met by date) | RMSE, MAPE; Brier score for P(goal met) |
| Coaching | Grounded conditional text generation with tool use | User question + tool results → plain-English answer | Grounding accuracy; rubric score |

**Why unsupervised for anomalies:** real users never label their own anomalies, so the design must not depend on labels. Synthetic labels are used only to evaluate.

## System overview

The tool server is the single contract between the assistant layer and the intelligence layer; nothing above it touches models or data directly.

&#91;embedded content: system overview · modules by layer\]

The built-in coach and any external assistant use the same tools, so answers are consistent across clients. Model boxes name capabilities, not algorithms; those are chosen later (see Model selection approach).

## Modules

The system is nine modules in four layers, plus a cross-cutting evaluation harness; each owns one responsibility and is testable alone.

| Layer | Module | Responsibility | Inputs | Outputs |
| --- | --- | --- | --- | --- |
| Data | Data generator | Generate persona-based multi-user transactions, goals and planted events (unusual charges, spending spikes) with ground truth, from a parameter spec | Spec file (personas, catalog, events, seeds) | One SQLite file: model-visible tables + `truth_*` tables |
| Data | Data store | Hold transactions, users, goals, model outputs; data-access layer exposes only model-visible tables to models | Generator SQLite file | Queryable tables |
| Data | Feature pipeline | Clean merchant text; build per-user history and weekly/monthly aggregates | Raw transactions | Feature tables per model |
| Intelligence | Categorization service | Assign category + confidence | Merchant text, amount | Category, confidence |
| Intelligence | Anomaly service | Flag unusual transactions and spending spikes against the user's baseline | Transaction and aggregate features | Flags, scores, reasons |
| Intelligence | Forecasting service | Forecast savings and estimate goal likelihood | Monthly net-savings series, goal | Forecast, interval, P(goal met), gap |
| Access | Tool server | Expose intelligence services as typed tools; inject user identity | Tool calls + session identity | Structured tool results |
| Experience | Coach agent | Turn questions into tool calls and answers | User message, tool results | Grounded answer + tool trace |
| Experience | Web app | Dashboard, chat, user selection | User actions | Rendered views |
| Cross-cutting | Evaluation harness | Score every model and the coach against ground truth and baselines | Held-out data, test sets | Metrics report |

## Interfaces

Interfaces are fixed before any model is chosen, so model experiments can swap implementations without touching callers.

**Core data schemas**

| Entity | Key fields |
| --- | --- |
| Transaction | transaction\_id, user\_id, timestamp, amount, currency, merchant\_raw, channel |
| User | user\_id, persona, monthly\_income\_estimate |
| Goal | goal\_id, user\_id, name, target\_amount, created\_date, target\_date, as\_of\_date, current\_balance (as of as\_of\_date) |
| Anomaly flag | flag\_id, user\_id, level (transaction or period), ref (transaction\_id or period + category), score, reason |

**Ground truth** (eval only, in separate `truth_*` tables that models never read)

| Entity | Key fields |
| --- | --- |
| Transaction truth | transaction\_id, category, merchant\_id, process, is\_recurring, anomaly\_kind |
| Period truth (spikes) | user\_id, granularity, period\_start, category, multiplier |
| Goal truth | goal\_id, outcome\_class, met |

Full table definitions are in FR-1 Synthetic Data Generator — Feature Design.

**Service contracts** (each model service implements one of these)

- `categorize(merchant_raw, amount) → {category, confidence}`
- `score_transactions(user_id, transactions) → [{transaction_id, score, is_flagged, reason}]`
- `detect_spikes(user_id, period, granularity) → [{category, period, actual, expected, deviation, top_transactions}]`
- `forecast_goal(user_id, goal_id) → {projected_balance, interval, p_goal_met, gap, monthly_forecast[]}`

**Tools exposed to assistants** (no user\_id parameter; identity comes from the session)

| Tool | Purpose |
| --- | --- |
| get\_spending\_summary | Totals by category and month for a date range |
| get\_transactions | Filtered, categorized transactions |
| detect\_anomalies | Unusual transactions and spending spikes for a period |
| forecast\_goal | On-track status and gap for a goal |
| list\_goals | The user's goals |

All tool outputs are structured JSON with units and currency, so the assistant can quote numbers without doing arithmetic.

## Infrastructure

v1 runs as a single Python deployment on one machine; each component has a named upgrade path for the real-data release.

| Concern | v1 | Upgrade path (v2) |
| --- | --- | --- |
| Language / runtime | Python 3.11, pinned dependencies | Containerized services |
| Data store | SQLite (one file per generated dataset) | Managed Postgres with row-level security |
| Model artifacts | Versioned files in the repo's artifact folder | Model registry |
| Tool server | Local process speaking MCP over stdio / HTTP | Hosted service behind auth gateway |
| LLM | Hosted API (provider TBD), key from environment | Same, with rate limiting and cost budgets |
| Web app | Local Python web app | Hosted front end + API |
| Secrets | Environment variables, never committed | Secrets manager |
| CI | Lint, unit tests and evaluation suite on every push | Same, plus metric regression gates |
| Observability | Structured logs of tool calls and LLM latency / cost | Tracing, dashboards, alerting |
| Compute | CPU only | CPU; GPU only if an experiment justifies it |

## Architecture alternatives

Six high-level choices shape the system; each decision favors trustworthy numbers and reuse over raw flexibility.

| Decision | Options considered | Chosen | Why |
| --- | --- | --- | --- |
| Role of the LLM | (a) LLM analyzes raw transactions directly · (b) LLM writes SQL over the data · (c) LLM calls purpose-built model tools | (c) | Numbers come from tested models, not generation; (a) hallucinates arithmetic, (b) is flexible but hard to validate and secure |
| Tool interface | (a) Provider-specific function calling · (b) Custom REST API · (c) MCP server | (c) | One contract usable by any MCP-compatible assistant; no provider lock-in. Cost: an extra process and protocol to operate |
| Where personalization lives | (a) One global model · (b) One model per user · (c) Global model + per-user baseline features | Categorization (a); anomalies and forecasts (c) or (b), settled by experiment | Merchant meaning is shared across users; "normal spending" is personal |
| Compute timing | (a) Batch precompute nightly · (b) Compute on each request | (a) for forecasts and flags, (b) for categorization of new transactions | Keeps chat latency low; forecasts change slowly |
| Model hosting | (a) In-process library · (b) Separate model service | (a) | Simplest for v1; the service contracts allow a split later without caller changes |
| Data store | (a) Flat files (e.g. Parquet) · (b) SQLite · (c) Postgres | (b) | Zero ops for v1; standard library; one file per dataset; same SQL moves to Postgres in v2 |

## Security and data isolation

The assistant can never choose whose data it reads: user identity is bound to the session and attached to every tool call by the tool server.

1. The web app establishes the session user (v1: user selection; v2: real authentication).
2. The tool server receives the session identity out of band, never from the LLM's tool arguments.
3. Every data query is scoped by that identity at the data-access layer, not in each tool.
4. Tool results never include other users' records, so a prompt injection has nothing to leak.
5. The coach system prompt forbids investment advice and unverified numbers; tests enforce both.

**Other controls:** API keys only from environment variables; tool arguments validated against schemas; logs exclude raw transaction text beyond what debugging needs.

## Model selection approach

No model is chosen in this doc. Once each interface and feature set is fixed, candidates are either ruled out by theory or compared by experiment on the evaluation framework, and the winner is recorded as a decision.

**Selection criteria, in order:** primary metric on held-out data → beats the baseline → latency within budget → explainability of outputs → operational simplicity.

| Problem | Candidate families to evaluate | Decided by |
| --- | --- | --- |
| Categorization | Linear models on n-gram text features · gradient-boosted trees · sentence-embedding classifiers · small fine-tuned transformer | Experiment, incl. the unseen-merchant test |
| Unusual transactions | Isolation-based ensembles · one-class boundary methods · density methods · robust statistical rules | Experiment on precision / recall at a fixed alert rate |
| Spending spikes | Robust per-user statistics on aggregates · seasonal decomposition residuals · forecast-residual methods | Experiment; theory narrows to seasonality-aware options |
| Goal forecasting | Additive trend + seasonality models · ARIMA-family models · exponential smoothing | Experiment via rolling backtest; theory rules out options needing long history |
| Coach LLM | Hosted models of different size and cost tiers | Experiment on grounding accuracy, rubric score, cost per answer |

Each experiment's setup, results and decision are logged in the evaluation report so the choice is reproducible.

## Evaluation framework design

Every model is scored against planted ground truth and a simple baseline, with one command that regenerates data, retrains and writes the report.

| Problem | Data split | Baseline | Metrics |
| --- | --- | --- | --- |
| Categorization | Stratified 80/20 by transaction within train users (known merchants), plus test users' transactions at holdout merchants never seen in training | Keyword rules | Macro F1, per-class F1, confusion matrix |
| Unusual transactions | All transactions scored; labels hidden from training | Per-user z-score on amount | Precision, recall, PR-AUC, precision at fixed alert rate |
| Spending spikes | Weekly and monthly aggregates; labels hidden | Per-user mean ± k·std per category | Period-level precision and recall |
| Goal forecasting | Rolling-origin backtest: train on months 1..k, predict k+1..k+3 | Seasonal-naive | RMSE, MAPE; Brier score for P(goal met) |
| Coach | \~30 scripted questions with expected facts, plus adversarial cases | None | Grounding accuracy, refusal accuracy, rubric score (LLM judge, 1–5) |

**Controls against flattering results**

- Ambiguous merchants and messy merchant text generated into the data.
- Label noise applied by the training pipeline to loaded training data; the generator itself only generates.
- Separate seeds for training and test users.
- Goal examples use only transactions with `ts <= as_of_date` (the goal's backtest origin). The full ledger covers the target month, so later transactions reveal whether the goal was met. This is an acceptance criterion for the feature that builds training and evaluation splits.
- The judge model differs from the coach model, and a sample of judge scores is checked by hand.

## Build order and open questions

Modules are built bottom-up so each layer is tested before the next depends on it.

1. Foundations: repo layout, pinned dependencies, configuration, CI.
2. Data generator, data store and data quality checks.
3. Feature pipeline and service interfaces with baseline implementations.
4. Evaluation harness running on the baselines.
5. Model experiments per problem; record decisions.
6. Tool server with session-scoped identity.
7. Coach agent and coach test suite.
8. Web app: user selection, dashboard, chat.
9. Evaluation report and technical docs.

**Open questions**

- [ ] LLM provider: Anthropic or OpenAI?
- [ ] Web framework for v1 (a Python dashboard framework vs. a separate front end)?
- [ ] Tool server transport for v1: stdio only, or HTTP as well?
