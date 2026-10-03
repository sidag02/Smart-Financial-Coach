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
| Transaction categorization | Supervised multi-class text classification | Transaction rows (merchant text, amount, channel, time) → one of 12 spending categories or Income + calibrated confidence | Macro F1 over the 12 spending categories |
| Unusual transactions | Unsupervised, per-user point anomaly detection | Transaction features relative to the user's history → anomaly score + flag + reason | Precision, recall, PR-AUC |
| Spending spikes | Per-user anomaly detection on aggregated time series | Monthly spend per category → flagged period + deviation size + driving transactions (weekly deferred; see FR-2) | Precision, recall at period level |
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
| Data | Data store | Hold transactions, users and goals; data-access layer exposes only model-visible tables to models. Model outputs live in separate predictions files, one per model version and dataset, so the generator's file stays a pure function of its spec | Generator SQLite file; predictions files | Queryable tables |
| Data | Feature pipeline | Clean merchant text; build per-user history and weekly/monthly aggregates | Raw transactions | Feature tables per model |
| Intelligence | Categorization service | Assign category + confidence, in batches on ingestion | Batches of transaction rows | Category, confidence, model version |
| Intelligence | Anomaly service | Flag unusual transactions and spending spikes against the user's baseline | Transaction and aggregate features | Flags, scores, reasons |
| Intelligence | Forecasting service | Forecast savings and estimate goal likelihood | Monthly net-savings series, goal | Forecast, interval, P(goal met), gap |
| Access | Tool server | Expose intelligence services as typed tools; inject user identity | Tool calls + session identity | Structured tool results |
| Experience | Coach agent | Turn questions into tool calls and answers | User message, tool results | Grounded answer + tool trace |
| Experience | Web app | Dashboard, chat, user selection | User actions | Rendered views |
| Cross-cutting | Evaluation harness | Build training and evaluation splits (each example sees only data up to its origin); score every model and the coach against ground truth and baselines | Generator SQLite file, test sets | Splits, metrics report |

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
| Transaction truth | transaction\_id, category, merchant\_id, process, is\_recurring, anomaly\_kind, tier, related\_transaction\_id |
| Period truth (spikes) | spike\_id, user\_id, granularity, period\_start, period\_end, category, multiplier, expected\_count, expected\_spend, base\_spend, extra\_spend, tier |
| Expected spend | user\_id, category, granularity, period\_start, expected\_count, expected\_spend |
| Goal truth | goal\_id, outcome\_class, met |

Full table definitions are in FR-1 Synthetic Data Generator — Feature Design and FR-2 Ground Truth Labels — Feature Design. How flags are scored against these tables (the label contract) is in FR-2 and in `data/labels.py`.

**Service contracts** (each model service implements one of these)

- `categorize(transactions) → [{transaction_id, category, confidence, model_version}]`: batch-first over model-visible transaction rows (`transaction_id, user_id, ts, amount, currency, merchant_raw, channel`), one output row per input in order. Whole rows keep the contract stable whichever features a model uses; a single transaction is a one-row batch. Every service's output is checked against its contract at runtime (FR-3)
- `score_transactions(user_id, transactions) → [{transaction_id, score, is_flagged, reason_code, reason}]` (`reason_code` is `duplicate`, `amount_unusual` or `new_merchant`, so reason accuracy can be scored)
- `detect_spikes(user_id, period, granularity) → [{category, period, actual, expected, deviation, top_transactions}]` (v1: `granularity="month"` only; at most 5 `top_transactions`)
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
| Model artifacts | The repo's `artifacts/` holds each service's `PROMOTED` pointer, promotion log and model manifests; model files are attached to GitHub Releases and verified against the committed manifest on first use (FR-3) | Container registry: continuous deployment bakes the promoted model into the serving image |
| Experiment tracking | MLflow with a local store: runs, metrics and the model registry (`champion` alias); serving never reads it | Shared MLflow server |
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
| Compute timing | (a) Batch precompute nightly · (b) Compute on each request · (c) Batch on ingestion | (a) for forecasts and flags, (c) for categorization of new transactions | Keeps chat latency low; forecasts change slowly. Transactions arrive from bank feeds in batches, so categorizing them on ingestion, batched across users, keeps per-transaction latency off any user's path. Serving cost becomes a cluster-sizing question (FR-3, decision Oct 2, 2026) |
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

**Selection criteria, in order:** primary metric on held-out data → beats the baseline → latency within budget → explainability of outputs → operational simplicity. For batch-precomputed services (categorization, flags, forecasts), "latency" means batch serving cost, used to break ties and size the cluster rather than as a gate. Only paths a user waits on, such as chat answers (NFR-5), get hard latency gates.

| Problem | Candidate families to evaluate | Decided by |
| --- | --- | --- |
| Categorization | Linear models on n-gram text features · gradient-boosted trees · sentence-embedding classifiers · small fine-tuned transformer | Experiment, incl. the unseen-merchant test. **Decided (Oct 2, 2026):** logistic regression over character n-grams and frozen `bge-base-en-v1.5` embeddings, calibrated per familiarity group (FR-3 Categorization Model Selection) |
| Unusual transactions | Isolation-based ensembles · one-class boundary methods · density methods · robust statistical rules | Experiment on precision / recall at a fixed alert rate |
| Spending spikes | Robust per-user statistics on aggregates · seasonal decomposition residuals · forecast-residual methods | Experiment; theory narrows to seasonality-aware options |
| Goal forecasting | Additive trend + seasonality models · ARIMA-family models · exponential smoothing | Experiment via rolling backtest; theory rules out options needing long history |
| Coach LLM | Hosted models of different size and cost tiers | Experiment on grounding accuracy, rubric score, cost per answer |

Each experiment's setup, results and decision are logged in the evaluation report so the choice is reproducible.

## Evaluation framework design

Every model is scored against planted ground truth and a simple baseline, with one command that regenerates data, retrains and writes the report.

| Problem | Data split | Baseline | Metrics |
| --- | --- | --- | --- |
| Categorization | Stratified 80/20 by transaction within train users (known merchants), plus test users' transactions at holdout merchants never seen in training. Hyperparameters and calibrators come from grouped cross-fitting over merchants within train users | Keyword rules | Macro F1 over the 12 spending categories (Income on its own line), per-class F1, confusion matrix; a merchant-level bootstrap interval for unseen merchants; calibration (Brier, ECE) on every test set |
| Unusual transactions | All transactions scored; labels hidden from training; thresholds tuned on train users, reported on test users | Per-user z-score on amount | Precision, recall, PR-AUC, precision at fixed alert rate, reason accuracy |
| Spending spikes | Monthly aggregates of all spend on true categories; labels hidden; thresholds tuned on train users, reported on test users | Per-user mean ± k·std per category | Period-level precision and recall (all and `clear` labels); excess coverage of driving transactions vs. a top-5-by-amount baseline |
| Goal forecasting | Rolling-origin backtest: train on months 1..k, predict k+1..k+3 | Seasonal-naive | RMSE, MAPE; Brier score for P(goal met) |
| Coach | \~30 scripted questions with expected facts, plus adversarial cases | None | Grounding accuracy, refusal accuracy, rubric score (LLM judge, 1–5) |

**How models are trained, compared and promoted** (built in FR-3; each later model extends it rather than replacing it)

- A generic model interface and registry: a candidate is a class plus a config file. Wrappers (calibration, routing, lookup) compose models through the same interface.
- Composable splitters with leak checks that stop a run before any fit. A task owns each problem's examples, labels, splits, metrics, baseline and gates.
- An experiment runner that logs every run's config, data hash, split hashes, code version and validation metrics to MLflow. A run is skipped on resume only if all of them match.
- **Experiments are compared on validation data.** Each task writes its decision rule before the runs; test sets are scored once, for at most three finalists plus the baselines, enforced in code. Departing from the rule needs a recorded override reason.
- Promotion checks the task's gates in the selection criteria's order, exports the model, records it in a committed promotion log, and is the only way a model reaches callers. Callers load "the promoted model" for a service, never a model class.

**Controls against flattering results**

- Ambiguous merchants and messy merchant text generated into the data.
- Label noise applied by the training pipeline to loaded training data; the generator itself only generates.
- Separate seeds for training and test users.
- Anomaly thresholds tuned on train users and reported on test users, never both on the same users.
- Spike metrics scored on true categories, so categorizer errors don't leak into FR-8.
- An oracle with the generator's true expected spend must reach the precision target before a model is judged against it (`validate` gate; FR-2).
- Goal examples use only transactions with `ts <= as_of_date` (the goal's backtest origin). The full ledger covers the target month, so later transactions reveal whether the goal was met. The evaluation harness enforces this when it builds splits (build order step 4).
- The judge model differs from the coach model, and a sample of judge scores is checked by hand.

## Learning from user feedback (direction, not yet designed)

Categorization ships with a cold-start model that only needs to be decent (FR-3). After launch, quality comes from feedback: low-confidence review (FR-5) and user corrections (FR-6), both P1 for v1.1. A correction says more than "the model was wrong". It also shows how a user *prefers* to see their money. This section sets the direction, so v1 doesn't close doors. A feature design comes with FR-5 and FR-6.

**A correction means one of two things, and the system has to tell them apart.**

- **The model was wrong** ("Taco Bell isn't Shopping"). The fix should reach everyone.
- **The user sees it differently** ("I count Costco as Groceries, not Shopping"). The fix should change only that user's view.

Training personal preferences into the global model leaks one user's view into everyone's categories. Keeping real errors as per-user overrides makes every user fix the same mistake.

**Direction**

1. **Per-user overrides in front of the model.** A user's correction for a merchant applies to that user at once, with no retraining. It is deterministic and explainable.
2. **Global labels only by agreement.** A correction becomes a training label when enough distinct users relabel the same merchant the same way.
3. **Scheduled retraining through the evaluation harness.** The same splits, gates and promotion as the cold model, so a bad batch of feedback can't ship unnoticed. Models already take labels as a training argument for this reason.
4. **Clusters of preferences are product signal.** Many users pushing merchants towards a category that doesn't exist is a taxonomy question, not a model fix.

**Constraints the design must meet**

- **Privacy (user story 5: "I only ever see my own data").** Some merchant strings are personal: person-to-person payments (`ZELLE TO <name>`), a landlord's name, a babysitter's payment app handle. Promoted to a global label, one user's private text and their label for it would end up in a model every user is served by. So the agreement rule's threshold is a minimum number of **distinct users**, never a count of corrections. A string that only ever occurs for one user never becomes a global label; it stays that user's override.
- **Overrides apply per user, after shared inference.** Categorization runs in batches that mix users, which lets one merchant string seen by many users be embedded once. Per-user overrides must therefore be keyed by `user_id` and applied to each row after the shared model has scored the batch, never folded into the batch's shared work, or one user's override could reach another user's rows (user story 5).
- **Measure gain on users who didn't correct.** Global gain is measured on users who supplied none of the corrections, as FR-2 separates train and test users. Otherwise it partly measures the correctors' own overrides, or the model memorizing their transactions, rather than generalization. The time-ordered replay handles leakage in time; this handles leakage between users.

**What "good" will mean** (to measure on synthetic users with preference profiles and simulated correction behavior, replayed in time order):

- **Personal accuracy after feedback:** corrections needed until a user's categories match their preferences, and how often a user corrects the same merchant twice.
- **Global gain:** how fast unseen-merchant F1 rises once the first users' corrections arrive (a realistic route to FR-4's target), measured on users who didn't supply them.
- **Isolation:** one user's preferences never change another user's categories.
- **Robustness:** accidental or adversarial corrections don't move the global model.
- **Calibration after retraining:** FR-5's thresholds keep their meaning.

**What v1 already does for it:** labels are a training argument, predictions are stored with `model_version`, retraining and promotion are one command with gates, and the confidence is calibrated per familiarity group. The deferred FR-3 experiment rounds (composition, training regime) are the starting menu for the retraining design.

## Build order and open questions

Modules are built bottom-up so each layer is tested before the next depends on it.

1. Foundations: repo layout, pinned dependencies, configuration, CI.
2. Data generator, data store and data quality checks.
3. Feature pipeline and service interfaces with baseline implementations.
4. Evaluation harness: training and evaluation splits (incl. the `as_of_date` cutoff for goal examples), running on the baselines.
5. Model experiments per problem; record decisions.
6. Tool server with session-scoped identity.
7. Coach agent and coach test suite.
8. Web app: user selection, dashboard, chat.
9. Evaluation report and technical docs.

**Open questions**

- [x] LLM provider: Anthropic (owner, Oct 2, 2026).
- [x] Web framework for v1: server-rendered Python, FastAPI with templates and htmx (owner, Oct 2, 2026; Delivery Plan).
- [ ] Tool server transport for v1: stdio only, or HTTP as well?
- [ ] Feedback and retraining (FR-5, FR-6): the agreement rule for global labels (a minimum of distinct users; single-user strings stay private), retraining cadence, whether "cheap to retrain" joins the model selection criteria, and whether the shipped model trains on injected label noise at all (the promoted categorizer's Travel fallback for unfamiliar merchants is its likely effect; FR-3 Categorization Model Selection). Settled in the FR-5/FR-6 feature design ([Learning from user feedback](#learning-from-user-feedback-direction-not-yet-designed)).
