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
| Data | Feature pipeline | Clean merchant text; build per-user point-in-time history and weekly/monthly aggregates; build merchant profiles (a merchant's typical price and spread across users, as of each month, from ≥ 3 distinct other users; FR-7); build category season profiles (how much more or less people buy in a category in each month of the year, as of each month, the scored user left out exactly, from ≥ 20 other users; FR-8) | Raw transactions | Feature tables per model |
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
| Goal | goal\_id, user\_id, name, target\_amount, created\_date, target\_date (a month end), as\_of\_date, current\_balance (as of as\_of\_date). The dataset holds the generated goals; users' own goals and changes are events in a separate goal store, replayed over them per user and session (FR-10 Savings Goals — Feature Design, §1–2) |
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

- `categorize(transactions) → [{transaction_id, category, confidence, model_version, familiar}]`: batch-first over model-visible transaction rows (`transaction_id, user_id, ts, amount, currency, merchant_raw, channel`), one output row per input in order. Whole rows keep the contract stable whichever features a model uses; a single transaction is a one-row batch. Every service's output is checked against its contract at runtime (FR-3). `familiar` says whether the row's normalized merchant string occurs in the model's training rows; the review policy uses it (FR-5 and FR-6 design, §1)
- `score_transactions(rows) → [{transaction_id, score, is_flagged, reason_code, evidence, model_version}]`: batch-first, like `categorize`, over the outflows of one or more users, each with its full earlier history, plus merchant-profile columns (`scoring_rows`). A charge is scored only against what came before it (point in time): rows in later minutes never change its score, and two identical charges in the same minute count as each other's repeat, since their order is unknowable. `reason_code` is `duplicate`, `amount_unusual` or `new_merchant`, so reason accuracy can be scored; `evidence` holds the numbers the plain-language reason is rendered from (templates, not model or LLM text). A flag without a reason breaks the contract (FR-7)
- `score_periods(rows) → [{period_id, score, is_flagged, evidence, model_version}]`: batch-first, like `score_transactions`, over one row per (user, spending category, complete month) with at least 3 earlier months: the monthly spend and purchase count, their point-in-time history, and the category season profile columns (`scoring_periods`). The category is whatever the caller fills in: true categories in evaluation, the session's categories (with corrections) in serving. A flag must pass two product rules, enforced by the contract: spend ≥ 1.3× the user's average month over the past year, and at least 2 purchases in that average month. Its evidence holds only the user's own numbers (actual, usual, excess, ratio, counts), and its reason is a template over them. The driving transactions are the period's 5 largest charges, added by the tool (FR-8). This replaces the earlier `detect_spikes(user_id, period, granularity)` sketch; v1 is monthly only
- **Sensitivity presets (FR-9):** each alert model's score has three cutoffs. Balanced is the promoted cutoff; Less often and More often flag 0.5× and 2× as many rows after the warm-up, placed once on every user without labels by `sfc-model presets` and committed beside the model's manifest. Flag files store down to More often; in a person's first 90 days the cutoff is the stricter of their level and Balanced
- `forecast_goal(rows) → [{example_id, status, p_goal_met, projected_balance, range_lo, range_hi, gap, extra_per_month, share, share_source, net_next_6, may_draw_down, model_version}]`: batch-first over goal rows (the goal, its goal set and the user's monthly net savings up to `as_of`), so a user's goals and a draft are forecast together over one future. `range_lo`/`range_hi` are the 10th and 90th percentiles: the 80% interval. Status bands are fixed at 0.7 and 0.3; a reached goal sets `may_draw_down`. Serving runs the promoted model on per-user states precomputed nightly (`forecasts.json`), simulating 1,000 paths per request from a seed of (user, `as_of` month, model version) (FR-11 and FR-12 design, §1, §7)

**Tools exposed to assistants** (no user\_id parameter; identity comes from the session)

| Tool | Purpose |
| --- | --- |
| get\_spending\_summary | Totals by category and month for a date range |
| get\_transactions | Filtered, categorized transactions, with each one's review flag and whether the user set its category (FR-5, FR-6) |
| detect\_anomalies | Unusual charges for a period, each with its kind, reason and evidence (FR-7); spending spikes: complete months in the range when a category ran well above the user's usual, each with its reason, size and largest charges, scored on request on the session's categories, and a status when there's too little history or the month isn't over (FR-8). At the person's sensitivity (FR-9), leaving out the alerts they hid, with how many; each alert has a `flag_id` |
| forecast\_goal | Whether a goal is on track: status, chance, likely amount and its 80% range, gap or how far ahead, the monthly top-up that would put it on track, the monthly path, and what the share of savings assumes; a reached goal hides its chance, range and top-up. Without a promoted model, a simple projection of the pace so far (FR-11, FR-12) |
| list\_goals | The user's goals: status (active, reached, ended), months left and the amount needed per month, each running goal's forecast status, chance and likely amount, and the user's median monthly savings |
| check\_goal | Validates a new goal or an edit and states the facts for the setup screen, with the draft's forecast and fit badge once a forecasting model is promoted; writes nothing (FR-10, FR-11) |
| create\_goal, update\_goal, archive\_goal | Change the user's goals. The Goals page applies on submit; the coach and outside assistants get a preview until the user agrees (`confirm`) |
| undo\_goal\_change | Undo a goal's latest change |
| list\_review\_items | Categories the model is unsure about, one item per merchant, most unreviewed spending first (FR-5) |
| resolve\_review\_item | Confirm or correct a review item for every transaction at its merchant; the coach previews a change to more than one transaction until the user agrees (FR-5, FR-6) |
| correct\_category | Change any transaction's category, for its merchant or just that one (FR-6) |
| undo\_correction, list\_corrections | Undo a change; the user's recent changes (FR-6) |
| get\_alert\_settings, set\_alert\_sensitivity | How often alerts are pointed out: Less often, Balanced or More often, half or twice Balanced's alerts after a person's first 90 days; the coach and outside assistants preview a change until the user agrees (FR-9) |
| act\_on\_flag, undo\_flag\_action | Act on an alert: "I recognize this" hides it and later ones for the same reason at the merchant (a possible duplicate only itself), "Not me" marks it with fixed guidance, "Expected, all good" hides a month's spike; undo shows it again. Per person, never the model or anyone else (FR-9) |

All tool outputs are structured JSON with units and currency, so the assistant can quote numbers without doing arithmetic.

## Infrastructure

v1 runs as a single Python deployment on one machine; each component has a named upgrade path for the real-data release.

| Concern | v1 | Upgrade path (v2) |
| --- | --- | --- |
| Language / runtime | Python 3.11, pinned dependencies | Containerized services |
| Data store | SQLite (one file per generated dataset) | Managed Postgres with row-level security |
| Feedback store (FR-5, FR-6, FR-9) | SQLite file the app writes (category corrections, alert settings and flag actions), apart from the read-only data; in the demo, keyed by browser session and reset whenever the container restarts or is redeployed | Postgres with row-level security, keyed by the signed-in user |
| Model artifacts | The repo's `artifacts/` holds each service's `PROMOTED` pointer, promotion log and model manifests, and the alert models' sensitivity presets (`presets.json`, FR-9); model files are attached to GitHub Releases and verified against the committed manifest on first use (FR-3) | Container registry: continuous deployment bakes the promoted model into the serving image |
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
| Categorization | Linear models on n-gram text features · gradient-boosted trees · sentence-embedding classifiers · small fine-tuned transformer | Experiment, incl. the unseen-merchant test. **Decided (Oct 3, 2026, FR-4):** logistic regression over character n-grams and frozen `bge-small-en-v1.5` embeddings, without class weights, trained on clean labels and calibrated per familiarity group. It replaced FR-3's Oct 2 choice (bge-base, trained under the injected noise) (FR-3 Categorization Model Selection; FR-4 Categorization — Round Results) |
| Unusual transactions | Isolation-based ensembles · one-class boundary methods · density methods · robust statistical rules | Experiment on recall at a fixed alert rate. **Decided (Oct 3, 2026, FR-7):** an isolation forest across users on one-sided, point-in-time relative features (merchant z, merchant-profile price ratio, first visit, exact repeat, history rank), cut at precision 0.80 on train users; it beat robust rules and a per-(user, merchant) Student-t at 0.11 flags per user-month (FR-7 Unusual Transactions — Round Results). Reasons come from the kind of charge, rendered from stored evidence |
| Spending spikes | Robust per-user statistics on aggregates · seasonal decomposition residuals · forecast-residual methods | Experiment; theory narrows to seasonality-aware options. **Round run (Oct 4, 2026, validation only; FR-8):** spend-based statistics fail (one-off charges dominate), and seasonal decomposition is ruled out by history length. A forecast-residual method on the month's purchase **count** wins: the expected count is the user's usual count times a season (their own a year earlier, shrunk toward the category profile) times income^β. The negative-binomial tail ranks first at 0.521 recall at 0.035 flags per user-month, tied with the Poisson (0.516), on decision 6's tie-break. Both product rules apply to every model. **Promoted (Oct 4, 2026):** on test users it scored precision 0.705 (0.634–0.770), a thin margin, and recall at the rate 0.521, against the baseline's 0.158. It's served on request; before a model is promoted, the simple count rule serves, labelled as such (FR-8 Spending Spikes — Round Results) |
| Goal forecasting | Additive trend + seasonality models · ARIMA-family models · exponential smoothing | Experiment. **Decided (Oct 4, 2026, FR-11 and FR-12):** simulated paths of each user's monthly net savings: a 24-month level; a month-of-year profile under each persona's prior (the user's own profile shrunk toward it), with each path following one persona drawn by how much the user's own history looks like it; the user's own deviations scaled by a spread tuned on coverage of realized balances. Every goal is arithmetic over the paths. Without a promoted model, naive pace stands behind the same pipeline as a simple projection, with no chance or range (owner decision 10 on #54). It beat a flat level and a damped trend; exponential smoothing lost in round 4. Promoted after a second test scoring (owner decisions 10–13) (FR-11 and FR-12 Goal Forecasting — Round Results) |
| Coach LLM | Hosted models of different size and cost tiers | Experiment on grounding accuracy, rubric score, cost per answer |

Each experiment's setup, results and decision are logged in the evaluation report so the choice is reproducible.

## Evaluation framework design

Every model is scored against planted ground truth and a simple baseline, with one command that regenerates data, retrains and writes the report.

| Problem | Data split | Baseline | Metrics |
| --- | --- | --- | --- |
| Categorization | Stratified 80/20 by transaction within train users (known merchants), plus test users' transactions at holdout merchants never seen in training. Hyperparameters and calibrators come from grouped cross-fitting over merchants within train users | Keyword rules | Macro F1 over the 12 spending categories (Income on its own line), per-class F1, confusion matrix; a merchant-level bootstrap interval for unseen merchants; calibration (Brier, ECE) on every test set. Gates: known ≥ 0.90 and unseen ≥ 0.66 in v1, each above keyword |
| Unusual transactions | Every outflow scored; labels hidden from scorers and used only to place cutoffs and tune parameters within user-grouped folds; validation profiles from train users only; reported on test users | Per-user z-score on amount | Recall at 0.11 flags per post-warm-up user-month (ranking), precision at the run's own cutoff, recall per kind and on `clear` labels, reason accuracy, PR-AUC; user-bootstrap intervals. Gates: precision ≥ 0.70 and recall above the baseline's at the same flag rate |
| Spending spikes | Monthly aggregates of all spend on true categories, after the warm-up; labels hidden; thresholds tuned on train users in user-grouped folds, reported on test users | Per-user mean ± k·std per category, with the product rules every flag obeys (owner, Oct 4, 2026) | Recall at 0.035 flags per user-month (ranking), precision at the run's own cutoff tuned to 0.80; period-level precision and recall (all and `clear` labels); excess coverage of the 5 largest charges; the error of the quoted "usual"; recall on simulated basket-size spikes (the main risk: count models miss them by construction); on test, also on predicted categories. Gates: precision ≥ 0.70 and recall above the baseline's at the same flag rate, and every flag with evidence and valid drivers |
| Goal forecasting | Goals with a known outcome at their `as_of`, from 10 sampler draws per user with targets set from a projection at `as_of` (round 5); scored as tracked and as new; user-grouped folds within train users. Test users are scored for finalists; scoring them again is an owner decision after a root-cause analysis on train and validation data (owner decision 13 on #54) | Naive pace; a flat 50% | Brier per path and persona; met rate per status band; coverage of the 80% range; RMSE of the next 6 months' net savings vs last-month naive and seasonal-naive; the share of realized balances below and above the range (reported); user-bootstrap intervals. Gates: Brier below naive pace and a flat 50% with a track record, below a flat 50% for new goals; bands calibrated within tolerance; coverage 70–90%; RMSE ≤ 0.85 of naive and ≤ seasonal-naive (owner decision 1 on #50) |
| Coach | \~30 scripted questions with expected facts, plus adversarial cases | None | Grounding accuracy, refusal accuracy, rubric score (LLM judge, 1–5) |

**How models are trained, compared and promoted** (built in FR-3; each later model extends it rather than replacing it)

- A generic model interface and registry: a candidate is a class plus a config file. Wrappers (calibration, routing, lookup) compose models through the same interface.
- Composable splitters with leak checks that stop a run before any fit. A task owns each problem's examples, labels, splits, metrics, baseline and gates.
- An experiment runner that logs every run's config, data hash, split hashes, code version and validation metrics to MLflow. A run is skipped on resume only if all of them match.
- **Experiments are compared on validation data.** Each task writes its decision rule before the runs; test sets are scored once, for at most three finalists plus the baselines, enforced in code. Departing from the rule needs a recorded override reason.
- **Comparison runs and shipping twins** (FR-4). Candidates are compared under the injected label noise. Each also trains a twin with the task's shipping params (`label_noise: 0`) written out, and only twins are finalized and promoted. Promotion refuses a config that leaves a shipping param implicit, and records its value.
- **The decision rule** (FR-4 §5):
  - ranking and the F1 tie set come from comparison runs;
  - ties are broken on the twins' validation metrics, the first tie-breaker (calibration) with its own paired tie test, so cost decides between runs tied on both;
  - a twin that beats rank 1's twin on validation stops `finalize` until it's investigated.
- Promotion checks the task's gates in the selection criteria's order, exports the model, records it in a committed promotion log, and is the only way a model reaches callers. Callers load "the promoted model" for a service, never a model class.

**Controls against flattering results**

- Ambiguous merchants and messy merchant text generated into the data.
- Label noise applied by the training pipeline to loaded training data, for runs that compare candidates; shipped models train on clean labels, as shipping twins (owner, Oct 2, 2026). The generator itself only generates.
- Separate seeds for training and test users.
- Anomaly thresholds tuned on train users and reported on test users, never both on the same users.
- Spike metrics scored on true categories, so categorizer errors don't leak into FR-8.
- An oracle with the generator's true expected spend must reach the precision target before a model is judged against it (`validate` gate; FR-2).
- Goal examples use only transactions with `ts <= as_of_date` (the goal's backtest origin). The full ledger covers the target month, so later transactions reveal whether the goal was met. The evaluation harness enforces this when it builds splits (build order step 4).
- Goal targets: stage 9 plants a target as a multiple of the balance actually reached, which leaks the future into the target. FR-11's evaluation goals (since round 5) take their targets from a projection at `as_of` instead; that projection sits near the model's own median, so band calibration is a check around it, not of how people set targets (FR-11 and FR-12 design, Round 5).
- The judge model differs from the coach model, and a sample of judge scores is checked by hand.

## Learning from user feedback

Categorization ships with a cold-start model that only needs to be decent (FR-3). After launch, quality comes from feedback: low-confidence review (FR-5) and user corrections (FR-6), P1 requirements built for the v1 demo. A correction says more than "the model was wrong". It also shows how a user *prefers* to see their money. This section set the direction; the design is FR-5 and FR-6 Review, Corrections and Retraining — Feature Design.

**As built (Oct 2026):**

- **Review:** each promoted categorizer carries a review policy, one threshold per familiarity group chosen on its validation predictions by a written rule; the batch flags spending rows below it.
- **Corrections:** events in a feedback store; overrides are replayed from them and applied per user after the shared predictions (a transaction override, then a merchant override, then the model). Undo marks an event; nothing is deleted. In the demo, feedback is keyed by browser session, since visitors share accounts.
- **Agreement:** a merchant's category becomes a training label when at least N distinct people (3) have a view, two thirds agree, and at least one corrected rather than accepted a suggestion; re-evaluated with every vote. N, the majority and the cadence are provisional.
- **Retraining:** through the evaluation harness's parts, on clean labels: original rows relabelled at agreed merchants, contributors' rows before the cutoff, a leak check; gated on users who supplied no labels, against their own view, with truth-based numbers reported.
- **Measured** in a simulated replay (FR-5 Feedback Replay — Results): new-merchant macro F1 for people who never corrected rose from 0.75 to 0.87 against the true categories, and held at 0.83 with a fifth of correctors acting at random, though 7 of 52 labels held only because of their votes and every promoted model carried 3 to 7 of them; N stays provisional. The demo shows it on "How it learns"; no replay model is promoted, and visitors' agreement is illustrative only (owner, Oct 3, 2026).

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
- [x] Tool server transport for v1: HTTP only, MCP over Streamable HTTP with bearer tokens carrying the user (owner, Oct 2, 2026; Web App UI, decision 8). stdio isn't part of v1.
- [x] Feedback and retraining (FR-5, FR-6): designed and built (FR-5 and FR-6 feature design). Settled: the agreement rule's shape (distinct users, a two-thirds majority, at least one correction; single-user strings stay private) and clean labels for shipped and retrained models (owner, Oct 2, 2026). Still provisional after the replay: N, the majority and the retraining cadence; deferred: "cheap to retrain" as a selection criterion.
