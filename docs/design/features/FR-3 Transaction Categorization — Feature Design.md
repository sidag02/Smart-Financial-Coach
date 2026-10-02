# FR-3 Transaction Categorization — Feature Design

Oct 1, 2026 (accepted Oct 2) · @Sidd · Status: **Accepted** · PRs #6–#14

## Summary

This feature builds the categorization service: every transaction gets exactly one category from the fixed taxonomy, with a confidence, behind a contract that later model experiments can swap without touching callers.

- **Requirement:** FR-3 (P0): *"Assign every transaction one category from a fixed set of about 12."* Success metric (PRD): macro F1 ≥ 0.90 on known merchants.
- **Neighbour:** FR-4 (P0, *"categorize merchants the system has never seen before"*, macro F1 ≥ 0.80) runs on the same service. This design builds the service and measures FR-4 from day one, but leaves closing the FR-4 gap to its own design (see [Scope](#scope-fr-3-versus-fr-4)).
- **Feasibility, measured on the default dataset** ([evidence](#feasibility)): a plain linear model on character n-grams already reaches **0.99** macro F1 on known merchants. On unseen merchants it reaches **0.50**. Pretrained sentence embeddings lift unseen merchants to **0.64** (95% interval 0.56–0.73), alone or combined with n-grams. The combination keeps known merchants at 0.99. FR-3's target is easy; FR-4's is not, and even the interval's upper end is below 0.80.
- **Approach:**
  1. A **model training and evaluation framework** (the Technical Design's evaluation harness, build order step 4): a generic model interface and registry, composable splitters with leak checks, an experiment runner tracked in **MLflow**, and a promotion step that is the only way a model reaches callers. FR-3 builds what categorization needs and states how later features extend it ([Framework](#training-and-evaluation-framework)).
  2. A batch-first **service contract** that takes model-visible transaction rows and returns category, confidence and model version. Callers load "the promoted categorizer", never a specific model class.
  3. A shared **merchant-text normalizer** in the feature pipeline, reused later by the anomaly service.
  4. A **keyword baseline** built from generic words only, never merchant names.
  5. A **linear text categorizer** (character n-grams + embeddings + amount, sign and channel) as the starting candidate, ported from the POC.
  6. An **experiment programme** run on the framework: baselines, feature and classifier variants, composition and calibration, selected by a rule written down before the runs ([Experiment plan](#experiment-plan)). The winner is promoted, and FR-3 Categorization Model Selection records the decision.
- **Principle (carried from FR-2):** the data is not tuned to make targets pass. The ambiguous-merchant ceiling is reported, not trained around.
- **Principle (new):** models are chosen on validation data. Test sets are scored once, for the finalists, so running many experiments can't select a model that won by luck.

## Context

Categorization is the first model in the system, and its outputs feed nearly everything after it.

| Consumer | What it needs from categorization |
| --- | --- |
| Dashboard (FR-17) | A category on every transaction, for spend by category and trend |
| Coach tools (FR-13, FR-14) | `get_spending_summary` and `get_transactions` group by predicted category |
| Unusual transactions (FR-7) | A shared merchant identity (normalized text) for "new merchant" and duplicate features |
| Low-confidence review (FR-5, P1) | A confidence that means what it says (calibrated) |
| Category correction (FR-6, P1) | Corrections become new labels, so training must take labels as an argument rather than read them from a fixed table |
| Spending spikes (FR-8) | **Not** a consumer for evaluation: spike metrics are scored on true categories (FR-2), so categorizer errors don't leak into FR-8 |

What the data looks like (default spec, 360 users, 1.1M transactions):

| Fact | Value | Why it matters |
| --- | --- | --- |
| Class sizes | Dining 418k … Travel 2.9k transactions | Macro F1 weighs Travel as much as Dining, so class weighting matters |
| Distinct `merchant_raw` strings | 194k for 324 merchants | Rendering is messy; normalization collapses them to 12.8k |
| Positive spending amounts | 5,422 Shopping refunds | Sign alone doesn't mean Income |
| Ambiguous merchants | 5 (warehouse clubs, marketplaces) | 1.3% of spending transactions can't be categorized from text and amount alone (FR-2 §6) |
| Holdout merchants | 58, used only by test users | The FR-4 test set: 78k transactions, but only 3–10 merchants per class |

## Scope: FR-3 versus FR-4

Both requirements use one service and one model, so the line is drawn by what each design *gates on*.

| | FR-3 (this design) | FR-4 (next design) |
| --- | --- | --- |
| Builds | Framework, contract, normalizer, baseline, candidate models, experiments, promoted model, CLI | Whatever closes the unseen-merchant gap, as experiments on the same framework (see [Decisions and open questions](#decisions-and-open-questions)) |
| Reports | Known-merchant, all-test-user and unseen-merchant macro F1 | Same report |
| Gates on | Known-merchant macro F1 ≥ 0.90 and beats the baseline | Unseen-merchant macro F1 ≥ 0.80 |

## Feasibility

**Evidence:** POC branch `poc/fr-3-categorization`. All links here are pinned to commit [`55d4197`](https://github.com/sidag02/Smart-Financial-Coach/tree/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization), so they can't change. `feasibility.py` trains and scores the models, and `plots.py` draws the charts. Every number below is in the [results report](https://github.com/sidag02/Smart-Financial-Coach/blob/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization/results/feasibility.md). `feasibility.json` adds per-class precision and recall, confusion matrices, per-merchant accuracy and calibration. There is also a [full-page screenshot](https://github.com/sidag02/Smart-Financial-Coach/blob/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization/results/screenshots/report.png) of the HTML report and each [chart](https://github.com/sidag02/Smart-Financial-Coach/tree/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization/results/figures). The run used the default dataset (spec hash `130e78f55383`) and seed 0. A second run gives identical metrics.

Setup: the Technical Design's splits, no label noise, no calibration. The model is logistic regression with balanced class weights, trained on 194k train-user transactions (at most 20k per class). Macro F1 is over exactly the 12 spending categories.

**How the regularization strength was chosen.** C was set by hand before the first run, at 10 for n-grams (many sparse features) and 3 for the embedding variants (384 dense ones). It was not tuned on test results or anything else. A sensitivity check afterwards, at C = 1, 3 and 10, was not used to choose C. It moves unseen-merchant F1 by at most 0.03 and known-merchant F1 by at most 0.002, and changes no conclusion. If anything, C = 10 slightly understates n-grams on unseen merchants (0.504 vs 0.531 at C = 1–3). The design chooses C on cross-fitted validation predictions (§5). The unseen-merchant interval comes from 1,000 bootstrap resamples of merchants within each category (see [§6](#6-evaluation)).

| Features | Known merchants (145k txns) | All test users (375k) | Unseen merchants (78k) | Unseen, 95% interval |
| --- | --- | --- | --- | --- |
| Character 2–4-grams of `merchant_raw` + amount bin, sign, channel, hour | 0.986 | 0.868 | 0.504 | 0.44–0.56 |
| Sentence embedding of normalized text (`bge-small-en-v1.5`, 384-d) + same side features | 0.973 | 0.913 | **0.643** | 0.55–0.73 |
| Both | **0.988** | **0.929** | 0.638 | 0.56–0.73 |

*Corrected in review:* an earlier version showed 0.842 and 0.593 for the embeddings row. Macro F1 had averaged over a 13th label (Income, F1 0) after one spending transaction was predicted as Income. The POC now passes the 12 spending labels explicitly, and no other row changed.

![Macro F1 by test set](https://github.com/sidag02/Smart-Financial-Coach/blob/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization/results/figures/headline.png?raw=true)

No bootstrap resample of any model reaches 0.80; the best of 1,000 is 0.793.

![Unseen-merchant bootstrap distribution](https://github.com/sidag02/Smart-Financial-Coach/blob/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization/results/figures/unseen_bootstrap.png?raw=true)

Unseen-merchant accuracy by catalog scope (transaction-weighted):

| Scope | Merchants | Transactions | N-grams | Embeddings | Both |
| --- | --- | --- | --- | --- | --- |
| National (real chains) | 25 | 34.0k | 0.46 | 0.62 | 0.69 |
| Local (fictional) | 19 | 26.7k | 0.72 | 0.82 | 0.80 |
| Online services | 14 | 17.4k | 0.43 | 0.93 | 0.87 |

![Accuracy by merchant type](https://github.com/sidag02/Smart-Financial-Coach/blob/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization/results/figures/unseen_by_scope.png?raw=true)

What the errors show:

- **N-grams can't name what an unseen brand sells.** N-gram accuracy is 0.11 on Taco Bell, 0.09 on McDonald's, 0.01 on Caviar and 0.53 on Walmart.com. Embeddings get all four right: 1.00, 0.47, 1.00 and 1.00. That knowledge comes from outside the training data.
- **Some merchants defeat both.** Arco (0.01), Rite Aid (0.01), Giant Eagle (0.00), Alamo Drafthouse (0.00) and Sittercity (0.03) are wrong under every feature set. A small embedding model doesn't know these brands well enough. These five account for 6.8k unseen transactions.
- **On unseen merchants, combining features doesn't help overall.** Embeddings alone tie the combination (0.643 vs 0.638). By scope, embeddings alone are ahead on online services (0.93 vs 0.87) and local names (0.82 vs 0.80), and behind on real chains (0.62 vs 0.69). On Caviar the combination drops from 1.00 to 0.64, and on Corner Fresh Grocery from 0.86 to 0.04: n-grams memorized from training merchants outvote the embedding when a string is unfamiliar. The combination still wins on known merchants and all test users, so it stays the FR-3 default. FR-4 should try routing by familiarity.
- **Descriptive local names are easy.** Driftwood Coffee Co, Golden Hour Bakery and Little Owl Espresso are 1.00 under every feature set.
- **Housing, Insurance & Fees, Utilities and Subscriptions transfer well.** Per-class F1 on unseen merchants is 0.91–1.00 with both feature sets, because their text ("APARTMENTS", "AUTOPAY", "FEE") and the `ach` channel are shared across merchants.
- **Income is trivial.** Its F1 is 1.000 on known merchants under every feature set, which is why it's left out of the headline ([§4](#4-income)).
- **Confidence is calibrated on known merchants but over-confident on unseen ones.** The model is uncalibrated in the POC. The ranking is still useful: confidence ≥ 0.9 is at least 96.5% accurate on unseen merchants. This drives the calibration design in §3 and §5.

  | Features | Calibration error (ECE), known | ECE, unseen | Unseen: accuracy at confidence ≥ 0.9 | Unseen: share of rows at ≥ 0.9 |
  | --- | --- | --- | --- | --- |
  | N-grams | 0.008 | 0.160 | 0.965 | 32% |
  | Embeddings | 0.012 | 0.100 | 0.967 | 61% |
  | Both | 0.009 | 0.056 | 0.992 | 39% |

  ![Calibration](https://github.com/sidag02/Smart-Financial-Coach/blob/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization/results/figures/calibration.png?raw=true)

- **Embedding cost is negligible.** 12.8k unique normalized strings embed in about 10 s on a laptop CPU through `fastembed` (ONNX, no PyTorch). The file `fastembed` actually downloads is Qdrant's quantized ONNX export, `Qdrant/bge-small-en-v1.5-onnx-Q` at revision `aa8f8b060edb`, not the original model, so results depend on that exact file (see [Model artifact](#model-artifact)).

## Goals and non-goals

**Goals**

1. Every transaction gets one of the 12 spending categories or Income, plus a confidence in [0, 1].
2. Known-merchant macro F1 ≥ 0.90, above the keyword baseline, reproducible on rerun (NFR-8).
3. Unseen-merchant macro F1 is measured and reported with a confidence interval from the first commit, so FR-4 starts from a number.
4. Confidence is calibrated on **unseen as well as known** merchants, well enough for FR-5 to threshold it later. Expected calibration error is reported on every test set and for seen and unseen strings separately. The POC's uncalibrated model is at 0.009 on known merchants and 0.056 on unseen ones.
5. Model code never reads ground truth. Labels reach the model only as a training argument.
6. Serving cost is measured and reported (batch time for 10k transactions, single-call p95), not gated: categorization runs in batches on ingestion, so cost is a cluster-sizing question ([Metrics and why](#metrics-and-why), point 7).
7. Swapping the categorization model is a promotion, not a code change: callers load the promoted model through one function.
8. Every experiment is tracked (config, data, splits, code version, metrics) and can be rerun to the same numbers.

**Non-goals**

- Reaching 0.80 on unseen merchants (FR-4).
- Low-confidence review, user corrections, per-user overrides (FR-5, FR-6; P1).
- Tasks, splitters and metrics for the other models (anomalies, spikes, forecasting). The interfaces are checked against them, and [Path to a general framework](#path-to-a-general-framework) says how each feature adds its own, but FR-3 doesn't build them.
- Hosting an MLflow server. FR-3 uses a local tracking store; a shared server is a configuration change (see [option F](#f-experiment-tracking)).
- Persisting predictions inside the generator's SQLite file (see [option C](#c-where-predictions-live)).

## What it will look like

### Usage

```sh
# Data scientist: run experiments, compare them in MLflow, promote one
uv run sfc-experiment run configs/experiments/categorization/linear_both.yaml --data data/synthetic/default.sqlite
uv run sfc-experiment run configs/experiments/categorization/          # every config in a folder
uv run sfc-experiment finalize --task categorization --runs <run_id> <run_id>   # score finalists on test, once
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
uv run sfc-model promote --task categorization --run <run_id>

# Pipeline: categorize with whatever is promoted
uv run sfc-model predict --task categorization --data data/synthetic/default.sqlite --out data/predictions/default.sqlite
```

```python
from smart_financial_coach.data.store import load_transactions
from smart_financial_coach.intelligence.service import load_service

categorizer = load_service("categorization")  # the promoted model, whichever it is
txns = load_transactions("data/synthetic/default.sqlite", user_id="u_te_yp_0007")
categorizer.categorize(txns)  # -> DataFrame[transaction_id, category, confidence, model_version]
```

Promoting a different run changes what `load_service` returns. No calling code changes.

### Service contract

The Technical Design fixes `categorize(merchant_raw, amount) → {category, confidence}`. This design proposes widening it to whole transaction rows, batch-first:

```python
class Categorizer(Model, Protocol):  # Model: the framework's generic interface
    categories: tuple[str, ...]  # 12 spending categories + Income, from meta

    def categorize(self, transactions: pd.DataFrame) -> pd.DataFrame: ...

    #   in:  transaction_id, user_id, ts, amount, currency, merchant_raw, channel
    #   out: transaction_id, category, confidence, model_version
```

- **Contract enforced, not trusted:** every categorizer is served through a wrapper that checks the output: one row per input, in input order, a known category, confidence in [0, 1]. A new model that breaks the contract fails in tests and at promotion, not in the dashboard.
- **Why whole rows:** amount sign and channel were measurably useful in feasibility, and the hour may be. Passing the row keeps the contract stable whichever features win, which is the Technical Design's reason for fixing interfaces before models.
- **Why batch-first:** the dashboard categorizes thousands of rows at once and the vectorizers are batch operations. A single transaction is a one-row frame.
- **`fit` takes labels as an argument:** the model never knows where labels come from, whether truth tables now or user corrections (FR-6) later. The training script under `evaluation/` reads truth; `intelligence/` doesn't (FR-2 isolation test).

### Model output table

Predictions are written to a separate SQLite file, not the generator's file (option C):

**transaction_categories**

| Column | Type | Notes |
| --- | --- | --- |
| transaction_id | TEXT PK | |
| user_id | TEXT | Indexed, for user-scoped reads by the data-access layer |
| category | TEXT | One of `meta.categories` or `Income` |
| confidence | REAL | Calibrated probability of the predicted class |
| model_version | TEXT | Artifact version that produced it |

### Model artifact

Every run logs its fitted model to MLflow. Promotion copies the chosen one to `artifacts/categorization/<version>/` and points `artifacts/categorization/PROMOTED` at it; serving reads only that folder (see [Promotion and serving](#promotion-and-serving)).

The folder holds the fitted model (`model.joblib`) and `manifest.json`: registry name, params, config hash, training data `spec_hash`, git commit, seeds, split hashes, MLflow run ID, metrics, the model file's SHA-256, and the embedding file's source repository, revision and SHA-256. `fastembed` downloads the model at run time, so loading refuses to run if the cached file's checksum differs from the manifest. The version is `<config hash>-<data hash>-<code version>` (first 8 characters each), so retraining after a code fix yields a new version and release tag rather than a second file under the old name. The code version was added after the first promotion (review on PR #12): `3f0ccc82-2f0e60a6`, promoted before the change, keeps its two-part name, since renaming it would change the file `finalize` scored. `promote` and the release step refuse a version that already exists with a different file. Only artifacts from this folder are loaded, since joblib files are pickles.

**Where the files live (decision Oct 2, 2026).** The manifest is committed; `model.joblib` is not. Saved models are 9.5–12.7 MB, above the repo's 5 MB file limit, and every retraining would add one to git history for good. Each promoted model file is attached to a **GitHub Release** of this repository, tagged `categorization-<version>`. The repository is public, so anyone can download it with a plain URL and no credentials. The committed manifest's `model_sha256` is the trust anchor: a downloaded file is unpickled only if it matches, so a replaced release asset is refused rather than run.

## Training and evaluation framework

Categorization is the first model, but the Technical Design has four learning problems, and each splits its data differently. FR-3 therefore builds the harness as small, composable pieces: what categorization needs is implemented now, and the interfaces are checked against the other problems so later features extend it rather than replace it.

| Problem | How it splits | Labels in training |
| --- | --- | --- |
| Categorization (FR-3, FR-4) | Train vs test users; stratified rows within train users; holdout merchants; merchant-grouped folds | Yes |
| Unusual transactions (FR-7) | Thresholds tuned on train users, reported on test users | No; labels only score |
| Spending spikes (FR-8) | Same, on monthly per-category aggregates | No; labels only score |
| Goal forecasting | Rolling origin: train on months 1..k, predict k+1..k+3, with an `as_of_date` cutoff | Future values |

### Pieces and responsibilities

| Piece | Lives in | Responsibility | Reads truth |
| --- | --- | --- | --- |
| `Model` protocol, registry, wrappers | `intelligence/models/` | What every model implements; build a model from `{type, params}`; compose models | No |
| Service contracts (`Categorizer`, later `AnomalyScorer`, `Forecaster`) | `intelligence/<service>/` | Fixed input and output schema per service, checked at runtime | No |
| `load_service` | `intelligence/service.py` | Load the promoted model for a service from `artifacts/` | No |
| `Task` | `evaluation/tasks/` | Load examples, attach labels, define default splits, metrics, baseline and gates | Yes |
| Splitters and leak checks | `evaluation/splits/` | Turn examples into named ID sets and validation folds; fail on leaks | Yes (split definitions) |
| Metrics | `evaluation/metrics/` | Pure functions over predictions and labels | No |
| Runner | `evaluation/runner.py` | One experiment: split, tune on folds, fit, validate, log to MLflow | Yes |
| Finalize and promote | `evaluation/promote.py` | Score finalists on test once; check gates; register and export the winner | Yes |

The truth boundary doesn't move: `intelligence/` never reads truth and never imports `mlflow`. The isolation test is extended to check both, and to cover `data/store.py` and `data/features/`.

### Model interface

```python
class Model(Protocol):
    name: ClassVar[str]  # registry key, e.g. "categorization/linear_text"
    version: str

    @property
    def params(self) -> Mapping[str, Any]: ...

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self: ...

    def predict(self, x: pd.DataFrame) -> pd.DataFrame: ...
```

- **Persistence is not the model's job:** `artifact.py` saves every model the same way (joblib + manifest + checksum). A model holding something that can't be pickled, such as an ONNX session, drops it in `__getstate__` and reopens it on first use.
- **`y` is optional** because anomaly and spike models are unsupervised; categorization always passes it.
- **Registry:** implementations register under a name (`@register("categorization/linear_text")`) and are built from config, so a new candidate is a class plus a YAML file. The runner, MLflow logging and promotion don't change.
- **Composition:** wrappers implement the same protocol, so FR-4's ideas are configurations, not new plumbing:
  - `Calibrated(base, by="familiarity", method=...)`: calibrates confidence per group (§3). It maps only the predicted class's probability, so the predicted category never changes.
  - `Routed(seen=..., unseen=...)`: n-grams for familiar strings, embeddings for unfamiliar ones.
  - `Lookup(fallback=...)`: exact normalized-string lookup in front of a model.

### Tasks and splits

A `Task` owns everything problem-specific, so the runner stays generic. The categorization task:

- **Examples:** model-visible transaction rows from `data/store.py`, plus `merchant_id`, `holdout` and the true category attached on the evaluation side.
- **Splits** (declared in the experiment config, defaults from the Technical Design):

  | Set | Splitter | Used for |
  | --- | --- | --- |
  | `train` | `by_user_split(train)` minus `test_known` | Fitting, and the validation folds inside it |
  | `test_known` | `stratified_rows(0.2, by=category)` within train users | Known-merchant gate, finalists only |
  | `test_all` | `by_user_split(test)` | All test users, finalists only |
  | `test_unseen` | `test_all` ∩ `membership(holdout merchants)` | Unseen merchants, finalists only |
  | Validation folds | `group_kfold(merchant_id, k=5, stratify=category, eligible=holdout-eligible merchants)` over `train`, plus a seeded 10% seen-merchant row sample per fold (§5) | Choosing hyperparameters, fitting calibrators, **comparing experiments** |

- **Validation-unseen matches `test_unseen`'s population.** The catalog's holdout rule never picks some merchants: each category's most popular ones, the top merchant of each subtype, and merchants in excluded categories. On the default spec these protected merchants are 64 of train users' 233 spending merchants and carry 34% of their spending transactions (87% of Health & Fitness). None of them can ever be in `test_unseen`. They are mostly big chains, which pretrained embeddings know best, so letting them rotate through validation would favor some candidates. So they stay in every fold's training rows, and only the ~169 holdout-eligible merchants are held out, still about 3× the 58 in `test_unseen`. The eligible set comes from the generator's own holdout rule, factored into one function that both the generator and the task call, so the two can't drift apart. The dataset doesn't currently record it (`truth_merchants` has no popularity, and `meta` doesn't hold the holdout settings), so the generator writes it as a new `truth_merchants.holdout_eligible` column (schema version 3; existing datasets must be regenerated).

- **Leak checks** run on every split, before any fit: no ID in two sets; no fold's held-out merchant in that fold's training rows; no protected merchant in any fold's held-out group; `test_unseen` only holdout merchants; label noise only on rows a model trains on. A failure stops the run.
- **Split hashes** (a hash of each set's sorted IDs) are logged with every run, so two runs can be compared only if they used the same splits. The runner refuses to put runs with different split hashes on one leaderboard.

### Experiment runner and tracking (MLflow)

One experiment is one YAML file: task, data, splits, model `{type, params}`, a hyperparameter grid, seeds, label noise. `sfc-experiment run` does, in order:

1. Load examples and labels through the task; build and check the splits.
2. For each grid point, fit on each validation fold and pool the out-of-fold predictions (§5). Choose hyperparameters by the task's validation metric.
3. Fit wrapper stages that need held-out predictions (calibrators) on the pooled predictions.
4. Fit the final model on all of `train` with the chosen hyperparameters.
5. Log to MLflow and stop. **Test sets are not scored here.**

| MLflow | What goes there |
| --- | --- |
| Experiment | One per task: `categorization` |
| Run | One per experiment config; nested child runs per grid point |
| Params | The flattened config: model type and params, features, grid, seeds, noise rate |
| Tags | Git commit, data `spec_hash`, split hashes, config hash, task, registry name |
| Metrics | `val_*` (validation, every run); `test_*` (finalists only, written by `finalize`); fit and predict timings |
| Artifacts | Fitted model, manifest, validation report (per-class F1, confusion matrices, per-merchant accuracy, calibration curves) |

- **Tracking store:** local by default (`sqlite:///mlruns/mlflow.db`, artifacts under `mlruns/`, both gitignored). `SFC_MLFLOW_TRACKING_URI` points it at a shared server instead; nothing else changes.
- **Trust:** `finalize` and `promote` unpickle models downloaded from the tracking store. The manifest checksum shows a file arrived intact, not who wrote it, so anyone who can write to a shared server's artifact store could run code on the machine that runs those commands. The local default is safe; a shared server must allow only trusted writers.
- **Caching:** embeddings are cached per normalized string and keyed by the embedding file's checksum, so a sweep embeds once.
- **Resumable:** a config is skipped only if a finished run has the same config hash, **data content hash** (a hash of every loaded row, not just the spec and version strings), split hash and **code version** (a hash of everything that can change results, `src/`, `configs/models/`, `configs/data/`, `pyproject.toml` and `uv.lock`, as it is now: committed, modified or untracked; docs, tests, CI and new experiment configs don't change it, since each run's own config is in its config hash), unless `--force`. A model that reads a file names its content in its params (e.g. the keyword baseline's required `keywords_sha256`), so editing the file changes the config hash. The leaderboard compares runs on the same data and splits, warns when they come from different code versions, and `finalize` refuses such a mix without an override reason. A missing required baseline (keyword) is named rather than leaving every candidate silently ineligible.

### Selection discipline

With 58 unseen merchants and a 0.17-wide interval, enough experiments will find one that wins by chance. Three rules prevent that:

1. **Compare on validation only.** Merchant-grouped folds stand in for unseen merchants; the seen-merchant fold rows stand in for known ones. Every run gets `val_known_macro_f1`, `val_unseen_macro_f1` and `val_unseen_ece`.
2. **Write the decision rule before the runs** ([Experiment plan](#experiment-plan)).
3. **Score test sets once, for finalists, enforced in code.** `sfc-experiment finalize` takes the leaderboard's top three itself (tagging each with its rank) and refuses a split whose candidates were already test-scored. It scores them on `test_known`, `test_all` and `test_unseen`, tags them as finalists, and records that the test sets were used. The task's round 0 baselines are always scored alongside them and don't count toward the three, so the "beats the baseline" gate has test numbers to compare against. The test numbers are reported, not used to choose among the finalists again. The one exemption is the POC reproduction check (§3).

### Promotion and serving

`sfc-model promote --task categorization --run <id>`:

1. Requires the run to be the rank-1 finalist: finalists are never re-ranked on test scores.
2. Checks the task's gates in the Technical Design's order: primary metric on held-out data (FR-3: `test_known` ≥ 0.90), beats the baseline, and a written note on explainability and operations (`--note`, stored on the run).
3. Registers the model in the MLflow model registry as `categorization` and moves the `champion` alias to it.
4. Exports it to `artifacts/categorization/<version>/` and writes `artifacts/categorization/PROMOTED`, which holds the version.
5. Uploads `model.joblib` to a GitHub Release tagged `categorization-<version>` (through the `gh` CLI, so it needs a login with write access to the repository). The manifest and pointer stay in the repo; the model file is git-ignored.
6. Appends a line to the committed `artifacts/categorization/promotions.jsonl`: version, MLflow run ID, date, the gate results, the note and the model file's download URL. With the default local tracking store, the registry lives in a gitignored database on one machine, so this log is the promotion history every clone can see. Tests check that `PROMOTED` matches the log's last entry; the check against the MLflow alias runs only where that tracking store is reachable.

The promotion is complete when the pointer, manifest and log are committed and merged. Until then, other clones still serve the previous version.

**Departing from the rule is possible but visible.** Naming finalists, a second round of test scoring on the same splits, or promoting a finalist other than #1 each need `--override "<reason>"`. The reason is recorded as a tag on every run it touches and in `promotions.jsonl`.

`load_service("categorization")` reads `PROMOTED`, loads that folder, verifies the manifest and the embedding checksum, and returns the model wrapped in its contract check. If the folder has no `model.joblib` yet, it first downloads the file from the URL in the promotion log and checks it against the committed manifest before loading, the same pattern as the embedding file. Each download goes to its own temporary file in that folder and is renamed into place only once verified, so workers starting together race harmlessly and never read a partial file. So a fresh clone runs with `uv sync` and `sfc-model predict`; the first call fetches the model.

**Interim limits of first-use download**, until continuous deployment bakes the model into the serving image (option G):

- A fresh worker needs **GitHub to be reachable** at cold start. An outage blocks new pods, not running ones, which already hold the file.
- The version folder must be **writable**, which a read-only container filesystem isn't.

**Serving never talks to MLflow:** the dashboard and coach depend only on files, so an MLflow outage or a missing server can't take categorization down (NFR-6's spirit), and the Technical Design's in-process hosting choice holds. Rolling back is promoting the previous run.

### Path to a general framework

FR-3 builds only what categorization needs. This is how each later feature extends it, and what it must not change:

| Piece | Built in FR-3 | Extended later by | Extension |
| --- | --- | --- | --- |
| `Model` protocol, registry, `load_service`, promotion | Complete | — | Used as is; each service adds a contract |
| Service contracts | `Categorizer` | FR-7, FR-8, forecasting | `AnomalyScorer` (score, flag, reason), `SpikeDetector`, `Forecaster` (forecast, interval, P(goal met)) |
| Example unit | Transaction rows | FR-8 | Aggregated examples `(user, category, month)`; splitters already work on example IDs and group columns, not on transactions |
| Splitters | `by_user_split`, `stratified_rows`, `group_kfold`, `membership` | FR-7, FR-8: `by_user_split` with a `tune` set. Forecasting: `rolling_origin(k, horizon)` | New splitter classes; the `Splits` structure (named sets + folds) is unchanged |
| Leak checks | Set disjointness, group separation, noise scope | Forecasting | Time cutoff: no example sees data after its `as_of_date` |
| Metrics | Macro F1, per-class F1, confusion, merchant bootstrap, ECE | FR-7, FR-8, forecasting | Precision/recall/PR-AUC and precision at an alert rate (using `Truth.score_transactions`), period scoring (`Truth.score_periods`), RMSE/MAPE/Brier |
| Validation | Grouped cross-fitting | FR-7, FR-8 | Threshold tuning on train users; the "compare on validation" rule applies unchanged |
| Wrappers | `Calibrated`, `Routed`, `Lookup` | As needed | `Calibrated` is classification-only; others are generic |
| Report | Categorization report | Each feature | One report section per task; the harness's "one command" report (Technical Design) concatenates them |

**Rule for later features:** extend by adding a task, splitter, metric or contract. If a feature needs to change the `Model` protocol, the `Splits` structure, the runner or promotion, that is a design change to this section, reviewed on its own.

## Design

### 1. Merchant-text normalization

A pure function in the feature pipeline, `normalize_merchant(merchant_raw) -> str`, shared with the anomaly service:

- Collapse whitespace first (feeds double spaces, even inside `ACH  DEBIT`), then strip processor and channel prefixes (`SQ *`, `TST*`, `CLV*`, `PAYPAL *`, `SP *`, `POS DEBIT`, `ACH DEBIT`, `ACH CREDIT`) **repeatedly**, because they stack (`POS DEBIT SQ *`).
- Drop reference codes after `*`, store numbers (and anything after them), the `PPD ID:` label and other digit runs.
- Lower-case. Trailing locations are otherwise kept, since truncation makes them unreliable to strip and n-grams tolerate them. If nothing is left, the raw text is kept, so the result is never empty.
- **Fixed from the POC:** the POC stripped one prefix, so stacked prefixes collapsed to the processor's name (`POS DEBIT SQ *PHO SAIGON` → `sq`), and it missed `CLV*`. That affected 3.1% of the default dataset's transactions (34k), losing their merchant identity for the embedding. `normalize_merchant_poc` keeps the POC's behavior only for the reproduction check (§3).

The prefix list is generic bank-feed vocabulary, written by hand from the rendering rules' *types*. It is not read from the catalog, so no holdout names leak in. In feasibility it collapsed 194k raw strings to 12.8k.

The raw text keeps its own feature (character n-grams on `merchant_raw`), because prefixes carry signal: `TST*` is Toast, a restaurant point-of-sale system.

### 2. Keyword baseline

The Technical Design's baseline is keyword rules. To be a fair baseline, not a leak:

- A committed `configs/models/category_keywords.yaml` maps generic words to categories (`coffee`, `cafe`, `grill`, `pizza` → Dining; `airlines`, `hotel` → Travel; `insurance`, `fee` → Insurance & Fees; `electric`, `water`, `wireless` → Utilities …).
- **No merchant names**, so the baseline gets no unfair advantage from the catalog. A unit test fails if any keyword is a token of a catalog name and isn't on the test's reviewed allowlist of generic words (`coffee`, `market`, `parking` …).
- The category with the most keyword matches wins, so a refund at a shop keeps its category. Positive amounts with no keyword match → Income. Anything else unmatched → the training set's most frequent spending category (Dining).
- Confidence is the training precision of the rule that fired, as for the other round 0 baselines (majority class; exact normalized-string lookup).
- "Beats the baseline" means beating the **keyword** baseline. Lookup is near-perfect on known merchants (0.98 on validation) and near-useless on unseen ones (0.08), so it's a reference point, not the bar.

### 3. Starting candidate: linear text categorizer

The feasibility "both" configuration, made into a reproducible scikit-learn pipeline and registered as `categorization/linear_text`. Its feature blocks are config switches, so the n-grams-only and embeddings-only variants are the same class:

| Feature block | Source | Notes |
| --- | --- | --- |
| Character 2–4-grams (`char_wb`), TF-IDF, sublinear | `merchant_raw` | Fitted on training rows only |
| Sentence embedding (384-d) | `normalize_merchant(merchant_raw)` | Pretrained, frozen; cached per unique normalized string |
| Amount | `log1p(abs(amount))`, binned | Bins, not a raw number, so the linear model can learn non-monotone price bands |
| Sign | `amount > 0` | Separates Income from refunds together with text |
| Channel | one-hot | `card_present`, `online`, `ach`, `other` |
| Hour of day | 3-hour bins of local `ts` | Kept only if the ablation shows it helps (milestone 3) |

- **Classifier:** multinomial logistic regression with balanced class weights. C is chosen from {1, 3, 10} on cross-fitted validation predictions (§5).
- **Confidence:** the predicted class's probability after calibration, done separately for **seen** and **unseen** strings. A string is seen if its normalized form occurs in the training rows. The vocabulary is built from all training rows *before* the per-class cap, so a known merchant's string isn't treated as unseen just because the cap sampled it out. In the POC split, that cuts known-merchant test rows wrongly treated as unseen from 1.3% to 0.9%, and test users' rows at known merchants from 2.6% to 1.8% (measured in review). Familiarity is model-visible, so it works the same way in production. Each familiarity group gets its own calibrator, fitted on the cross-fitted predictions in §5. The method (none, temperature or isotonic) is chosen per group by cross-validated Brier score, not ECE; see [Metrics and why](#metrics-and-why), point 5.
- **Why split calibration by familiarity:** in feasibility the model is already calibrated on known merchants (ECE 0.009). A calibrator fitted on known merchants would change almost nothing there and leave unseen merchants over-confident (ECE up to 0.16). That is the case FR-5's low-confidence review exists for.
- **Why start here:** it meets the FR-3 target by a wide margin in feasibility, trains in seconds on CPU, and stays explainable (top n-grams per class). It is the candidate to beat, not a foregone conclusion: the [experiment plan](#experiment-plan) decides what is promoted.
- **First check of the framework (milestone 3, done).** With label noise off, calibration off, C fixed as in the POC and the POC's normalizer:

  | Configuration | Known merchants | All test users | Unseen merchants | Unseen, 95% interval |
  | --- | --- | --- | --- | --- |
  | N-grams | 0.988 (POC 0.986) | 0.877 (0.868) | 0.511 (0.504) | 0.45–0.58 |
  | Embeddings | 0.972 (0.973) | 0.912 (0.913) | 0.630 (0.643) | 0.56–0.70 |
  | Both | 0.989 (0.988) | 0.932 (0.929) | 0.649 (0.638) | 0.57–0.74 |

  - **The port is exact.** On the POC's own split (`train_test_split` and its cap), with the POC's embedding batching, the n-gram and combined models reproduce the POC to three decimals. The remaining differences above come from the framework's split drawing different training and test rows.
  - **Unseen-merchant F1 has a noise floor of about ±0.015.** Changing embedding vectors by about 1e-4 (batch padding) moved it by up to 0.014, because one merchant can flip a class's F1. So the planned ±0.005 tolerance holds for known merchants and all test users (±0.01, the split) but cannot hold for unseen merchants. **Proposed criterion:** ±0.005 on known merchants, ±0.015 on unseen merchants, plus the exact-split check. The paired tie rule already treats differences this small as ties.
  - **Fixed along the way:** in a padded batch, a string's embedding shifts slightly with its neighbours. Training batched, while serving embeds one string, so the two would see different features. Strings are now embedded one at a time: every call gives the same vector, at about 21 s instead of 10 s per run (cached).
- **The one exemption from "test sets only in `finalize`":** this check has to score the test sets in round 1. It re-scores a configuration whose test numbers are already published, and nothing is chosen with it. The runner allows it only with `--reproduce-poc`, which accepts only the POC's configurations (by config hash), tags the run `reproduce_poc`, and excludes the run from the leaderboard and from `finalize`. There is no general way to score test sets outside `finalize`.

### 4. Income

Income is predicted as a 13th class, and macro F1 is reported over the 12 spending categories, with Income's F1 reported on its own line.

- A sign rule alone fails: 5,422 Shopping refunds are positive.
- Keeping Income in the model lets text plus sign separate payroll from refunds, and `get_spending_summary` needs both anyway.
- Excluding Income from the headline matches the PRD's "about 12" and keeps a near-perfect class from inflating macro F1.

This closes the open question carried from FR-1 and FR-2, pending review.

### 5. Training pipeline

This is what the [runner](#experiment-runner-and-tracking-mlflow) does for the categorization task (`evaluation/tasks/categorization.py`), which is allowed to read truth:

1. Load model-visible transactions and `truth_transactions.category` for train users.
2. **Known-merchant test:** the Technical Design's stratified 20% by transaction within train users. The other 80% are the training rows.
3. **Grouped cross-fitting** chooses C and fits the calibrators without holding any merchant out of the shipped model:
   - Split the holdout-eligible merchants that appear in the training rows into K = 5 seeded groups, stratified by category so each group has merchants from most categories. Protected merchants are never held out ([Tasks and splits](#tasks-and-splits)).
   - For each group k, train a fold model on the training rows outside group k, minus a seeded 10% row sample of those merchants. It predicts on group k's rows (merchants it has never seen) and on the 10% sample (merchants it has seen).
   - Pool the out-of-fold predictions. Each row's familiarity is computed against its own fold model's training vocabulary, exactly as production computes it against the full training set.
   - Choose C by the mean of pooled macro F1 on unseen-merchant rows and on seen-merchant rows. Then fit one calibrator per familiarity group on the pooled predictions for that C.
   - **The shipped model** is then trained on all training rows with the chosen C, and the calibrators are attached. No merchant is left out, so all train-user merchants are learned.
   - Cost: K × 3 values of C = 15 fits, plus the shipped model's fit: 16 fits of about 30 s each, around 8 minutes on a laptop CPU per configuration.
   - The calibrators come from fold models, each trained on about 80% of the merchants, and are applied to the shipped model. That approximation is checked directly: §6 reports the shipped model's calibration error on the known-merchant test, all test users and unseen merchants.
4. **Label noise:** applied *after* splitting, to the rows each model trains on (fold models and the shipped model). It flips a configurable share of spending labels uniformly to another spending category; Income labels are never flipped, and nothing is flipped to Income, since a payroll labeled Dining isn't a realistic correction error; the default is 2%, a Technical Design control against flattering results. It never touches held-out fold rows, calibration rows or test rows, so calibration can't learn the noise. The report states the rate.
   - **Measured effect (milestone 3; rule kept by decision, Oct 1, 2026).** Uniform destinations send about the same number of wrong labels into every category, so small categories take most of the damage. In the default training rows, 39% of Travel's labels are wrong, and 12% of Housing's and Childcare & Education's, against 0.3% of Dining's. On validation fold 0 (C = 3, n-grams + embeddings), macro F1 goes from 0.989 known / 0.573 unseen without noise to 0.953 / 0.368 with it. Choosing the wrong category in proportion to category frequency (every class about 2% wrong) gave 0.984 / 0.553. Every experiment is compared under this noise, so the comparisons are fair, but absolute numbers sit well below the noise-free ones, especially for rare categories and unseen merchants. The round 1 candidate `linear_both` scored 0.967 known / 0.439 unseen on validation.
5. **Cap per class:** at most `max_rows_per_class` training rows per class (default 20k) for each model, sampled with a fixed seed. Dining has 418k rows, so this keeps training fast and balanced without dropping rare classes. The familiarity vocabulary is built before the cap (§3).
6. Log the run to MLflow: model, manifest (including the chosen C, K, the fold seed and each calibrator) and validation metrics. The test sets wait for `finalize`.

### 6. Evaluation

Every run reports the validation versions of these metrics (pooled out-of-fold predictions: unseen-merchant folds and seen-merchant fold rows). The test sets below are scored only by `finalize`.

| Test set | Definition | Metric |
| --- | --- | --- |
| Known merchants | Train users, the held-out 20% | Macro F1 (12 spending), per-class F1, confusion matrix. **Gate: ≥ 0.90 and > baseline** |
| All test users | Every test-user transaction | Macro F1. The realistic mix of known and new |
| Unseen merchants | Test-user transactions at holdout merchants | Macro F1 with a 95% interval from a **merchant-level** bootstrap. Reported, not gated (FR-4) |
| Ambiguity ceiling | Majority-category oracle on known merchants | Per-class F1 for Groceries and Shopping, shown next to the model's |
| Refunds | Positive-amount spending transactions | Accuracy, so refunds aren't silently called Income |

Also reported: expected calibration error on every test set and for seen and unseen strings, accuracy at confidence ≥ 0.9 and the share of transactions above that threshold (FR-5 groundwork), and p50 / p95 latency for a one-row batch and for 10k rows.

**Robustness of the known-merchant gate (checked in review).** 84.5% of known-merchant test strings appear verbatim in training, so memorization was a concern. But the n-gram model scores 0.986 on strings it has never seen as well. With a split by user instead of by transaction, it scores 0.985 overall and 0.970 on new strings. The gate doesn't rest on memorized strings. The report adds a never-seen-string slice of the known test so this stays visible.

**Why a merchant-level bootstrap:** the unseen set has 78k transactions but only 58 merchants, as few as 3 for Travel and Entertainment. One merchant decides a class's F1, so a transaction-level interval would be falsely narrow.

### Metrics and why

Each metric has one job. This section records why it was chosen, which alternatives would also have made sense, and which would not. The test-set and validation metrics have the same names (`val_*`, `test_*`).

**1. Is the category right? Macro F1 over the 12 spending categories** (headline, FR-3 gate, FR-4 target, selection).

- **Why:** every category is a line on the dashboard and a bucket in `get_spending_summary`. A user who checks Travel cares about Travel as much as one who checks Dining, so each category counts equally, however rare (Travel is 0.3% of rows). F1 needs both precision (don't pollute a category with others' spend) and recall (don't lose a category's spend to others). Income is scored separately, so a near-perfect easy class doesn't inflate the average (§4). It is also the PRD's metric.
- **Would also make sense:**
  - Macro recall (balanced accuracy). Simpler, but blind to false positives, which inflate a category's total.
  - Matthews correlation or Cohen's kappa. Single numbers that are robust to imbalance, but they don't break down by category, and they aren't the PRD's language.
  - **Dollar-weighted error**, which is adopted as a secondary metric (point 6).
- **Would not:**
  - Accuracy or micro F1. Dining is 38% of rows, so a model that never predicts Travel loses under 1%.
  - Weighted F1. Same domination by Dining.
  - Top-k accuracy. The product shows one category.
  - Per-class ROC-AUC or PR-AUC. They rank scores without a threshold, while the product takes the top class, so they don't measure the decision that ships.

**2. Which C does a run use? The mean of validation macro F1 on known and unseen merchants** (`tuning_macro_f1`).

- **Why:** C trades memorizing merchants against generalizing to new ones, and the shipped model serves both. Tuning on unseen merchants alone could accept a known-merchant regression that the 0.90 eligibility floor doesn't catch. Tuning on known merchants alone can't separate settings, because they all score 0.97–0.99.
- **Would also make sense:** weighting by the production mix (about 22% of test-user rows are at unfamiliar strings). It needs a mix assumption that validation doesn't have, so the simpler symmetric mean was used.

**3. Which run wins? Validation macro F1 on unseen merchants**, among runs at ≥ 0.90 on known merchants and above the keyword baseline (the [decision rule](#decision-rule)).

- **Why:** every candidate already clears the known-merchant gate, so the unseen score is where they differ, and every new merchant a user meets is unseen.

**4. How sure are we? A 95% interval from a merchant-level bootstrap, stratified by category; ties use a paired version on the same merchants.**

- **Why:** the rows aren't independent; merchants are. One merchant can decide a class's F1 (§6), so resampling transactions gives intervals that are falsely narrow. Pairing removes the variation the two runs share, so a real difference isn't hidden by it.
- **Measured (milestone 3):** perturbing embedding vectors by about 1e-4, from batch padding (since fixed), moved unseen-merchant F1 of the embeddings-only model by 0.014. Differences of that size are noise, which is what the paired tie rule absorbs.
- **Would also make sense:** repeating the run with different seeds. That measures training variability, not merchant sampling, so it complements the bootstrap rather than replacing it.
- **Would not:**
  - McNemar's test on transactions. It assumes independent rows.
  - "The two intervals don't overlap". That test is too conservative for paired runs, and it isn't a test of their difference.

**5. Can FR-5 trust the confidence? Brier score of the top-class confidence, to choose; ECE and the 0.9 operating point, to report.**

- **Why Brier:**
  - It is a proper score: it is lowest only when confidences are both calibrated and discriminating (right answers confident, wrong ones not). That is what a review threshold needs.
  - It is bounded, so one confident mistake can't dominate it.
  - It chooses the calibration method per familiarity group, cross-validated across folds so a flexible method can't win by fitting the rows it is scored on. It also breaks ties in the decision rule.
- **Why ECE is reported, not optimized:** "0.9 means right 90% of the time" is easy to read. But ECE isn't proper (a constant confidence equal to the accuracy scores 0), and it depends on the 15 bins.
- **The FR-5 operating point:** accuracy at confidence ≥ 0.9, and the share of transactions above it.
- **Would also make sense:**
  - Log loss, which is proper and punishes confident mistakes hardest. It is unbounded, though, and isotonic calibration outputs exact 0 and 1, so it would need arbitrary clipping. It is used where it fits naturally: fitting the temperature.
  - The area under the risk–coverage curve, which is FR-5's whole trade-off curve. It belongs in the FR-5 design.
- **Would not:**
  - ECE as an objective.
  - Mean confidence.
  - Per-class ECE across 13 classes with a handful of unseen merchants each, which is mostly noise.

**6. What does the user feel? Misallocated spend: the share of spending dollars put in the wrong category** (`*_misallocated_spend`).

- **Why:** the dashboard and the coach sum dollars by category, and misfiling a $2,000 rent payment matters more than misfiling a $4 coffee. Macro F1 counts transactions, not dollars.
- **Reported, not gated:** a few large, easy recurring merchants (Housing, Utilities) dominate it, so it can look good while small categories are wrong. It complements macro F1 rather than replacing it.

**7. What does it cost to serve? Reported, never gated** (decision Oct 2, 2026, made after the launch round's results were known; recorded as such in the launch-round report).

- **Why no gate:** categorization runs **on ingestion, in batches** across users, not per request. New transactions arrive from bank feeds in batches, and no user waits on a single categorization. A slower model costs more CPU, which the cluster is sized for, and the categorization workers can run on bigger nodes. Rejecting an otherwise better model for a few milliseconds per transaction would optimize the wrong thing. The milestone 3 gate (5 ms p95 per transaction) also turned out to depend on the machine: 3.1 ms on a laptop, 8.95 ms on a CI runner.
- **The cost metric that matters: `latency_batch_ms`,** the time for one fixed 10k-row batch (the runner's latency sample). It is the second tie-breaker among statistically tied runs, so a cheaper model wins a real tie. It is measured in the same run environment for every candidate, so it ranks consistently even though its absolute value depends on the machine.
- **Also reported:**
  - `latency_p95_ms`: one-transaction calls on distinct rows from a cold start (caches cleared with `reset_caches`), for the chat path's occasional single lookups;
  - `latency_warm_p95_ms`: the same rows a second time, with warm caches.

  Measured (milestone 3): 3.1 ms cold and 1.2 ms warm on a laptop; 8.95 ms cold on a CI runner.
- **Measured with `sfc-model predict`** (bge-base, default dataset, laptop CPU, one process): about 12,000 transactions per second; 0.6 s to load the model once its file is downloaded. `predict` reads the dataset in batch-sized chunks (milestone 6, from review on #13), so peak memory depends on the batch size, not the dataset: 1.96 GB at 20k-row batches (the default) and 6.6 GB at 100k, at the same throughput. Before chunking, the whole dataset was loaded first: 2.1 GB and 7.2 GB. The model's per-batch working set, not the data, is most of the peak.
- **Serving at scale is a deployment concern, for the serving design:**
  - throughput per core and cost per million transactions;
  - pod memory: the ONNX model is 67 MB for bge-small and about twice that for bge-base, and the embedding cache is a bounded LRU of up to about 150 MB;
  - cold start: model load, plus each new pod embedding strings on first sight; ship the training vocabulary's vectors (about 20 MB) or warm on `load_service`;
  - ONNX thread settings: one thread per worker, CPU requests equal to limits, so container CPU throttling doesn't create tail latency.
- **Before profiling,** a single call took 8.4 ms: pandas `isin` rebuilt the 13k-string vocabulary set on every call. Fixed in milestone 3.

**8. Diagnostics, reported and never gated:**

- per-class F1;
- Income F1 (it counts spending rows called Income);
- refund accuracy;
- the never-seen-string slice of the known test;
- the ambiguity ceiling for Groceries and Shopping;
- confidence quality for familiar and unfamiliar strings separately;
- the unseen calibrator's report (cross-validated Brier per method and group).

### 7. Truth isolation

- `intelligence/categorization/` reads no truth table and imports nothing from `data.labels`. The existing isolation test covers it unchanged.
- `data/store.py` (new, minimal) is the data-access layer from the Technical Design: `load_transactions`, `load_users`, `load_goals`. It reads model-visible tables only and can scope by `user_id`. The tool server builds on it in build order step 6.
- `evaluation/` reads `truth_transactions.category` and `truth_merchants.holdout`, which only define labels and splits.

### 8. Module layout

```
src/smart_financial_coach/
  data/
    store.py                        model-visible data access (new)
    features/merchant_text.py       normalize_merchant (new)
  intelligence/
    models/
      base.py                       Model protocol, BaseModel
      registry.py                   register, build from {type, params}
      contract.py                   Contract, Checked (output checks on every prediction)
      wrappers.py                   Calibrated, Routed, Lookup
      artifact.py                   save / load folder, manifest, checksum, PROMOTED, promotion log
    service.py                      load_service: PROMOTED pointer -> contract-checked model
    categorization/
      contract.py                   Categorizer protocol, output check
      baseline.py                   majority, KeywordCategorizer
      linear.py                     linear_text (n-gram / embedding / side-feature switches)
      embeddings.py                 cached sentence embeddings (fastembed), stub for tests
      ...                           further candidates from the experiment plan
  evaluation/
    tasks/base.py                   Task protocol, Examples, task registry
    tasks/categorization.py         examples, labels, splits, metrics, gates
    splits.py                       Splits, splitters, leak checks
    metrics/classification.py       macro F1, per class, bootstrap, ECE
    experiment.py                   experiment config (YAML), config hash, grid
    selection.py                    the decision rule (ties against the leader)
    tracking.py                     MLflow logging (the only module that imports mlflow)
    runner.py                       run one experiment config
    promote.py                      leaderboard, finalize, promote, export
    cli.py                          sfc-experiment run | leaderboard | finalize ; sfc-model promote | show | predict
configs/
  experiments/categorization/*.yaml one file per experiment
  models/category_keywords.yaml     baseline rules
```

New dependencies:

- `fastembed` (ONNX runtime, about 130 MB model download, cached). Tests that need the embedding model are marked `slow`. Unit tests use a stub embedder so CI doesn't download it (see [option B](#b-text-representation)).
- `mlflow`, imported only by `evaluation/tracking.py` and `promote.py`, in a `train` dependency group rather than `[project.dependencies]`. The full package brings a server stack (Flask, SQLAlchemy, Alembic and more) that the dashboard doesn't need; a serving install leaves the group out. The group is the full package rather than `mlflow-skinny` because `mlflow ui` and `mlflow server` need it. `[tool.uv] default-groups = ["dev", "train"]`, so `uv sync --locked` in both CI workflows (`ci.yml`, `slow.yml`) and on developer machines installs it with no workflow change; a serving install uses `--no-default-groups`. Unit tests log to a temporary local store. `fastembed` stays a runtime dependency, because serving embeds new merchant strings.

## Experiment plan

The framework exists so that choosing a model is an experiment, not an argument. This plan is fixed before the runs. Changes to it are recorded with a reason.

### Decision rule

1. **Eligible:** validation known-merchant macro F1 ≥ 0.90 and above the keyword baseline's validation score.
2. **Ranked by validation unseen-merchant macro F1.** Every feasibility candidate already clears the known-merchant gate (0.97–0.99), so the unseen-merchant score is where candidates differ and where users notice: every new merchant is unseen.
3. **Ties, defined against the leader only** (pairwise ties aren't transitive, so they wouldn't give one order across ~40 runs):
   - The leader is the eligible run with the best point estimate.
   - Its tie set is every eligible run whose paired difference with the leader has a 95% interval containing 0 (merchant-level bootstrap over the same validation folds). The leader is in its own tie set.
   - Within the tie set, runs are ordered by validation unseen-merchant Brier score after calibration (see [Metrics and why](#metrics-and-why), point 5; the plan first said ECE), then batch serving cost (`latency_batch_ms`; the plan first said p95 latency, see [Metrics and why](#metrics-and-why), point 7), then explainability, then operational simplicity (fewer components and dependencies).
   - If the tie set has fewer than three runs, the rest of the order is by point estimate.
4. **Finalists:** the top three in that order go to `finalize`, and the baselines are scored with them (see [Selection discipline](#selection-discipline)). The rule's winner is promoted if it passes the test gates. If it fails one, that's investigated before anything is promoted; the finalists are never re-ranked on test scores.

### Experiment rounds

| Round | Variants | Question it answers |
| --- | --- | --- |
| 0. Baselines | Majority class; keyword rules; exact normalized-string lookup | The floor every model must beat |
| 1. Reproduce | POC's n-grams / embeddings / both | Does the framework match the POC (§3)? |
| 2. Text features | N-gram ranges (2–4, 1–5, word unigrams + bigrams); raw vs normalized text for n-grams; embedding models, all supported by `fastembed` 0.8.1: `BAAI/bge-small-en-v1.5` (POC), `BAAI/bge-base-en-v1.5` and `BAAI/bge-large-en-v1.5` (does size help?), `sentence-transformers/all-MiniLM-L6-v2` and `snowflake/snowflake-arctic-embed-m` (other training lineages), `minishlab/potion-base-8M` (static embeddings: much faster, how much worse?) | How much is the text representation worth on unseen merchants? |
| 3. Side features | Remove amount, sign, channel, hour one at a time | Does hour stay in? Does amount help with ambiguous merchants? |
| 4. Training regime | Per-class cap 5k / 20k / none; one row per unique normalized string; balanced vs no class weights; label noise 0 / 2 / 5% | Do frequent merchants drown out rare ones? How sensitive is the model to noise? |
| 5. Classifiers | Logistic regression; linear SVM + calibration; kNN over embeddings; gradient-boosted trees (`HistGradientBoostingClassifier`) on dense features | Is the classifier or the features the bottleneck? |
| 6. Composition | `Routed(seen=n-grams, unseen=embeddings)`; `Lookup(fallback=best)`; stacking the round 2 winners | The FR-4 ideas, measured |
| 7. Calibration | None / temperature / isotonic; one calibrator vs split by familiarity | Can FR-5 trust a 0.9 threshold on new merchants? |
| 8. Optional | Fine-tuned small transformer | Only if rounds 2–6 plateau well below 0.80 on unseen merchants |

- Later rounds start from the best of earlier ones, so the grid grows with the number of rounds, not their product.
- **Additions** to any round after the runs start are recorded in this section with a reason.

### Trimmed for launch (Oct 1, 2026)

**Decision:** before launch, categorization needs a *decent* cold-start model, not the best possible one. What decides quality after launch is how well the model learns from user feedback (FR-5 review, FR-6 corrections; see the Technical Design, "Learning from user feedback"). So milestone 4 runs only what the cold model and FR-5 depend on.

**Why the plan didn't fit anyway:** a calibrated n-grams + embeddings configuration took about **45 minutes**, not the planned 8: three values of C × five folds plus the final fit, each with embeddings and calibration. The full plan (about 40 configurations) would take about 30 hours.

| Round | Kept for launch | Deferred |
| --- | --- | --- |
| 2. Text features | `bge-base-en-v1.5` vs `bge-small` (both with n-grams); n-grams 1–5 vs 2–4. Larger embedding models only if bge-base wins | Word n-grams, raw vs normalized text, the other embedding models |
| 3. Side features | — | All ablations (hour stays on, as in the POC) |
| 4. Training regime | — | Caps, one row per string, class weights, noise levels |
| 5. Classifiers | — | Linear SVM, kNN, boosted trees |
| 6. Composition | — | `Routed`, `Lookup` fallback, stacking: they belong with feedback-driven learning |
| 7. Calibration | One calibrator vs one per familiarity group (`by: none`). The three methods are already compared within every calibrated run (cross-validated Brier in `held_out.*`) | — |
| 8. Optional | — | Fine-tuned transformer |

- **Cost controls:** C is fixed at 1.0, round 1's choice for the n-grams + embeddings family, so each configuration fits 3 fold models and the final model instead of 16 fits. The launch round runs on **3 folds**.
- **One split for the whole launch round.** `k` changes the folds and the split hash, and the leaderboard and `finalize` only compare runs on the same split, with the keyword baseline on it. So the launch round's folder (`configs/experiments/categorization/launch/`) reruns everything it is compared against on the same 3 folds: the three round 0 baselines (minutes) and the round 1 reference (n-grams + `bge-small`, C = 1.0). Seven configurations in all, estimated at about 1–1.5 hours; milestone 4 records the measured time.
- **Output:**
  - the MLflow experiment with every launch-round run;
  - a comparison table in the evaluation report, with validation metrics and merchant-bootstrap intervals for every run, and test metrics for the finalists once milestone 5 finalizes them;
  - an updated FR-3 Categorization Model Selection (milestone 5), recording what won, what lost and why, what was deferred, and what stays open for FR-4.
- **Deferred rounds aren't dropped.** They are the menu for the first retraining design, once real corrections show where the cold model is wrong.
- **New selection consideration:** how cheaply a model retrains and absorbs corrections. It isn't in the Technical Design's criteria yet; until it is, it's recorded in milestone 5's promotion note.

## Options considered

### A. Contract shape

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Keep `categorize(merchant_raw, amount)` from the Technical Design | Already documented | Drops channel, which helps; every feature experiment would change the contract |
| **(b) Batch of model-visible transaction rows (recommended)** | Stable whichever features win; matches batch dashboard use | Callers build a frame for one transaction |
| (c) Separate scalar and batch methods | Convenient | Two code paths to keep in sync |

### B. Text representation

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Character n-grams only | No new dependency; 0.99 on known merchants | 0.50 on unseen merchants: no knowledge from outside the training data |
| **(b) N-grams + frozen sentence embeddings (recommended)** | Best on known merchants and all test users (0.99 / 0.93); ties embeddings alone on unseen merchants (0.64); CPU, seconds | New dependency and a model download; pins an external model file (checksum in the manifest) |
| (d) Frozen sentence embeddings only | Ties the best on unseen merchants (0.64); 0.91 on all test users | 0.97 on known merchants: loses exact-merchant memory |
| (c) Fine-tuned small transformer | Possibly best on unseen merchants | Training cost, GPU likely, heavier ops; a milestone 3 candidate, not the default |

### C. Where predictions live

| Option | Pros | Cons |
| --- | --- | --- |
| (a) New table in the generator's SQLite file | One file | Generator output stops being a pure function of the spec; the content hash breaks |
| **(b) Separate predictions SQLite file (recommended)** | Generator file stays immutable; predictions are versioned by model | Two files to pass around; `ATTACH` to join |
| (c) Don't persist; categorize on every request | Nothing to keep in sync | Repeats work for every dashboard load; conflicts with the batch-precompute choice in the Technical Design |

### D. Income

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Sign rule before the model | Simple | Refunds become Income (5,422 rows) |
| **(b) Predicted class, excluded from headline macro F1 (recommended)** | Handles refunds; honest headline | 13 outputs where the PRD says "about 12" |
| (c) Predicted class, included in macro F1 | One number | A near-perfect class inflates macro F1 |

### E. Label noise

| Option | Pros | Cons |
| --- | --- | --- |
| (a) None | Simplest | Ignores a Technical Design control; synthetic labels are cleaner than user corrections will be |
| **(b) Uniform flips at a configured rate, default 2% (recommended)** | Implements the control; rate reported | Uniform noise is easier than real, confusable noise |
| (c) Confusion-shaped flips (e.g. Groceries ↔ Shopping) | More realistic | Needs a confusion prior we don't have yet |

### F. Experiment tracking

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Run folders + a leaderboard file in the repo | No dependency | Comparing runs, charts and sharing are all hand-built |
| **(b) MLflow (recommended, chosen)** | Standard run comparison UI, params, metrics and artifacts per run; model registry with aliases; local store by default, shared server by configuration | Heavy dependency (server, database migrations); must be kept out of the serving path |
| (c) Hosted tracker (e.g. Weights & Biases) | Sharing with no server to run | Account and vendor dependency; data leaves the machine |

Sharing needs a tracking server everyone can reach. Until one is chosen, runs live in the local store and the evaluation report in the repo carries the comparison.

### G. Promotion and serving

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Serve directly from the MLflow registry (`models:/categorization@champion`) | One source of truth | The dashboard depends on the tracking server and on `mlflow` at runtime |
| **(b) MLflow alias + exported folder and `PROMOTED` pointer (recommended)** | Serving needs only files; registry still records what was promoted and when | Registry history is machine-local until a shared server exists, so `promote` also appends to a committed `promotions.jsonl`; tests check `PROMOTED` against it |
| (c) Pointer file only, no registry | Simplest | Loses promotion history in the tracker |

**Where the exported model file is stored** (decided Oct 2, 2026, for option b):

| Option | Pros | Cons |
| --- | --- | --- |
| (i) Commit it | Nothing to download | 10–13 MB, over the 5 MB file limit; every retraining grows git history for good |
| (ii) Shrink it, then commit (float32 coefficients + xz) | Fits: bge-base drops from 10.0 MB to 3.6 MB | Borderline for larger models (n-grams 1–5 about 4.5 MB); the served model differs numerically from the one `finalize` scored; history still grows |
| (iii) Git LFS | Stays in the repo | Every clone and CI job needs LFS; storage and bandwidth quotas |
| **(iv) GitHub Release asset per promoted version (chosen)** | No new service; public download URL; kept until deleted; files up to 2 GiB | `promote` needs `gh` with write access; first use downloads the file |
| (v) MLflow server or object storage (S3, GCS) | The standard production store | A server or a cloud account to run, for a single file per promotion |
| (vi) Container registry (ghcr.io) | Fits Kubernetes deployment | No generic file store; it means baking the model into an image |

Releases are the store until continuous deployment is built. CD will then use the container registry (vi): the serving image is built with the promoted model in it, checked against the same committed manifest. Moving the file elsewhere only changes the URL in the promotion log; the pointer, manifest and checksum stay as they are.

### H. How experiments are compared

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Compare on test sets | Simple, matches the POC | With many runs and 58 unseen merchants, picks lucky winners |
| **(b) Compare on merchant-grouped validation folds; test once for finalists (recommended)** | Test numbers stay honest; validation reuses the cross-fitting already needed for C and calibration | Validation merchants are train-user merchants, a slightly different population from holdout merchants |
| (c) A separate validation user split | Mirrors the test split exactly | Takes users away from training; the merchant-level unit still decides variance |

## Testing

- **Unit:** label noise never touches validation, calibration or test rows. Loading an artifact whose embedding file checksum doesn't match the manifest fails. In cross-fitting, every holdout-eligible training merchant is held out in exactly one fold, protected merchants in none, and no held-out merchant appears in that fold model's training rows. The shipped model trains on every training merchant. The familiarity vocabulary includes strings the per-class cap sampled out. The normalizer handles every rendering distortion type (prefix, store number, reference code, truncation, `PPD ID:`), idempotent, never empty for non-empty input. Baseline keyword file contains no catalog merchant names. The contract returns one row per input with a known category and confidence in [0, 1]. An empty frame returns an empty frame. `fit` then `categorize` is deterministic for a fixed seed. Label noise flips the configured share and never keeps the original label. Splits: no transaction in two splits, unseen set contains only holdout merchants.
- **Framework (fake models, no real data):** the registry builds a model from config and rejects unknown names. Each splitter's leak check fails on a constructed leak (shared ID, held-out merchant in fold training rows, protected merchant in a held-out group, non-holdout merchant in the unseen set). The generator and the task get the same holdout-eligible merchants from the shared function. The runner logs params, tags, split hashes and `val_*` metrics to a temporary MLflow store and writes no `test_*` metrics; `--reproduce-poc` refuses any config that isn't a POC config. `finalize` refuses more than three runs and runs with different split hashes, and always scores the baselines. The decision rule orders a constructed set of runs where pairwise ties aren't transitive (A ties B, B ties C, A beats C) the same way every time. `promote` refuses a non-finalist and a run that fails a gate, leaves the MLflow alias and `PROMOTED` pointing at the same version, and appends to `promotions.jsonl`. Nothing under `intelligence/` imports `mlflow`. `load_service` returns a contract-checked model, and a model that returns too few rows, an unknown category or confidence outside [0, 1] fails the check. A resumed sweep skips finished configs.
- **Integration (`small.yaml`, stub embedder):** run → finalize → promote → `load_service` → predict end to end; the model beats the baseline on known merchants; a rerun gives identical metrics (NFR-8). Swapping the promoted run to a different model type needs no code change.
- **Slow (`default.yaml`, real embedder):** reproduces the POC within ±0.005 (§3); known-merchant macro F1 ≥ 0.90; serving cost is logged.
- **Isolation:** the existing test covers `intelligence/`; it is extended to forbid `mlflow` imports there and to cover `data/store.py` and `data/features/`.

## Milestones

One PR per milestone.

1. **Framework:** `Model` protocol, registry, wrappers, artifact and manifest; `Splits`, the four splitters and leak checks; `Task` protocol; runner with MLflow tracking; finalize, promote, `load_service`; CLI. Tested with fake models on `small.yaml`.
2. **Categorization task and baselines:** `data/store.py`, `normalize_merchant`, the categorization task and its metrics (macro F1, bootstrap, ECE, report), majority, keyword and lookup baselines. Round 0 sets the baseline numbers.
3. **Linear text model:** n-grams, embeddings, side features, cross-fitting for C, `Calibrated`. Round 1 reproduces the POC.
4. **Experiments (trimmed for launch, see [Trimmed for launch](#trimmed-for-launch-oct-1-2026)):** round 2's embedding-size and n-gram-range variants, and round 7's calibration grouping. Mostly configs and results.
5. **Promote and predict:** finalize, promote the winner, `predict` writes `transaction_categories`, serving cost logged (not gated). Update FR-3 Categorization Model Selection.
6. **Docs:** Technical Design (contract, Income, predictions store, evaluation harness), close the Income question in FR-1 and FR-2.

## Status and handoff (Oct 2, 2026)

Written for whoever continues FR-3. It records where the work stands, what remains, and how the work has been done so far.

### Done

| Milestone | PR | What landed |
| --- | --- | --- |
| Design | #6 | Training framework, MLflow, experiment plan |
| 1. Framework | #7 | Model protocol, registry, contracts, artifacts, splits and leak checks, runner, leaderboard, finalize, promote, `load_service`, CLIs |
| 2. Task and baselines | #8 | Data store, normalizer, `holdout_eligible` (schema 3), categorization task, round 0 baselines; run identity = config + data content + split + code version |
| 3. Linear model | #9 | `linear_text`, embeddings (one string per batch, pinned file), `calibrated` per familiarity group, "Metrics and why", POC reproduction |
| Plan trim | #10 | Milestone 4 trimmed for launch; Technical Design "Learning from user feedback" |
| 4. Launch round | #11 | Launch-round configs and results (`docs/reports/FR-3 Categorization — Launch Round Results.md`), `sfc-experiment report`; latency gate removed (batched on ingestion) |
| 5. Promote and predict | #13 | Model files in GitHub Releases; `finalize` (once); bge-base promoted (`3f0ccc82-2f0e60a6`); `sfc-model predict`; Model Selection doc and launch report updated |
| 6. Docs | #14 | Technical Design updated (contract, predictions store, evaluation, artifacts); Income question closed in FR-1 and FR-2; this doc Accepted. Review follow-ups from #13: per-run temporary predictions files, chunked reads in `predict`, releases tagged at the training commit |

**Where it stands.** `finalize` scored the top three (bge-base, n-grams 1–5, bge-small) and the baselines on the test sets once. bge-base passed both gates (known 0.970 against 0.90 and keyword's 0.461) and is promoted. Its known issue, Travel as the fallback guess for unfamiliar merchants, was shipped knowingly by the owner's decision. FR-3 Categorization Model Selection and the launch report hold the results. The test sets on split `ce93ef87…` are now used: a further round needs `--override`.

### Remaining

FR-3 is complete. What it leaves open belongs to other designs:

- **FR-4** (unseen merchants, 0.462 on test against 0.80): the approach, and whether the 0.80 target is meaningful with 3–10 holdout merchants per class (open questions below).
- **FR-5 and FR-6** (review and corrections, the retraining design): the Travel fallback for unfamiliar merchants, and whether the shipped model should train on injected label noise at all; FR-5's review threshold from the reliability curve (FR-3 Categorization Model Selection, "Known issues").
- **Serving design and CD:** model files baked into the serving image through the container registry, which removes first-use download's interim limits.

### Practical notes for the next session

- **The promoted model doesn't need `mlruns/`.** `load_service("categorization")` downloads it from the GitHub Release on first use and checks it against the committed manifest. `uv run sfc-model predict --task categorization --data data/synthetic/default.sqlite --out data/predictions/default.sqlite` takes about 90 s on the default dataset.
- **The launch round's MLflow runs exist only on the machine that ran them**, in `mlruns/` (git-ignored, local SQLite store). Run IDs: keyword `6830162a…`, lookup `95a4e68b…`, majority `bd8bb11b…`, bge-small `dd8d5e85…`, bge-base `000ef7d3…`, n-grams 1–5 `6e25c1a9…`, one calibrator `87290714…`; split hash `ce93ef87…`.
  - On another machine: regenerate the dataset and rerun the launch folder, about 40 minutes: `uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite --force`, then `uv run sfc-experiment run configs/experiments/categorization/launch/ --data data/synthetic/default.sqlite`.
  - Those runs were made at an earlier code version. `finalize` compares finalists and baselines with each other, not with the current code, so they remain usable. A rerun of the launch folder creates new runs rather than resuming.
- **Embedding model files** download to `data/models/fastembed` on first use (git-ignored; about 67 MB for bge-small, about 130 MB for bge-base).
- **Long runs:** run them in the background with `PYTHONUNBUFFERED=1` so logs stream. A calibrated linear configuration with fixed C on 3 folds takes about 8 minutes (bge-base about 14).
- **How the work has been done** (keep it):
  - one PR per milestone, from a branch based on the latest `main` (rebase unpushed work onto `main` before pushing; the `main` ruleset requires an up-to-date branch and a green `check`);
  - before every push, check the open PR for review comments as a separate step and address them first, replying on each thread;
  - design or rule changes are proposed to the owner before they're made, recorded in this doc with their reason, and labelled when they come after seeing results;
  - test sets are touched only by `finalize` (and `--reproduce-poc`);
  - a promotion publishes its release before the promotion PR merges (the promotion takes effect only when `PROMOTED` merges). **If a promotion PR is rejected, delete its release** (`gh release delete <service>-<version> --cleanup-tag`), so the releases list never shows a model that didn't ship.

## Decisions and open questions

**Decisions** (reviewed on PR #4, PR #6 and PR #12)

- [x] FR-3 / FR-4 split as above: FR-3 gates on known merchants, FR-4 owns the unseen-merchant gap.
- [x] Batch contract over model-visible rows (A-b).
- [x] N-grams + frozen sentence embeddings as the default, adding `fastembed` (B-b). Condition: the manifest records the embedding file's source, revision and checksum, and loading verifies it.
- [x] Predictions in a separate SQLite file (C-b).
- [x] Income as a predicted class, excluded from the headline macro F1 (D-b).
- [x] 2% uniform training label noise (E-b). Condition: applied after splitting, never to validation, calibration or test rows.
- [x] C and calibration chosen by grouped cross-fitting over merchants; the shipped model trains on all training rows; calibration split by familiarity, with the vocabulary built before the cap (from review).
- [x] FR-3 builds the training and evaluation framework (generic model interface, registry, splitters, runner, promotion), extended feature by feature along [Path to a general framework](#path-to-a-general-framework).
- [x] MLflow for experiment tracking and the model registry (F-b), installed through a `train` dependency group so serving installs stay lean (from review).
- [x] Validation folds hold out only holdout-eligible merchants, so validation-unseen matches `test_unseen`'s population; ties defined against the leader; baselines always scored in `finalize`; a committed promotion log (from review on PR #6).
- [x] Serving from an exported folder and `PROMOTED` pointer, never from MLflow (G-b); model files in GitHub Releases, verified against the committed manifest (decided Oct 2, 2026).
- [x] Experiments compared on validation; test sets scored once for at most three finalists (H-b), with the decision rule in [Experiment plan](#experiment-plan).

**Technical Design updates** (made in milestone 6)

- [x] `categorize` contract: batch of transaction rows → category, confidence, model version.
- [x] Data store: model outputs live in a separate predictions file per model version.
- [x] Categorization evaluation row: headline macro F1 over 12 spending categories; merchant-level bootstrap interval for unseen merchants; grouped cross-fitting over merchants for C and calibration; ECE on every test set.
- [x] Evaluation framework: generic model interface, registry, promotion, MLflow tracking (the Infrastructure table's "model registry" becomes the MLflow registry plus exported artifacts); experiments compared on validation, test sets scored once.

**Open questions**

- [ ] **Where the shared MLflow server runs.** Sharing runs needs a server everyone can reach (self-hosted `mlflow server` with a database and artifact store, or a managed MLflow). Until then the local store works and the report in the repo carries the comparison.
- [x] **Promoted artifacts exceed the repo's 5 MB file limit** (9.5–12.7 MB measured). Settled Oct 2, 2026: model files go to GitHub Releases; the manifest, pointer and log are committed ([Model artifact](#model-artifact), option G).

- [ ] **FR-4 approach** (for the FR-4 design). Embeddings get 0.64 (interval 0.56–0.73) against the 0.80 target. Options for the FR-4 design: a larger embedding model; training on one row per unique merchant string, so frequent merchants don't dominate; or an LLM fallback for low-confidence, never-seen strings, cached per normalized merchant so cost scales with merchants, not transactions. The LLM option would make categorization depend on the LLM provider, which NFR-6 avoids for the dashboard. A cached fallback would keep the dashboard working during an outage, but new merchants would wait.
- [ ] Is the unseen-merchant target realistic with 3–10 merchants per class? The feasibility interval is 0.17 wide, so near the target a pass or fail would be mostly luck. FR-4 may need a larger holdout or a merchant-level metric. The review agrees this should be settled in the FR-4 design, the way FR-2 decided weekly spikes.
