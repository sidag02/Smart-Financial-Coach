# FR-3 Transaction Categorization — Feature Design

Oct 1, 2026 · @Sidd · Status: **Proposed** · Branch: `feature/fr-3`

## Summary

This feature builds the categorization service: every transaction gets exactly one category from the fixed taxonomy, with a confidence, behind a contract that later model experiments can swap without touching callers.

- **Requirement:** FR-3 (P0): *"Assign every transaction one category from a fixed set of about 12."* Success metric (PRD): macro F1 ≥ 0.90 on known merchants.
- **Neighbour:** FR-4 (P0, *"categorize merchants the system has never seen before"*, macro F1 ≥ 0.80) runs on the same service. This design builds the service and measures FR-4 from day one, but leaves closing the FR-4 gap to its own design (see [Scope](#scope-fr-3-versus-fr-4)).
- **Feasibility, measured on the default dataset** ([evidence](#feasibility)): a plain linear model on character n-grams already reaches **0.99** macro F1 on known merchants. On unseen merchants it reaches **0.50**. Adding pretrained sentence embeddings lifts unseen merchants to **0.64** (95% interval 0.56–0.73) and keeps known merchants at 0.99. FR-3's target is easy; FR-4's is not, and even the interval's upper end is below 0.80.
- **Approach:**
  1. A batch-first **service contract** that takes model-visible transaction rows and returns category, confidence and model version.
  2. A shared **merchant-text normalizer** in the feature pipeline, reused later by the anomaly service.
  3. A **keyword baseline** built from generic words only, never merchant names.
  4. A **linear text categorizer** (character n-grams + embeddings + amount, sign and channel). It is the default candidate, and the Technical Design's other model families compete with it by experiment.
  5. A **training and evaluation slice** for categorization: splits, label noise, metrics and report. It becomes the first part of the evaluation harness (build order step 4).
- **Principle (carried from FR-2):** the data is not tuned to make targets pass. The ambiguous-merchant ceiling is reported, not trained around.

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
| Builds | Contract, normalizer, baseline, default model, training and evaluation slice, CLI, artifact | Whatever closes the unseen-merchant gap (see [Decisions and open questions](#decisions-and-open-questions)) |
| Reports | Known-merchant, all-test-user and unseen-merchant macro F1 | Same report |
| Gates on | Known-merchant macro F1 ≥ 0.90 and beats the baseline | Unseen-merchant macro F1 ≥ 0.80 |

## Feasibility

**Evidence:** branch `poc/fr-3-categorization`, script `experiments/fr3_categorization/feasibility.py`. The full report is `experiments/fr3_categorization/results/feasibility.md`, with every number below plus per-class precision and recall in `feasibility.json`. The run used the default dataset (spec hash `130e78f55383`) and seed 0. A second run gives identical metrics.

Setup: the Technical Design's splits, no label noise, no tuning. The model is logistic regression with balanced class weights, trained on 194k train-user transactions (at most 20k per class). Macro F1 is over the 12 spending categories. The unseen-merchant interval comes from 1,000 bootstrap resamples of merchants within each category (see [§6](#6-evaluation)).

| Features | Known merchants (145k txns) | All test users (375k) | Unseen merchants (78k) | Unseen, 95% interval |
| --- | --- | --- | --- | --- |
| Character 2–4-grams of `merchant_raw` + amount bin, sign, channel, hour | 0.986 | 0.868 | 0.504 | 0.44–0.56 |
| Sentence embedding of normalized text (`bge-small-en-v1.5`, 384-d) + same side features | 0.973 | 0.842 | 0.593 | 0.55–0.73 |
| Both | **0.988** | **0.929** | **0.638** | 0.56–0.73 |

Unseen-merchant accuracy by catalog scope (transaction-weighted):

| Scope | Merchants | Transactions | N-grams | Embeddings | Both |
| --- | --- | --- | --- | --- | --- |
| National (real chains) | 25 | 34.0k | 0.46 | 0.62 | 0.70 |
| Local (fictional) | 19 | 26.7k | 0.72 | 0.82 | 0.80 |
| Online services | 14 | 17.4k | 0.43 | 0.93 | 0.87 |

What the errors show:

- **N-grams can't name what an unseen brand sells.** N-gram accuracy is 0.11 on Taco Bell, 0.09 on McDonald's, 0.01 on Caviar and 0.53 on Walmart.com. Embeddings get all four right: 1.00, 0.47, 1.00 and 1.00. That knowledge comes from outside the training data.
- **Some merchants defeat both.** Arco (0.01), Rite Aid (0.01), Giant Eagle (0.00), Alamo Drafthouse (0.00) and Sittercity (0.03) are wrong under every feature set. A small embedding model doesn't know these brands well enough. These five account for 6.8k unseen transactions.
- **Combining features sometimes hurts.** Embeddings alone beat both on online services (0.93 vs 0.87) and local names (0.82 vs 0.80). On Caviar the combination drops from 1.00 to 0.64, and on Corner Fresh Grocery from 0.86 to 0.04. N-grams memorized from training merchants outvote the embedding when a string is unfamiliar. FR-4 should therefore try routing by familiarity rather than a single combined model.
- **Descriptive local names are easy.** Driftwood Coffee Co, Golden Hour Bakery and Little Owl Espresso are 1.00 under every feature set.
- **Housing, Insurance & Fees, Utilities and Subscriptions transfer well.** Per-class F1 on unseen merchants is 0.91–1.00 with both feature sets, because their text ("APARTMENTS", "AUTOPAY", "FEE") and the `ach` channel are shared across merchants.
- **Income is trivial.** Its F1 is 1.000 on known merchants under every feature set, which is why it's left out of the headline ([§4](#4-income)).
- **Embedding cost is negligible.** 12.8k unique normalized strings embed in about 10 s on a laptop CPU through `fastembed` (ONNX, no PyTorch).

## Goals and non-goals

**Goals**

1. Every transaction gets one of the 12 spending categories or Income, plus a confidence in [0, 1].
2. Known-merchant macro F1 ≥ 0.90, above the keyword baseline, reproducible on rerun (NFR-8).
3. Unseen-merchant macro F1 is measured and reported with a confidence interval from the first commit, so FR-4 starts from a number.
4. Confidence is calibrated well enough for FR-5 to threshold it later (reported as expected calibration error).
5. Model code never reads ground truth. Labels reach the model only as a training argument.
6. Single-transaction latency is well inside the chat budget (NFR-5): target < 5 ms per transaction on CPU for the default model.

**Non-goals**

- Reaching 0.80 on unseen merchants (FR-4).
- Low-confidence review, user corrections, per-user overrides (FR-5, FR-6; P1).
- The full evaluation harness and report for other models. This design builds the categorization slice and the shared pieces it needs.
- Persisting predictions inside the generator's SQLite file (see [option C](#c-where-predictions-live)).

## What it will look like

### Usage

```sh
uv run sfc-categorize train    --data data/synthetic/default.sqlite --config configs/models/categorization.yaml
uv run sfc-categorize evaluate --data data/synthetic/default.sqlite --model artifacts/categorizer/<version>
uv run sfc-categorize predict  --data data/synthetic/default.sqlite --model artifacts/categorizer/<version> --out data/predictions/default.sqlite
```

```python
from smart_financial_coach.data.store import load_transactions
from smart_financial_coach.intelligence.categorization import load_categorizer

categorizer = load_categorizer("artifacts/categorizer/<version>")
txns = load_transactions("data/synthetic/default.sqlite", user_id="u_te_yp_0007")  # model-visible only
categorizer.categorize(txns)  # -> DataFrame[transaction_id, category, confidence, model_version]
```

### Service contract

The Technical Design fixes `categorize(merchant_raw, amount) → {category, confidence}`. This design proposes widening it to whole transaction rows, batch-first:

```python
class Categorizer(Protocol):
    version: str
    categories: tuple[str, ...]          # 12 spending categories + Income, from meta

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> Self: ...
    def categorize(self, transactions: pd.DataFrame) -> pd.DataFrame: ...
    #   in:  transaction_id, user_id, ts, amount, currency, merchant_raw, channel
    #   out: transaction_id, category, confidence, model_version
```

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

`artifacts/categorizer/<version>/` holds the fitted pipeline (`model.joblib`), the embedding model name and revision, and `manifest.json`: config hash, training data `spec_hash`, git commit, seeds, metrics. The version is derived from the config hash and the data `spec_hash`. Loading checks the manifest. Only artifacts from this folder are loaded, since joblib files are pickles.

## Design

### 1. Merchant-text normalization

A pure function in the feature pipeline, `normalize_merchant(merchant_raw) -> str`, shared with the anomaly service:

- Strip processor and channel prefixes (`SQ *`, `TST*`, `PAYPAL *`, `SP *`, `POS DEBIT`, `ACH DEBIT`, `ACH CREDIT`).
- Drop reference codes after `*`, store and terminal numbers, `PPD ID:` and other digit runs.
- Lower-case and collapse whitespace. Trailing location suffixes are kept, since truncation makes them unreliable to strip and n-grams tolerate them.

The prefix list is generic bank-feed vocabulary, written by hand from the rendering rules' *types*. It is not read from the catalog, so no holdout names leak in. In feasibility it collapsed 194k raw strings to 12.8k.

The raw text keeps its own feature (character n-grams on `merchant_raw`), because prefixes carry signal: `TST*` is Toast, a restaurant point-of-sale system.

### 2. Keyword baseline

The Technical Design's baseline is keyword rules. To be a fair baseline, not a leak:

- A committed `configs/models/category_keywords.yaml` maps generic words to categories (`coffee`, `cafe`, `grill`, `pizza` → Dining; `airlines`, `hotel` → Travel; `insurance`, `fee` → Insurance & Fees; `electric`, `water`, `wireless` → Utilities …).
- **No merchant names**, so the baseline gets no unfair advantage from the catalog. A unit test fails if any keyword equals a catalog canonical name token that isn't a common English word.
- Positive amounts with no keyword match → Income. Anything else unmatched → the training set's most frequent spending category (Dining).

### 3. Default model: linear text categorizer

The feasibility "both" configuration, made into a reproducible scikit-learn pipeline:

| Feature block | Source | Notes |
| --- | --- | --- |
| Character 2–4-grams (`char_wb`), TF-IDF, sublinear | `merchant_raw` | Fitted on training rows only |
| Sentence embedding (384-d) | `normalize_merchant(merchant_raw)` | Pretrained, frozen; cached per unique normalized string |
| Amount | `log1p(abs(amount))`, binned | Bins, not a raw number, so the linear model can learn non-monotone price bands |
| Sign | `amount > 0` | Separates Income from refunds together with text |
| Channel | one-hot | `card_present`, `online`, `ach`, `other` |
| Hour of day | 3-hour bins of local `ts` | Kept only if the ablation shows it helps (milestone 3) |

- **Classifier:** multinomial logistic regression, balanced class weights, regularization chosen on a validation split.
- **Confidence:** the predicted class's probability after calibration on the validation split (temperature or isotonic, chosen by expected calibration error).
- **Why this default:** it meets the FR-3 target by a wide margin, trains in seconds on CPU, and stays explainable (top n-grams per class). The other families in the Technical Design (gradient-boosted trees, small fine-tuned transformer) compete with it in milestone 3 under the selection criteria, in order.

### 4. Income

Income is predicted as a 13th class, and macro F1 is reported over the 12 spending categories, with Income's F1 reported on its own line.

- A sign rule alone fails: 5,422 Shopping refunds are positive.
- Keeping Income in the model lets text plus sign separate payroll from refunds, and `get_spending_summary` needs both anyway.
- Excluding Income from the headline matches the PRD's "about 12" and keeps a near-perfect class from inflating macro F1.

This closes the open question carried from FR-1 and FR-2, pending review.

### 5. Training pipeline

Owned by the evaluation side (`evaluation/categorization.py`), which is allowed to read truth:

1. Load model-visible transactions and `truth_transactions.category` for train users.
2. **Split:** the Technical Design's stratified 80/20 by transaction within train users is the known-merchant test. A further 10% of the 80% is the validation split, for regularization and calibration.
3. **Label noise:** flip a configurable share of training labels uniformly to another category. The default is 2%, a Technical Design control against flattering results. It applies to training rows only, and the report states the rate.
4. **Cap per class:** at most `max_rows_per_class` training rows per class (default 20k), sampled with a fixed seed. Dining has 418k rows, so this keeps training fast and balanced without dropping rare classes.
5. Fit, calibrate, write the artifact and manifest.

### 6. Evaluation

| Test set | Definition | Metric |
| --- | --- | --- |
| Known merchants | Train users, the held-out 20% | Macro F1 (12 spending), per-class F1, confusion matrix. **Gate: ≥ 0.90 and > baseline** |
| All test users | Every test-user transaction | Macro F1. The realistic mix of known and new |
| Unseen merchants | Test-user transactions at holdout merchants | Macro F1 with a 95% interval from a **merchant-level** bootstrap. Reported, not gated (FR-4) |
| Ambiguity ceiling | Majority-category oracle on known merchants | Per-class F1 for Groceries and Shopping, shown next to the model's |
| Refunds | Positive-amount spending transactions | Accuracy, so refunds aren't silently called Income |

Also reported: expected calibration error, accuracy at confidence ≥ 0.9 and the share of transactions above that threshold (FR-5 groundwork), and p50 / p95 latency for a one-row batch and for 10k rows.

**Why a merchant-level bootstrap:** the unseen set has 78k transactions but only 58 merchants, as few as 3 for Travel and Entertainment. One merchant decides a class's F1, so a transaction-level interval would be falsely narrow.

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
  intelligence/categorization/
    __init__.py                     Categorizer protocol, load_categorizer
    baseline.py                     KeywordCategorizer
    linear.py                       TextCategorizer (n-grams + embeddings + side features)
    embeddings.py                   cached sentence embeddings (fastembed)
    artifact.py                     save / load, manifest
  evaluation/
    splits.py                       categorization splits (reused by later models)
    categorization.py               train, evaluate, report
    cli.py                          sfc-categorize train | evaluate | predict
configs/models/
  categorization.yaml               features, regularization grid, label noise, seeds, caps
  category_keywords.yaml            baseline rules
```

New dependency: `fastembed` (ONNX runtime, about 130 MB model download, cached). Tests that need the embedding model are marked `slow`. Unit tests use a stub embedder so CI doesn't download it (see [option B](#b-text-representation)).

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
| **(b) N-grams + frozen sentence embeddings (recommended)** | Best on every test set (0.99 / 0.93 / 0.64); CPU, seconds | New dependency and a model download; pins an external model revision |
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

## Testing

- **Unit:** normalizer on every rendering distortion type (prefix, store number, reference code, truncation, `PPD ID:`), idempotent, never empty for non-empty input. Baseline keyword file contains no catalog merchant names. The contract returns one row per input with a known category and confidence in [0, 1]. An empty frame returns an empty frame. `fit` then `categorize` is deterministic for a fixed seed. Label noise flips the configured share and never keeps the original label. Splits: no transaction in two splits, unseen set contains only holdout merchants.
- **Integration (`small.yaml`, stub embedder):** train → evaluate → predict runs end to end, beats the baseline on known merchants, and a rerun gives identical metrics (NFR-8).
- **Slow (`default.yaml`, real embedder):** known-merchant macro F1 ≥ 0.90; latency within budget.
- **Isolation:** the existing test covers `intelligence/categorization/`.

## Milestones

1. `data/store.py`, `normalize_merchant`, and the keyword baseline, with the evaluation slice (splits, metrics, report) running on the baseline. This sets the baseline number.
2. The default linear categorizer with n-grams and side features, artifact and CLI. Confirms the 0.90 gate.
3. Embeddings, calibration, and the experiment round: feature ablation (hour, channel), gradient-boosted trees, and a small fine-tuned transformer if time allows. Record the decision in the report.
4. `predict` writes `transaction_categories`; latency check.
5. Update the Technical Design (contract, Income, predictions store) and close the Income question in FR-1 and FR-2.

## Decisions and open questions

**Decisions needed from review**

- [ ] FR-3 / FR-4 split as above: FR-3 gates on known merchants, FR-4 owns the unseen-merchant gap.
- [ ] Batch contract over model-visible rows (A-b).
- [ ] N-grams + frozen sentence embeddings as the default, adding `fastembed` (B-b).
- [ ] Predictions in a separate SQLite file (C-b).
- [ ] Income as a predicted class, excluded from the headline macro F1 (D-b).
- [ ] 2% uniform training label noise (E-b).

**Technical Design updates (after approval)**

- [ ] `categorize` contract: batch of transaction rows → category, confidence, model version.
- [ ] Data store: model outputs live in a separate predictions file per model version.
- [ ] Categorization evaluation row: headline macro F1 over 12 spending categories; merchant-level bootstrap interval for unseen merchants.

**Open questions**

- [ ] **FR-4 approach.** Embeddings get 0.64 (interval 0.56–0.73) against the 0.80 target. Options for the FR-4 design: a larger embedding model; training on one row per unique merchant string, so frequent merchants don't dominate; or an LLM fallback for low-confidence, never-seen strings, cached per normalized merchant so cost scales with merchants, not transactions. The LLM option would make categorization depend on the LLM provider, which NFR-6 avoids for the dashboard. A cached fallback would keep the dashboard working during an outage, but new merchants would wait.
- [ ] Is the unseen-merchant target realistic with 3–10 merchants per class? The feasibility interval is 0.17 wide, so near the target a pass or fail would be mostly luck. FR-4 may need a larger holdout or a merchant-level metric, decided the way FR-2 decided weekly spikes.
