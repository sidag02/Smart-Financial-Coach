# FR-3 Categorization Model Selection

Oct 1, 2026 · @Sidd · Status: **Proposed** · Companion to FR-3 Transaction Categorization — Feature Design

## Summary

This note records which categorization model v1 starts with, what else was considered, why the choice was made, and what evidence would change it.

- **Selected:** multinomial logistic regression over character n-grams of the raw merchant text, frozen sentence embeddings of the normalized text, and amount, sign and channel features. Confidence is calibrated separately for merchant strings the model has seen and strings it hasn't.
- **Why:** it clears the FR-3 gate by a wide margin (0.988 macro F1 on known merchants against 0.90), is the best measured option on known merchants and all test users, trains in seconds on a CPU, and stays explainable.
- **Not settled:** no measured option reaches FR-4's 0.80 on unseen merchants (best 0.643). Embeddings alone tie the combination there, so FR-4 may change the model; the triggers are in [What would change the selection](#what-would-change-the-selection).
- **Method:** the Technical Design's selection criteria, in order: primary metric on held-out data → beats the baseline → latency within budget → explainability → operational simplicity. Candidates are ruled out by theory where possible and compared by experiment otherwise.

All numbers come from the POC on `poc/fr-3-categorization`, pinned to commit [`55d4197`](https://github.com/sidag02/Smart-Financial-Coach/tree/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization) (default dataset, seed 0). Macro F1 is over the 12 spending categories.

## What the model has to do

The choice is constrained by more than accuracy. Each requirement below rules options in or out.

| Requirement | Source | Consequence for the model |
| --- | --- | --- |
| Macro F1 ≥ 0.90 on known merchants | PRD, FR-3 | Must memorize merchant strings well, including rare classes (Travel has 2.9k rows against Dining's 418k) |
| Macro F1 ≥ 0.80 on merchants never seen | PRD, FR-4 | Needs knowledge from outside the training data, since an unseen brand name says nothing to a model trained only on other names |
| Every transaction gets a category, consistently | FR-3, NFR-8 | Deterministic output; same input, same category |
| Confidence that FR-5 can threshold | FR-5 (P1) | Calibrated probabilities, on new merchants as well as known ones |
| Learns from user corrections | FR-6 (P1) | Labels come in as a training argument; cheap to retrain or override |
| Dashboard keeps working without the LLM | NFR-6 | The primary path can't depend on a hosted LLM |
| Dashboard < 2 s; chat < 8 s at p95 | NFR-5 | Milliseconds per transaction on CPU |
| AI cost per user tracked and in budget | NFR-9 | Per-transaction API calls are a cost risk at 1M+ rows |
| Model code never reads ground truth | FR-2 §5 | Training code passes labels in; nothing under `intelligence/` reads `truth_*` |

## Options

Each option is either measured in the POC, ruled out by theory, or scheduled for an experiment.

| Option | How it works | Measured result (known / all test users / unseen) | Strengths | Weaknesses | Status |
| --- | --- | --- | --- | --- | --- |
| Keyword rules | Generic words (`coffee`, `airlines`, `fee`) map to categories | Not yet measured | Transparent; no training | Brittle on messy text and brand names | **Baseline** the model must beat (Technical Design) |
| Exact-match lookup | A table from normalized merchant string to its most common training category | Not measured | Trivial, fast, near-perfect on known strings | Undefined for any new string, so it can't serve FR-4 alone | Useful as a component: per-user overrides for FR-6 |
| Linear model on character n-grams | TF-IDF of 2–4-character grams + side features, logistic regression | 0.986 / 0.868 / 0.504 | Strong memorization; fast; explainable by top n-grams | Knows nothing beyond training strings; over-confident on new ones (ECE 0.160) | Measured; becomes one half of the selected model |
| Linear model on sentence embeddings | Frozen `bge-small-en-v1.5` vector of the normalized text + side features | 0.973 / 0.913 / **0.643** | Brings in outside knowledge ("Taco Bell" is food); best on online services | Weaker on known merchants; loses exact-string identity | Measured; candidate route for unfamiliar strings (FR-4) |
| **Linear model on both** | Both feature blocks concatenated | **0.988 / 0.929** / 0.638 | Best on known merchants and all test users; best calibrated (ECE 0.009 known, 0.056 unseen) | N-grams sometimes outvote the embedding on unfamiliar strings (Caviar 1.00 → 0.64) | **Selected for FR-3** |
| Gradient-boosted trees | Trees over dense features: embeddings, amount, channel, hour | Not measured | Non-linear interactions (e.g. amount × channel for ambiguous merchants); strong on tabular data | Poor fit for 200k sparse n-gram columns; larger artifacts; less transparent | Milestone 3 experiment, on dense features only |
| Nearest neighbours over embeddings | Category of the closest known merchant strings | Not measured | No training; easy to add merchants; natural "similar to X" explanation | Needs an index; sensitive to embedding quality; hubness | FR-4 candidate |
| Fine-tuned small transformer | Fine-tune a small text model on merchant strings | Not measured | Possibly best on unseen merchants | GPU likely; slower; heavier operations and artifacts | Milestone 3 if time allows, or FR-4 if cheaper options fall short |
| LLM, zero- or few-shot, every transaction | Prompt a hosted LLM with the merchant text | Ruled out for the primary path | World knowledge; handles new categories without retraining | Cost and latency at 1M+ rows (NFR-5, NFR-9); outages break the dashboard (NFR-6); output can vary between calls (NFR-8) | **Ruled out by theory** as the primary path |
| LLM fallback, cached per merchant | Only low-confidence, never-seen strings go to the LLM; the answer is cached per normalized merchant | Not measured | Cost scales with new merchants, not transactions; uses world knowledge where it is missing | Depends on the LLM provider; new merchants wait during an outage | FR-4 candidate |
| External merchant database | Look up merchants in an enrichment service | Not applicable to v1 | Real-world coverage | Synthetic merchants aren't in it; vendor dependency | Revisit for v2 (real bank data) |

## Why this model

The selection follows the Technical Design's criteria in order.

1. **Primary metric on held-out data.** The combination is best on known merchants (0.988) and on all test users (0.929). On unseen merchants it ties embeddings alone (0.638 vs 0.643; the 95% intervals, 0.56–0.73 and 0.55–0.73, overlap almost completely). FR-3 gates on known merchants, so the combination wins.
2. **Beats the baseline.** The keyword baseline is built in milestone 1. With the model at 0.988, a gap is expected; the milestone confirms it.
3. **Latency.** Inference is one sparse matrix product plus an embedding lookup. Embeddings are cached per unique normalized string (12.8k on the default dataset, embedded in about 10 s), so a new transaction usually needs no embedding call at all.
4. **Explainability.** Linear weights show which n-grams and features drove a category. That supports FR-16 (show where numbers came from) later.
5. **Operational simplicity.** scikit-learn and an ONNX embedding model on CPU, with no GPU and no service. Training takes about 30 s.

**Why logistic regression rather than another classifier.** It gives probabilities that are already well calibrated on known merchants (ECE 0.009). It handles 200k sparse features natively, trains deterministically, and copes with class imbalance through class weights. Nothing measured so far suggests the classifier, rather than the features, is the bottleneck.

**Why the regularization strength doesn't decide anything.** C was fixed by hand before the first run (10 for n-grams, 3 for embeddings) and not tuned. At C = 1, 3 and 10, unseen-merchant F1 moves by at most 0.03 and known-merchant F1 by at most 0.002. The ranking of options doesn't change. The production pipeline chooses C on validation splits.

**Why calibration is split by familiarity.** The model is calibrated where it is already confident and right (known merchants) and over-confident where it is often wrong (unseen merchants). One calibrator fitted on known merchants would leave that untouched. Calibrating seen and unseen strings separately targets the case FR-5 needs, and "seen" is model-visible, so it works the same way in production.

## What would change the selection

These are the conditions under which another model should be tried, and the evidence that would decide it. Each is checked on the same splits, seeds and metrics as the POC, and the result is recorded in the evaluation report.

| Trigger | What it means | Options to evaluate | Decided by |
| --- | --- | --- | --- |
| **FR-4 gate not met** (expected; best is 0.643 against 0.80) | The default can't categorize new merchants well enough | Routing by familiarity (n-grams for seen strings, embeddings for unseen); a larger embedding model; training on one row per unique string; nearest neighbours; cached LLM fallback | Unseen-merchant macro F1 and its merchant-bootstrap interval on the merchant-holdout validation split, then the test users |
| **The 0.80 gate itself changes** | With 3–10 holdout merchants per class, the interval is 0.17 wide | Same as above, judged against the revised gate | FR-4 design, as FR-2 did for weekly spikes |
| **Calibration error on unseen merchants stays high after calibration** | FR-5 can't trust thresholds on new merchants | Ensembles or models with better uncertainty; abstaining (routing to review) below a threshold | ECE on unseen strings; accuracy and coverage at the review threshold |
| **A different model wins milestone 3** | Gradient-boosted trees or a fine-tuned transformer beat the default on held-out macro F1 | Switch only if it also passes latency, explainability and operations, in that order | The Technical Design's selection criteria |
| **Ambiguous merchants matter more** | Warehouse clubs and marketplaces cap Groceries and Shopping at about 1.3% error from text alone | Features beyond text: amount bands per merchant, the user's own history at that merchant; trees handle these interactions well | Groceries and Shopping per-class F1 against the majority-category oracle |
| **Real bank data (v2)** | Messier text, many more merchants, different class mix | Re-run every option on real data before launch; enrichment services become possible | All metrics re-validated on real data (PRD launch gate) |
| **User corrections arrive (FR-6)** | Labels change continuously | Per-user override table in front of the model, plus periodic retraining; models that update cheaply are favored | Correction rate and whether corrected merchants stay corrected |
| **Latency or cost budget tightens** (NFR-5, NFR-9) | Embedding or LLM calls become too slow or expensive | Keep the string-level embedding cache; drop to n-grams only for seen strings | p95 latency and cost per active user |
| **The taxonomy changes** | Categories are added or split | Retraining handles it for supervised models; zero-shot options handle it without labels | Per-class F1 on the new categories |

**Not a trigger:** a small improvement on known merchants alone. The default is already at 0.988, and the remaining errors are mostly the ambiguous-merchant ceiling.

## Open questions

- [ ] Whether FR-4 routes by familiarity, which would make the model a small ensemble, or keeps a single model with better features.
- [ ] Whether the LLM fallback is acceptable under NFR-6, given that new merchants would wait during an outage.
