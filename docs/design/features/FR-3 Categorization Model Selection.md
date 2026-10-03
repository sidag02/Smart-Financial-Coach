# FR-3 Categorization Model Selection

Oct 1, 2026 (decided Oct 2; superseded Oct 3) · @Sidd · Status: **Accepted, superseded by FR-4's promotion** · Companion to FR-3 Transaction Categorization — Feature Design

## Current selection (Oct 3, 2026): superseded by FR-4

FR-4's round replaced the model below. The rest of this note records FR-3's choice as it was made.

- **Promoted:** `21_small_unweighted.ship`, version `20eea4fb-44781bc4-c0274576`, in the GitHub Release `categorization-20eea4fb-44781bc4-c0274576`. It is the same family: calibrated logistic regression over character n-grams, frozen embeddings, amount, sign, channel and hour.
- **What changed:**
  - **bge-small** instead of bge-base: tied on quality, about a third of the batch cost;
  - **no class weights**;
  - **clean training labels** (`label_noise: 0`; injected noise stays for comparing candidates);
  - trained on the default dataset with FR-4's 40% holdout (data hash `44781bc4e4a5`).
- **Test set** (scored once, by FR-4's `finalize`): known merchants 0.983, all test users 0.943, unseen merchants **0.735** (0.67–0.81). It passes FR-3's gates and FR-4's (unseen ≥ 0.66 and above keyword).
- **The known issues below are resolved:**
  - Travel is no longer the fallback for unfamiliar merchants (158 predictions against 142 true at test users' unseen merchants);
  - known-merchant confidence is calibrated (ECE 0.009, 95% of rows at 0.9 or more).
- **New known issue:** on unseen merchants, Health & Fitness is barely predicted (validation F1 0.105) and Entertainment is confused with Dining (0.49). It's diagnosed in the round's report, to be fixed after launch through feedback.
- **Evidence:** FR-4 Unseen Merchant Categorization — Feature Design, and `docs/reports/FR-4 Categorization — Round Results.md`.

## Summary

This note records which categorization model v1 starts with, what else was considered, why the choice was made, and what evidence would change it.

- **Selected and promoted (Oct 2, 2026):** multinomial logistic regression over character 2–4-grams of the raw merchant text, frozen `bge-base-en-v1.5` sentence embeddings of the normalized text, and amount, sign, channel and hour features. Confidence is calibrated separately for strings the model has seen and strings it hasn't. Version `3f0ccc82-2f0e60a6`, from MLflow run `000ef7d3` (`20_linear_base`); its model file is in the GitHub Release `categorization-3f0ccc82-2f0e60a6`.
- **Why:** the FR-3 decision rule ranked it first. It ties every launch candidate on unseen merchants and has the best-calibrated confidence there. On the test sets, scored once, it passes both gates: known-merchant macro F1 **0.970** against 0.90 and the keyword baseline's 0.461.
- **Known issue, shipped knowingly:** on unseen merchants, Travel is its fallback guess (30% of those transactions, nearly all wrong, at confidence about 0.5). FR-5 review and feedback retraining are the remedy ([Known issues](#known-issues)).
- **Not settled:** nothing reaches FR-4's 0.80 on unseen merchants (0.462 on test). That is FR-4's design.
- **Method:** the Technical Design's selection criteria, applied through the FR-3 decision rule on validation data. Test sets were scored once, for three finalists and the baselines. Evidence: FR-3 Categorization — Launch Round Results (`docs/reports/`), and the POC on `poc/fr-3-categorization` (commit [`55d4197`](https://github.com/sidag02/Smart-Financial-Coach/tree/55d4197cef6c69e23bae81d2721394e27a7f9a4f/experiments/fr3_categorization)).

Macro F1 is over the 12 spending categories. Every launch-round number is under 2% uniform training label noise (FR-3 §5); the POC's numbers are noise-free.

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
| Categorized on ingestion, in batches across users | Technical Design (compute timing) | Serving cost sizes the cluster; it is not a gate (decision Oct 2, 2026) |
| AI cost per user tracked and in budget | NFR-9 | Per-transaction API calls are a cost risk at 1M+ rows |
| Model code never reads ground truth | FR-2 §5 | Training code passes labels in; nothing under `intelligence/` reads `truth_*` |

## Options

Each option is measured, ruled out by theory, or deferred. Launch-round numbers are validation (known / unseen merchants) unless marked test.

| Option | How it works | Measured result | Strengths | Weaknesses | Status |
| --- | --- | --- | --- | --- | --- |
| Keyword rules | Generic words (`coffee`, `airlines`, `fee`) map to categories | 0.474 / 0.425; test 0.461 known, 0.482 unseen | Transparent; no training; honest confidence (best unseen Brier) | Brittle on brand names; refund accuracy 0.03 | **The baseline** every model must beat |
| Exact-match lookup | Normalized string → its most common training category | 0.980 / 0.074; test 0.874 on all test users | Near-perfect on known strings; robust to uniform label noise | Useless on any new string | Reference point. Candidate front stage: `Lookup(fallback=model)` (deferred round 6) |
| Linear, n-grams + `bge-small` | The POC's "both" configuration | 0.968 / 0.503; test 0.970 / 0.471 | Half the batch cost of bge-base; best test unseen Brier among finalists | Tied, ranked third on unseen Brier | Finalist #3 |
| Linear, n-grams 1–5 + `bge-small` | Wider n-gram range | 0.965 / 0.504; test 0.967 / 0.472 | Same cost as bge-small | No measurable change from 2–4 | Finalist #2 |
| **Linear, n-grams + `bge-base`** | Larger embedding model | **0.969 / 0.512; test 0.970 / 0.462** | Best validation unseen confidence (Brier 0.216, accuracy at 0.9: 97%) | 2.8× bge-small's batch cost; twice its ONNX memory | **Selected and promoted** |
| One calibrator for all strings | Same model, calibration not split by familiarity | 0.968 / 0.503 | Simpler | Unseen confidence ≥ 0.9 right 61% of the time instead of 92% | Rejected: the split stays |
| Gradient-boosted trees | Trees over dense features | Not measured | Non-linear interactions (amount × channel for ambiguous merchants) | Poor fit for sparse n-grams; larger artifacts | Deferred (round 5) |
| Nearest neighbours over embeddings | Category of the closest known strings | Not measured | No training; "similar to X" explanation | Needs an index; hubness | Deferred to FR-4 |
| Fine-tuned small transformer | Fine-tune a small text model | Not measured | Possibly best on unseen merchants | GPU likely; heavier operations | Deferred (round 8) |
| LLM on every transaction | Prompt a hosted LLM | Ruled out | World knowledge | Cost at 1M+ rows (NFR-9); outages break the dashboard (NFR-6); not deterministic (NFR-8) | **Ruled out by theory** |
| LLM fallback, cached per merchant | Only low-confidence new strings go to an LLM | Not measured | Cost scales with new merchants | Provider dependency; new merchants wait in an outage | FR-4 candidate |
| External merchant database | Enrichment service | Not applicable to v1 | Real-world coverage | Synthetic merchants aren't in it | Revisit for v2 |

## Why this model

The Technical Design's criteria, in order, as applied by the FR-3 decision rule:

1. **Primary metric on held-out data.** The four launch candidates tie on validation unseen-merchant macro F1 (0.503–0.512; every paired difference from the leader spans 0). All clear the 0.90 known-merchant floor.
2. **Beats the baseline.** All do, on validation and on test (0.970 against keyword's 0.461 known).
3. **Tie-breaks.** First, unseen-merchant Brier score after calibration: bge-base is best (0.216, against 0.224 and 0.229). Second, batch serving cost, which wasn't needed because Brier already separated every tied run.
4. **Explainability.** Linear weights show which n-grams, embedding dimensions and side features drove a category.
5. **Operational simplicity.** scikit-learn and an ONNX embedding model on CPU. Measured on a laptop: about 12,200 transactions per second per process, 2.1 GB peak memory at 20k-row batches, 0.6 s to load.

**Retraining cost** (new consideration for the feedback loop, not yet in the Technical Design's criteria): one calibrated configuration takes about 14 minutes on a laptop CPU with fixed C and 3 folds, and about 8 minutes with bge-small.

**Test scores weren't used to choose.** On test, bge-small and n-grams 1–5 score slightly higher on unseen merchants (0.471–0.472 against 0.462) and on unseen Brier (0.208 against 0.248). All three intervals overlap almost completely (about 0.39–0.59), and the rule doesn't re-rank finalists on test scores.

## What lost, and why

- **bge-small and n-grams 1–5:** tied on F1, ranked below bge-base on validation confidence. Either would be a reasonable choice; bge-small is the fallback if serving cost matters more later, at about a third of the batch time.
- **One calibrator:** same categories, much worse confidence on unseen merchants. Per-familiarity calibration stays.
- **Lookup:** best on known strings and on all test users (test 0.874 against 0.796), worthless on unseen merchants. It returns as the front stage of a composition, not as the model.
- **Keyword rules:** the baseline. They edge the finalists on test unseen F1 (0.482, overlapping intervals) but misallocate more unseen spend (49% of dollars against 36%) and almost never get refunds right.

## Deferred

The launch round was trimmed (FR-3 design, "Trimmed for launch"). These rounds aren't dropped. They are the menu for the first retraining design:

- **Round 2, rest:** word n-grams, raw vs normalized text for n-grams, other embedding models (bge-large was conditional on bge-base winning on F1; it tied).
- **Round 3:** side-feature ablations (hour stays on).
- **Round 4:** training regime: per-class caps, one row per unique string, class weights, **noise levels**.
- **Round 5:** linear SVM, kNN, boosted trees.
- **Round 6:** `Routed`, `Lookup(fallback=...)`, stacking.
- **Round 8:** fine-tuned transformer.

## Decisions made along the way

- **2% uniform label noise kept** (Oct 1, 2026). Every candidate is compared under it, so comparisons are fair. Absolute numbers sit well below the noise-free POC, especially for rare categories and unseen merchants. The shipped model is trained under it too, which is the likely cause of the Travel issue below.
- **Serving cost reported, not gated** (Oct 2, 2026, after the launch round's results). Categorization runs batched on ingestion; batch cost is the second tie-breaker.
- **Promoted model files live in GitHub Releases** (Oct 2, 2026). The manifest with the file's checksum is committed and is what a downloaded file must match; continuous deployment will move to a container registry.
- **Ship bge-base despite the Travel issue** (owner, Oct 2, 2026, after seeing test results). A decent cold-start model is enough; feedback retraining (FR-5, FR-6) decides quality after launch.

## Known issues

- **Travel is the fallback for unfamiliar merchants.** On the default dataset the promoted model calls 27,271 transactions Travel against 2,851 true ones. At holdout merchants it calls 30% of rows Travel, nearly all wrongly, at average confidence about 0.5; at known merchants, accuracy is 0.983. Likely mechanism: uniform noise puts as many wrong labels into Travel as into any category (39% of Travel's training labels are wrong), and balanced class weights give that rarest class the most weight. On the dashboard, a user's Travel spend will be inflated by new merchants until they are reviewed or corrected.
- **Under-confident on known merchants.** Known-merchant ECE is 0.153 on test: only 33% of rows reach confidence 0.9, and those are all right. Cross-validated Brier chose no calibration for familiar strings. FR-5 should pick its threshold from the reliability curve.

## What would change the selection

Each trigger is checked on the same splits and metrics, and the result recorded in the evaluation report.

| Trigger | What it means | Options to evaluate | Decided by |
| --- | --- | --- | --- |
| **Travel fallback persists after the first feedback** | New merchants keep inflating Travel | Training without injected noise (decided for the next shipped model in the FR-5 and FR-6 design); no class weights; `Lookup` plus a calibrated abstain | Travel precision on unseen merchants; misallocated spend |
| **FR-4 gate not met** (expected; 0.462 on test against 0.80) | The model can't categorize new merchants well enough | Routing by familiarity; larger embedding models; one row per unique string; nearest neighbours; cached LLM fallback | Unseen-merchant macro F1 on validation, then on test with its merchant-bootstrap interval |
| **The 0.80 gate itself changes** | With 3–10 holdout merchants per class, the interval is about 0.2 wide | Same as above, judged against the revised gate | FR-4 design |
| **FR-5 can't find a useful threshold** | Confidence too flat on known merchants or too sharp on unseen ones | Recalibrate familiar strings; ensembles; abstention | Accuracy and coverage along the reliability curve |
| **A deferred round wins** | Trees, kNN or a transformer beat the default on held-out macro F1 | Switch if it also passes explainability and operations | The decision rule |
| **Ambiguous merchants matter more** | Warehouse clubs and marketplaces cap Groceries and Shopping | Per-merchant amount bands; the user's own history at that merchant | Groceries and Shopping F1 against the majority-category oracle |
| **Real bank data (v2)** | Messier text, more merchants, different class mix | Re-run every option on real data before launch | All metrics re-validated (PRD launch gate) |
| **User corrections arrive (FR-6)** | Labels change continuously | Per-user overrides in front of the model; scheduled retraining; models that retrain cheaply | Correction rate and whether corrected merchants stay corrected |
| **Serving cost becomes binding** | Cluster cost for the ingestion batch grows | bge-small (about a third of the batch time, tied on F1); n-grams only for seen strings | Rows per second per core; cost per million transactions |
| **The taxonomy changes** | Categories are added or split | Retraining handles it for supervised models | Per-class F1 on the new categories |

## Open questions

- [ ] Whether FR-4 routes by familiarity, which would make the model a small ensemble, or keeps a single model with better features.
- [ ] Whether the LLM fallback is acceptable under NFR-6, given that new merchants would wait during an outage.
- [x] Whether the shipped model should train on injected label noise at all, or only the experiments that compare candidates (the Travel issue). **Decided Oct 2, 2026** in the FR-5 and FR-6 design: shipped and retrained models train on clean labels; injected noise stays for comparing candidates.
