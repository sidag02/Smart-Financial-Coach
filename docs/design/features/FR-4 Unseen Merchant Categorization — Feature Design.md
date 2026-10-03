# FR-4 Unseen Merchant Categorization — Feature Design

Oct 2, 2026 · @Sidd · Status: **Accepted** (owner, Oct 2, 2026; PR #17)

## Summary

This feature makes the categorizer usable on merchants it has never seen, measures that honestly, and sets a v1 target the evidence supports.

- **Requirement:** FR-4 (P0), *"Categorize merchants the system has never seen before."* PRD success metric: macro F1 ≥ 0.80 on new merchants.
- **Starting point:** the promoted model (bge-base, `3f0ccc82-2f0e60a6`) scores **0.462** (0.40–0.55) on unseen test merchants and 0.512 on validation. On unfamiliar merchants, Travel is its fallback guess (FR-3 Categorization Model Selection).
- **Feasibility, measured on validation data only** ([evidence](#feasibility)):
  - **Clean labels are the lever.** Training without the injected 2% label noise lifts unseen-merchant macro F1 from 0.512 to **0.714**. Dropping balanced class weights as well gives **0.745** (0.68–0.80). Known merchants rise from 0.969 to 0.988, the Travel fallback disappears (Travel F1 0.025 to 0.886), and misallocated unseen spend falls from 27% to 11%.
  - **Nothing else measured helps.** Routing unfamiliar strings to an embeddings-only model ties (0.704); nearest neighbours over embeddings is worse than the shipped model (0.468); bge-small ties bge-base (0.715) at a third of the batch cost.
  - **Today's test holdout is too small to judge 0.80.** With 58 holdout merchants, the same model scores anywhere from 0.43 to 0.62 depending on which merchants are drawn. A 40% holdout gives 115 merchants, narrows the interval by about a fifth and gives a fresh test set; the generator's data checks pass with it.
- **Approach:**
  1. **Ship a clean-label model.** The owner decided on PR #15 that shipped and retrained models train on clean labels, while injected noise stays for comparing candidates. FR-4 makes that mechanical: every compared configuration has a **shipping twin** with `label_noise: 0`, and the twins are what get finalized and promoted.
  2. **Add the class-weight choice to the candidates**, since unweighted training led on validation.
  3. **Measure on a bigger, fresh holdout:** the default dataset gets a 40% holdout with a new seed. This doubles the unseen merchants to 115, and gives a test set nothing has been scored on.
  4. **The v1 gate is 0.66** (owner decision, Oct 3, 2026; first accepted at 0.70 and lowered on the FR-4 round's validation results, before any test scoring; §4), with 0.80 as the v1.1 goal reached through feedback (FR-5/FR-6), as FR-2 did for weekly spikes. The PRD changes in milestone 5.
- **Excluded by owner decision:** an LLM fallback for unfamiliar strings. Categorization stays LLM-free; the LLM stays in the coach.
- **Principle (carried from FR-3):** models are chosen on validation data; test sets are scored once, for finalists.

## Context

Where the shipped model loses unseen-merchant macro F1 (validation, per category):

| Category | Merchants | Shipped | Clean labels | Clean, no class weights |
| --- | --- | --- | --- | --- |
| Travel | 8 | **0.025** (63,545 predicted, 848 true) | 0.683 | 0.886 |
| Health & Fitness | 10 | 0.075 | 0.142 | **0.156** |
| Entertainment | 8 | 0.289 | 0.358 | **0.463** |
| Insurance & Fees | 8 | 0.342 | 0.770 | 0.760 |
| Childcare & Education | 10 | 0.396 | 0.652 | 0.706 |
| Shopping | 26 | 0.580 | 0.758 | 0.801 |
| Groceries | 17 | 0.588 | 0.815 | 0.801 |
| Dining | 33 | 0.670 | 0.893 | 0.886 |
| Housing | 8 | 0.689 | 0.985 | 0.981 |
| Transportation | 14 | 0.696 | 0.731 | 0.719 |
| Subscriptions | 15 | 0.879 | 0.883 | 0.890 |
| Utilities | 12 | 0.916 | 0.905 | 0.892 |
| **Macro F1 / accuracy** | 169 | 0.512 / 0.581 | 0.714 / 0.817 | 0.745 / 0.825 |

- **The shipped model's losses were mostly the noise.** Uniform flips put about 39% wrong labels into the rarest class, Travel, and balanced class weights gave that class the most weight. Unfamiliar strings fell into it. Clean labels fix most categories at once.
- **What's left is two categories.** Health & Fitness (gyms, medical, pharmacies; over-predicted about 2×) and Entertainment (under-predicted) hold macro F1 below 0.80. Each counts one twelfth of the macro average.

## Scope

| | FR-3 (accepted) | FR-4 (this design) | FR-5/FR-6 (PR #15) |
| --- | --- | --- | --- |
| Gates on | Known merchants ≥ 0.90, beats the keyword baseline | Unseen merchants ≥ the v1 target (0.66), beats the keyword baseline on unseen merchants; FR-3's gates still apply | Personal accuracy, global gain after feedback |
| Changes | Framework, first model | Training labels, candidates, the holdout, the target | Review, corrections, retraining |

## Feasibility

**Evidence:** POC branch `poc/fr-4-unseen-merchants`, pinned to commit [`2541dc6`](https://github.com/sidag02/Smart-Financial-Coach/tree/2541dc654043c591afca906c543f6bf4240fc538/experiments/fr4_unseen). Every number below is in its [results](https://github.com/sidag02/Smart-Financial-Coach/blob/2541dc654043c591afca906c543f6bf4240fc538/experiments/fr4_unseen/results/feasibility.md), with the commands that reproduce them.

Setup: the launch round's 3 merchant-grouped validation folds (169 held-out merchants of train users, split hash `ce93ef87`), default dataset, one code version. All candidates are calibrated per familiarity group with C = 1.0. The runs went into a separate MLflow store, so nothing joined the FR-3 leaderboard. **No test set was scored.**

### Candidates

| Run | Unseen macro F1 (95% CI) | Paired vs leader | Known | Unseen Brier | Unseen misallocated spend | Batch ms per 10k |
| --- | --- | --- | --- | --- | --- | --- |
| bge-base, clean, no class weights | **0.745** (0.68–0.80) | – | 0.988 | 0.126 | 11.3% | 4,538 |
| bge-small, clean | 0.715 (0.65–0.77) | −0.058 to +0.005 (tied) | 0.988 | 0.134 | 12.4% | 1,562 |
| bge-base, clean | 0.714 (0.65–0.78) | −0.051 to −0.001 | 0.988 | 0.130 | 11.9% | 4,579 |
| Embeddings only, clean | 0.704 (0.64–0.77) | −0.077 to +0.005 (tied) | 0.980 | 0.130 | 14.1% | 4,384 |
| bge-base, noise, no class weights | 0.666 (0.59–0.72) | −0.121 to −0.043 | 0.988 | 0.148 | 16.7% | 4,691 |
| Embeddings only, noise | 0.536 (0.47–0.60) | −0.258 to −0.141 | 0.972 | 0.184 | 25.4% | 4,577 |
| **Shipped configuration** (bge-base, noise) | 0.512 (0.45–0.58) | −0.280 to −0.173 | 0.969 | 0.216 | 26.8% | 4,386 |
| Nearest neighbours (k = 10), clean or noise | 0.468 (0.38–0.55) | −0.352 to −0.201 | 0.97 | 0.21 | 51.7% | 4,450 |
| Keyword baseline | 0.425 | | 0.474 | 0.197 | 45.3% | 51 |

The reference reproduces the launch round's bge-base exactly (0.512).

What the runs show:

- **Clean labels: +0.20.** The largest effect measured in FR-3 or FR-4, on both unseen and known merchants. Calibration improves with it: unseen Brier falls from 0.216 to 0.130.
- **No class weights: +0.03 with clean labels, +0.15 with noise.** Balanced weights mostly amplified the noise. Without it they still cost a little, mainly on Travel and Entertainment.
- **Routing doesn't help.** FR-3 suggested sending unfamiliar strings to an embeddings-only model, because n-grams memorized from training merchants outvoted the embedding. With clean labels, embeddings-only scores 0.704 against the combined model's 0.714, so routing would at best tie.
- **Nearest neighbours is worse.** One example per known string, the weighted vote of the 10 closest strings. It is robust to noise (noise makes no difference to it), but the linear model generalizes much better from brand names to categories.
- **bge-small ties bge-base under clean labels** (0.715 against 0.714) at a third of the batch cost. The bge-small, unweighted combination wasn't run; it belongs in the FR-4 round.

### How precise is the unseen-merchant measurement?

Category-stratified subsets of the 169 validation merchants, scored with the shipped model's validation predictions. The merchant bootstrap resamples with replacement, so it treats merchants as independent draws, as a fresh holdout's are:

| Holdout merchants | Bootstrap SD | 95% interval width | P(≥ 0.70) if the true score is 0.745 | … if 0.73 |
| --- | --- | --- | --- | --- |
| **58** (today's test holdout) | 0.048 | 0.186 | 83% | 73% |
| 87 | 0.042 | 0.163 | 86% | 76% |
| **115** | 0.038 | 0.148 | 88% | 78% |

- **With 58 merchants, one holdout draw swings the score by about ±0.10:** subsets of the 169 put the same model anywhere from 0.43 to 0.62. FR-3's 0.462 on test against 0.512 on validation is within that luck. A 0.80 pass or fail on this holdout would say little about the model.
- **115 merchants narrow the interval by about a fifth** (SD 0.048 to 0.038; √(58/115) predicts about 29% for independent merchants). Three to six merchants per category is still few, so intervals stay wide; the target below is set with that in mind.
- 0.73 allows for 0.745 being the best of nine runs on the same folds. The pass rates use a normal approximation and the SD of this model (unseen F1 0.51), not the leader's.
- *Corrected in review:* the first version used the spread of subsets drawn without replacement, which share most merchants as N grows and so understate a fresh holdout's spread (SD 0.029 at 115, against 0.038). It said 115 merchants "halve" the spread.

### A larger holdout

A trial spec (`holdout40.yaml` in the POC) extends the default with a holdout share of 0.4, a new holdout seed and a lower test-user bias:

- **115 holdout merchants** against 58: Travel, Housing and Entertainment 6 each (from 3), Dining 19 (from 10).
- **23.5% of test users' spending at holdout merchants**, inside the generator's 15–25% check; 85k unseen-merchant test transactions. All data quality checks pass.
- With the default test-user bias (1.25), 43% of test spending lands at holdout merchants, so the bias has to come down with the share (0.45).
- **Cost:** train users see 176 spending merchants instead of 233. The 3-fold validation models above already train on about that many (two thirds of the holdout-eligible merchants plus the protected ones), so their numbers approximate what a model trained under the larger holdout can do.

## Goals and non-goals

**Goals**

1. A promoted categorizer trained on clean labels, chosen by the FR-3 decision rule among candidates compared under noise.
2. Unseen-merchant macro F1 measured on at least 100 holdout merchants, on a test set scored once.
3. A v1 target for unseen merchants that a good model passes reliably on that measurement, with 0.80 kept as the goal after feedback.
4. FR-3's gates (known merchants ≥ 0.90, beats the keyword baseline) still pass for the promoted model.
5. Shipping without injected noise is explicit and enforced, never a silent default.

**Non-goals**

- An LLM fallback (owner decision).
- Fixing Health & Fitness and Entertainment specifically: they are reported, and left to feedback and later rounds.
- The deferred FR-3 rounds (side features, caps, other classifiers) beyond the class-weight choice.
- Learning from feedback (FR-5/FR-6).

## Design

### 1. Comparison runs and shipping twins

The owner's rule (PR #15) separates two things that used to be one run:

- **Comparison runs** train under the injected 2% label noise, the Technical Design's control against flattering results. The leaderboard and the decision rule rank only these.
- **Shipping twins** are the same configuration with `label_noise: 0`, written explicitly. Only twins are finalized and promoted.

How the framework carries it:

- A config declares its twin with `ship: {task_params: {label_noise: 0}}`. `sfc-experiment run` runs both and tags the twin `sfc.twin_of=<comparison run>`.
- `sfc-experiment finalize` takes the decision rule's top three comparison runs, as now, and scores **their twins** on the test sets, once. A finalist without a finished twin is refused.
- `sfc-model promote` accepts only a twin whose config sets `label_noise` explicitly, and records the value in `promotions.jsonl`. A config that leaves it to the task default is refused, so a promotion can't silently train with or without noise.
- The comparison run and its twin share a split hash, so validation numbers for both are on one leaderboard for reporting. Ranking ignores twins.

**Does ranking under noise pick the right clean model?** It did here: without class weights wins under noise (0.666 against 0.512) and without noise (0.745 against 0.714). It isn't guaranteed, so the rule says in advance what each kind of run decides (§5):

- **Eligibility, ranking and the tie set** come from the comparison runs, as the owner's decision requires.
- **Tie-breaks come from the twins' validation metrics.** Noise roughly doubles unseen Brier (0.216 against 0.130 for the same bge-base configuration), so ordering tied runs by noisy Brier would choose on the calibration of models that don't ship. The twins' validation numbers are on the same splits and use no test data.
- **A reversal on F1 is a stop.** Every candidate has a twin, so the check covers **every eligible run's twin**, not only finalists: if any of them beats rank 1's twin on validation unseen F1 by a paired interval excluding 0, nothing is finalized until it's investigated, like a failed gate. It is not resolved by quietly promoting the other twin.

### 2. Candidates

The FR-4 round runs, on the new dataset (§3), each as a comparison run with a shipping twin:

| Candidate | Why |
| --- | --- |
| bge-base, balanced class weights | The shipped family |
| bge-base, no class weights | The validation leader |
| bge-small, balanced class weights | Ties bge-base under clean labels at a third of the cost |
| bge-small, no class weights | Not measured; the cheapest candidate that could lead |
| Keyword, lookup, majority (baselines) | The floor, as in FR-3 |

`class_weight` (`balanced` or `none`) becomes a `linear_text` parameter, as on the POC branch. Routing and nearest neighbours aren't carried forward (Feasibility). Calibration stays per familiarity group.

### 3. A larger, fresh holdout

- **The default spec changes:** holdout share 0.2 to **0.4**, holdout seed 7 to **8**, test-user bias 1.25 to **0.45**. The default dataset is regenerated.
- **Why change the default** rather than add a second dataset: every model and report reads one dataset. Two datasets would double every run and leave FR-3's and FR-4's numbers on different data.
- **Fresh test set:** the new seed holds out different merchants, and the split hash changes, so FR-4's finalists are scored on test data nothing has seen. FR-3's numbers on the old dataset remain as recorded; the FR-4 report re-states FR-3's gates on the new data.
- **One regeneration, with FR-5/FR-6's schema 4** (owner decision on #15, Oct 2, 2026): milestone 1 also adds that design's `truth_preferences`, so the default dataset is regenerated once. It builds from **#15's accepted data contract**: the owner accepted `truth_preferences` and the preference-aware label contract (#15 §7) ahead of the rest of that design, so milestone 1 doesn't implement an unaccepted design. The FR-5/FR-6 replay then runs on FR-4's promoted twin and this dataset.
- **FR-3's runs stay reproducible.** `rebuild_splits` refuses a dataset whose hash differs from a run's, so after the regeneration no FR-3 run can be re-finalized, re-reported or re-scored against the new file. Before milestone 1, the last commit with the old default spec and generator is tagged `data/fr3-default` (data hash `2f0e60a6`); generating from that tag reproduces FR-3's dataset. Serving the currently promoted model doesn't depend on it: `load_service` and `predict` read only the artifact.

### 4. Target

- **v1 target (accepted, owner, Oct 2, 2026): unseen-merchant macro F1 ≥ 0.70** on the 115-merchant test holdout, with its merchant-bootstrap interval reported, and above the keyword baseline's unseen score.
  - The validation leader scores 0.745 (0.68–0.80). With a bootstrap SD of about 0.038 at 115 merchants, a model that good passes 0.70 on roughly eight to nine draws in ten (88%; 78% if its true score is 0.73, allowing for its being the best of nine runs). So the gate mostly tests the model, not the draw. The SD was measured on the shipped model's predictions; the FR-4 round re-measures it with the merchant bootstrap on the leader's twin before the gate is used.
- **Revised to 0.66 (owner, Oct 3, 2026), after the FR-4 round's validation results and before any test scoring.** This is a rule change made after seeing results, so it's labelled as one. The round (FR-4 Categorization — Round Results) put its leader, bge-small without class weights, at **0.712** on validation unseen merchants, not the POC's 0.745. Re-measured as above (bootstrap SD 0.038 at 115 merchants), that leader would pass 0.70 on only about 53–63% of fresh holdouts. 0.66 restores the pass rate the target was set for (about 85–90% at 0.712; 0.66–0.67 for those rates, 0.65–0.66 allowing for best-of-four). What it rejects: a model like the old noisy configuration (about 0.45 on this holdout). The round's balanced-weights twins (0.67–0.69) would sit just above it, so the gate separates the clean-label families from the noisy one, not clean candidates from each other; the decision rule does that. No test data was involved; the test set is first scored in milestone 4.
  - A 0.80 gate would fail most draws of today's best model, and its interval's upper end only touches 0.80.
- **0.80 stays as the goal for v1.1**, reached through feedback (FR-5/FR-6) and measured by that design's replay as global gain on users who supplied no corrections. The Technical Design already names feedback as "a realistic route to FR-4's target".
- **This changes the PRD** (success metrics, FR-4), as FR-2 changed FR-8 for weekly spikes. The change lands in milestone 5.

### 5. Gates and selection

- **Decision rule,** fixed before the FR-4 round runs:
  - eligibility, ranking and the tie set on validation comparison runs (unseen macro F1, ties against the leader by paired merchant bootstrap), as in FR-3;
  - within the tie set, order by the twins' validation unseen Brier **with its own tie test**, judged against a named reference as F1 ties are: the **Brier leader** is the lowest-Brier twin in the F1 tie set, and runs whose paired merchant-bootstrap interval of the Brier difference against it contains 0 are tied with it (pairwise ties aren't transitive, so they aren't judged pair by pair);
  - among runs tied with the Brier leader, the lower batch cost wins, then explainability, then operational simplicity; the rest follow by Brier;
  - a twin-level F1 reversal, by any eligible run's twin, stops the round (§1).
- **Why a Brier tie test:** with Brier as a strict tie-break, cost is never consulted. FR-3 found Brier "separated every tied run", and here bge-small's clean Brier (0.134) trails bge-base's (0.130, unweighted 0.126) by only 0.004–0.008, at a third of the batch cost. Whether such a gap is real is exactly what a tie test answers. Serving is batched (FR-3's latency decision), so cost should decide between models that are tied on both F1 and calibration.
- **Made knowing the feasibility numbers.** They suggest this change would favour bge-small if it ties again. It uses no test data, it is fixed before the FR-4 round, and it applies from FR-4 on; FR-3's selection stands.
- **Promotion gates** (on the twin's test scores): known-merchant macro F1 ≥ 0.90 and above keyword (FR-3); unseen-merchant macro F1 ≥ the v1 target and above keyword's unseen score (FR-4).
- **If the rank-1 twin fails the FR-4 gate,** that is investigated, not resolved by promoting #2 (FR-3's rule).

## Metrics and why

FR-3's metrics carry over unchanged ("Metrics and why", FR-3 design). What FR-4 adds:

1. **Unseen-merchant macro F1 is a gate now**, with its interval reported. The interval is merchant-level, as in FR-3, because merchants, not transactions, are the unit that varies.
2. **Twin versus comparison run on validation,** reported side by side, so a ranking reversal between noisy and clean training is visible (§1).
3. **Per-category unseen F1, and where each category's errors go,** with Health & Fitness and Entertainment called out, since they are where the remaining gap is. The round's report includes each category's confusions on unseen merchants (which categories its errors land in, and which categories' errors land in it), so a later fix knows what it's fixing (owner decision: diagnose now, fix after launch).

## Options considered

### A. Closing the unseen-merchant gap

| Option | Evidence | Status |
| --- | --- | --- |
| **Clean training labels** | 0.512 to 0.714 (validation) | **Adopted** (owner decision, PR #15) |
| **No balanced class weights** | +0.031 with clean labels | **Candidate** |
| Larger embedding model (bge-base over bge-small) | Ties under clean labels | Kept as candidates; cost breaks ties |
| Routing by familiarity | Embeddings-only ties the combined model | Not adopted |
| Nearest neighbours over embeddings | 0.468 | Not adopted |
| LLM fallback for unfamiliar strings | Not measured | **Excluded** (owner decision) |
| bge-large, fine-tuned transformer | Not measured | Deferred; size didn't help from small to base |

### B. Measurement

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Keep the 58-merchant holdout | No regeneration | ±0.10 luck; test set already used by FR-3 |
| **(b) 40% holdout, new seed, in the default dataset (recommended)** | 115 merchants; an interval about a fifth narrower; a fresh test set | Training sees 176 instead of 233 spending merchants; all data regenerates |
| (c) Several holdout seeds, scores averaged | Least luck | A full dataset and training per seed |
| (d) Validation only, no test gate | Already 169 merchants | Validation merchants are train users' merchants; no untouched final check |

### C. Target

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Keep 0.80 as a v1 gate | The PRD as written | Today's best model fails most draws; the gate would test luck |
| **(b) 0.70 for v1 (revised to 0.66 after the round, see §4), 0.80 for v1.1 after feedback (recommended)** | Passes on roughly eight to nine draws in ten for a model as good as the leader; keeps 0.80 as the goal, with a route to it | A PRD change |
| (c) A gate on the interval's lower bound | Conservative | With 3–6 merchants per category, the lower bound is far below the point estimate; it would gate on sample size |

## Testing

- **Unit:** `class_weight` accepts `balanced` and `none` and rejects anything else; a twin is the same config except `label_noise`; promotion refuses a config without an explicit `label_noise`, and the log records it.
- **Framework (toy task):** `finalize` scores twins of the top three and refuses a finalist without one; the leaderboard ranks only comparison runs; the report shows twins beside their comparison runs.
- **Data:** the default spec passes validation with the new holdout (115 merchants, test share within 15–25%).
- **Slow (default data):** the promoted twin passes FR-3's and FR-4's gates.

## Milestones

One PR per milestone.

1. **Dataset:** tag `data/fr3-default` first; then the default spec's new holdout (share, seed, bias) together with `truth_preferences` (schema 4) and the preference-aware label contract, from #15's accepted data contract; regenerated once and validated; FR-1 and FR-2 docs note the change.
2. **Framework:** shipping twins in run, finalize and promote; explicit `label_noise` required at promotion; tie-breaks on twins with the Brier tie test; the reversal stop; `class_weight` in `linear_text`.
3. **FR-4 round:** baselines and the four candidates, each with its twin, on the new dataset; the comparison report, with per-category confusions on unseen merchants.
4. **Finalize and promote:** twins of the top three scored once on the fresh test set; promote rank 1 if it passes both features' gates; `sfc-model predict`; FR-3 Categorization Model Selection updated.
5. **Docs:** the PRD's FR-4 success metric (0.66 for v1, 0.80 for v1.1 through feedback), the Technical Design (twins, the gate, the decision-rule changes).

## Status and handoff (Oct 3, 2026)

Written for whoever continues FR-4, human or agent. It records where the work stands, what remains with the exact commands, the owner's decisions, and how the work has been done. Read it together with §1–§5 above, which define the rules being implemented.

### Done

| Milestone | PR | What landed |
| --- | --- | --- |
| Design | #17 | This design, accepted by the owner; feasibility on POC branch `poc/fr-4-unseen-merchants` (pinned `2541dc6`) |
| 1. Dataset | #20 | Default spec: holdout share 0.4, seed 8, `test_user_bias` 0.45 (115 holdout merchants). Schema 4: `truth_preferences` and label contract 2 (`Truth.user_categories()`), from #15's accepted data contract. Regenerated default dataset: data hash **`44781bc4e4a5`**, 23.5% of test spending at holdout merchants, all checks pass |
| 2. Framework | #21 | Shipping twins (`ship:` in configs, `run_with_twin`, tag `sfc.twin_of`); `Task.shipping_params` (categorization: `label_noise`); tie-breaks on twins with the Brier tie test against the Brier leader; the reversal stop in `finalize`; `promote` requires explicit shipping params and logs them; twin columns in the report; `class_weight` in `linear_text` |

If #21 isn't merged when you read this, check it first: it's the base for everything below.

**Also in place:** the git tag `data/fr3-default` (on `bfc07ac`) reproduces FR-3's dataset (data hash `2f0e60a6`). The promoted model is still FR-3's bge-base `3f0ccc82-2f0e60a6`, trained on that old dataset; it keeps serving until milestone 4 promotes a replacement.

### Remaining

**Milestone 3: the FR-4 round.** One branch from the latest `main`, one PR. Code first, then the run, then the report.

1. **Add the FR-4 gate before anything is test-scored.** `CategorizationTask.gates` (`evaluation/tasks/categorization.py`) checks only FR-3's gates today (known ≥ 0.90, above keyword). Add, per §5:
   - `unseen_macro_f1`: `test_unseen_macro_f1 ≥ 0.70` (a constant like `KNOWN_GATE`, e.g. `UNSEEN_GATE = 0.70`);
   - `beats_keyword_unseen`: `test_unseen_macro_f1` above the keyword baseline's `test_unseen_macro_f1`.

   Tests: a twin at 0.69 unseen fails; one at 0.71 that's below keyword's unseen score fails; one above both passes. `test_gates_and_eligibility` in `tests/unit/evaluation/test_categorization_task.py` is the pattern.
2. **Add per-category unseen-merchant confusions to the round's report** (owner decision: diagnose Health & Fitness and Entertainment now, fix after launch; "Metrics and why", point 3). For each spending category on validation unseen-merchant rows: its F1, where its errors go (top predicted categories for its true rows), and what lands in it (top true categories among rows predicted as it). Report it for rank 1's twin at least. `experiments/fr4_unseen/unseen_errors.py` on the POC branch is a working starting point (count merchants once, under their majority category).
3. **Write the round's configs** in `configs/experiments/categorization/fr4/`, one file per run, each **with `ship: {task_params: {label_noise: 0}}`** except the baselines:
   - baselines: `keyword`, `lookup`, `majority`, copied from `launch/` (no `ship`);
   - `base_balanced`: bge-base, `class_weight: balanced`;
   - `base_unweighted`: bge-base, `class_weight: none`;
   - `small_balanced`: bge-small, `class_weight: balanced`;
   - `small_unweighted`: bge-small, `class_weight: none`.

   All four candidates: `categorization/calibrated` (`method: auto`, `by: familiarity`) around `categorization/linear_text` with `C: 1.0`, and `task_params: {k: 3}`, like the launch round and the POC (`configs/experiments/categorization/launch/20_linear_base.yaml` is the template). The task's default `label_noise` (0.02) applies to the comparison runs; the twins set 0 explicitly. Don't add other candidates: §2 fixed the list before the round.
4. **Regenerate the dataset** if you don't have it (git-ignored): `uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite --force` (about 40 s). Confirm the data hash is `44781bc4e4a5`.
5. **Run the round** in the background, logging to a file (`PYTHONUNBUFFERED=1`):

   ```sh
   uv run sfc-experiment run configs/experiments/categorization/fr4/ --data data/synthetic/default.sqlite
   ```

   Expect about 1.5 hours on a laptop CPU: each calibrated bge-base configuration took about 14 minutes on 3 folds in the POC and bge-small about 8, and every candidate trains twice (comparison run and twin). Use the default local MLflow store, not the POC's `mlruns/poc-fr4.db`.
6. **Read the leaderboard and the report.** `uv run sfc-experiment leaderboard --task categorization --data data/synthetic/default.sqlite` and `... report ... --out <file>`. Check:
   - the order comes from comparison runs, with ties broken on twins (§5);
   - **no reversal** is marked. If one is, stop: it's investigated, not resolved by promoting another twin (§1). Raise it with the owner;
   - the twins' numbers look like the POC's (clean bge-base about 0.71–0.75 unseen on validation, on a different holdout seed now).
7. **Re-measure the bootstrap SD on the leader's twin** before the 0.70 gate is relied on (§4): the merchant bootstrap of its validation unseen-merchant macro F1, as `experiments/fr4_unseen/holdout_size.py` on the POC branch does, but with replacement only and on the twin's run. Record the SD and the pass-rate estimate in the round's report.
8. **Write the round's report** in `docs/reports/` (like `FR-3 Categorization — Launch Round Results.md`): the generated tables, the confusions, the SD, and findings. Open the PR with the configs, the code and the report. **Don't run `finalize` in this milestone.**

**Milestone 4: finalize and promote.** After milestone 3 merges, on a new branch.

1. `uv run sfc-experiment finalize --task categorization --data data/synthetic/default.sqlite` scores the twins of the top three, and the baselines, on the fresh test set, **once**. A second round needs `--override "<reason>"`; don't spend it casually.
2. If rank 1's twin passes every gate (FR-3's and FR-4's): `uv run sfc-model promote --task categorization --run <rank-1 twin run ID> --note "<explainability, operations, retraining cost, known issues>"`. It publishes the model file to a GitHub Release (`gh` with write access), writes `PROMOTED`, the manifest and `promotions.jsonl` (with `"task_params": {"label_noise": 0.0}`), and the release is tagged at the training commit only if the run was trained from a clean tree. **If rank 1 fails a gate, investigate; don't promote #2 without an override reason and the owner.**
3. `uv run sfc-model predict --task categorization --data data/synthetic/default.sqlite --out data/predictions/default.sqlite`, and record rows per second and peak memory.
4. Update FR-3 Categorization Model Selection (what's selected now, why, what lost) and the round's report (the test table from `sfc-experiment report`).
5. **If the promotion PR is rejected, delete its release:** `gh release delete categorization-<version> --cleanup-tag`.

**Milestone 5: docs.**

- **PRD:** FR-4's success metric becomes ≥ 0.70 on new merchants for v1, with 0.80 for v1.1 through feedback (owner decision, open question 1 below).
- **Technical Design:** shipping twins, the FR-4 gate, and the decision-rule changes (tie-breaks on twins, the Brier tie test, the reversal stop).
- **This section:** update it to "complete".

### What depends on FR-4

**FR-5/FR-6 (draft #15).** Its simulated replay runs on the model milestone 4 promotes, and its Feasibility section is re-measured on that model (owner decision on #15). Its data contract is already in the dataset (#20).

### Practical notes

- **The POC's runs** are in a local MLflow store on the machine that ran them (`mlruns/poc-fr4.db`), not in the repo. The POC branch has the configs, scripts and results, so they can be rerun anywhere.
- **Embedding model files** download to `data/models/fastembed` on first use (git-ignored; about 67 MB for bge-small, 130 MB for bge-base).
- **Another session may be working in the repository's main checkout** (the web app track). Do branch work in a separate git worktree (`git worktree add <path> -b <branch> origin/main`), and never check out, reset or stash in the main checkout.
- **Install the git hooks** in a fresh checkout (`uv run pre-commit install`). They run ruff, mypy and the 5 MB file check; CI fails on what they'd catch.
- **How the work has been done** (keep it):
  - one PR per milestone, from a branch on the latest `main`; `main` requires a PR, a green `check` and an up-to-date branch, so rebase onto `main` when it moves and rerun the tests;
  - before every push, check the open PR for review comments and address them first, replying on each thread;
  - rule or design changes are proposed to the owner before they're made, recorded in the design with their reason, and labelled when they come after seeing results;
  - test sets are touched only by `finalize`.

## Decisions and open questions

**Decisions** (all approved by the owner, Oct 2, 2026, on PR #17)

- [x] Comparison runs under noise; shipping twins with explicit `label_noise: 0` are what is finalized and promoted.
- [x] `class_weight` as a candidate dimension; routing and nearest neighbours not carried forward.
- [x] The default dataset's holdout becomes 40% with a new seed (115 merchants, fresh test set).
- [x] FR-4 gate: unseen macro F1 ≥ the v1 target and above keyword, alongside FR-3's gates.
- [x] Tie-breaks on the twins' validation metrics; a paired-bootstrap Brier tie test against the lowest-Brier twin in the F1 tie set, so cost can decide; a reversal by any eligible run's twin stops the round (from review).
- [x] Tag `data/fr3-default` before regenerating, so FR-3's dataset stays reproducible (from review).
- [x] Milestone 1 builds schema 4 from #15's accepted data contract (owner, Oct 2, 2026, on #15).
- [x] Regenerate the dataset once, with FR-5/FR-6's schema 4; the replay runs on FR-4's promoted twin (owner, Oct 2, 2026, on #15).
- [x] No LLM fallback in categorization (owner, Oct 2, 2026).

**Open questions**

1. [x] **The v1 target. Decided (owner, Oct 2, 2026): ≥ 0.70 for v1,** above keyword's unseen score, with the merchant-bootstrap interval reported; **0.80 for v1.1** through feedback, measured by the FR-5/FR-6 replay as global gain. Basis: today's best model (0.745 on validation) would fail a 0.80 gate on most draws, while 0.70 passes for a model that good on roughly eight to nine draws in ten. Follows FR-2's precedent for weekly spikes. The PRD change lands in milestone 5. **Revised (owner, Oct 3, 2026): 0.66**, on the FR-4 round's validation results and before any test scoring (§4).
2. [x] **Health & Fitness and Entertainment** stay weakest (0.16 and 0.46 on validation). **Decided (owner, Oct 2, 2026): diagnose now, fix after launch.** The round's report shows where their errors go (milestone 3; "Metrics and why", point 3). No targeted change before launch: tuning toward two known-weak categories on validation data risks fitting the evaluation, and these are the gaps feedback is meant to close. Revisit if the FR-5/FR-6 replay shows corrections don't lift them.
3. [x] **bge-small as the default if it ties again.** Answered by the decision-rule change in §5 (from review): with a Brier tie test, cost decides among runs tied on F1 and calibration. Under a strict Brier tie-break it never could.
