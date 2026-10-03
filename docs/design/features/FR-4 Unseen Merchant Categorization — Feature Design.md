# FR-4 Unseen Merchant Categorization — Feature Design

Oct 2, 2026 · @Sidd · Status: **Proposed** · Branch: `docs/fr-4-design`

## Summary

This feature makes the categorizer usable on merchants it has never seen, measures that honestly, and sets a v1 target the evidence supports.

- **Requirement:** FR-4 (P0), *"Categorize merchants the system has never seen before."* PRD success metric: macro F1 ≥ 0.80 on new merchants.
- **Starting point:** the promoted model (bge-base, `3f0ccc82-2f0e60a6`) scores **0.462** (0.40–0.55) on unseen test merchants and 0.512 on validation. On unfamiliar merchants, Travel is its fallback guess (FR-3 Categorization Model Selection).
- **Feasibility, measured on validation data only** ([evidence](#feasibility)):
  - **Clean labels are the lever.** Training without the injected 2% label noise lifts unseen-merchant macro F1 from 0.512 to **0.714**. Dropping balanced class weights as well gives **0.745** (0.68–0.80). Known merchants rise from 0.969 to 0.988, the Travel fallback disappears (Travel F1 0.025 to 0.886), and misallocated unseen spend falls from 27% to 11%.
  - **Nothing else measured helps.** Routing unfamiliar strings to an embeddings-only model ties (0.704); nearest neighbours over embeddings is worse than the shipped model (0.468); bge-small ties bge-base (0.715) at a third of the batch cost.
  - **Today's test holdout is too small to judge 0.80.** With 58 holdout merchants, the same model scores anywhere from 0.43 to 0.62 depending on which merchants are drawn. A 40% holdout gives 115 merchants and halves that spread, and the generator's data checks pass with it.
- **Approach:**
  1. **Ship a clean-label model.** The owner decided on PR #15 that shipped and retrained models train on clean labels, while injected noise stays for comparing candidates. FR-4 makes that mechanical: every compared configuration has a **shipping twin** with `label_noise: 0`, and the twins are what get finalized and promoted.
  2. **Add the class-weight choice to the candidates**, since unweighted training led on validation.
  3. **Measure on a bigger, fresh holdout:** the default dataset gets a 40% holdout with a new seed. This doubles the unseen merchants to 115, and gives a test set nothing has been scored on.
  4. **Set the v1 target at 0.70**, with 0.80 as the v1.1 target reached through feedback (FR-5/FR-6), as FR-2 did for weekly spikes. This is a PRD change for the owner.
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
| Shopping | 28 | 0.580 | 0.758 | 0.801 |
| Groceries | 18 | 0.588 | 0.815 | 0.801 |
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
| Gates on | Known merchants ≥ 0.90, beats the keyword baseline | Unseen merchants ≥ the v1 target (proposed 0.70), beats the keyword baseline on unseen merchants; FR-3's gates still apply | Personal accuracy, global gain after feedback |
| Changes | Framework, first model | Training labels, candidates, the holdout, the target | Review, corrections, retraining |

## Feasibility

**Evidence:** POC branch `poc/fr-4-unseen-merchants`, pinned to commit [`108e5aa`](https://github.com/sidag02/Smart-Financial-Coach/tree/108e5aafc758a8f2831723b34029ccaba469f519/experiments/fr4_unseen). Every number below is in its [results](https://github.com/sidag02/Smart-Financial-Coach/blob/108e5aafc758a8f2831723b34029ccaba469f519/experiments/fr4_unseen/results/feasibility.md), with the commands that reproduce them.

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

Category-stratified subsets of the 169 validation merchants, scored with the shipped model's validation predictions:

| Holdout merchants | Same model, 95% of holdout draws | Draw-to-draw SD | Bootstrap interval width |
| --- | --- | --- | --- |
| **58** (today's test holdout) | 0.425–0.624 | 0.051 | 0.186 |
| 87 | 0.447–0.599 | 0.035 | 0.163 |
| **115** | 0.461–0.570 | 0.029 | 0.148 |

- **With 58 merchants, one holdout draw swings the score by about ±0.10.** FR-3's 0.462 on test against 0.512 on validation is within that luck. A 0.80 pass or fail on this holdout would say little about the model.
- **115 merchants halve the spread** (SD 0.029). Three to six merchants per category is still few, so intervals stay wide; the target below is set with that in mind.

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

**Does ranking under noise pick the right clean model?** It did here: without class weights wins under noise (0.666 against 0.512) and without noise (0.745 against 0.714). It isn't guaranteed. The report shows each finalist's twin on validation next to its comparison run, so a reversal would be visible before the test sets are touched.

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
- **Coordination with FR-5/FR-6:** that design adds `truth_preferences` (schema 4). If both land close together, regenerate once with both changes.

### 4. Target

- **Proposed v1 target: unseen-merchant macro F1 ≥ 0.70** on the 115-merchant test holdout, with its merchant-bootstrap interval reported, and above the keyword baseline's unseen score.
  - The validation leader scores 0.745 (0.68–0.80). With a draw-to-draw SD of about 0.03 at 115 merchants, a model that good passes 0.70 on about nine draws in ten, so the gate tests the model, not the draw. (The SD was measured on the shipped model's predictions; the FR-4 round re-measures it for the leader before the gate is used.)
  - A 0.80 gate would fail most draws of today's best model, and its interval's upper end only touches 0.80.
- **0.80 stays as the goal for v1.1**, reached through feedback (FR-5/FR-6) and measured by that design's replay as global gain on users who supplied no corrections. The Technical Design already names feedback as "a realistic route to FR-4's target".
- **This changes the PRD** (success metrics, FR-4), as FR-2 changed FR-8 for weekly spikes. It is the owner's decision, listed below.

### 5. Gates and selection

- **Decision rule:** unchanged from FR-3: eligibility and ranking on validation comparison runs, ties against the leader by paired merchant bootstrap, then unseen Brier, then batch cost.
- **Promotion gates** (on the twin's test scores): known-merchant macro F1 ≥ 0.90 and above keyword (FR-3); unseen-merchant macro F1 ≥ the v1 target and above keyword's unseen score (FR-4).
- **If the rank-1 twin fails the FR-4 gate,** that is investigated, not resolved by promoting #2 (FR-3's rule).

## Metrics and why

FR-3's metrics carry over unchanged ("Metrics and why", FR-3 design). What FR-4 adds:

1. **Unseen-merchant macro F1 is a gate now**, with its interval reported. The interval is merchant-level, as in FR-3, because merchants, not transactions, are the unit that varies.
2. **Twin versus comparison run on validation,** reported side by side, so a ranking reversal between noisy and clean training is visible (§1).
3. **Per-category unseen F1,** with Health & Fitness and Entertainment called out, since they are where the remaining gap is.

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
| **(b) 40% holdout, new seed, in the default dataset (recommended)** | 115 merchants; half the spread; a fresh test set | Training sees 176 instead of 233 spending merchants; all data regenerates |
| (c) Several holdout seeds, scores averaged | Least luck | A full dataset and training per seed |
| (d) Validation only, no test gate | Already 169 merchants | Validation merchants are train users' merchants; no untouched final check |

### C. Target

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Keep 0.80 as a v1 gate | The PRD as written | Today's best model fails most draws; the gate would test luck |
| **(b) 0.70 for v1, 0.80 for v1.1 after feedback (recommended)** | Passes reliably for a model as good as the leader; keeps 0.80 as the goal, with a route to it | A PRD change |
| (c) A gate on the interval's lower bound | Conservative | With 3–6 merchants per category, the lower bound is far below the point estimate; it would gate on sample size |

## Testing

- **Unit:** `class_weight` accepts `balanced` and `none` and rejects anything else; a twin is the same config except `label_noise`; promotion refuses a config without an explicit `label_noise`, and the log records it.
- **Framework (toy task):** `finalize` scores twins of the top three and refuses a finalist without one; the leaderboard ranks only comparison runs; the report shows twins beside their comparison runs.
- **Data:** the default spec passes validation with the new holdout (115 merchants, test share within 15–25%).
- **Slow (default data):** the promoted twin passes FR-3's and FR-4's gates.

## Milestones

One PR per milestone.

1. **Dataset:** the default spec's new holdout (share, seed, bias), regenerated and validated; FR-1 and FR-2 docs note the change (coordinated with FR-5/FR-6's schema 4 if close).
2. **Framework:** shipping twins in run, finalize and promote; explicit `label_noise` required at promotion; `class_weight` in `linear_text`.
3. **FR-4 round:** baselines and the four candidates, each with its twin, on the new dataset; the comparison report.
4. **Finalize and promote:** twins of the top three scored once on the fresh test set; promote rank 1 if it passes both features' gates; `sfc-model predict`; FR-3 Categorization Model Selection updated.
5. **Docs:** PRD target (if accepted), Technical Design (twins, the gate), this design accepted.

## Decisions and open questions

**Decisions for review**

- [ ] Comparison runs under noise; shipping twins with explicit `label_noise: 0` are what is finalized and promoted.
- [ ] `class_weight` as a candidate dimension; routing and nearest neighbours not carried forward.
- [ ] The default dataset's holdout becomes 40% with a new seed (115 merchants, fresh test set).
- [ ] FR-4 gate: unseen macro F1 ≥ the v1 target and above keyword, alongside FR-3's gates.
- [x] No LLM fallback in categorization (owner, Oct 2, 2026).

**Open questions**

1. [ ] **The v1 target.** Proposed 0.70 for v1, 0.80 for v1.1 through feedback. Needs the owner's decision and a PRD change.
2. [ ] **Health & Fitness and Entertainment** stay weakest (0.16 and 0.46 on validation). Worth a targeted look (subtype confusions, side features) before or after launch?
3. [ ] **bge-small as the default** if it ties again: about a third of the batch cost and memory. The decision rule's tie-breaks (Brier, then cost) decide; worth confirming that's the intended order now that serving is batched.
