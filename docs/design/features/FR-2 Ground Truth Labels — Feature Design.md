# FR-2 Ground Truth Labels — Feature Design

Oct 1, 2026 · @Sidd · Status: **Revised after review; ready to implement** · Branch: `feature/fr-2-ground-truth`

## Summary

This feature makes the ground truth that FR-1 already writes precise, trustworthy and safe to use, so every quality number in the evaluation report means what it says.

- **Requirement:** FR-2 (P0): *"Synthetic data carries known true categories and known anomalies (unusual charges and spending spikes), so quality can be measured."*
- **Where we start:** FR-1 already plants unusual charges and spending spikes as generator events and writes `truth_*` tables (see FR-1 Synthetic Data Generator — Feature Design). FR-2 does **not** add a second injector.
- **What is missing:** an audit of the default dataset (below) shows the labels record what was *planned*, not what the data *shows*. 18% of weekly spikes leave no visible trace. 22% of unlabeled weeks look like a spike to a simple rule. No rule says how a weekly flag inside a labeled monthly spike should be scored.
- **Approach:** four additions, all generic and spec-driven:
  1. A written **label contract**: what is positive, negative or ignored at each level.
  2. **Realized-effect annotations** on every planted spike, so weak labels are visible.
  3. An **oracle ceiling**: the best precision any detector could reach on this data. Monthly spikes are gated on it; weekly targets are set relative to it.
  4. **Label checks** in `validate`, and a `labels` report command.
- **Decisions** were made on Oct 1, 2026 and revised after the PR review the same day; see [Decisions and open questions](#decisions-and-open-questions).
- **Principle:** the modeled user behavior is not tuned to make targets pass. Where the behavior makes a target unreachable, the target moves.

## Context

Each FR-2 label feeds one success metric in the PRD. A metric is only as good as the label it is scored against.

| Label | Source table | Feeds metric (PRD) | Target |
| --- | --- | --- | --- |
| True category | `truth_transactions.category` | Categorization macro F1 (FR-3, FR-4) | ≥ 0.90 known; ≥ 0.80 new merchants |
| Unusual charge | `truth_transactions.anomaly_kind` | Unusual-transaction precision / recall (FR-7) | Precision ≥ 0.70; recall above baseline |
| Spending spike | `truth_periods` | Period-level precision / recall (FR-8) | Precision ≥ 0.70; recall above baseline |
| Goal outcome | `truth_goals` | Brier score (FR-11) | Better calibrated than naive |

Goal labels and their `as_of_date` cutoff were settled in FR-1 and are not changed here.

## Label audit of the current data

Measured on `data/synthetic/default.sqlite` (default spec, 300 users, 896,322 transactions). Spike lift is the period's discretionary spend in the category divided by the user's median for that category and granularity over the whole history.

### Spending spikes

| | Monthly spikes | Weekly spikes |
| --- | --- | --- |
| Labeled | 497 | 493 |
| Median realized lift | 2.28 | 2.45 |
| Labeled with lift < 1.3 (no visible spike) | 8% | **18%** |
| Unlabeled periods with lift ≥ 1.8 | 13% (8,524) | **22%** (40,627) |

- **Weak labels come from low-volume categories.** Among weekly spikes, the share with lift < 1.3 falls as purchase volume rises: 27% when the user averages ≤ 1.5 purchases a week in that category, 6% above 6 a week. Groceries (median 1.2 a week) is weak 25% of the time, Dining (6 a week) 8%.
- **Unlabeled look-alikes outnumber labels about 80 to 1 at week level.** Most are seasonality and Poisson noise, which a good per-user model should explain away. A whole-history median can't, so this is an upper bound. It still shows that weekly scoring is dominated by negatives that look positive.
- **Spikes cover only four categories** (Dining, Groceries, Shopping, Transportation), because `min_weekly_rate` excludes the rest.

### Unusual charges

| Kind | Labeled | Share above the user's p99 for that category | Median amount |
| --- | --- | --- | --- |
| `amount_outlier` | 467 | 83% | $423 |
| `new_merchant_large` | 435 | 88% | $705 |
| `duplicate` | 424 | **1%** | $28 |
| *Normal one-offs (for comparison)* | 9,319 | 16% | |

- **Duplicates are invisible to amount-based detectors.** Only pair features (same text and amount within hours) find them. Separately, 30 normal discretionary purchases are exact same-day repeats of another (e.g. two identical coffees). These are realistic hard negatives.
- **Large first visits are mostly normal.** 1,837 one-offs and 628 recurring first payments are first-time merchants at ≥ $250, against 435 labeled `new_merchant_large`. This is intended; it is what makes the problem non-trivial.
- **The original of a duplicate is labeled normal.** A detector that flags the earlier of the two charges is scored as a false positive today.

### Categories

- 3.5% of spending transactions are at ambiguous merchants (warehouse clubs, marketplaces), split between Groceries and Shopping.
- 39% of those are not the merchant's majority category, so **1.3% of all spending transactions can't be categorized correctly from text and amount alone**. This sets a ceiling on Groceries and Shopping F1 and should be reported, not treated as model error.

### Scoring gaps

The current tables don't answer these, so two evaluators could score the same model differently:

1. Does a weekly flag inside a labeled monthly spike count as correct, wrong or neither?
2. Does a month that contains a labeled weekly spike count as a positive month?
3. Is a period that is high only because of one planted unusual charge a spike false positive?
4. Are the first 90 days (no anomalies planted) scored at all?

## Goals and non-goals

**Goals**

1. A label contract that turns every flag into exactly one of: true positive, false positive or ignored, with no evaluator judgment.
2. Every planted spike carries its realized effect, so weak labels can be reported separately instead of silently lowering recall.
3. Measure, per dataset, the best precision an ideal detector can reach, and set targets that are achievable on that data rather than tuning the data to fit the targets.
4. Generator bugs that corrupt labels fail `validate` before any model sees the data.

**Non-goals**

- Metric computation, train/eval splits and the evaluation report. These belong to the evaluation harness (Technical Design build step 4). FR-2 supplies the labels and the matching rules it calls.
- Labeling "natural" anomalies the generator didn't plant. Positives are planted events only; see option A.
- Changing goal labels.

## What it will look like

### Usage

```sh
uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite
uv run sfc-data labels data/synthetic/default.sqlite      # label report: counts, tiers, ceilings
```

```python
from smart_financial_coach.data.labels import load_truth

truth = load_truth("data/synthetic/default.sqlite")   # evaluation harness only

truth.spikes(granularity="week", tier="clear")         # labeled periods
truth.score_periods(flags, granularity="week")         # -> TP / FP / FN / ignored per flag
truth.score_transactions(flags)                        # same, at transaction level
```

### Schema changes

All additions are columns or tables in the same SQLite file. Model-visible tables don't change.

**truth_transactions** (one new column)

| Column | Type | Notes |
| --- | --- | --- |
| related_transaction_id | TEXT, nullable | For `duplicate`: the original charge. For `refund`: the purchase refunded. The ledger already tracks this as `copy_of`; it is now written out |

**truth_periods** (new columns)

| Column | Type | Notes |
| --- | --- | --- |
| spike_id | TEXT | Stable ID; becomes the primary key |
| period_end | TEXT | Inclusive; saves every consumer recomputing week and month ends |
| expected_count | REAL | Expected number of Poisson-process purchases (discretionary and one-off) in the category and period without the spike |
| expected_spend | REAL | Expected spend in the category and period without the spike, all processes (see [What a period contains](#what-a-period-contains)) |
| base_spend | REAL | Realized spend from everything in the period except the spike's extra purchases |
| extra_spend | REAL | Realized spend from the spike's extra purchases |
| tier | TEXT | `clear` or `weak` (see [Spike tiers](#3-spike-tiers)) |

**truth_expected** (new, eval only)

| Column | Type | Notes |
| --- | --- | --- |
| user_id, category, granularity, period_start | TEXT | Primary key; only categories eligible for spikes |
| expected_count | REAL | Expected Poisson-process purchases for the user's normal behavior: seasonality, income coupling and day-of-week included, spikes and unusual charges excluded |
| expected_spend | REAL | Expected spend on the same basis, all processes |

Size on the default spec: about 360 users × 4 categories × (156 weeks + 36 months) ≈ 280k rows, a few MB. It powers the oracle ceiling.

Expectations are per **true** category. Ambiguous merchants draw their category per transaction, so a Groceries stream at a warehouse club contributes to Shopping's expectation in proportion to that merchant's category mix.

**meta** (new keys): `label_contract_version`, `oracle_*` ceiling stats, `ambiguous_error_share`.

## Design

### 1. Label contract

The contract is code (`data/labels.py`) and is versioned in `meta.label_contract_version`. It lives in the data layer because `validate` needs the oracle; the evaluation harness imports it. Every rule below is a spec parameter with the default shown.

**Transaction level (FR-7)**

| Flagged transaction | Outcome |
| --- | --- |
| Has `anomaly_kind` | True positive |
| Is the original of a labeled `duplicate` | **Ignored.** Which of two identical charges is "the duplicate" is arbitrary |
| In the user's first `baseline_days` (90) | Ignored. Nothing is planted there and detectors have no history |
| Anything else | False positive |
| Unflagged with `anomaly_kind`, outside the warm-up | False negative |

Reason accuracy: among true positives, the flag's reason code must match the kind (`duplicate`, `amount_unusual`, `new_merchant`). This measures NFR-7's "every flag has a reason" against truth, not just for presence.

**Period level (FR-8)**

A period is (user, category, granularity, period start). Weeks start Monday; months start on the 1st.

#### What a period contains

- **Category:** spike detection is scored on **true** categories. The harness feeds detectors true categories for FR-8 evaluation, so spike metrics aren't confounded by categorizer errors. FR-3 and FR-4 measure categorization separately.
- **Spend:** a period's spend is the net outflow of **every** transaction in the category: recurring bills, discretionary purchases, one-offs, refunds and unusual charges. This is what a deployed model sees. `expected_spend` covers the same set, so the ceiling and the models measure the same quantity.

| Flagged period | Outcome |
| --- | --- |
| Matches a `truth_periods` row | True positive |
| Week inside a labeled **monthly** spike of the same category | Ignored. It really is elevated, but it isn't the labeled unit |
| Month containing a labeled **weekly** spike of the same category | Ignored. Lift is diluted to roughly 1.2–1.5×, too ambiguous to call either way |
| Contains a planted unusual charge in the same category | Ignored. The FR-7 metric already scores that charge |
| Starts in the user's first `baseline_months` (3) | Ignored |
| Anything else | False positive |

Recall is reported on all labels and on `clear` labels only. Precision is the same for both.

**Driving transactions** (FR-8 "the transactions driving it"): the returned set is capped at `max_drivers` (default 5) transactions, each of which must be in the period and category. It is scored by **excess coverage**: the returned spend divided by the period's excess over `expected_spend`, capped at 1. With the cap, returning small or irrelevant purchases scores low, so the metric can fail.

Precision and recall against the overlay's extra purchases were considered and rejected. Extra purchases are drawn from the same merchants and prices as normal ones, so no detector can tell them apart; that score would measure luck.

### 2. Spikes as an overlay process

Today a spike multiplies the discretionary purchase rate inside one Poisson draw. The normal and extra purchases can't be told apart, so the realized effect is unknown.

**Change:** draw normal purchases at the unspiked rate λ, then draw extra purchases at rate λ·(m − 1) for the spike's days with a separate seeded generator. The sum has the same distribution as today (Poisson superposition), so the modeled behavior is unchanged.

Three implementation rules keep normal purchases independent of spikes:

1. **Calibration on normal purchases only.** `calibrate_discretionary` today sets the rate scale from discretionary spend with spikes included, so turning spikes off would change every purchase. It now calibrates on normal purchases; extras are drawn afterwards at the calibrated rate.
2. **A separate ledger process.** Extras are added under an internal `spike_extra` process with its own counter, so transaction IDs of normal purchases don't shift. They are written as `discretionary` in `truth_transactions`.
3. **Merchant text is not covered.** Rendering draws from one random sequence per user, in transaction order, so extra rows change the `merchant_raw` of later transactions. The independence guarantee covers IDs, dates, amounts, merchants and true categories, not rendered text.

What this buys:

- `base_spend` and `extra_spend` are exact, so each label shows how much of the period was the spike and how much was noise.
- With spikes turned off in an otherwise identical spec, normal purchases keep their IDs, dates, amounts and merchants: a controlled comparison. (`clean.yaml` also turns rendering off, so against it only amounts and dates are comparable.)

Cost: calibration changes slightly for users with spikes, and the default dataset's content hash changes once; regeneration is required.

### 3. Spike tiers

`tier = weak` when the period's realized total `base_spend + extra_spend` is below `expected_spend × weak_lift` (default 1.3). Otherwise `clear`.

Weak labels stay in the data. Dropping them would hide how often realistic noise cancels a real overspend. Reporting both recalls keeps the headline number honest without letting weak labels dominate model selection.

The spike volume floor (`min_weekly_rate`, 1 purchase a week, checked against the user's rate before calibration as today) stays the same for both granularities, so low-volume categories such as Groceries keep weekly spikes. Expect about 18% of weekly labels to be `weak`; the audit shows weekly labels below 1.5 purchases a week are weak 27% of the time. A higher weekly floor was considered and rejected under the principle above; see option F.

### 4. Oracle ceiling

The generator knows each user's true expected purchase count per period. A spike multiplies the purchase *rate*, so the test that matches the generative process is on counts. Spend adds heavy-tailed price noise on top, so a spend ratio is not the ceiling.

- **Oracle score:** the Poisson upper-tail probability of the period's realized Poisson-process purchase count, given `expected_count`. Recurring bills are excluded from the score, since their count is fixed by schedule. The oracle may use truth freely; it is a ceiling, not a model.
- **Operating point:** precision at a fixed recall of 0.5, which doesn't depend on any baseline.
- **Monthly spikes:** `validate` fails when oracle precision is below the PRD target (0.70).
- **Weekly spikes:** no gate. The oracle's weekly precision is recorded in `meta`, and the weekly target is **relative**: a model must reach 70% of the oracle's precision at the same recall.
- **Unusual charges:** a transaction oracle (true merchant price distribution plus a perfect duplicate check), gated at 0.70.

**Feasibility, from the PR review** (default spec, discretionary purchases, precision at recall 0.5):

| Oracle | Monthly | Weekly |
| --- | --- | --- |
| Spend ratio | 0.29 | 0.02 |
| Purchase count (Poisson tail) | 0.98 | 0.05 |

Restricting weekly spikes to categories with ≥ 6 purchases a week after calibration would lift the weekly count oracle to 0.71, but that would change the modeled behavior, so weekly gets a relative target instead (option F). On this data, weekly spike detection is close to impossible for any detector. Weekly results are diagnostic, and the PRD target for weekly spikes changes accordingly.

The oracle is an upper bound on the ranking problem only. Real detectors also have to estimate the expected count, so a model at the oracle is not expected.

### 5. Truth isolation

Isolation stays a convention, as in FR-1 option D-b: model-visible tables have plain names, truth tables are prefixed `truth_`, and model code reads only the former. There is no runtime guard.

- `load_truth(path)` in `data/labels.py` is the intended reader of truth tables; model code under `intelligence/` must not import it.
- A unit test fails if any file under `intelligence/` contains `truth_` or imports `data.labels`. This is cheap and catches accidental reads in CI.
- Code review covers the rest. A guarded connection (an SQLite authorizer denying reads of `truth_*`) was considered and deferred; see option D.

### 6. Category labels

No generator change. FR-2 records:

- The ambiguity ceiling (`meta.ambiguous_error_share`, 1.3% on default). The label report shows per-class F1 for the majority-category oracle, so Groceries and Shopping are judged against what is achievable.
- Refunds keep their purchase's category and have a positive amount. The contract scores them like any spending transaction.
- Whether Income is a predicted class stays open (from FR-1).

### 7. Module layout

```
src/smart_financial_coach/data/
  labels.py                 load_truth; label contract: matching rules, ignore masks, oracle
  generator/
    events.py               spike overlay (`spike_extra` process), realized spend, tiers
    calibration.py          calibrate on normal purchases only
    expected.py             expected count and spend per period (new)
    dataset.py              schema additions
    validate.py             label checks, oracle gate
    cli.py                  `sfc-data labels`
tests/unit/test_truth_isolation.py   nothing under intelligence/ reads truth
configs/data/default.yaml   weak_lift, max_drivers, contract params; test users 40 per persona
```

New dependency: `scipy`, for the Poisson tail probability.

## Options considered

### A. What counts as a positive

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Planted events only; weak ones tiered (chosen)** | Labels are exact and independent of any detector; tiers expose weak labels | Natural extremes count against precision |
| (b) Planted events plus natural extremes found by a rule | Fewer "unfair" false positives | Labels then come from a detector, so evaluation becomes circular |
| (c) Planted events only, weak ones dropped | Simplest | Hides that noise cancels real overspends; recall overstated |

### B. How to know a spike's realized effect

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Overlay process (chosen)** | Exact split of normal vs extra spend; normal spending independent of events | Changes the content hash once |
| (b) Measure lift after generation against a median | No generator change | A measurement, not truth; depends on the baseline chosen |
| (c) Generate twice, with and without spikes | Exact counterfactual | Doubles runtime; calibration may diverge between runs |

### C. Granularity overlap (week inside month)

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Ignore (chosen)** | Neither rewards nor punishes a correct but differently-sized flag | Slightly fewer scored periods |
| (b) Count as true positive | Rewards detecting the spike | A detector can inflate TP counts by flagging all four weeks |
| (c) Count as false positive | Strict | Punishes a real, visible elevation |

### D. Enforcing truth isolation

| Option | Pros | Cons |
| --- | --- | --- |
| (a) SQLite authorizer on the model connection | Enforced at query time; one file kept | Only protects code that uses the guarded connection |
| (b) Separate `truth.sqlite` file | Structurally impossible to read by accident | Two files to keep together; reverses an FR-1 decision |
| **(c) Convention plus code review (chosen)** | No work; matches FR-1 | Leaks would be silent |

### E. Test-set size for anomaly metrics

The test population has 268 unusual charges and 168 spike periods. At 0.70 precision on about 170 flags, the 95% interval is about ±0.07, too wide to tell 0.68 from 0.75.

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Tune thresholds on train users, report on test users; double test users to 40 per persona (chosen)** | Keeps realistic prevalence; interval about ±0.05 | 360 users instead of 300; generation about 20% slower |
| (b) Raise event rates in an eval-only spec | Many more labels cheaply | Changes prevalence, and precision depends directly on prevalence |
| (c) Report on all users | Most labels | Thresholds tuned and reported on the same users |

### F. Weekly spike targets

The review showed that the count oracle reaches only 0.05–0.08 precision at recall 0.5 on weekly spikes with the floor at 1 purchase a week.

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Raise the weekly floor to about 6 purchases a week | Gate holds at 0.70 | Changes the modeled behavior; about 70 weekly spikes left, mostly Dining; sparse categories untested |
| (b) Larger weekly multipliers | Keeps every category | Changes the modeled behavior; spikes become unrealistically easy |
| **(c) Target relative to the oracle; no weekly gate (chosen)** | Behavior unchanged; target always achievable by construction | Weekly results are diagnostic only on this data; PRD changes |
| (d) Report weekly, no target | No PRD change | No bar on weekly quality at all |

## Data quality checks

New checks in `validate`, all failing loudly:

- **Label consistency:** `anomaly_kind` is set if and only if `process = unusual_charge`. Every `duplicate` has a `related_transaction_id` with the same user, text and amount, earlier by at most 90 minutes. Every `new_merchant_large` merchant is new to that user at that time. Every `amount_outlier` is above the merchant's price p99 at the user's scale.
- **Spike integrity:** periods inside the calendar and after the warm-up; Monday or month-first starts; no overlap within a category (including the ±7-day buffer); no month above `peak_threshold`; `base_spend + extra_spend` matches the ledger to the cent.
- **Tiers:** `weak` share within a spec range (default ≤ 25% for weeks, ≤ 15% for months; the audit measured about 18% and 8%).
- **Ceiling:** count-oracle precision at recall 0.5 ≥ 0.70 for monthly spikes and for unusual charges. Weekly oracle precision is recorded in `meta`, not gated.

## Testing

- **Unit:** each contract rule on hand-built frames: duplicate original ignored, week-in-month ignored, warm-up ignored, unusual-charge period ignored, reason mismatch counted. Overlay: with spikes off, normal purchases keep the same IDs, dates, amounts and merchants. Driver cap enforced. Isolation test: a planted `truth_` reference under `intelligence/` fails it.
- **Statistical:** on `small.yaml`, the mean realized lift matches the mean planted multiplier within tolerance (superposition preserved).
- **Integration:** `generate small.yaml` → `validate` passes → `labels` report runs → a random-score detector gets precision near prevalence, and the monthly oracle beats 0.70.

## Milestones

1. **Feasibility first:** expected counts and spend (`expected.py`, `truth_expected`) and the count oracle, on the current generator. Confirm the monthly and unusual-charge gates pass and record the weekly oracle. The script in the PR review mostly covers this; everything below depends on its result.
2. Label contract and `data/labels.py` on the existing tables. This unblocks the evaluation harness.
3. `related_transaction_id`; spike overlay with calibration on normal purchases and the `spike_extra` process; realized spend and tiers. Regenerate default.
4. `sfc-data labels`; label checks and oracle gate in `validate`; isolation test; test-population size change.
5. Update the FR-1 doc, the Technical Design and the PRD.

## Decisions and open questions

**Decisions** (Oct 1, 2026)

- [x] Positives are planted events only, with `clear` / `weak` tiers (A-a).
- [x] Spikes become an overlay process; one-time content hash change (B-a).
- [x] Granularity overlaps, warm-up, duplicate originals and unusual-charge periods are ignored (C-a).
- [x] Truth isolation stays a convention; no guarded connection, plus a unit test that nothing under `intelligence/` reads truth (D-c).
- [x] Anomaly thresholds tuned on train users, reported on test users; test users 40 per persona (E-a).
- [x] Modeled behavior is not tuned to targets. The spike volume floor stays at 1 purchase a week, before calibration, for both granularities; weak threshold 1.3× expected.
- [x] Oracle is the count-based Poisson tail at a fixed recall of 0.5. `validate` fails when it is below 0.70 for monthly spikes or unusual charges.
- [x] Weekly spikes: no gate; target is 70% of the oracle's precision at the same recall (F-c).
- [x] Spike metrics are scored on true categories; a period's spend is every transaction in the category, and expectations cover the same set.
- [x] Driving transactions: at most 5, scored by excess coverage.

**Document updates (to apply)**

- [ ] Technical Design ground truth table: add `related_transaction_id`, the new `truth_periods` columns and `truth_expected`.
- [ ] Technical Design: `score_transactions` returns a reason **code** with the plain-language reason, so reason accuracy can be scored.
- [ ] Technical Design controls: "anomaly thresholds tuned on train users, reported on test users" and "spike metrics scored on true categories".
- [ ] PRD success metrics: spending-spike precision ≥ 0.70 for monthly periods; weekly periods at ≥ 70% of the oracle's precision at the same recall.

**Open questions**

- [ ] Is Income a class the categorizer predicts? (Carried from FR-1.)
- [ ] Does the unusual-charge oracle reach 0.70? Not yet measured; milestone 1 answers it. If not, its target moves the way the weekly one did, rather than the behavior changing.
- [ ] Do we want a "silent spike" kind, a gradual multi-week drift? It is realistic and FR-8 doesn't exclude it, but it needs its own label shape.
