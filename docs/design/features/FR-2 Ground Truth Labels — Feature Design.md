# FR-2 Ground Truth Labels — Feature Design

Sep 30, 2026 · @Sidd · Status: **Decisions made, ready to implement** · Branch: `feature/fr-2-ground-truth`

## Summary

This feature makes the ground truth that FR-1 already writes precise, trustworthy and safe to use, so every quality number in the evaluation report means what it says.

- **Requirement:** FR-2 (P0): *"Synthetic data carries known true categories and known anomalies (unusual charges and spending spikes), so quality can be measured."*
- **Where we start:** FR-1 already plants unusual charges and spending spikes as generator events and writes `truth_*` tables (see FR-1 Synthetic Data Generator — Feature Design). FR-2 does **not** add a second injector.
- **What is missing:** an audit of the default dataset (below) shows the labels record what was *planned*, not what the data *shows*. 18% of weekly spikes leave no visible trace. 22% of unlabeled weeks look like a spike to a simple rule. No rule says how a weekly flag inside a labeled monthly spike should be scored.
- **Approach:** four additions, all generic and spec-driven:
  1. A written **label contract**: what is positive, negative or ignored at each level.
  2. **Realized-effect annotations** on every planted spike, so weak labels are visible.
  3. An **oracle ceiling** that shows the PRD targets are reachable on this data.
  4. **Label checks** in `validate`, and a `labels` report command.
- **Decisions** were made on Oct 1, 2026 and are recorded in [Decisions and open questions](#decisions-and-open-questions).

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
3. Prove, per dataset, that the PRD precision targets are reachable by an ideal detector. If they aren't, the spec is wrong, not the model.
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
from smart_financial_coach.evaluation.labels import load_truth

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
| expected_spend | REAL | Expected discretionary spend in the period without the spike, from the generator's own rates and prices |
| base_spend | REAL | Realized spend from normal purchases in the period |
| extra_spend | REAL | Realized spend from the spike's extra purchases |
| tier | TEXT | `clear` or `weak` (see [Spike tiers](#3-spike-tiers)) |

**truth_expected_spend** (new, eval only)

| Column | Type | Notes |
| --- | --- | --- |
| user_id, category, granularity, period_start | TEXT | Primary key; only categories eligible for spikes |
| expected_spend | REAL | Expected discretionary spend for the user's normal behavior: seasonality, income coupling and day-of-week included, spikes excluded |

Size on the default spec: about 300 users × 4 categories × (156 weeks + 36 months) ≈ 230k rows, a few MB. It powers the oracle ceiling.

**meta** (new keys): `label_contract_version`, `oracle_*` ceiling stats, `ambiguous_error_share`.

## Design

### 1. Label contract

The contract is code (`evaluation/labels.py`) and is versioned in `meta.label_contract_version`. Every rule below is a spec parameter with the default shown.

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

| Flagged period | Outcome |
| --- | --- |
| Matches a `truth_periods` row | True positive |
| Week inside a labeled **monthly** spike of the same category | Ignored. It really is elevated, but it isn't the labeled unit |
| Month containing a labeled **weekly** spike of the same category | Ignored. Lift is diluted to roughly 1.2–1.5×, too ambiguous to call either way |
| Contains a planted unusual charge in the same category | Ignored. The FR-7 metric already scores that charge |
| Starts in the user's first `baseline_months` (3) | Ignored |
| Anything else | False positive |

Recall is reported on all labels and on `clear` labels only. Precision is the same for both.

**Driving transactions** (FR-8 "the transactions driving it"): extra spike purchases are statistically identical to normal ones, so no transaction is uniquely "the driver". The contract checks that every returned transaction is in the period and category, and that together they cover at least half of the excess over `expected_spend`.

### 2. Spikes as an overlay process

Today a spike multiplies the discretionary purchase rate inside one Poisson draw. The normal and extra purchases can't be told apart, so the realized effect is unknown.

**Change:** draw normal purchases at the unspiked rate λ as today, then draw extra purchases at rate λ·(m − 1) for the spike's days with a separate seeded generator. Both are `discretionary` in `truth_transactions`. The sum has the same distribution as today (Poisson superposition), so realism is unchanged.

What this buys:

- `base_spend` and `extra_spend` are exact, so each label shows how much of the period was the spike and how much was noise.
- Normal purchases no longer depend on whether spikes are on. `clean.yaml` and `default.yaml` users share identical normal spending, a free controlled comparison.

Cost: the default dataset's content hash changes once; regeneration is required.

### 3. Spike tiers

`tier = weak` when the period's realized total `base_spend + extra_spend` is below `expected_spend × weak_lift` (default 1.3). Otherwise `clear`.

Weak labels stay in the data. Dropping them would hide how often realistic noise cancels a real overspend. Reporting both recalls keeps the headline number honest without letting weak labels dominate model selection.

The spike volume floor (`min_weekly_rate`, 1 purchase a week) stays the same for both granularities, so low-volume categories such as Groceries keep weekly spikes. Expect about 18% of weekly labels to be `weak`; the audit shows weekly labels below 1.5 purchases a week are weak 27% of the time. A higher weekly floor (e.g. 3 a week) was considered and rejected: tiers already separate weak labels, and keeping every category eligible tests detectors on sparse categories too.

### 4. Oracle ceiling

The generator knows each user's true expected spend per period. The **oracle** ranks every period by `realized / expected_spend`. This is the best possible "unusual for this user" score, with no estimation error.

- `sfc-data labels` reports the oracle's precision at the recall a simple baseline reaches, and its best precision at recall ≥ 0.5, per granularity.
- `validate` fails when oracle precision at the baseline's recall is below the PRD target (0.70). Then no model can pass on this data, and the spec needs tuning (spike multiplier, volume floor), not the model.
- The same is done for unusual charges with a transaction oracle: the true merchant price distribution plus a perfect duplicate check.

The oracle is an upper bound on the ranking problem only. Real detectors also have to estimate `expected_spend`, so a model at the oracle is not expected.

### 5. Truth isolation

Isolation stays a convention, as in FR-1 option D-b: model-visible tables have plain names, truth tables are prefixed `truth_`, and model code reads only the former. FR-2 adds nothing to enforce it.

- `load_truth(path)` in `evaluation/labels.py` is the intended reader of truth tables; model code under `intelligence/` should not import it.
- Code review is the control. A guarded connection (an SQLite authorizer denying reads of `truth_*`) was considered and deferred; see option D.

### 6. Category labels

No generator change. FR-2 records:

- The ambiguity ceiling (`meta.ambiguous_error_share`, 1.3% on default). The label report shows per-class F1 for the majority-category oracle, so Groceries and Shopping are judged against what is achievable.
- Refunds keep their purchase's category and have a positive amount. The contract scores them like any spending transaction.
- Whether Income is a predicted class stays open (from FR-1).

### 7. Module layout

```
src/smart_financial_coach/data/
  generator/
    events.py               spike overlay, realized spend, tiers
    expected.py             expected spend per period (new)
    dataset.py              schema additions
    validate.py             label checks
    cli.py                  `sfc-data labels`
src/smart_financial_coach/evaluation/
  labels.py                 load_truth; label contract: matching rules, ignore masks, oracle
configs/data/default.yaml   weak_lift, contract params; test users 40 per persona
```

No new dependencies.

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

## Data quality checks

New checks in `validate`, all failing loudly:

- **Label consistency:** `anomaly_kind` is set if and only if `process = unusual_charge`. Every `duplicate` has a `related_transaction_id` with the same user, text and amount, earlier by at most 90 minutes. Every `new_merchant_large` merchant is new to that user at that time. Every `amount_outlier` is above the merchant's price p99 at the user's scale.
- **Spike integrity:** periods inside the calendar and after the warm-up; Monday or month-first starts; no overlap within a category (including the ±7-day buffer); no month above `peak_threshold`; `base_spend + extra_spend` matches the ledger to the cent.
- **Tiers:** `weak` share within a spec range (default ≤ 25% for weeks, ≤ 15% for months; the audit measured about 18% and 8%).
- **Ceiling:** oracle precision at the baseline's recall ≥ 0.70 at both granularities and for unusual charges.

## Testing

- **Unit:** each contract rule on hand-built frames: duplicate original ignored, week-in-month ignored, warm-up ignored, unusual-charge period ignored, reason mismatch counted. Overlay: with spikes off, normal purchases are byte-identical to spikes on.
- **Statistical:** on `small.yaml`, the mean realized lift matches the mean planted multiplier within tolerance (superposition preserved).
- **Integration:** `generate small.yaml` → `validate` passes → `labels` report runs → a random-score detector gets precision near prevalence, and the oracle beats the target.

## Milestones

1. Label contract and `evaluation/labels.py` on the existing tables (no regeneration). This unblocks the evaluation harness immediately.
2. `related_transaction_id`, spike overlay, realized spend and tiers; regenerate default.
3. `truth_expected_spend`, oracle ceiling and `sfc-data labels`.
4. Label checks in `validate`; test-population size change; update the FR-1 doc and Technical Design.

## Decisions and open questions

**Decisions** (Oct 1, 2026)

- [x] Positives are planted events only, with `clear` / `weak` tiers (A-a).
- [x] Spikes become an overlay process; one-time content hash change (B-a).
- [x] Granularity overlaps, warm-up, duplicate originals and unusual-charge periods are ignored (C-a).
- [x] Truth isolation stays a convention; no guarded connection (D-c).
- [x] Anomaly thresholds tuned on train users, reported on test users; test users 40 per persona (E-a).
- [x] Spike volume floor stays at 1 purchase a week for both granularities; weak threshold 1.3× expected.
- [x] `validate` fails when the oracle can't reach the PRD precision target.

**Technical Design updates (to apply)**

- [ ] Ground truth table: add `related_transaction_id`, the new `truth_periods` columns and `truth_expected_spend`.
- [ ] `score_transactions` returns a reason **code** with the plain-language reason, so reason accuracy can be scored.
- [ ] Controls: "anomaly thresholds tuned on train users, reported on test users".

**Open questions**

- [ ] Is Income a class the categorizer predicts? (Carried from FR-1.)
- [ ] Should the PRD precision targets be stated per granularity? The audit suggests weekly spikes are inherently noisier than monthly ones.
- [ ] With the volume floor at 1, can the oracle reach 0.70 precision on weekly spikes? If not, the oracle gate fails `validate`, and we must raise the floor or the multiplier, or restate the weekly target.
- [ ] Do we want a "silent spike" kind, a gradual multi-week drift? It is realistic and FR-8 doesn't exclude it, but it needs its own label shape.
