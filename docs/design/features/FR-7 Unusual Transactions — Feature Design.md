# FR-7 Unusual Transactions — Feature Design

Oct 3, 2026 · @Sidd · Status: **Accepted** (owner, Oct 3, 2026, on #30) · Implementation in review: #33, #34, #37, #39, #40 ([Status](#status-oct-3-2026))

## Summary

This feature flags individual charges that are unusual for the user who made them, each with a plain-language reason, and measures that against FR-2's planted labels.

- **Requirement:** FR-7 (P0), *"Flag individual transactions that are unusual for this user, with a plain-language reason."* PRD success metric: precision ≥ 0.70, recall above a simple rule-based alternative. NFR-7: every flag has a reason.
- **Starting point:** FR-1 plants three kinds of unusual charge, FR-2 scores flags against them (`Truth.score_transactions`, reason accuracy included), and FR-2's oracle reaches 0.98 precision at recall 0.5, so the target is reachable on this data. Nothing detects anything yet: `detect_anomalies` returns "not available yet", and the "Worth a look" screen (1f) links to its mockup.
- **Feasibility, measured on train users only** ([evidence](#feasibility)):
  - **The Technical Design's baseline fails.** A per-user z-score on amount finds 4.5% of planted charges at the flag volume below. Planted charges are unusual *for a merchant*, not large for the user.
  - **Three simple rules, one per kind, reach precision 0.71 at recall 0.73** (user bootstrap: 0.68–0.74 and 0.71–0.76), with 97% of reasons right. That's about 1.7 flags per user per year.
  - **Duplicates are solved** (precision 0.97, recall 0.98). **New merchants need a population prior:** what other users pay at the same merchant (precision 0.77). **Amount outliers are the hard part** (precision 0.42, recall 0.48), and they decide the round.
- **Approach:**
  1. **Score each charge against the user's earlier history only** (point in time), as a nightly job would, so a flag never depends on what happened after it.
  2. **Merchant profiles from many users,** built by the feature pipeline from model-visible data, with at least 3 distinct users per merchant. They supply a merchant's typical price and spread. They change scores only; reasons quote only the user's own numbers (NFR-2).
  3. **A round of four candidates** through the FR-3 framework: the baseline, the rules above, a single probabilistic score, and an isolation forest over the same features. Ranked on out-of-fold recall at a common flag rate fixed before the round, as FR-3 and FR-4 ranked on F1.
  4. **Tune to precision 0.80, gate at 0.70** (owner, Oct 3, 2026). Measured precision swings by about ±0.03 between draws of users, so tuning to exactly 0.70 would pass the gate half the time. The margin costs about 6 points of recall in the POC.
  5. **Reasons are templates over structured evidence,** not model or LLM text, so the dashboard works without the LLM (NFR-6) and the coach quotes exact numbers (FR-14).
- **Owner decisions** (Oct 3, 2026, on #30) settled the tuning target, the baseline comparison, fees and medical bills, and the "Worth a look" window; see [Decisions and open questions](#decisions-and-open-questions). The rest of the design is still for review.
- **Principle (carried from FR-2):** the modeled behavior is not tuned to make the target pass, and test users are scored once, for finalists.

## Context

### What is planted

FR-1 plants about 1.5 unusual charges per user per year, after a 90-day warm-up. On the default dataset (main at `b9902fd`, 360 users) that is 1,591 charges: 1,058 for train users and 533 for test users.

| Kind | How it's planted | Reason code | Hard negatives in the data |
| --- | --- | --- | --- |
| `duplicate` | A copy of a discretionary purchase: same day, same text and amount, 1–89 minutes later | `duplicate` | Real same-day repeats (two identical coffees) |
| `amount_outlier` | 6–15× the merchant's price (at the user's own scale) at one of the user's favorite merchants | `amount_unusual` | Heavy-tailed one-offs; marketplaces where normal orders reach 6–15× the median (FR-2 tags those labels `weak`) |
| `new_merchant_large` | A first charge at a merchant new to the user, in Shopping, Travel or Entertainment, at 6–15× its price and at least $250 | `new_merchant` | About 2,000 normal first visits at ≥ $250 among train users: one-offs, first bills, first purchases |

### What the label contract already decides (FR-2)

- **Positive:** a flagged charge with `anomaly_kind`. **Ignored:** the original of a planted duplicate, and anything in the user's first 90 days. **False positive:** everything else.
- **Recall** is reported on all labels and on `clear` labels only.
- **Reason accuracy:** among true positives, the flag's reason code must match the kind.
- **Thresholds** are tuned on train users and reported on test users (FR-2, option E-a), and test users were doubled to 40 per persona for this metric.

### What the mockup asks for (screen 1f, Web App UI)

Each flag in "Worth a look" has a title, an amount, a date, a kind and a reason line:

| Kind | Mockup reason |
| --- | --- |
| Larger than usual | "About 9× your usual charge from this merchant ($20.99)." |
| Possible duplicate | "Same amount, same merchant, 6 minutes apart." |
| New merchant, large amount | "First purchase here, and your largest Shopping charge in 2 years." |

The Web App UI note leaves flag actions ("I recognize this", "Not me — what now?") and the sensitivity control (FR-9) to this design (gap 6).

## Scope

| | FR-7 (this design) | FR-8 | FR-9 (P1) |
| --- | --- | --- | --- |
| Unit | One charge | One (user, category, month) | Either |
| Gates on | Precision ≥ 0.70 and recall above the baseline, on test users | Monthly precision ≥ 0.70 and recall above its baseline | — |
| Delivers | `score_transactions`, the unusual-charges half of `detect_anomalies`, "Worth a look" charges | `detect_spikes`, the spikes half | A sensitivity control and flag actions |

FR-7 and FR-8 share one tool and one screen but have separate models, rounds and gates. Under the Delivery Plan's sync rule, the spikes half stays "not available yet" until FR-8 promotes.

## Feasibility

**Evidence:** POC branch `poc/fr-7-unusual-charges`, commit `7e0d092`: `experiments/fr7_unusual/feasibility.py` and its output, `results/feasibility.md`. Each number below comes from that output, and one command reproduces it.

Setup: the default dataset generated from main at `b9902fd` (content hash `b4d43bf4`; `validate` passes; transaction oracle 0.983). **Train users only:** 636,475 scored outflows, 1,058 planted charges (0.17%) and 7,930 user-months. FR-2's ignore rules apply. **No test user was read.**

What the POC simplifies, so its numbers are optimistic:

- **Thresholds were chosen and measured on the same users.** The round tunes on folds and measures out of fold (§6).
- **Merchant profiles pool all train users' charges for all time,** not as of each charge's month, and the scored user's own charges count towards the typical price.

### Components alone

| Component | Threshold | Flags | Precision | What it finds |
| --- | --- | --- | --- | --- |
| Duplicate: the same text and amount as an earlier charge, within 90 minutes | — | 337 | **0.973** | 328 of 336 duplicates |
| Amount against the user's own history at the merchant, the user's own spread | z ≥ 5 | 1,537 | 0.155 | 237 amount outliers |
| Same, with the spread borrowed from other users at the merchant | z ≥ 4 | 733 | 0.325 | 238 amount outliers |
| Same, borrowed spread | z ≥ 5 | 375 | 0.416 | 156 amount outliers |
| New merchant: amount ranked in the user's own earlier charges | top 1% | 497 | 0.239 | 116 new-merchant charges |
| New merchant: amount against what other users pay there | ≥ 5× | 379 | **0.765** | 268 new-merchant charges, 22 amount outliers |

- **Duplicates are a rule, not a model.** The 9 false positives are real repeat purchases within 90 minutes, which the data plants deliberately as hard negatives.
- **The user's own history is too short to know a merchant's spread.** Half of all (user, merchant) pairs have 5 or fewer charges (median 6). Borrowing the spread from other users at the same merchant, as FR-2's oracle borrows the catalog's, doubles precision at the same recall: about 237 outliers found at 0.33 instead of 0.16.
- **"Large" can't be judged from the user's history alone for a new merchant.** A first visit in the top 1% of the user's earlier charges is planted 24% of the time: one-offs are large too. Against what other users pay at the merchant, a first visit at 5× or more is planted 77% of the time. Planted charges sit at a median 9.8× the merchant's typical price; the 2,040 normal first visits of $250 or more sit at 1.2×. 93% of planted new-merchant charges are at a merchant with a profile.

### Combined (best recall at a precision target; grid on the same users)

| Target | Flags | Per user-month | Precision | Recall | Recall (clear) | Duplicate | Amount outlier | New merchant | Reason accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.70 | 1,091 | 0.138 | 0.709 | **0.732** | 0.736 | 0.976 | 0.482 | 0.759 | 0.972 |
| 0.75 | 963 | 0.121 | 0.759 | 0.691 | 0.695 | 0.976 | 0.339 | 0.788 | 0.971 |
| 0.80 | 883 | 0.111 | 0.807 | **0.674** | 0.678 | 0.976 | 0.341 | 0.734 | 0.969 |

Rule order: a duplicate first, then a new merchant, then an unusual amount. Each charge gets one reason code.

At the 0.70 point:

- **By reason, precision is 0.97 for duplicates, 0.77 for new merchants and 0.42 for amounts.** Amount flags are a third of all flags and most of the false positives.
- **Where the false positives come from:** 174 discretionary purchases and 143 one-offs. The most frequent merchants are medical (Kaiser Permanente, 27), bank fees ("monthly service fee", 13; "overdraft item fee", 11), events and travel (Ticketmaster, Airbnb, United). These are real large or rare charges that the data calls normal. A user might well want to see an overdraft fee. Under the contract it's still a false positive, and the generator isn't changed to suit the detector.
- **Missed new-merchant charges (85):** 23 at merchants with fewer than 3 other train users, so no profile. The other 62 sit at a median 1.7× the merchant's typical price, despite being planted at 6–15× its catalog price. **Not yet diagnosed.** A likely cause is that rarely visited merchants' "typical price" is computed from few users, some of them other planted charges. The round reports recall split by profile size to check.
- **Missed amount outliers (191):** a median z of 3.8; 185 of them are `clear` labels, so these are real misses, not label noise.
- **Uncertainty:** a user bootstrap gives precision 0.68–0.74 and recall 0.71–0.76. On 120 test users the interval is wider still.

### The baseline at the same flag volume

| Baseline | Flags | Precision | Recall |
| --- | --- | --- | --- |
| Per-user z-score on amount (Technical Design) | 1,091 | 0.044 | 0.045 |
| The same plus the duplicate rule | 1,091 | 0.342 | 0.353 |

The Technical Design's baseline reaches precision 0.70 only on its 17 highest scores (recall 0.011), so "recall above the baseline" is compared at equal flag volume (§5).

## Goals and non-goals

**Goals**

1. A promoted `AnomalyScorer` that passes precision ≥ 0.70 on test users and beats the baseline's recall at the same flag volume.
2. Every flag carries a reason code, a plain-language reason and the numbers behind it, all checked at runtime (NFR-7).
3. Point-in-time scoring: a charge is judged only on what was known when it posted.
4. Flags reach the dashboard ("Worth a look"), the coach and outside assistants through `detect_anomalies`, scoped to the session's user.
5. One command reproduces the round and its report (NFR-8).

**Non-goals**

- Spending spikes (FR-8): their own design, model and gate.
- The sensitivity control and flag actions (FR-9, P1). v1 leaves room for them (§8).
- Fraud detection, card blocking or any action on money (out of scope in the PRD).
- Natural anomalies the generator didn't plant. Positives are planted events only (FR-2, option A).
- Real-time alerts. v1 scores nightly (Technical Design, compute timing).

## Design

### 1. Contract and point-in-time scoring

The service contract follows FR-3's batch-first shape rather than the Technical Design's per-user signature, so that one nightly batch can score every user:

```
score_transactions(rows) → one row per outflow in `rows`:
  transaction_id, score, is_flagged, reason_code, evidence, model_version
```

- **Input:** model-visible transaction rows (`transaction_id, user_id, ts, amount, currency, merchant_raw, channel`) for one or more users, with each user's full history up to the scoring date, plus the merchant-profile columns the feature pipeline joins (§2). A single user is a batch of one.
- **Point in time:** each charge is scored only against the same user's *earlier* rows and against merchant profiles as of the first of its month. Rows after a charge never change its score. This is what a nightly job sees, and it means a duplicate's later copy is flagged rather than the original, which matches the label contract.
- **What is scored:** outflows only. Income and refunds are never flagged.
- **Runtime checks** (the FR-3 `Checked` wrapper):
  - one output row per input outflow, in order;
  - `reason_code` is `duplicate`, `amount_unusual` or `new_merchant` when `is_flagged`, and null otherwise;
  - `evidence` holds the fields that code's reason template needs (§5).

  A flag without a reason is a contract error, not a quality issue (NFR-7).
- **The Technical Design's signature** `score_transactions(user_id, transactions)` becomes a thin per-user call over this one. The Technical Design changes in milestone 5.

### 2. Features and merchant profiles

All features come from model-visible columns. Nothing reads `truth_*`, and the isolation test already covers new code under `intelligence/`.

| Feature | Source | Used for |
| --- | --- | --- |
| Merchant key | `normalize_merchant(merchant_raw)` (FR-3's normalizer, which was built for this reuse) | Grouping a user's charges by merchant; first visits |
| Exact repeat | Same user, raw text and amount as an earlier charge, minutes apart | Duplicates |
| The user's history at the key | Count, running median and spread of log amount, earlier charges only | Unusual amounts |
| The user's history overall | Rank of the amount among the user's earlier charges; largest earlier charge per predicted category | Reason text for new merchants; a fallback score without a profile |
| **Merchant profile** | Per key: the typical log amount (median of users' medians), the typical spread within a user (the median over users of each user's median absolute deviation, from users with 2+ charges there), and the number of distinct users | Unusual amounts (spread), new merchants (typical price) |

**Merchant profiles are a feature table, not model parameters.**

- The feature pipeline builds them from model-visible rows of all users, **as of the first of each month**, so a charge never sees later prices.
- A user's own charges are left out of the profile used to score that user.
- A key gets a profile only when **at least 3 distinct other users** have charges there. This is the same distinct-users idea as the feedback agreement rule (Technical Design, "Learning from user feedback"): a string that occurs for one user only, such as `ZELLE TO <name>`, never gets a profile.
- Profiles are rebuilt nightly. A merchant new to the platform gets a profile once three users have been there, without retraining or promoting anything.
- **Why not fit them into the model:** a promoted model is a fixed, versioned artifact. A profile table fitted at training time would know only the train users' merchants, so FR-4's holdout merchants would never get a profile in serving, even after hundreds of users had shopped there.

**In evaluation**, profiles are built the same way, with no labels, from two pools (from review):

- **Validation profiles come from train users only.** Each fold's charges are scored against profiles of all train users, with the scored user's own charges left out. Test users' rows, and the unusual charges planted in them, never shape a validation score or model selection, so the round keeps FR-2's train/test separation.
- **Test and serving profiles come from all users.** Test users' charges at holdout merchants then get profiles from other test users who shopped there earlier, which is what serving would do.

**Categories stay out of the score in v1.** The predicted category (the categorization predictions file) appears only in reason text ("your largest Shopping charge since …"). So categorizer errors can't move an anomaly score, and FR-7's metrics don't depend on which categorizer is promoted. A candidate that uses category features is a later round.

### 3. Candidates

Each candidate outputs one score on a common scale plus a reason code, and each is a class plus a config in the FR-3 registry.

| Candidate | What it is | Why |
| --- | --- | --- |
| **Baseline:** per-user z on amount | The Technical Design's baseline, with no merchant grouping | Required; the PRD's "simple alternative" |
| **Rules** | The POC: an exact-repeat rule, a z against the user's history at the merchant with the larger of the user's and the profile's spread, and a profile price ratio for first visits. Component thresholds are grid parameters; the score is the component that fired, ordered duplicate, new merchant, amount | The POC leader; reasons come for free |
| **Probabilistic** | One predictive model per (user, merchant): a Student-t on log amount whose center and spread shrink from the merchant profile towards the user's own history as it grows. A new merchant is scored against the profile alone. Score = the negative log tail probability; an exact repeat scores as certain. The reason is the term that contributed most | One threshold for all kinds, so the FR-9 sensitivity is one knob; a principled version of the rules' borrowed spread |
| **Isolation forest** | The same relative features (amount ratio and z at the merchant, profile ratio, first visit, minutes since an exact repeat, rank in the user's history), one forest across users. The reason is the feature with the largest contribution | The Technical Design's isolation-based family, on features that make it per-user |

- **Not carried forward:** one-class boundary methods and density estimates on raw features. They need per-user models, and most users have too few charges at most merchants for that (Feasibility, "Components alone").
- **Isolation forest is the weakest on explainability** (NFR-7): its reason is an attribution, not the rule that fired, so it has to win on recall to be chosen (§4).

### 4. Decision rule

Fixed before the round runs. It follows FR-3 and FR-4 where it can.

- **A common operating point** (from review). Every run's cutoff is tuned to precision 0.80 on other folds, but out of fold it lands at different precisions. Ranking on recall at each run's own cutoff would favour the run whose cutoff slips furthest towards 0.70, which is also the run most likely to fail the test gate. So runs are ranked at one flag rate, fixed now: **0.11 flags per user-month**, the POC's 0.80 point. In each held-out fold, a run flags its highest-scoring charges up to that rate (the rate times the fold's user-months). That cutoff needs no labels, so every run is compared on the same number of alerts.
- **Eligibility:**
  - out-of-fold precision ≥ 0.70 at the run's own tuned cutoff;
  - recall at the common flag rate above the baseline's at the same rate.
- **Ranking:** out-of-fold recall at the common flag rate. Ties against the leader are judged by a paired **user** bootstrap: users, not charges, are the unit that varies, as merchants were in FR-4.
- **Reported beside the ranking:** each run's out-of-fold precision and flag rate at its own tuned cutoff. That precision is what predicts the test gate, and it's read before `finalize`.
- **Tie-breaks, in order:**
  1. reason accuracy, with its own paired tie test against the reason-accuracy leader in the tie set (FR-4 §5's pattern);
  2. out-of-fold precision at the run's own cutoff: the higher one is likelier to hold the gate on test users;
  3. batch cost;
  4. explainability (rules and probabilistic before the forest);
  5. operational simplicity.
- **No shipping twins.** Twins exist because comparison runs train under injected label noise (FR-4 §1). Anomaly models don't train on labels, so the task's `shipping_params` is empty and finalize scores the runs themselves.
- **Test users are scored once,** for at most three finalists plus the baseline, through `finalize`.

### 5. The operating point and the gates

**Where the cutoff comes from.** Scores are unsupervised; labels set only the cutoff. A wrapper does it, as `Calibrated` does for categorization: `Thresholded(base, precision=0.80)`. The wrapper's `fit` passes no labels to the base scorer and uses them only to place the cutoff at the target precision on the training users. Within validation folds, the cutoff for a fold comes from the other folds' users.

**Every label-tuned parameter is fitted within the folds** (from review), not only the cutoff. The Rules candidate's grid (the z, the profile ratio and the minimum amount), and any tuned parameter of another candidate, is chosen for each fold on the other folds' users, by recall at the common flag rate (§4), as FR-3's `C` is. A grid chosen once on all train users would carry the POC's in-sample optimism into every validation score.

No change to the `Model` protocol, the runner or promotion is needed (FR-3, "Rule for later features").

**Tune to 0.80, gate at 0.70 (owner, Oct 3, 2026).** The round still reports the 0.75 point for comparison.

- Precision measured on a set of users swings by about ±0.03 (POC user bootstrap, train users). Test users are half as many, so their swing is wider.
- A cutoff tuned to exactly 0.70 on train users would land below 0.70 on test users about half the time: the gate would test the draw, not the model. The same reasoning set FR-4's target.
- In the POC, tuning to 0.80 costs 6 points of recall (0.732 to 0.674) and 19% fewer flags. Duplicate recall doesn't change; most of the loss is amount outliers.
- The tuning target is a parameter of the wrapper and is recorded at promotion. FR-9 later lets users move their own cutoff around it.

**Promotion gates** (on test users):

1. Precision ≥ 0.70 (PRD), with its user-bootstrap interval reported.
2. Recall above the baseline's recall **at the same flag volume** (owner, Oct 3, 2026). The baseline is the Technical Design's per-user z. "Baseline plus the duplicate rule" is reported alongside, not gated: choosing the bar after seeing which rule works would move it. The Technical Design's baseline reaches precision 0.70 only on its 17 highest scores (recall 0.011), so "recall at precision 0.70" would compare against almost nothing. At equal flag volume the comparison asks the PRD's question: does the model find more of what's planted with the same number of alerts?
3. Every flag has a reason (a contract check, so it can't fail silently).

Reason accuracy and recall on `clear` labels are reported, not gated. The PRD sets no target for them.

**If the rank-1 finalist fails a gate,** that is investigated, not resolved by promoting rank 2 (FR-3's rule).

**For real data (v2).** Without labels, a precision target can't place a cutoff. So promotion also records the **flag rate** at the cutoff (flags per user-month; 0.11 at the POC's 0.80 point). On real data, `Thresholded(rate=…)` sets the cutoff from that rate, with no labels, until real feedback (FR-9 actions) can recalibrate it.

### 6. Evaluation task

A new task, `unusual_transactions`, built from the FR-3 framework's parts. FR-3 named most of them in "Path to a general framework".

- **Examples:** every outflow of every user, after the warm-up, with model-visible columns, the merchant-profile columns, and `anomaly_kind` and `tier` attached on the evaluation side.
- **Splits:** `by_user_split`: train users for validation, test users for test. Validation uses 5 folds of train users, grouped by user and stratified by persona. Each fold is scored by a model whose cutoff and tuned parameters were fitted on the other four folds' users, against profiles built from train users only (§2).
- **Leak checks:**
  - no user in two sets;
  - profiles leave out the scored user's own charges;
  - validation profiles contain no test user's rows;
  - each charge's profile is as of the first of its month;
  - the model's input has no truth columns.
- **Metrics** through `Truth.score_transactions`, so the label contract decides every outcome:
  - precision, recall, recall on `clear` labels, and reason accuracy;
  - recall per kind;
  - recall at the common flag rate (the ranking measure), and precision and flag rate at each run's own cutoff;
  - PR-AUC;
  - flags per user-month;
  - user-bootstrap intervals.
- **Diagnostics** for the report (FR-4's pattern):
  - false positives by merchant key and by amount band;
  - missed charges by kind, and by the size of their merchant profile (0, 3–5, 6+ users), which checks the undiagnosed new-merchant misses; and by the user's history length at the charge.
- **Baselines:** the per-user z, run through the same `Thresholded` wrapper and compared at the common flag rate and at the candidate's flag volume; the same plus the duplicate rule, reported only.

### 7. Reasons

A flag's reason is rendered by a small deterministic function from its `evidence`, not by the model and not by the LLM.

| Reason code | Evidence | Reason (example) |
| --- | --- | --- |
| `duplicate` | `original_transaction_id`, `minutes_apart`, `amount` | "Same amount at the same merchant, 6 minutes after an earlier charge." |
| `amount_unusual` | `usual_amount` (the user's median there), `ratio`, `prior_charges` | "About 9× your usual charge here ($20.99, from 14 earlier charges)." |
| `new_merchant` | `rank_in_history`, `largest_since` (date), `category` (predicted) | "First charge here, and your largest Shopping charge since Oct 2024." |

- **Only the user's own numbers appear in a reason.** The merchant profile changes the score, but "other people usually pay $X here" is never shown. It's an aggregate over at least 3 users, but the coach and outside assistants would repeat it, and NFR-2 is release-blocking. The mockup's copy needs only the user's own history.
- **Why templates:**
  - the dashboard renders flags without the LLM (NFR-6) and within 2 s (NFR-5);
  - every number in a reason is a stored value the coach can quote (FR-14, NFR-1);
  - the wording can change without retraining or promoting anything.
- **The kind label** shown in the UI maps from the code: "Possible duplicate", "Larger than usual", "New merchant, large amount".

### 8. Serving

- **Nightly batch job** (Technical Design compute timing (a); a CronJob in the Delivery Plan's topology). It builds merchant profiles, scores each user's charges since the last run with their history, and writes **flag files** the way categorization writes predictions files (FR-3 option C-b): a separate SQLite file per model version and dataset, with one row per flagged charge. The row holds the flag's id, user, transaction, score, reason code, evidence and model version, and the file is written to a temporary name and renamed into place when complete.
- **The tool:** `detect_anomalies(start_date, end_date)` returns the session user's flags in the range, scoped at the data-access layer like every tool:

  ```json
  {"currency": "USD",
   "unusual_transactions": [{"transaction_id": "…", "date": "2026-09-14", "merchant": "…",
     "amount": 189.00, "kind": "Larger than usual", "reason_code": "amount_unusual",
     "reason": "About 9× your usual charge here ($20.99, from 14 earlier charges).",
     "evidence": {"usual_amount": 20.99, "ratio": 9.0, "prior_charges": 14}}],
   "spending_spikes": {"status": "not_available", "message": "…FR-8…"}}
  ```

  `spending_spikes` stays "not available yet" until FR-8 promotes (the Delivery Plan's sync rule). The unusual-charges half switches to real flags in the PR that promotes FR-7's model.
- **The dashboard:** "Worth a look" on the Overview and screen 1f list the last 60 days' flags, newest first, with the mockup's title, amount, date, kind and reason. Transactions (1e) mark flagged rows.
- **The demo bundle** (Web App UI, "Demo build") precomputes flags into its read-only database, as it does categories.
- **Short histories** (PRD risk): no minimum history beyond FR-2's 90-day warm-up (the 30-charge minimum in the first version was unmeasured and dropped in review). New-merchant flags rest on the merchant's profile, not the user's history; when that history is short, the reason says "one of your first charges" instead of "your largest … since". A merchant with no earlier charges from the user gets no amount flag. The coach says when history is too short to judge, instead of saying nothing is unusual. The round reports misses by history length (§6).

**Flag actions (deferred to v1.1, with FR-9).** "I recognize this", "Not me — what now?" and the sensitivity control all write per-user state, and the demo's data is read-only. v1 shows flags without actions. What v1 fixes now, so v1.1 needs no migration:

- a flag's id is stable: the model version plus the transaction id;
- an action will be stored per user, keyed by flag id, in FR-5/FR-6's feedback store;
- "I recognize this" will suppress later flags with the same reason at the same merchant, for that user only. It never changes another user's flags or the model, under the same isolation rule as category overrides;
- "Not me" will show guidance only (contact your bank). Moving money or blocking cards is out of scope.

## Metrics and why

1. **Precision at the tuned cutoff** is the PRD's metric and the gate. False alarms are the PRD's named risk (alert fatigue), and precision measures them directly.
2. **Recall at a common flag rate,** to rank candidates and to compare against the baseline. Every run is judged on the same number of alerts, so a run can't buy recall with a cutoff that slips, and "better than a simple rule" is fair when the baseline can't reach the precision target at all.
3. **Recall per kind and on `clear` labels.** One kind can hide behind another: duplicates alone give 0.31 recall at precision 0.97. Per-kind recall shows whether a candidate wins on the hard kind (amount outliers) or only on the easy one.
4. **Reason accuracy** turns NFR-7's "every flag has a reason" from a presence check into a correctness check. A flag with the wrong reason ("larger than usual" on a duplicate) erodes trust as much as a wrong number.
5. **Flags per user-month** is the alert burden in the user's terms. At the POC's 0.80 point it is 0.11, about 1.3 flags per user per year.
6. **User-bootstrap intervals,** because users are what a fresh sample draws. Charges of one user are not independent.

## Options considered

### A. Where the merchant's typical price comes from

| Option | Pros | Cons |
| --- | --- | --- |
| (a) The user's own history only | No cross-user data; simplest | New merchants can't be judged (precision 0.24 for a top-1% first visit); amount spreads come from a median of 6 charges |
| **(b) Merchant profiles from other users, as a nightly feature table (recommended)** | New-merchant precision 0.77; spreads from many charges; new merchants get profiles without retraining | Cross-user aggregates need a privacy rule (≥ 3 distinct users, own charges left out, never shown in reasons) |
| (c) Profiles fitted into the model at training | One artifact | Holdout and newly popular merchants never get a profile until the next promotion |
| (d) The merchant catalog's price | Exact on synthetic data | Doesn't exist for real data; it's what the oracle uses, as a ceiling |

### B. The cutoff

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Tune to precision 0.70 | Most recall | Passes the test gate about half the time |
| **(b) Tune to 0.80, gate at 0.70 (recommended)** | Passes reliably; fewer alerts | About 6 points of recall in the POC |
| (c) A fixed flag rate | Needs no labels; carries to real data | Ignores the precision target on the data that has labels |

(c) is recorded at promotion for v2 (§5).

### C. "Recall above a simple rule-based alternative"

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Recall at precision 0.70 against the baseline's | Literal | The baseline reaches 0.70 only on 17 flags (recall 0.011), so the gate is nearly empty |
| **(b) Recall at equal flag volume (recommended)** | Fair at any precision; the same number of alerts either way | A comparison point the PRD doesn't spell out |
| (c) Baseline plus the duplicate rule | A stronger "simple rule" | Moves the bar after seeing which rule works; can be reported alongside |

### D. Reason text

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Templates over stored evidence (recommended)** | Exact numbers; works without the LLM; wording changes freely | Less varied phrasing |
| (b) The coach LLM writes reasons | Natural phrasing | The dashboard depends on the LLM (NFR-6); numbers aren't guaranteed (FR-14) |
| (c) The model writes reason strings | One output | Retrain to reword; can't be scored apart from the code |

### E. When charges are scored

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Nightly batch (recommended; Technical Design)** | Matches flags and forecasts; profiles rebuilt with it | A duplicate is flagged the next day |
| (b) On ingestion, with categorization | Same-hour alerts | Profiles and history needed on the ingestion path; no user-facing alert channel in v1 to use the speed |
| (c) On request | Always fresh | Dashboard latency (NFR-5); repeated work |

## Testing

- **Unit:**
  - the exact-repeat rule (the window's edges, different text, different amounts);
  - point-in-time scoring: appending later rows never changes an earlier charge's score;
  - profiles: fewer than 3 users means no profile, and the scored user's own charges are left out;
  - the contract rejects a flag without a reason code or with missing evidence;
  - each reason template renders from its evidence alone;
  - `Thresholded` passes no labels to the base scorer.
- **Isolation:** nothing under `intelligence/` reads truth (the existing test). `detect_anomalies` returns user A's flags to A and nothing to B. Reason text contains no number that isn't in the user's own rows.
- **Framework (toy task):** out-of-fold cutoffs and grid choices (a grid fitted on all users fails the test); recall at the common flag rate; validation profiles without test users; empty `shipping_params` means no twins; finalize scores test users once.
- **Slow (default data):** the promoted scorer passes the gates on test users; the flag file passes its contract.

## Milestones

One PR per milestone.

1. **Contract and features:** the `AnomalyScorer` contract and runtime checks; merchant keys and point-in-time history features; merchant profiles in the feature pipeline (as of month, own charges left out, ≥ 3 users); reason templates.
2. **Task:** `unusual_transactions` with user-grouped folds, validation profiles from train users, label-tuned parameters fitted within folds, `Thresholded`, metrics through the label contract, recall at the common flag rate, diagnostics and the decision rule of §4.
3. **The round:** the baseline and the three candidates on validation, and the comparison report, with the new-merchant misses diagnosed.
4. **Finalize and promote:** finalists scored once on test users; promote if the gates pass. Then the nightly flag job, `detect_anomalies` (the unusual-charges half), "Worth a look" and the demo bundle.
5. **Docs:** the Technical Design (the batch-first contract, merchant profiles in the feature pipeline, the FR-7 row in model selection, the gate at equal flag volume), the PRD's FR-7 metric wording, and the Web App UI's gap 6.

## Implementation notes

Departures recorded from milestone 1 (#33), each reviewed there:

- **The profile spread is the median over users of each user's median absolute deviation** (§2), not the POC's MAD over pooled charges. Leaving the scored user out is then exact and cheap, and a spread needs at least 3 other users with 2 or more charges each.
- **The new-merchant reason uses the user's whole history** ("your largest charge since Oct 2024", §7). The per-category wording in the mockup ("your largest Shopping charge …") needs the predicted category, which exists only at serving time, so it moves to milestone 4's flag job.
- **"One of your first charges" applies below 30 earlier charges,** as wording only. It never decides whether a charge is flagged (§8; the 30-charge flagging rule was dropped in review).
- **`×` in user-facing reason text:** ruff's ambiguous-character rule (RUF001, RUF002) is ignored for the reasons module and its test only.

Later milestones and reviews:

- **Point in time, with a same-minute exception** (§1; #33's re-review). Timestamps have minute resolution, so the order of two identical charges in one minute is unknowable. Each counts as the other's repeat. Rows in *later minutes* never change a charge's score; a same-minute row can.
  - Planted duplicates in their original's minute were otherwise missed whenever the copy's ID sorted first: 10 of 512.
  - On the default dataset, 24 pairs end up with both members flagged: 23 planted pairs, whose originals the contract ignores, and 1 real repeat, which adds one false positive.
  - `detect_anomalies` shows one flag per such pair (#39).
- **Flag rates are per post-warm-up user-month** (§4, §6): the common rate, reported rates and the search budget count only rows the label contract scores. A correctness review before the round found the warm-up being counted (#34), and the first round was discarded.
- **Label-tuned parameters are searched inside `Thresholded`'s fit** (§5): each point is a fresh scorer fitted without labels, so the search runs per fold.
- **The test gate against the baseline uses the common flag rate** (§5), which is equal volume by construction, because gates see only logged metrics (#34).
- **New-merchant reasons get per-category wording at serving time,** from the user's own ledger (#39).

## Status (Oct 3, 2026)

| Milestone | PR | Outcome |
| --- | --- | --- |
| Design | #30 | Accepted (owner); feasibility on `poc/fr-7-unusual-charges` (pinned `7e0d092`) |
| 1. Contract and features | #33 | The contract with runtime checks; point-in-time history; merchant profiles (as of month, leave-user-out, ≥ 3 users); reasons |
| 2. Task | #34 | `unusual_transactions`: labels from the contract, user-grouped folds, train-only validation profiles, ranking at 0.11 flags per user-month, gates; `Thresholded`; the baseline |
| 3. The round | #37 | `isolation_forest` ranks first on validation, at 0.708 against rules 0.659, probabilistic 0.610 and baseline 0.044 (FR-7 Unusual Transactions — Round Results) |
| 4. Finalize, promote, serve | #39 | Test, scored once: precision 0.814 (0.78–0.85), recall at the rate 0.715 against the baseline's 0.051. Promoted `fb6dab21-8b9632e6-b5c5488f`. Flag files, the nightly job, `detect_anomalies`, "Worth a look" and the demo bundle serve its flags |
| 5. Docs | #40 | Technical Design, PRD, Web App UI |

**Open, for the owner (#39):** the forest's history-rank feature is two-sided, contrary to §3. 57 of 1,597 flags (3.6%) call a cheap first visit "large". A one-sided rank costs nothing measurable on validation. Fixing it means re-finalizing on the used test set with a recorded override.

**Known limit** (Round Results): new-merchant charges at merchants that fewer than 6 other users have visited are mostly missed. Neither profile-rule option measured on validation helped. A category-level price prior is the v1.1 idea.

## Decisions and open questions

**Decisions for review**

- [x] Point-in-time scoring, batch-first across users; the Technical Design's per-user signature becomes a thin wrapper (§1) (owner, Oct 3, 2026, on #30).
- [x] Merchant profiles from other users as a nightly feature table: ≥ 3 distinct users, the scored user's charges left out, as of the month; scores only, never shown in reasons (§2, option A-b) (owner, Oct 3, 2026, on #30).
- [x] Categories stay out of the v1 score; the predicted category is used in reason text only (§2) (owner, Oct 3, 2026, on #30).
- [x] Candidates: the baseline, rules, probabilistic, isolation forest (§3) (owner, Oct 3, 2026, on #30).
- [x] Decision rule: rank on out-of-fold recall at a common flag rate (0.11 per user-month), with each run's own-cutoff precision reported; user-bootstrap ties, then reason accuracy, own-cutoff precision, cost and explainability; no shipping twins (§4; revised in review) (owner, Oct 3, 2026, on #30).
- [x] Tune to precision 0.80, gate at 0.70 on test users (§5, option B-b) (owner, Oct 3, 2026, on #30).
- [x] "Recall above the baseline": the Technical Design's per-user z at equal flag volume (§5, option C-b); baseline plus the duplicate rule reported, not gated (owner, Oct 3, 2026, on #30).
- [x] Reasons as templates over stored evidence (§7, option D-a) (owner, Oct 3, 2026, on #30).
- [x] Nightly scoring into flag files; `detect_anomalies` serves the unusual-charges half when FR-7 promotes (§8) (owner, Oct 3, 2026, on #30).
- [x] Flag actions and sensitivity in v1.1 with FR-9; v1 fixes the flag id and the per-user scope now (§8) (owner, Oct 3, 2026, on #30).
- [x] Label-tuned parameters fitted within folds; validation profiles from train users only; no 30-charge minimum (§2, §5, §8; from review) (owner, Oct 3, 2026, on #30).

**Open questions**

1. [x] **The tuning target. Decided (owner, Oct 3, 2026, on #30): tune to precision 0.80, gate at 0.70 on test users.** The same reasoning as FR-4's target: a model as good as the leader passes reliably, not on the draw. About 1.3 flags per user per year at the POC's 0.80 point. The round still reports the 0.75 point (0.69 recall against 0.67 in the POC) for comparison.
2. [x] **Fees and medical bills. Decided (owner, Oct 3, 2026, on #30): left as measured for v1.** Under the label contract they're false positives, and labels aren't changed to suit a detector (FR-2's principle). A separate "fees" insight is a v1.1-or-later idea, outside this design's scope.
3. [x] **The PRD's "simple rule-based alternative". Decided (owner, Oct 3, 2026, on #30): the Technical Design's per-user z baseline, at equal flag volume** (option C-b). Baseline plus the duplicate rule (C-c) is reported alongside, not gated: choosing the bar after seeing which rule works would move it.
4. [x] **The "Worth a look" window. Decided (owner, Oct 3, 2026, on #30): a fixed 60 days in v1.** "Since you last looked" needs per-user state, so it goes to v1.1 with FR-9's flag actions.
