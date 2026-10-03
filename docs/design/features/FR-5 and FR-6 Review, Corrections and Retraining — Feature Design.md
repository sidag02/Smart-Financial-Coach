# FR-5 and FR-6 Review, Corrections and Retraining — Feature Design

Oct 2, 2026 · @Sidd · Status: **Accepted** (owner, Oct 3, 2026); N, the majority and the retraining cadence are **provisional** until the replay · Branch: `docs/fr-5-design` · **Start with [Status and handoff](#status-and-handoff-oct-3-2026)**

## Summary

This feature closes the loop that FR-3's cold-start categorizer was built for. Users review the categories the model is unsure about and correct the ones that are wrong. Their corrections fix their own view at once, and retrain the shared model when enough users agree.

- **Requirements:** FR-5 (P1), *"Mark low-confidence categories for user review"*, and FR-6 (P1), *"Let users correct a category"*. They are designed together, as the Technical Design's "Learning from user feedback" asks: a review queue only matters if corrections flow back.
- **Starting point:** the promoted categorizer (FR-4: bge-small, clean labels, `20eea4fb-44781bc4-c0274576`) is wrong on about 3% of transactions at merchant strings it knows and about 19% at strings it doesn't (validation). It is a good cold model, so review flags only what it is unsure about; its confident errors are left to corrections and retraining.
- **Feasibility, measured on validation data** ([evidence](#feasibility)):
  - One threshold **per familiarity group**: familiar below 0.95 and unfamiliar below 0.8 flag about 13% of a realistic mix of transactions and catch about 74% of its errors.
  - Reviewing per **merchant string**, not per transaction, keeps the burden small: about 4–5 review items in a user's first month, then fewer than 0.4 a month.
  - The simulated feedback replay runs on FR-4's promoted model and regenerated dataset (owner decision, Oct 2, 2026).
- **Approach:**
  1. **Review policy with the model.** The categorizer reports whether each string is familiar, and the promoted artifact carries per-group review thresholds chosen on validation. Flags are computed in the ingestion batch.
  2. **One review item per user and merchant string**, ranked by spend, with a reason the user can read.
  3. **Corrections are events; overrides are state.** A correction or confirmation applies to that user at once, by merchant (default) or for one transaction. Precedence: transaction override, then merchant override, then the model. Every correction can be undone.
  4. **Global labels only by agreement:** a merchant string's category becomes a training label when at least 3 distinct users agree (to be tuned by the replay), including at least one who corrected rather than accepted a suggestion. A string only one user ever sees never leaves that user.
  5. **Scheduled retraining through the FR-3 framework**, on clean labels, gated and promoted the same way, evaluated on data that arrived after its training cutoff from users who supplied no labels. Review thresholds are re-derived with every promotion.
  6. **Use-case ready:** tools on the tool server (`access/tools.py`, and the MCP server) and their JSON shapes, which the web app and the coach call, so the dashboard and the coach can't disagree.
  7. **A simulator** of synthetic users with their own category preferences, who review, correct, slip and occasionally misbehave, replayed month by month to measure the loop.
- **Principle (carried from FR-3):** models are compared on data they didn't train on. For retraining, that means data from later months and from users who supplied no labels.

## Context

What FR-3 and FR-4 hand over:

| From FR-3 and FR-4 | Consequence here |
| --- | --- |
| Confidence calibrated per familiarity group (familiar / unfamiliar string) | The review policy uses the same groups |
| Unfamiliar strings are wrong far more often (19% against 3%), and many of those errors are confident | Thresholds differ per group; review can't catch every error, so corrections and retraining carry the rest |
| The old model's Travel fallback for unfamiliar strings is gone with clean labels (FR-4) | Review items are a mix of categories, not one fallback |
| Predictions are stored per model version in a separate file (`transaction_categories`) | Overrides live in their own store and survive model changes |
| `fit` takes labels as an argument | Retraining on feedback labels needs no model change |
| Batches mix users (one string embedded once) | Overrides are applied per user **after** the shared inference (Technical Design, feedback constraints) |

What the user sees today, without this feature: a category on every transaction, no indication of uncertainty, and no way to fix a wrong one.

## Feasibility

**Evidence:** the promoted model's twin's out-of-fold **calibrated** confidences from the FR-4 round (`21_small_unweighted.ship`, run `6bc58706`, validation only), for model `20eea4fb-44781bc4-c0274576` on data hash `44781bc4e4a5`. For the review burden: test users' model-visible transactions and the promoted model's vocabulary. No test labels are used. The script and its results are on the POC branch `poc/fr-5-review` (`54bc286`, `experiments/fr5_review/`). These replace the first version's numbers, measured on the old model `3f0ccc82` (bge-base trained under injected noise: 2.0% and 41.9% error rates).

Rows at held-out merchants stand in for unfamiliar strings; the seen-merchant sample stands in for familiar ones. The "mix" columns weight them to production at 23.7% unfamiliar rows, the share among test users. Error rates: **2.8%** familiar, **18.6%** unfamiliar.

| Threshold (flag if confidence below) | Familiar: flagged | Familiar: errors caught | Familiar: flags that are errors | Unfamiliar: flagged | Unfamiliar: errors caught | Unfamiliar: flags that are errors | Mix: flagged | Mix: errors caught |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.1% | 1% | 45% | 7.2% | 22% | 57% | 1.8% | 15% |
| 0.60 | 3.3% | 58% | 50% | 15.8% | 39% | 46% | 6.2% | 45% |
| 0.70 | 4.4% | 77% | 49% | 25.8% | 53% | 38% | 9.5% | 61% |
| 0.80 | 5.8% | 98% | 48% | 33.1% | 61% | 34% | 12.3% | 73% |
| 0.90 | 6.0% | 100% | 47% | 43.1% | 67% | 29% | 14.8% | 78% |
| 0.95 | 6.2% | 100% | 46% | 53.5% | 71% | 25% | 17.4% | 81% |

- **Familiar strings:** almost every error sits below 0.8, and flags are nearly half errors at every threshold. Below 0.95 flags 6.2% of familiar rows and catches all their errors.
- **Unfamiliar strings:** no threshold catches 80% of their errors (71% at 0.95), because many of the clean model's remaining unfamiliar errors are confident. Raising the threshold mostly adds correct items: at 0.95, three in four flags are right.
- **The chosen thresholds** (§1; owner, Oct 3, 2026): familiar below 0.95 and unfamiliar below 0.8 flag 6.2% and 33.1% of their groups (about 13% of the mix), catching all familiar errors and 61% of unfamiliar ones (about 74% of the mix's). A third of unfamiliar flags are real errors, so a flagged item is worth a user's glance.
- **Rejected: flag every unfamiliar string on first sight.** It catches every error but asks about strings the model gets right four times in five, which says the cold model can't be trusted. FR-3 and FR-4 exist to make it trustworthy; review flags what it is unsure about, and the confident errors are what corrections and retraining are for (owner, Oct 3, 2026).

**Review burden**, counted per distinct normalized merchant string per user, since one review settles every transaction at that string:

| Period | Familiar strings met | Unfamiliar strings met | Review items at (0.95, 0.8) |
| --- | --- | --- | --- |
| First month | 27.4 | 8.7 | about 4.6 |
| Each month from month 4 (new strings) | 1.6 | 0.8 | about 0.4 |

Items are estimated by applying the per-group flag rates, measured on transactions, to strings. That is an approximation: a string's transactions differ only in amount, channel and hour, so they are usually flagged together, but this wasn't measured.

**Retraining evidence from FR-4's feasibility work:** the same configuration trained without the injected 2% label noise reaches **0.714** validation unseen-merchant macro F1, against 0.512 with it, and 0.988 against 0.969 on known merchants. Clean labels matter far more than anything else measured so far. This bears on what retraining from feedback can achieve and on [open question 3](#open-questions).

**Still to measure:** the simulated replay ([Simulation](#7-simulation-and-replay)): personal accuracy after feedback, global gain on users who supplied no corrections, entrenched errors, and robustness to wrong corrections.

- **Sequencing** (owner decision, Oct 2, 2026): FR-4's milestone 1 regenerates the default dataset once, with this design's `truth_preferences` (schema 4). The replay is built after FR-4's milestone 2 (shipping twins and explicit `label_noise`) and runs on FR-4's promoted twin, so its numbers describe the model that ships.
- The data contract (§7) was accepted on Oct 2 so FR-4's milestone 1 could build it; the rest of the design was accepted on Oct 3, with N, the majority and the cadence provisional until the replay.

## Goals and non-goals

**Goals**

1. Every categorized transaction carries whether it needs review and why, computed with the model's own review policy.
2. A user sees at most a handful of review items a month after onboarding, ranked by how much money they affect.
3. A user can confirm or correct any category (flagged or not) from the dashboard or the coach, for one transaction or every transaction at that merchant, and undo it.
4. A correction changes that user's categories, totals and coach answers at once, with no retraining.
5. One user's corrections never change another user's categories, except through a retrained model built on labels that enough distinct users agree on.
6. Retraining from feedback goes through the FR-3 framework's splits, gates and promotion, so a bad batch of feedback can't ship unnoticed.
7. The loop is measured before launch on synthetic users with preferences and realistic correction behavior, replayed in time order.

**Non-goals**

- Alert sensitivity (FR-9) and showing where numbers come from (FR-16), the other v1.1 items.
- Changing the taxonomy. Clusters of corrections toward a missing category are reported as product signal, not modelled.
- Real authentication (v2). Identity comes from the session as in v1.
- An LLM in categorization (owner decision for FR-4, carried here).
- Redesigning the web app or the coach. They exist (#19, #23); this design adds the review and correction tools, the pages that call them, and the coach's use of them.

## What it will look like

### Flows

| Flow | Where | What happens |
| --- | --- | --- |
| **Review** | Dashboard: a "Review" badge with the count of open items | The user sees a short list: merchant, a sample description, number of transactions, total spend, the suggested category and why it's flagged ("new merchant", "not sure"). They confirm or pick another category. The default scope is every transaction at that merchant |
| **Correct anything** (FR-6) | Dashboard transaction list; spending-by-category drill-down | Any transaction's category can be changed, flagged or not. The user chooses "this merchant" or "just this transaction" |
| **Correct through the coach** | Chat | "That Costco charge is groceries." The coach calls the correction tool, then states what changed ("I've moved 14 Costco transactions, $1,240, to Groceries"). Bulk changes are confirmed before they're applied |
| **Undo** | Dashboard: "Recent changes" | Each correction can be undone; the categories return to what they were |
| **Answers stay in sync** | Dashboard totals, coach tools | Totals, trends, spikes context and coach answers use the **effective** category: the user's override where there is one, otherwise the model's |

A confirmation is an action too: it pins the category for that user, so a later model version can't silently change it, and it counts as a label.

### Tools (tool server)

No tool takes a `user_id`; identity comes from the session (Technical Design, "Security and data isolation"). The web app calls the same tools as the coach, over the tool server's HTTP transport, so the dashboard and the coach can't disagree.

| Tool | Input | Output |
| --- | --- | --- |
| `list_review_items` | `limit` (default 10), `status` (`open`) | Items: `item_id`, `merchant` (display name), `sample_description`, `transaction_count`, `total_spend` (amount and currency), `suggested_category`, `confidence_band` (`low`, `medium`), `reason` (`new_merchant`, `low_confidence`), `first_seen` |
| `resolve_review_item` | `item_id`, `action` (`confirm`, `correct`), `category` (for `correct`), `scope` (`merchant`, `transaction`; default `merchant`) | The correction record and the effect: `transactions_changed`, `spend_moved` per category |
| `correct_category` | `transaction_id`, `category`, `scope` | Same as above |
| `undo_correction` | `correction_id` | The restored categories |
| `list_corrections` | `limit` | The user's recent corrections, newest first, with `undoable` |
| `get_transactions` (existing) | as today | Adds `category_source` (`model`, `you`), `needs_review`, `review_item_id` |
| `get_spending_summary` (existing) | as today | Totals use effective categories; adds `open_review_items` and `unreviewed_spend` so the coach can say how firm a number is |

All outputs are structured JSON with units and currency, like the existing tools. Categories are validated against the taxonomy; an unknown category is rejected, not stored.

### Service contract change

The categorizer's output gains one column:

```
in:  transaction_id, user_id, ts, amount, currency, merchant_raw, channel
out: transaction_id, category, confidence, model_version, familiar
```

`familiar` is model-visible (the normalized string occurs in the model's training rows), so production computes it exactly as validation does. The review policy then needs nothing the service doesn't already know.

## Design

```mermaid
flowchart LR
    ING[Ingestion batch<br/>all users] --> CAT[Categorizer<br/>category, confidence, familiar]
    CAT --> POL[Review policy<br/>per-group thresholds]
    POL --> PRED[(Predictions file<br/>per model version)]
    PRED --> EFF[Effective categories<br/>per user]
    OVR[(Feedback store<br/>corrections, overrides,<br/>review items)] --> EFF
    EFF --> TOOLS[Tool server<br/>dashboard and coach]
    TOOLS -- review, correct, undo --> OVR
    OVR --> AGR[Agreement rule<br/>distinct users]
    AGR --> TRAIN[Retraining<br/>FR-3 framework]
    TRAIN --> GATE[Gates and promotion]
    GATE --> CAT
```

### 1. Review policy

- **Per familiarity group thresholds**, stored with the promoted model (`review_policy` in the manifest), so every model version brings thresholds that match its own confidence.
- **Chosen at promotion on validation**, by a written rule rather than by hand:
  - unfamiliar threshold: the lowest that catches at least **60%** of unfamiliar-string errors on validation;
  - familiar threshold: the highest at which at least **25%** of familiar flags are real errors, **capped at 0.95**.

  On the promoted model's validation predictions, the rule gives 0.8 and 0.95 (Feasibility). The first version asked for 80% of unfamiliar errors, which the clean FR-4 model can't meet at any threshold; the owner lowered it to 60% rather than flag every new merchant (Oct 3, 2026). The cap is needed because this model's familiar flags are at least 25% errors at every threshold measured, so without it the rule's "highest" would run to 1.0 and flag nearly every familiar row.
- **Flags are computed in the ingestion batch** and written to the predictions file (`needs_review`, `review_reason`), so the dashboard reads them without calling the model.
- **Reason:** `new_merchant` for an unfamiliar string below its threshold, `low_confidence` for a familiar one.
- **Why not one threshold:** see Feasibility. **Why not a fixed review budget per user** (top-k by uncertainty): it hides how uncertain the model really is, and makes the queue's meaning change with the user's volume. The budget is applied at display time instead (goal 2), and items are ranked by spend.

### 2. Review items

- **One item per (user, merchant string)**, where the merchant string is the normalized string FR-3 uses (`normalize_merchant`). Raw variants of the same merchant share one item.
- **Opened** when a flagged transaction arrives for a string with no open item and no override for that user. **Updated** (count, spend) as more transactions arrive. **Closed** when resolved, or superseded when a new model version no longer flags the string.
- **Ranked** by unreviewed spend, then recency. The dashboard shows the top 10.
- **Model changes:** after a promotion, predictions are recomputed in the next batch. Strings with an override are never re-flagged. Open items whose string the new model no longer flags close as superseded.

### 3. Corrections, overrides and effective categories

- **Feedback store:** a SQLite file per deployment in v1 (Postgres with row-level security in v2, as the data store), separate from the generated dataset and from predictions files. Predictions are regenerated per model version; feedback is durable user state.
- **In the Oct 6 demo** (owner, Oct 3, 2026): the same tables in a `feedback.sqlite` in a writable folder inside the container, next to the read-only bundle. The app runs as one replica, so there is one store. It lasts until the container restarts or redeploys (every merge to `main`), which is enough for a demo; mounting Azure Files on that folder would make it durable. Visitors share demo accounts, so feedback is keyed by the **browser session**, as chat history already is (#19): two visitors on one account never see each other's corrections, and the agreement rule (§4) counts each session as a distinct user.

  | Table | Key | Holds |
  | --- | --- | --- |
  | `category_corrections` | `correction_id` | Event log: `user_id`, `scope` (`merchant`, `transaction`), `merchant_key`, `transaction_id`, `from_category`, `to_category`, `action` (`confirm`, `correct`), `source` (`review`, `edit`, `coach`), `model_version`, `created_at`, `undone_at` |
  | `category_overrides` | (`user_id`, `merchant_key`) or (`user_id`, `transaction_id`) | Current state, derived from the event log; rebuilt from it at any time |
  | `review_items` | `item_id` | `user_id`, `merchant_key`, `first_seen`, `suggested_category`, `confidence`, `reason`, `status` (`open`, `confirmed`, `corrected`, `superseded`), `model_version` |

- **Precedence:** transaction override, then merchant override, then the model's prediction. A transaction override handles the ambiguous merchants (a warehouse club where one purchase was electronics) without overriding the merchant.
- **Effective categories are computed per user, after the shared inference.** The data-access layer joins that user's predictions to that user's overrides, scoped by the session's `user_id`. Overrides never enter the batch's shared work, so one user's override can't reach another user's rows (Technical Design, feedback constraints).
- **Undo** marks the event undone and rebuilds that user's affected overrides. Nothing is deleted, so the log stays an audit trail.
- **Coach-initiated bulk changes** (more than one transaction) are confirmed in the chat before the tool applies them; the tool takes a `confirm` flag the coach sets only after the user agrees.
- **Aggregates and spike detection use one categorization at a time.** Two things change a user's per-category totals without any change in spending: a promotion recomputes predictions across the user's whole history, and a merchant-scope correction moves a merchant's spend between categories. Spending spikes (FR-8) compare a period with a baseline, so both are always computed from the **current** effective categories, never from a cached baseline under an earlier categorization. Otherwise a relabelling would be flagged as a spike. A test checks that a correction or a promotion alone never produces a spike alert.

### 4. From corrections to global labels

A correction means one of two things: the model was wrong, or the user sees it differently (Technical Design). The rule separates them by agreement across users.

- **Agreement rule:** a merchant string's category becomes a global training label when at least **N distinct users** (default 3) have confirmed or corrected it, and at least **two thirds** of them agree on the category.
- **Confirmations are weaker evidence than corrections (automation bias).** A confirmation accepts the model's own suggestion, and users often accept suggestions without checking. A third of unfamiliar flags are wrong at the chosen thresholds (Feasibility; about half on the old model, many of them its Travel fallback), so three habitual confirmations could turn a model error into a global label, and retraining would entrench it. So:
  - a global label needs **at least one independent correction** to that category, a choice the user made rather than accepted; confirmations alone never create one;
  - confirmations still count toward N and the majority once a correction exists, and still pin the category for the user who confirmed (§3).

  The simulator models accept-the-suggestion behavior (§7), and the replay reports how often a model error becomes a global label.
- **Privacy:** the threshold counts distinct users, never corrections. A string seen by fewer than N users in total (a person-to-person payment, a landlord's name) can never become a global label; it stays that user's override (user story 5).
- **Preferences stay personal:** a merchant where users split (some say Groceries, some Shopping) fails the two-thirds rule and doesn't move the model; each user keeps their override.
- **Agreement is re-evaluated as votes accumulate** (owner, Oct 2, 2026). With N = 3 and two candidate categories, one side always holds at least two of the first three votes, so the two-thirds rule is always met at that point. A preference held by 30% of users wins two or three of the first three votes about one time in five (50%: one in two; 70%: about four in five). So a global label is provisional: every new vote re-evaluates the string, and a string whose agreement falls below two thirds **loses its global label**. The next retraining drops it, through the same gates as any change. The replay reports, per remap, how often a minority or split preference became a global label and after how many votes it was revoked, and N is tuned against that.
- **Robustness:** a label needs several independent users, and a correction from a user whose corrections disagree with consensus unusually often is down-weighted (tuned in the replay). A single user, however active, can't create a global label.
- **Product signal:** strings that repeatedly fail agreement between the same two categories are reported, as taxonomy questions rather than model fixes.

### 5. Retraining

- **Training data:** the original synthetic training rows, plus the **contributing users' own transactions** at globally labelled strings, **from before the training cutoff**, with the agreed category.
- **A global label supersedes the original label at that string** (owner, Oct 2, 2026). Original training rows at a globally labelled string are relabelled to the agreed category. Otherwise a majority preference at a known merchant could never move the model: at a streaming string, every train user's Subscriptions rows would outnumber the few contributors' Entertainment rows, and the replay would report a training-data artifact as a property of the agreement rule. When a label is revoked (§4), the original labels return at the next retraining.
- **Leak check:** non-contributors' transactions never enter training, even at a labelled string: the evaluation below scores non-contributors in later months, and their rows in training would leak into it. A leak check, like FR-3's, stops a retraining whose training rows include any evaluation user's transaction or any transaction after the cutoff. The model's `fit` already takes labels as an argument.
- **Clean labels** (decision, Oct 2, 2026; [open question 3](#open-questions)): the shipped model and every model retrained from feedback train **without injected label noise**. Injected noise stays for experiments that compare candidates. The two differ only in that setting, so it is written explicitly in the configuration that is finalized and promoted (`label_noise: 0`), promotion refuses a configuration that leaves it implicit, and the promotion log records it.
- **Cadence:** scheduled (proposed monthly), and skipped when fewer than a minimum number of new global labels arrived. The replay sets both numbers.
- **Evaluation without reusing a test set:** each retraining is scored on data its training never saw:
  - **later months** than its training cutoff;
  - from **users who supplied no labels** used in that training, as FR-2 separates train and test users.

  New data arrives every cycle, so no test set is scored twice.
- **Gates** (the FR-3 order): primary metric on that held-out data, scored against the non-contributors' own view (no regression on familiar strings; improvement on the strings that gained labels), above the incumbent, and calibration preserved (unfamiliar Brier no worse). Promotion is the FR-3 command with a recorded note.
- **Which gates govern a retrained promotion** (owner, Oct 2, 2026): §5's gates above, scored against the non-contributors' own view, **decide** promotion. FR-3's and FR-4's truth-based numbers (known- and unseen-merchant macro F1 against the true category) are **reported alongside** every retrained candidate, so a shift toward an agreed preference is visible, but they don't block it. A model that correctly learns a majority preference (streaming → Entertainment at 70%) is, by construction, wrong against the truth on those strings, so truth-based gates would block exactly the outcome the loop exists to produce. FR-4's own candidates are unchanged: they are gated on the true category.
- **After promotion:** the review policy is re-derived (§1), predictions are recomputed in the next batch, and overrides persist untouched.
- **Rollback** is promoting the previous version, as in FR-3.

### 6. Isolation and safety

- Tools take no `user_id`; every read and write of the feedback store is scoped by the session's user at the data-access layer.
- Effective categories are computed per user after shared inference (§3).
- Global labels require distinct-user agreement (§4); the training data for retraining contains only globally labelled strings, never a single user's overrides.
- The coach can't change categories without the user's request, and bulk changes need confirmation (§3).
- The correction log records the source and the model version of every change.

### 7. Simulation and replay

The loop is measured on synthetic users before any real user sees it.

- **Preferences (generator, FR-1/FR-2 extension):** each test user gets a preference profile: they see some merchant subtypes in another category. Each remap's adoption rate is chosen to test one outcome of the agreement rule (§4):

  | Subtype | Default category | Some users call it | Adopted by | Tests |
  | --- | --- | --- | --- | --- |
  | warehouse | Groceries | Shopping | 30% of test users | A minority preference stays personal |
  | gym | Health & Fitness | Subscriptions | 30% | Minority stays personal (in the weakest category) |
  | pharmacy | Health & Fitness | Shopping | 30% | Minority stays personal |
  | rideshare | Transportation | Travel | 30% | Minority stays personal (the old fallback category) |
  | books | Shopping | Entertainment | **50%** | Split: early labels flip and are revoked as votes accumulate |
  | streaming | Subscriptions | Entertainment | **70%** | A majority preference becomes a global label |

  - Each test user adopts each remap independently, for every merchant of that subtype.
  - Only test users get preferences; train users keep the default categories, so the cold model's training labels don't change.
  - Preferences are drawn from their own random seed, so adding them changes no other generated row: transactions, merchants and the holdout are identical to a dataset without preferences.

  A new truth table, `truth_preferences(user_id, merchant_id, category)`, holds each user's view (schema version 4; default datasets regenerate). The label contract gains a "user's category": the preference where there is one, otherwise the true category. **FR-4's gates and evaluation stay on the true category;** the user's category is used only by FR-5/FR-6 measures.

  > **Accepted (owner, Oct 2, 2026): this data contract**, the `truth_preferences` table (schema 4) and the preference-aware label contract, is accepted ahead of the rest of this design. FR-4's milestone 1 implements it in its single regeneration. Any later change to it needs a new regeneration, so changes go through review like an accepted design. The preference profiles above (remaps, adoption rates, independence, test users only, their own seed) are part of the accepted contract (owner, Oct 2, 2026), so the contract is complete for FR-4's milestone 1.

- **Behavior:** each month, each simulated user opens the review queue with some probability (engagement). For each item they resolve, they confirm if the suggestion matches their view and correct otherwise, slipping to a wrong category with a small probability. Some users **accept the suggestion without checking** at a set rate (automation bias), confirming wrong suggestions too. Some users also correct unflagged errors they notice, more often for large amounts. A small share of users correct at random (adversarial).
- **Replay:** month by month over the test users' history: ingest, categorize, flag, simulate responses, update overrides, and retrain on the schedule when the agreement rule produces enough labels.
- **A stress setting:** with six remaps at 30% and more, nearly every test user holds at least one preference (76% from the four 30% remaps alone, 1 − 0.7⁴; 96% overall, 1 − 0.7⁴ × 0.5 × 0.3). The replay's burden and corrections-needed are therefore pessimistic, and its report says so.
- **Measures** (Technical Design, "What good will mean"):

  | Measure | Definition |
  | --- | --- |
  | Personal accuracy | Share of a user's spending transactions whose effective category matches their view, over time |
  | Corrections needed | Corrections per user until their categories match their view; repeat corrections of the same string |
  | Global gain | Unseen-merchant macro F1 of each retrained model, on users who supplied none of its labels and on months after its cutoff, **scored against those users' own view** (their "user's category"). Truth-based F1 is reported alongside, so a shift toward an agreed preference shows as a preference, not as a regression |
  | Entrenched errors | Global labels whose category differs from the contributing users' views (e.g. confirmed model errors) |
  | Isolation | Effective categories of users who never corrected don't change except through promoted models |
  | Robustness | Global gain with 5% and 20% adversarial users; no global label from a single user |
  | Calibration after retraining | Unfamiliar Brier and the review policy's catch rate keep their meaning |
  | Burden | Review items per user per month; share of items resolved |
  | Preference outcomes, per remap | Share of non-contributors whose effective category at that subtype matches their view; how often a minority or split preference became a global label, and after how many votes it was revoked. Reported **with intervals** (a user-level bootstrap): with 120 test users, a 30% remap has about 36 holders, enough for the N = 3 rule but few for a precise per-remap rate (review on #20) |

## Metrics and why

1. **Personal accuracy** is the user-facing outcome. The model's accuracy is only part of it: overrides fix what the model gets wrong for that user.
2. **Errors caught per item shown** measures the review queue: how much a minute of the user's attention fixes. Accuracy at a threshold alone would ignore the burden.
3. **Global gain on users who didn't correct** is the only honest measure of learning. Measuring on the correctors would partly measure their own overrides.
4. **Distinct-user agreement** is a privacy rule first and an accuracy rule second (§4).
5. **Brier score of the unfamiliar group** guards the review policy: if retraining makes confidence less honest, thresholds stop meaning what they did.

## Options considered

### A. Review granularity

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Per transaction | Simple | 85 transactions a month per user; the same merchant asked about again and again |
| **(b) Per user and merchant string (recommended)** | One answer fixes every transaction at that merchant; about 0.4 items a month after onboarding | Ambiguous merchants need a per-transaction exception, which the scope choice provides |
| (c) Per merchant across users | Least burden | Mixes users; breaks isolation |

### B. Review threshold

| Option | Pros | Cons |
| --- | --- | --- |
| (a) One fixed threshold | Easy to explain | At 0.8 it catches only 61% of unfamiliar errors; at 0.95, three in four unfamiliar flags are right (on the old model, 0.9 flagged 66% of familiar rows) |
| **(b) Per familiarity group, chosen at promotion by a written rule (recommended)** | Matches how the model's confidence behaves; moves with each model version | Two numbers to explain; needs `familiar` in the contract |
| (c) Top-k most uncertain per user | Fixed burden | Hides real uncertainty; meaning varies with volume |

### C. Where overrides apply

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Inside the model (per-user features or fine-tuning) | One path | Retraining for every correction; leaks across users in shared batches |
| **(b) Per user, after shared inference, in the data-access layer (recommended)** | Immediate, explainable, isolated | A join on every read (small, indexed) |
| (c) Rewrite the predictions file | No join | Predictions stop being a pure function of the model; lost on the next model version |

### D. Turning corrections into training labels

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Every correction is a label | Fastest learning | Personal preferences and private strings leak into the shared model |
| (b) Count of corrections | Simple | One active user can create a label |
| **(c) Distinct-user agreement with a majority (recommended)** | Private strings stay private; splits stay personal; robust to single users | Slower; rare merchants may never qualify |

### E. Retraining trigger

| Option | Pros | Cons |
| --- | --- | --- |
| (a) After every correction | Freshest | Constant churn; no meaningful gate |
| **(b) Scheduled, skipped below a minimum of new labels (recommended)** | Predictable; enough new evidence to gate on | Labels wait up to a cycle (overrides cover the user meanwhile) |
| (c) On drift detection | Retrains when needed | Needs a drift signal this design doesn't have yet |

## Testing

- **Unit:** override precedence (transaction over merchant over model); undo restores the previous state and leaves the log intact; the agreement rule counts distinct users, never corrections, and rejects strings below N users; review item open, update, close and supersede; the review policy rule on constructed confidences.
- **Isolation:** a correction by user A leaves user B's effective categories, totals and tool outputs unchanged; tools reject a `user_id` argument; a feedback-store read without a session user fails.
- **Contract:** tool inputs and outputs validate against their JSON schemas; the web app's calls round-trip; unknown categories are rejected.
- **Integration (small data, stub embedder):** ingest, flag, review, correct, effective totals change, undo, totals restore; a retraining on agreed labels runs through run, gates and promotion, and the review policy is re-derived.
- **Replay (default data):** the measures in §7, with the numbers recorded in the evaluation report.

## Milestones

One PR per milestone.

1. **Contract and review policy:** `familiar` in the categorizer output, `review_policy` in the manifest, chosen at promotion; `needs_review` and `review_reason` in the predictions file.
2. **Feedback store and effective categories:** tables, precedence, undo, per-user effective categories in the data-access layer, isolation tests.
3. **Tools:** `list_review_items`, `resolve_review_item`, `correct_category`, `undo_correction`, `list_corrections`; effective categories in `get_transactions` and `get_spending_summary`; JSON schemas and contract tests against the web app and the coach.
4. **The simulator:** preference profiles and simulated review and correction behavior. The data contract they write into (`truth_preferences`, schema 4, and the preference-aware label contract) is accepted and lands with FR-4's milestone 1, in the same regeneration as FR-4's new holdout.
5. **Global labels and retraining:** the agreement rule (with the correction requirement), a feedback-aware training task, time-forward evaluation on non-contributing users scored against their own view, a leak check on training rows (no evaluation user, nothing after the cutoff), gates, policy re-derivation, and an explicit `label_noise` required at promotion.
6. **Replay and decisions:** the replay on FR-4's promoted twin and the regenerated dataset; settle N and the cadence; PRD and Technical Design updates. (The Feasibility section was re-measured on that model on Oct 3.)

## Status and handoff (Oct 3, 2026)

Written for the session that continues FR-5 and FR-6, human or agent. Read it first.

### Where it stands

- **Accepted by the owner (Oct 3, 2026),** with N = 3, the two-thirds majority and the retraining cadence **provisional** until the replay. Earlier owner decisions stand:
  - **the data contract** (§7): `truth_preferences`, schema 4, the preference-aware label contract (`Truth.user_categories()`), and the preference profiles. It's built and in the default dataset (#20; data hash `44781bc4e4a5`);
  - **clean labels:** the shipped model and models retrained from feedback train with `label_noise: 0`;
  - **retrained promotions** use §5's gates against users' own view; FR-3's and FR-4's truth-based numbers are reported, not gated;
  - **a global label needs at least one independent correction**; labels are re-evaluated as votes accumulate and can be revoked; a global label supersedes the original training label at its string.
- **FR-4 is complete:** its model **`20eea4fb-44781bc4-c0274576`** (bge-small, no class weights, clean labels) is promoted (#28) and live in the demo. Feasibility above is re-measured on it.
- **The product exists:** the web app (#19: `src/smart_financial_coach/experience/web/`, designed in `docs/design/Smart Financial Coach — Web App UI.md`), the tools and MCP server (#23: `access/tools.py`, `access/ledger.py`, `access/mcp_server.py`), and the demo, deployed to Azure for **Oct 6, 2026** by `.github/workflows/deploy.yml` on every merge to `main`. The demo serves a read-only bundle (`experience/demo.py`, `sfc-web build-demo`) from one replica; visitors share a few demo accounts.

### Owner decisions, Oct 3, 2026

1. **FR-5, FR-6 and the retraining pipeline are in the Oct 6 demo.** The project exists to show how the whole system is built, and the feedback loop is one of the strongest things to show.
2. **Retraining is shown as the precomputed replay, walked step by step:** a "Learning" page goes from corrections, to agreement across users, to training data, to retraining, to the gates, to promoted or rejected, with the replay's numbers at each step. No live retraining in the container.
3. **Accept the design now;** N, the majority and the cadence are provisional until the replay.
4. **The review rule** (§1): unfamiliar strings below the lowest threshold catching at least 60% of their errors, familiar strings below the highest threshold whose flags are at least 25% errors, capped at 0.95. On the promoted model: 0.8 and 0.95. Flagging every new merchant was rejected: the point of FR-3 and FR-4 was a cold model good enough to trust, so review flags only what it is unsure about.
5. **The replay runs on test users** (only they hold preferences), **once, with N = 3 and the default settings; nothing is chosen from its results.** Their data was scored once at FR-4's `finalize`, and the replay answers a different question, so that is acceptable; tuning on it would not be. When N or the cadence is tuned later, split the test users into a tuning half and a reporting half.
6. **A retrained model that passes §5's gates is promoted,** with the normal `promote` command. Two consequences, recorded in the promotion log:
   - for the demo: a promotion **by the end of Oct 5** is rebuilt into the demo and clicked through; after that, the demo keeps `20eea4fb`;
   - the promoted model has trained on contributing test users' transactions, so its truth-based numbers are reported on non-contributing test users only; FR-3's and FR-4's full test sets no longer score it cleanly.
7. **The replay's must-have measures:** global gain on non-contributors against their own view (truth alongside), per-remap outcomes with bootstrap intervals, burden, robustness at 5% and 20% adversarial users, and isolation. Deferred: down-weighting users who often disagree with consensus, and "cheap to retrain". Retraining is quarterly in the replay (about an hour of compute).
8. **Visitors' corrections feed the real loop:** a `feedback.sqlite` in the container (§3), keyed by browser session, so each visitor is a distinct user for the agreement rule. The "Learning" page shows live agreement counts from visitors ("Netflix → Entertainment: 2 of 3 users needed"). No scheduled retraining from visitors' feedback.
9. **Web app ownership:** this work touches the web app and the MCP server only where FR-5 and FR-6 need it.
10. **The coach mentions unconfirmed spend** when it affects an answer (open question 5).
11. **The PRD and the Technical Design are updated** in a small docs PR after #29 lands: FR-5 and FR-6 move into the v1 demo, the feedback section is no longer "not yet designed", and the tools table gains the new tools.

### Plan for the Oct 6 demo

One PR per step, each small, since the demo deploys on every merge. After each merge, confirm the deploy succeeded (its smoke test checks `/healthz`) and click through what changed.

1. **`familiar` and the review policy.** `Calibrated` already knows familiarity from `base.scores`; add `familiar` to the contract's output columns and the predictions file. Derive the promoted model's policy with §1's rule and store it next to its manifest (`artifacts/categorization/20eea4fb-44781bc4-c0274576/review_policy.json`; adding it to the manifest would change its checksum fields). Future promotions derive it at `promote`.
   - The twin's predictions (run `6bc58706…`) were in the FR-4 round's session-local MLflow store: treat it as gone. Rerun the twin with `configs/experiments/categorization/fr4/21_small_unweighted.yaml` (about 9 minutes) and use its pooled predictions. Don't `finalize` it.
2. **`needs_review` and `review_reason`** computed in the batch (`batch.py`), written to predictions and the demo bundle.
3. **The feedback store and effective categories:** §3's tables in `feedback.sqlite`, keyed by session in the demo; precedence and undo; effective categories applied per user **after** the shared predictions, in `access/ledger.py`.
4. **Tools:** `list_review_items`, `resolve_review_item`, `correct_category`, `undo_correction`, `list_corrections` in `access/tools.py` and the MCP server; `get_transactions` and `get_spending_summary` return effective categories, `category_source`, `needs_review` and `unreviewed_spend`.
5. **Web pages:** a review badge and queue; correct from the transaction list ("this merchant" or "just this one"); "recent changes" with undo; totals and coach answers updating at once.
6. **The replay** (in parallel with 1–5): built on the POC branch `poc/fr-5-review`:
   - the simulator (§7 Behavior): engagement, slips, an accept-the-suggestion rate, unflagged corrections weighted by amount, adversarial users;
   - the agreement rule (§4), overrides (§3), retraining (§5: `linear_text` + `calibrated`, `label_noise: 0`, original rows relabelled at globally labelled strings, contributors' rows from before the cutoff, a leak check);
   - time-forward evaluation on non-contributors against their own view, truth alongside; the measures in decision 7;
   - its output, the step-by-step record and the measures, saved as a file the demo bundle includes.
7. **The "Learning" page:** the replay's steps and numbers, the gate outcome, and live agreement counts from visitors.
8. **Promotion** if a retrained candidate passes §5's gates (decision 6).
9. **Tests:**
   - two sessions on the same demo account never see each other's corrections;
   - precedence; undo; unknown categories are rejected; tool schemas;
   - a correction alone never raises a spending spike (§3).

### Practical notes

- **Data:** `uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite --force` (about 40 s); confirm data hash `44781bc4e4a5`. Embedding models download to `data/models/fastembed` on first use.
- **Other sessions use the main checkout** (web app and MCP work). Do branch work in a separate git worktree (`git worktree add <path> -b <branch> origin/main`), and never check out, reset or stash in the main checkout.
- **Install the git hooks** (`uv run pre-commit install`). They run ruff, mypy and the 5 MB file check.
- **Working practice:**
  - one PR per milestone from the latest `main` (`main` requires a PR, a green `check` and an up-to-date branch: rebase and rerun the tests when it moves);
  - before every push, check the open PR for review comments and address them first, replying on each thread;
  - rule or design changes go to the owner first, are recorded here with their reason, and are labelled when made after seeing results;
  - test sets are touched only by `finalize`;
  - a rejected promotion's release is deleted.

## Decisions and open questions

**Decisions**

- [x] FR-5 and FR-6 designed together, with retraining.
- [x] Review per (user, merchant string); per-familiarity thresholds chosen at promotion by the rule in §1 (60% of unfamiliar errors; familiar flags at least 25% errors, capped at 0.95; owner, Oct 3, 2026).
- [x] `familiar` added to the categorizer contract.
- [x] Overrides applied per user after shared inference, with transaction over merchant over model.
- [x] Global labels by distinct-user agreement (N = 3, two-thirds majority, both provisional until the replay).
- [x] Retraining scheduled, evaluated on later months from non-contributing users, promoted through the FR-3 gates.
- [x] The web app and the coach use the same tools; bulk coach changes need confirmation.
- [x] A global label needs at least one independent correction; confirmations alone never create one (from review).
- [x] Retraining uses only contributing users' transactions from before the cutoff, with a leak check (from review).
- [x] Global gain is scored against non-contributors' own view, with truth-based F1 alongside (from review).
- [x] Spike baselines and periods always use the current effective categories (from review).
- [x] The shipped model and retrained models train on clean labels; injected noise only for comparing candidates (owner, Oct 2, 2026).
- [x] The replay runs after FR-4, on its promoted twin and the dataset regenerated once with schema 4 (owner, Oct 2, 2026).
- [x] The data contract in §7 (`truth_preferences`, schema 4, and the preference-aware label contract) is accepted now; the rest stays in draft until the replay (owner, Oct 2, 2026). This breaks the loop where FR-4's milestone 1 would build a schema from an unaccepted design.
- [x] The preference profiles: four remaps at 30%, books at 50%, streaming at 70%; independent per user, test users only, their own seed; FR-4 stays on the true category (owner, Oct 2, 2026).
- [x] A global label supersedes the original label at that string; agreement is re-evaluated as votes accumulate and labels can be revoked; the replay reports per-remap preference outcomes (owner, Oct 2, 2026).
- [x] Retrained models are promoted on §5's gates (non-contributors' own view); FR-3's and FR-4's truth-based numbers are reported alongside, not gated. FR-4's own candidates stay gated on the true category (owner, Oct 2, 2026).
- [x] The design is accepted without the replay; N, the majority and the cadence are provisional (owner, Oct 3, 2026).
- [x] FR-5, FR-6 and the retraining pipeline are in the Oct 6 demo; retraining is shown as the precomputed replay, walked step by step (owner, Oct 3, 2026).
- [x] The replay runs once on test users with N = 3 and default settings; nothing is chosen from it; later tuning splits test users in half (owner, Oct 3, 2026).
- [x] A retrained model that passes §5's gates is promoted; for the demo, by the end of Oct 5 (owner, Oct 3, 2026).
- [x] In the demo, feedback lives in a `feedback.sqlite` in the container, keyed by browser session, and feeds the agreement rule (owner, Oct 3, 2026).

**Open questions**

1. [ ] **N and the majority** for global labels: the replay measures how fast global gain arrives against how often a personal preference or an adversarial user leaks.
2. [ ] **Retraining cadence** and the minimum number of new labels per retraining. Quarterly in the replay for now.
3. [x] **Should retraining (and the shipped model) train on injected label noise?** **Decided (owner, Oct 2, 2026): no.** Injected noise stays for experiments that compare candidates, the Technical Design's control against flattering results. The shipped model and models retrained from feedback train on clean labels, with `label_noise: 0` explicit in the promoted configuration (§5). Robustness to the natural noise in feedback labels is measured in the replay instead. Basis: without the injected noise, validation unseen-merchant macro F1 is 0.714 against 0.512 (known 0.988 against 0.969; FR-4 feasibility), and the Travel fallback is the noise's most visible cost in the shipped model. This refines the Oct 1 decision to keep noise, which was about comparing candidates; that discipline is unchanged.
4. [ ] **"Cheap to retrain" as a selection criterion** (carried from the Technical Design): deferred past the demo (owner, Oct 3, 2026); the replay records retraining time per cycle.
5. [x] **Coach answers during review:** should the coach mention open review items when they affect an answer ("$120 of this is still unconfirmed")? **Yes** (owner, Oct 3, 2026), using `unreviewed_spend`.
