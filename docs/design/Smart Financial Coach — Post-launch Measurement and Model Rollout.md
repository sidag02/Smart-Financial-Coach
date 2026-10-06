# Smart Financial Coach — Post-launch Measurement and Model Rollout

Oct 6, 2026 · @Sidd · Status: **Accepted** (reviewer, Oct 6, 2026, on #77); **a plan to present, not to build** (owner) · Branch: `docs/post-launch-measurement`

## What go-live looks like

This is a plan: what going live with real users would look like. Nothing in it is built or scheduled (owner, Oct 6, 2026).

- **What a person sees:**
  - a 👍 / 👎 under each coach answer;
  - once a goal reaches its date, one question: "Did you reach it?".
  Everything else they already do (correct a category, review the queue, act on an alert) becomes a signal.
- **What the team watches each week:**
  - **six measures:** the correction rate and review-queue completion; the "I recognize this" dismissal rate; forecast calibration as goals resolve; the coach's ratings and grounding spot checks; and retention, the north star;
  - **their counter-measures:** alert-fatigue switches, the coach's safe-message rate and review burden;
  - each read only once there's enough data to mean something.
- **How a new model reaches people:**
  - offline gates, as today;
  - **shadow mode:** it runs on real data and nobody sees it;
  - **staged rollout:** 10% → 50% → 100%, with automatic rollback if a guardrail breaks;
  - an A/B test only when there are enough people for one to answer the question.

## Summary

This design says how we would know the product works once people use it, and how retrained models would reach them without risk. Every model so far was judged on synthetic ground truth. Live, there is no ground truth, only what people do.

- **Six measures**, each from something people already do in the app:

  | Measure | For | Signal it reads |
  | --- | --- | --- |
  | Correction rate; review-queue completion | Categorization (FR-3 to FR-6) | Corrections and confirmations |
  | "I recognize this" dismissal rate | Unusual charges and spikes (FR-7 to FR-9) | Alert actions |
  | Forecast calibration as goals resolve | Goal forecasts (FR-11, FR-12) | Goals reaching their dates |
  | Thumbs up or down, and grounding spot checks | The coach (FR-13 to FR-15) | A rating per answer, and a human review sample |
  | Retention | The product (north star) | People coming back and acting |
  | Shadow mode, then staged or A/B rollout | Retrained models (FR-5, FR-6 and later) | All of the above, by model version |

- **The main gap is exposure, not actions.** The feedback, alert and goal stores already record every correction, confirmation, alert action and goal change, with a time, and corrections carry the model version. Nothing records what a person was **shown**. So none of these rates has a denominator: corrections per transaction seen, items resolved per item shown, dismissals per alert shown. An exposure log would come first.
- **Two signals don't exist yet:** a coach rating (thumbs) and a goal outcome. A goal that passes its date is marked "ended" whatever was saved (FR-10), so whether it was met is never recorded. Calibration needs that outcome.
- **Each proxy is biased, and the design says how.** For example, "I recognize this" means *I know this charge*, not *this flag was wrong*. A recognized first visit to a new merchant was still a correct flag. So the dismissal rate is checked against the sensitivity levels (More often must dismiss more than Balanced) and against a labelled sample before it's trusted as precision.
- **Retention needs real people.** In the demo, visitors share four accounts and the "user" is a browser session that resets on every redeploy, so retention can't be measured until v2 brings real sign-in. Everything else could be checked first on demo traffic and on the FR-5 replay's simulated users.
- **Rollout:** offline gates (as today), then **shadow mode** (the candidate scores the same data, nothing is shown), then a **staged rollout** by user (10% → 50% → 100%) with guardrails on the measures above. A true **A/B test** is used only where the expected effect can be detected in the time available. With the user numbers v1 and early v2 will have, that's rare, and the design says so rather than run underpowered tests.

## Context

### What's recorded today

| Store | Events | Who | Model version | When |
| --- | --- | --- | --- | --- |
| Feedback (`category_corrections`) | `confirm` or `correct`, merchant or transaction scope, from and to category, source (review, edit, coach) | `subject` and `user_id` | Yes | `seq`, time |
| Alerts (`flag_actions`, `alert_settings`) | `recognize`, `not_me`, `expected`; sensitivity changes | `subject` and `user_id` | In the `flag_id` (version prefix) | `seq`, time |
| Goals (`goal_revisions`) | create, update, archive, with saved amounts | `subject` and `user_id` | No | `seq`, time |
| Coach log line (FR-13 to FR-15 §7) | Per answer: backend, model, seconds, tool calls, tokens, dollars, first-attempt and final grounding, retried | No user, by design | Model and effort | Log time |

**Not recorded:** what was shown (transactions, review items, alerts, forecasts), page visits, coach ratings, goal outcomes, the answer text, and any stable person across visits in the demo.

### The demo is not a population

The demo has four shared accounts, a browser-session `subject`, stores that reset on redeploy, and visitors who explore rather than manage their money. Numbers measured there test the pipeline, not the product. Real targets wait for v2's real users. The FR-5 replay's simulated users (with known preferences and truth) can check each proxy against ground truth before then.

## Scope

| | In this design | Not in it |
| --- | --- | --- |
| Measures | The six above: definitions, denominators, segments, biases, alert thresholds | Business metrics (revenue, acquisition) |
| Instrumentation | An exposure log, a coach rating, a goal outcome prompt, forecast snapshots | Third-party analytics |
| Computation | A daily job over the stores and the exposure log, and a metrics report | Real-time dashboards (v2's managed tracing) |
| Rollout | Shadow mode, staged rollout, A/B where it has the power, guardrails, rollback | Changes to offline gates and promotion (FR-3 framework) |
| Data | Demo traffic and the replay's simulated users now; real users in v2 | Real users' data in v1 |

## Design

### 1. The exposure log (the missing denominators)

One append-only table, `exposures`, in the feedback store's file. It's written by the web app and the tools when something is shown:

| Field | Meaning |
| --- | --- |
| `subject`, `user_id` | As in the other stores (session in the demo, the signed-in user in v2) |
| `surface` | `transactions`, `review_queue`, `worth_a_look`, `goal_detail`, `coach` |
| `item_kind`, `item_ids` | `transaction`, `review_item`, `flag`, `goal`; the ids shown, at most 50 per row |
| `model_version` | The model behind what was shown |
| `variant` | `control` or a candidate's version, for rollout (§7) |
| `shown_at` | Time |

- **Counted once per item per day per person**, so reloading a page doesn't inflate the denominators.
- **Coach exposures are what an answer cited, not what its tools returned** (review on #77). A `get_transactions` result with 50 rows doesn't mean 50 transactions were shown. So for each answer, the items in results the answer cited are logged under `surface: coach`. Outside assistants call the same tools but their answers are never seen here, so they log no exposures. Coach exposures are left out of the correction-rate denominator, which is defined on transactions shown on the pages.
- **Privacy:** ids only, never descriptions, amounts or text. It's kept as long as the stores, and in v2 it follows the same row-level security.

### 2. Categorization: correction rate and review-queue completion

**Correction rate.** Corrections per 100 spending transactions a person was shown, per user-month:
- A merchant-scope correction counts every transaction it moved that the person had been shown.
- Split by `familiar` group (FR-5), source (review, edit, coach) and model version.

**"Stayed corrected".** The share of corrected merchants that the person does **not** correct again within 90 days. A drop after a promotion means the new model is undoing people's fixes (FR-3's "whether corrected merchants stay corrected").

**Review-queue completion.**
- Of the review items shown in a week, the share resolved within 14 days.
- The confirm-to-correct split of those resolutions.
- *Errors caught per item shown* (FR-5's burden measure): corrections ÷ items shown.

**Biases and how they're handled:**
- **People correct only what they notice.** The correction rate is a lower bound on the error rate, and it falls when people engage less. So it's read per **engaged** user-month (one with at least one Transactions or review visit), next to an engagement count, never alone.
- **Confirmations can be automation bias** (FR-5 §4). A rising confirm share with a flat correction rate can mean people are clicking through. The replay measures what that looks like, and the time to resolve an item is logged to help tell them apart.
- **A correction mixes two cases:** the model was wrong, or the person sees it differently. Live, they can't be split per event. The agreement step (FR-5 §4) splits them in aggregate: corrections at merchants where people agree are model errors.

**Thresholds (decision 4):** an alert when the engaged correction rate rises 25% over the trailing 8-week median, or when the stayed-corrected rate drops below 90%.

### 3. Alerts: the "I recognize this" dismissal rate as a precision proxy

**Definition.** Per sensitivity level, model version and kind (charge or spike), per week:
- **Dismissal rate:** `recognize` (charges) or `expected` (spikes) ÷ alerts shown.
- **Action rate:** any action ÷ shown.
- **Not-me rate:** `not_me` ÷ shown, the closest thing to a confirmed true positive for a charge.

**Why it's only a proxy:**
- "I recognize this" means *I know this charge*, not *the flag was wrong*. A first visit to a new merchant, or a real duplicate the person already knows about, is a correct flag that gets recognized. So the dismissal rate measures **unhelpful** alerts, which is what alert fatigue is about, and only approximately precision.
- **Most alerts get no action.** A dismissal rate over shown alerts is diluted by non-response. Over acted-on alerts, it's skewed toward people who act. Both are reported.

**Before it would be trusted** (roadmap step M3):
1. **Monotonic in sensitivity.** More often must dismiss more than Balanced, and Less often less. If not, the proxy isn't tracking what the cutoff changes.
2. **Against truth, in simulation.** Give the FR-5 replay's simulated people an alert behavior: recognize ordinary charges with high probability and planted anomalies with low probability, with rates from a small labelled sample. Then fit the mapping from dismissal rate to precision on the replay's ground truth.
3. **Against a labelled sample, live (v2).** Each quarter, 100 shown alerts are labelled by hand. The fitted mapping is checked against them.

**Thresholds (decision 4):** an alert if Balanced's dismissal rate rises 30% over its 8-week median. Separately, a sharp rise in Less often switches is the alert-fatigue counter-measure.

### 4. Goals: forecast calibration as goals actually resolve

Two things are missing. An outcome, and the forecast as it was before the outcome was known.

- **Outcome.** When a goal reaches its date, the Goals page and the coach ask once: "Did you reach your vacation fund?" (yes, partly, no), with the saved amount prefilled. A goal that was **reached** before its date resolves as met without asking. Unanswered goals are recorded as unknown and **left out**, and the share left out is reported, since non-response is likely biased (people skip goals they missed). The alternative, inferring the outcome from the ledger, uses the same share assumption as the forecast, so it would grade the forecast against itself (decision 5).
- **Snapshots.** On the first of each month, each running goal's forecast would be written to a `forecast_snapshots` table:
  - `p_goal_met`, the band, the 80% range and `model_version`;
  - **the goal's revision** (`seq` in `goal_revisions`) it forecast;
  - **next month's net savings:** the median and the 80% range for the person's own monthly net. Today's forecast output is the goal's balance, not next month's net, so this has to be added to the snapshot for the fast proxy below (review on #77).

  Calibration is scored at fixed horizons before the target date (1, 3 and 6 months), so a goal counts once per horizon, not once per day.
- **Goals edited after a snapshot.** People change a goal's target or date. A snapshot taken before that edit forecast a different goal, so it's scored only if no target or date edit followed it. A later edit of the saved amount alone doesn't change the goal and is kept. The share of snapshots dropped this way is reported next to the calibration.
- **Metrics:** at each horizon, the Brier score, met rate per status band (on track ≥ 70%, either way, off track < 30%, the same bands as FR-11) and coverage of the 80% range. The intervals are bootstrapped by person, since one person's goals are correlated.

**This is slow.** Goals resolve months after they're set, so the first real calibration numbers come about six months into v2. A **fast proxy** resolves every month: each person's next-month net savings against the forecaster's own one-month median and 80% range. It covers the same model with no outcome needed and no wait (FR-11's coverage measure, live). Both are reported; the fast one is the early warning.

**Thresholds (decision 4):** an alert if the one-month coverage leaves 70–90% for two months in a row, or if a band's met rate leaves its FR-11 calibration tolerance once 50 goals in that band have resolved.

### 5. The coach: thumbs up or down, and grounding spot checks

**Thumbs.**
- A 👍 / 👎 under each answer, with an optional reason after a 👎: wrong number, didn't answer, tone, or other.
- Stored per answer with: conversation and turn ids, coach model, prompt hash, backend, the tools called, the grounding results (first attempt and final), whether a retry or the safe message happened, and the source ids cited. Not the question or the answer text (decision 6).
- **Read as relative, not absolute.** Few people rate (expect 1–5% of answers), and the unhappy rate more. So 👎 share is compared between prompt or model versions and over time, never against a target.

**Deterministic signals, already logged for every answer:**
- first-attempt grounding rate;
- retry rate;
- safe-message rate;
- refusal rate;
- latency p95 and dollars per answer (NFR-5, NFR-9).
These need no human. Their expected values come from the API, which is what serves people. The M3 suite's API run (4c53318, 51 cases, one run each) measured:
- 94.9% of answers grounded first time, and 100% after the one retry, so no safe message was shown;
- latency p50 4.1 s and p95 6.84 s;
- $0.0112 an answer.

The subscription runs (96.6–100% first time) don't count here: they have no retry and run a different loop.

**Grounding spot checks.**
- Each week, a person reviews a sample of 25 answers: 10 rated 👎, 5 that hit the safe message, and 10 at random.
- They read each answer next to the tool results it cited and mark: every number right, a number with the wrong label (the check's known blind spot: values, not meaning), unsupported claims, and tone.
- **This needs the answer and tool results stored for the sample**, which today's logging deliberately avoids (decision 6). The review uses the same rubric as the M3 suite's judge, so the judge's live agreement can be tracked too.

**Thresholds (decision 4):**
- an alert if the safe-message rate exceeds 2% of answers over a rolling 7 days, with at least 300 answers. The API run showed none after the retry, and about 5% first-attempt failures that the retry fixed. A daily rate on a few dozen answers would fire on one bad answer;
- an alert if 👎 share doubles its 4-week median;
- an immediate review if any spot check finds a wrong-label number.

### 6. Retention, the north star

**Proposed definition (decision 3):** the share of **activated** people who come back and **act** in their fourth week (W4 retention).
- **Activated:** in their first week, a person did at least one of: reviewed or corrected a category, acted on an alert, set or checked a goal, or asked the coach a question.
- **Act:** the same list. A visit alone doesn't count.
- It's reported weekly, by cohort (signup week), with W1, W4 and W12.

**Why acting, not visiting:** the product's promise is that people understand and act on their money. A dashboard people open and ignore would pass a visit-based metric.

**Counter-measures, read alongside it:**
- Alert fatigue: switches to Less often, and the dismissal rate.
- Coach trust: the 👎 and safe-message rates.
- Review burden: queue completion.

A rise in retention bought with nagging should show up in these.

**When:** v2, with real sign-in. In the demo there's no stable person. The pipeline is built and tested on the replay's simulated people, who have engagement built in (FR-5 §7).

### 7. Rolling out retrained models: shadow mode, staged rollout and A/B

Today a model reaches people through offline gates and a reviewed promotion PR. The plan adds two stages in between, for any model whose retrained version would change what people see.

```
offline gates → shadow (2–4 weeks) → staged rollout 10% → 50% → 100% → promoted
                                        (or A/B, when it has the power)
```

**Shadow mode.** The candidate runs on the same inputs as the champion, and its outputs are stored under its own `model_version` and never shown. The predictions store already keeps one file per model version, so nothing changes for callers.

| Model | Compared in shadow | Pass condition (decision 7) |
| --- | --- | --- |
| Categorizer | Agreement with the champion. **Churn:** the share of each person's past categories it would change, a UX cost of its own. **Corrections arriving during shadow:** the share where the candidate already had the person's category | It already has the corrected category for ≥ 50% of shadow-period corrections at merchants where people agree (the model's errors, FR-5 §4). It keeps ≥ 98% of the categories people confirmed. Churn < 2% of transactions for 95% of people |
| Unusual charges, spikes | Flags per user-month at each preset (the rates FR-9 fixed). Overlap with the champion. Actions on overlapping flags | Flag rate within 10% of the preset. Not-me charges the champion flagged are still flagged |
| Goal forecasts | Next-month coverage and Brier on the fast proxy (§4), scored as months resolve | Coverage in 70–90%; Brier no worse than the champion's |

Shadow can't show how people *react* to the candidate's outputs, only how they compare. That's what the staged rollout is for.

**Staged rollout.**
- **Assignment** is by a hash of `user_id` (the `subject` in the demo), so a person sees one model across pages, the coach and outside assistants. Overrides always win (FR-6).
- **Sticky assignment matters most for the categorizer.** Moving a person between models would reshuffle their past categories and totals, and could fire spurious spikes (FR-5 §3 already computes spikes from current categories only).
- **10%, then 50%, then 100%.** Each stage lasts at least a week **and** until each arm has its minimum exposure:
  - at least 200 people per arm;
  - for the categorizer, at least 5,000 shown transactions whose category the candidate changes;
  - for alerts, at least 300 shown flags.

  At early-v2 numbers, 10% may take several weeks to get there. A stage never advances on time alone (review on #77).
- **Guardrails** are checked daily against the control group: correction rate, review items per person, dismissal and not-me rates, coach safe-message rate, and errors.
  - Each guardrail has a minimum count (§8). Below it, the guardrail reports "not enough data" and neither passes nor stops the stage.
  - A breach of any guardrail, with enough data and its interval excluding no change, stops the rollout and sends everyone back to the champion.
  - Rollback is repointing assignment, with no redeploy.
- Each stage's numbers go into the promotion PR, which is still how a model becomes the champion.

**A/B tests.** These only run when they can answer the question:
- The **power** is computed before starting, from the measure's variance in the control group and the number of people available.
- The correction rate is the likeliest candidate: it's per user-month with a known variance.
- Retention isn't: a 2-point W4 change needs tens of thousands of people per arm.
- When the power isn't there, the staged rollout with guardrails is the decision, and the design states that plainly, rather than run a test that can't fail.

**Seasonality and novelty.** Spend and alerts are seasonal (FR-8, FR-12), so control and candidate always run in the same weeks. A first-week dip in corrections can be novelty, so stage decisions use week 2 onward.

### 8. Computing and reporting

- `sfc-metrics daily`: a job that reads the stores, the exposure log, the ratings, the snapshots and the coach log lines, and writes one row per measure, segment and day to a `metrics` table. It's computed from events, never edited by hand, and can be recomputed for any day (NFR-8).
- `sfc-metrics report`: a weekly Markdown report with every measure, its trailing median and the alerts that fired. In v1 this lives in the repo's reports. In v2, the managed tracing and dashboards from the Technical Design take over (decision 2).
- **Demo honesty:** each report states its population (demo sessions, replay simulation or real users) at the top. Demo numbers are never compared with targets.
- **Minimum volumes** (review on #77). A measure below its minimum reports "not enough data", and its alert neither fires nor passes:

  | Measure | Minimum, per window |
  | --- | --- |
  | Correction rate, stayed corrected | 50 engaged user-months; 100 corrected merchants for stayed-corrected |
  | Review-queue completion | 200 shown review items |
  | Dismissal, action and not-me rates | 300 shown alerts per sensitivity level |
  | Forecast calibration | 50 resolved goals per band; 200 person-months for the one-month proxy |
  | Coach safe-message, 👎 share | 300 answers (7 days rolling); 100 ratings for 👎 share |
  | Retention | 100 activated people per cohort |

- **Before eight weeks of data exist,** each "8-week median" threshold uses an expected value instead:
  - the coach: the M3 API run;
  - the dismissal and flag rates: FR-9's measured presets;
  - the correction rate: the FR-5 replay's simulated users;
  - forecast coverage: FR-11's 70–90%.

  Each is replaced by the live median once eight weeks with enough data exist.

## Roadmap (if built)

The order it would be built in, if it ever is. Nothing here is scheduled (decision 1).

| Step | Content | Done when |
| --- | --- | --- |
| M1 | Exposure log; coach 👍/👎 store and UI; forecast snapshots; goal outcome prompt | Each writes on the demo; tests show no text or amounts are stored |
| M2 | `sfc-metrics daily` and `report`; the deterministic coach measures from the log lines | A report on demo traffic, labelled as demo |
| M3 | The proxy checks: the replay with alert behavior and goal outcomes; the dismissal-rate monotonic check; the correction rate against the replay's truth | A results report saying which proxies hold, with the fitted mappings |
| M4 | Shadow mode for the categorizer (the first retrained model, FR-5 §5) | The first retrained model runs in shadow, with its report |
| M5 | Staged rollout: assignment, guardrails, rollback | A rehearsal on the demo with a deliberately worse candidate stops at 10% |
| v2 | Retention cohorts, the live labelled alert sample, grounding spot checks on real answers | With real sign-in |

## Decisions and open questions

Decision 1 is decided. Decisions 2 to 9 are the plan's proposals, and nothing else is decided. The owner can take them up if this is ever built.

1. **Scope:** a plan to present, not to build. Nothing in it is built now, and the roadmap is the order it would be built in. **Decided** (owner, Oct 6, 2026, on #77).
2. **Where metrics live in v1:** a `metrics` table and a weekly report in the repo (*proposed*), or Azure Log Analytics now. The report is free and reproducible; Log Analytics is the v2 path.
3. **North star:** W4 retention of activated people, where acting counts and visiting doesn't (§6). *Proposed.* The alternative is weekly active users, which is simpler but rewards visits.
4. **Alert thresholds** for each measure (§2–§5), with the minimum volumes and pre-data expected values in §8. *Proposed* as starting points, revisited after eight weeks of data.
5. **Goal outcomes:** ask the person at the target date, and leave unanswered goals out (*proposed*), or infer from the ledger, which grades the forecast against its own assumption.
6. **Coach text for spot checks:** store answers and cited tool results for the weekly sample only, kept 30 days, demo data only in v1, and opt-in in v2. In v2, stored tool results hold real transaction descriptions and amounts, so they need the same row-level security as the rest of a person's data, plus an access log of who reviewed what. *Proposed.* Today nothing is stored, which keeps spot checks impossible.
7. **Shadow pass conditions** (§7 table), as in the table. *Proposed*; the categorizer's churn limit is the one most worth the owner's view.
8. **A/B only with power, otherwise staged rollout with guardrails.** *Proposed.*
9. **Open:** who would do the weekly grounding spot check and the quarterly alert labelling. A named reviewer, once there are real users.
