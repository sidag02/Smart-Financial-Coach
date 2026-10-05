# Smart Financial Coach — PRD

Sep 29, 2026 · @Sidd

## Overview

Smart Financial Coach is an AI assistant that helps people understand and act on their own spending, in plain English, with answers they can trust.

- **Pitch:** "Ask your money a question and get an answer you can trust."
- **Why now:** People already accept AI assistants for everyday questions, but generic chatbots guess at numbers. People need guidance that is both conversational and correct about their own money.
- **This release (v1):** validates that the coach gives accurate, personalized insight, using realistic synthetic data before real bank data is connected.
- **How it is built:** see Smart Financial Coach — Technical Design.

## Problem statement

People have plenty of transaction data but little insight from it. Four problems stand between the data and useful guidance.

1. **Messy data.** Descriptions like "SQ \*STARBUCKS #4321" don't say what the money was spent on, so people can't see where it goes.
2. **Unnoticed changes.** An unusual charge, or a month of overspending in one category, often goes unnoticed until it hurts.
3. **Uncertain goals.** People set savings goals but can't tell if they are on track, especially when spending is seasonal or income is irregular.
4. **Hard-to-get answers.** Budgeting apps show charts but don't answer the question the user actually has, in their own words.

**Why it matters:** the problems compound. Miscategorized data hides spikes, hidden spikes derail goals, and without clear answers people disengage. Solving them together turns passive records into guidance people act on.

## Goals and scope

v1 proves that an AI coach can give accurate, personalized answers about a person's own money. It runs on synthetic data so quality can be measured against known ground truth before real user data is connected.

**Goals**

1. Categorize transactions accurately enough that users rarely need to fix them.
2. Surface spending that is unusual for each user, with a clear reason, without alert fatigue.
3. Tell users whether they are on track for a savings goal, and by how much.
4. Answer money questions in plain English, using only the user's real numbers.
5. Make the same insights available in other AI assistants the user already uses.

**In scope (v1)**

- Transaction categorization
- Detection of unusual charges and spending spikes
- Savings-goal tracking and on-track prediction
- Conversational coach and a dashboard
- Access for third-party AI assistants
- Multiple users with strict data separation
- Realistic synthetic data with known ground truth

**Out of scope (later releases)**

- Real bank connections and real user data
- Real sign-in and account management
- Mobile app and push notifications
- Moving money, bill pay or other transactions
- Regulated investment advice and product recommendations

## Users and personas

The primary user is an individual managing their own money who wants answers, not spreadsheets. Three personas cover the spending patterns v1 must handle; the synthetic data generator is built from them.

| Persona | Profile | Key question for the coach |
| --- | --- | --- |
| Young professional | Salaried, heavy dining and rideshare, saving for a trip | "Am I on track for my vacation fund?" |
| Family budgeter | Two incomes, groceries and childcare, strong school-year seasonality | "Why was this month so expensive?" |
| Freelancer | Irregular income, lumpy spending | "Can I afford this, given my income swings?" |

**Secondary users:** developers and partners who connect other AI assistants to reuse the same financial intelligence.

## User stories and key scenarios

Each user story maps to one ML capability, and each key scenario is an end-to-end flow v1 must support.

**User stories**

1. As a user, I see my transactions cleanly categorized so I know where my money goes.
2. As a user, I am alerted when a single charge, or my spending over a month, is unusual for me, with a reason.
3. As a user, I set a savings goal and learn if I am on track, and by how much.
4. As a user, I ask questions in plain English and get answers that use my real numbers.
5. As a user, I only ever see my own data.
6. As a developer, I connect another AI assistant and get the same grounded answers.

**Key scenarios**

1. **Monthly check-in:** the user opens the dashboard and sees categories, the monthly trend, flagged anomalies and goal progress.
2. **Goal question:** "Am I going to hit my $3,000 vacation goal by December?" → the coach explains whether they are on track, the gap, and what would close it.
3. **Spending spike:** "Why was August so high?" → the coach points to the spending spike and the charges behind it.
4. **Personalization:** two users ask the same question and get different answers based on their own history.
5. **Isolation:** a user asks for someone else's data; the system has no way to return it.
6. **Third-party assistant:** the same question asked through another AI assistant returns the same numbers.

## Functional requirements

Requirements are grouped by capability and say what the system must do, not how. P0 is required for v1, P1 is planned for v1.1. FR-5 and FR-6 are P1 but were built for the v1 demo (owner, Oct 3, 2026), and so was FR-9 (owner, Oct 4, 2026).

| ID | Capability | Requirement | Priority |
| --- | --- | --- | --- |
| FR-1 | Data | v1 operates on synthetic data for many users over at least two years, reflecting real-world variability: messy merchant text, varied income patterns, seasonality | P0 |
| FR-2 | Data | Synthetic data carries known true categories and known anomalies (unusual charges and spending spikes), so quality can be measured | P0 |
| FR-3 | Categorization | Assign every transaction one category from a fixed set of about 12 | P0 |
| FR-4 | Categorization | Categorize merchants the system has never seen before | P0 |
| FR-5 | Categorization | Mark low-confidence categories for user review | P1 (in the v1 demo) |
| FR-6 | Categorization | Let users correct a category | P1 (in the v1 demo) |
| FR-7 | Unusual spending | Flag individual transactions that are unusual for this user, with a plain-language reason | P0 |
| FR-8 | Unusual spending | Flag spending spikes: monthly spend in a category significantly above the user's normal level, with the size of the deviation and the transactions driving it. Weekly spikes are deferred: even a perfect detector would be wrong most of the time on weekly data (see FR-2) | P0 |
| FR-9 | Unusual spending | Let users adjust alert sensitivity, and act on an alert: recognize it, say a charge isn't theirs, or say a spike was expected | P1 (in the v1 demo) |
| FR-10 | Goals | Let users define a savings goal with an amount and a target date | P0 |
| FR-11 | Goals | Tell the user whether they are on track, the projected amount, the gap, and how certain that is | P0 |
| FR-12 | Goals | Account for seasonal patterns and each user's own income and spending behavior | P0 |
| FR-13 | Coach | Answer natural-language questions about spending, unusual activity and goals | P0 |
| FR-14 | Coach | Use only numbers from the user's data, never estimates; say so when data is missing | P0 |
| FR-15 | Coach | Keep a supportive, non-judgmental tone; do not give investment advice | P0 |
| FR-16 | Coach | Show users where the numbers in an answer came from | P1 |
| FR-17 | Dashboard | Show spending by category, monthly trend, flagged activity and goal progress | P0 |
| FR-18 | Users | Support multiple users; each user sees only their own data | P0 |
| FR-19 | Integrations | Let third-party AI assistants access the same insights and give the same answers | P0 |
| FR-20 | Quality | Measure every capability against the success metrics before each release, repeatably | P0 |

## Non-functional requirements

The most important quality is trust: the coach must never invent numbers or reveal another user's data. Targets apply to v1.

| ID | Category | Requirement | Target |
| --- | --- | --- | --- |
| NFR-1 | Accuracy | Every number the coach states matches the user's data | ≥ 95% of tested answers fully correct |
| NFR-2 | Privacy | No user can see another user's data, however they ask | 100% of attempts blocked |
| NFR-3 | Security | Credentials and keys are never exposed in code, logs or answers | Zero exposures |
| NFR-4 | Safety | No regulated investment advice; defers to a professional when asked | 100% of such requests handled |
| NFR-5 | Responsiveness | Dashboard and chat feel immediate | Dashboard < 2 s; chat answer < 8 s at p95 |
| NFR-6 | Reliability | Dashboard and alerts keep working if the AI assistant is unavailable | Clear message in chat; no dashboard impact |
| NFR-7 | Explainability | Every flag has a reason; every forecast has a range | 100% of outputs |
| NFR-8 | Reproducibility | Quality results can be regenerated and match | Same metrics on rerun |
| NFR-9 | Cost | AI cost per active user is tracked and within budget | Budget TBD |
| NFR-10 | Extensibility | New assistants and new insights can be added without changing existing capabilities | No regressions in existing metrics |

## Success metrics

v1 succeeds when every capability meets its target and each beats a simple rule-based alternative. Targets are initial and will be revisited after the first measurement.

| Capability | Metric | Target |
| --- | --- | --- |
| Categorization | Macro F1 | ≥ 0.90 on known merchants; on new merchants ≥ 0.66 in v1 and ≥ 0.80 in v1.1, through user feedback (FR-5, FR-6) |
| Unusual transactions | Precision, recall | Precision ≥ 0.70; recall above a simple rule-based alternative at the same number of alerts |
| Spending spikes | Precision, recall per monthly period | Precision ≥ 0.70; recall above a simple rule-based alternative |
| Goal forecasting | Forecast error (RMSE); accuracy of the on-track call | RMSE ≥ 15% lower than a naive forecast; on-track call better calibrated than naive. **Target changed before test (owner decision 1 on FR-11's #50):** RMSE at least 15% below last-month naive and no worse than seasonal-naive, since nothing beat seasonal-naive by 15% (the true level in hindsight is only 17–19% better); Brier and calibration by status band are the primary gates. **Measured (v1, Oct 4, 2026):** RMSE 0.28 of last-month naive and 0.88 of seasonal-naive; Brier 0.168 for goals with a track record and 0.202 for new goals, against 0.25 for a flat 50% and 0.396 for naive pace; every status band within its calibration gate. Two caveats: test users were scored a second time, so this is optimistic; and evaluation targets come from a projection close to the model's own median, so band calibration mainly checks around that projection, and the Brier isn't comparable with the first round's (owner decisions 12 and 13 on FR-11's #54; FR-11 and FR-12 Goal Forecasting — Round Results) |
| Coach grounding | Share of answers with every number correct | ≥ 95% |
| Coach safety | Share of unsafe or cross-user requests correctly refused | 100% |
| Coach quality | Rated helpfulness, clarity, empathy, personalization (1–5) | ≥ 4.0 average |

**Revisited after the first measurement:** the new-merchant target was 0.80 for v1. A cold model learns new brands only from text, and the best measured one scores about 0.71 on validation, so v1 gates at 0.66, a level a model that good passes reliably. 0.80 moves to v1.1, where corrections from users teach the model new merchants (FR-4 Unseen Merchant Categorization — Feature Design, §4). The v1 model scored 0.735 on new merchants.

**Spending spikes, first measurement (Oct 4, 2026):** on validation, the simple alternative, a per-user mean ± k·std on monthly spend, finds 16% of planted spikes at 0.035 flags per user-month (about one flag per user every 2.4 years), and no cutoff of it reaches the 0.80 tuning target. That's with the two rules every flag obeys (spend ≥ 1.3× the usual month, at least 2 purchases in it); without them it finds 1.5% under the same harness (4% in the feasibility study, which left ignored months out of its budget). A model that scores how improbable the month's purchase count is finds 52% at the same rate, at precision 0.80 out of fold. It misses spikes from bigger baskets or higher prices by construction, the design's main stated risk. **On test users** (scored once), the v1 model scored precision **0.705** (0.634–0.770) and recall **0.521** at that rate, against the alternative's 0.158, so it passes both targets and was promoted. The precision margin is thin: it was 0.80 on validation. On the categorizer's predicted categories rather than the true ones, precision is 0.777 and recall 0.526 (FR-8 Spending Spikes — Feature Design; FR-8 Spending Spikes — Round Results).

**Learning from feedback, simulated (FR-5, FR-6):** in a replay of three years with 120 synthetic people who see some categories their own way, corrections from 69 of them, turned into training labels only where at least 3 people agreed, raised new-merchant macro F1 for the 51 who never corrected anything from 0.75 to 0.87 against the true categories. It held at 0.83 with a fifth of correctors acting at random, though labels that only their random votes carried reached every retrained model that was promoted, so the agreement threshold stays provisional. That's the route to v1.1's 0.80 target; it's a simulation, and no retrained model is promoted in v1 (FR-5 Feedback Replay — Results).

**Unusual transactions, first measurement:** a per-user z-score on amount, the simple alternative, can reach 0.70 precision only on a handful of alerts, so recall is compared at the same number of alerts: 0.11 per user-month (FR-7 Unusual Transactions — Feature Design, §5). The v1 model scored precision 0.834 at its cutoff, and recall 0.720 at that rate against the alternative's 0.051, from a second scoring of the test users after a feature fix: the first promoted model's history rank was two-sided, so cheap first visits read as large. That model had scored 0.814 and 0.715 on the first scoring (FR-7 Unusual Transactions — Round Results).

**Alert sensitivity (FR-9), measured out of fold (Oct 4, 2026):** Less often and More often flag half and twice as many alerts per post-warm-up user-month as Balanced, the promoted models' cutoffs; on held-out users they do (0.49× and 1.98× for unusual charges, 0.50× and 2.01× for spikes). Precision at Less, Balanced and More is 0.939, 0.792 and 0.455 for unusual charges, and 0.963, 0.799 and 0.491 for spikes. The targets above apply to Balanced, the default: More often is the person's choice to see more borderline alerts, and the page says more of them will be ordinary. In a person's first 90 days, More often waits and shows Balanced's alerts (FR-9 Alert Sensitivity and Flag Actions — Feature Design; FR-9 Alert Sensitivity — Results).

**Caveat:** v1 is measured on synthetic data, which is easier than real data. Targets must be re-validated on real data before launch.

## Risks, assumptions and open questions

The biggest risk is that quality measured on synthetic data does not hold on real transactions; v1 treats real-data validation as a launch gate.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Synthetic data is easier than real data; metrics overstate quality | High | Build realistic noise into the data; validate on real data before launch |
| Coach states a wrong number or gives unsafe advice | High | Grounding and safety are release-blocking metrics |
| Cross-user data exposure | High | Data separation is a release-blocking requirement, tested adversarially |
| Alert fatigue from false alarms | Medium | Precision target; user-adjustable sensitivity and per-alert actions (FR-9, in the v1 demo) |
| New users have too little history | Medium | Say clearly when an insight is limited by short history |
| AI provider outage | Medium | Dashboard and alerts work without the assistant |
| Users distrust AI financial guidance | Medium | Show where numbers come from; supportive tone |

**Assumptions**

- Monthly granularity is enough for goal tracking.
- About 12 spending categories cover most user needs.
- Users accept an assistant that explains but does not move money.

**Open questions**

- [ ] Which bank-data aggregator for the real-data release?
- [ ] What regulatory review is needed before launch (e.g. financial advice disclaimers)?
- [ ] What default alert sensitivity do users prefer? v1 defaults to Balanced, the promoted models' cutoffs; real users' choices will say whether that's right.

## Releases

Each release widens what the coach can do or who it serves; real user data arrives in v2.

| Release | Scope |
| --- | --- |
| v1 | All P0 requirements on synthetic data, plus FR-5 and FR-6: review of uncertain categories, corrections and undo, and how corrections teach the model, shown from a simulated replay; and FR-9: alert sensitivity and actions on alerts (Oct 6, 2026 demo) |
| v1.1 | The other P1 requirement, answer sources; retraining from real feedback, promoted through the gates, reaching the 0.80 new-merchant target |
| v2 | Real bank data, real sign-in, validated quality on real data, clear handling of short histories |
| v3 | Proactive nudges, mobile notifications, deeper goal planning, weekly spending-spike alerts if they can be made reliable |
