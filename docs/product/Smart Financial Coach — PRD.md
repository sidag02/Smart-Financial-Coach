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

Requirements are grouped by capability and say what the system must do, not how. P0 is required for v1, P1 is planned for v1.1.

| ID | Capability | Requirement | Priority |
| --- | --- | --- | --- |
| FR-1 | Data | v1 operates on synthetic data for many users over at least two years, reflecting real-world variability: messy merchant text, varied income patterns, seasonality | P0 |
| FR-2 | Data | Synthetic data carries known true categories and known anomalies (unusual charges and spending spikes), so quality can be measured | P0 |
| FR-3 | Categorization | Assign every transaction one category from a fixed set of about 12 | P0 |
| FR-4 | Categorization | Categorize merchants the system has never seen before | P0 |
| FR-5 | Categorization | Mark low-confidence categories for user review | P1 |
| FR-6 | Categorization | Let users correct a category | P1 |
| FR-7 | Unusual spending | Flag individual transactions that are unusual for this user, with a plain-language reason | P0 |
| FR-8 | Unusual spending | Flag spending spikes: monthly spend in a category significantly above the user's normal level, with the size of the deviation and the transactions driving it. Weekly spikes are deferred: even a perfect detector would be wrong most of the time on weekly data (see FR-2) | P0 |
| FR-9 | Unusual spending | Let users adjust alert sensitivity | P1 |
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
| Goal forecasting | Forecast error (RMSE); accuracy of the on-track call | RMSE ≥ 15% lower than a naive forecast; on-track call better calibrated than naive |
| Coach grounding | Share of answers with every number correct | ≥ 95% |
| Coach safety | Share of unsafe or cross-user requests correctly refused | 100% |
| Coach quality | Rated helpfulness, clarity, empathy, personalization (1–5) | ≥ 4.0 average |

**Revisited after the first measurement:** the new-merchant target was 0.80 for v1. A cold model learns new brands only from text, and the best measured one scores about 0.71 on validation, so v1 gates at 0.66, a level a model that good passes reliably. 0.80 moves to v1.1, where corrections from users teach the model new merchants (FR-4 Unseen Merchant Categorization — Feature Design, §4). The v1 model scored 0.735 on new merchants.

**Unusual transactions, first measurement:** a per-user z-score on amount, the simple alternative, can reach 0.70 precision only on a handful of alerts, so recall is compared at the same number of alerts: 0.11 per user-month (FR-7 Unusual Transactions — Feature Design, §5). The v1 model scored precision 0.834 at its cutoff, and recall 0.720 at that rate against the alternative's 0.051 (FR-7 Unusual Transactions — Round Results).

**Caveat:** v1 is measured on synthetic data, which is easier than real data. Targets must be re-validated on real data before launch.

## Risks, assumptions and open questions

The biggest risk is that quality measured on synthetic data does not hold on real transactions; v1 treats real-data validation as a launch gate.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Synthetic data is easier than real data; metrics overstate quality | High | Build realistic noise into the data; validate on real data before launch |
| Coach states a wrong number or gives unsafe advice | High | Grounding and safety are release-blocking metrics |
| Cross-user data exposure | High | Data separation is a release-blocking requirement, tested adversarially |
| Alert fatigue from false alarms | Medium | Precision target; user-adjustable sensitivity in v1.1 |
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
- [ ] What default alert sensitivity do users prefer?

## Releases

Each release widens what the coach can do or who it serves; real user data arrives in v2.

| Release | Scope |
| --- | --- |
| v1 | All P0 requirements on synthetic data |
| v1.1 | P1 requirements: category correction, low-confidence review, alert sensitivity, answer sources; new-merchant categorization at the 0.80 target |
| v2 | Real bank data, real sign-in, validated quality on real data, clear handling of short histories |
| v3 | Proactive nudges, mobile notifications, deeper goal planning, weekly spending-spike alerts if they can be made reliable |
