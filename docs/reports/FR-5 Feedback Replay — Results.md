# FR-5 Feedback Replay — Results

Oct 3, 2026 · the simulated feedback replay of the FR-5 and FR-6 design (§7), run once with N = 3 and the default settings; nothing was chosen from its results (owner, Oct 3, 2026). The full result is `fr5-replay.json`, which the demo's "How it learns" page reads.

```sh
uv run sfc-experiment replay --data data/synthetic/default.sqlite --out docs/reports/fr5-replay.json
```

Data hash `44781bc4e4a5`; starting model `20eea4fb-44781bc4-c0274576`, the promoted FR-4 model, with its review policy (familiar < 0.95, unfamiliar < 0.65); retraining with `21_small_unweighted.ship` (clean labels). About 37 minutes on a laptop CPU. Reproduced exactly by later runs and by the reviewer on #44; the review fixes (isolation and burden measures, the Income rule for overrides, the Brier gate on the same rows) changed no decision or headline number.

## Setup

- **People:** the 120 test users, split by a hash of their id into 69 *feedback* users, who review and correct (3 of them adversarial, correcting at random), and 51 *evaluation* users, who never give feedback. Gains are measured on the evaluation users, against their own view (their "user's category") unless marked "truth".
- **Behavior:** each month a feedback user opens the review queue with probability 0.6 and resolves its top 5 items by spend; confirms the suggestion if it matches their view, otherwise corrects (3% slips); accepts suggestions without checking 15% of the time; notices an unflagged string they see differently 5% of the time a month.
- **Learning:** the agreement rule (§4: N = 3, two thirds, at least one correction) every quarter from month 6; retraining (§5) on the original rows relabelled at agreed strings plus contributors' rows before the cutoff; gates against the incumbent on the evaluation users' quarter before the cutoff (never trained on).

## Results

| Measure (evaluation users unless noted) | First quarter | Last quarter |
| --- | --- | --- |
| Accuracy against their own view | 0.904 | 0.938 |
| Macro F1, own view | 0.813 | 0.858 |
| **New merchants: macro F1, true categories** | **0.751** | **0.872** |
| New merchants: macro F1, own view | 0.612 | 0.800 |
| All merchants: macro F1, true categories | 0.946 | 0.937 |
| Feedback users: accuracy of what they see (with their corrections) | 0.908 | 0.942 |
| Feedback users: the model alone, without their corrections | 0.881 | 0.923 |

- **Retrainings:** 10, of which 3 were promoted (2024-06, 2024-09, 2024-12) and 7 rejected. The first candidate (2024-03, 29 labels) relabelled 48,804 training rows and lost accuracy at the labelled merchants (0.697 → 0.522) and against the truth (0.943 → 0.846), and the gates rejected it. **Most of that came from the known Uber collision** (FR-5 design, Feasibility): the 30% rideshare preference won the first votes at `uber → Travel`, and `uber` is the string Uber Eats (Dining) and Uber rides share, so 42,116 of the relabelled rows were at `uber`, including about 13,000 Uber Eats rows turned to Travel. The gates did their job, mostly on a known data defect rather than on agreement alone. Every candidate after 2024-12 scored 0.007–0.026 below the incumbent on macro F1 against the evaluation users' view, and was rejected.
- **Global labels:** 65 from 1,825 votes (615 corrections). 50 fix real model errors (One Medical, Chick-fil-A, Subway…); 11 are the 70% streaming preference (Netflix, Max, Disney+ → Entertainment), which the design means to be learned; 4 are minority preferences or an ambiguous merchant that met N = 3: Walgreens (two strings) → Shopping from the 30% pharmacy remap, Costco → Shopping (2 of 3) from the 30% warehouse remap, and one Target store → Groceries.
- **Burden (§7):** about 1.3 items in a feedback user's review queue a month, of which they resolved 0.7 (57%).
- **Isolation (§7):** evaluation users' categories changed only in the months right after the three promotions (2024-07, 2024-10, 2025-01), and they cast no votes; each feedback user's view applies only their own overrides.
- **A retrained model, not deployed (owner, Oct 3, 2026):** `sfc-experiment retrain-from-replay --month 2024-12` rebuilt the last promoted candidate from the recorded inputs (48 labels, cutoff 2025-01-01; 584,551 training rows, 6,877 relabelled, 5,180 from contributors; the same review policy) and logged it as a tracked run of kind `feedback-retrain` (`5f4906cd` in the local store). It isn't a finalist, so it can't be promoted without `finalize` and `promote`; `PROMOTED` stays `20eea4fb`.

**Per remap** (evaluation users in the last quarter: share of their transactions at that subtype whose category matches their view, with a 95% user-level bootstrap interval):

| Remap | Holders (of 120) | Labels learned | Match | Interval |
| --- | --- | --- | --- | --- |
| books: Shopping → Entertainment (50%) | 13 | none | 0.67 | 0.00–1.00 |
| gym: Health & Fitness → Subscriptions (30%) | 22 | none | 0.65 | 0.41–0.88 |
| pharmacy: Health & Fitness → Shopping (30%) | 38 | 2 (Shopping) | 0.76 | 0.61–0.87 |
| rideshare: Transportation → Travel (30%) | 25 | none (6 made and revoked) | 0.70 | 0.55–0.82 |
| streaming: Subscriptions → Entertainment (70%) | 90 | 11 (Entertainment) | 0.59 | 0.50–0.68 |
| warehouse: Groceries → Shopping (30%) | 20 | 1 (Shopping) | 0.31 | 0.21–0.44 |

## What it says

1. **The loop works.** Corrections from some people improved the model for people who gave none, most on merchants it had never seen (new-merchant macro F1 against the truth 0.75 → 0.87, past the PRD's 0.80 goal for v1.1, "through feedback"), with the gates promoting improvements and catching a bad batch.
2. **N = 3 lets minority preferences through.** Three labels came from 30% preferences, as §4 predicted (a 30% preference wins the first three votes about one time in five), and they are why later candidates fell below the incumbent for the 70% who see those merchants the default way. Re-evaluation revoked many (rideshare: 6 made, 6 revoked) but not all. This is the evidence for open question 1 (N and the majority), to be tuned on a separate half of the test users (owner, Oct 3, 2026).
3. **Preferences cost a little truth.** Learning the 70% streaming preference makes the model "wrong" against the true category on those strings by construction; overall macro F1 against the truth moved 0.946 → 0.937 while agreement with what people see rose.

## Robustness

The design asks for global gain with 5% and 20% adversarial users (users who correct at random).

- **5%** (3 of 69 feedback users): the results above.
- **About 9%** (6 of 69; a run configured at 20% before the replay picked an exact count): gain held (accuracy against their own view 0.904 → 0.941, new-merchant macro F1 against the truth 0.751 → 0.868; 4 of 10 promoted), and **no random correction became a global label**: the 13 labels that differ from the truth are all real shared preferences (streaming, Costco, Rite Aid). All-merchant macro F1 against the truth ended at 0.912, as more of the streaming preference was learned.
- **20%** (14 of 69): see the follow-up below once it's in.

## Caveats

- Synthetic people with simple, stationary behavior; one seed; one run.
- The evaluation users are FR-4's test users, scored once at its `finalize`; the replay answers a different question, and nothing was tuned on it (owner, Oct 3, 2026).
- Gate tolerances (familiar accuracy −0.005, Brier +0.005) are provisional, like N, the majority and the cadence.
- **The gates and the headline use the same 51 evaluation users** (earlier quarters for the gates, the last quarter for the headline), and their preferences don't change, so the gain against their own view is somewhat optimistic: candidates were rejected partly because these users don't share the Walgreens and Costco preferences. A gate half and a report half would fix it, alongside tuning N on a separate half (owner, Oct 3, 2026).
- The Uber / Uber Eats normalizer collision is in the data the replay ran on (fix deferred by the owner to the first retrained model meant for promotion).
- Two of the three demo accounts (Maya, Ada) are feedback users here.
- **No replay model is promoted to the demo (owner, Oct 3, 2026).** Retraining stays inside the replay; the demo shows the loop from these results on its "How it learns" page and keeps serving the FR-4 categorizer `20eea4fb`. The retrained December 2024 model is kept as a tracked run only.
