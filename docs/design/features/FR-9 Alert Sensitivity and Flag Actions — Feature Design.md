# FR-9 Alert Sensitivity and Flag Actions — Feature Design

Oct 4, 2026 · @Sidd · Status: **Draft for review**; decisions 1–6 confirmed by the owner (Oct 4, 2026), questions 7–10 open · Branch: `feature/fr9`

## Summary

This feature lets users choose how often "Worth a look" points things out, and act on what it shows them.

- **Requirement:** FR-9 (P1), *"Let users adjust alert sensitivity."* The PRD names it as the mitigation for alert fatigue, and FR-7 §8 and FR-8 §8 deferred their flag actions to it. Like FR-5 and FR-6, it's a P1 built for the Oct 6 demo (owner, Oct 4, 2026).
- **Starting point:**
  - Both alert models are promoted and served: unusual charges (FR-7, an isolation forest, precision 0.834 on test users) and spending spikes (FR-8, a negative-binomial count model, precision 0.705).
  - Each turns one score into a flag at one cutoff, tuned to precision 0.80 on train users. The cutoff is the single knob FR-7 §5 and FR-8 §3 left for this feature.
  - Flag ids are already stable for actions to key on. Actions are already scoped: per user, in the FR-5/FR-6 feedback store, never touching the model or another user.
- **What the user gets** (mockup 1f):
  - **"How often should we point things out?"** with three settings: **Less often · Balanced · More often**. One control covers both unusual charges and spending spikes.
  - **On each unusual charge:** "I recognize this" (hide it and later ones like it) and "Not me — what now?" (guidance only).
  - **On each spending spike:** "Expected, all good" (hide that month's spike).
  - Undo for every action, as for corrections.
- **Presets are alert rates** (owner, Oct 4, 2026): Less = half as many alerts as Balanced, More = twice as many, Balanced = the promoted cutoff unchanged.
- **Feasibility, measured** ([evidence](#feasibility)):
  - **Less often** roughly halves alerts at precision 0.94–0.96, against 0.80 at Balanced. It finds less: recall 0.46 against 0.76 for unusual charges, and 0.32 against 0.52 for spikes.
  - **More often** doubles alerts at precision about 0.5: half of what it adds is ordinary. Recall rises 8–9 points.
  - **In the demo,** "More often" adds nothing to the three demo accounts' 60-day window. A fourth account, `u_te_fb_0019`, shows the effect (decision 6).
- **The PRD's precision ≥ 0.70 applies to Balanced,** the default and the setting everyone gets. More often is the user's choice to see more borderline alerts, and the page says so (decision 5).

## Context

### What exists

| | Unusual charges (FR-7) | Spending spikes (FR-8) |
| --- | --- | --- |
| Promoted model | `e0b67433-8b9632e6-2e033606`, isolation forest | `8c428c54-d85b4650-64917ea6`, `count_negbin` |
| Cutoff | Tuned to precision 0.80 on train users (`Thresholded`) | The same (`SpikeThresholded`) |
| Test precision | 0.834 (0.805–0.863) | 0.705 (0.634–0.770), a thin margin |
| Alerts at Balanced | 0.13 per user-month, about 1.5 a year | 0.035 per user-month, about 1 every 2.4 years |
| How serving gets flags | The nightly job writes flagged rows only to a flag file (`data/flags.py`) | Scored on request from the session's ledger with `spikes.json` (about 75 ms a user) |
| Flag id | Model version + transaction id | Model version + user, category, `period_start` |
| Fixed rules at any cutoff | An exact repeat scores +inf, so a duplicate is always flagged | Spend ≥ 1.3× usual and ≥ 2 purchases in a usual month, enforced by the contract |

### What the mockup asks for (screen 1f)

"Worth a look" opens with *"How often should we point things out?"* and a three-way switch: Less often · Balanced · More often. The spike card has "Ask the coach about this" and "Expected, all good". Single charges have "I recognize this" and "Not me — what now?" (Web App UI, gap 6).

### What earlier designs settled that FR-9 reuses

- **FR-7 §5:** "The tuning target is a parameter of the wrapper and is recorded at promotion. FR-9 later lets users move their own cutoff around it." Promotion also records the flag rate, so a rate cutoff can be placed without labels (the v2 path).
- **FR-7 §8:** actions are stored per user, keyed by flag id, in the FR-5/FR-6 feedback store. "I recognize this" suppresses later flags with the same reason at the same merchant, for that user only. "Not me" shows guidance only; moving money or blocking cards is out of scope.
- **FR-8 §8:** "Expected, all good" suppresses that category's flag for that month only.
- **FR-5/FR-6 §3:** feedback is events, and state is replayed from them. Undo marks an event; nothing is deleted. In the demo, the subject is the browser session, so visitors who share an account never see each other's feedback, and the store resets on redeploy.
- **FR-10:** changes made by the coach or an outside assistant are previewed until the user agrees (`confirm`); the web page applies on submit.

## Scope

| | In FR-9 | Not in FR-9 |
| --- | --- | --- |
| Sensitivity | Three presets, one control for both halves; per user (per session in the demo) | A continuous slider; per-category or per-kind settings |
| Actions | "I recognize this", "Not me — what now?", "Expected, all good", and undo | Blocking cards, disputes, contacting a bank; anything that moves money |
| Models | Cutoffs placed around the promoted ones | Retraining from actions; new models; changes to either model's Balanced cutoff |
| Window | "Worth a look" keeps its fixed 60 days | "Since you last looked" (v1.1 or later) |

## Feasibility

**Evidence:** `scripts/fr9_presets_feasibility.py` on the default dataset (content hash `b4d43bf4`, the data FR-7 and FR-8 used), with the promoted models. One command reproduces every number below:

```
uv run python scripts/fr9_presets_feasibility.py data/synthetic/default.sqlite \
    configs/web/demo_accounts.yaml u_te_fb_0000,u_te_fb_0019
```

### Precision and recall per preset

Train users only (240), through the label contract. FR-7's merchant profiles and FR-8's season profiles come from train users only, and FR-8 uses true categories, as validation does. Preset cutoffs flag 0.5× and 2× as many rows as the promoted cutoff on the same users. **No test user's labels are read.**

| Half | Preset | Flags | Flags per user-month | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Unusual charges | Less often | 532 | 0.067 | 0.942 | 0.458 |
| Unusual charges | **Balanced** | 1,065 | 0.128 | **0.801** | 0.759 |
| Unusual charges | More often | 2,130 | 0.223 | 0.510 | 0.843 |
| Spending spikes | Less often | 138 | 0.0174 | 0.964 | 0.321 |
| Spending spikes | **Balanced** | 277 | 0.0350 | **0.801** | 0.518 |
| Spending spikes | More often | 554 | 0.0699 | 0.487 | 0.611 |

- **Optimistic for the cutoffs:** both promoted cutoffs were placed on these users, which is why Balanced reproduces the 0.80 tuning target exactly. Milestone 1 measures each preset out of fold (§6). On test users, Balanced was 0.834 for unusual charges and 0.705 for spikes.
- **More often trades a lot of precision for a little recall.** Doubling the alerts adds 8–9 points of recall, so about half of what it shows is ordinary. That's the honest description, and the page uses it (§4).
- **Unusual charges at More often are 1.74× Balanced in flags per post-warm-up user-month, not 2×,** although the flag count doubles exactly. About 300 of the 1,065 extra flags (from FR-7's 7,930 post-warm-up user-months) fall in users' first 90 days, which the rate leaves out (FR-2's warm-up, when a user's history is short). For real users, that means More often is noisier in their first months. Milestone 1 reports where the extra flags fall.
- **Every preset includes every duplicate.** Exact repeats score +inf, so no cutoff drops them; rates are matched over all flags, duplicates included.
- **Less often keeps the clearest alerts:** about 19 in 20 are planted. It finds a third of planted spikes rather than half.

### What the demo shows

Serving's way: every user is scored against the whole pool, spikes on the promoted categorizer's predictions. Preset cutoffs are matched to the pool's flag count, and each account's "Worth a look" covers Aug 2 – Sep 30, 2026. No labels are read.

| Preset | Unusual charges (all users) | Users with one in the window | Spikes (all users) | Users with one in the window |
| --- | --- | --- | --- | --- |
| Less often | 801 | 35 | 200 | 11 |
| Balanced | 1,602 | 72 | 400 | 28 |
| More often | 3,204 | 116 | 800 | 54 |

| Demo account | Less often | Balanced | More often |
| --- | --- | --- | --- |
| Maya (`u_te_yp_0030`) | nothing | Groceries spike, Sep | same as Balanced |
| Ada (`u_te_fb_0003`) | nothing | Dining spike, Sep | same as Balanced |
| Jordan (`u_te_fl_0010`) | 1 duplicate | 1 duplicate + Transportation spike, Aug | same as Balanced |
| **`u_te_fb_0019`** (proposed fourth) | nothing | nothing | 1 charge + 1 spike |

- **The three accounts show "Less often", and none of them shows "More often".** Their planted spikes are not among the strongest half, so Less often hides them. Nothing borderline falls in their window at 2×. No test user both has a spike at Balanced and gains anything at 2×, so one account can't show all three settings.
- **`u_te_fb_0019`** (a family budgeter) has nothing at Balanced. At More often it shows:
  - *"First charge here, and larger than 86% of your earlier charges."* Crate & Barrel, $147.15, Aug 20.
  - *"You spent $2,338 on Groceries in September 2026, $716 more than your average month over the past year ($1,622). That came from 20 purchases, against about 14 in an average month."*

  Neither is planted. They're the borderline, ordinary alerts that More often is described as adding.
- **`u_te_fb_0000` was the first choice** and behaves the same way. It was dropped because its added charge reads "About 5× your usual charge here ($31.77, from 1 earlier charge)": one earlier charge is a thin basis to show.

## Goals and non-goals

**Goals**

1. Three presets per alert model, at 0.5×, 1× and 2× the promoted model's alert rate, with Balanced exactly the promoted cutoff.
2. One setting per user (per session in the demo) that "Worth a look", the Overview, the coach and outside assistants all honor.
3. The three flag actions and undo, stored as events in the feedback store, scoped like corrections.
4. Precision, recall and alerts per user-month for each preset, measured out of fold on validation and published.
5. The demo shows every preset and every action.

**Non-goals**

- Changing either model, its Balanced cutoff or its gates.
- Learning from actions: no retraining, recalibration or cross-user effect. Actions are signal for v2's real-data calibration (FR-7 §5), stored so it can be read later.
- Scoring test users again. Presets aren't a model choice, and test users have been scored twice for FR-7 and once for FR-8.
- Per-category sensitivity, notification channels or "since you last looked".

## What it will look like

### Flows

1. **Changing how often.** On "Worth a look", the user picks More often. Both cards reload with the extra alerts, and a note under the switch reads *"More often: you'll see more, and more of them will turn out to be ordinary."* The setting sticks for the session. The Overview's "Worth a look" count follows it.
2. **I recognize this.** On an unusual charge, the user clicks "I recognize this". It disappears with an Undo toast, and later charges flagged for the same reason at the same merchant don't appear.
3. **Not me — what now?** The flag stays, marked "You said this wasn't you", with guidance: contact the card issuer or bank using the number on the card, check for other charges at the merchant, and change passwords if the charge was online. Nothing else happens; the app can't block cards or dispute charges.
4. **Expected, all good.** On a spike, the user clicks it. That month's spike in that category disappears, with Undo.
5. **The coach.** "Why haven't you flagged anything?" → the coach knows the setting and how many alerts the user hid, and says so rather than "nothing is unusual". "Show me more alerts" → the coach previews the change and applies it once the user agrees (open question 7).

### Tools (tool server)

| Tool | Change |
| --- | --- |
| `detect_anomalies` | Uses the session's sensitivity. Adds `sensitivity` (`less`, `balanced`, `more`) and `hidden` (counts of flags the user acted on, by action) to its output. Each flag gains `flag_id`, and `your_action` when the user said "Not me". |
| `get_alert_settings` (new) | The session's sensitivity and what each setting means, in plain words. |
| `set_alert_sensitivity` (new) | Set `level`. The page applies on submit; the coach and outside assistants preview until `confirm: true`. |
| `act_on_flag` (new) | `flag_id`, `action` (`recognize`, `not_me`, `expected`). The same preview rule applies. |
| `undo_flag_action` (new) | Undo one action by id. |

A token without a feedback subject (read-only, FR-19) sees Balanced, can't act on flags, and gets no new write tools.

## Design

### 1. Presets as rate cutoffs

- **Each preset is a cutoff on the model's own score.** Less often is the score that flags half as many rows as the promoted cutoff over the reference pool, More often the score that flags twice as many. Balanced is the promoted cutoff itself, unchanged.
- **Placed once, without labels, then fixed.** A command, `sfc-model presets --task <task> --data <pool>`, scores the pool with the promoted model, places the two cutoffs, and writes `presets.json` next to the model's manifest (`artifacts/<task>/<version>/presets.json`). The file holds the cutoffs, the multipliers, the pool's data hash and the flag counts they produced. It's committed like the manifest, so serving reads fixed numbers and the build stays reproducible. This follows FR-8's rule that a rate cutoff is fixed when it's fitted, so one user's months are judged against the same line as everyone's.
- **The reference pool is every user,** as serving's merchant and season profiles are (FR-7 §2, FR-8 §2). For spikes, it uses the promoted categorizer's predicted categories, the same basis as the season table.
- **The fixed rules hold at every preset.** Duplicates are flagged at every setting (score +inf). Spike flags always need spend ≥ 1.3× usual and ≥ 2 purchases in a usual month; the contract checks every flag, whatever its cutoff.
- **Promotion stays the only route to callers.** A new promoted model needs its own `presets.json` before serving uses presets; without one, serving offers Balanced only and hides the switch. A test enforces that a presets file matches its model version.

### 2. Serving

**Unusual charges (flag files).**

- The nightly job writes every row at or above the **More often** cutoff, with its score. The file's meta records the three cutoffs.
- Serving keeps flags at or above the session's preset cutoff. The flag id doesn't depend on the preset, so an action on a flag holds at every setting.
- A flag file without preset cutoffs in its meta (written before FR-9) serves Balanced only.
- Storage grows about 2×: 3,204 rows instead of 1,602 for the default data.

**Spending spikes (on request).**

- `spikes.json` gains the three cutoffs from `presets.json`.
- `SpikeState.score` takes the preset and flags with that cutoff; the product rules and the contract checks are unchanged.
- The simple-rule fallback (decision 12 on #58) has its own rate cutoff and gets presets the same way, at 0.5× and 2× its rate.

**`detect_anomalies`** reads the session's preset (§3), filters flags by it, removes flags the user hid (§3), and reports what it hid. The coach's guidance in the tool description: when nothing is shown but `hidden` isn't empty, or the setting is Less often, say so; never say that nothing was unusual.

### 3. Settings and actions in the feedback store

Two new event tables in the FR-5/FR-6 store, in the same SQLite file and scoped the same way: `subject` is who acted (the browser session in the demo), `user_id` is whose ledger it's about. Neither ever comes from a tool argument.

```
alert_settings  setting_id, seq, subject, user_id, level ('less'|'balanced'|'more'),
                source ('page'|'coach'), created_at
flag_actions    action_id, seq, subject, user_id, flag_id, kind ('charge'|'spike'),
                action ('recognize'|'not_me'|'expected'), merchant_key, reason_code,
                category, period_start, transaction_ts, source, created_at, undone_at
```

- **The setting is the latest event;** with none, Balanced. Changing it is its own undo, so settings have no `undone_at`.
- **What each action hides,** replayed from the events that aren't undone:
  - `recognize` (unusual charges): this flag, and every later flag for this user with the same `reason_code` at the same `merchant_key`, by `transaction_ts` (FR-7 §8). Earlier flags stay.
  - `expected` (spikes): this category and month only (FR-8 §8).
  - `not_me`: hides nothing. The flag is shown with the marker and the guidance.
- **The action stores what it suppresses by,** `merchant_key` and `reason_code` or `category` and `period_start`, so replay doesn't need the flag file. A new model version gives flags new ids, but its flags are still matched by merchant and reason, so "I recognize this" keeps working across promotions.
- **Isolation:** actions never reach the model, another user, or another session on the same account. In the demo the store resets on redeploy, as corrections do.

### 4. Web pages

- **"Worth a look" (1f):** the switch at the top, as in the mockup. It posts and redirects, so it works without JavaScript; htmx swaps the two cards in place. Under it, one line per setting:
  - Less often: *"Fewer alerts: only the clearest ones. You may miss some."*
  - Balanced: *"Our standard setting."*
  - More often: *"You'll see more, and more of them will turn out to be ordinary."*
- **Unusual charges:** "I recognize this" and "Not me — what now?" on each row. "Not me" expands into the guidance in place.
- **Spending spikes:** "Expected, all good" on each spike, beside "Ask the coach about this".
- **Undo toast** after every action, as for corrections.
- **A line at the foot of each card** when something is hidden: "2 alerts you recognized aren't shown." with a link that shows them again for the page view only.
- **Overview:** the "Worth a look" count follows the setting and leaves out hidden flags.
- **Transactions (1e):** a flagged row's marker follows the setting and hidden flags in the same way.

### 5. The coach

- The system prompt gains one rule: when `detect_anomalies` shows nothing, check `sensitivity` and `hidden` before saying nothing was unusual, and mention them.
- The coach can change the setting and act on flags through the new tools, previewed until the user agrees (open question 7).
- "Not me" guidance is fixed text from the tool result, not generated, so the coach never improvises advice about fraud.

### 6. Measurement

- **Out of fold, on validation, for each promoted config.** The existing tasks already rank at a common rate in held-out folds (`val_recall_at_rate`). Milestone 1 adds the same measurement at 0.5× and 2× of each task's common rate (FR-7: 0.055 and 0.22; FR-8: 0.0175 and 0.07): precision, recall and flags per user-month, with user-bootstrap intervals. Only the promoted configs are rerun, on the round's splits.
- **Reported, not gated.** Balanced keeps its gates. The presets have no gate, so there's nothing to choose and no test scoring.
- **Results** go in a short report, FR-9 Alert Sensitivity — Results, beside this script's in-sample estimate. The PRD's success-metrics note gains a line.
- **Demo check:** the build prints each demo account's alert count per preset, and `/healthz` reports that presets are loaded for both halves.

### 7. The demo

- **A fourth demo account,** `u_te_fb_0019` (a family budgeter; name and email to be chosen), in `configs/web/demo_accounts.yaml`. Its comment changes, since it has no spike at Balanced.
- **The story:**
  - On Maya, Ada or Jordan, Less often hides the spike and More often adds nothing: alerts are rare by design.
  - On the fourth account, Balanced is empty and More often shows a borderline charge and a borderline month. Act on both.
  - Ask the coach why nothing was flagged.
- The 60-day window stays (decision 6; FR-7 open question 4).

## Metrics and why

1. **Precision per preset:** what a user trades for more alerts. It's the quantity the More often copy describes.
2. **Recall per preset:** what Less often costs.
3. **Flags per user-month per preset:** the alert burden, the user's own terms for "how often". It checks the 0.5× and 2× targets out of fold.
4. **Balanced's gates are unchanged:** the PRD's precision ≥ 0.70 and recall above the baseline apply to the default.

## Options considered

### A. What a preset fixes

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) An alert rate: 0.5×, 1×, 2× (decided)** | Answers "how often", visibly different settings; needs no labels, so it carries to real data (FR-7 §5) | Precision at More often is about 0.5, below the PRD's bar |
| (b) A precision target per preset, e.g. 0.90 / 0.80 / 0.70 | Every setting keeps a stated precision | More often at 0.70 adds about a quarter more alerts (FR-7 feasibility: 883 to 1,091 flags), and 2× adds nothing to the demo accounts' window, so neither would show; needs labels to place |
| (c) A continuous slider | Fine control | Hard to explain; every position needs a measurement |

### B. A floor for More often

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) None; the copy says more will be ordinary (decided)** | One control moves both halves; spike flags still obey the product rules, so a "false" spike is still a visibly high month | About half of More often's alerts are ordinary |
| (b) Keep spikes at Balanced on More often | Spike precision stays at its tested level | "More often" silently does half of what it says |
| (c) Cap More often where precision falls to 0.70 | PRD bar holds everywhere | For spikes that's Balanced itself (0.705 on test): no room |

### C. Where preset cutoffs come from

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Placed once on the pool, committed beside the model (recommended)** | Fixed, reproducible, versioned with the model; one user is judged against the same line as everyone | A step after each promotion |
| (b) Placed by each nightly run | Follows the pool as it grows | Cutoffs drift from night to night; spikes are scored on request, so they'd need a nightly file anyway |
| (c) Per user, top 2× of their own scores | Every user sees a change | A user with nothing unusual gets ordinary charges flagged; FR-8 rejected re-ranking within one user for this reason |

### D. Who can change the setting and act on flags

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) The page directly; the coach and outside assistants with a preview and `confirm` (recommended)** | Matches goals and corrections; "show me more alerts" works in chat | Two more write tools to test |
| (b) The page only | Smallest surface | The coach can explain the setting but not change it |

### E. The demo window

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Keep 60 days; add a fourth account (decided)** | Keeps the decision on #30; honest about how rare alerts are | The demo switches accounts to show More often |
| (b) Show earlier alerts too | Every account changes with the setting | Reopens the window decision; a longer page |

## Testing

- **Unit:**
  - presets: the cutoffs flag 0.5× and 2× as many rows on the pool they were placed on; Balanced equals the promoted cutoff; a presets file for another model version is refused;
  - every preset flags every duplicate; every spike flag at every preset passes the product rules (the contract);
  - flag file: rows down to the More often cutoff are stored; a file without preset meta serves Balanced only;
  - replay: `recognize` hides later flags with the same reason at the same merchant and not earlier ones; `expected` hides one month; `not_me` hides nothing; undo restores.
- **Isolation:** one session's setting and actions never change another session's or another user's view of the same account (demo), or anything for a token without a subject.
- **Tools:** `detect_anomalies` reports `sensitivity` and `hidden`; the coach's writes preview until `confirm`.
- **Web:** the switch posts and redirects without JavaScript; the Overview count, "Worth a look" and Transactions agree at every setting.
- **Demo bundle:** the fourth account shows nothing at Balanced and a charge and a spike at More often; `/healthz` reports presets for both halves.

## Milestones

One PR per milestone, stacked.

1. **Presets:** `sfc-model presets` and committed `presets.json` for both promoted models (and the simple rule); flag files down to More often; spikes by preset; `detect_anomalies` with a `sensitivity` argument internally (Balanced by default); the out-of-fold measurement and FR-9 Alert Sensitivity — Results.
2. **Settings and actions:** the two event tables, replay and undo; the new tools and `detect_anomalies`' session setting and `hidden`; the switch, buttons, guidance, toasts and footers on "Worth a look"; the Overview and Transactions; the coach's prompt rule.
3. **Demo and docs:** the fourth account and `/healthz`; the PRD (FR-9 in the v1 demo; More often's precision), the Technical Design (tools, presets) and the Web App UI (1f, gap 6).

## Decisions and open questions

**Decisions** (owner, Oct 4, 2026)

1. [x] **FR-9 is in the Oct 6 demo.**
2. [x] **One control for both halves,** as in the mockup.
3. [x] **The flag actions ship with it:** "I recognize this", "Not me — what now?", "Expected, all good".
4. [x] **Presets are alert rates:** Less = 0.5×, Balanced = the promoted cutoff unchanged, More = 2×. Precision is measured on validation, out of fold; no new test scoring (option A-a).
5. [x] **No precision floor for More often, on either half.** The product rules still hold, the 0.70 gate applies to Balanced, and the page says more alerts will be ordinary (option B-a).
6. [x] **A fourth demo account shows More often; the 60-day window stays** (option E-a). The first choice was `u_te_fb_0000`, pending a look at its flags. This design proposes `u_te_fb_0019` instead ([What the demo shows](#what-the-demo-shows)).

**Open questions, for review**

7. [ ] **Can the coach and outside assistants change the setting and act on flags,** with a preview until the user agrees? Recommended: yes (option D-a), as for goals and corrections.
8. [ ] **Preset cutoffs placed once on the pool and committed beside the model** (option C-a)? Recommended.
9. [ ] **"I recognize this" hides later flags only,** with the same reason at the same merchant, as FR-7 §8 wrote? Or every flag with that reason at that merchant, earlier ones included?
10. [ ] **The fourth account's name and email.**
