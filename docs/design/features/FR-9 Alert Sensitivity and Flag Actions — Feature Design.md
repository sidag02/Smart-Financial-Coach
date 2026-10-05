# FR-9 Alert Sensitivity and Flag Actions — Feature Design

Oct 4, 2026 · @Sidd · Status: **Accepted** (owner, Oct 4, 2026, on #65); decisions 1–17 confirmed, none open · **Implemented** (#66, #67, #68; [Status](#status-oct-5-2026)), in review · Branch: `feature/fr9`

## Summary

This feature lets users choose how often "Worth a look" points things out, and act on what it shows them.

- **Requirement:** FR-9 (P1), *"Let users adjust alert sensitivity."* The PRD names it as the mitigation for alert fatigue, and FR-7 §8 and FR-8 §8 deferred their flag actions to it. Like FR-5 and FR-6, it's a P1 built for the Oct 6 demo (owner, Oct 4, 2026).
- **Starting point:**
  - Both alert models are promoted and served: unusual charges (FR-7, an isolation forest, precision 0.834 on test users) and spending spikes (FR-8, a negative-binomial count model, precision 0.705).
  - Each turns one score into a flag at one cutoff, tuned to precision 0.80 on train users. The cutoff is the single knob FR-7 §5 and FR-8 §3 left for this feature.
  - Flag ids are already stable for actions to key on. Actions are already scoped: per user, in the FR-5/FR-6 feedback store, never touching the model or another user.
- **What the user gets** (mockup 1f):
  - **"How often should we point things out?"** with three settings: **Less often · Balanced · More often**. One control covers both unusual charges and spending spikes.
  - **On each unusual charge:** "I recognize this" (hide it and later ones like it; a duplicate hides only itself) and "Not me — what now?" (guidance only).
  - **On each spending spike:** "Expected, all good" (hide that month's spike).
  - Undo for every action, as for corrections. The coach and outside assistants can do the same, previewed until the user agrees.
- **Presets are alert rates after the warm-up** (decisions 4 and 11): Less = half as many alerts per post-warm-up user-month as Balanced, More = twice as many, Balanced = the promoted cutoff unchanged.
- **Feasibility, measured** ([evidence](#feasibility)):
  - **Less often** halves alerts at precision 0.95–0.96, against 0.80 at Balanced. It finds less: recall 0.44 against 0.76 for unusual charges, and 0.32 against 0.52 for spikes.
  - **More often** doubles alerts at precision 0.45–0.49: about half of what it shows is ordinary. Recall rises 9–10 points.
  - **More often is much noisier in a user's first 90 days** for unusual charges: 489 warm-up flags on train users against 47 at Balanced, about 0.69 per user-month there. They're outside the precision, since the label contract scores nothing in the warm-up. So during a user's first 90 days, serving uses the stricter of their setting and Balanced (decision 17).
  - **In the demo,** "More often" adds nothing to the three demo accounts' 60-day window, so a fourth account, Sam Patel (`u_te_fb_0023`), shows it (decisions 6, 10 and 16).
- **The PRD's precision ≥ 0.70 applies to Balanced,** the default and the setting everyone gets. More often is the user's choice to see more borderline alerts, and the page says so (decision 5).

## Context

### What exists

| | Unusual charges (FR-7) | Spending spikes (FR-8) |
| --- | --- | --- |
| Promoted model | `e0b67433-8b9632e6-2e033606`, isolation forest | `8c428c54-d85b4650-64917ea6`, `count_negbin` |
| Cutoff | Tuned to precision 0.80 on train users (`Thresholded`) | The same (`SpikeThresholded`) |
| Test precision | 0.834 (0.805–0.863) | 0.705 (0.634–0.770), a thin margin |
| Alerts at Balanced | 0.13 per post-warm-up user-month, about 1.5 a year | 0.035 per user-month, about 1 every 2.4 years |
| How serving gets flags | The nightly job writes flagged rows only to a flag file (`data/flags.py`) | Scored on request from the session's ledger with `spikes.json` (about 75 ms a user) |
| Flag id | Model version + transaction id | Model version + user, category, `period_start` |
| Fixed rules at any cutoff | An exact repeat scores +inf, so a duplicate is always flagged | Spend ≥ 1.3× usual and ≥ 2 purchases in a usual month, enforced by the contract |

### What the mockup asks for (screen 1f)

"Worth a look" opens with *"How often should we point things out?"* and a three-way switch: Less often · Balanced · More often. The spike card has "Ask the coach about this" and "Expected, all good". Single charges have "I recognize this" and "Not me — what now?" (Web App UI, gap 6).

### What earlier designs settled that FR-9 reuses

- **FR-7 §5:** "The tuning target is a parameter of the wrapper and is recorded at promotion. FR-9 later lets users move their own cutoff around it." Promotion also records the flag rate, so a rate cutoff can be placed without labels (the v2 path).
- **FR-7 §8:** actions are stored per user, keyed by flag id, in the FR-5/FR-6 feedback store. "I recognize this" suppresses later flags with the same reason at the same merchant, for that user only. "Not me" shows guidance only; moving money or blocking cards is out of scope.
- **FR-8 §8:** "Expected, all good" suppresses that category's flag for that month only. A rate cutoff is fixed when it's fitted, so one user's months are judged against the same line as everyone's.
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

**Evidence:** `scripts/fr9_presets_feasibility.py` on the default dataset (content hash `b4d43bf4`, the data FR-7 and FR-8 used), with the promoted models. One command prints every number in this section:

```
uv run python scripts/fr9_presets_feasibility.py data/synthetic/default.sqlite \
    configs/web/demo_accounts.yaml u_te_fb_0000,u_te_fb_0019,u_te_fb_0023
```

The script places presets the way §1's command will (decision 11):

- count the Balanced flags on post-warm-up rows;
- take the k highest post-warm-up scores, k = 0.5× or 2× that count, with `top_k`'s tie-break by id;
- the lowest of them is the preset's cutoff.

The warm-up is each user's first 90 days (charges) or first 3 months (periods) from their first transaction. On this data, that's exactly the label contract's warm-up, and the script checks it.

### Precision and recall per preset

Train users only (240), through the label contract. FR-7's merchant profiles and FR-8's season profiles come from train users only, and FR-8 uses true categories, as validation does. **No test user's labels are read.**

| Half | Preset | Flags after warm-up | Per post-warm-up user-month | Precision | Recall | Flags in warm-up (outside precision) |
| --- | --- | --- | --- | --- | --- | --- |
| Unusual charges | Less often | 509 | 0.064 | 0.945 | 0.441 | 1 |
| Unusual charges | **Balanced** | 1,018 | 0.128 | **0.801** | 0.759 | 47 |
| Unusual charges | More often | 2,036 | 0.257 | 0.450 | 0.859 | 489 |
| Spending spikes | Less often | 138 | 0.0174 | 0.964 | 0.321 | 0 |
| Spending spikes | **Balanced** | 277 | 0.0350 | **0.801** | 0.518 | 0 |
| Spending spikes | More often | 554 | 0.0699 | 0.487 | 0.611 | 0 |

- **Optimistic for the cutoffs:** both promoted cutoffs were placed on these users, which is why Balanced reproduces the 0.80 tuning target exactly. Milestone 1 measures each preset out of fold (§6). On test users, Balanced was 0.834 for unusual charges and 0.705 for spikes.
- **More often trades a lot of precision for a little recall.** Doubling the alerts adds 9–10 points of recall, so about half of what it shows is ordinary. That's the honest description, and the page uses it (§4).
- **Warm-up flags are reported apart, not hidden** (decision 11). Unusual charges at More often put 489 flags in users' first 90 days, about 0.69 per user-month there (240 users × about 3 months). That's 10× Balanced's warm-up rate (47) and 2.7× More often's own rate after the warm-up. Short histories make many charges look new or large. Their precision is unmeasured, because the label contract scores nothing in the warm-up. Spike periods need 3 earlier months to be scored at all, so spikes have no warm-up flags. Decision 17 keeps More often's warm-up flags away from new users.
- **Every preset includes every duplicate.** Exact repeats score +inf, so no cutoff drops them.
- **Less often keeps the clearest alerts:** about 19 in 20 are planted. It finds a third of planted spikes rather than half.

### What the demo shows

Serving's way: every user is scored against the whole pool, spikes on the promoted categorizer's predictions, and presets are placed on the pool. Each account's "Worth a look" covers Aug 2 – Sep 30, 2026. No labels are read.

| Preset | Unusual charges (all users, all time) | Users with one in the window | Spikes (all users, all time) | Users with one in the window |
| --- | --- | --- | --- | --- |
| Less often | 771 | 34 | 200 | 11 |
| Balanced | 1,602 | 72 | 400 | 28 |
| More often | 3,749 | 124 | 800 | 54 |

| Account | Less often | Balanced | More often |
| --- | --- | --- | --- |
| Maya (`u_te_yp_0030`) | nothing | Groceries spike, Sep | same as Balanced |
| Ada (`u_te_fb_0003`) | nothing | Dining spike, Sep | same as Balanced |
| Jordan (`u_te_fl_0010`) | 1 duplicate | 1 duplicate + Transportation spike, Aug | same as Balanced |
| `u_te_fb_0019` (not chosen) | nothing | nothing | 1 charge + 1 spike |
| **Sam Patel (`u_te_fb_0023`), the fourth account** | 1 duplicate + Groceries spike, Aug | same as Less often | adds 1 charge |

- **The three accounts show "Less often", and none of them shows "More often".** Their planted spikes are not among the strongest half, so Less often hides them, and nothing borderline falls in their window at 2×.
- **No account shows all three settings.** Of the 120 test users, 30 gain something in the window at More often. Only one of them, `u_te_fb_0023`, also has a spike at Balanced, and its alerts are strong enough that Less often doesn't change it.
- **`u_te_fb_0019`** (a family budgeter) has nothing at Balanced. At More often it shows two ordinary, unplanted alerts:
  - *"First charge here, and larger than 86% of your earlier charges."* Crate & Barrel, $147.15, Aug 20.
  - *"You spent $2,338 on Groceries in September 2026, $716 more than your average month over the past year ($1,622). That came from 20 purchases, against about 14 in an average month."*
- **`u_te_fb_0023`** (a family budgeter) has two planted alerts at every setting:
  - a duplicate, Peet's Coffee, $5.66, 9 minutes after an earlier charge on Sep 17;
  - *"You spent $6,963 on Groceries in August 2026, $5,579 more than your average month over the past year ($1,384). That came from 40 purchases, against about 9 in an average month."*

  More often adds one ordinary charge: *"About 11× your usual charge here ($38.44, from 42 earlier charges)."* The Gift Nook, $421.53, Sep 30. This one account shows More often and all three actions, including the duplicate exception (decision 9).
- **`u_te_fb_0000` was the first choice.** At More often it now shows two charges and a spike. One of the charges reads "About 5× your usual charge here ($31.77, from 1 earlier charge)": one earlier charge is a thin basis to show.

## Goals and non-goals

**Goals**

1. Three presets per alert model, at 0.5×, 1× and 2× the promoted model's alert rate per post-warm-up user-month, with Balanced exactly the promoted cutoff.
2. One setting per user (per session in the demo) that "Worth a look", the Overview, the coach and outside assistants all honor.
3. The three flag actions and undo, stored as events in the feedback store, scoped like corrections, and kept across model promotions.
4. Precision, recall and alerts per post-warm-up user-month for each preset, measured out of fold on validation and published. Warm-up flags are counted apart.
5. The demo shows every preset and every action.

**Non-goals**

- Changing either model, its Balanced cutoff or its gates.
- Learning from actions: no retraining, recalibration or cross-user effect. Actions are signal for v2's real-data calibration (FR-7 §5), stored so it can be read later.
- Scoring test users again. Presets aren't a model choice, and test users have been scored twice for FR-7 and once for FR-8.
- Per-category sensitivity, notification channels or "since you last looked".

## What it will look like

### Flows

1. **Changing how often.** On "Worth a look", the user picks More often. Both cards reload with the extra alerts, and a note under the switch reads *"More often: you'll see more, and more of them will turn out to be ordinary."* The setting sticks for the session. The Overview's "Worth a look" count follows it.
2. **I recognize this.** On an unusual charge, the user clicks "I recognize this". It disappears with an Undo toast, and later charges flagged for the same reason at the same merchant don't appear. On a possible duplicate, it hides only that charge: a later double charge at the same merchant still shows.
3. **Not me — what now?** The flag stays, marked "You said this wasn't you", with guidance:
   - contact the card issuer or bank, using the number on the card;
   - check for other charges at the merchant;
   - change passwords if the charge was online.

   Nothing else happens; the app can't block cards or dispute charges.
4. **Expected, all good.** On a spike, the user clicks it. That month's spike in that category disappears, with Undo.
5. **The coach.** "Why haven't you flagged anything?" → the coach knows the setting and how many alerts the user hid, and says so rather than "nothing is unusual". "Show me more alerts" → the coach previews the change and applies it once the user agrees (decision 7).

### Tools (tool server)

| Tool | Change |
| --- | --- |
| `detect_anomalies` | Uses the session's sensitivity. Adds `sensitivity` (`less`, `balanced`, `more`) and `hidden` to its output. `hidden` counts the flags in range the user hid, by `recognize` and `expected` only, so it never counts a flag that's on the screen (decision 15). Each flag gains `flag_id`, and `your_action: "not_me"` when the user said so. |
| `get_alert_settings` (new) | The session's sensitivity and what each setting means, in plain words. |
| `set_alert_sensitivity` (new) | Set `level`. The page applies on submit; the coach and outside assistants preview until `confirm: true` (decision 7). |
| `act_on_flag` (new) | `flag_id`, `action` (`recognize`, `not_me`, `expected`). The same preview rule applies. |
| `undo_flag_action` (new) | Undo one action by id. |

A token without a feedback subject (read-only, FR-19) sees Balanced, can't act on flags, and gets no new write tools.

## Design

### 1. Presets as rate cutoffs

- **Each preset is a cutoff on the model's own score,** placed by alert rate after the warm-up (decision 11):
  - count the promoted cutoff's flags on the pool's post-warm-up rows;
  - take the k highest post-warm-up scores, k = 0.5× or 2× that count, ties broken by id (`top_k`, as `rate_cutoff` does);
  - the lowest score among them is the cutoff.

  Flags are still taken at score ≥ cutoff, so a tie at the cutoff can add a few. Balanced is the promoted cutoff itself, unchanged.
- **The warm-up without labels** is each user's first 90 days from their first transaction (charges), or their first 3 months (spike periods): FR-2's warm-up lengths, counted per user, so the rule carries to real accounts that start on different days. On the synthetic data it equals the label contract's warm-up. Warm-up rows aren't in the rate. Serving flags them at **the stricter of the session's preset and Balanced** (decision 17): More often starts after a user's first 90 days, and Less often stays Less often throughout.
- **Placed once, without labels, then fixed** (decision 8). A command, `sfc-model presets --task <task> --data <pool>`, scores the pool with the promoted model, places the two cutoffs, and writes `presets.json` next to the model's manifest (`artifacts/<task>/<version>/presets.json`). The file holds:
  - the cutoffs and the multipliers;
  - the pool's data hash;
  - the flag counts after the warm-up and in it, per preset.

  It's committed like the manifest, so serving reads fixed numbers and the build stays reproducible. This follows FR-8's rule that a rate cutoff is fixed when it's fitted.
- **The reference pool is every user,** as serving's merchant and season profiles are (FR-7 §2, FR-8 §2). For spikes, it uses the promoted categorizer's predicted categories, the same basis as the season table.
- **The fixed rules hold at every preset.** Duplicates are flagged at every setting (score +inf). Spike flags always need spend ≥ 1.3× usual and ≥ 2 purchases in a usual month; the contract checks every flag, whatever its cutoff.
- **Promotion stays the only route to callers.** A new promoted model needs its own `presets.json` before serving uses presets. Without one, serving offers Balanced only and hides the switch. A test enforces that a presets file matches its model version.

### 2. Serving

**Unusual charges (flag files).**

- The nightly job writes every row at or above the **More often** cutoff, with its score. The file's meta records the three cutoffs.
- Serving keeps flags at or above the session's preset cutoff. The flag id doesn't depend on the preset, so an action on a flag holds at every setting.
- A flag file without preset cutoffs in its meta (written before FR-9) serves Balanced only.
- Storage grows about 2.3×: 3,749 rows instead of 1,602 for the default data.

**Spending spikes (on request).**

- `spikes.json` gains the three cutoffs from `presets.json`.
- `SpikeState.score` takes the preset and flags with that cutoff. The product rules and the contract checks are unchanged.
- **The simple-rule fallback** (decision 12 on #58) has no promoted artifact, so it has no `presets.json` (decision 13). `build_state` fits its Less and More cutoffs at build time, the way it fits its own: rate cutoffs at 0.5× and 2× its rate (0.0175 and 0.07 flags per user-month) on the pool, with no labels. `spikes.json` stores all three.

**`detect_anomalies`** reads the session's preset (§3), filters flags by it, removes the flags the user hid (§3), and reports `hidden`. The coach's guidance in the tool description: when nothing is shown but `hidden` isn't empty, or the setting is Less often, say so; never say that nothing was unusual.

### 3. Settings and actions in the feedback store

Two new event tables in the FR-5/FR-6 store, in the same SQLite file and scoped the same way: `subject` is who acted (the browser session in the demo), and `user_id` is whose ledger it's about. Neither ever comes from a tool argument.

```
alert_settings  setting_id, seq, subject, user_id, level ('less'|'balanced'|'more'),
                source ('page'|'coach'), created_at
flag_actions    action_id, seq, subject, user_id, flag_id, kind ('charge'|'spike'),
                action ('recognize'|'not_me'|'expected'), transaction_id, transaction_ts,
                merchant_key, reason_code, category, period_start, source, created_at,
                undone_at
```

- **The setting is the latest event;** with none, Balanced. Changing it back is the undo, so settings have no `undone_at`.
- **What each action hides,** replayed from the events that aren't undone (decisions 9 and 14):

  | Action | Hides | Matched on |
  | --- | --- | --- |
  | `recognize` on an amount or new-merchant flag | This flag and every **later** flag for this user with the same `reason_code` at the same `merchant_key`. Earlier flags stay (FR-7 §8) | `merchant_key`, `reason_code`, `transaction_ts` |
  | `recognize` on a **duplicate** | **This flag only.** A later double charge at the same merchant still shows, at every preset | `transaction_id` |
  | `expected` (spikes) | This category and month only (FR-8 §8) | `category`, `period_start` |
  | `not_me` | Nothing. The flag is shown with the marker and the guidance | `transaction_id` |

- **Every action is matched on stored fields, never on `flag_id` alone,** so it survives a promotion. A new model version gives flags new ids, but the merchant, reason, transaction and month stay the same. `flag_id` is kept for the audit trail.
- **Isolation:** actions never reach the model, another user, or another session on the same account. In the demo the store resets on redeploy, as corrections do.

### 4. Web pages

- **"Worth a look" (1f):** the switch at the top, as in the mockup. It posts and redirects, so it works without JavaScript; htmx swaps the two cards in place. Under it, one line per setting:
  - Less often: *"Fewer alerts: only the clearest ones. You may miss some."*
  - Balanced: *"Our standard setting."*
  - More often: *"You'll see more, and more of them will turn out to be ordinary."* In a user's first 90 days, it adds: *"More often starts once we know your usual pattern, after your first 90 days."*
- **Unusual charges:** "I recognize this" and "Not me — what now?" on each row. "Not me" expands into the guidance in place.
- **Spending spikes:** "Expected, all good" on each spike, beside "Ask the coach about this".
- **Undo toast** after every action, as for corrections.
- **A line at the foot of each card** when something is hidden: "2 alerts you recognized aren't shown." with a link that shows them again for the page view only.
- **Overview:** the "Worth a look" count follows the setting and leaves out hidden flags.
- **Transactions (1e):** a flagged row's marker follows the setting and hidden flags in the same way.

### 5. The coach

- The system prompt gains one rule: when `detect_anomalies` shows nothing, check `sensitivity` and `hidden` before saying nothing was unusual, and mention them.
- The coach and outside assistants can change the setting and act on flags through the new tools, previewed until the user agrees (decision 7).
- "Not me" guidance is fixed text from the tool result, not generated, so the coach never improvises advice about fraud.

### 6. Measurement

- **Out of fold, on validation, the way the presets ship** (decision 12). Only the promoted configs are rerun, on their rounds' splits. In each held-out fold:
  - take the Balanced cutoff fitted on the other folds' users;
  - place Less and More on the other folds' users with §1's rule (post-warm-up rows, `top_k` tie-break), as `sfc-model presets` places them on the pool;
  - apply all three to the held-out users.

  That measures precision and also whether 0.5× and 2× hold on users the cutoffs weren't placed on.
- **Reported per preset:** precision, recall and flags per post-warm-up user-month, with user-bootstrap intervals; warm-up flags counted apart. These replace the earlier plan to measure at 0.5× and 2× of each task's common rate: FR-7's common rate (0.11) isn't its Balanced rate (0.126–0.129).
- **Reported, not gated.** Balanced keeps its gates. The presets have no gate, so there's nothing to choose and no test scoring.
- **Results** go in a short report, FR-9 Alert Sensitivity — Results, beside this script's in-sample estimate. The PRD's success-metrics note gains a line.
- **Demo check:** the build prints each demo account's alert count per preset, and `/healthz` reports that presets are loaded for both halves.

### 7. The demo

- **A fourth demo account,** **Sam Patel, `sam@example.com`** (`u_te_fb_0023`, a family budgeter), in `configs/web/demo_accounts.yaml` (decisions 10 and 16).
- **The story, with `u_te_fb_0023`:**
  - On Maya, Ada or Jordan, Less often hides the spike and More often adds nothing: alerts are rare by design.
  - On Sam, Balanced shows a planted duplicate and a planted Groceries spike. More often adds a borderline Gift Nook charge.
  - Act on all three: "I recognize this" on the duplicate hides only that charge, "Expected, all good" on the spike, and "Not me" on the Gift Nook charge.
  - Ask the coach why something was or wasn't flagged.
- The 60-day window stays (decision 6; FR-7 open question 4).

## Metrics and why

1. **Precision per preset:** what a user trades for more alerts. It's the quantity the More often copy describes.
2. **Recall per preset:** what Less often costs.
3. **Flags per post-warm-up user-month per preset:** the alert burden, in the user's own terms for "how often". It checks the 0.5× and 2× targets out of fold.
4. **Warm-up flags per preset,** counted apart: the alert burden a new user sees, which the precision doesn't cover.
5. **Balanced's gates are unchanged:** the PRD's precision ≥ 0.70 and recall above the baseline apply to the default.

## Options considered

### A. What a preset fixes

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) An alert rate after the warm-up: 0.5×, 1×, 2× (decided)** | Answers "how often", with visibly different settings; needs no labels, so it carries to real data (FR-7 §5) | Precision at More often is 0.45–0.49, below the PRD's bar |
| (b) A precision target per preset, e.g. 0.90 / 0.80 / 0.70 | Every setting keeps a stated precision | More often at 0.70 adds about a quarter more alerts (FR-7 feasibility: 883 to 1,091 flags), and 2× adds nothing to the demo accounts' window, so neither would show; needs labels to place |
| (c) A continuous slider | Fine control | Hard to explain; every position needs a measurement |

### A2. What "2×" counts (review of #65, finding 2)

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Flags per post-warm-up user-month (decided)** | The rate the PRD and decision 4 mean; the label contract's precision covers exactly these flags | Warm-up flags have to be reported separately |
| (b) All flags, warm-up included | One count | Unusual charges at More often were then 1.74× per post-warm-up user-month, and about 17% of More often's flags fell outside the precision |

### B. A floor for More often

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) None; the copy says more will be ordinary (decided)** | One control moves both halves; spike flags still obey the product rules, so a "false" spike is still a visibly high month | About half of More often's alerts are ordinary |
| (b) Keep spikes at Balanced on More often | Spike precision stays at its tested level | "More often" silently does half of what it says |
| (c) Cap More often where precision falls to 0.70 | The PRD bar holds everywhere | For spikes that's Balanced itself (0.705 on test): no room |

### C. Where preset cutoffs come from

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Placed once on the pool, committed beside the model (decided)** | Fixed, reproducible, versioned with the model; one user is judged against the same line as everyone | A step after each promotion; the simple-rule fallback fits its own at build time instead |
| (b) Placed by each nightly run | Follows the pool as it grows | Cutoffs drift from night to night; spikes are scored on request, so they'd need a nightly file anyway |
| (c) Per user, top 2× of their own scores | Every user sees a change | A user with nothing unusual gets ordinary charges flagged; FR-8 rejected re-ranking within one user for this reason |

### D. Who can change the setting and act on flags

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) The page directly; the coach and outside assistants with a preview and `confirm` (decided)** | Matches goals and corrections; "show me more alerts" works in chat | Two more write tools to test |
| (b) The page only | Smallest surface | The coach can explain the setting but not change it |

### E. The demo window

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Keep 60 days; add a fourth account (decided)** | Keeps the decision on #30; honest about how rare alerts are | The demo switches accounts to show More often |
| (b) Show earlier alerts too | Every account changes with the setting | Reopens the window decision; a longer page |

### F. Presets in a user's first 90 days (decision 17)

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) During the warm-up, the stricter of the preset and Balanced (decided)** | A new user isn't flooded (about 0.69 unmeasured alerts per user-month at More often, against 0.07 at Balanced); Less often stays Less often (1 warm-up flag on train users, not Balanced's 47) | More often changes nothing for a new account's first 90 days, which needs a line of copy |
| (a′) During the warm-up, Balanced for every setting | Simplest to state | Loosens Less often for new users: Balanced's warm-up alerts, more than they asked for (review on #65) |
| (b) Apply the preset everywhere, as measured | One rule | The noisiest alerts land on the users with the least history, and their precision is unknown |

The demo accounts have years of history, so neither option changes the demo.

## Testing

- **Unit:**
  - presets: on the pool they were placed on, the cutoffs flag 0.5× and 2× as many post-warm-up rows as Balanced (up to ties at the cutoff); ties are broken by id; Balanced equals the promoted cutoff; a presets file for another model version is refused;
  - the per-user warm-up equals the label contract's on the default data;
  - during the warm-up, More often flags at Balanced's cutoff and Less often keeps its own, stricter one (both directions of decision 17);
  - every preset flags every duplicate; every spike flag at every preset passes the product rules (the contract);
  - the simple-rule fallback's Less and More cutoffs are fitted at build time at 0.5× and 2× its rate;
  - flag file: rows down to the More often cutoff are stored; a file without preset meta serves Balanced only;
  - replay: `recognize` hides later flags with the same reason at the same merchant and not earlier ones; `recognize` on a duplicate hides that flag only, and a later duplicate at the merchant still shows; `expected` hides one month; `not_me` hides nothing; undo restores;
  - promotion: after the flags' model version changes, every kind of action still applies (`not_me` and duplicate `recognize` by `transaction_id`).
- **Isolation:** one session's setting and actions never change another session's or another user's view of the same account (demo), or anything for a token without a subject.
- **Tools:** `detect_anomalies` reports `sensitivity` and `hidden`, and `hidden` never counts `not_me`; the coach's writes preview until `confirm`.
- **Web:** the switch posts and redirects without JavaScript; the Overview count, "Worth a look" and Transactions agree at every setting.
- **Demo bundle:** the fourth account shows its measured alerts at each preset; `/healthz` reports presets for both halves.

## Milestones

One PR per milestone, stacked.

1. **Presets:**
   - `sfc-model presets` and committed `presets.json` for both promoted models;
   - the simple-rule fallback's Less and More cutoffs fitted at build time;
   - flag files down to More often, and spikes by preset;
   - `detect_anomalies` with a `sensitivity` argument internally (Balanced by default);
   - the out-of-fold measurement of §6 and FR-9 Alert Sensitivity — Results.
2. **Settings and actions:** the two event tables, replay and undo; the new tools, and `detect_anomalies`' session setting and `hidden`; the switch, buttons, guidance, toasts and footers on "Worth a look"; the Overview and Transactions; the coach's prompt rule.
3. **Demo and docs:**
   - the fourth account and `/healthz`;
   - the PRD (FR-9 in the v1 demo; More often's precision);
   - the Technical Design (tools, presets);
   - the Web App UI (1f, gap 6).

## Implementation notes

From milestone 1 (presets):

- **Out of fold, the presets hold** (FR-9 Alert Sensitivity — Results). Less often and More often flag 0.49× and 1.98× Balanced's unusual charges per post-warm-up user-month on held-out users, and 0.50× and 2.01× Balanced's spikes. Precision is 0.939 / 0.792 / 0.455 for unusual charges and 0.963 / 0.799 / 0.491 for spikes. Balanced reproduces each round's own out-of-fold precision.
- **Committed presets** (`sfc-model presets` on the default dataset, all 360 users):
  - unusual charges: Less 0.7783, Balanced 0.7300, More 0.6855;
  - spikes: Less 12.64, Balanced 7.682, More 5.553.
- **Balanced is always the model's own cutoff.** `SpikeState.model_at("balanced")` returns the model itself, never a copy at the presets' number, so anything that reads or adjusts the model's cutoff sees Balanced.
- **The warm-up is counted from a user's first transaction of any kind,** income included. On the synthetic data that's the label contract's warm-up exactly, and a test checks it.
- **`/healthz` reports which halves have presets** (`presets.unusual_charges`, `presets.spending_spikes`). This was planned for milestone 3 and done here, since milestone 1 builds the files it reads.
- **The flag file records its cutoffs in meta** (`presets`, JSON). A file without it serves Balanced at every level, so a bundle built before FR-9 still works.
- **`detect_anomalies` reports the level it applied** (`sensitivity`): Balanced whenever neither half has presets, whatever was asked. Milestone 2 takes the level from the session.

From milestone 2 (settings and actions):

- **`access/alerts.py`** holds the store (`AlertStore`, two tables in the feedback store's file) and the replay (`AlertView`). The replay keeps the action that hides each alert, so a hidden alert can be shown again by undoing it.
- **Sources are `page`, `coach` and `assistant`,** as the tokens' `client` claim gives them; §3 named `page` and `coach` only.
- **An action is refused if the same action on the same alert is already active** ("you've already done that"). After an undo, it can be taken again.
- **Any stored charge, and any spike at the loosest level, can be acted on,** whatever the session's setting, so an alert seen at More often can still be acted on after switching to Balanced.
- **Undo is the subject's own, on the current account.** An action taken on another demo account can't be undone from this one.
- **The hidden-alert footer counts the page's own lists,** each card's window, so the number is what "Show them" brings back. `detect_anomalies` takes an internal `include_hidden` for it; it isn't a tool argument.
- **The coach's prompt** gains the alert rules: when nothing is listed but `hidden` isn't zero, or the setting is Less often, say so; change the setting or act only when asked; preview, then `confirm`. When someone says a charge isn't theirs, the coach offers `not_me` and gives its fixed guidance.

From milestone 3 (demo and docs):

- **Sam Patel (`u_te_fb_0023`, `sam@example.com`) is the fourth demo account.** On the built bundle, "Worth a look" shows, as (unusual charges, spikes) at Less / Balanced / More:
  - Maya: (0, 0) / (0, 1) / (0, 1);
  - Ada: (0, 0) / (0, 1) / (0, 1);
  - Jordan: (1, 0) / (1, 1) / (1, 1);
  - Sam: (1, 1) / (1, 1) / (2, 1).

  The demo bundle's flag file holds 52 charges down to More often, against 12 at Balanced.
- **The build prints those counts** (`sfc-web build-demo`), so a rebuilt bundle shows at once whether the demo story still holds.
- **The deploy's smoke test requires presets for both halves** in `/healthz`, as it already requires a spike scorer.
- **Docs:** the PRD (FR-9 in the v1 demo; the out-of-fold numbers; the risk and releases), the Technical Design (tools, presets, the feedback store) and the Web App UI (1f; gap 6 closed).

## Status (Oct 5, 2026)

| Milestone | PR | Outcome |
| --- | --- | --- |
| Design | #65 | Accepted (owner, Oct 4, 2026); decisions 1–17 |
| 1. Presets | #66 | `sfc-model presets`; presets committed for both promoted models; flag files down to More often; spikes by level; out of fold 0.49×/1.98× and 0.50×/2.01× of Balanced's rate (FR-9 Alert Sensitivity — Results) |
| 2. Settings and actions | #67 | The alert store and replay; four tools over MCP; the switch, actions, undo and hidden alerts on "Worth a look"; the coach's rules |
| 3. Demo and docs | #68 | Sam Patel's account; per-level counts in the build; `/healthz` presets in the deploy smoke test; PRD, Technical Design, Web App UI |

## Decisions and open questions

**Decisions** (owner, Oct 4, 2026)

1. [x] **FR-9 is in the Oct 6 demo.**
2. [x] **One control for both halves,** as in the mockup.
3. [x] **The flag actions ship with it:** "I recognize this", "Not me — what now?", "Expected, all good".
4. [x] **Presets are alert rates:** Less = 0.5×, Balanced = the promoted cutoff unchanged, More = 2×, per post-warm-up user-month (decision 11). Precision is measured on validation, out of fold; no new test scoring (option A-a).
5. [x] **No precision floor for More often, on either half.** The product rules still hold, the 0.70 gate applies to Balanced, and the page says more alerts will be ordinary (option B-a).
6. [x] **A fourth demo account shows More often; the 60-day window stays** (option E-a). The first choice was `u_te_fb_0000`, pending a look at its flags. Which account is decision 16.

**Decided on the review of #65** (owner, Oct 4, 2026)

7. [x] **The coach and outside assistants can change the setting and act on flags,** previewed until `confirm` (option D-a).
8. [x] **Preset cutoffs are placed once on the pool and committed beside the model** (option C-a).
9. [x] **"I recognize this" hides later flags only,** with the same reason at the same merchant (FR-7 §8). **Except on a duplicate, where it hides that flag only,** so later duplicates at the merchant still surface at every preset (review finding 3).
11. [x] **"2×" is the alert rate after the warm-up** (option A2-a; finding 2). Less and More are matched on post-warm-up rows, so 0.5× and 2× hold per post-warm-up user-month. Warm-up flags are reported at each preset separately, so they're never silently outside the precision.
12. [x] **The out-of-fold measurement places presets the way the command will** (finding 1). In each fold: the fitted Balanced cutoff, then Less and More by §1's rule. It reports precision, recall and flags per post-warm-up user-month (§6).
13. [x] **The simple-rule fallback fits its Less and More cutoffs at build time,** at 0.5× and 2× its rate (finding 4).
14. [x] **`not_me` is matched on `transaction_id`,** which `flag_actions` stores, so the marker survives promotions (finding 5).
15. [x] **`hidden` counts `recognize` and `expected` only** (finding 6).

**Decided to close the open questions** (owner, Oct 4, 2026, on #65)

10. [x] **The fourth account is Sam Patel, `sam@example.com`.**
16. [x] **The fourth account is `u_te_fb_0023`,** not `u_te_fb_0019`. At Balanced it has a planted duplicate and a planted Groceries spike, and More often adds a borderline charge, so one account shows More often and all three actions, including the duplicate exception.
17. [x] **In a user's first 90 days, serving uses the stricter of the session's preset and Balanced** (option F-a, with the review's fix). More often starts after the warm-up, where its unusual charges would otherwise flag about 0.69 per user-month, 10× Balanced's rate, at unmeasured precision; Less often stays Less often throughout.
