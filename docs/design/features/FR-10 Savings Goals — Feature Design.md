# FR-10 Savings Goals — Feature Design

Oct 3, 2026 · @Sidd · Status: **Accepted** (owner, Oct 3, 2026) · Branch: `docs/fr-10-design`

## Summary

This feature lets a user set up a savings goal with an amount and a target date, and change or remove it later. It also fixes the goal record that FR-11 and FR-12 forecast from.

- **Requirement:** FR-10 (P0), *"Let users define a savings goal with an amount and a target date."* User story 3 and key scenario 2 depend on it. FR-11 (on track, gap, certainty) and FR-12 (seasonality, the user's own behavior) build on its records.
- **Starting point:**
  - Every synthetic user has 1–2 generated goals: 545 goals for 360 users on the default dataset, 297 of them still running on the app's "today" (Sep 30, 2026). They are read-only.
  - `list_goals` lists them, and `forecast_goal` returns "not available yet".
  - The Goals page is a "coming next" panel, and the setup screen (mockup 1h) has no tool behind it (Web App UI, gap 4).
- **Approach:**
  1. **Goals are user state, kept outside the dataset.** A user's goals are the generated goals with that user's changes applied on top. The changes are stored as an event log in a `goals.sqlite`, next to FR-5's feedback store and built the same way. Every change can be undone. The generator's file stays a pure function of its spec.
  2. **A goal records the amount saved so far, entered by the user.** v1 has no savings accounts, so the app can't observe a balance, and inferring one would invent a number (FR-14). The amount defaults to $0 and can be updated later.
  3. **The setup screen shows only facts, never a verdict.** While the user types, `check_goal` returns the monthly amount needed, what other goals already need each month, and the user's median monthly savings over the last 12 months. Each number comes from a tool. The "Within reach" or "A stretch" badge waits for FR-11's probability, so v1 never shows two different answers to "can I make it?".
  4. **One set of tools for the web app, Wren and outside assistants:** `check_goal`, `create_goal`, `update_goal`, `archive_goal` and `undo_goal_change`, with an extended `list_goals`. Wren and outside assistants change goals only after the user says yes, as in FR-5.
  5. **Isolation works as in FR-5.** Tools never take a `user_id`. In the demo, goal changes are keyed by browser session, so two visitors on one demo account never see each other's goals.
- **No model and no evaluation round.** FR-10 is a feature built on records. Its quality checks are validation, isolation and contract tests, plus coach-suite questions. FR-11 owns the forecast and its metrics.
- **Owner decisions** (Oct 3, 2026) settled all six questions: goals are in the Oct 6 demo, FR-10 stores no contribution plan, no fit badge before FR-11, the limits, coach and assistant writes with confirmation, and an "Ended" section. See [Decisions and open questions](#decisions-and-open-questions).

## Context

### What exists

| Piece | State on `main` (`5689d39`) |
| --- | --- |
| Generated goals (FR-1) | `goals` table: `goal_id`, `user_id`, `name`, `target_amount`, `created_date`, `target_date`, `as_of_date`, `current_balance`. Target dates are month ends. `truth_goals` holds `outcome_class` and `met`, for evaluation only |
| How a generated balance is made | A hidden share of the user's monthly net savings, added up from `created_date` to `as_of_date`. No transaction moves money into a goal |
| Goals whose target date is past | 248 of 545. Their `as_of_date` is the backtest origin, 3–12 months before the target, so the balance shown is stale and the outcome is unknown to the app |
| `list_goals` | Name, amount, target date, saved, saved as of; `"forecast": "not_available"`. No `goal_id` |
| `forecast_goal` | Takes `goal_name`; returns "not available yet" |
| Web | Overview shows the first active goal's balance; `/goals` is the "coming next" panel |
| Demo accounts | Maya: Vacation fund ($17,600 by Jul 2027) and a New laptop goal that ended in May 2025. Ada: Emergency fund ($14,050 by Jun 2027) and a College fund that ended in Jan 2025. Jordan: Tax reserve ($8,900 by Apr 2027) |

### What the mockups ask for

- **1h Goal setup:** fields for name, amount and date ("By"), then a "How it fits" box with a badge ("Within reach", "A stretch"), "$250 a month to get there" and a note such as "Together with your emergency fund that's about $1,160 a month in savings. You've averaged $820."
- **1g, 1l Goal detail:** saved of target, a likely range, on-track status and what would close the gap. That is FR-11's job, not this one.
- **Web App UI, gap 4:** "How it fits" needs numbers for an unsaved goal, those numbers must come from a tool, and creating and editing goals needs a write tool.

### What FR-5 already settled that FR-10 reuses

- User writes go to a SQLite file in a writable folder, kept apart from the read-only demo bundle. In v2 they move to Postgres with row-level security.
- Every read and write is scoped by a **subject**: the signed-in user in production, the browser session in the demo. The subject travels in the session and in signed MCP tokens (`fb` claim, #36). It never comes from a tool argument.
- Changes are events, and current state is replayed from them, so undo is a mark and the log stays an audit trail.
- The coach applies a change only with a `confirm` flag that it sets after the user agrees.

## Scope

**In:** creating, editing, archiving and undoing goals; validation; the facts in the setup check; the Goals page with its setup form; the overview goal card; tools over MCP; the coach's use of them; the goal record FR-11 and FR-12 read.

**Out:**

- Forecasts, on-track status, ranges and what would close the gap (FR-11).
- Seasonality and income swings (FR-12).
- The fit badge, until FR-11.
- Linking goals to accounts or transfers (v2, with real bank data).
- What-if scenarios (Web App UI, gap 5).
- Recurring goals and goals without a date.

## Goals and non-goals

**Goals**

1. A user can create a goal (name, amount, date, amount saved so far) from the Goals page or by asking Wren, and sees it immediately in the dashboard and in coach answers.
2. Every number on the setup screen and in a coach answer about a new goal comes from a tool (FR-14, NFR-1).
3. Invalid goals are refused with a message the user can act on, the same message in the form and in chat.
4. Every change can be undone, and nobody ever sees another user's goals, or in the demo another session's (NFR-2).
5. FR-11 gets one record per goal with everything a forecast needs, the same for generated and user-created goals.

**Non-goals**

- Saying whether a goal is achievable. FR-10 states the arithmetic; FR-11 gives the judgment with its range (NFR-7).
- Tracking deposits. In v1 the saved amount changes only when the user updates it.

## What it will look like

### Flows

| Flow | Where | What happens |
| --- | --- | --- |
| **Set up a goal** | Goals page, "New goal" (1h); on mobile, a full page | The user types a name and an amount, picks a month ("By June 2027") and optionally enters what they've saved. After each pause in typing, the "How it fits" box updates with the facts from `check_goal`, or with the field-level problem. "Create goal" saves it at once (the submit is the user's yes), and a toast offers Undo |
| **Set up through the coach** | Chat | "I want to save $3,000 for a trip by next June." Wren calls `check_goal`, repeats the date it returns ("by Jun 30, 2027") and the monthly amount, and asks to confirm. Then it calls `create_goal` with `confirm: true` and says what it saved |
| **Edit** | Goal row, "Edit" | The same form, prefilled; the check excludes the goal from "other goals". Changing "saved so far" records it as of today |
| **Archive** | Goal row, "Remove" | The goal leaves the lists and coach answers; a toast offers Undo. Nothing is deleted. A "Removed goals" disclosure at the foot of the page lists archived goals with "Restore", so a removal can be undone after the toast is gone |
| **Reached goals** | Goals page, in the list with the running goals | A goal whose saved amount has reached its target before its date: a "Reached" state, saved of target, no monthly amount. It can be edited (a higher target makes it active again) or archived |
| **Ended goals** | Goals page, an "Ended" section | Generated goals whose date has passed: "Saved $1,184 as of Aug 31, 2024". No outcome, since the app doesn't know it. They can be archived, not edited |
| **Overview** | Goal card | The active goal with the nearest target date: saved of target, months left and the monthly amount needed. With no active goal, a reached one ("Reached: $2,100 of $2,100"). With neither: "Set up a savings goal". The on-track line stays "arrives with goal forecasting" until FR-11 |

### The setup check ("How it fits")

Before FR-11, the box shows three facts and no badge:

> **$445 a month** to get there by Jun 30, 2027 (9 months).
> With your Emergency fund, that's about **$1,160 a month** across your goals. Over the last 12 months you've saved a median of **$820 a month**.

- **A goal is due at the end of a month.** Any day given is stored as the last day of its month, and every result returns that date, so "by next June" means Jun 30, 2027 whether Wren resolves it to Jun 1 or Jun 30. That's the shape of the generated goals and what FR-11's monthly forecast runs to.
- **Needed per month** = (target − saved) ÷ months left, rounded up to the dollar. Months left counts the month ends after today up to the target date: from Sep 30, 2026 to Jun 30, 2027 is 9.
- **Across your goals** is the sum of "needed per month" over the user's other active goals, plus this one. Reached goals need nothing and aren't counted. It's left out when there are no other goals.
- **Median monthly savings** is income minus spending per calendar month, the median over the last 12 full months. It's the sum of the ledger's amounts, so it doesn't depend on categories, and a FR-6 correction can't change it. It's a median, not a mean, so one bonus month or one big one-off doesn't move it. With fewer than 3 full months it says so and gives no figure (PRD risk: short histories).

When FR-11 lands, `check_goal` gains `p_goal_met` and the range for the draft, and the badge appears, derived from that probability. The three facts stay.

### Tools (tool server)

No tool takes a `user_id`. The web app and the coach call the same tools, so they can't disagree. Goals are addressed by `goal_id`, which `list_goals` returns. All amounts are in USD, like the other tools, and dates are `YYYY-MM-DD`.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `list_goals` (extended) | `include_ended` (default `true`), `include_archived` (default `false`) | Per goal: `goal_id`, `name`, `target_amount`, `target_date`, `saved`, `saved_as_of`, `created_date`, `status` (`active`, `reached`, `ended`, and `archived` when asked for), `origin` (`yours`, `existing`), `undo_revision_id` (the goal's latest change that can be undone, or `null`), and for active goals `months_left` and `needed_per_month`. Plus `median_monthly_savings_12m`. `forecast` stays `not_available` until FR-11 |
| `check_goal` (read-only) | `name`, `target_amount`, `target_date`, `saved` (default 0), `goal_id` (when editing) | `valid`, `problems` (field, code and message), and when valid `months_left`, `needed_per_month`, `other_goals_per_month`, `all_goals_per_month`, `median_monthly_savings_12m`, `months_of_history`. Writes nothing |
| `create_goal` | Same as `check_goal` without `goal_id`, plus `confirm` | The goal and the `revision_id`. From Wren or an outside assistant without `confirm: true`: the check, with `"status": "needs_confirmation"`, and nothing written. The Goals page applies on submit |
| `update_goal` | `goal_id`, and any of `name`, `target_amount`, `target_date`, `saved`, plus `confirm` | As `create_goal`. An updated `saved` is recorded as of today |
| `archive_goal` | `goal_id`, `confirm` | The archived goal and the `revision_id` |
| `undo_goal_change` | `revision_id` | The goal as it is now (or `"status": "removed"` when undoing a creation). Refused, with the validation codes, when the result would break a rule (§2) |
| `forecast_goal` (changed) | `goal_id`, replacing `goal_name` | Still "not available yet". The Technical Design's contract already uses `goal_id`, and nothing depends on the name |

The write tools carry MCP's `destructiveHint: false` and `idempotentHint: false`; `check_goal` and `list_goals` carry `readOnlyHint: true`, so outside assistants can run reads without asking.

**Where a write comes from, and when it needs `confirm`.** As in FR-5 (#36), the caller's context sets `source`; no tool argument does.

- `edit`: the Goals page, FR-5's name for dashboard changes. It applies on submit.
- `coach` and `assistant`: calls over MCP, told apart by the token's `client` claim. Wren's in-process tokens carry `client="coach"`; any other client is `assistant`. Both need `confirm: true` for every write, since a goal change is always one the user should see first.
- `undo_goal_change` applies directly from every caller, as FR-5's `undo_correction` does: the user has just asked for it, and it only takes the goal back to the state they last saw. The coach prompt allows it only when the user asks to undo, and Wren says what changed afterwards. The undone event stays in the log.
- A token without a subject gets no goal writes, as in #36: the write tools refuse with "not available for this sign-in".

## Design

### 1. The goal record

One shape for every goal, generated or created by the user. FR-11 and FR-12 read only this.

| Field | Meaning |
| --- | --- |
| `goal_id` | Generated: `g_<user>_<n>` (FR-1). The user's: `gu_` + 12 hex characters, so the two can't collide |
| `name` | 1–40 characters after trimming; unique among the user's running goals (active or reached), ignoring case |
| `target_amount` | $50 to $1,000,000. Stored as integer cents; tools take and return dollars, and refuse amounts with fractions of a cent |
| `target_date` | The last day of a month, from the month after today's to 120 months ahead. Any day given is moved to the end of its month |
| `created_date` | The day it was created: the app's today (the dataset's `calendar_end`; Web App UI, gap 8) |
| `saved`, `saved_as_of` | The amount saved toward the goal (integer cents, like the target) and the day it was recorded. Generated goals keep `current_balance` and `as_of_date`, converted to cents on read |
| `status` | Checked in this order. `ended`: the target date is today or before, whatever was saved. `reached`: saved ≥ target. `active`: everything else. Archived goals are listed only when asked for |
| `origin` | `existing` (generated, as if it predated the app) or `yours` |

**Reached goals stay in view.** On the default dataset 19 of the 297 running generated goals already have saved ≥ target on Sep 30, 2026. They stay in the list with the running goals, show "Reached" and no monthly amount, and don't count toward the 10-goal limit.

**Why "saved" is entered, not computed.** In the synthetic data no money moves into a goal: the generator adds up a hidden share of net savings. In v1 the only honest source for "saved so far" is the user. The dataset's balances for generated goals stand in for a balance they would have entered, and FR-11 already treats them that way. In v2 a linked account replaces this field.

### 2. Store, events and effective goals

`access/goals.py`, modelled on FR-5's `access/feedback.py`:

```sql
CREATE TABLE goal_revisions (
    revision_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,  -- order of events; later wins
    subject TEXT NOT NULL,
    user_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    op TEXT NOT NULL CHECK (op IN ('create', 'update', 'archive')),
    name TEXT NOT NULL,           -- the goal's full state after this event
    target_amount_cents INTEGER NOT NULL,
    target_date TEXT NOT NULL,    -- a month end
    saved_cents INTEGER NOT NULL,
    saved_as_of TEXT NOT NULL,
    created_date TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('edit', 'coach', 'assistant')),
    created_at TEXT NOT NULL,
    undone_at TEXT
);
CREATE INDEX idx_goal_revisions_subject ON goal_revisions (subject, user_id, seq);
```

- **Each event stores the goal's full state after it**, not a diff, so replaying is "the last event per `goal_id` that isn't undone". An `archive` event hides the goal.
- **Effective goals** = the dataset's goals for `user_id`, then the subject's events replayed over them by `goal_id`. A generated goal that was edited is replaced, one that was archived is hidden, and the user's new goals are added.
- **Undo** marks an event undone and replays. Only the latest live event of a goal can be undone, so an undo never resurrects a state the user didn't see last. Undoing a `create` removes the goal. Someone else's revision is indistinguishable from one that doesn't exist.
- **Undo is validated like a write.** The latest-event rule is per goal, so on its own it would allow this: archive *Trip*, create a new *Trip*, then undo the archive, which leaves two active *Trip* goals. The same steps with 10 goals give an 11th. So undo replays the result inside the write lock and checks the rules that depend on other goals (`name_in_use`, `too_many_goals`). If the result breaks one, the undo is refused with the same code and message, and nothing changes.
- **Writes are serialized** by a lock, and validation runs again inside it. Two tabs creating "Trip" at once get one goal and one "name in use".
- **Setting:** `SFC_GOALS_DB`, defaulting to `data/goals.sqlite`. The demo image sets `/var/lib/sfc/goals.sqlite`, next to FR-5's feedback store. A separate file rather than a table in `feedback.sqlite`: the two features land independently, and in v2 they become two tables in one Postgres database anyway.
- `Ledger` stays read-only and cached. Effective goals are computed per call from the cached dataset goals and a small per-subject read. A subject has at most a few dozen events, well inside NFR-5's 2 s budget.

### 3. Validation

The same function serves `check_goal`, the write tools and the form, so a message reads the same everywhere. Each problem has a field, a code and a message:

| Code | Rule | Message (example) |
| --- | --- | --- |
| `name_missing`, `name_too_long` | 1–40 characters | "Give the goal a name." |
| `name_in_use` | Unique among running goals (active or reached), ignoring case; checked on undo too | "You already have a goal called Vacation fund." |
| `amount_range` | $50 to $1,000,000 | "Goals start at $50." |
| `amount_invalid` | The amount (or saved amount) is a number | "Enter an amount in dollars." |
| `amount_cents` | Whole cents only, for the target and the saved amount. Parsed through `Decimal(str(x))`, never `x * 100`, which turns 19.99 into 1998.999… | "Use dollars and cents." |
| `date_invalid` | The date reads as `YYYY-MM-DD` (the Goals page's month picker shows "Pick a month.") | "Use a date like 2027-06-30." |
| `date_too_soon` | The month after today's or later | "Pick October 2026 or later." |
| `date_too_far` | At most 120 months after today's | "Pick a date within 10 years." |
| `saved_range` | Never negative. When creating, below the target. When updating, the target or more is allowed and makes the goal `reached` | "That's already the whole amount." |
| `too_many_goals` | At most 10 active goals (reached and ended goals don't count); checked on undo too | "You have 10 active goals; finish or remove one first." |
| `goal_not_editable` | Ended goals can only be archived | "This goal's date has passed." |

The limits are product choices, not model needs (decision 4).

### 4. Web pages

- **`/goals`** replaces the "coming next" panel: running goals as cards (saved of target, progress bar, then months left and needed per month, or "Reached"), a "New goal" card, the "Ended" section and the "Removed goals" disclosure. FR-11's range and status slot into each card later.
- **Setup and edit form:** a dialog on wide screens and a page under 600 px, as in 1h. "By" is a month picker. `hx-post="/goals/check"` with `hx-trigger="input changed delay:300ms"` swaps in the "How it fits" partial. Submit posts to `/goals` (or `/goals/{id}`); the response re-renders the list with an Undo toast. With JavaScript off, the form still works as a plain post.
- **Overview card** as in [Flows](#flows). The "Soon" badge leaves the Goals link in the sidebar.
- **State changes are same-site POSTs,** so the session cookie's `SameSite=Lax` covers CSRF (Web App UI, decision 1).

### 5. The coach

- **The system prompt gains:** call `check_goal` before suggesting or creating a goal; quote its numbers and never compute a monthly amount; ask before any change; call a write tool with `confirm: true` only after the user agrees in the conversation; after a change, say what changed and that it can be undone.
- **Asking "will I make it?"** still gets `forecast_goal`'s "not available yet" until FR-11. Wren states the facts from `check_goal` without a verdict (FR-14).
- **Relative dates** ("by next June") are resolved by Wren against the `as_of` in every tool result. It repeats the month-end date `check_goal` returns, not its own reading, before confirming.
- **Undo later:** "undo that" in a new chat uses `undo_revision_id` from `list_goals`, so a change can be undone after the toast or the chat that made it is gone.
- **Outside assistants** reach the same tools over MCP, scoped by their token's user and subject. Their writes have `source: assistant` (from the token's `client` claim) and need `confirm` like Wren's.

### 6. Isolation and safety

- Tools take no `user_id` or subject; both come from the session or the signed token (Technical Design, "Security and data isolation"). The schemas reject unknown arguments, as today.
- Every read of `goal_revisions` is filtered by subject and user in the store, not in each tool. A `goal_id` or `revision_id` that belongs to someone else gets the same "no such goal" as one that doesn't exist.
- A goal name is the user's own text. It's shown back only to the same subject, never logged at INFO, and passed to the coach as tool data, not instructions.
- No investment advice: the setup check talks about saving, never about where to put the money (NFR-4).

### 7. What FR-11 and FR-12 get

- `forecast_goal(user_id, goal_id)` reads the effective goal: `target_amount`, `target_date`, `saved`, `saved_as_of`.
- FR-11 forecasts from `saved_as_of`, using only transactions up to that day. For active goals that day is today. For evaluation, generated goals with known outcomes keep their backtest origin, as FR-1 set up.
- **How savings split across goals is FR-11's call** (decision 2). FR-10 stores no contribution plan. The setup check's "across your goals" line is plain arithmetic and makes no claim about allocation.
- User-created goals have no ground truth, so they're never scored. FR-11's metrics use the generated goals only.

## Options considered

### A. Where goals live

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Write into the dataset's `goals` table | One table | Breaks "the generator's file is a pure function of its spec"; the demo bundle is read-only; evaluation would read user writes |
| (b) A mutable `goals` table in a writable store | Simple reads | No history, no undo, no audit |
| **(c) An event log over the generated goals (chosen)** | Undo and audit for free; the same shape as FR-5; generated goals stay untouched for FR-11's evaluation | A replay per read (a few dozen rows) |

### B. The amount saved so far

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Always $0 at creation | Simplest form | Wrong for anyone who has already started; forecasts start low |
| **(b) Entered by the user, default $0 (chosen)** | Honest; one optional field | A number the app can't verify (v1 has no accounts) |
| (c) Inferred from net savings since some date | No typing | Invents a number (FR-14); the generator's allocation share is hidden by design |

### C. The fit badge before FR-11

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) No badge; facts only (chosen)** | Every number is checked; no verdict that FR-11 may later contradict | The setup box is less punchy than 1h |
| (b) Fixed thresholds now (needed per month against the median and lower-quartile months) | Matches 1h at once | A second, model-free "can I make it?" that disagrees with FR-11's probability for seasonal or irregular earners (FR-12's whole point); no range (NFR-7) |
| (c) LLM-written fit note | Reads well | Breaks with the LLM down (NFR-6); numbers not from a tool |

### D. Previewing a draft

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) A separate read-only `check_goal` (chosen)** | Outside assistants can run it without approval; clean live checking in the form | One more tool |
| (b) `create_goal` with `confirm: false` as the preview | One tool fewer | A preview counts as a write to MCP clients, so every keystroke check would need approval |

### E. Who can change goals

| Option | Pros | Cons |
| --- | --- | --- |
| (a) Only the Goals page | No risk of the coach changing data | Breaks "ask in plain English" for the most natural goal request; unlike FR-5 |
| **(b) The page, Wren and outside assistants, with confirmation (chosen)** | Same tools everywhere (FR-19); consistent with FR-5 | Relies on the coach honoring `confirm`; tested in the coach suite |

### F. Goals in the shared demo

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) Keyed by browser session, in the container (chosen; as FR-5)** | Visitors never see each other's goals; the seeded goals are always there | Gone on restart or redeploy |
| (b) Keyed by demo account | Persists across visitors | One visitor's goals show for the next (NFR-2 in spirit) |
| (c) Nightly reset | Clean every day | Visitors still collide during the day |

## Testing

- **Validation:** each rule at its boundary (40 and 41 characters, $49.99 and $50, $50.001 refused while 19.99 and 0.29 are accepted, the first allowed month, 120 and 121 months, a case-insensitive name clash with an active and with a reached goal, the 11th active goal with a reached one not counted). Moving dates to month ends (Jun 1 and Jun 30 give the same goal, February in a leap year). Status order: a goal both reached and past its date is `ended`. An update to saved ≥ target gives `reached`; a negative saved amount is refused.
- **Store:** replay order; editing and archiving a generated goal; undoing a creation, an edit and an archive; undoing anything but the latest event is refused; concurrent creates with one name give one goal. Undo is refused when it would bring back a duplicate name (archive *Trip*, create *Trip*, undo the archive) or an 11th active goal. Editing a reached goal back to active (a higher target or a lower saved amount) while 10 goals are active is refused too. Generated balances convert to cents exactly.
- **Confirmation and source:** a Goals-page write applies at once with `source: edit`; a `coach` or `assistant` write without `confirm` writes nothing; the source follows the token's `client` claim; a token without a subject gets no writes.
- **Isolation (adversarial):** another user's or another session's `goal_id` and `revision_id`, on every tool, give "no such goal"; `user_id` or `subject` as a tool argument is rejected; an MCP token for user A never sees B's goals; two sessions on one demo account see the seeded goals and only their own changes.
- **Contract:** JSON shapes of the six tools, the same through `Tools` and through MCP. `needed_per_month` and `median_monthly_savings_12m` checked against hand-computed values for one demo user.
- **Web:** create, edit, archive and undo through the pages; restore from "Removed goals" after a reload; the check partial shows field problems; the overview card with zero goals, a reached goal only, and one and two active goals; the form without JavaScript.
- **Coach suite** (`llm` marker, release gate): "set a $3,000 goal for next June" leads to `check_goal`, a confirmation question and no write before the yes; "will I make it?" leads to no verdict before FR-11; every number in the answers comes from a tool result (grounding).

## Milestones

One PR each, small, since the demo deploys on every merge.

1. **Store and validation** (#42): `access/goals.py` (events, replay, undo, effective goals), the validation table, month counting, `SFC_GOALS_DB`; unit and isolation tests.
2. **Tools** (#43): extended `list_goals`, `check_goal`, the write tools, `undo_goal_change`, `forecast_goal` by `goal_id`; MCP registration with hints; coach prompt rules; contract tests; the Dockerfile's `SFC_GOALS_DB`. Builds on #36's session subject in tokens.
3. **Web** (#45): the Goals page, the setup and edit form with the live check, archive and undo, the overview card; a click-through on the deployed demo.
4. **Docs** (this milestone): the Technical Design's Goal schema and tools table; the Web App UI's gap 4 (closed except the badge), screen table and open question on mutable demo data.

## Implementation notes

Where the build departs from the design above, or settles what it left open:

- **Two more validation codes,** `amount_invalid` and `date_invalid`, for input that isn't a number or a date. The table assumed well-formed input; the form and the coach can both send otherwise.
- **The Dockerfile's `SFC_GOALS_DB` moved from milestone 3 to milestone 2.** Milestone 2 makes the app open the goal store at startup, and the image's `/app` isn't writable by the app user, so without it the deployed app wouldn't start.
- **Median monthly savings:** the first month of data counts as partial, so a history from Oct 1, 2024 to Sep 30, 2026 has 23 full months. The median is in cents, over the last 12 full months; a month with no transactions counts as zero.
- **Blank form fields stay blank** when the Goals page sends them, so a cleared name or amount is reported rather than silently left unchanged. A blank saved amount means $0 for a new goal and "unchanged" for an edit.
- **The coach suite's goal questions** (Testing) wait for the coach suite itself: the repo has only the `llm` marker so far. The prompt rules are in `experience/coach.py`.

## Decisions and open questions

1. [x] **In the Oct 6 demo:** yes, after FR-5's #35 and #36 land, since FR-10 reuses their session subject. Milestones 1–3 are small and touch no model (owner, Oct 3, 2026).
2. [x] **How savings split across goals** is FR-11's decision, so FR-10 stores no contribution plan. FR-11 starts from splitting in proportion to "needed per month" (owner, Oct 3, 2026).
3. [x] **No fit badge before FR-11** (option C(a)). The setup box shows facts only until FR-11's probability is available (owner, Oct 3, 2026).
4. [x] **Limits:** $50 to $1,000,000; a month from next month to 10 years ahead; names up to 40 characters; 10 active goals per user (owner, Oct 3, 2026).
5. [x] **Wren and outside assistants may create, edit and archive goals** after an explicit yes (option E(b)), as in FR-5 (owner, Oct 3, 2026).
6. [x] **Ended generated goals** are shown in an "Ended" section with their last recorded balance and no outcome, rather than hidden (owner, Oct 3, 2026).
