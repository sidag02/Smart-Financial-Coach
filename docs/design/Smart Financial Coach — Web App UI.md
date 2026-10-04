# Smart Financial Coach — Web App UI

Oct 2, 2026 · Owner: @Sidd · Status: **Accepted** (owner decisions, Oct 2, 2026; recorded in the Delivery Plan and the Technical Design too) · Branch: `design/mockup`

## Summary

Hi-fi mockups for the v1 web app are in [`mockups/`](mockups/). This note maps each screen to its requirement, the tools behind it and the milestone that ships it, records the UI decisions, and lists what the mockups need that no design or tool provides yet.

- **Build to the mockups, in the Delivery Plan's order.** Panels ship in a "not available yet" or "coming next" state first and switch to real numbers in the PR that promotes the model behind them (the Delivery Plan's sync rule).
- **Demo for Oct 6, 2026:** sign-in, Overview, Transactions and coach chat run on real data; Worth a look shows unusual charges since FR-7's promotion (spikes follow with FR-8); Goals runs on real data with FR-10 (setting up, editing and removing goals) and FR-11–12 (each goal's status, the goal detail with its likely range and top-up, and the setup check's fit badge). See [Demo build](#demo-build-oct-6-2026).
- **Open items** are in [Gaps](#gaps-the-mockups-need-that-nothing-provides-yet) and [Open questions](#open-questions).

## Mockups

Open `mockups/Smart Financial Coach - Light & Dark.dc.html` from a local server (`python -m http.server` in that folder); the files load React from unpkg. The light and dark files hold screens 1a–1l and 2a–2l; `Sidebar*.dc.html` is the shared nav and `support.js` the viewer runtime.

- Data is mocked per persona with the repo's 13 categories, through Sep 30, 2026. Persona is a design-time switch only; customers never see persona labels.
- Coach copy in the mockups uses only numbers a tool could return (FR-14) and gives no tax or investment advice (NFR-4); answers that need a what-if forecast are gap 5.
- Visual direction: cream grounds, apricot accent, sage for good news, amber for "worth a look"; Bricolage Grotesque and Figtree.

## Screens

| Screen | Requirement | Tools | Ships in |
| --- | --- | --- | --- |
| 1i Sign in | FR-18 | Session | Demo; P2 |
| 1a Overview (classic grid), 1j mobile | FR-17 | `get_spending_summary`, `get_transactions`, `list_goals` | Demo; P2 (flags panel "coming next"; the goal card with FR-10, its status and likely amount with FR-11) |
| 1e Transactions list | FR-3, FR-17 | `get_transactions` with confidence | Demo, read-only; P2 |
| 1e "Not sure?" review panel | FR-5, FR-6 (P1) | #15: `list_review_items`, `resolve_review_item`, `correct_category`, `undo_correction` | After #15 milestone 3 |
| 1d, 1k Coach chat with sources | FR-13–16 | Read tools, plus goal and category writes with confirmation | Demo; P3 |
| 1f Worth a look | FR-7, FR-8, FR-9 | `detect_anomalies` | Unusual charges: live since FR-7's promotion (Oct 3, 2026), last 60 days; spikes when FR-8 promotes; actions with FR-9 (v1.1) |
| 1h Goal setup, and the Goals list | FR-10, FR-11 | `list_goals`, `check_goal`, `create_goal`, `update_goal`, `archive_goal`, `undo_goal_change` | FR-10 (#42, #43, #45); the fit badge and statuses in FR-11 (#56) |
| 1g, 1l Goal detail | FR-11, FR-12 | `forecast_goal`, `list_goals` | Built (#56): status, likely amount and 80% range, top-up, a chart of each month's range, what it accounts for, the short-history notice and the assumption line. The "what this accounts for" bullets are generic, not 1g's user-specific facts ("December spending usually runs about 15% higher for you") |
| 1b Coach-first summary, 1c Mosaic | — | — | Not built (see decision 2) |
| Connect an assistant (no mockup) | FR-19 | Read tools, plus goal and category writes with confirmation, over MCP | Demo |

## Decisions

1. **Sign-in: demo accounts.** Keep screen 1i. A few synthetic users get a name and email in `configs/web/demo_accounts.yaml`, so the generator's file is unchanged. One shared password comes from `SFC_DEMO_PASSWORD` and is checked against a scrypt hash (standard library). Real sign-in (v2) replaces only the login route. The password also gates the public demo (NFR-9). Hardening, since it's the only gate on a public URL:
   - **One-click "Continue as …" fails closed:** off unless a local-only setting turns it on (`SFC_QUICK_SIGNIN`, set by `sfc-web serve --dev`), so a forgotten variable can't open the hosted demo.
   - **Session cookie:** signed with `SFC_SESSION_SECRET` from the environment (NFR-3), `HttpOnly`, `Secure` (on unless a local setting turns it off), `SameSite=Lax`, expiring after 12 hours. It holds the `user_id` and a random session id, and it's the only place the app learns who the user is. Signing in clears the old session.
   - **CSRF:** `SameSite=Lax` keeps the cookie off cross-site posts (sign-in, chat, and later corrections), so no form token is needed while every state change is a same-site post.
   - **Rate limits:** sign-in attempts per client address, since everyone shares one password; chat messages per session.
2. **Dashboard layout: 1a.** 1b leads with an LLM-written summary, which would break the dashboard when the LLM is unavailable (NFR-6) and slow it past 2 s (NFR-5). 1b's summary can come later as a precomputed card.
3. **Essentials: the bare minimum to live on,** shown apart from everything else in the money-flow chart (owner, Oct 2, 2026).
   - **Default:** Housing, Utilities, Groceries, Insurance & Fees, Childcare & Education. Transportation and Health & Fitness fall under "Everything else".
   - **A setting, not a constant:** `SFC_ESSENTIALS` (a JSON list) changes the set without a code change.
   - **Checked at startup:** every name must be a category in the dataset's taxonomy, and Income can't be one; otherwise the app doesn't start.
   - **Display only:** no change to the taxonomy, the models or the tools.
4. **Responsive and dark mode from the start.** The mockups' palette becomes CSS tokens with a dark set; layouts hold down to 390 px. The PRD rules out a mobile app, not a responsive web app.
5. **Coach name** is a setting, defaulting to "Wren"; tests don't depend on it. Check the name doesn't collide with a financial product before a public launch.
6. **Coach LLM: Anthropic** (Technical Design open question). The key comes from the environment, never the image (NFR-3).
7. **Web framework:** FastAPI with server-rendered templates and htmx (Delivery Plan recommendation). Charts are server-rendered SVG.
8. **One MCP server for Wren and outside assistants, over HTTP only** (owner, Oct 2, 2026; FR-19, key scenario 6). The tools are served over MCP (Streamable HTTP) at `/mcp`, so the same question gets the same numbers from Wren and from, say, Claude Desktop.
   - **Identity comes only from a bearer token,** never from a tool argument. Tokens are signed with `SFC_SESSION_SECRET`, name one demo user and expire.
   - **Revocation is all or nothing.** A single token can't be revoked: it works until it expires, even after sign-out or a change of the demo password. Rotating `SFC_SESSION_SECRET` revokes every token and signs everyone out. Acceptable for a short demo on synthetic data; real sign-in (v2) needs per-token revocation. The Connect page is sent with `Cache-Control: no-store`, since it shows a token.
   - **Outside assistants** use a personal access token from the "Connect an assistant" page, valid for `SFC_MCP_TOKEN_DAYS` (default 7). The page gives ready-made setup for Claude Code and Claude Desktop (through `mcp-remote`).
   - **Wren** calls the same endpoint in-process, through the full HTTP stack, with a 5-minute token for the session's user, and takes its tool list from the server.
   - stdio isn't part of v1: desktop assistants connect over HTTP. It can be revisited if a concrete need comes up.

## Demo build (Oct 6, 2026)

A short-lived deployment for a presentation, up from Oct 3 and torn down on Oct 6. It takes shortcuts the Delivery Plan's CD stages will replace:

- **Hosting:** Azure Container Apps, one replica, HTTPS on the platform address, in its own resource group. Deleting the resource group removes everything. A short-lived shortcut, outside the Delivery Plan's option D, which stays open for staging.
- **Read-only data in the image.** The dataset, categorization predictions and demo accounts are built into a read-only SQLite file at image build time; no ingestion worker, no volume. Nothing a visitor does changes shared data.
- **Chat is per session, not per user.** Several visitors can sign in as the same demo user, and a question can contain anything, so conversation history is kept in memory by the session's random id: never keyed by `user_id`, never in the database. Two sessions as the same user don't see each other's chat (NFR-2), and a test checks it.
- **The MCP server runs in the web app's process** (decision 8). Wren and outside assistants call the tools through it; the dashboard calls the same functions directly. Identity comes only from a bearer token, every query is scoped at the data-access layer, and the isolation tests run. Moving the tool server into its own workload is milestone P1.
- **Chat cost:** a spending cap on the API key and a per-visitor rate limit.

## Gaps the mockups need that nothing provides yet

1. **Readable merchant names.** 1e and 1f show "Blue Bottle Coffee" over `SQ *BLUE BOTTLE #4321`, and #15's `list_review_items` returns a `merchant` display name. FR-3's normalizer makes matching keys, not names. Needs an owner: FR-4 or a small display-name step. The demo shows the normalizer's key, capitalized, above the raw text.
2. **Alternative categories.** The review panel offers 2–3 alternatives per item; #15 returns only `suggested_category`. Add the top few.
3. **Sources for every number (FR-16).** 1d links each number to a numbered source ("Savings forecast · 21 months"). Tools need to return source metadata and the coach a number-to-source mapping. P1, but cheaper to build into the first chat than to add later.
4. **Goal tools.** Closed by FR-10 and FR-11: `check_goal` gives 1h's "How it fits" facts, the fit badge and a probability for an unsaved goal; `forecast_goal` gives "Set aside $75 more a month" (`extra_per_month`) and the 80% interval; `create_goal`, `update_goal`, `archive_goal` and `undo_goal_change` write.
5. **What-if forecasts.** Two mockup answers forecast a hypothetical: the freelancer's "Spending $900 in November would lower your likely tax reserve … to about $5,800" and the family's "Even a December like last year's keeps you on track". `forecast_goal` forecasts only a saved goal on actual history, and the number must come from a tool (FR-14, NFR-1). Either add a scenario input (an extra expense in a month, or a month replaced by last year's) or drop those answers. Deferred by FR-11 (owner decision 7 on #50).
6. **Flag actions.** "I recognize this", "Not me — what now?", "Expected, all good" and the sensitivity control (FR-9). *Settled in FR-7 §8 (owner, Oct 3, 2026, on #30):* v1 shows flags without actions; actions and sensitivity come in v1.1 with FR-9, stored per user and keyed by a stable flag id (model version plus transaction id). "Not me" gives guidance only.
7. **Copy.** 1e says a correction updates "your goal forecast"; forecasts are precomputed nightly and net savings rarely depend on categories. Change the copy or define the behavior.
8. **The app's date.** Data ends Sep 30, 2026, so "today" comes from the dataset's `as_of`, not the clock.
9. **Missing screens:** "not available yet" and "coming next" panels, chat with the LLM unavailable (NFR-6), short histories, loading and error states.

## Open questions

- [ ] **What "usual" means** ("$150 more than your usual month"). May come from a model's baseline (FR-8's expected spend) rather than a fixed statistic. Until then, the dashboard shows the month's amount without a comparison.
- [ ] Owner for readable merchant names (gap 1).
- [x] Mutable data in a shared demo: neither a nightly reset nor a copy per session. Corrections and goal changes are keyed by browser session in SQLite files in the container, kept until it restarts or redeploys (owner, Oct 3, 2026; FR-5 and FR-6 design, §3; FR-10 design, option F).
