# Smart Financial Coach — Web App UI

Oct 2, 2026 · Owner: @Sidd · Status: **Accepted** (decisions below) · Branch: `design/mockup`

## Summary

Hi-fi mockups for the v1 web app are in [`mockups/`](mockups/). This note maps each screen to its requirement, the tools behind it and the milestone that ships it, records the UI decisions, and lists what the mockups need that no design or tool provides yet.

- **Build to the mockups, in the Delivery Plan's order.** Panels ship in a "not available yet" or "coming next" state first and switch to real numbers in the PR that promotes the model behind them (the Delivery Plan's sync rule).
- **Demo for Oct 6, 2026:** sign-in, Overview, Transactions and coach chat run on real data; Worth a look and Goals link to their mockups until FR-7/8 and FR-10–12 land. See [Demo build](#demo-build-oct-6-2026).
- **Open items** are in [Gaps](#gaps-the-mockups-need-that-nothing-provides-yet) and [Open questions](#open-questions).

## Mockups

Open `mockups/Smart Financial Coach - Light & Dark.dc.html` from a local server (`python -m http.server` in that folder); the files load React from unpkg. The light and dark files hold screens 1a–1l and 2a–2l; `Sidebar*.dc.html` is the shared nav and `support.js` the viewer runtime.

- Data is mocked per persona with the repo's 13 categories, through Sep 30, 2026. Persona is a design-time switch only; customers never see persona labels.
- Visual direction: cream grounds, apricot accent, sage for good news, amber for "worth a look"; Bricolage Grotesque and Figtree.

## Screens

| Screen | Requirement | Tools | Ships in |
| --- | --- | --- | --- |
| 1i Sign in | FR-18 | Session | Demo; P2 |
| 1a Overview (classic grid), 1j mobile | FR-17 | `get_spending_summary`, `get_transactions`, `list_goals` | Demo; P2 (flags and goals panels "coming next") |
| 1e Transactions list | FR-3, FR-17 | `get_transactions` with confidence | Demo, read-only; P2 |
| 1e "Not sure?" review panel | FR-5, FR-6 (P1) | #15: `list_review_items`, `resolve_review_item`, `correct_category`, `undo_correction` | After #15 milestone 3 |
| 1d, 1k Coach chat with sources | FR-13–16 | All read tools | Demo; P3 |
| 1f Worth a look | FR-7, FR-8, FR-9 | `detect_anomalies` | When FR-7/8 promote |
| 1g, 1l Goal detail; 1h Goal setup | FR-10–12 | `forecast_goal`, `list_goals`, a goal-write tool | When FR-10–12 land |
| 1b Coach-first summary, 1c Mosaic | — | — | Not built (see decision 2) |

## Decisions

1. **Sign-in: demo accounts.** Keep screen 1i. A seed step gives a few synthetic users a name and email in their own table, so the generator's file is unchanged. One shared password comes from `SFC_DEMO_PASSWORD` and is checked against a scrypt hash (standard library). A signed session cookie holds the `user_id`, and it is the only place the app learns who the user is. One-click "Continue as …" buttons are on locally and off in the hosted demo. Real sign-in (v2) replaces only the login route. The password also gates the public demo (NFR-9), with rate limits on sign-in and chat.
2. **Dashboard layout: 1a.** 1b leads with an LLM-written summary, which would break the dashboard when the LLM is unavailable (NFR-6) and slow it past 2 s (NFR-5). 1b's summary can come later as a precomputed card.
3. **Essentials grouping** for the money-flow chart: Housing, Utilities, Groceries, Insurance & Fees, Childcare & Education. Everything else is "Everything else". This is a display grouping, not a taxonomy change.
4. **Responsive and dark mode from the start.** The mockups' palette becomes CSS tokens with a dark set; layouts hold down to 390 px. The PRD rules out a mobile app, not a responsive web app.
5. **Coach name** is a setting, defaulting to "Wren"; tests don't depend on it. Check the name doesn't collide with a financial product before a public launch.
6. **Coach LLM: Anthropic** (Technical Design open question). The key comes from the environment, never the image (NFR-3).
7. **Web framework:** FastAPI with server-rendered templates and htmx (Delivery Plan recommendation). Charts are server-rendered SVG.

## Demo build (Oct 6, 2026)

A short-lived deployment for a presentation, up from Oct 3 and torn down on Oct 6. It takes shortcuts the Delivery Plan's CD stages will replace:

- **Hosting:** Azure Container Apps, one replica, HTTPS on the platform address. Deleting the resource group removes everything.
- **Read-only data in the image.** The dataset, categorization predictions and demo accounts are built into a read-only SQLite file at image build time; no ingestion worker, no volume. Nothing a visitor does changes shared data, so shared demo accounts are safe.
- **Tools in-process.** The tool functions run inside the web app rather than a separate tool server, but identity still comes only from the session, every query is scoped at the data-access layer, and the isolation tests run. Splitting the tool server out is milestone P1.
- **Chat cost:** a spending cap on the API key and a per-visitor rate limit.

## Gaps the mockups need that nothing provides yet

1. **Readable merchant names.** 1e and 1f show "Blue Bottle Coffee" over `SQ *BLUE BOTTLE #4321`, and #15's `list_review_items` returns a `merchant` display name. FR-3's normalizer makes matching keys, not names. Needs an owner: FR-4 or a small display-name step. The demo shows the raw text.
2. **Alternative categories.** The review panel offers 2–3 alternatives per item; #15 returns only `suggested_category`. Add the top few.
3. **Sources for every number (FR-16).** 1d links each number to a numbered source ("Savings forecast · 21 months"). Tools need to return source metadata and the coach a number-to-source mapping. P1, but cheaper to build into the first chat than to add later.
4. **Goal tools.** 1h's live "How it fits" needs a forecast for an unsaved goal (`forecast_goal` takes a `goal_id`). "Set aside $75 more a month" must come from the tool, not LLM arithmetic. Creating and editing goals needs a write tool. Fix the interval at 80% in the contract.
5. **Flag actions.** "I recognize this", "Not me — what now?", "Expected, all good" and the sensitivity control (FR-9) belong in the FR-7/8 design.
6. **Copy.** 1e says a correction updates "your goal forecast"; forecasts are precomputed nightly and net savings rarely depend on categories. Change the copy or define the behavior.
7. **The app's date.** Data ends Sep 30, 2026, so "today" comes from the dataset's `as_of`, not the clock.
8. **Missing screens:** "not available yet" and "coming next" panels, chat with the LLM unavailable (NFR-6), short histories, loading and error states.

## Open questions

- [ ] **What "usual" means** ("$150 more than your usual month"). May come from a model's baseline (FR-8's expected spend) rather than a fixed statistic. Until then, the dashboard shows the month's amount without a comparison.
- [ ] Owner for readable merchant names (gap 1).
- [ ] Mutable data in a shared demo (after corrections and goal writes land): nightly reset from the generator spec, or a copy per visitor session.
