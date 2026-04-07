# Nova Schwab Manual

Last updated: 2026-04-06

## 1) What This App Does

Nova Schwab is a web dashboard for:
- Viewing Schwab account, cash, position, and risk data
- Scanning options strategies from Schwab chains
- Reviewing open put spreads with live scenario columns
- Planning trades in the credit spread simulator
- Reviewing spread setups on the chart page with spot and level guides
- Managing a watchlist, movers workflow, and draft tickets
- Monitoring alerts with rule-based exit checks
- Logging trades in a journal and reviewing closed-trade stats
- Using Nova for chain analysis, filter suggestions, and chat

This app is advisory-only. It does not place orders automatically.

---

## 2) Main Navigation

Top row pages:
- `Overview`
- `Options Chain`
- `Open Options`
- `Positions`
- `Alerts`
- `Watchlist`
- `Movers Agent`
- `Nova`

Second row pages:
- `Spread Sim`
- `Risk`
- `Summary`
- `Journal`
- `Stats`
- `Tools`
- `Manual`
- `Tickets`

---

## 3) Access and Login

1. Open the dashboard URL.
2. Log in with the app username/password.
3. Use the two-row top navigation to move between workflows.

If login is failing, verify:
- `AUTH_USERNAME`
- `AUTH_PASSWORD_HASH` or `AUTH_PASSWORD`

---

## 4) Schwab Token and Connectivity

The app needs a valid Schwab OAuth token.

### Standard refresh flow

1. Run:
   ```powershell
   python connect.py
   ```
2. Complete Schwab auth.
3. Paste the redirect URL or code immediately when prompted.
4. Confirm `token.json` is updated locally.
5. Update deployment env with `TOKEN_JSON_B64` or `TOKEN_JSON`.
6. Redeploy if you need the new token to survive restart.

### Important notes

- Authorization codes expire quickly.
- Runtime refresh from the app updates the current running instance only.
- To persist token state across restart/redeploy, update Render env `TOKEN_JSON_B64`.
- If both `TOKEN_JSON` and `TOKEN_JSON_B64` exist, the app uses the newer payload.

---

## 5) Overview

Use `Overview` as the landing page for account-level snapshot data. If Schwab connectivity is degraded, check `Tools` next.

---

## 6) Options Chain (`/options`)

This is the main strategy scanner page.

### Inputs

- `Symbol`
- `Expiration`
- `Strategy`: `Bull Put`, `Bear Call`, or `Iron Condor`
- `Max Width`
- `Min POP (%)`
- `Contracts`
- `Pricing mode`: `Mid`, `Natural (Bid/Ask)`, or `Custom limit`
- `Custom limit (per spread)`
- `Cash Balance`
- `Max Loss`
- `Show raw results`
- `Auto-refresh (60s)`

### What the page does

- Pulls the Schwab chain for the selected expiry
- Computes filtered trade candidates
- Shows a research snapshot for the symbol
- Shows market regime context
- Supports Nova actions for the full chain or a selected row
- Shows a workflow-first layout with `Setup`, `Research`, `Nova`, and `Results` jump links
- Uses a live quote for the displayed spot when available instead of relying only on the chain payload

### Nova actions on this page

- `Ask Nova about this chain`
- `Suggest filters`
- `Explain selected trade`
- `Create trade ticket`

### Spread simulator handoff

For `Bull Put` and `Bear Call`, each row includes an `Open` link in the `Sim` column. This passes the selected trade into `Spread Sim`.

### Pricing notes

- `Mid` uses midpoint pricing
- `Natural` uses bid/ask style pricing
- `Custom limit` lets scanner math match your intended ticket credit exactly

### Error handling notes

- The page now tries to keep scan results visible even if research context calls fail separately.
- If Schwab returns malformed chain rows, the scanner degrades to fewer or no results instead of crashing the page.

---

## 7) Open Options (`/options-open`)

Use this page to review live bull put spreads already open in the account.

### Inputs

- `Interval ($)`
- `Range (+/- steps)`
- `Scenario Pricing`
  - `Model (Black-Scholes)`
  - `Expiration Payoff`

### Output

The table shows:
- underlying and expiry
- short and long put symbols
- strikes and width
- quantity
- entry credit
- current spread mark
- underlying last
- unrealized P/L and P/L %
- breakeven
- scenario P/L columns around the short strike

---

## 8) Positions (`/positions`)

Use this page for a live cross-account position table.

Columns include:
- account
- symbol
- asset type
- quantity
- average price
- market value
- P/L and P/L %
- DTE
- last, mark, delta, theta, IV
- action / why / note

This is the broadest open-position view. `Alerts` uses similar data but only shows positions that meet active exit rules.

---

## 9) Alerts (`/alerts`)

This page defines your monitoring rules and shows positions needing attention.

### Rule settings

- `Profit Target (% of credit)`
- `Max Loss (% of max loss)`
- `Time-based Exit (DTE)`
- `Nova judgment notes`
- `Polling interval (minutes)`
- `Monitor only during market hours`
- `Market open`
- `Market close`
- `Monitor weekdays only`
- `Market holidays`
- `Nova model`
- `Nova role`
- `Movers lookback`
- `Movers count`
- `Include current positions in movers universe`
- `Include S&P 500 universe`
- `Movers scan timeout`
- `Movers universe`

### Output

The `Action Required` table shows only rows that meet alert conditions, with:
- position metrics
- greeks / IV / DTE
- action
- why
- note

If email is not configured, the page shows a warning.

---

## 10) Watchlist (`/watchlist`)

Use the watchlist for fast idea triage.

### Supported actions

- Add a symbol
- Remove a symbol
- Jump straight into `Options Chain` with `Scan`
- Run `Scan All` across the current watchlist

### Watchlist scan inputs

- `Strategy`
- `Max Width`
- `Min POP (%)`
- `Contracts`
- `Cash Balance`
- `Max Loss`
- `Expiration` (optional)
- `Show raw results`

### Watchlist scan output

Each row shows:
- symbol
- expiry
- spot
- trades found
- top trade
- error, if any

---

## 11) Movers Agent (`/movers-agent`)

This is a ranking workflow separate from the options scanner.

### Purpose

- Scan a curated optionable universe
- Rank names by recent movement
- Review scan errors separately
- Add selected symbols to the watchlist
- Reuse cached market-history work so repeated scans complete faster
- Use bounded concurrency so the scan can cover more of the universe before timeout

### Typical inputs

- lookback window
- result count
- max scan time

### Typical outputs

- snapshot metadata
- ranked results
- timeout/scanned-count status
- scan error section

Primary files used by this workflow:
- `optionable_universe.json`
- `movers_snapshot.json`

---

## 12) Spread Sim (`/spread-sim`)

This page was added after the original manual and is now a core workflow.

Use it to model a `Bull Put` or `Bear Call` vertical before entry.

### Inputs

- `Symbol`
- `Spread Type`
- `Preset`
- `Stock Value (Now)`
- `Short Strike`
- `Long Strike`
- `Net Credit (per spread)`
- `Fill Mode`
- `Slippage (Natural, per spread)`
- `Contracts`
- `DTE (model)`
- `IV Short`
- `IV Long`
- `Risk-Free Rate`
- `Price From`
- `Price To`
- `Price Step`

### Presets

Presets apply default scenario values and can also reset the simulation bounds. Use `Apply Preset Defaults` to load those values into the form.

### Main actions

- `Run Simulation`
- `Open Chart`
- `Export Setup to Watchlist`
- `Export Setup to Ticket`
- `Prefill Journal Entry`
- `Save Simulation File`
- `Clear`
- `Print`
- `Load Simulation File`

### What the page shows

- Trade Summary
- return on risk
- risk/reward ratio
- breakeven
- current spot model P/L
- quoted credit vs entry credit used
- POP score
- current spread value
- model inputs
- spot-to-short-strike buffer

### Analysis sections

- `How This Is Calculated`
- `Key Levels`
- `Profit Target Markers`
- `Stop And Defense Lines`
- `Vertical Scenario Ladder`

### Chart handoff

Use `Open Chart` to send the current spread setup into the chart page.

The chart page carries over:
- symbol
- spread type
- current spot
- short strike
- long strike
- credit and fill assumptions
- contracts and DTE

The chart page shows:
- underlying price history
- horizontal guide lines for `Spot`, `Profit Limit`, `Breakeven`, and `Loss Limit`
- hover crosshairs with the selected close plus a live cursor price readout on the Y axis
- print support
- `Back To Spread Sim`
- `Clear`

If the latest chart close is materially different from the simulator spot, the page shows a warning with both values and the percent gap. Treat that as a data sanity check before relying on the chart for decision support.

### Save/load notes

- Saved files are JSON
- The page validates file type and schema version on load
- Older versions can still load with a warning when supported

### Handoff notes

- `Options Chain` can open a selected spread directly in this page
- `Open Chart` sends the active setup into the chart page
- `Export Setup to Ticket` creates a draft ticket
- `Export Setup to Watchlist` pushes the setup into watchlist storage

---

## 13) Risk Dashboard (`/risk`)

Use this page for portfolio exposure review.

It shows:
- total account value
- invested market value
- cash and cash investments
- total P/L
- count of options with DTE <= 7
- exposure by asset type
- top symbols by exposure
- largest positions

This is the fastest page for concentration checks.

---

## 14) Summary (`/summary`)

Purpose:
- show open + closed P/L grouped by symbol and period
- split results into `Stocks`, `Options`, `Cash`, and `Other Assets`
- show subtotals plus a combined total
- start with a dashboard-style overview before the detailed tables
- include section jump links so you can move directly to each asset block

### Filters

- `Period`: `week`, `month`, `quarter`, `year`
- `Status`: `all`, `open`, `closed`
- `Sort`: `profit`, `loss`, `symbol`
- `Year`
- `Month`

### Notes

- Open rows come from current positions and balances.
- Closed rows come from Schwab transactions.
- The page shows summary cache freshness when available.
- If option totals look wrong, compare against `Transaction Reconcile Debug`.
- The top overview cards reflect the current filters, not all-time totals.

---

## 15) Transaction Reconcile Debug (`/debug/txn-reconcile`)

Use this page when summary totals do not match Schwab transaction history.

### Inputs

- symbol filter
- asset filter
- start date
- end date
- include mode

### Outputs

- summary totals by cashflow method
- per-symbol totals
- raw transaction-level rows
- identifiers and field breakdown used by the summary logic

---

## 16) Journal (`/journal`)

Use the journal to record trade quality, execution discipline, and outcome.

### Supported workflows

- add a trade
- edit an existing trade
- update status to `open`, `closed`, or `expired`
- record realized P/L and outcome
- delete a trade

### Core fields

- symbol
- strategy
- status
- entry / expiry / exit dates
- max loss
- target profit
- realized P/L
- outcome
- thesis
- notes

### Entry gate checklist

Open trades require the checklist to pass:
- setup matches proven criteria
- max loss within cap
- POP / quality threshold met
- position size valid
- no rule violations

The logged trades table also shows whether each trade passed or failed the gate.

---

## 17) Stats (`/journal/stats`)

This page summarizes closed-trade performance from the journal.

It includes:
- closed trades count
- win rate
- net P/L
- average return
- max drawdown
- breakdown by strategy
- breakdown by month
- closed trades detail by symbol/date

Use this page together with `Journal` to evaluate repeatability.

---

## 18) Tickets (`/tickets`)

Tickets are draft trade previews. Order submission is not wired yet.

### What you can do

- review draft tickets created from `Options Chain` or `Spread Sim`
- see available account numbers
- inspect top trade summary and legs
- `Confirm`
- `Send`
- `Clear`

Treat this page as a staging area, not an execution system.

---

## 19) Nova Chat (`/nova`)

This is the general chat page for advisory-only guidance.

### Actions

- ask free-form questions
- run `Find Movers`
- clear chat history

`Find Movers` uses the movers-related settings from `Alerts`.

### Page layout

The page now opens with:
- session status cards
- `Chat Log`, `Quick Actions`, and `Compose` jump links
- a dedicated quick-action block for movers scans
- a separate compose area for free-form prompts

Use `Find Movers` when you want Nova to start from the current movers universe. Use the compose box when you already know the symbol, setup, or question you want to discuss.

---

## 20) Tools (`/tools`)

Use this page for system health, exports, and operational actions.

### Sections

- `Monitor Status`
- `Timezone`
- `Exports`
- `Schwab Status Check`
- `Error Log`

### Available actions

- save app timezone
- export alerts CSV
- export tickets CSV
- export summary CSV
- open transaction reconcile debug
- show `TOKEN_JSON_B64`
- copy `TOKEN_JSON_B64`
- run Schwab status check
- send test email
- clear error log

### Schwab token workflow

Use `Tools` to follow the token instructions and generate `TOKEN_JSON_B64`, but run the actual OAuth refresh locally.

Recommended flow:
1. Run `python connect.py`
2. Complete the Schwab browser login and approval flow
3. Confirm local `token.json` is updated
4. Run `python scripts/render_token_b64.py` or use `Tools` -> `Show TOKEN_JSON_B64`
5. Paste that value into Render env `TOKEN_JSON_B64`
6. Redeploy

Do not rely on an in-app token refresh button for Schwab OAuth.

Use `Tools` first when data looks stale, alerts appear quiet, or token state is unclear.

---

## 21) Persistent Data on Render

Without persistent storage, JSON-backed app data resets on redeploy.

### Recommended setup

1. Add a persistent disk in Render.
2. Mount it at `/var/data`.
3. Set env var `DATA_DIR=/var/data`.
4. Redeploy.

### Common persisted files

- `settings.json`
- `alerts.json`
- `alerts_state.json`
- `watchlist.json`
- `tickets.json`
- `trade_journal.json`
- `monitor_state.json`
- `error_log.json`
- `movers_snapshot.json`

---

## 22) Common Troubleshooting

### A) `refresh_token_authentication_error` or `unsupported_token_type`

Cause:
- expired or invalid token payload

Fix:
1. Run `python connect.py`
2. generate a fresh token
3. use `Tools` -> `Show TOKEN_JSON_B64`
4. update Render env
5. redeploy

### B) Scanner credit does not match Schwab ticket

Cause:
- different pricing assumption

Fix:
- use `Pricing mode = Custom limit` in `Options Chain`
- or match `Fill Mode` and slippage assumptions in `Spread Sim`

### C) Movers results are too small

Cause:
- timeout reached early

Fix:
- increase movers scan timeout
- reduce universe scope
- review scan errors in `Movers Agent`

Notes:
- The movers workflows are faster than before because history fetches now run concurrently with limits.
- If results are still thin, the bottleneck is usually Schwab endpoint quality or universe size, not local page rendering.

### D) Summary totals look wrong

Cause:
- Schwab transaction fields can vary by asset and activity type

Fix:
- open `Transaction Reconcile Debug`
- compare raw cashflow fields against summary math

### E) Runtime token refresh worked, but app lost it after restart

Cause:
- token was refreshed in memory only

Fix:
- run `python connect.py` locally first
- then copy the `TOKEN_JSON_B64` value from `Tools`
- update Render env
- redeploy

### F) Option chain spot looks different from another Schwab view

Cause:
- the chain payload and the live quote feed may not match exactly at the same moment

Fix:
- the page now prefers the live quote when available
- if the difference still matters, refresh once and compare against `Spread Chart` or another live quote page

---

## 23) File Quick Reference

- Main app: `app.py`
- Token helper: `connect.py`
- Manual source: `docs/manual.md`
- Layout/nav: `templates/layout.html`
- Options scanner: `templates/options.html`
- Open options: `templates/options_open.html`
- Positions: `templates/positions.html`
- Alerts: `templates/alerts.html`
- Watchlist: `templates/watchlist.html`
- Movers page: `templates/movers_agent.html`
- Spread simulator: `templates/spread_sim.html`
- Spread chart page: `templates/chart.html`
- Risk dashboard: `templates/risk.html`
- Summary page: `templates/summary.html`
- Journal page: `templates/trade_journal.html`
- Stats page: `templates/trade_stats.html`
- Tickets page: `templates/tickets.html`
- Tools page: `templates/tools.html`
- Reconcile debug: `templates/txn_reconcile.html`
- Universe file: `optionable_universe.json`
- Movers snapshot: `movers_snapshot.json`

---

## 24) Recommended Weekly Workflow

1. Run `Movers Agent` and add selected names to the watchlist.
2. Review names in `Watchlist` and open `Options Chain` scans.
3. For promising spreads, open `Spread Sim` from the results table.
4. Use the simulator to review target markers, defense lines, and ladder scenarios.
5. Export the best setup to `Tickets`.
6. Log the trade in `Journal`.
7. Monitor live positions from `Alerts`, `Open Options`, `Positions`, and `Risk`.
8. Review `Summary` and `Stats` at the end of the week or month.
