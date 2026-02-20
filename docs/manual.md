# Nova Schwab Manual

## 1) What This App Does

Nova Schwab is a web dashboard for:
- Viewing Schwab account and positions data
- Scanning options strategies (bull put, bear call, iron condor)
- Managing a watchlist and draft tickets
- Running a separate Movers Agent and adding picks to watchlist
- Using Nova (LLM assistant) to explain or analyze scan output

This app is advisory-only. It does not place orders automatically.

---

## 2) Access and Login

1. Open the dashboard URL.
2. Log in with the app username/password.
3. Use the top navigation to move between pages.

If login is failing, verify `AUTH_USERNAME` and `AUTH_PASSWORD_HASH` (or `AUTH_PASSWORD`) in environment config.

---

## 3) Token and Schwab Connectivity

The app needs a valid Schwab OAuth token.

### Standard token refresh flow

1. Run:
   ```powershell
   python connect.py
   ```
2. Complete Schwab auth.
3. Paste redirect URL or code when prompted.
4. Confirm `token.json` is updated.
5. Update Render env (`TOKEN_JSON_B64` or `TOKEN_JSON`) and redeploy.

### Important notes

- Authorization codes expire quickly. Paste immediately.
- Keep token secrets private.
- If both `TOKEN_JSON` and `TOKEN_JSON_B64` are present, the app selects the newer payload.

---

## 4) Options Chain Scanner (`/options`)

This is the main strategy scanner page.

### Required inputs

- `Symbol`
- `Expiration`
- `Strategy`: `Bull Put`, `Bear Call`, or `Iron Condor`
- `Contracts`

### Risk/filter inputs

- `Max Width`
- `Min POP (%)`
- `Cash Balance`
- `Max Loss`
- `Show raw results` (ignores filter cutoffs)

### New pricing controls

- `Pricing mode`
  - `Mid`: uses midpoint for each leg
  - `Natural (Bid/Ask)`: sell at bid, buy at ask (more conservative)
  - `Custom limit`: uses your entered net credit per spread
- `Custom limit (per spread)`
  - Enter format like `0.46` (per 1 spread)
  - Internally converted to dollars per spread (`0.46 * 100 = 46`)

### Credit columns in results

- `Credit (Realistic)`: active pricing basis from selected mode
- `Credit (Mid $)`: per-spread credit at midpoint
- `Credit (Natural $)`: per-spread credit at natural fill
- `Total Credit ($)`: active per-spread credit times contracts

Use `Custom limit` when you want scanner math to match your Schwab ticket limit exactly.

---

## 5) Nova Actions on Options Page

From the scan results table:
- Select a trade row
- Use one of:
  - `Ask Nova about this chain`
  - `Suggest filters`
  - `Explain selected trade`
  - `Create trade ticket`

Nova explanation uses the selected row’s numeric fields (`Total Credit`, `Max Loss`, `Contracts`, `Breakeven`) plus current pricing mode context.

---

## 6) Movers Agent (`/movers-agent`)

This is a separate workflow from the options scanner.

### Purpose

- Rank symbols from a curated optionable universe
- Return top names
- Let you check symbols and add them to watchlist

### Inputs

- `Lookback (trading days)`
- `Return count`
- `Max scan time (seconds)`

### Outputs

- Snapshot metadata (time, scanned count, timeout status)
- Ranked table with checkbox per symbol
- `Add Checked To Watchlist` button

Universe source file:
- `optionable_universe.json`

Snapshot persistence file:
- `movers_snapshot.json`

---

## 7) Watchlist (`/watchlist`)

Supports:
- Add/remove symbols
- Scan all watchlist symbols with selected strategy settings
- Jump from chip to options scan page

---

## 8) Alerts (`/alerts`)

Configures:
- Profit target, max loss, DTE exit rules
- Polling interval and market-hours controls
- Nova model/role settings
- Movers settings used in Nova chat context

---

## 9) Tools (`/tools`)

Operational page for:
- Token status checks
- Manual refresh call path
- Other system utilities and diagnostics

Use this page first when Schwab data appears stale or unavailable.

---

## 10) Common Troubleshooting

### A) `refresh_token_authentication_error` or `unsupported_token_type`

Cause:
- Expired/invalid token payload

Fix:
1. Run `python connect.py` and generate fresh token
2. Update token env var in deployment
3. Redeploy

### B) `502` / worker timeout on heavy scans

Cause:
- Request path doing too much work in one request

Fix:
- Use Movers Agent workflow for ranking, not full-universe request scans
- Reduce scan timeouts / universe size if needed

### C) Scanner credit doesn’t match Schwab ticket

Cause:
- Different pricing assumption

Fix:
- Use `Pricing mode = Custom limit` and enter the same limit used in Schwab ticket

### D) Very small list from movers results

Cause:
- Timeout reached early

Fix:
- Increase `Max scan time (seconds)` on Movers Agent

---

## 11) File Quick Reference

- Main app: `app.py`
- Options templates: `templates/options.html`
- Movers page template: `templates/movers_agent.html`
- Layout/nav: `templates/layout.html`
- Styles: `static/styles.css`
- Universe file: `optionable_universe.json`
- Snapshot file: `movers_snapshot.json`
- Settings storage: `settings.json`
- Watchlist storage: `watchlist.json`
- Token helper: `connect.py`

---

## 12) Recommended Weekly Workflow

1. Open `Movers Agent`, run weekly scan, add selected names to watchlist.
2. Open `Options Chain` for each symbol and scan spreads.
3. Set pricing mode to match execution intent (`Custom limit` if ticket-driven).
4. Use Nova explain/analyze for final review.
5. Create ticket drafts.

