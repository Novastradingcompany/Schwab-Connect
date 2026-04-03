import datetime as dt
import time

from nova.external_services import ExternalServiceError


def fetch_price_history(symbol, days=365, *, get_client, schwab_json, cache, cache_ttl):
    cache_key = (str(symbol or "").upper().strip(), int(days or 365))
    now = time.time()
    cached = cache.get(cache_key)
    if cached and (now - cached["ts"]) <= cache_ttl:
        return cached["data"]

    client = get_client()
    end_date = dt.datetime.now(dt.timezone.utc)
    start_date = end_date - dt.timedelta(days=days)
    data = schwab_json(
        lambda: client.get_price_history(
            symbol,
            period_type=client.PriceHistory.PeriodType.YEAR,
            frequency_type=client.PriceHistory.FrequencyType.DAILY,
            frequency=1,
            start_datetime=start_date,
            end_datetime=end_date,
            need_extended_hours_data=False,
            need_previous_close=True,
        ),
        f"get_price_history:{symbol}",
    )
    cache[cache_key] = {"ts": now, "data": data}
    return data


def fetch_positions(
    *,
    get_client,
    schwab_json,
    cache,
    cache_ttl,
    parse_expiration,
    parse_option_symbol,
):
    now = time.time()
    cached = cache.get("value", [])
    if cached and (now - cache.get("ts", 0.0)) <= cache_ttl:
        return cached

    client = get_client()
    data = schwab_json(lambda: client.get_accounts(fields=["positions"]), "get_accounts_positions")
    positions = []
    for entry in data:
        account = entry.get("securitiesAccount", {})
        acct_number = account.get("accountNumber")
        acct_type = account.get("type")
        for pos in account.get("positions", []) or []:
            instrument = pos.get("instrument", {})
            asset_type = instrument.get("assetType", "")
            symbol = instrument.get("symbol") or instrument.get("underlyingSymbol")
            long_qty = float(pos.get("longQuantity", 0) or 0)
            short_qty = float(pos.get("shortQuantity", 0) or 0)
            qty = long_qty - short_qty
            if qty == 0:
                qty = float(pos.get("quantity", 0) or 0)

            avg_price = (
                pos.get("averagePrice")
                or pos.get("averageLongPrice")
                or pos.get("averageShortPrice")
            )
            market_value = pos.get("marketValue")
            pnl = (
                pos.get("openProfitLoss")
                or pos.get("longOpenProfitLoss")
                or pos.get("shortOpenProfitLoss")
            )
            pnl_pct = (
                pos.get("openProfitLossPercent")
                or pos.get("longOpenProfitLossPercent")
                or pos.get("shortOpenProfitLossPercent")
            )

            if pnl is None and avg_price is not None and market_value is not None and qty:
                multiplier = 100 if asset_type == "OPTION" else 1
                cost_basis = float(avg_price) * float(qty) * multiplier
                pnl = float(market_value) - cost_basis
                if cost_basis != 0:
                    pnl_pct = (pnl / abs(cost_basis)) * 100

            if str(asset_type).upper() in {"EQUITY", "ETF", "MUTUAL_FUND", "STOCK"}:
                if avg_price is None and pnl_pct is None:
                    pnl = None

            expiration = parse_expiration(instrument.get("expirationDate") or instrument.get("maturityDate"))
            if expiration is None and asset_type == "OPTION":
                parsed_option = parse_option_symbol(symbol)
                expiration = parsed_option.get("expiry")
            dte = None
            if expiration:
                dte = (expiration - dt.date.today()).days

            positions.append({
                "accountNumber": acct_number,
                "accountType": acct_type,
                "symbol": symbol,
                "assetType": asset_type,
                "instrument": instrument,
                "qty": qty,
                "avgPrice": avg_price,
                "marketValue": market_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "expiration": expiration.isoformat() if expiration else None,
                "dte": dte,
            })

    cache["ts"] = now
    cache["value"] = positions
    return positions


def normalize_quote(raw, *, safe_float):
    if not isinstance(raw, dict):
        return {}
    quote = raw.get("quote", raw)
    ref = raw.get("reference", {})
    return {
        "symbol": raw.get("symbol") or ref.get("symbol"),
        "last": safe_float(quote.get("lastPrice") or quote.get("last") or quote.get("mark")),
        "mark": safe_float(quote.get("mark")),
        "bid": safe_float(quote.get("bidPrice") or quote.get("bid")),
        "ask": safe_float(quote.get("askPrice") or quote.get("ask")),
        "delta": safe_float(quote.get("delta")),
        "theta": safe_float(quote.get("theta")),
        "gamma": safe_float(quote.get("gamma")),
        "vega": safe_float(quote.get("vega")),
        "iv": safe_float(quote.get("volatility") or quote.get("impliedVolatility")),
    }


def fetch_accounts_summary(*, get_client, schwab_json, cache, cache_ttl):
    now = time.time()
    cached = cache.get("value", [])
    if cached and (now - cache.get("ts", 0.0)) <= cache_ttl:
        return cached

    client = get_client()
    data = schwab_json(lambda: client.get_accounts(), "get_accounts_summary")
    cache["ts"] = now
    cache["value"] = data
    return data


def fetch_quotes(symbols, *, get_client, schwab_json, cache, cache_ttl, normalize_quote):
    if not symbols:
        return {}
    now = time.time()
    cached = {}
    missing = []
    for sym in symbols:
        entry = cache.get(sym)
        if entry and now - entry["ts"] <= cache_ttl:
            cached[sym] = entry["quote"]
        else:
            missing.append(sym)

    if not missing:
        return cached

    client = get_client()
    data = schwab_json(lambda: client.get_quotes(missing), "get_quotes")
    quotes = {}
    for symbol, payload in data.items():
        quotes[symbol] = normalize_quote(payload)
        cache[symbol] = {"ts": now, "quote": quotes[symbol]}
    quotes.update(cached)
    return quotes


def fetch_option_chain(symbol, *, get_client, schwab_response, cache, cache_ttl, error_cache):
    now = time.time()
    cached = cache.get(symbol)
    if cached and now - cached["ts"] <= cache_ttl:
        return cached["data"]
    err = error_cache.get(symbol)
    if err and now - err["ts"] <= 120:
        raise RuntimeError(f"Recent Schwab error for {symbol}; retry in a bit.")
    client = get_client()
    response = schwab_response(
        lambda: client.get_option_chain(symbol, include_underlying_quote=True),
        "get_option_chain",
    )
    if response.status_code == 502:
        body = ""
        try:
            body = response.text or ""
        except Exception:
            body = ""
        too_big = "TooBigBody" in body or "Body buffer overflow" in body
        if too_big:
            response = schwab_response(
                lambda: client.get_option_chain(
                    symbol,
                    include_underlying_quote=True,
                    strike_count=30,
                ),
                "get_option_chain_limited",
            )
    try:
        data = response.json()
    except Exception as exc:
        raise ExternalServiceError(
            "Schwab",
            "get_option_chain",
            "Schwab returned an invalid option chain response. Try again.",
            original=exc,
        ) from exc
    cache[symbol] = {"ts": now, "data": data}
    return data


def get_chain_data(
    symbol,
    *,
    fetch_option_chain,
    fetch_quotes,
    safe_float,
    flatten_option_map,
):
    chain = fetch_option_chain(symbol)
    spot_price = None
    try:
        live_quote = fetch_quotes([symbol]).get(symbol, {})
        spot_price = safe_float(
            live_quote.get("last")
            or live_quote.get("mark")
            or live_quote.get("bid")
            or live_quote.get("ask")
        )
    except Exception:
        spot_price = None
    if spot_price is None:
        spot_price = chain.get("underlyingPrice") or chain.get("underlying", {}).get("last")
        if spot_price is not None:
            spot_price = float(spot_price)
    call_map = flatten_option_map(chain.get("callExpDateMap", {}))
    put_map = flatten_option_map(chain.get("putExpDateMap", {}))

    expirations = []
    exp_dates = sorted(set(call_map.keys()) | set(put_map.keys()))
    for exp_date in exp_dates:
        dte = None
        if exp_date in call_map and call_map[exp_date]["dte"] is not None:
            dte = call_map[exp_date]["dte"]
        if exp_date in put_map and put_map[exp_date]["dte"] is not None:
            dte = put_map[exp_date]["dte"]
        expirations.append({"date": exp_date, "dte": dte})

    return spot_price, call_map, put_map, expirations


def get_account_hashes(*, get_client, schwab_json, cache, cache_ttl):
    now = time.time()
    cached = cache.get("value", {})
    if cached and (now - cache.get("ts", 0.0)) <= cache_ttl:
        return cached

    client = get_client()
    data = schwab_json(lambda: client.get_account_numbers(), "get_account_numbers")
    hashes = {row.get("accountNumber"): row.get("hashValue") for row in data}
    cache["ts"] = now
    cache["value"] = hashes
    return hashes


def fetch_account_balance_totals(*, fetch_accounts_summary, safe_float):
    data = fetch_accounts_summary()
    totals = {"cash": 0.0, "liquidation": 0.0, "equity": 0.0}
    for entry in data or []:
        acct = (entry or {}).get("securitiesAccount", {}) or {}
        bal = acct.get("currentBalances", {}) or {}
        totals["cash"] += safe_float(bal.get("cashBalance")) or 0.0
        totals["liquidation"] += safe_float(bal.get("liquidationValue")) or 0.0
        totals["equity"] += safe_float(bal.get("equity")) or 0.0
    return totals
