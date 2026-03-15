import datetime as dt
import base64
import json
import logging
import os
import threading
import time
import smtplib
from functools import wraps
from email.message import EmailMessage
import csv
import io
import re
import math
import shutil
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from openai import OpenAI
import httpx
from schwab.auth import client_from_token_file
from werkzeug.security import check_password_hash, generate_password_hash

from nova.nova_rules import get_max_loss_threshold, NOVA_RULES
from nova.strategies.bull_put import scan_bull_put
from nova.strategies.bear_call import scan_bear_call
from nova.strategies.iron_condor import scan_iron_condor


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or os.urandom(24)
_CLIENT = None


def _data_file(name):
    data_dir = (os.getenv("DATA_DIR") or "").strip()
    if not data_dir:
        return name
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, name)
    # One-time migration from repo-root file to persistent disk file.
    if not os.path.exists(path) and os.path.exists(name):
        try:
            shutil.copyfile(name, path)
        except Exception:
            pass
    return path


SETTINGS_PATH = _data_file("settings.json")
NOVA_CHAT = []
_OPENAI_CLIENT = None
NOVA_ERROR = None
ALERTS_PATH = _data_file("alerts.json")
ALERTS_STATE_PATH = _data_file("alerts_state.json")
ALERTS = []
_ALERTS_LOCK = threading.Lock()
_MONITOR_THREAD = None
_MONITOR_LOCK = threading.Lock()
NOVA_OPTIONS_RESPONSE = None
NOVA_OPTIONS_ERROR = None
WATCHLIST_PATH = _data_file("watchlist.json")
WATCHLIST_NOTES_PATH = _data_file("watchlist_notes.json")
TICKETS_PATH = _data_file("tickets.json")
TRADE_JOURNAL_PATH = _data_file("trade_journal.json")
MONITOR_STATE_PATH = _data_file("monitor_state.json")
ERROR_LOG_PATH = _data_file("error_log.json")
SP500_PATH = os.path.join("nova-options-scanner-main", "nova-options-scanner-main", "sp500_symbols.json")
OPTIONABLE_UNIVERSE_PATH = "optionable_universe.json"
MOVERS_SNAPSHOT_PATH = _data_file("movers_snapshot.json")
MANUAL_PATH = os.path.join("docs", "manual.md")
_QUOTE_CACHE = {}
_QUOTE_CACHE_TTL = 15
MARKET_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]
_AUTH_CACHE = None
_LAST_SCHWAB_ERROR = None
_CHAIN_CACHE = {}
_CHAIN_CACHE_TTL = 60
_CHAIN_ERROR = {}
_OPTIONABLE_CACHE = {}
_OPTIONABLE_CACHE_TTL = 86400
_ACCOUNT_HASH_CACHE = {"ts": 0.0, "value": {}}
_ACCOUNT_HASH_CACHE_TTL = 900
_TRANSACTIONS_CACHE = {"ts": 0.0, "value": []}
_TRANSACTIONS_CACHE_TTL = 900
_TRANSACTIONS_FETCH_BUDGET_SEC = 20
SPREAD_SIM_SAVE_TYPE = "credit_spread_simulation"
SPREAD_SIM_SAVE_VERSION = 1
SPREAD_SIM_PRESETS = {
    "custom": {
        "label": "Custom",
        "description": "Keep your current assumptions and shape the ladder manually.",
        "defaults": {},
        "scenario": {"upside_pct": 0.03, "downside_pct": 0.03, "width_buffer": 2.0, "price_step": 0.5},
    },
    "conservative": {
        "label": "Conservative",
        "description": "Longer duration, natural fills, and a wider stress ladder for defensive planning.",
        "defaults": {"fill_mode": "natural", "slippage": 0.10, "dte": 30, "contracts": 1, "iv_short": 0.22, "iv_long": 0.20},
        "scenario": {"upside_pct": 0.06, "downside_pct": 0.12, "width_buffer": 3.0, "price_step": 1.0},
    },
    "income": {
        "label": "Income",
        "description": "Balanced biweekly setup with tighter ladder spacing for routine premium selling.",
        "defaults": {"fill_mode": "mid", "slippage": 0.05, "dte": 14, "contracts": 1, "iv_short": 0.25, "iv_long": 0.23},
        "scenario": {"upside_pct": 0.05, "downside_pct": 0.08, "width_buffer": 2.5, "price_step": 0.5},
    },
    "high_pop": {
        "label": "High POP",
        "description": "Favor distance from the short strike with a broader ladder and lower-vol assumptions.",
        "defaults": {"fill_mode": "natural", "slippage": 0.08, "dte": 21, "contracts": 1, "iv_short": 0.20, "iv_long": 0.18},
        "scenario": {"upside_pct": 0.04, "downside_pct": 0.15, "width_buffer": 3.5, "price_step": 1.0},
    },
}
NOVA_MODEL_OPTIONS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4o-mini",
]


def monitor_enabled():
    return os.getenv("NOVA_MONITOR_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}


def ensure_monitor_thread():
    global _MONITOR_THREAD
    if not monitor_enabled():
        return False
    if _MONITOR_THREAD and _MONITOR_THREAD.is_alive():
        return True
    with _MONITOR_LOCK:
        if _MONITOR_THREAD and _MONITOR_THREAD.is_alive():
            return True
        _MONITOR_THREAD = threading.Thread(
            target=monitor_loop,
            name="nova-monitor",
            daemon=True,
        )
        _MONITOR_THREAD.start()
        logging.info("Started monitor thread.")
        return True


def _get_env(key):
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


def _spread_sim_price_bounds(stock_price, short_strike, long_strike, preset):
    profile = SPREAD_SIM_PRESETS.get(preset, SPREAD_SIM_PRESETS["custom"]).get("scenario", {})
    width = max(abs(short_strike - long_strike), 0.01)
    upside_pct = profile.get("upside_pct", 0.03)
    downside_pct = profile.get("downside_pct", 0.03)
    width_buffer = profile.get("width_buffer", 2.0)
    higher_strike = max(short_strike, long_strike)
    lower_strike = min(short_strike, long_strike)
    upper = max(stock_price * (1.0 + upside_pct), higher_strike + width * width_buffer)
    lower = min(stock_price * (1.0 - downside_pct), lower_strike - width * width_buffer)
    return round(upper, 2), round(lower, 2)


def _spread_sim_action_summary(spread_type, zone, markers):
    marker_set = set(markers or [])
    if "3.0x Credit Stop" in marker_set:
        return "Hard stop or exit."
    if "2.5x Credit Stop" in marker_set:
        return "Defend now or cut risk."
    if "2.0x Credit Stop" in marker_set:
        return "Debit stop hit. Tighten risk."
    if "Short Strike Breach" in marker_set:
        if spread_type == "bull_put":
            return "Short strike breached. Roll or reduce delta."
        return "Short strike breached. Roll up/out or reduce delta."
    if "75% Target" in marker_set:
        return "Take most profits."
    if "50% Target" in marker_set:
        return "Primary take-profit zone."
    if "25% Target" in marker_set:
        return "Consider scaling out."
    if zone == "Max Profit Zone":
        return "Hold or close for a clean win."
    if zone == "Profit Zone":
        return "Manage winner and watch targets."
    if zone == "Loss Zone":
        return "Monitor closely and prep defense."
    if zone == "Max Loss Zone":
        return "Exit or roll decisively."
    return "Hold and reassess."


def load_settings():
    defaults = {
        "profit_target_pct": 50,
        "max_loss_pct": 50,
        "dte_exit": 7,
        "nova_judgment": True,
        "nova_model": "gpt-4o",
        "nova_role": "senior partner with 20 years of options experience and decision-making authority",
        "polling_minutes": 60,
        "timezone": "America/New_York",
        "monitor_market_hours_only": True,
        "market_open_time": "09:30",
        "market_close_time": "16:00",
        "monitor_weekdays_only": True,
        "market_holidays": "",
        "movers_lookback_days": 5,
        "movers_count": 10,
        "movers_include_positions": True,
        "movers_universe": "",
        "movers_use_sp500": False,
        "movers_max_seconds": 20,
    }
    if not os.path.exists(SETTINGS_PATH):
        save_settings(defaults)
        return defaults
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults.update({k: data.get(k, v) for k, v in defaults.items()})
        return defaults
    except Exception:
        return defaults


def save_settings(data):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _nova_model_options(current):
    options = list(NOVA_MODEL_OPTIONS)
    if current and current not in options:
        options.insert(0, current)
    return options


def _nova_role(settings):
    role = (settings.get("nova_role") or "").strip()
    if not role:
        return "disciplined options assistant"
    return role


def _get_auth_config():
    global _AUTH_CACHE
    if _AUTH_CACHE is not None:
        return _AUTH_CACHE
    username = os.getenv("AUTH_USERNAME", "").strip()
    password_hash = os.getenv("AUTH_PASSWORD_HASH", "").strip()
    password = os.getenv("AUTH_PASSWORD", "")
    if not password_hash and password:
        password_hash = generate_password_hash(password)
    if not username or not password_hash:
        settings = load_settings()
        username = username or (settings.get("auth_username") or "").strip()
        password_hash = password_hash or (settings.get("auth_password_hash") or "").strip()
    _AUTH_CACHE = {"username": username, "password_hash": password_hash}
    return _AUTH_CACHE


def _auth_ready():
    auth = _get_auth_config()
    return bool(auth["username"] and auth["password_hash"])


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authed"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

@app.before_request
def require_login():
    try:
        ensure_monitor_thread()
    except Exception as exc:
        logging.warning("Failed to start monitor thread: %s", exc)
        log_error(exc, context="ensure_monitor_thread")

    # These are endpoint names, NOT URL paths
    if request.endpoint in {"login", "logout", "static", "callback"}:
        return None

    if not session.get("authed"):
        return redirect(url_for("login", next=request.path))

    return None



def load_alerts():
    if not os.path.exists(ALERTS_PATH):
        return []
    try:
        with open(ALERTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_alerts(data):
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_alert_state():
    if not os.path.exists(ALERTS_STATE_PATH):
        return {"sent_ids": []}
    try:
        with open(ALERTS_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"sent_ids": []}


def save_alert_state(data):
    with open(ALERTS_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_monitor_state():
    if not os.path.exists(MONITOR_STATE_PATH):
        return {}
    try:
        with open(MONITOR_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_monitor_state(data):
    with open(MONITOR_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _parse_iso_dt(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except Exception:
        return None


def _parse_hhmm(value):
    if not value:
        return None
    try:
        parts = value.strip().split(":")
        if len(parts) != 2:
            return None
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return hour, minute
    except Exception:
        return None


def _parse_date_list(value):
    dates = set()
    if not value:
        return dates
    for item in str(value).split(","):
        token = item.strip()
        if not token:
            continue
        try:
            dates.add(dt.date.fromisoformat(token))
        except ValueError:
            continue
    return dates


def log_error(message, context=""):
    tz = ZoneInfo(load_settings().get("timezone", "UTC"))
    entry = {
        "timestamp": dt.datetime.now(tz).isoformat(timespec="seconds"),
        "message": str(message),
        "context": context,
    }
    data = []
    if os.path.exists(ERROR_LOG_PATH):
        try:
            with open(ERROR_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    data = data[-200:]
    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _set_schwab_error(exc, context):
    global _LAST_SCHWAB_ERROR
    _LAST_SCHWAB_ERROR = {
        "timestamp": time.time(),
        "error": str(exc),
        "context": context,
    }


def load_error_log():
    if not os.path.exists(ERROR_LOG_PATH):
        return []
    try:
        with open(ERROR_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def schwab_request(action, context):
    last_exc = None
    for attempt in range(3):
        try:
            return action()
        except Exception as exc:
            last_exc = exc
            _set_schwab_error(exc, context)
            log_error(exc, context=f"schwab:{context}:attempt{attempt + 1}")
            time.sleep(1 + attempt)
    raise last_exc


def fetch_price_history(symbol, days=365):
    client = get_client()
    end_date = dt.datetime.now(dt.timezone.utc)
    start_date = end_date - dt.timedelta(days=days)
    response = schwab_request(
        lambda: client.get_price_history(
            symbol,
            start_datetime=start_date,
            end_datetime=end_date,
        ),
        f"get_price_history:{symbol}",
    )
    response.raise_for_status()
    return response.json()


def _extract_closes(history):
    candles = history.get("candles", []) if isinstance(history, dict) else []
    closes = []
    for candle in candles:
        close = candle.get("close")
        if close is None:
            continue
        try:
            close_val = float(close)
        except (TypeError, ValueError):
            continue
        if close_val <= 0:
            continue
        closes.append(close_val)
    return closes


def compute_trend_metrics(closes):
    if not closes or len(closes) < 2:
        return {}
    def pct_change(n):
        if len(closes) < n + 1:
            return None
        return ((closes[-1] - closes[-1 - n]) / closes[-1 - n]) * 100
    def sma(n):
        if len(closes) < n:
            return None
        return sum(closes[-n:]) / n
    last = closes[-1]
    sma20 = sma(20)
    sma50 = sma(50)
    sma200 = sma(200)
    return {
        "last": last,
        "ret_5d": pct_change(5),
        "ret_20d": pct_change(20),
        "ret_60d": pct_change(60),
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "trend_20_50": "bull" if sma20 and sma50 and sma20 > sma50 else "bear",
        "trend_50_200": "bull" if sma50 and sma200 and sma50 > sma200 else "bear",
    }


def get_research_summary(symbol):
    summary = {}
    try:
        history = fetch_price_history(symbol, days=365)
        closes = _extract_closes(history)
        summary["symbol"] = symbol
        summary["metrics"] = compute_trend_metrics(closes)
    except Exception as exc:
        summary["symbol"] = symbol
        summary["error"] = str(exc)
    return summary


def get_market_research():
    research = []
    for symbol in MARKET_TICKERS:
        research.append(get_research_summary(symbol))
    return research


def load_sp500_symbols():
    if not os.path.exists(SP500_PATH):
        return []
    try:
        with open(SP500_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        symbols = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    sym = (entry.get("symbol") or "").upper().strip()
                else:
                    sym = str(entry).upper().strip()
                if sym:
                    symbols.append(sym)
        return symbols
    except Exception:
        return []


def _parse_symbol_list(raw):
    if not raw:
        return []
    parts = re.split(r"[,\s;]+", str(raw).upper())
    return [p.strip().upper() for p in parts if p.strip()]


def build_movers_universe(settings, positions=None):
    symbols = set()
    symbols.update(load_watchlist())
    symbols.update(MARKET_TICKERS)
    symbols.update(_parse_symbol_list(settings.get("movers_universe", "")))
    if settings.get("movers_use_sp500"):
        symbols.update(load_sp500_symbols())

    if settings.get("movers_include_positions", True):
        if positions is None:
            try:
                positions = fetch_positions()
            except Exception:
                positions = []
        for pos in positions or []:
            if (pos.get("assetType") or "").upper() == "OPTION":
                continue
            symbol = (pos.get("symbol") or "").upper().strip()
            if symbol:
                symbols.add(symbol)

    return sorted(symbols)


def _rotate_symbols_for_week(symbols):
    if not symbols:
        return []
    ordered = sorted(symbols)
    today = dt.date.today()
    week_key = (today.year * 100) + today.isocalendar().week
    offset = week_key % len(ordered)
    return ordered[offset:] + ordered[:offset]


def symbol_has_options(symbol):
    now = time.time()
    cached = _OPTIONABLE_CACHE.get(symbol)
    if cached and now - cached["ts"] <= _OPTIONABLE_CACHE_TTL:
        return bool(cached["value"])
    try:
        chain = fetch_option_chain(symbol)
        call_map = chain.get("callExpDateMap") or {}
        put_map = chain.get("putExpDateMap") or {}
        has_data = bool(call_map or put_map)
    except Exception:
        has_data = False
    _OPTIONABLE_CACHE[symbol] = {"ts": now, "value": has_data}
    return has_data


def scan_stock_movers(symbols, lookback_days=5, max_count=10, max_seconds=20):
    if not symbols:
        return {"movers": [], "errors": [], "lookback_days": lookback_days, "universe_size": 0}

    try:
        lookback_days = int(lookback_days)
    except (TypeError, ValueError):
        lookback_days = 5
    lookback_days = max(1, lookback_days)

    try:
        max_count = int(max_count)
    except (TypeError, ValueError):
        max_count = 10
    max_count = max(1, max_count)

    try:
        max_seconds = int(max_seconds)
    except (TypeError, ValueError):
        max_seconds = 20
    max_seconds = max(5, max_seconds)

    fetch_days = max(lookback_days * 2, lookback_days + 5)
    symbols = _rotate_symbols_for_week(symbols)
    movers = []
    errors = []
    start = time.monotonic()

    for symbol in symbols:
        if time.monotonic() - start >= max_seconds:
            errors.append({"symbol": None, "error": f"scan_timeout_after_{max_seconds}s"})
            break
        try:
            history = fetch_price_history(symbol, days=fetch_days)
            closes = _extract_closes(history)
            if len(closes) <= lookback_days:
                errors.append({"symbol": symbol, "error": "insufficient history"})
                continue
            last = closes[-1]
            prior = closes[-1 - lookback_days]
            if not prior or prior <= 0:
                errors.append({"symbol": symbol, "error": "invalid prior close"})
                continue
            change_pct = ((last - prior) / prior) * 100
            movers.append({
                "symbol": symbol,
                "last": last,
                "prior": prior,
                "change_pct": change_pct,
                "direction": "up" if change_pct >= 0 else "down",
            })
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    movers.sort(key=lambda item: abs(item.get("change_pct") or 0), reverse=True)

    optionable = []
    for item in movers:
        if len(optionable) >= max_count:
            break
        if time.monotonic() - start >= max_seconds:
            errors.append({"symbol": None, "error": f"optionable_filter_timeout_after_{max_seconds}s"})
            break
        symbol = item.get("symbol")
        if symbol and symbol_has_options(symbol):
            optionable.append(item)

    return {
        "movers": optionable,
        "errors": errors,
        "lookback_days": lookback_days,
        "universe_size": len(symbols),
    }


def _format_movers_context(snapshot):
    movers = snapshot.get("movers") or []
    if not movers:
        return "Movers scan returned no results."
    rows = []
    for entry in movers:
        change = entry.get("change_pct")
        last = entry.get("last")
        symbol = entry.get("symbol")
        if change is None or last is None:
            continue
        rows.append(f"{symbol} {change:+.2f}% (last={last:.2f})")
    lookback = snapshot.get("lookback_days")
    universe_size = snapshot.get("universe_size")
    return (
        f"Movers over last {lookback} trading days from universe size {universe_size}: "
        + "; ".join(rows)
    )


def _wants_movers(text):
    text = (text or "").lower()
    triggers = (
        "mover",
        "movement",
        "stocks to watch",
        "top stocks",
        "weekly",
        "for the week",
        "top picks",
    )
    return any(token in text for token in triggers)


def load_watchlist():
    if not os.path.exists(WATCHLIST_PATH):
        save_watchlist([])
        return []
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_watchlist(data):
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_watchlist_notes():
    if not os.path.exists(WATCHLIST_NOTES_PATH):
        save_watchlist_notes({})
        return {}
    try:
        with open(WATCHLIST_NOTES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_watchlist_notes(data):
    with open(WATCHLIST_NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_optionable_universe_files():
    files = []
    try:
        for name in os.listdir("."):
            lowered = name.lower()
            if lowered.endswith(".json") and "universe" in lowered:
                files.append(name)
    except Exception:
        files = []
    if OPTIONABLE_UNIVERSE_PATH not in files and os.path.exists(OPTIONABLE_UNIVERSE_PATH):
        files.append(OPTIONABLE_UNIVERSE_PATH)
    return sorted(set(files), key=lambda x: x.lower())


def load_optionable_universe(universe_path=None):
    path = universe_path or OPTIONABLE_UNIVERSE_PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        universe = []
        if not isinstance(data, list):
            return []
        for item in data:
            if isinstance(item, dict):
                symbol = (item.get("symbol") or "").upper().strip()
                name = (item.get("name") or symbol).strip()
            else:
                symbol = str(item).upper().strip()
                name = symbol
            if symbol:
                universe.append({"symbol": symbol, "name": name})
        return universe
    except Exception:
        return []


def load_movers_snapshot():
    if not os.path.exists(MOVERS_SNAPSHOT_PATH):
        return None
    try:
        with open(MOVERS_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def save_movers_snapshot(data):
    with open(MOVERS_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_manual_text():
    if not os.path.exists(MANUAL_PATH):
        return "Manual not found."
    try:
        with open(MANUAL_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Failed to load manual: {exc}"


def _realized_vol_pct(closes, window=20):
    if len(closes) < window + 1:
        return None
    rets = []
    for i in range(-window, 0):
        prev_close = closes[i - 1]
        curr_close = closes[i]
        if prev_close <= 0:
            continue
        rets.append((curr_close / prev_close) - 1.0)
    if len(rets) < 2:
        return None
    mean_ret = sum(rets) / len(rets)
    variance = sum((r - mean_ret) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(variance) * 100


def _normalize_movers_strategy(value):
    strategy = (value or "bull_put").strip().lower()
    if strategy in ("bull_put", "bear_call"):
        return strategy
    return "bull_put"


def _movers_directional_score(strategy, ret_5d, ret_20d, vol_20d):
    r5 = ret_5d or 0.0
    r20 = ret_20d or 0.0
    vol = vol_20d or 0.0
    if strategy == "bear_call":
        # Favor downside momentum for bearish call credit spreads.
        return max(0.0, -r5) + (max(0.0, -r20) * 0.5) + (vol * 0.35)
    # Default to bullish momentum for put credit spreads.
    return max(0.0, r5) + (max(0.0, r20) * 0.5) + (vol * 0.35)


def run_optionable_weekly_scan(
    universe,
    lookback_days=5,
    top_n=10,
    max_seconds=25,
    strategy="bull_put",
    previous=None,
):
    strategy = _normalize_movers_strategy(strategy)
    rows_cache = {}
    if isinstance(previous, dict):
        prev_lookback = int(previous.get("lookback_days", lookback_days))
        prev_strategy = _normalize_movers_strategy(previous.get("strategy", "bull_put"))
        if prev_lookback == int(lookback_days) and prev_strategy == strategy:
            for item in previous.get("rows_cache", []) or []:
                symbol = (item.get("symbol") or "").upper().strip()
                if symbol:
                    rows_cache[symbol] = item
    errors = []
    started = time.monotonic()
    scanned = 0
    fetch_days = max(lookback_days + 35, 60)
    total_symbols = len(universe)
    if total_symbols == 0:
        return {
            "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
            "lookback_days": lookback_days,
            "top_n": top_n,
            "max_seconds": max_seconds,
            "timed_out": False,
            "universe_size": 0,
            "scanned": 0,
            "scanned_this_run": 0,
            "covered_count": 0,
            "next_index": 0,
            "rows": [],
            "rows_cache": [],
            "error_count": 0,
            "error_samples": [],
        }

    start_index = 0
    if isinstance(previous, dict):
        prev_lookback = int(previous.get("lookback_days", lookback_days))
        prev_strategy = _normalize_movers_strategy(previous.get("strategy", "bull_put"))
        if prev_lookback == int(lookback_days) and prev_strategy == strategy:
            start_index = int(previous.get("next_index", 0)) % total_symbols

    for step in range(total_symbols):
        if time.monotonic() - started >= max_seconds:
            break
        idx = (start_index + step) % total_symbols
        item = universe[idx]
        symbol = item.get("symbol")
        name = item.get("name") or symbol
        if not symbol:
            continue
        scanned += 1
        try:
            history = fetch_price_history(symbol, days=fetch_days)
            closes = _extract_closes(history)
            metrics = compute_trend_metrics(closes)
            ret_5d = metrics.get("ret_5d")
            ret_20d = metrics.get("ret_20d")
            vol_20d = _realized_vol_pct(closes, window=20)
            if ret_5d is None and ret_20d is None:
                continue
            score = _movers_directional_score(strategy, ret_5d, ret_20d, vol_20d)
            rows_cache[symbol] = {
                "symbol": symbol,
                "name": name,
                "last": metrics.get("last"),
                "ret_5d": ret_5d,
                "ret_20d": ret_20d,
                "vol_20d": vol_20d,
                "score": score,
                "direction": "up" if (ret_5d or 0) >= 0 else "down",
            }
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    rows = list(rows_cache.values())
    rows.sort(key=lambda x: x.get("score") or 0, reverse=True)
    timed_out = (time.monotonic() - started) >= max_seconds
    next_index = (start_index + scanned) % total_symbols if total_symbols else 0
    covered_count = len(rows_cache)
    snapshot = {
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "lookback_days": lookback_days,
        "top_n": top_n,
        "max_seconds": max_seconds,
        "strategy": strategy,
        "timed_out": timed_out,
        "universe_size": len(universe),
        "scanned": scanned,
        "scanned_this_run": scanned,
        "covered_count": covered_count,
        "next_index": next_index,
        "rows": rows[:top_n],
        "rows_cache": rows,
        "error_count": len(errors),
        "error_samples": errors[:5],
    }
    return snapshot


def load_tickets():
    if not os.path.exists(TICKETS_PATH):
        save_tickets([])
        return []
    try:
        with open(TICKETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_tickets(data):
    with open(TICKETS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_trade_journal():
    if not os.path.exists(TRADE_JOURNAL_PATH):
        save_trade_journal([])
        return []
    try:
        with open(TRADE_JOURNAL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_trade_journal(data):
    with open(TRADE_JOURNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _parse_date_safe(raw):
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except Exception:
        return None


def _trade_gate_result(trade):
    checks = trade.get("checks", {}) or {}
    required = {
        "setup_match": "Setup criteria mismatch",
        "risk_within_cap": "Max loss exceeds cap",
        "min_pop_met": "Minimum POP not met",
        "size_valid": "Position sizing invalid",
        "no_rule_violation": "Rule violation flagged",
    }
    failed = [msg for key, msg in required.items() if not checks.get(key)]
    return {
        "passed": len(failed) == 0,
        "failed_reasons": failed,
    }


def compute_trade_stats(trades):
    today = dt.date.today()
    closed_rows = []
    for t in trades:
        status = str(t.get("status", "")).lower()
        expiry = _parse_date_safe(t.get("expiry_date"))
        exit_date = _parse_date_safe(t.get("exit_date"))
        realized_pnl = _safe_float(t.get("realized_pnl"))
        max_loss = _safe_float(t.get("max_loss"))
        if realized_pnl is None:
            continue
        include = status in {"closed", "expired"} or (expiry is not None and expiry <= today)
        if not include:
            continue
        return_pct = None
        if max_loss and max_loss != 0:
            return_pct = (realized_pnl / abs(max_loss)) * 100.0
        closed_rows.append({
            "id": t.get("id"),
            "symbol": t.get("symbol"),
            "strategy": t.get("strategy"),
            "pnl": realized_pnl,
            "return_pct": return_pct,
            "date": exit_date or expiry or _parse_date_safe(t.get("entry_date")) or today,
        })

    closed_rows.sort(key=lambda x: x["date"])
    total = len(closed_rows)
    wins = len([r for r in closed_rows if r["pnl"] > 0])
    losses = len([r for r in closed_rows if r["pnl"] < 0])
    net = sum(r["pnl"] for r in closed_rows)
    avg_return = None
    return_vals = [r["return_pct"] for r in closed_rows if r["return_pct"] is not None]
    if return_vals:
        avg_return = sum(return_vals) / len(return_vals)

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    monthly = {}
    for r in closed_rows:
        equity += r["pnl"]
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_drawdown:
            max_drawdown = dd
        month_key = r["date"].strftime("%Y-%m")
        if month_key not in monthly:
            monthly[month_key] = {"month": month_key, "count": 0, "pnl": 0.0}
        monthly[month_key]["count"] += 1
        monthly[month_key]["pnl"] += r["pnl"]

    by_strategy = {}
    for r in closed_rows:
        key = r["strategy"] or "UNKNOWN"
        if key not in by_strategy:
            by_strategy[key] = {"strategy": key, "count": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        by_strategy[key]["count"] += 1
        by_strategy[key]["pnl"] += r["pnl"]
        if r["pnl"] > 0:
            by_strategy[key]["wins"] += 1
        elif r["pnl"] < 0:
            by_strategy[key]["losses"] += 1

    win_rate = ((wins / total) * 100.0) if total else 0.0
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_pnl": net,
        "avg_return_pct": avg_return,
        "max_drawdown": max_drawdown,
        "closed_rows": closed_rows,
        "by_month": sorted(monthly.values(), key=lambda x: x["month"], reverse=True),
        "by_strategy": sorted(by_strategy.values(), key=lambda x: x["pnl"], reverse=True),
    }


def _parse_token_payload(raw):
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    token = payload.get("token", payload)
    if not isinstance(token, dict) or not token.get("refresh_token"):
        return None
    expires_at = token.get("expires_at") or payload.get("expires_at") or 0
    creation_ts = payload.get("creation_timestamp") or 0
    try:
        expires_at = int(expires_at) if expires_at else 0
    except Exception:
        expires_at = 0
    try:
        creation_ts = int(creation_ts) if creation_ts else 0
    except Exception:
        creation_ts = 0
    return {
        "raw": json.dumps(payload, separators=(",", ":"), sort_keys=True),
        "creation_ts": creation_ts,
        "expires_at": expires_at,
    }


def _best_token_candidate(candidates):
    valid = [candidate for candidate in candidates if candidate]
    if not valid:
        return None
    valid.sort(key=lambda item: (item["creation_ts"], item["expires_at"]), reverse=True)
    return valid[0]


def get_client():
    global _CLIENT
    if _CLIENT is None:
        token_path = ensure_token_file()

        # No token yet - user must log in first
        if not token_path or not os.path.exists(token_path):
            raise RuntimeError("No token file found. Log in to Schwab first.")

        _CLIENT = client_from_token_file(
            token_path,
            _get_env("SCHWAB_API_KEY"),
            _get_env("SCHWAB_APP_SECRET"),
            enforce_enums=False,
        )
    return _CLIENT



def ensure_token_file():
    token_path = os.getenv("TOKEN_PATH", "token.json")
    raw_file = None
    if os.path.exists(token_path):
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                raw_file = f.read()
        except Exception:
            raw_file = None

    token_json = os.getenv("TOKEN_JSON")
    token_b64 = os.getenv("TOKEN_JSON_B64")
    decoded_b64 = None
    if token_b64:
        try:
            decoded_b64 = base64.b64decode(token_b64).decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Failed to decode TOKEN_JSON_B64: {exc}") from exc

    selected = _best_token_candidate(
        [
            _parse_token_payload(raw_file),
            _parse_token_payload(token_json),
            _parse_token_payload(decoded_b64),
        ]
    )

    if selected:
        token_dir = os.path.dirname(token_path)
        if token_dir and not os.path.exists(token_dir):
            os.makedirs(token_dir, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(selected["raw"])
        return token_path

    # No token file and no usable env token - return None
    return None

def get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        http_client = httpx.Client(timeout=30.0, trust_env=False)
        _OPENAI_CLIENT = OpenAI(
            api_key=_get_env("OPENAI_API_KEY"),
            http_client=http_client,
        )
    return _OPENAI_CLIENT


def email_settings():
    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    to_addr = os.getenv("ALERT_EMAIL_TO")
    from_addr = os.getenv("ALERT_EMAIL_FROM") or user
    if not all([host, port, user, password, to_addr]):
        return None
    return {
        "host": host,
        "port": int(port),
        "user": user,
        "password": password,
        "to_addr": to_addr,
        "from_addr": from_addr,
    }


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["bid"] >= 0.01) & (df["ask"] >= 0.01)]
    df = df[df["ask"] >= df["bid"]]
    if "openInterest" in df.columns:
        df = df[df["openInterest"] > 0]
    return df


def parse_expiration(exp_str):
    if not exp_str:
        return None
    text = str(exp_str).strip()
    if not text:
        return None
    exp_str = text[:10]
    try:
        return dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
    except ValueError:
        pass
    # Some payloads provide compact YYYYMMDD.
    compact = re.sub(r"\D", "", text)
    if len(compact) == 8:
        try:
            return dt.datetime.strptime(compact, "%Y%m%d").date()
        except ValueError:
            pass
    # Some payloads provide Unix timestamps (seconds or milliseconds).
    if compact.isdigit() and len(compact) in (10, 13):
        try:
            ts = int(compact)
            if len(compact) == 13:
                ts = ts / 1000.0
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()
        except (ValueError, OverflowError, OSError):
            pass
    return None


def parse_option_symbol(symbol):
    # Accepts common OCC-style strings, with or without spaces.
    text = (symbol or "").strip().upper()
    if not text:
        return {}
    compact = re.sub(r"\s+", "", text)
    match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", compact)
    if not match:
        return {}
    underlying, expiry_raw, cp_flag, strike_raw = match.groups()
    expiry = None
    try:
        expiry = dt.datetime.strptime(expiry_raw, "%y%m%d").date()
    except ValueError:
        expiry = None
    return {
        "underlying": underlying,
        "expiry": expiry,
        "type": "CALL" if cp_flag == "C" else "PUT",
        "strike": int(strike_raw) / 1000.0,
    }


def fetch_positions():
    client = get_client()
    response = schwab_request(lambda: client.get_accounts(fields=["positions"]), "get_accounts_positions")
    response.raise_for_status()
    data = response.json()
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

            # Some equities come through with incomplete basis/price and show N/A in Schwab UI.
            # Treat those as unknown P/L so totals do not drift from Schwab account-level numbers.
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

    return positions


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        neg = False
        if text.startswith("(") and text.endswith(")"):
            neg = True
            text = text[1:-1]
        text = text.replace("$", "").replace(",", "").replace("%", "").strip()
        if text.endswith("CR"):
            text = text[:-2].strip()
        if text.endswith("DR"):
            text = text[:-2].strip()
            neg = True
        try:
            val = float(text)
            return -val if neg else val
        except (TypeError, ValueError):
            return None
    return None


def normalize_quote(raw):
    if not isinstance(raw, dict):
        return {}
    quote = raw.get("quote", raw)
    ref = raw.get("reference", {})
    return {
        "symbol": raw.get("symbol") or ref.get("symbol"),
        "last": _safe_float(quote.get("lastPrice") or quote.get("last") or quote.get("mark")),
        "mark": _safe_float(quote.get("mark")),
        "bid": _safe_float(quote.get("bidPrice") or quote.get("bid")),
        "ask": _safe_float(quote.get("askPrice") or quote.get("ask")),
        "delta": _safe_float(quote.get("delta")),
        "theta": _safe_float(quote.get("theta")),
        "gamma": _safe_float(quote.get("gamma")),
        "vega": _safe_float(quote.get("vega")),
        "iv": _safe_float(quote.get("volatility") or quote.get("impliedVolatility")),
    }


def fetch_quotes(symbols):
    if not symbols:
        return {}
    now = time.time()
    cached = {}
    missing = []
    for sym in symbols:
        entry = _QUOTE_CACHE.get(sym)
        if entry and now - entry["ts"] <= _QUOTE_CACHE_TTL:
            cached[sym] = entry["quote"]
        else:
            missing.append(sym)

    if not missing:
        return cached

    client = get_client()
    response = schwab_request(lambda: client.get_quotes(missing), "get_quotes")
    response.raise_for_status()
    data = response.json()
    quotes = {}
    for symbol, payload in data.items():
        quotes[symbol] = normalize_quote(payload)
        _QUOTE_CACHE[symbol] = {"ts": now, "quote": quotes[symbol]}
    quotes.update(cached)
    return quotes


def evaluate_position(pos, settings):
    action = "Hold"
    reasons = []
    pnl_pct = pos.get("pnl_pct")
    dte = pos.get("dte")

    if pnl_pct is not None:
        if pnl_pct >= settings["profit_target_pct"]:
            action = "Close"
            reasons.append(f"Profit target hit ({pnl_pct:.1f}%)")
        if pnl_pct <= -settings["max_loss_pct"]:
            action = "Close"
            reasons.append(f"Max loss hit ({pnl_pct:.1f}%)")

    if dte is not None and dte <= settings["dte_exit"]:
        action = "Close"
        reasons.append(f"Expires in {dte} days (threshold reached)")

    note = ""
    if settings.get("nova_judgment"):
        near_profit = pnl_pct is not None and (settings["profit_target_pct"] - pnl_pct) <= 5
        near_loss = pnl_pct is not None and (pnl_pct + settings["max_loss_pct"]) <= 5
        near_dte = dte is not None and dte <= settings["dte_exit"] + 2
        if near_profit or near_loss or near_dte:
            note = "Nova: near threshold, review context."

    return action, reasons, note


def build_alerts(positions, settings):
    alerts = []
    for pos in positions:
        action, reasons, note = evaluate_position(pos, settings)
        if action != "Hold":
            alerts.append({
                "accountNumber": pos.get("accountNumber"),
                "symbol": pos.get("symbol"),
                "assetType": pos.get("assetType"),
                "qty": pos.get("qty"),
                "pnl": pos.get("pnl"),
                "pnl_pct": pos.get("pnl_pct"),
                "dte": pos.get("dte"),
                "action": action,
                "reasons": "; ".join(reasons),
                "note": note,
                "quote": pos.get("quote", {}),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            })
    return alerts


def _alert_id(alert):
    return "|".join([
        str(alert.get("accountNumber")),
        str(alert.get("symbol")),
        str(alert.get("assetType")),
        str(alert.get("action")),
        str(alert.get("reasons")),
    ])


def send_email_alerts(alerts):
    settings = email_settings()
    if settings is None:
        return
    state = load_alert_state()
    sent_ids = set(state.get("sent_ids", []))
    new_alerts = [a for a in alerts if _alert_id(a) not in sent_ids]
    if not new_alerts:
        return
    summary = []
    for alert in new_alerts[:5]:
        summary.append(f"{alert['symbol']} {alert['action']}: {alert['reasons']}")
    body = "Nova alert: " + " | ".join(summary)
    msg = EmailMessage()
    msg["Subject"] = "Nova Alert"
    msg["From"] = settings["from_addr"]
    msg["To"] = settings["to_addr"]
    msg.set_content(body)

    with smtplib.SMTP(settings["host"], settings["port"]) as server:
        server.starttls()
        server.login(settings["user"], settings["password"])
        server.send_message(msg)
    tz = ZoneInfo(load_settings().get("timezone", "UTC"))
    state = load_monitor_state()
    state["last_email_sent"] = dt.datetime.now(tz).isoformat(timespec="seconds")
    save_monitor_state(state)
    for alert in new_alerts:
        sent_ids.add(_alert_id(alert))
    state["sent_ids"] = list(sent_ids)[-500:]
    save_alert_state(state)


def send_test_email_alert():
    settings = email_settings()
    if settings is None:
        raise RuntimeError("Email alerts are not configured. Missing SMTP_* or ALERT_EMAIL_* env vars.")

    now = dt.datetime.now(ZoneInfo(load_settings().get("timezone", "UTC"))).isoformat(timespec="seconds")
    msg = EmailMessage()
    msg["Subject"] = "Nova Alert Test"
    msg["From"] = settings["from_addr"]
    msg["To"] = settings["to_addr"]
    msg.set_content(
        "This is a Nova test email.\n"
        f"Sent at: {now}\n"
        "If you received this message, alert delivery is working."
    )

    with smtplib.SMTP(settings["host"], settings["port"]) as server:
        server.starttls()
        server.login(settings["user"], settings["password"])
        server.send_message(msg)


def monitor_loop():
    while True:
        settings = load_settings()
        tz = ZoneInfo(settings.get("timezone", "UTC"))
        state = load_monitor_state()
        paused_until = _parse_iso_dt(state.get("paused_until"))
        if paused_until and paused_until > dt.datetime.now(tz):
            time.sleep(60)
            continue
        now = dt.datetime.now(tz)
        if settings.get("monitor_weekdays_only") and now.weekday() >= 5:
            time.sleep(60)
            continue
        holidays = _parse_date_list(settings.get("market_holidays", ""))
        if holidays and now.date() in holidays:
            time.sleep(60)
            continue
        if settings.get("monitor_market_hours_only"):
            open_time = _parse_hhmm(settings.get("market_open_time"))
            close_time = _parse_hhmm(settings.get("market_close_time"))
            if open_time and close_time:
                market_open = now.replace(hour=open_time[0], minute=open_time[1], second=0, microsecond=0)
                market_close = now.replace(hour=close_time[0], minute=close_time[1], second=0, microsecond=0)
                if not (market_open <= now <= market_close):
                    time.sleep(60)
                    continue
        try:
            positions = fetch_positions()
            option_positions = [p for p in positions if p.get("assetType") == "OPTION"]
            if not option_positions:
                state = load_monitor_state()
                state["last_monitor_run"] = now.isoformat(timespec="seconds")
                state["last_alert_count"] = 0
                state["failure_count"] = 0
                save_monitor_state(state)
                time.sleep(max(1, int(settings.get("polling_minutes", 60))) * 60)
                continue
            symbols = sorted({p.get("symbol") for p in positions if p.get("symbol")})
            quotes = fetch_quotes(symbols)
            for pos in positions:
                pos_symbol = pos.get("symbol")
                if pos_symbol in quotes:
                    pos["quote"] = quotes[pos_symbol]
            alerts = build_alerts(positions, settings)
            with _ALERTS_LOCK:
                global ALERTS
                ALERTS = alerts
                save_alerts(alerts)
            send_email_alerts(alerts)
            state = load_monitor_state()
            state["last_monitor_run"] = dt.datetime.now(tz).isoformat(timespec="seconds")
            state["last_alert_count"] = len(alerts)
            state["failure_count"] = 0
            state["paused_until"] = None
            save_monitor_state(state)
        except Exception as exc:
            logging.warning("Monitor error: %s", exc)
            log_error(exc, context="monitor_loop")
            state = load_monitor_state()
            failures = int(state.get("failure_count") or 0) + 1
            state["failure_count"] = failures
            if failures >= 3:
                state["paused_until"] = (dt.datetime.now(tz) + dt.timedelta(minutes=30)).isoformat(timespec="seconds")
            save_monitor_state(state)
        time.sleep(max(1, int(settings.get("polling_minutes", 60))) * 60)


def flatten_option_map(exp_map):
    flattened = {}
    for exp_key, strikes in exp_map.items():
        parts = exp_key.split(":")
        exp_date = parts[0]
        dte = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        rows = []
        for strike, options in strikes.items():
            for opt in options:
                delta_raw = opt.get("delta")
                try:
                    delta_raw = float(delta_raw)
                except (TypeError, ValueError):
                    delta_raw = None
                iv_raw = opt.get("volatility", opt.get("impliedVolatility", 0))
                try:
                    iv_raw = float(iv_raw)
                except (TypeError, ValueError):
                    iv_raw = None
                rows.append({
                    "strike": float(opt.get("strikePrice", strike)),
                    "bid": float(opt.get("bid", 0) or 0),
                    "ask": float(opt.get("ask", 0) or 0),
                    "openInterest": int(opt.get("openInterest", 0) or 0),
                    "impliedVolatility": iv_raw,
                    "impliedVolatility_raw": iv_raw,
                    "delta": delta_raw,
                    "delta_raw": delta_raw,
                })
        flattened[exp_date] = {"dte": dte, "rows": rows}
    return flattened


def fetch_option_chain(symbol):
    now = time.time()
    cached = _CHAIN_CACHE.get(symbol)
    if cached and now - cached["ts"] <= _CHAIN_CACHE_TTL:
        return cached["data"]
    err = _CHAIN_ERROR.get(symbol)
    if err and now - err["ts"] <= 120:
        raise RuntimeError(f"Recent Schwab error for {symbol}; retry in a bit.")
    client = get_client()
    response = schwab_request(
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
            response = schwab_request(
                lambda: client.get_option_chain(
                    symbol,
                    include_underlying_quote=True,
                    strike_count=30,
                ),
                "get_option_chain_limited",
            )
    response.raise_for_status()
    data = response.json()
    _CHAIN_CACHE[symbol] = {"ts": now, "data": data}
    return data


def get_token_status():
    token_path = os.getenv("TOKEN_PATH", "token.json")
    if not os.path.exists(token_path):
        return None
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        token = payload.get("token", payload)
        expires_at = token.get("expires_at")
        if not expires_at:
            return None
        now = int(time.time())
        return {
            "expires_at": int(expires_at),
            "seconds_left": int(expires_at) - now,
        }
    except Exception:
        return None


@app.context_processor
def inject_schwab_status():
    if not _LAST_SCHWAB_ERROR:
        return {}
    age = time.time() - _LAST_SCHWAB_ERROR["timestamp"]
    if age > 1800:
        return {}
    return {"schwab_banner": _LAST_SCHWAB_ERROR}


def get_chain_data(symbol):
    chain = fetch_option_chain(symbol)
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


def run_scan(symbol, expiry, strategy, width, max_loss, min_pop, raw_mode, contracts,
             pricing_mode="mid", custom_limit=None):
    spot_price, call_map, put_map, expirations = get_chain_data(symbol)
    if not expiry:
        return None, spot_price, expirations
    if spot_price is None:
        raise RuntimeError("Missing underlying price from option chain response.")
    exp_calls = call_map.get(expiry, {"rows": [], "dte": None})
    exp_puts = put_map.get(expiry, {"rows": [], "dte": None})
    dte = exp_calls["dte"] or exp_puts["dte"] or 0
    T = dte / 365.0 if dte else 0

    calls_df = clean_chain(pd.DataFrame(exp_calls["rows"]))
    puts_df = clean_chain(pd.DataFrame(exp_puts["rows"]))

    if strategy == "bull_put":
        results_df = scan_bull_put(
            puts_df, spot_price, expiry, dte, T,
            width, max_loss, min_pop, raw_mode, contracts,
            pricing_mode=pricing_mode,
            custom_limit=custom_limit,
        )
    elif strategy == "bear_call":
        results_df = scan_bear_call(
            calls_df, spot_price, expiry, dte, T,
            width, max_loss, min_pop, raw_mode, contracts,
            pricing_mode=pricing_mode,
            custom_limit=custom_limit,
        )
    else:
        results_df = scan_iron_condor(
            {"puts": puts_df, "calls": calls_df},
            spot_price, expiry, dte, T,
            width, max_loss, min_pop, raw_mode, contracts,
            pricing_mode=pricing_mode,
            custom_limit=custom_limit,
        )

    return results_df, spot_price, expirations


def summarize_scan(results_df):
    if results_df is None or results_df.empty:
        return {"count": 0, "top": None}
    df = results_df.copy()
    if "POP %" in df.columns:
        df = df.sort_values(by="POP %", ascending=False)
    top = df.iloc[0].to_dict() if not df.empty else None
    return {"count": len(df), "top": top}


def parse_legs_from_trade(trade_text):
    if not trade_text:
        return []
    parts = str(trade_text).split("/")
    if len(parts) < 2:
        return []
    sell_part = parts[0].strip()
    buy_part = parts[1].strip()
    legs = []
    try:
        sell_tokens = sell_part.split()
        buy_tokens = buy_part.split()
        sell_strike = float(sell_tokens[1])
        buy_strike = float(buy_tokens[1])
        opt_type = buy_tokens[2] if len(buy_tokens) > 2 else sell_tokens[-1]
        legs.append({"action": "SELL", "strike": sell_strike, "type": opt_type})
        legs.append({"action": "BUY", "strike": buy_strike, "type": opt_type})
    except Exception:
        return []
    return legs


def build_spread_sim_link(symbol, strategy, pricing_mode, row):
    if strategy not in {"bull_put", "bear_call"} or not isinstance(row, dict):
        return None
    legs = parse_legs_from_trade(row.get("Trade"))
    if len(legs) < 2:
        return None
    short_leg = next((leg for leg in legs if leg.get("action") == "SELL"), None)
    long_leg = next((leg for leg in legs if leg.get("action") == "BUY"), None)
    if not short_leg or not long_leg:
        return None

    credit_dollars = _safe_float(row.get("Credit (Realistic)"))
    implied_vol = _safe_float(row.get("Implied Vol"))
    if implied_vol is not None and implied_vol > 1.0:
        implied_vol = implied_vol / 100.0

    fill_mode = "natural" if pricing_mode == "natural" else "mid"
    params = {
        "symbol": symbol,
        "spread_type": strategy,
        "preset": "custom",
        "stock_price": _safe_float(row.get("Spot")),
        "short_strike": _safe_float(short_leg.get("strike")),
        "long_strike": _safe_float(long_leg.get("strike")),
        "credit": (round(credit_dollars / 100.0, 4) if credit_dollars is not None else None),
        "fill_mode": fill_mode,
        "contracts": int(_safe_float(row.get("Contracts")) or 1),
        "dte": int(_safe_float(row.get("DTE")) or 0),
        "iv_short": implied_vol,
        "iv_long": implied_vol,
    }
    params = {key: value for key, value in params.items() if value is not None}
    return url_for("spread_sim", **params)


def build_management_plan(ticket, trade_row=None, assumptions=None, overrides=None, settings=None):
    trade = trade_row or (ticket.get("trade") if isinstance(ticket, dict) else {}) or {}
    details = assumptions or (ticket.get("assumptions") if isinstance(ticket, dict) else {}) or {}
    overrides = overrides or {}
    settings = settings or load_settings()
    contracts = int(
        _safe_float(trade.get("Contracts"))
        or _safe_float(details.get("contracts"))
        or len((ticket or {}).get("legs", [])) // 2
        or 1
    )
    credit_total = _safe_float(trade.get("Total Credit ($)"))
    if credit_total is None:
        entry_credit_per_spread = _safe_float(details.get("entry_credit_per_spread"))
        if entry_credit_per_spread is not None:
            credit_total = entry_credit_per_spread * 100.0 * contracts
    if credit_total is None:
        realistic_credit = _safe_float(trade.get("Credit (Realistic)"))
        if realistic_credit is not None:
            if realistic_credit > 10:
                credit_total = realistic_credit * contracts
            else:
                credit_total = realistic_credit * 100.0 * contracts
    max_loss = _safe_float(trade.get("Max Loss ($)") or trade.get("Max Loss"))
    if max_loss is None:
        max_loss = _safe_float(details.get("max_loss"))
    if credit_total is None:
        credit_total = 0.0
    if max_loss is None:
        max_loss = 0.0

    profit_target_pct = _safe_float(overrides.get("profit_target_pct"))
    if profit_target_pct is None:
        profit_target_pct = _safe_float(settings.get("profit_target_pct")) or 50.0
    profit_target_pct = max(1.0, min(profit_target_pct, 99.0))
    profit_target_amount = round(credit_total * (profit_target_pct / 100.0), 2)
    primary_stop_multiple = _safe_float(overrides.get("primary_stop_multiple"))
    if primary_stop_multiple is None:
        primary_stop_multiple = 2.0
    primary_stop_multiple = min(max(primary_stop_multiple, 2.0), 3.0)
    debit_stops = []
    for multiple in (2.0, 2.5, 3.0):
        debit_stops.append({
            "label": f"{multiple:.1f}x credit",
            "multiple": multiple,
            "buyback_debit_total": round(credit_total * multiple, 2),
            "is_primary": abs(multiple - primary_stop_multiple) < 1e-9,
        })
    strategy = str((ticket or {}).get("strategy") or "").lower()
    if strategy == "bull_put":
        breach_rule_default = "If spot breaks below the short strike, defend or exit. Do not wait for max loss."
    elif strategy == "bear_call":
        breach_rule_default = "If spot breaks above the short strike, defend or exit. Do not wait for max loss."
    else:
        breach_rule_default = "If the short strike is breached, defend or exit quickly."
    breach_rule = (overrides.get("breach_rule") or breach_rule_default).strip()
    hard_rule = (overrides.get("hard_rule") or "Do not hold to full loss. A small win is better than a full loser.").strip()
    primary_stop = next((stop for stop in debit_stops if stop["is_primary"]), debit_stops[0])

    return {
        "profit_target_pct": round(profit_target_pct, 2),
        "profit_target_amount": profit_target_amount,
        "primary_stop_multiple": primary_stop_multiple,
        "debit_stops": debit_stops,
        "breach_rule": breach_rule,
        "hard_max_loss": round(max_loss, 2),
        "hard_rule": hard_rule,
        "summary": (
            f"Take profit near ${profit_target_amount:,.2f}. "
            f"Exit or defend on short-strike breach. "
            f"Primary debit stop at ${primary_stop['buyback_debit_total']:,.2f}."
        ),
    }


def build_ticket(symbol, strategy, expiry, trade_row):
    ticket = {
        "id": dt.datetime.now().strftime("%Y%m%d%H%M%S"),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "strategy": strategy,
        "expiry": expiry,
        "status": "draft",
        "trade": trade_row,
        "legs": [],
    }
    if strategy == "iron_condor":
        legs = []
        legs += parse_legs_from_trade(trade_row.get("Put Spread"))
        legs += parse_legs_from_trade(trade_row.get("Call Spread"))
        ticket["legs"] = legs
    else:
        ticket["legs"] = parse_legs_from_trade(trade_row.get("Trade"))
    ticket["management_plan"] = build_management_plan(ticket, trade_row=trade_row)
    return ticket


def build_risk_summary(positions, cash_total=0.0):
    invested_mv = 0.0
    total_pnl = 0.0
    by_type = {}
    by_symbol = {}
    dte_buckets = {"<=7": 0, "<=14": 0, "<=30": 0, ">30": 0, "unknown": 0}

    for pos in positions:
        mv = _safe_float(pos.get("marketValue")) or 0.0
        pnl = _safe_float(pos.get("pnl")) or 0.0
        invested_mv += mv
        total_pnl += pnl

        asset_type = pos.get("assetType") or "UNKNOWN"
        by_type.setdefault(asset_type, {"marketValue": 0.0, "pnl": 0.0, "count": 0})
        by_type[asset_type]["marketValue"] += mv
        by_type[asset_type]["pnl"] += pnl
        by_type[asset_type]["count"] += 1

        symbol = pos.get("symbol") or "UNKNOWN"
        by_symbol.setdefault(symbol, {"symbol": symbol, "marketValue": 0.0, "pnl": 0.0, "count": 0})
        by_symbol[symbol]["marketValue"] += mv
        by_symbol[symbol]["pnl"] += pnl
        by_symbol[symbol]["count"] += 1

        dte = pos.get("dte")
        if dte is None:
            dte_buckets["unknown"] += 1
        elif dte <= 7:
            dte_buckets["<=7"] += 1
        elif dte <= 14:
            dte_buckets["<=14"] += 1
        elif dte <= 30:
            dte_buckets["<=30"] += 1
        else:
            dte_buckets[">30"] += 1

    top_positions = sorted(
        positions, key=lambda p: abs(_safe_float(p.get("marketValue")) or 0.0), reverse=True
    )[:10]
    symbols_sorted = sorted(by_symbol.values(), key=lambda x: abs(x["marketValue"]), reverse=True)[:10]
    cash_val = _safe_float(cash_total) or 0.0
    if cash_val:
        by_type.setdefault("CASH", {"marketValue": 0.0, "pnl": 0.0, "count": 0})
        by_type["CASH"]["marketValue"] += cash_val
        by_type["CASH"]["count"] += 1

    total_mv = invested_mv + cash_val

    return {
        "invested_mv": invested_mv,
        "cash_total": cash_val,
        "total_mv": total_mv,
        "total_pnl": total_pnl,
        "by_type": by_type,
        "by_symbol": symbols_sorted,
        "dte_buckets": dte_buckets,
        "top_positions": top_positions,
    }


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_put_price(spot, strike, t_years, rate, sigma):
    spot = max(_safe_float(spot) or 0.0, 0.01)
    strike = max(_safe_float(strike) or 0.0, 0.01)
    t_years = max(_safe_float(t_years) or 0.0, 0.0)
    sigma = max(_safe_float(sigma) or 0.0, 0.0)
    if t_years <= 0.0 or sigma <= 0.0:
        return max(0.0, strike - spot)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return max(0.0, (strike * math.exp(-rate * t_years) * _norm_cdf(-d2)) - (spot * _norm_cdf(-d1)))


def _bs_call_price(spot, strike, t_years, rate, sigma):
    spot = max(_safe_float(spot) or 0.0, 0.01)
    strike = max(_safe_float(strike) or 0.0, 0.01)
    t_years = max(_safe_float(t_years) or 0.0, 0.0)
    sigma = max(_safe_float(sigma) or 0.0, 0.0)
    if t_years <= 0.0 or sigma <= 0.0:
        return max(0.0, spot - strike)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return max(0.0, (spot * _norm_cdf(d1)) - (strike * math.exp(-rate * t_years) * _norm_cdf(d2)))


def _option_iv(quote, fallback=0.25):
    iv = _safe_float(
        (quote or {}).get("volatility")
        or (quote or {}).get("volatilityInPercent")
        or (quote or {}).get("iv")
    )
    if iv is None or iv <= 0:
        return fallback
    if iv > 1.0:
        iv = iv / 100.0
    return min(max(iv, 0.01), 5.0)


def _put_intrinsic(spot, strike):
    return max(0.0, strike - spot)


def _call_intrinsic(spot, strike):
    return max(0.0, spot - strike)


def build_put_spread_open_rows(positions, quotes, underlying_quotes, step, span, pricing_mode="model"):
    rate = _safe_float(os.getenv("RISK_FREE_RATE")) or 0.045
    levels = [round(i * step, 2) for i in range(-span, span + 1)]
    mode = (pricing_mode or "model").strip().lower()
    if mode not in {"model", "expiry"}:
        mode = "model"
    grouped = {}
    for pos in positions:
        if pos.get("assetType") != "OPTION":
            continue
        symbol = pos.get("symbol")
        if not symbol:
            continue
        instrument = pos.get("instrument", {}) or {}
        symbol_meta = parse_option_symbol(symbol)
        opt_type = (instrument.get("putCall") or instrument.get("optionType") or symbol_meta.get("type") or "").upper()
        if opt_type != "PUT":
            continue
        strike = _safe_float(instrument.get("strikePrice"))
        if strike is None:
            strike = _safe_float(symbol_meta.get("strike"))
        underlying = instrument.get("underlyingSymbol") or symbol_meta.get("underlying")
        expiration = parse_expiration(instrument.get("expirationDate") or instrument.get("maturityDate")) or symbol_meta.get("expiry")
        qty = _safe_float(pos.get("qty")) or 0.0
        if strike is None or not underlying or expiration is None or qty == 0:
            continue
        avg_price = _safe_float(
            pos.get("avgPrice")
            or pos.get("averagePrice")
            or pos.get("averageLongPrice")
            or pos.get("averageShortPrice")
        ) or 0.0
        quote = quotes.get(symbol, {}) or {}
        mark = _safe_float(quote.get("mark") or quote.get("last") or quote.get("bid") or quote.get("ask")) or 0.0
        leg = {
            "symbol": symbol,
            "strike": strike,
            "qty": qty,
            "remaining": abs(qty),
            "avg_price": avg_price,
            "mark": mark,
            "iv": _option_iv(quote),
        }
        key = (underlying, expiration.isoformat())
        grouped.setdefault(key, {"shorts": [], "longs": []})
        if qty < 0:
            grouped[key]["shorts"].append(leg)
        else:
            grouped[key]["longs"].append(leg)

    rows = []
    for (underlying, expiry_iso), bucket in grouped.items():
        shorts = sorted(bucket.get("shorts", []), key=lambda x: x["strike"], reverse=True)
        longs = sorted(bucket.get("longs", []), key=lambda x: x["strike"], reverse=True)
        expiry_date = parse_expiration(expiry_iso)
        t_years = 0.0
        if expiry_date:
            t_years = max((expiry_date - dt.date.today()).days, 0) / 365.0
        under_quote = underlying_quotes.get(underlying, {}) or {}
        underlying_last = _safe_float(
            under_quote.get("last") or under_quote.get("mark") or under_quote.get("bid") or under_quote.get("ask")
        )

        for short_leg in shorts:
            while short_leg["remaining"] > 0:
                candidates = [leg for leg in longs if leg["remaining"] > 0 and leg["strike"] < short_leg["strike"]]
                if not candidates:
                    break
                long_leg = max(candidates, key=lambda x: x["strike"])
                spread_qty = min(short_leg["remaining"], long_leg["remaining"])
                short_leg["remaining"] -= spread_qty
                long_leg["remaining"] -= spread_qty

                entry_credit = short_leg["avg_price"] - long_leg["avg_price"]
                current_spread_value = short_leg["mark"] - long_leg["mark"]
                current_pl = (entry_credit - current_spread_value) * spread_qty * 100
                entry_credit_total = entry_credit * spread_qty * 100
                pl_pct = None
                if entry_credit_total:
                    pl_pct = (current_pl / abs(entry_credit_total)) * 100
                breakeven = short_leg["strike"] - entry_credit

                pl_levels = {}
                for delta in levels:
                    level_price = short_leg["strike"] + delta
                    if mode == "expiry":
                        short_val = _put_intrinsic(level_price, short_leg["strike"])
                        long_val = _put_intrinsic(level_price, long_leg["strike"])
                    else:
                        short_val = _bs_put_price(level_price, short_leg["strike"], t_years, rate, short_leg["iv"])
                        long_val = _bs_put_price(level_price, long_leg["strike"], t_years, rate, long_leg["iv"])
                    spread_value = short_val - long_val
                    pl_levels[delta] = round((entry_credit - spread_value) * spread_qty * 100, 2)

                rows.append({
                    "underlying": underlying,
                    "expiry": expiry_iso,
                    "short_symbol": short_leg["symbol"],
                    "long_symbol": long_leg["symbol"],
                    "short_strike": round(short_leg["strike"], 2),
                    "long_strike": round(long_leg["strike"], 2),
                    "width": round(short_leg["strike"] - long_leg["strike"], 2),
                    "qty": round(spread_qty, 2),
                    "entry_credit": round(entry_credit, 4),
                    "mark_spread": round(current_spread_value, 4),
                    "breakeven": round(breakeven, 2),
                    "underlying_last": underlying_last,
                    "current_pl": round(current_pl, 2),
                    "pl_pct": round(pl_pct, 2) if pl_pct is not None else None,
                    "pl_levels": pl_levels,
                })

    rows.sort(key=lambda r: (r["underlying"], r["expiry"], -r["short_strike"], -r["long_strike"]))
    return levels, rows


def get_account_hashes():
    now = time.time()
    cached = _ACCOUNT_HASH_CACHE.get("value", {})
    if cached and (now - _ACCOUNT_HASH_CACHE.get("ts", 0.0)) <= _ACCOUNT_HASH_CACHE_TTL:
        return cached

    client = get_client()
    response = schwab_request(lambda: client.get_account_numbers(), "get_account_numbers")
    response.raise_for_status()
    data = response.json()
    hashes = {row.get("accountNumber"): row.get("hashValue") for row in data}
    _ACCOUNT_HASH_CACHE["ts"] = now
    _ACCOUNT_HASH_CACHE["value"] = hashes
    return hashes


def fetch_transactions_one_year():
    now = time.time()
    cached = _TRANSACTIONS_CACHE.get("value", [])
    if cached and (now - _TRANSACTIONS_CACHE.get("ts", 0.0)) <= _TRANSACTIONS_CACHE_TTL:
        return cached

    client = get_client()
    end_date = dt.datetime.now(dt.timezone.utc)
    # Yearly Summary should be year-to-date (calendar year), not rolling 365 days.
    start_date = dt.datetime(end_date.year, 1, 1, tzinfo=dt.timezone.utc)
    account_hashes = get_account_hashes()
    all_txns = []
    deadline = time.monotonic() + _TRANSACTIONS_FETCH_BUDGET_SEC

    def _txn_key(txn, acct_num):
        item = (txn or {}).get("transactionItem", {}) or {}
        instr = (item.get("instrument") or {}) if isinstance(item, dict) else {}
        return (
            str(acct_num or ""),
            str((txn or {}).get("transactionId") or ""),
            str((txn or {}).get("transactionDate") or (txn or {}).get("tradeDate") or ""),
            str((txn or {}).get("transactionType") or ""),
            str(item.get("instruction") or ""),
            str(instr.get("symbol") or instr.get("underlyingSymbol") or item.get("symbol") or (txn or {}).get("symbol") or ""),
        )

    best_by_key = {}

    def _txn_quality(txn):
        item = (txn or {}).get("transactionItem", {}) or {}
        return (
            (1 if _safe_float((txn or {}).get("netAmount")) not in (None, 0) else 0)
            + (1 if _safe_float(item.get("cost")) not in (None, 0) else 0)
            + (1 if _safe_float(item.get("price")) not in (None, 0) else 0)
            + (1 if _safe_float(item.get("amount")) not in (None, 0) else 0)
            + (1 if _safe_float(item.get("quantity")) not in (None, 0) else 0)
            + (1 if bool((txn or {}).get("transferItems")) else 0)
        )

    def _ingest_txn(txn, acct_num):
        txn["accountNumber"] = acct_num
        key = _txn_key(txn, acct_num)
        current = best_by_key.get(key)
        if current is None or _txn_quality(txn) > _txn_quality(current):
            best_by_key[key] = txn

    for acct_num, acct_hash in account_hashes.items():
        if time.monotonic() >= deadline:
            break
        # Pull newest-first so timeout budget still returns the most recent activity.
        window_end = end_date
        while window_end > start_date:
            if time.monotonic() >= deadline:
                break
            window_start = max(start_date, window_end - dt.timedelta(days=59))
            try:
                # Pull default transaction view.
                response = schwab_request(
                    lambda: client.get_transactions(
                        acct_hash,
                        start_date=window_start,
                        end_date=window_end,
                    ),
                    "get_transactions",
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    for txn in data:
                        _ingest_txn(txn, acct_num)

                # Pull TRADE-only view as well. Schwab sometimes places the
                # usable option cashflow here while default rows show zeros.
                trade_response = schwab_request(
                    lambda: client.get_transactions(
                        acct_hash,
                        start_date=window_start,
                        end_date=window_end,
                        transaction_types=[client.Transactions.TransactionType.TRADE],
                    ),
                    "get_transactions_trade_only",
                )
                trade_response.raise_for_status()
                trade_data = trade_response.json()
                if isinstance(trade_data, list):
                    for txn in trade_data:
                        _ingest_txn(txn, acct_num)
            except Exception as exc:
                log_error(exc, context="fetch_transactions_one_year")
                if cached:
                    return cached
                break
            window_end = window_start - dt.timedelta(seconds=1)

    all_txns = list(best_by_key.values())
    if all_txns:
        _TRANSACTIONS_CACHE["ts"] = now
        _TRANSACTIONS_CACHE["value"] = all_txns
    elif cached:
        return cached
    return all_txns


def _txn_symbol(txn):
    item = txn.get("transactionItem", {}) or {}
    instr = item.get("instrument", {}) or {}
    if instr.get("symbol"):
        return instr.get("symbol")
    if instr.get("underlyingSymbol"):
        return instr.get("underlyingSymbol")
    if item.get("symbol"):
        return item.get("symbol")
    if txn.get("symbol"):
        return txn.get("symbol")
    for transfer in txn.get("transferItems", []) or []:
        t_instr = (transfer or {}).get("instrument", {}) or {}
        if t_instr.get("symbol"):
            return t_instr.get("symbol")
        if t_instr.get("underlyingSymbol"):
            return t_instr.get("underlyingSymbol")

    # Some option transaction payloads omit instrument.symbol but include a concise description.
    # Example: "SNDK 02/27/2026 577.50 P"
    desc_candidates = [
        instr.get("description"),
        item.get("description"),
        txn.get("description"),
    ]
    for raw in desc_candidates:
        text = str(raw or "").strip().upper()
        if not text:
            continue
        if parse_option_symbol(text):
            return text
        if re.match(r"^[A-Z]{1,6}\s+\d{1,2}/\d{1,2}/\d{2,4}\s+\d+(\.\d+)?\s+[CP]$", text):
            return text
    return None


def _txn_qty(txn):
    item = txn.get("transactionItem", {}) or {}
    instruction = _txn_instruction(txn)
    asset_type = _txn_asset_type(txn)

    # For options, Schwab may place non-contract values in "amount" (e.g. 0.01).
    # Prefer explicit contract quantity fields first.
    if asset_type == "OPTION":
        qty = (
            _safe_float(item.get("quantity"))
            or _safe_float(item.get("quantityNumber"))
            or _safe_float(item.get("amount"))
        )
    else:
        qty = (
            _safe_float(item.get("amount"))
            or _safe_float(item.get("quantity"))
            or _safe_float(item.get("quantityNumber"))
        )
    if qty is None:
        transfer_qty = 0.0
        found_transfer_qty = False
        for transfer in txn.get("transferItems", []) or []:
            t_qty = (
                _safe_float((transfer or {}).get("amount"))
                or _safe_float((transfer or {}).get("quantity"))
                or _safe_float((transfer or {}).get("quantityNumber"))
            )
            if t_qty is None:
                continue
            transfer_qty += t_qty
            found_transfer_qty = True
        if found_transfer_qty:
            qty = transfer_qty

    if qty is None:
        return 0.0

    sell_instructions = {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}
    buy_instructions = {"BUY", "BUY_TO_OPEN", "BUY_TO_CLOSE", "BUY_TO_COVER"}
    if instruction in sell_instructions:
        return -abs(qty)
    if instruction in buy_instructions:
        return abs(qty)
    return qty


def _sum_numeric_values(obj):
    total = 0.0
    if isinstance(obj, dict):
        for val in obj.values():
            num = _safe_float(val)
            if num is not None:
                total += num
    return total


def _compute_cashflow_from_item(txn):
    item = txn.get("transactionItem", {}) or {}
    instruction = _txn_instruction(txn)
    asset_type = _txn_asset_type(txn)

    if asset_type == "OPTION":
        qty = _safe_float(item.get("quantity")) or _safe_float(item.get("quantityNumber"))
        if qty is None:
            qty = _safe_float(item.get("amount"))
    else:
        qty = _safe_float(item.get("amount"))
        if qty is None:
            qty = _safe_float(item.get("quantity")) or _safe_float(item.get("quantityNumber"))
    price = _safe_float(item.get("price"))
    fees = abs(_sum_numeric_values(txn.get("fees") or {})) + abs(_sum_numeric_values(item.get("fees") or {}))
    cost = _safe_float(item.get("cost"))
    if cost is not None and cost != 0:
        return cost - fees

    if qty is None or price is None:
        return None

    multiplier = 100.0 if asset_type == "OPTION" else 1.0
    gross = abs(qty) * abs(price) * multiplier

    sell_instructions = {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}
    buy_instructions = {"BUY", "BUY_TO_OPEN", "BUY_TO_CLOSE", "BUY_TO_COVER"}
    if instruction in sell_instructions:
        return gross - fees
    if instruction in buy_instructions:
        return -gross - fees

    return None


def _compute_cashflow_from_transfers(txn):
    total = 0.0
    found = False
    asset_type = _txn_asset_type(txn)
    for transfer in txn.get("transferItems", []) or []:
        t = transfer or {}
        # For options, transfer "amount" is often contract count (e.g. 1),
        # while "cost"/"netAmount" carries dollars.
        if asset_type == "OPTION":
            keys = ("netAmount", "cost", "amount")
        else:
            keys = ("netAmount", "cost", "amount")
        for key in keys:
            num = _safe_float(t.get(key))
            if num is None:
                continue
            total += num
            found = True
            break
    if found:
        return total
    return None


def _extract_price_from_text(text):
    raw = str(text or "")
    if not raw:
        return None
    # Common broker text formats: "... @ 6.92" or "... @ $6.92"
    match = re.search(r"@\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", raw, re.IGNORECASE)
    if match:
        val = _safe_float(match.group(1))
        if val is not None and 0 < val < 100:
            return val
        return None
    # Conservative fallback for cases like "PRICE 6.92" / "AT 6.92".
    match = re.search(r"\b(?:PRICE|AT)\s+\$?\s*([0-9]+(?:\.[0-9]+)?)\b", raw, re.IGNORECASE)
    if match:
        val = _safe_float(match.group(1))
        if val is not None and 0 < val < 100:
            return val
    return None


def _compute_cashflow_from_description(txn):
    desc = " ".join([
        str(txn.get("description") or ""),
        str((txn.get("transactionItem") or {}).get("description") or ""),
    ]).strip()
    if not desc:
        return None

    asset_type = _txn_asset_type(txn)
    if asset_type != "OPTION":
        return None

    qty = _txn_qty(txn)
    if not qty:
        return None

    instruction = _txn_instruction(txn)
    desc_up = desc.upper()
    if not instruction:
        if "BUY TO OPEN" in desc_up or "BUY TO CLOSE" in desc_up or "BUY TO COVER" in desc_up:
            instruction = "BUY"
        elif "SELL TO OPEN" in desc_up or "SELL TO CLOSE" in desc_up or "SELL SHORT" in desc_up:
            instruction = "SELL"

    price = _extract_price_from_text(desc)
    if price is None:
        return None

    gross = abs(qty) * abs(price) * 100.0
    fees = abs(_sum_numeric_values(txn.get("fees") or {})) + abs(_sum_numeric_values((txn.get("transactionItem") or {}).get("fees") or {}))

    sell_instructions = {"SELL", "SELL_SHORT", "SELL_TO_OPEN", "SELL_TO_CLOSE"}
    buy_instructions = {"BUY", "BUY_TO_OPEN", "BUY_TO_CLOSE", "BUY_TO_COVER"}
    if instruction in sell_instructions:
        return gross - fees
    if instruction in buy_instructions:
        return -gross - fees

    # If instruction is still unknown, infer sign from qty convention.
    # In our summary qty > 0 is buy-like and qty < 0 is sell-like.
    return (-gross - fees) if qty > 0 else (gross - fees)


def _txn_pnl(txn):
    net = _safe_float(txn.get("netAmount"))

    # Some Schwab transaction payloads report option trade economics only as
    # instruction/quantity/price (+fees), with netAmount omitted or zero.
    item_cashflow = _compute_cashflow_from_item(txn)
    transfer_cashflow = _compute_cashflow_from_transfers(txn)
    desc_cashflow = _compute_cashflow_from_description(txn)

    candidates = [v for v in [net, item_cashflow, transfer_cashflow, desc_cashflow] if v is not None and v != 0]
    if candidates:
        if _txn_asset_type(txn) == "OPTION":
            # Option payload variants can disagree; prefer the largest-magnitude
            # non-zero candidate to avoid under-reporting as +/-1.
            return max(candidates, key=lambda x: abs(x))
        # For non-options, prefer explicit net amount first.
        if net is not None and net != 0:
            return net
        return candidates[0]

    # Preserve explicit zero if present; otherwise default to 0.
    return net or 0.0


def _txn_date(txn, tz_name="America/New_York"):
    def _to_local_date(raw):
        if not raw:
            return None
        try:
            parsed = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                try:
                    parsed = parsed.replace(tzinfo=ZoneInfo(tz_name))
                except Exception:
                    pass
            try:
                parsed = parsed.astimezone(ZoneInfo(tz_name))
            except Exception:
                pass
            return parsed.date()
        except Exception:
            return None

    trade_like_fields = [
        txn.get("transactionDate"),
        txn.get("tradeDate"),
        txn.get("activityDate"),
        txn.get("orderDate"),
    ]
    settlement_fields = [
        txn.get("settlementDate"),
        txn.get("effectiveDate"),
    ]
    trade_like_dates = [d for d in (_to_local_date(v) for v in trade_like_fields) if d is not None]
    settlement_dates = [d for d in (_to_local_date(v) for v in settlement_fields) if d is not None]

    txn_type = str(txn.get("transactionType") or "").strip().upper()
    instruction = _txn_instruction(txn)
    is_trade_activity = (
        "TRADE" in txn_type
        or instruction in {
            "BUY",
            "SELL",
            "BUY_TO_OPEN",
            "SELL_TO_OPEN",
            "BUY_TO_CLOSE",
            "SELL_TO_CLOSE",
            "SELL_SHORT",
            "BUY_TO_COVER",
        }
    )

    # For trade activity, use the earliest trade-like date when available.
    if is_trade_activity and trade_like_dates:
        return min(trade_like_dates)
    if trade_like_dates:
        return trade_like_dates[0]
    if settlement_dates:
        return settlement_dates[0]
    try:
        return dt.datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        return dt.date.today()


def _summary_asset_type(raw_asset_type, symbol=None):
    asset_text = str(raw_asset_type or "").strip().upper()
    if "CASH" in asset_text:
        return "CASH"
    if "OPTION" in asset_text:
        return "OPTION"
    if parse_option_symbol(symbol):
        return "OPTION"
    if asset_text in {"EQUITY", "ETF", "MUTUAL_FUND"}:
        return "STOCK"
    if asset_text in {"STOCK", "EQUITY_OR_INDEX"}:
        return "STOCK"
    return asset_text or "OTHER"


def fetch_account_balance_totals():
    client = get_client()
    response = schwab_request(lambda: client.get_accounts(), "get_accounts_balances")
    response.raise_for_status()
    data = response.json()
    totals = {"cash": 0.0, "liquidation": 0.0, "equity": 0.0}
    for entry in data or []:
        acct = (entry or {}).get("securitiesAccount", {}) or {}
        bal = acct.get("currentBalances", {}) or {}
        totals["cash"] += _safe_float(bal.get("cashBalance")) or 0.0
        totals["liquidation"] += _safe_float(bal.get("liquidationValue")) or 0.0
        totals["equity"] += _safe_float(bal.get("equity")) or 0.0
    return totals


def _txn_asset_type(txn):
    item = txn.get("transactionItem", {}) or {}
    instr = item.get("instrument", {}) or {}
    raw_asset = instr.get("assetType", "")
    symbol = _txn_symbol(txn)
    return _summary_asset_type(raw_asset, symbol)


def _txn_instruction(txn):
    item = txn.get("transactionItem", {}) or {}
    return str(item.get("instruction") or txn.get("transactionSubType") or "").strip().upper()


def _include_in_closed_summary(txn):
    txn_type = str(txn.get("transactionType") or "").strip().upper()
    asset_type = _txn_asset_type(txn)
    instruction = _txn_instruction(txn)
    desc = " ".join([
        str(txn.get("description") or ""),
        str((txn.get("transactionItem") or {}).get("description") or ""),
    ]).upper()
    net_amount = _txn_pnl(txn)

    if "TRADE" in txn_type:
        return True
    if instruction in {
        "BUY",
        "SELL",
        "BUY_TO_OPEN",
        "SELL_TO_OPEN",
        "BUY_TO_CLOSE",
        "SELL_TO_CLOSE",
        "SELL_SHORT",
        "BUY_TO_COVER",
    }:
        return True
    if any(
        phrase in desc
        for phrase in (
            "BUY TO OPEN",
            "SELL TO OPEN",
            "BUY TO CLOSE",
            "SELL TO CLOSE",
            "ASSIGNMENT",
            "EXERCISE",
            "EXPIRATION",
        )
    ):
        return True
    if asset_type == "OPTION" and net_amount != 0:
        return True
    return False


def _section_totals(rows):
    totals = {"qty": 0.0, "profit": 0.0, "loss": 0.0, "pnl": 0.0}
    for row in rows:
        totals["qty"] += _safe_float(row.get("qty")) or 0.0
        totals["profit"] += _safe_float(row.get("profit")) or 0.0
        totals["loss"] += _safe_float(row.get("loss")) or 0.0
        totals["pnl"] += _safe_float(row.get("pnl")) or 0.0
    return totals


def bucket_period(date_val, period):
    if period == "week":
        year, week, _ = date_val.isocalendar()
        return f"{year}-W{week:02d}"
    if period == "month":
        return date_val.strftime("%Y-%m")
    if period == "quarter":
        q = (date_val.month - 1) // 3 + 1
        return f"{date_val.year}-Q{q}"
    return str(date_val.year)


def period_bucket_to_year_month(bucket, period):
    text = str(bucket or "")
    try:
        if period == "month":
            year_str, month_str = text.split("-", 1)
            return int(year_str), int(month_str)
        if period == "quarter":
            year_str, q_str = text.split("-Q", 1)
            q = int(q_str)
            month = ((q - 1) * 3) + 1
            return int(year_str), month
        if period == "week":
            year_str, week_str = text.split("-W", 1)
            d = dt.date.fromisocalendar(int(year_str), int(week_str), 1)
            return d.year, d.month
        if period == "year":
            return int(text), None
    except Exception:
        return None, None
    return None, None


def build_yearly_summary(period, status_filter, tz_name="America/New_York"):
    entries = []
    try:
        today = dt.datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        today = dt.date.today()

    # Open positions (unrealized P/L)
    positions = fetch_positions()
    for pos in positions:
        if status_filter in ("closed",):
            continue
        symbol = pos.get("symbol") or "UNKNOWN"
        entries.append({
            "symbol": symbol,
            "assetType": _summary_asset_type(pos.get("assetType"), symbol),
            "qty": _safe_float(pos.get("qty")) or 0.0,
            "pnl": _safe_float(pos.get("pnl")) or 0.0,
            "status": "open",
            "date": today,
        })

    # Cash balance as an open row so yearly summary reflects account cash.
    if status_filter in ("all", "open"):
        balances = fetch_account_balance_totals()
        cash_total = _safe_float(balances.get("cash")) or 0.0
        entries.append({
            "symbol": "CASH",
            "assetType": "CASH",
            "qty": cash_total,
            "pnl": 0.0,
            "status": "open",
            "date": today,
        })

    # Closed trades (net cash flow)
    if status_filter in ("all", "closed"):
        for txn in fetch_transactions_one_year():
            if not _include_in_closed_summary(txn):
                continue
            symbol = _txn_symbol(txn)
            if not symbol:
                continue
            asset_type = _txn_asset_type(txn)
            entries.append({
                "symbol": symbol,
                "assetType": asset_type,
                "qty": _txn_qty(txn),
                "pnl": _txn_pnl(txn),
                "status": "closed",
                "date": _txn_date(txn, tz_name=tz_name),
            })

    # Group by period + symbol
    grouped = {}
    for entry in entries:
        if status_filter != "all" and entry["status"] != status_filter:
            continue
        bucket = bucket_period(entry["date"], period)
        key = (bucket, entry["symbol"], entry["status"], entry["assetType"])
        if key not in grouped:
            grouped[key] = {
                "period": bucket,
                "symbol": entry["symbol"],
                "status": entry["status"],
                "assetType": entry["assetType"],
                "qty": 0.0,
                "pnl": 0.0,
            }
        grouped[key]["qty"] += entry["qty"]
        grouped[key]["pnl"] += entry["pnl"]

    rows = list(grouped.values())
    for row in rows:
        pnl = _safe_float(row.get("pnl")) or 0.0
        row["profit"] = pnl if pnl > 0 else 0.0
        row["loss"] = pnl if pnl < 0 else 0.0
    return rows


def format_results(df):
    if df is None or df.empty:
        return [], []
    view = df.copy()

    money_cols = ["Credit (Realistic)", "Credit (Mid $)", "Credit (Natural $)", "Total Credit ($)", "Max Loss ($)", "Breakeven", "Spot"]
    percent_cols = ["POP %", "Distance %"]

    for col in money_cols:
        if col in view.columns:
            view[col] = view[col].apply(
                lambda x: f"${float(x):,.2f}" if str(x).replace('.', '', 1).isdigit() else x
            )
    for col in percent_cols:
        if col in view.columns:
            view[col] = view[col].apply(
                lambda x: f"{float(x):.1f}%" if str(x).replace('.', '', 1).isdigit() else x
            )

    columns = list(view.columns)
    rows = view.to_dict(orient="records")
    return columns, rows

@app.route("/callback")
def callback():
    from schwab.auth import easy_client

    api_key = _get_env("SCHWAB_API_KEY")
    app_secret = _get_env("SCHWAB_APP_SECRET")
    callback_url = _get_env("SCHWAB_CALLBACK_URL")
    token_path = "token.json"

    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=callback_url,
            token_path=token_path,
            enforce_enums=False,
        )
        session["authed"] = True
        return redirect(url_for("index"))
    except Exception as exc:
        log_error(exc, context="callback")
        return f"Callback error: {exc}", 500

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if session.get("authed"):
        return redirect(url_for("index"))
    if not _auth_ready():
        error = "Auth is not configured. Set AUTH_USERNAME and AUTH_PASSWORD_HASH."
        return render_template("login.html", error=error)
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        auth = _get_auth_config()
        if username == auth["username"] and check_password_hash(auth["password_hash"], password):
            session["authed"] = True
            next_url = request.args.get("next")
            return redirect(next_url or url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    error = None
    summary = []
    try:
        client = get_client()
        response = client.get_accounts()
        response.raise_for_status()
        data = response.json()
        for entry in data:
            account = entry.get("securitiesAccount", {})
            balances = account.get("currentBalances", {})
            summary.append({
                "accountNumber": account.get("accountNumber"),
                "type": account.get("type"),
                "liquidationValue": balances.get("liquidationValue"),
                "cashBalance": balances.get("cashBalance"),
                "equity": balances.get("equity"),
            })
    except Exception as exc:
        error = str(exc)

    return render_template("index.html", summary=summary, error=error)


@app.route("/positions")
def positions():
    settings = load_settings()
    error = None
    rows = []
    try:
        positions = fetch_positions()
        symbols = sorted({p.get("symbol") for p in positions if p.get("symbol")})
        quotes = fetch_quotes(symbols)
        for pos in positions:
            pos_symbol = pos.get("symbol")
            if pos_symbol in quotes:
                pos["quote"] = quotes[pos_symbol]
            action, reasons, note = evaluate_position(pos, settings)
            pos["action"] = action
            pos["reasons"] = "; ".join(reasons)
            pos["note"] = note
            rows.append(pos)
    except Exception as exc:
        error = str(exc)

    return render_template("positions.html", rows=rows, error=error)


@app.route("/alerts", methods=["GET", "POST"])
def alerts():
    settings = load_settings()
    if request.method == "POST":
        settings["profit_target_pct"] = float(request.form.get("profit_target_pct", settings["profit_target_pct"]))
        settings["max_loss_pct"] = float(request.form.get("max_loss_pct", settings["max_loss_pct"]))
        settings["dte_exit"] = int(request.form.get("dte_exit", settings["dte_exit"]))
        settings["nova_judgment"] = request.form.get("nova_judgment") == "on"
        settings["polling_minutes"] = int(request.form.get("polling_minutes", settings["polling_minutes"]))
        requested_model = request.form.get("nova_model", settings["nova_model"])
        if requested_model in _nova_model_options(settings["nova_model"]):
            settings["nova_model"] = requested_model
        settings["nova_role"] = request.form.get("nova_role", settings.get("nova_role", "")).strip()
        settings["monitor_market_hours_only"] = request.form.get("monitor_market_hours_only") == "on"
        settings["monitor_weekdays_only"] = request.form.get("monitor_weekdays_only") == "on"
        settings["market_open_time"] = request.form.get("market_open_time", settings["market_open_time"]).strip()
        settings["market_close_time"] = request.form.get("market_close_time", settings["market_close_time"]).strip()
        settings["market_holidays"] = request.form.get("market_holidays", settings.get("market_holidays", "")).strip()
        settings["movers_lookback_days"] = int(request.form.get(
            "movers_lookback_days",
            settings.get("movers_lookback_days", 5),
        ))
        settings["movers_count"] = int(request.form.get(
            "movers_count",
            settings.get("movers_count", 10),
        ))
        settings["movers_include_positions"] = request.form.get("movers_include_positions") == "on"
        settings["movers_universe"] = request.form.get("movers_universe", settings.get("movers_universe", "")).strip()
        settings["movers_use_sp500"] = request.form.get("movers_use_sp500") == "on"
        settings["movers_max_seconds"] = int(request.form.get(
            "movers_max_seconds",
            settings.get("movers_max_seconds", 20),
        ))
        save_settings(settings)
        return redirect(url_for("alerts"))

    error = None
    try:
        with _ALERTS_LOCK:
            rows = ALERTS or load_alerts()
    except Exception as exc:
        rows = []
        error = str(exc)

    def _fmt_num(value):
        num = _safe_float(value)
        return f"{num:,.2f}" if num is not None else ""

    def _fmt_money(value):
        num = _safe_float(value)
        return f"${num:,.2f}" if num is not None else ""

    def _fmt_pct(value):
        num = _safe_float(value)
        return f"{num:.2f}%" if num is not None else ""

    display_rows = []
    for row in rows:
        quote = (row.get("quote") or {}) if isinstance(row, dict) else {}
        display_rows.append({
            "accountNumber": (row.get("accountNumber") if isinstance(row, dict) else ""),
            "symbol": (row.get("symbol") if isinstance(row, dict) else ""),
            "assetType": (row.get("assetType") if isinstance(row, dict) else ""),
            "qty_fmt": _fmt_num(row.get("qty") if isinstance(row, dict) else None),
            "market_value_fmt": _fmt_money(row.get("marketValue") if isinstance(row, dict) else None),
            "pnl_fmt": _fmt_money(row.get("pnl") if isinstance(row, dict) else None),
            "pnl_pct_fmt": _fmt_pct(row.get("pnl_pct") if isinstance(row, dict) else None),
            "dte_fmt": _fmt_num(row.get("dte") if isinstance(row, dict) else None),
            "last_fmt": _fmt_money(quote.get("last")),
            "mark_fmt": _fmt_money(quote.get("mark")),
            "delta_fmt": _fmt_num(quote.get("delta")),
            "theta_fmt": _fmt_num(quote.get("theta")),
            "iv_fmt": _fmt_num(quote.get("iv")),
            "action": (row.get("action") if isinstance(row, dict) else ""),
            "reasons": (row.get("reasons") if isinstance(row, dict) else ""),
            "note": (row.get("note") if isinstance(row, dict) else ""),
        })

    return render_template(
        "alerts.html",
        rows=display_rows,
        error=error,
        settings=settings,
        nova_models=_nova_model_options(settings.get("nova_model")),
        email_ready=bool(email_settings()),
    )


@app.route("/watchlist", methods=["GET", "POST"])
def watchlist():
    error = None
    results = []
    watchlist_symbols = load_watchlist()
    watchlist_notes = load_watchlist_notes()

    if request.method == "POST":
        action = request.form.get("action", "")
        symbol = request.form.get("symbol", "").upper().strip()

        if action == "add" and symbol:
            if symbol not in watchlist_symbols:
                watchlist_symbols.append(symbol)
                watchlist_symbols = sorted(set(watchlist_symbols))
                save_watchlist(watchlist_symbols)
            return redirect(url_for("watchlist"))

        if action == "remove" and symbol:
            if symbol in watchlist_symbols:
                watchlist_symbols = [s for s in watchlist_symbols if s != symbol]
                save_watchlist(watchlist_symbols)
                if symbol in watchlist_notes:
                    watchlist_notes.pop(symbol, None)
                    save_watchlist_notes(watchlist_notes)
            return redirect(url_for("watchlist"))

        if action == "scan":
            strategy = request.form.get("strategy", "bull_put")
            width = float(request.form.get("width", 2.5))
            min_pop = float(request.form.get("min_pop", 80))
            contracts = int(request.form.get("contracts", 1))
            raw_mode = request.form.get("raw", "") == "1"
            cash_balance = int(request.form.get("cash", 2000))
            max_loss = float(request.form.get("max_loss", get_max_loss_threshold(cash_balance)))
            expiry = request.form.get("expiry", "").strip()

            for sym in watchlist_symbols:
                try:
                    if expiry:
                        use_expiry = expiry
                    else:
                        _, _, _, expirations = get_chain_data(sym)
                        use_expiry = expirations[0]["date"] if expirations else ""
                    results_df, spot_price, _ = run_scan(
                        sym, use_expiry, strategy, width, max_loss, min_pop, raw_mode, contracts
                    )
                    summary = summarize_scan(results_df)
                    results.append({
                        "symbol": sym,
                        "expiry": use_expiry,
                        "spot": spot_price,
                        "count": summary["count"],
                        "top": summary["top"],
                    })
                except Exception as exc:
                    results.append({
                        "symbol": sym,
                        "expiry": expiry,
                        "spot": None,
                        "count": 0,
                        "top": None,
                        "error": str(exc),
                    })

    return render_template(
        "watchlist.html",
        watchlist=watchlist_symbols,
        watchlist_notes=watchlist_notes,
        results=results,
        error=error,
    )


@app.route("/movers-agent", methods=["GET", "POST"])
def movers_agent():
    info = None
    error = None
    snapshot = load_movers_snapshot()
    universe_files = list_optionable_universe_files()
    selected_universe_file = (
        request.values.get("universe_file")
        or (snapshot or {}).get("universe_file")
        or OPTIONABLE_UNIVERSE_PATH
    ).strip()
    if selected_universe_file not in universe_files:
        selected_universe_file = (
            OPTIONABLE_UNIVERSE_PATH
            if OPTIONABLE_UNIVERSE_PATH in universe_files
            else (universe_files[0] if universe_files else OPTIONABLE_UNIVERSE_PATH)
        )
    universe = load_optionable_universe(selected_universe_file)
    lookback_days = int((snapshot or {}).get("lookback_days", 5))
    top_n = int((snapshot or {}).get("top_n", 10))
    max_seconds = int((snapshot or {}).get("max_seconds", 25))
    strategy = _normalize_movers_strategy((snapshot or {}).get("strategy", "bull_put"))

    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "run_scan":
            try:
                selected_universe_file = (request.form.get("universe_file") or selected_universe_file).strip()
                if selected_universe_file not in universe_files:
                    raise RuntimeError(f"Unknown universe file: {selected_universe_file}")
                universe = load_optionable_universe(selected_universe_file)
                lookback_days = max(1, int(request.form.get("lookback_days", 5)))
                top_n = max(1, int(request.form.get("top_n", 10)))
                max_seconds = max(10, int(request.form.get("max_seconds", 25)))
                strategy = _normalize_movers_strategy(request.form.get("strategy", strategy))
                if not universe:
                    raise RuntimeError(f"No optionable universe loaded from {selected_universe_file}.")
                previous_snapshot = snapshot
                if (snapshot or {}).get("universe_file") != selected_universe_file:
                    previous_snapshot = None
                snapshot = run_optionable_weekly_scan(
                    universe,
                    lookback_days=lookback_days,
                    top_n=top_n,
                    max_seconds=max_seconds,
                    strategy=strategy,
                    previous=previous_snapshot,
                )
                snapshot["universe_file"] = selected_universe_file
                save_movers_snapshot(snapshot)
                rows_count = len(snapshot.get("rows") or [])
                covered = int(snapshot.get("covered_count", 0))
                total = int(snapshot.get("universe_size", 0))
                label = "Bull Put" if strategy == "bull_put" else "Bear Call"
                info = f"Weekly scan completed ({label}). Returned {rows_count} symbols. Coverage {covered}/{total}."
            except Exception as exc:
                error = str(exc)
        elif action == "clear_snapshot":
            try:
                if os.path.exists(MOVERS_SNAPSHOT_PATH):
                    os.remove(MOVERS_SNAPSHOT_PATH)
                snapshot = None
                info = "Movers snapshot cleared."
            except Exception as exc:
                error = str(exc)
        elif action == "add_selected":
            selected = [s.upper().strip() for s in request.form.getlist("symbols") if s.strip()]
            if not selected:
                error = "No symbols selected."
            else:
                watchlist_symbols = set(load_watchlist())
                before = len(watchlist_symbols)
                watchlist_symbols.update(selected)
                save_watchlist(sorted(watchlist_symbols))
                added = len(watchlist_symbols) - before
                info = f"Added {added} symbol(s) to watchlist."
            snapshot = load_movers_snapshot()
            selected_universe_file = (snapshot or {}).get("universe_file") or selected_universe_file
            strategy = _normalize_movers_strategy((snapshot or {}).get("strategy", strategy))

    universe = load_optionable_universe(selected_universe_file)

    return render_template(
        "movers_agent.html",
        snapshot=snapshot,
        universe_size=len(universe),
        universe_files=universe_files,
        selected_universe_file=selected_universe_file,
        info=info,
        error=error,
        defaults={
            "lookback_days": lookback_days,
            "top_n": top_n,
            "max_seconds": max_seconds,
            "strategy": strategy,
        },
    )


@app.route("/tickets", methods=["GET", "POST"])
def tickets():
    error = None
    tickets_data = load_tickets()
    settings = load_settings()
    accounts = []
    try:
        client = get_client()
        response = client.get_accounts()
        response.raise_for_status()
        accounts = [
            entry.get("securitiesAccount", {}).get("accountNumber")
            for entry in response.json()
        ]
    except Exception:
        pass

    if request.method == "POST":
        ticket_id = request.form.get("ticket_id")
        action = request.form.get("action")
        if action == "clear":
            tickets_data = [t for t in tickets_data if t.get("id") != ticket_id]
            save_tickets(tickets_data)
            return redirect(url_for("tickets"))

        for ticket in tickets_data:
            if ticket.get("id") == ticket_id:
                if action == "confirm":
                    ticket["status"] = "confirmed"
                    ticket["confirmed_at"] = dt.datetime.now().isoformat(timespec="seconds")
                elif action == "send":
                    ticket["status"] = "send_requested"
                    ticket["send_requested_at"] = dt.datetime.now().isoformat(timespec="seconds")
                elif action == "update_plan":
                    overrides = {
                        "profit_target_pct": _safe_float(request.form.get("profit_target_pct")),
                        "primary_stop_multiple": _safe_float(request.form.get("primary_stop_multiple")),
                        "breach_rule": (request.form.get("breach_rule") or "").strip(),
                        "hard_rule": (request.form.get("hard_rule") or "").strip(),
                    }
                    overrides = {key: value for key, value in overrides.items() if value not in {None, ""}}
                    ticket["management_plan"] = build_management_plan(
                        ticket,
                        trade_row=ticket.get("trade"),
                        assumptions=ticket.get("assumptions"),
                        overrides=overrides,
                        settings=settings,
                    )
                save_tickets(tickets_data)
                break
        return redirect(url_for("tickets"))

    return render_template(
        "tickets.html",
        tickets=tickets_data,
        accounts=accounts,
        error=error,
    )


@app.route("/tickets/create", methods=["POST"])
def create_ticket():
    symbol = request.form.get("symbol", "NVDA").upper().strip()
    strategy = request.form.get("strategy", "bull_put")
    width = float(request.form.get("width", 2.5))
    min_pop = float(request.form.get("min_pop", 80))
    contracts = int(request.form.get("contracts", 1))
    raw_mode = request.form.get("raw", "") == "1"
    cash_balance = int(request.form.get("cash", 2000))
    max_loss = float(request.form.get("max_loss", get_max_loss_threshold(cash_balance)))
    expiry = request.form.get("expiry", "")
    selected_index = request.form.get("trade_index")

    try:
        results_df, _, _ = run_scan(
            symbol, expiry, strategy, width, max_loss, min_pop, raw_mode, contracts
        )
        results_records = []
        if results_df is not None and not results_df.empty:
            results_records = results_df.to_dict(orient="records")
        idx = int(selected_index) if selected_index is not None else None
        trade = results_records[idx] if idx is not None and idx < len(results_records) else None
        if trade is None:
            raise RuntimeError("Select a trade row to create a ticket.")
        ticket = build_ticket(symbol, strategy, expiry, trade)
        tickets_data = load_tickets()
        tickets_data.insert(0, ticket)
        save_tickets(tickets_data)
    except Exception as exc:
        global NOVA_OPTIONS_ERROR
        NOVA_OPTIONS_ERROR = str(exc)

    return redirect(url_for(
        "options_chain",
        symbol=symbol,
        expiry=expiry,
        strategy=strategy,
        width=width,
        min_pop=min_pop,
        contracts=contracts,
        cash=cash_balance,
        max_loss=max_loss,
        raw="1" if raw_mode else "",
    ))


@app.route("/journal", methods=["GET", "POST"])
def trade_journal():
    error = None
    message = None
    trades = load_trade_journal()
    edit_trade = None

    def _coerce_trade_from_form(existing=None):
        symbol = request.form.get("symbol", "").strip().upper()
        strategy = request.form.get("strategy", "").strip()
        thesis = request.form.get("thesis", "").strip()
        status = request.form.get("status", "open").strip().lower()
        entry_date = request.form.get("entry_date", "").strip()
        expiry_date = request.form.get("expiry_date", "").strip()
        exit_date = request.form.get("exit_date", "").strip()
        max_loss = _safe_float(request.form.get("max_loss"))
        target = _safe_float(request.form.get("target"))
        realized_pnl = _safe_float(request.form.get("realized_pnl"))
        notes = request.form.get("notes", "").strip()
        outcome = request.form.get("outcome", "").strip()

        checks = {
            "setup_match": request.form.get("check_setup_match") == "1",
            "risk_within_cap": request.form.get("check_risk_within_cap") == "1",
            "min_pop_met": request.form.get("check_min_pop_met") == "1",
            "size_valid": request.form.get("check_size_valid") == "1",
            "no_rule_violation": request.form.get("check_no_rule_violation") == "1",
        }

        trade = {
            "id": (existing or {}).get("id") or dt.datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "created_at": (existing or {}).get("created_at") or dt.datetime.now().isoformat(timespec="seconds"),
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "strategy": strategy,
            "thesis": thesis,
            "status": status,
            "entry_date": entry_date,
            "expiry_date": expiry_date,
            "exit_date": exit_date,
            "max_loss": max_loss,
            "target": target,
            "realized_pnl": realized_pnl,
            "outcome": outcome,
            "notes": notes,
            "checks": checks,
        }
        gate = _trade_gate_result(trade)
        trade["gate_passed"] = gate["passed"]
        trade["gate_failed_reasons"] = gate["failed_reasons"]

        if not symbol:
            raise RuntimeError("Symbol is required.")
        if max_loss is None:
            raise RuntimeError("Max loss is required.")
        if status == "open" and not gate["passed"]:
            raise RuntimeError("Trade blocked by gate: " + "; ".join(gate["failed_reasons"]))
        return trade

    if request.method == "POST":
        action = request.form.get("action", "add")
        try:
            if action == "delete":
                trade_id = request.form.get("trade_id", "").strip()
                trades = [t for t in trades if str(t.get("id")) != trade_id]
                save_trade_journal(trades)
                message = "Trade deleted."
            elif action == "update":
                trade_id = request.form.get("trade_id", "").strip()
                target_index = next((i for i, t in enumerate(trades) if str(t.get("id")) == trade_id), None)
                if target_index is None:
                    raise RuntimeError("Trade not found for update.")
                updated = _coerce_trade_from_form(existing=trades[target_index])
                trades[target_index] = updated
                save_trade_journal(trades)
                message = "Trade updated."
            elif action == "add":
                trade = _coerce_trade_from_form(existing=None)
                trades.insert(0, trade)
                save_trade_journal(trades)
                message = "Trade logged."
            else:
                raise RuntimeError(f"Unsupported action: {action}")
        except Exception as exc:
            error = str(exc)

    edit_id = request.args.get("edit", "").strip()
    if edit_id:
        edit_trade = next((t for t in trades if str(t.get("id")) == edit_id), None)
        if edit_trade is None and not error:
            error = "Trade selected for edit was not found."

    trades = sorted(
        trades,
        key=lambda t: t.get("created_at") or "",
        reverse=True,
    )
    return render_template("trade_journal.html", trades=trades, error=error, message=message, edit_trade=edit_trade)


@app.route("/journal/stats")
def trade_stats():
    trades = load_trade_journal()
    stats = compute_trade_stats(trades)
    return render_template("trade_stats.html", stats=stats)


@app.route("/risk")
def risk_dashboard():
    error = None
    summary = {}
    balances = {"cash": 0.0, "liquidation": 0.0, "equity": 0.0}
    try:
        positions = fetch_positions()
        balances = fetch_account_balance_totals()
        summary = build_risk_summary(positions, cash_total=balances.get("cash", 0.0))
    except Exception as exc:
        error = str(exc)

    return render_template("risk.html", summary=summary, balances=balances, error=error)


@app.route("/options-open")
def options_open():
    error = None
    step = float(request.args.get("step", 1))
    span = int(request.args.get("span", 5))
    pricing_mode = (request.args.get("mode", "model") or "model").strip().lower()
    if pricing_mode not in {"model", "expiry"}:
        pricing_mode = "model"
    levels = []
    rows = []
    try:
        positions = fetch_positions()
        option_positions = [p for p in positions if p.get("assetType") == "OPTION"]
        option_symbols = sorted({p.get("symbol") for p in option_positions if p.get("symbol")})
        underlying_symbols = sorted({
            (
                (p.get("instrument", {}) or {}).get("underlyingSymbol")
                or parse_option_symbol(p.get("symbol")).get("underlying")
            )
            for p in option_positions
            if (
                (p.get("instrument", {}) or {}).get("underlyingSymbol")
                or parse_option_symbol(p.get("symbol")).get("underlying")
            )
        })
        quotes = fetch_quotes(option_symbols)
        underlying_quotes = fetch_quotes(underlying_symbols)
        levels, rows = build_put_spread_open_rows(
            option_positions, quotes, underlying_quotes, step, span, pricing_mode=pricing_mode
        )
    except Exception as exc:
        error = str(exc)

    return render_template(
        "options_open.html",
        levels=levels,
        rows=rows,
        step=step,
        span=span,
        pricing_mode=pricing_mode,
        error=error,
    )


@app.route("/spread-sim", methods=["GET", "POST"])
def spread_sim():
    error = (request.args.get("error") or "").strip() or None
    info = (request.args.get("info") or "").strip()
    load_warning = (request.args.get("load_warning") or "").strip() or None
    val = request.values.get
    symbol = (val("symbol") or "SNDK").strip().upper() or "SNDK"
    today = dt.date.today()
    spread_type = (val("spread_type") or "bull_put").strip().lower()
    if spread_type not in {"bull_put", "bear_call"}:
        spread_type = "bull_put"

    raw_stock_price = val("stock_price")
    raw_short_strike = val("short_strike")
    raw_long_strike = val("long_strike")
    raw_credit = val("credit")
    raw_fill_mode = val("fill_mode")
    raw_slippage = val("slippage")
    raw_contracts = val("contracts")
    raw_dte = val("dte")
    raw_iv_short = val("iv_short")
    raw_iv_long = val("iv_long")
    raw_rate = val("rate")
    raw_price_step = val("price_step")
    stock_price = _safe_float(raw_stock_price)
    short_strike = _safe_float(raw_short_strike)
    long_strike = _safe_float(raw_long_strike)
    credit = _safe_float(raw_credit)
    fill_mode = (raw_fill_mode or "mid").strip().lower()
    if fill_mode not in {"mid", "natural"}:
        fill_mode = "mid"
    preset = (val("preset") or "custom").strip().lower()
    if preset not in SPREAD_SIM_PRESETS:
        preset = "custom"
    slippage = _safe_float(raw_slippage)
    contracts = int(_safe_float(raw_contracts) or 1)
    dte = int(_safe_float(raw_dte) or 7)
    iv_short = _safe_float(raw_iv_short)
    iv_long = _safe_float(raw_iv_long)
    rate = _safe_float(raw_rate)
    price_step = _safe_float(raw_price_step)

    short_default = 582.5 if spread_type == "bull_put" else 602.5
    long_default = 580.0 if spread_type == "bull_put" else 605.0
    short_strike = short_strike if short_strike is not None else short_default
    long_strike = long_strike if long_strike is not None else long_default
    stock_price = stock_price if stock_price is not None else round((short_strike + long_strike) / 2.0, 2)
    credit = credit if credit is not None else 0.95
    iv_short = iv_short if iv_short is not None else 0.25
    iv_long = iv_long if iv_long is not None else 0.25
    rate = rate if rate is not None else (_safe_float(os.getenv("RISK_FREE_RATE")) or 0.045)
    price_step = price_step if price_step is not None and price_step > 0 else 0.5
    slippage = slippage if slippage is not None and slippage >= 0 else 0.05

    preset_profile = SPREAD_SIM_PRESETS.get(preset, SPREAD_SIM_PRESETS["custom"])
    preset_defaults = preset_profile.get("defaults", {})
    if raw_fill_mode in {None, ""}:
        fill_mode = preset_defaults.get("fill_mode", fill_mode)
    if raw_slippage in {None, ""}:
        slippage = preset_defaults.get("slippage", slippage)
    if raw_contracts in {None, ""}:
        contracts = preset_defaults.get("contracts", contracts)
    if raw_dte in {None, ""}:
        dte = preset_defaults.get("dte", dte)
    if raw_iv_short in {None, ""}:
        iv_short = preset_defaults.get("iv_short", iv_short)
    if raw_iv_long in {None, ""}:
        iv_long = preset_defaults.get("iv_long", iv_long)
    if raw_price_step in {None, ""}:
        price_step = preset_profile.get("scenario", {}).get("price_step", price_step)

    fill_mode = fill_mode if fill_mode in {"mid", "natural"} else "mid"

    contracts = max(1, contracts)
    dte = max(0, dte)
    t_years = dte / 365.0

    if spread_type == "bull_put":
        width = short_strike - long_strike
        if width <= 0:
            error = "For Bull Put, long strike must be below short strike."
    else:
        width = long_strike - short_strike
        if width <= 0:
            error = "For Bear Call, long strike must be above short strike."

    if spread_type == "bull_put":
        est_short = _bs_put_price(stock_price, short_strike, t_years, rate, iv_short)
        est_long = _bs_put_price(stock_price, long_strike, t_years, rate, iv_long)
    else:
        est_short = _bs_call_price(stock_price, short_strike, t_years, rate, iv_short)
        est_long = _bs_call_price(stock_price, long_strike, t_years, rate, iv_long)
    estimated_credit = max(est_short - est_long, 0.0)
    quoted_credit = credit
    credit_source = "manual"
    if quoted_credit is None:
        quoted_credit = round(estimated_credit, 4)
        credit_source = "estimated"
    entry_credit = quoted_credit if fill_mode == "mid" else max(quoted_credit - slippage, 0.0)

    price_from_default, price_to_default = _spread_sim_price_bounds(stock_price, short_strike, long_strike, preset)
    price_from = _safe_float(request.args.get("price_from"))
    price_to = _safe_float(request.args.get("price_to"))
    price_from = price_from if price_from is not None else price_from_default
    price_to = price_to if price_to is not None else price_to_default
    if price_from < price_to:
        price_from, price_to = price_to, price_from

    if spread_type == "bull_put":
        breakeven = short_strike - entry_credit
    else:
        breakeven = short_strike + entry_credit
    pop_score = None
    if t_years <= 0:
        if spread_type == "bull_put":
            pop_score = 1.0 if stock_price >= breakeven else 0.0
        else:
            pop_score = 1.0 if stock_price <= breakeven else 0.0
    else:
        sigma_pop = max((iv_short + iv_long) / 2.0, 0.0001)
        spot_for_pop = max(stock_price, 0.01)
        threshold = max(breakeven, 0.01)
        denom = sigma_pop * math.sqrt(t_years)
        d2_be = (math.log(spot_for_pop / threshold) + (rate - 0.5 * sigma_pop * sigma_pop) * t_years) / denom
        if spread_type == "bull_put":
            pop_score = _norm_cdf(d2_be)
        else:
            pop_score = _norm_cdf(-d2_be)
        pop_score = min(max(pop_score, 0.0), 1.0)
    max_profit = entry_credit * 100 * contracts
    max_loss = max(width - entry_credit, 0.0) * 100 * contracts
    risk_reward_ratio = (max_profit / max_loss) if max_loss > 0 else None
    return_on_risk_pct = ((max_profit / max_loss) * 100.0) if max_loss > 0 else None
    profit_targets = []
    for pct in (25, 50, 75):
        target_profit = max_profit * (pct / 100.0)
        profit_targets.append({
            "label": f"{pct}% Target",
            "pct": pct,
            "target_profit": round(target_profit, 2),
            "remaining_value_per_spread": round(max(entry_credit * (1.0 - (pct / 100.0)), 0.0), 2),
        })
    defense_lines = []
    for multiple in (2.0, 2.5, 3.0):
        stop_debit_per_spread = min(entry_credit * multiple, width)
        stop_loss_amount = max((stop_debit_per_spread - entry_credit) * 100.0 * contracts, 0.0)
        defense_lines.append({
            "label": f"{multiple:.1f}x Credit Stop",
            "kind": "debit_stop",
            "multiple": multiple,
            "stop_debit_per_spread": round(stop_debit_per_spread, 2),
            "stop_loss_amount": round(stop_loss_amount, 2),
            "pnl_threshold": round(-stop_loss_amount, 2),
        })
    defense_lines.append({
        "label": "Short Strike Breach",
        "kind": "short_strike_breach",
        "trigger_price": round(short_strike, 2),
    })

    if spread_type == "bull_put":
        current_short_model = _bs_put_price(stock_price, short_strike, t_years, rate, iv_short)
        current_long_model = _bs_put_price(stock_price, long_strike, t_years, rate, iv_long)
        current_short_exp = _put_intrinsic(stock_price, short_strike)
        current_long_exp = _put_intrinsic(stock_price, long_strike)
        current_zone = (
            "Max Profit Zone" if stock_price >= short_strike
            else "Max Loss Zone" if stock_price <= long_strike
            else "Profit Zone" if stock_price >= breakeven
            else "Loss Zone"
        )
        short_strike_breached_now = stock_price < short_strike
    else:
        current_short_model = _bs_call_price(stock_price, short_strike, t_years, rate, iv_short)
        current_long_model = _bs_call_price(stock_price, long_strike, t_years, rate, iv_long)
        current_short_exp = _call_intrinsic(stock_price, short_strike)
        current_long_exp = _call_intrinsic(stock_price, long_strike)
        current_zone = (
            "Max Profit Zone" if stock_price <= short_strike
            else "Max Loss Zone" if stock_price >= long_strike
            else "Profit Zone" if stock_price <= breakeven
            else "Loss Zone"
        )
        short_strike_breached_now = stock_price > short_strike
    current_spread_value_model = current_short_model - current_long_model
    current_spread_value_expiry = current_short_exp - current_long_exp
    current_pnl_model = (entry_credit - current_spread_value_model) * 100 * contracts
    current_pnl_expiry = (entry_credit - current_spread_value_expiry) * 100 * contracts
    for target in profit_targets:
        target["hit_now"] = current_pnl_model >= target["target_profit"] - 1e-9
    for line in defense_lines:
        if line["kind"] == "debit_stop":
            line["hit_now"] = current_spread_value_model >= line["stop_debit_per_spread"] - 1e-9
        else:
            line["hit_now"] = short_strike_breached_now

    strike_buffer_pct = None
    if stock_price and stock_price > 0:
        strike_buffer_pct = abs(stock_price - short_strike) / stock_price * 100.0
    risk_warning = None
    if strike_buffer_pct is not None and strike_buffer_pct <= 2.0:
        risk_warning = (
            f"Short strike is only {strike_buffer_pct:.2f}% from spot; "
            "assignment/defense risk is elevated."
        )

    if request.method == "POST":
        action = (request.form.get("action") or "").strip().lower()
        if action in {"export_watchlist", "export_ticket", "download_simulation", "load_simulation"}:
            params = {
                "symbol": symbol,
                "spread_type": spread_type,
                "stock_price": stock_price,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "credit": quoted_credit,
                "fill_mode": fill_mode,
                "preset": preset,
                "slippage": slippage,
                "contracts": contracts,
                "dte": dte,
                "iv_short": iv_short,
                "iv_long": iv_long,
                "rate": rate,
                "price_step": price_step,
                "price_from": _safe_float(val("price_from")) if _safe_float(val("price_from")) is not None else price_from_default,
                "price_to": _safe_float(val("price_to")) if _safe_float(val("price_to")) is not None else price_to_default,
            }
            try:
                if action == "download_simulation":
                    payload = {
                        "type": SPREAD_SIM_SAVE_TYPE,
                        "version": SPREAD_SIM_SAVE_VERSION,
                        "saved_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        "inputs": params,
                        "results_snapshot": {
                            "entry_credit": round(entry_credit, 4),
                            "credit_source": credit_source,
                            "estimated_credit": round(estimated_credit, 4),
                            "width": round(width, 2),
                            "breakeven": round(breakeven, 2),
                            "max_profit": round(max_profit, 2),
                            "max_loss": round(max_loss, 2),
                            "pop_score_pct": (round(pop_score * 100.0, 2) if pop_score is not None else None),
                            "strike_buffer_pct": (round(strike_buffer_pct, 2) if strike_buffer_pct is not None else None),
                            "risk_warning": risk_warning,
                        },
                    }
                    safe_symbol = re.sub(r"[^A-Za-z0-9._-]", "_", symbol or "spread")
                    filename = f"spread-sim-{safe_symbol}-{today.isoformat()}.json"
                    body = json.dumps(payload, indent=2)
                    return app.response_class(
                        body,
                        mimetype="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                    )
                elif action == "load_simulation":
                    upload = request.files.get("simulation_file")
                    if not upload or not upload.filename:
                        raise ValueError("Choose a simulation JSON file to load.")
                    raw = upload.read()
                    if not raw:
                        raise ValueError("Uploaded simulation file is empty.")
                    try:
                        parsed = json.loads(raw.decode("utf-8"))
                    except Exception as exc:
                        raise ValueError("Simulation file is not valid JSON.") from exc
                    if not isinstance(parsed, dict):
                        raise ValueError("Simulation file must be a JSON object.")
                    file_type = (parsed.get("type") or "").strip()
                    if file_type and file_type != SPREAD_SIM_SAVE_TYPE:
                        raise ValueError(
                            f"Unsupported simulation type '{file_type}'. Expected '{SPREAD_SIM_SAVE_TYPE}'."
                        )
                    raw_version = parsed.get("version")
                    if raw_version is None:
                        raise ValueError("Simulation file is missing required version metadata.")
                    try:
                        file_version = int(raw_version)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Simulation file version must be an integer.") from exc
                    if file_version > SPREAD_SIM_SAVE_VERSION:
                        raise ValueError(
                            f"Simulation file uses newer save format v{file_version}. "
                            f"This app supports up to v{SPREAD_SIM_SAVE_VERSION}."
                        )
                    loaded_inputs = parsed.get("inputs") if isinstance(parsed, dict) else None
                    if not isinstance(loaded_inputs, dict):
                        raise ValueError("Simulation file is missing an inputs object.")
                    allowed = {
                        "symbol",
                        "spread_type",
                        "stock_price",
                        "short_strike",
                        "long_strike",
                        "credit",
                        "fill_mode",
                        "preset",
                        "slippage",
                        "contracts",
                        "dte",
                        "iv_short",
                        "iv_long",
                        "rate",
                        "price_step",
                        "price_from",
                        "price_to",
                    }
                    restored = {}
                    for key in allowed:
                        value = loaded_inputs.get(key)
                        if value is None or value == "":
                            continue
                        restored[key] = value
                    required_keys = {"symbol", "spread_type", "short_strike", "long_strike"}
                    missing = [key for key in sorted(required_keys) if key not in restored]
                    if missing:
                        raise ValueError(f"Simulation file is missing required inputs: {', '.join(missing)}.")
                    restored_spread_type = str(restored.get("spread_type", "")).strip().lower()
                    if restored_spread_type not in {"bull_put", "bear_call"}:
                        raise ValueError("Simulation file has invalid spread_type. Must be bull_put or bear_call.")
                    if "fill_mode" in restored:
                        restored_fill_mode = str(restored.get("fill_mode", "")).strip().lower()
                        if restored_fill_mode not in {"mid", "natural"}:
                            raise ValueError("Simulation file has invalid fill_mode. Must be mid or natural.")
                    if "preset" in restored:
                        restored_preset = str(restored.get("preset", "")).strip().lower()
                        if restored_preset not in SPREAD_SIM_PRESETS:
                            raise ValueError(
                                "Simulation file has invalid preset. Must be custom, conservative, income, or high_pop."
                            )
                    numeric_fields = {
                        "stock_price": {"min": 0.01},
                        "short_strike": {"min": 0.01},
                        "long_strike": {"min": 0.01},
                        "credit": {"min": 0.0},
                        "slippage": {"min": 0.0},
                        "contracts": {"min": 1, "int": True},
                        "dte": {"min": 0, "int": True},
                        "iv_short": {"min": 0.01},
                        "iv_long": {"min": 0.01},
                        "rate": {"min": 0.0},
                        "price_step": {"min": 0.01},
                    }
                    for field, rules in numeric_fields.items():
                        if field not in restored:
                            continue
                        parsed_num = _safe_float(restored[field])
                        if parsed_num is None:
                            raise ValueError(f"Simulation field '{field}' must be numeric.")
                        if rules.get("int"):
                            if abs(parsed_num - round(parsed_num)) > 1e-9:
                                raise ValueError(f"Simulation field '{field}' must be an integer.")
                            parsed_num = int(round(parsed_num))
                        if parsed_num < rules["min"]:
                            raise ValueError(f"Simulation field '{field}' must be >= {rules['min']}.")
                        restored[field] = parsed_num
                    if not restored:
                        raise ValueError("Simulation file has no restorable inputs.")
                    redirect_params = dict(restored)
                    redirect_params["info"] = "Loaded simulation file."
                    if file_version < SPREAD_SIM_SAVE_VERSION:
                        redirect_params["load_warning"] = (
                            f"Loaded older save format v{file_version} in compatibility mode "
                            f"(current is v{SPREAD_SIM_SAVE_VERSION})."
                        )
                    return redirect(url_for("spread_sim", **redirect_params))
                elif action == "export_watchlist":
                    watch = set(load_watchlist())
                    watch_notes = load_watchlist_notes()
                    before = len(watch)
                    watch.add(symbol)
                    save_watchlist(sorted(watch))
                    watch_notes[symbol] = (
                        f"Preset={preset}; Type={spread_type}; Fill={fill_mode}; "
                        f"Credit={entry_credit:.2f}; DTE={dte}; "
                        f"Strikes={short_strike:.2f}/{long_strike:.2f}"
                    )
                    save_watchlist_notes(watch_notes)
                    added = len(watch) - before
                    info = f"Watchlist updated: {'added' if added else 'already had'} {symbol}."
                else:
                    expiry = (today + dt.timedelta(days=dte)).isoformat()
                    trade_text = (
                        f"SELL {short_strike:.2f} PUT / BUY {long_strike:.2f} PUT"
                        if spread_type == "bull_put"
                        else f"SELL {short_strike:.2f} CALL / BUY {long_strike:.2f} CALL"
                    )
                    trade_row = {
                        "Trade": trade_text,
                        "Credit (Realistic)": round(entry_credit * 100.0, 2),
                        "Max Loss": round(max_loss, 2),
                        "Source": "Spread Simulator",
                    }
                    ticket = build_ticket(symbol, spread_type, expiry, trade_row)
                    ticket["source"] = "spread_simulator"
                    ticket["assumptions"] = {
                        "preset": preset,
                        "fill_mode": fill_mode,
                        "slippage": round(slippage, 2),
                        "dte": dte,
                        "iv_short": round(iv_short, 4),
                        "iv_long": round(iv_long, 4),
                        "rate": round(rate, 4),
                        "entry_credit_per_spread": round(entry_credit, 4),
                        "quoted_credit_per_spread": round(quoted_credit, 4),
                        "breakeven": round(breakeven, 2),
                        "pop_score": round(pop_score * 100.0, 2) if pop_score is not None else None,
                        "max_profit": round(max_profit, 2),
                        "max_loss": round(max_loss, 2),
                    }
                    ticket["management_plan"] = build_management_plan(
                        ticket,
                        trade_row=trade_row,
                        assumptions=ticket["assumptions"],
                    )
                    tickets_data = load_tickets()
                    tickets_data.insert(0, ticket)
                    save_tickets(tickets_data)
                    info = f"Ticket created from spread sim: {ticket['id']}."
            except Exception as exc:
                error = str(exc)
            redirect_params = dict(params)
            if info:
                redirect_params["info"] = info
            if error:
                redirect_params["error"] = error
            return redirect(url_for("spread_sim", **redirect_params))

    rows = []
    price = price_from
    for _ in range(1000):
        if price < price_to - 1e-9:
            break

        if spread_type == "bull_put":
            short_exp = _put_intrinsic(price, short_strike)
            long_exp = _put_intrinsic(price, long_strike)
        else:
            short_exp = _call_intrinsic(price, short_strike)
            long_exp = _call_intrinsic(price, long_strike)
        spread_exp = short_exp - long_exp
        pnl_exp = (entry_credit - spread_exp) * 100 * contracts

        if spread_type == "bull_put":
            short_model = _bs_put_price(price, short_strike, t_years, rate, iv_short)
            long_model = _bs_put_price(price, long_strike, t_years, rate, iv_long)
        else:
            short_model = _bs_call_price(price, short_strike, t_years, rate, iv_short)
            long_model = _bs_call_price(price, long_strike, t_years, rate, iv_long)
        spread_model = short_model - long_model
        pnl_model = (entry_credit - spread_model) * 100 * contracts

        if spread_type == "bull_put":
            if price >= short_strike:
                zone = "Max Profit Zone"
            elif price <= long_strike:
                zone = "Max Loss Zone"
            elif price >= breakeven:
                zone = "Profit Zone"
            else:
                zone = "Loss Zone"
        else:
            if price <= short_strike:
                zone = "Max Profit Zone"
            elif price >= long_strike:
                zone = "Max Loss Zone"
            elif price <= breakeven:
                zone = "Profit Zone"
            else:
                zone = "Loss Zone"

        rows.append({
            "stock_price": round(price, 2),
            "markers": [],
            "zone": zone,
            "spread_value_model": round(spread_model, 4),
            "pnl_model": round(pnl_model, 2),
            "spread_value_expiry": round(spread_exp, 4),
            "pnl_expiry": round(pnl_exp, 2),
        })
        price = round(price - price_step, 8)

    if rows:
        spot_idx = min(range(len(rows)), key=lambda idx: abs(rows[idx]["stock_price"] - stock_price))
        rows[spot_idx]["markers"].append("Spot")
        for target in profit_targets:
            target_idx = min(range(len(rows)), key=lambda idx: abs(rows[idx]["pnl_model"] - target["target_profit"]))
            rows[target_idx]["markers"].append(target["label"])
        breach_idx = min(range(len(rows)), key=lambda idx: abs(rows[idx]["stock_price"] - short_strike))
        rows[breach_idx]["markers"].append("Short Strike Breach")
        for line in defense_lines:
            if line["kind"] != "debit_stop":
                continue
            stop_idx = min(range(len(rows)), key=lambda idx: abs(rows[idx]["spread_value_model"] - line["stop_debit_per_spread"]))
            rows[stop_idx]["markers"].append(line["label"])
        for row in rows:
            seen = []
            for marker in row["markers"]:
                if marker not in seen:
                    seen.append(marker)
            row["markers"] = seen
            row["action_summary"] = _spread_sim_action_summary(spread_type, row["zone"], row["markers"])

    return render_template(
        "spread_sim.html",
        error=error,
        info=info,
        load_warning=load_warning,
        sim_schema_version=SPREAD_SIM_SAVE_VERSION,
        spread_type=spread_type,
        symbol=symbol,
        today=today.isoformat(),
        preset=preset,
        preset_meta=SPREAD_SIM_PRESETS,
        preset_description=preset_profile.get("description"),
        stock_price=stock_price,
        short_strike=short_strike,
        long_strike=long_strike,
        credit=quoted_credit,
        entry_credit=round(entry_credit, 4),
        credit_source=credit_source,
        fill_mode=fill_mode,
        slippage=slippage,
        estimated_credit=round(estimated_credit, 4),
        contracts=contracts,
        dte=dte,
        iv_short=iv_short,
        iv_long=iv_long,
        rate=rate,
        price_from=price_from,
        price_to=price_to,
        price_step=price_step,
        width=round(width, 2),
        breakeven=round(breakeven, 2),
        max_profit=round(max_profit, 2),
        max_loss=round(max_loss, 2),
        risk_reward_ratio=(round(risk_reward_ratio, 2) if risk_reward_ratio is not None else None),
        return_on_risk_pct=(round(return_on_risk_pct, 2) if return_on_risk_pct is not None else None),
        current_zone=current_zone,
        current_spread_value_model=round(current_spread_value_model, 4),
        current_spread_value_expiry=round(current_spread_value_expiry, 4),
        current_pnl_model=round(current_pnl_model, 2),
        current_pnl_expiry=round(current_pnl_expiry, 2),
        profit_targets=profit_targets,
        defense_lines=defense_lines,
        short_strike_breached_now=short_strike_breached_now,
        pop_score=(round(pop_score * 100.0, 2) if pop_score is not None else None),
        strike_buffer_pct=(round(strike_buffer_pct, 2) if strike_buffer_pct is not None else None),
        risk_warning=risk_warning,
        rows=rows,
    )


@app.route("/summary")
def summary():
    period = request.args.get("period", "month")
    status_filter = request.args.get("status", "all")
    sort_by = request.args.get("sort", "profit")
    selected_year_raw = (request.args.get("year") or "").strip()
    selected_month_raw = (request.args.get("month") or "").strip()
    tz = request.args.get("tz")
    selected_year = int(selected_year_raw) if selected_year_raw.isdigit() else None
    selected_month = int(selected_month_raw) if selected_month_raw.isdigit() else None
    error = None
    rows = []
    year_options = []
    month_options = [
        {"value": 1, "label": "January"},
        {"value": 2, "label": "February"},
        {"value": 3, "label": "March"},
        {"value": 4, "label": "April"},
        {"value": 5, "label": "May"},
        {"value": 6, "label": "June"},
        {"value": 7, "label": "July"},
        {"value": 8, "label": "August"},
        {"value": 9, "label": "September"},
        {"value": 10, "label": "October"},
        {"value": 11, "label": "November"},
        {"value": 12, "label": "December"},
    ]
    stock_rows = []
    cash_rows = []
    option_rows = []
    other_rows = []
    stock_totals = _section_totals([])
    cash_totals = _section_totals([])
    option_totals = _section_totals([])
    other_totals = _section_totals([])
    combined_totals = _section_totals([])
    summary_last_updated = None
    summary_last_updated_ago = None
    try:
        settings = load_settings()
        if tz:
            settings["timezone"] = tz
            save_settings(settings)

        tz_name = settings.get("timezone", "America/New_York")
        rows = build_yearly_summary(period, status_filter, tz_name=tz_name)
        try:
            now_local = dt.datetime.now(ZoneInfo(tz_name))
        except Exception:
            now_local = dt.datetime.now()
        current_year = now_local.year

        period_parts = [(row, *period_bucket_to_year_month(row.get("period"), period)) for row in rows]
        discovered_years = {year for _, year, _ in period_parts if year}
        discovered_years.add(current_year)
        year_options = sorted(discovered_years, reverse=True)

        if selected_year:
            period_parts = [part for part in period_parts if part[1] == selected_year]
        if selected_month and period != "year":
            period_parts = [part for part in period_parts if part[2] == selected_month]
        rows = [row for row, _, _ in period_parts]

        cache_ts = _TRANSACTIONS_CACHE.get("ts", 0.0)
        if cache_ts:
            tz_name = settings.get("timezone", "America/New_York")
            try:
                tzinfo = ZoneInfo(tz_name)
            except Exception:
                tzinfo = dt.timezone.utc
            cache_dt = dt.datetime.fromtimestamp(cache_ts, tz=dt.timezone.utc).astimezone(tzinfo)
            summary_last_updated = cache_dt.strftime("%Y-%m-%d %I:%M:%S %p %Z")
            age_seconds = max(0, int(time.time() - cache_ts))
            if age_seconds < 60:
                summary_last_updated_ago = "just now"
            elif age_seconds < 3600:
                summary_last_updated_ago = f"{age_seconds // 60}m ago"
            else:
                summary_last_updated_ago = f"{age_seconds // 3600}h ago"

        if sort_by == "loss":
            rows = sorted(rows, key=lambda r: r["pnl"])
        elif sort_by == "symbol":
            rows = sorted(rows, key=lambda r: (r["symbol"], r["period"]))
        else:
            rows = sorted(rows, key=lambda r: r["pnl"], reverse=True)

        stock_rows = [r for r in rows if r.get("assetType") == "STOCK"]
        cash_rows = [r for r in rows if r.get("assetType") == "CASH"]
        option_rows = [r for r in rows if r.get("assetType") == "OPTION"]
        other_rows = [r for r in rows if r.get("assetType") not in {"STOCK", "OPTION", "CASH"}]

        stock_totals = _section_totals(stock_rows)
        cash_totals = _section_totals(cash_rows)
        option_totals = _section_totals(option_rows)
        other_totals = _section_totals(other_rows)
        combined_totals = _section_totals([r for r in rows if r.get("assetType") != "CASH"])
    except Exception as exc:
        error = str(exc)

    return render_template(
        "summary.html",
        rows=rows,
        stock_rows=stock_rows,
        cash_rows=cash_rows,
        option_rows=option_rows,
        other_rows=other_rows,
        stock_totals=stock_totals,
        cash_totals=cash_totals,
        option_totals=option_totals,
        other_totals=other_totals,
        combined_totals=combined_totals,
        error=error,
        period=period,
        status=status_filter,
        sort_by=sort_by,
        selected_year=selected_year,
        selected_month=selected_month,
        year_options=year_options,
        month_options=month_options,
        summary_last_updated=summary_last_updated,
        summary_last_updated_ago=summary_last_updated_ago,
    )


def _csv_response(filename, rows, headers):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    response = app.response_class(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
    return response


@app.route("/export/alerts")
def export_alerts():
    rows = load_alerts()
    headers = ["timestamp", "symbol", "assetType", "qty", "pnl", "pnl_pct", "dte", "action", "reasons", "note"]
    return _csv_response("alerts.csv", rows, headers)


@app.route("/export/tickets")
def export_tickets():
    rows = load_tickets()
    headers = ["id", "created_at", "symbol", "strategy", "expiry", "status"]
    flat = []
    for row in rows:
        flat.append({k: row.get(k) for k in headers})
    return _csv_response("tickets.csv", flat, headers)


@app.route("/export/summary")
def export_summary():
    period = request.args.get("period", "month")
    status_filter = request.args.get("status", "all")
    tz_name = load_settings().get("timezone", "America/New_York")
    rows = build_yearly_summary(period, status_filter, tz_name=tz_name)
    headers = ["period", "symbol", "status", "assetType", "qty", "profit", "loss", "pnl"]
    return _csv_response("summary.csv", rows, headers)


@app.route("/tools")
def tools():
    state = load_monitor_state()
    errors = load_error_log()
    status = None
    refresh_result = None
    token_b64_for_render = None
    token_b64_error = None
    email_test_result = None
    settings = load_settings()
    tz = request.args.get("tz")
    if tz:
        settings["timezone"] = tz
        save_settings(settings)
    ensure_monitor_thread()
    token_status = get_token_status()
    if request.args.get("clear_errors") == "1":
        save_monitor_state(state)
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        errors = []
    if request.args.get("refresh_token") == "1":
        try:
            client = get_client()
            response = schwab_request(lambda: client.get_accounts(), "refresh_token")
            refresh_result = {"ok": True, "status_code": response.status_code}
        except Exception as exc:
            refresh_result = {"ok": False, "error": str(exc)}
    if request.args.get("show_token_b64") == "1":
        try:
            token_path = os.getenv("TOKEN_PATH", "token.json")
            raw_file = None
            if os.path.exists(token_path):
                try:
                    with open(token_path, "r", encoding="utf-8") as f:
                        raw_file = f.read()
                except Exception:
                    raw_file = None

            token_json = os.getenv("TOKEN_JSON")
            token_b64 = os.getenv("TOKEN_JSON_B64")
            decoded_b64 = None
            if token_b64:
                try:
                    decoded_b64 = base64.b64decode(token_b64).decode("utf-8")
                except Exception as exc:
                    raise RuntimeError(f"TOKEN_JSON_B64 is not valid base64/utf-8: {exc}") from exc

            payload = _best_token_candidate(
                [
                    _parse_token_payload(raw_file),
                    _parse_token_payload(token_json),
                    _parse_token_payload(decoded_b64),
                ]
            )
            if not payload:
                raise RuntimeError("No usable token found in token file, TOKEN_JSON, or TOKEN_JSON_B64.")
            token_b64_for_render = base64.b64encode(
                payload["raw"].encode("utf-8")
            ).decode("ascii")
        except Exception as exc:
            token_b64_error = str(exc)
    if request.args.get("check_schwab") == "1":
        try:
            client = get_client()
            response = schwab_request(lambda: client.get_account_numbers(), "status_check")
            status = {"ok": True, "status_code": response.status_code}
        except Exception as exc:
            status = {"ok": False, "error": str(exc)}
    if request.args.get("send_test_email") == "1":
        try:
            send_test_email_alert()
            email_test_result = {"ok": True}
        except Exception as exc:
            email_test_result = {"ok": False, "error": str(exc)}

    tzinfo = ZoneInfo(settings.get("timezone", "UTC"))
    now = dt.datetime.now(tzinfo)
    polling_minutes = max(1, int(settings.get("polling_minutes", 60)))
    stale_seconds = (polling_minutes * 60) + 120

    def _coerce_tz(value):
        if not value:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=tzinfo)
        return value.astimezone(tzinfo)

    paused_until = _coerce_tz(_parse_iso_dt(state.get("paused_until")))
    last_run = _coerce_tz(_parse_iso_dt(state.get("last_monitor_run")))
    thread_alive = bool(_MONITOR_THREAD and _MONITOR_THREAD.is_alive())
    monitor_status = "stopped"
    if paused_until and paused_until > now:
        monitor_status = "paused"
    elif thread_alive:
        if last_run and (now - last_run).total_seconds() <= stale_seconds:
            monitor_status = "running"
        elif last_run:
            monitor_status = "stale"
        else:
            monitor_status = "starting"

    return render_template(
        "tools.html",
        state=state,
        errors=errors,
        status=status,
        refresh_result=refresh_result,
        token_b64_for_render=token_b64_for_render,
        token_b64_error=token_b64_error,
        email_test_result=email_test_result,
        token_status=token_status,
        monitor_status=monitor_status,
        monitor_thread_alive=thread_alive,
        monitor_enabled=monitor_enabled(),
        polling_minutes=polling_minutes,
        timezone=settings.get("timezone", "UTC"),
    )


def _parse_iso_date(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except Exception:
        return None


@app.route("/debug/txn-reconcile")
def debug_txn_reconcile():
    settings = load_settings()
    tz_name = settings.get("timezone", "America/New_York")
    try:
        local_today = dt.datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        local_today = dt.date.today()

    first_this_month = local_today.replace(day=1)
    default_end = first_this_month - dt.timedelta(days=1)
    default_start = default_end.replace(day=1)

    symbol_filter = (request.args.get("symbol") or "SNDK").strip().upper()
    asset_filter = (request.args.get("asset") or "OPTION").strip().upper()
    start_raw = (request.args.get("start") or default_start.isoformat()).strip()
    end_raw = (request.args.get("end") or default_end.isoformat()).strip()
    include_only_closed = request.args.get("closed", "1") != "0"

    start_date = _parse_iso_date(start_raw) or default_start
    end_date = _parse_iso_date(end_raw) or default_end
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    error = None
    rows = []
    totals = {
        "count": 0,
        "qty": 0.0,
        "pnl": 0.0,
        "net": 0.0,
        "item_cashflow": 0.0,
        "transfer_cashflow": 0.0,
        "desc_cashflow": 0.0,
    }
    by_symbol = {}
    try:
        for txn in fetch_transactions_one_year():
            txn_date = _txn_date(txn, tz_name=tz_name)
            if txn_date < start_date or txn_date > end_date:
                continue

            symbol = (_txn_symbol(txn) or "").upper()
            if symbol_filter and symbol_filter not in symbol:
                continue

            asset_type = _txn_asset_type(txn)
            if asset_filter != "ALL" and asset_type != asset_filter:
                continue

            is_closed_candidate = _include_in_closed_summary(txn)
            if include_only_closed and not is_closed_candidate:
                continue

            item = txn.get("transactionItem", {}) or {}
            instruction = _txn_instruction(txn)
            qty = _txn_qty(txn)
            pnl = _txn_pnl(txn)
            net = _safe_float(txn.get("netAmount"))
            item_cashflow = _compute_cashflow_from_item(txn)
            transfer_cashflow = _compute_cashflow_from_transfers(txn)
            desc_cashflow = _compute_cashflow_from_description(txn)
            fee_total = abs(_sum_numeric_values(txn.get("fees") or {})) + abs(_sum_numeric_values(item.get("fees") or {}))

            row = {
                "date": txn_date.isoformat(),
                "accountNumber": txn.get("accountNumber"),
                "symbol": symbol,
                "assetType": asset_type,
                "txnType": str(txn.get("transactionType") or "").strip().upper(),
                "instruction": instruction,
                "statusIncluded": is_closed_candidate,
                "qty": qty,
                "pnl": pnl,
                "net": net,
                "item_cashflow": item_cashflow,
                "transfer_cashflow": transfer_cashflow,
                "desc_cashflow": desc_cashflow,
                "fees": fee_total,
                "price": _safe_float(item.get("price")),
                "item_amount": _safe_float(item.get("amount")),
                "item_quantity": _safe_float(item.get("quantity")) or _safe_float(item.get("quantityNumber")),
                "item_cost": _safe_float(item.get("cost")),
                "txn_description": str(txn.get("description") or ""),
                "item_description": str(item.get("description") or ""),
                "transactionId": txn.get("transactionId"),
            }
            rows.append(row)

            totals["count"] += 1
            totals["qty"] += qty or 0.0
            totals["pnl"] += pnl or 0.0
            totals["net"] += net or 0.0
            totals["item_cashflow"] += item_cashflow or 0.0
            totals["transfer_cashflow"] += transfer_cashflow or 0.0
            totals["desc_cashflow"] += desc_cashflow or 0.0

            by_symbol.setdefault(symbol, 0.0)
            by_symbol[symbol] += pnl or 0.0

        rows.sort(key=lambda r: (r["date"], r["symbol"], str(r.get("instruction") or ""), str(r.get("transactionId") or "")))
    except Exception as exc:
        error = str(exc)

    by_symbol_rows = [{"symbol": sym, "pnl": pnl} for sym, pnl in sorted(by_symbol.items(), key=lambda kv: kv[0])]
    return render_template(
        "txn_reconcile.html",
        error=error,
        rows=rows,
        totals=totals,
        by_symbol_rows=by_symbol_rows,
        symbol_filter=symbol_filter,
        asset_filter=asset_filter,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        include_only_closed=include_only_closed,
    )


@app.route("/manual")
def manual():
    return render_template("manual.html", manual_text=load_manual_text())


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(error):
    log_error(error, context="http_500")
    return render_template("error.html", code=500, message="Something went wrong."), 500


@app.route("/nova", methods=["GET", "POST"])
def nova_chat():
    settings = load_settings()
    global NOVA_ERROR
    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "clear":
            NOVA_CHAT.clear()
            NOVA_ERROR = None
            return redirect(url_for("nova_chat"))
        message = request.form.get("message", "").strip()
        if action == "movers":
            lookback = settings.get("movers_lookback_days", 5)
            count = settings.get("movers_count", 10)
            message = (
                f"Scan for stock movers over the last {lookback} trading days and "
                f"recommend {count} symbols to explore. Include both positive and negative movers."
            )
        if message:
            NOVA_CHAT.append({"role": "user", "content": message})
            try:
                positions = fetch_positions()
                positions_context = positions[:20]
                with _ALERTS_LOCK:
                    current_alerts = ALERTS or load_alerts()
                role = _nova_role(settings)
                system_prompt = (
                    f"You are Nova, a {role}. Provide decisive, profit-maximizing recommendations "
                    "within the user's rules. "
                    "Advisory-only guidance; do not place trades or give direct order placement instructions. "
                    "Apply the user's rules without restating them verbatim unless asked. "
                    "Use the user's rules and current positions for context, and ask clarifying "
                    "questions if needed."
                )
                movers_snapshot = None
                if action == "movers" or _wants_movers(message):
                    movers_universe = build_movers_universe(settings, positions=positions)
                    movers_snapshot = scan_stock_movers(
                        movers_universe,
                        lookback_days=settings.get("movers_lookback_days", 5),
                        max_count=settings.get("movers_count", 10),
                        max_seconds=settings.get("movers_max_seconds", 20),
                    )
                    system_prompt += (
                        " Use the movers scan as the source of candidate symbols and "
                        "return a concise list of 5-10 picks to explore."
                    )
                context = (
                    f"Rules: profit_target={settings['profit_target_pct']}%, "
                    f"max_loss={settings['max_loss_pct']}%, dte_exit={settings['dte_exit']}, "
                    f"nova_judgment={settings['nova_judgment']}, "
                    f"polling_minutes={settings['polling_minutes']}. "
                    f"Positions (up to 20): {positions_context}. "
                    f"Alerts: {current_alerts[:10]}"
                )
                if movers_snapshot:
                    context = context + " " + _format_movers_context(movers_snapshot)
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=settings["nova_model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "system", "content": NOVA_RULES},
                        {"role": "system", "content": context},
                    ] + NOVA_CHAT[-10:],
                )
                reply = response.choices[0].message.content
                NOVA_CHAT.append({"role": "assistant", "content": reply})
                NOVA_ERROR = None
            except Exception as exc:
                NOVA_ERROR = str(exc)
        return redirect(url_for("nova_chat"))

    error = NOVA_ERROR
    NOVA_ERROR = None
    return render_template("nova.html", messages=NOVA_CHAT, error=error, settings=settings)


@app.route("/nova/options", methods=["POST"])
def nova_options():
    settings = load_settings()
    global NOVA_ERROR, NOVA_OPTIONS_RESPONSE, NOVA_OPTIONS_ERROR
    action = request.form.get("action", "")
    symbol = request.form.get("symbol", "NVDA").upper().strip()
    strategy = request.form.get("strategy", "bull_put")
    width = float(request.form.get("width", 2.5))
    min_pop = float(request.form.get("min_pop", 80))
    contracts = int(request.form.get("contracts", 1))
    raw_mode = request.form.get("raw", "") == "1"
    pricing_mode = request.form.get("pricing_mode", "mid").strip().lower()
    if pricing_mode not in {"mid", "natural", "custom"}:
        pricing_mode = "mid"
    custom_limit_raw = request.form.get("custom_limit", "").strip()
    custom_limit = None
    if custom_limit_raw:
        try:
            custom_limit = float(custom_limit_raw)
        except ValueError:
            custom_limit = None
    cash_balance = int(request.form.get("cash", 2000))
    max_loss = float(request.form.get("max_loss", get_max_loss_threshold(cash_balance)))
    expiry = request.form.get("expiry", "")
    selected_index = request.form.get("trade_index")

    try:
        results_df, spot_price, expirations = run_scan(
            symbol, expiry, strategy, width, max_loss, min_pop, raw_mode, contracts,
            pricing_mode=pricing_mode, custom_limit=custom_limit
        )
        results_records = []
        if results_df is not None and not results_df.empty:
            results_records = results_df.to_dict(orient="records")

        research = get_research_summary(symbol)
        market_research = get_market_research()

        if action == "suggest_filters":
            client = get_openai_client()
            exp_list = [e["date"] for e in expirations][:12]
            prompt = (
                "Suggest option chain scan filters. Return JSON with keys: "
                "expiry, strategy (bull_put|bear_call|iron_condor), width, min_pop, "
                "contracts, max_loss, raw_mode. Use one of the provided expiries."
            )
            context = (
                f"Symbol: {symbol}, spot: {spot_price}, expiries: {exp_list}, "
                f"current: strategy={strategy}, width={width}, min_pop={min_pop}, "
                f"contracts={contracts}, max_loss={max_loss}, cash_balance={cash_balance}. "
                f"Research: {research}. Market: {market_research}."
            )
            response = client.chat.completions.create(
                model=settings["nova_model"],
                messages=[
                    {"role": "system", "content": f"You are Nova, a {_nova_role(settings)}. Return JSON only."},
                    {"role": "system", "content": NOVA_RULES},
                    {"role": "user", "content": prompt + "\n" + context},
                ],
            )
            reply = response.choices[0].message.content
            NOVA_CHAT.append({"role": "assistant", "content": reply})
            NOVA_OPTIONS_RESPONSE = reply
            NOVA_OPTIONS_ERROR = None
            try:
                data = json.loads(reply)
                expiry = data.get("expiry", expiry)
                strategy = data.get("strategy", strategy)
                width = float(data.get("width", width))
                min_pop = float(data.get("min_pop", min_pop))
                contracts = int(data.get("contracts", contracts))
                max_loss = float(data.get("max_loss", max_loss))
                raw_mode = bool(data.get("raw_mode", raw_mode))
            except Exception:
                pass
            return redirect(url_for(
                "options_chain",
                symbol=symbol,
                expiry=expiry,
                strategy=strategy,
                width=width,
                min_pop=min_pop,
                contracts=contracts,
                cash=cash_balance,
                max_loss=max_loss,
                pricing_mode=pricing_mode,
                custom_limit=custom_limit if custom_limit is not None else "",
                raw="1" if raw_mode else "",
            ))

        if action == "explain_trade":
            idx = int(selected_index) if selected_index is not None else None
            trade = results_records[idx] if idx is not None and idx < len(results_records) else None
            if trade is None:
                raise RuntimeError("Select a trade row to explain.")
            prompt = (
                "Explain this options trade in plain English. Summarize risk/reward, "
                "breakeven, and an exit plan aligned to the user's rules. Advisory-only. "
                "Use the numeric fields exactly as provided for credit, max loss, contracts, and breakeven."
            )
            trade_facts = (
                f"trade_facts: total_credit={trade.get('Total Credit ($)')}, "
                f"credit_per_spread={trade.get('Credit (Realistic)')}, "
                f"max_loss={trade.get('Max Loss ($)')}, "
                f"contracts={trade.get('Contracts')}, "
                f"breakeven={trade.get('Breakeven')}, "
                f"pop={trade.get('POP %')}"
            )
            context = (
                f"Symbol: {symbol}, spot: {spot_price}, cash_balance={cash_balance}, trade: {trade}. "
                f"{trade_facts}. pricing_mode={pricing_mode}, custom_limit={custom_limit}. "
                f"Research: {research}. Market: {market_research}."
            )
        else:
            prompt = (
                "Analyze the current scan results and summarize the best opportunities. "
                "Use the user's rules and call out any trades that violate them."
            )
            context = f"Symbol: {symbol}, spot: {spot_price}, cash_balance={cash_balance}, results: {results_records[:10]}. Research: {research}. Market: {market_research}."

        client = get_openai_client()
        response = client.chat.completions.create(
            model=settings["nova_model"],
            messages=[
                {"role": "system", "content": f"You are Nova, a {_nova_role(settings)}. Advisory-only guidance."},
                {"role": "system", "content": NOVA_RULES},
                {"role": "user", "content": prompt + "\n" + context},
            ],
        )
        reply = response.choices[0].message.content
        NOVA_CHAT.append({"role": "assistant", "content": reply})
        NOVA_OPTIONS_RESPONSE = reply
        NOVA_OPTIONS_ERROR = None
    except Exception as exc:
        NOVA_ERROR = str(exc)
        NOVA_OPTIONS_ERROR = str(exc)

    return redirect(url_for(
        "options_chain",
        symbol=symbol,
        expiry=expiry,
        strategy=strategy,
        width=width,
        min_pop=min_pop,
        contracts=contracts,
        cash=cash_balance,
        max_loss=max_loss,
        pricing_mode=pricing_mode,
        custom_limit=custom_limit if custom_limit is not None else "",
        raw="1" if raw_mode else "",
    ))


@app.route("/nova/options/clear", methods=["POST"])
def nova_options_clear():
    global NOVA_OPTIONS_RESPONSE, NOVA_OPTIONS_ERROR
    NOVA_OPTIONS_RESPONSE = None
    NOVA_OPTIONS_ERROR = None
    symbol = request.form.get("symbol", "NVDA").upper().strip()
    strategy = request.form.get("strategy", "bull_put")
    width = float(request.form.get("width", 2.5))
    min_pop = float(request.form.get("min_pop", 80))
    contracts = int(request.form.get("contracts", 1))
    raw_mode = request.form.get("raw", "") == "1"
    pricing_mode = request.form.get("pricing_mode", "mid").strip().lower()
    if pricing_mode not in {"mid", "natural", "custom"}:
        pricing_mode = "mid"
    custom_limit_raw = request.form.get("custom_limit", "").strip()
    cash_balance = int(request.form.get("cash", 2000))
    max_loss = float(request.form.get("max_loss", get_max_loss_threshold(cash_balance)))
    expiry = request.form.get("expiry", "")
    return redirect(url_for(
        "options_chain",
        symbol=symbol,
        expiry=expiry,
        strategy=strategy,
        width=width,
        min_pop=min_pop,
        contracts=contracts,
        cash=cash_balance,
        max_loss=max_loss,
        pricing_mode=pricing_mode,
        custom_limit=custom_limit_raw,
        raw="1" if raw_mode else "",
    ))

@app.route("/options")
def options_chain():
    symbol = request.args.get("symbol", "NVDA").upper().strip()
    strategy = request.args.get("strategy", "bull_put")
    width = float(request.args.get("width", 2.5))
    min_pop = float(request.args.get("min_pop", 80))
    contracts = int(request.args.get("contracts", 1))
    raw_mode = request.args.get("raw", "") == "1"
    pricing_mode = request.args.get("pricing_mode", "mid").strip().lower()
    if pricing_mode not in {"mid", "natural", "custom"}:
        pricing_mode = "mid"
    custom_limit_raw = request.args.get("custom_limit", "").strip()
    custom_limit = None
    if custom_limit_raw:
        try:
            custom_limit = float(custom_limit_raw)
        except ValueError:
            custom_limit = None
    cash_balance = int(request.args.get("cash", 2000))
    max_loss = float(request.args.get("max_loss", get_max_loss_threshold(cash_balance)))
    expiry = request.args.get("expiry", "")
    auto_refresh = int(request.args.get("auto", 0))

    results_columns = []
    results_rows = []
    errors = []
    spot_price = None
    meta = {}
    research = {}
    market_research = []
    expirations = []

    try:
        results_df, spot_price, expirations = run_scan(
            symbol, expiry, strategy, width, max_loss, min_pop, raw_mode, contracts,
            pricing_mode=pricing_mode, custom_limit=custom_limit
        )
        if results_df is not None:
            results_columns, results_rows = format_results(results_df)
            raw_rows = results_df.to_dict(orient="records")
            for idx, row in enumerate(results_rows):
                raw_row = raw_rows[idx] if idx < len(raw_rows) else {}
                row["_spread_sim_url"] = build_spread_sim_link(symbol, strategy, pricing_mode, raw_row)
            meta = {
                "symbol": symbol,
                "expiry": expiry,
                "spot": spot_price,
                "strategy": strategy,
            }
        research = get_research_summary(symbol)
        market_research = get_market_research()
    except Exception as exc:
        errors.append(str(exc))

    global NOVA_OPTIONS_RESPONSE, NOVA_OPTIONS_ERROR
    nova_response = NOVA_OPTIONS_RESPONSE
    nova_error = NOVA_OPTIONS_ERROR
    NOVA_OPTIONS_RESPONSE = None
    NOVA_OPTIONS_ERROR = None

    return render_template(
        "options.html",
        symbol=symbol,
        strategy=strategy,
        width=width,
        min_pop=min_pop,
        contracts=contracts,
        raw_mode=raw_mode,
        pricing_mode=pricing_mode,
        custom_limit=custom_limit_raw,
        cash_balance=cash_balance,
        max_loss=max_loss,
        expiry=expiry,
        expirations=expirations,
        results_columns=results_columns,
        results_rows=results_rows,
        errors=errors,
        spot_price=spot_price,
        meta=meta,
        auto_refresh=auto_refresh,
        nova_response=nova_response,
        nova_error=nova_error,
        research=research,
        market_research=market_research,
    )


#if __name__ == "__main__":
  #  app.run(
     #   host="127.0.0.1",
      #  port=8001,
      #  ssl_context=("cert.pem", "key.pem"),
       # debug=True
   

