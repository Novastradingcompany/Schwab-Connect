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
SETTINGS_PATH = "settings.json"
NOVA_CHAT = []
_OPENAI_CLIENT = None
NOVA_ERROR = None
ALERTS_PATH = "alerts.json"
ALERTS_STATE_PATH = "alerts_state.json"
ALERTS = []
_ALERTS_LOCK = threading.Lock()
_MONITOR_THREAD = None
NOVA_OPTIONS_RESPONSE = None
NOVA_OPTIONS_ERROR = None
WATCHLIST_PATH = "watchlist.json"
TICKETS_PATH = "tickets.json"
MONITOR_STATE_PATH = "monitor_state.json"
ERROR_LOG_PATH = "error_log.json"
SP500_PATH = os.path.join("nova-options-scanner-main", "nova-options-scanner-main", "sp500_symbols.json")
OPTIONABLE_UNIVERSE_PATH = "optionable_universe.json"
MOVERS_SNAPSHOT_PATH = "movers_snapshot.json"
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
NOVA_MODEL_OPTIONS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4o-mini",
]


def _get_env(key):
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


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


def load_optionable_universe():
    if not os.path.exists(OPTIONABLE_UNIVERSE_PATH):
        return []
    try:
        with open(OPTIONABLE_UNIVERSE_PATH, "r", encoding="utf-8") as f:
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


def run_optionable_weekly_scan(universe, lookback_days=5, top_n=10, max_seconds=25, previous=None):
    rows_cache = {}
    if isinstance(previous, dict):
        prev_lookback = int(previous.get("lookback_days", lookback_days))
        if prev_lookback == int(lookback_days):
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
        if prev_lookback == int(lookback_days):
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
            score = abs(ret_5d or 0.0) + (abs(ret_20d or 0.0) * 0.5) + ((vol_20d or 0.0) * 0.5)
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

    # If the token file already exists, use it
    if os.path.exists(token_path):
        return token_path

    # If a token is stored in environment variables, rebuild it
    token_json = os.getenv("TOKEN_JSON")
    token_b64 = os.getenv("TOKEN_JSON_B64")

    decoded_b64 = None
    if token_b64:
        try:
            decoded_b64 = base64.b64decode(token_b64).decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Failed to decode TOKEN_JSON_B64: {exc}") from exc

    # If both are provided, prefer the newest token payload
    if token_json or decoded_b64:
        candidates = []
        for raw in (token_json, decoded_b64):
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            token = payload.get("token", payload) if isinstance(payload, dict) else {}
            expires_at = token.get("expires_at") or payload.get("expires_at") or 0
            creation_ts = payload.get("creation_timestamp") or 0
            candidates.append((expires_at, creation_ts, raw))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            token_json = candidates[0][2]

    if token_json:
        token_dir = os.path.dirname(token_path)
        if token_dir and not os.path.exists(token_dir):
            os.makedirs(token_dir, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(token_json)
        return token_path

    # No token file and no env token — return None
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
    exp_str = str(exp_str)[:10]
    try:
        return dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
    except ValueError:
        return None


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

            expiration = parse_expiration(instrument.get("expirationDate") or instrument.get("maturityDate"))
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
        reasons.append(f"DTE threshold ({dte} days)")

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
    return ticket


def build_risk_summary(positions):
    total_mv = 0.0
    total_pnl = 0.0
    by_type = {}
    by_symbol = {}
    dte_buckets = {"<=7": 0, "<=14": 0, "<=30": 0, ">30": 0, "unknown": 0}

    for pos in positions:
        mv = _safe_float(pos.get("marketValue")) or 0.0
        pnl = _safe_float(pos.get("pnl")) or 0.0
        total_mv += mv
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

    return {
        "total_mv": total_mv,
        "total_pnl": total_pnl,
        "by_type": by_type,
        "by_symbol": symbols_sorted,
        "dte_buckets": dte_buckets,
        "top_positions": top_positions,
    }


def _option_intrinsic(opt_type, strike, underlying):
    if opt_type == "CALL":
        return max(0.0, underlying - strike)
    if opt_type == "PUT":
        return max(0.0, strike - underlying)
    return 0.0


def build_option_open_rows(positions, quotes, underlying_quotes, step, span):
    rows = []
    levels = [i * step for i in range(-span, span + 1) if i != 0]
    for pos in positions:
        if pos.get("assetType") != "OPTION":
            continue
        instrument = pos.get("instrument", {})
        opt_type = instrument.get("putCall") or instrument.get("optionType") or ""
        strike = _safe_float(instrument.get("strikePrice"))
        underlying = instrument.get("underlyingSymbol")
        if strike is None or not underlying:
            continue

        avg_price = _safe_float(
            pos.get("avgPrice")
            or pos.get("averagePrice")
            or pos.get("averageLongPrice")
            or pos.get("averageShortPrice")
        ) or 0.0
        qty = _safe_float(pos.get("qty")) or 0.0
        cost_basis = avg_price * qty * 100

        opt_quote = quotes.get(pos.get("symbol"), {})
        mark = opt_quote.get("mark") or opt_quote.get("last") or opt_quote.get("bid") or opt_quote.get("ask") or 0.0
        current_pl = (mark * qty * 100) - cost_basis

        breakeven = strike + avg_price if opt_type == "CALL" else strike - avg_price
        under_quote = underlying_quotes.get(underlying, {})
        underlying_last = under_quote.get("last") or under_quote.get("mark") or under_quote.get("bid") or under_quote.get("ask")

        pl_levels = {}
        for delta in levels:
            level_price = breakeven + delta
            intrinsic = _option_intrinsic(opt_type, strike, level_price)
            pl_levels[delta] = round((intrinsic * qty * 100) - cost_basis, 2)

        rows.append({
            "symbol": pos.get("symbol"),
            "underlying": underlying,
            "type": opt_type,
            "strike": strike,
            "qty": qty,
            "avg_price": avg_price,
            "mark": mark,
            "current_pl": round(current_pl, 2),
            "breakeven": round(breakeven, 2),
            "underlying_last": underlying_last,
            "pl_levels": pl_levels,
        })

    return levels, rows


def get_account_hashes():
    client = get_client()
    response = schwab_request(lambda: client.get_account_numbers(), "get_account_numbers")
    response.raise_for_status()
    data = response.json()
    return {row.get("accountNumber"): row.get("hashValue") for row in data}


def fetch_transactions_one_year():
    client = get_client()
    end_date = dt.datetime.now(dt.timezone.utc)
    start_date = end_date - dt.timedelta(days=365)
    account_hashes = get_account_hashes()
    all_txns = []
    for acct_num, acct_hash in account_hashes.items():
        window_start = start_date
        while window_start < end_date:
            window_end = min(window_start + dt.timedelta(days=59), end_date)
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
                    txn["accountNumber"] = acct_num
                all_txns.extend(data)
            window_start = window_end + dt.timedelta(seconds=1)
    return all_txns


def _txn_symbol(txn):
    item = txn.get("transactionItem", {}) or {}
    instr = item.get("instrument", {}) or {}
    return instr.get("symbol") or item.get("symbol") or txn.get("symbol")


def _txn_qty(txn):
    item = txn.get("transactionItem", {}) or {}
    amt = item.get("amount")
    return _safe_float(amt) or 0.0


def _txn_pnl(txn):
    return _safe_float(txn.get("netAmount")) or 0.0


def _txn_date(txn):
    raw = txn.get("transactionDate") or txn.get("tradeDate")
    if not raw:
        return dt.date.today()
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except Exception:
        return dt.date.today()


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


def build_yearly_summary(period, status_filter):
    entries = []
    today = dt.date.today()

    # Open positions (unrealized P/L)
    positions = fetch_positions()
    for pos in positions:
        if status_filter in ("closed",):
            continue
        entries.append({
            "symbol": pos.get("symbol") or "UNKNOWN",
            "assetType": pos.get("assetType") or "UNKNOWN",
            "qty": _safe_float(pos.get("qty")) or 0.0,
            "pnl": _safe_float(pos.get("pnl")) or 0.0,
            "status": "open",
            "date": today,
        })

    # Closed trades (net cash flow)
    if status_filter in ("all", "closed"):
        for txn in fetch_transactions_one_year():
            if txn.get("transactionType") and "TRADE" not in str(txn.get("transactionType")):
                continue
            symbol = _txn_symbol(txn)
            if not symbol:
                continue
            entries.append({
                "symbol": symbol,
                "assetType": (txn.get("transactionItem", {}) or {}).get("instrument", {}).get("assetType", "UNKNOWN"),
                "qty": _txn_qty(txn),
                "pnl": _txn_pnl(txn),
                "status": "closed",
                "date": _txn_date(txn),
            })

    # Group by period + symbol
    grouped = {}
    for entry in entries:
        if status_filter != "all" and entry["status"] != status_filter:
            continue
        bucket = bucket_period(entry["date"], period)
        key = (bucket, entry["symbol"], entry["status"])
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

    return list(grouped.values())


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

    return render_template(
        "alerts.html",
        rows=rows,
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
        results=results,
        error=error,
    )


@app.route("/movers-agent", methods=["GET", "POST"])
def movers_agent():
    info = None
    error = None
    snapshot = load_movers_snapshot()
    universe = load_optionable_universe()
    lookback_days = int((snapshot or {}).get("lookback_days", 5))
    top_n = int((snapshot or {}).get("top_n", 10))
    max_seconds = int((snapshot or {}).get("max_seconds", 25))

    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "run_scan":
            try:
                lookback_days = max(1, int(request.form.get("lookback_days", 5)))
                top_n = max(1, int(request.form.get("top_n", 10)))
                max_seconds = max(10, int(request.form.get("max_seconds", 25)))
                if not universe:
                    raise RuntimeError("No optionable universe loaded. Check optionable_universe.json.")
                snapshot = run_optionable_weekly_scan(
                    universe,
                    lookback_days=lookback_days,
                    top_n=top_n,
                    max_seconds=max_seconds,
                    previous=snapshot,
                )
                save_movers_snapshot(snapshot)
                rows_count = len(snapshot.get("rows") or [])
                covered = int(snapshot.get("covered_count", 0))
                total = int(snapshot.get("universe_size", 0))
                info = f"Weekly scan completed. Returned {rows_count} symbols. Coverage {covered}/{total}."
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

    return render_template(
        "movers_agent.html",
        snapshot=snapshot,
        universe_size=len(universe),
        info=info,
        error=error,
        defaults={
            "lookback_days": lookback_days,
            "top_n": top_n,
            "max_seconds": max_seconds,
        },
    )


@app.route("/tickets", methods=["GET", "POST"])
def tickets():
    error = None
    tickets_data = load_tickets()
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


@app.route("/risk")
def risk_dashboard():
    error = None
    summary = {}
    try:
        positions = fetch_positions()
        summary = build_risk_summary(positions)
    except Exception as exc:
        error = str(exc)

    return render_template("risk.html", summary=summary, error=error)


@app.route("/options-open")
def options_open():
    error = None
    step = float(request.args.get("step", 1))
    span = int(request.args.get("span", 5))
    levels = []
    rows = []
    try:
        positions = fetch_positions()
        option_positions = [p for p in positions if p.get("assetType") == "OPTION"]
        option_symbols = sorted({p.get("symbol") for p in option_positions if p.get("symbol")})
        underlying_symbols = sorted({
            (p.get("instrument", {}) or {}).get("underlyingSymbol")
            for p in option_positions
            if (p.get("instrument", {}) or {}).get("underlyingSymbol")
        })
        quotes = fetch_quotes(option_symbols)
        underlying_quotes = fetch_quotes(underlying_symbols)
        levels, rows = build_option_open_rows(option_positions, quotes, underlying_quotes, step, span)
    except Exception as exc:
        error = str(exc)

    return render_template(
        "options_open.html",
        levels=levels,
        rows=rows,
        step=step,
        span=span,
        error=error,
    )


@app.route("/summary")
def summary():
    period = request.args.get("period", "month")
    status_filter = request.args.get("status", "all")
    sort_by = request.args.get("sort", "profit")
    tz = request.args.get("tz")
    error = None
    rows = []
    try:
        if tz:
            settings = load_settings()
            settings["timezone"] = tz
            save_settings(settings)
        rows = build_yearly_summary(period, status_filter)
        if sort_by == "loss":
            rows = sorted(rows, key=lambda r: r["pnl"])
        elif sort_by == "symbol":
            rows = sorted(rows, key=lambda r: (r["symbol"], r["period"]))
        else:
            rows = sorted(rows, key=lambda r: r["pnl"], reverse=True)
    except Exception as exc:
        error = str(exc)

    return render_template(
        "summary.html",
        rows=rows,
        error=error,
        period=period,
        status=status_filter,
        sort_by=sort_by,
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
    rows = build_yearly_summary(period, status_filter)
    headers = ["period", "symbol", "status", "assetType", "qty", "pnl"]
    return _csv_response("summary.csv", rows, headers)


@app.route("/tools")
def tools():
    state = load_monitor_state()
    errors = load_error_log()
    status = None
    refresh_result = None
    settings = load_settings()
    tz = request.args.get("tz")
    if tz:
        settings["timezone"] = tz
        save_settings(settings)
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
    if request.args.get("check_schwab") == "1":
        try:
            client = get_client()
            response = schwab_request(lambda: client.get_account_numbers(), "status_check")
            status = {"ok": True, "status_code": response.status_code}
        except Exception as exc:
            status = {"ok": False, "error": str(exc)}
    return render_template(
        "tools.html",
        state=state,
        errors=errors,
        status=status,
        refresh_result=refresh_result,
        token_status=token_status,
        timezone=settings.get("timezone", "UTC"),
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
   
