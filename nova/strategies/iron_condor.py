import re
import logging
import pandas as pd

from nova.strategies.bull_put import scan_bull_put
from nova.strategies.bear_call import scan_bear_call

logging.basicConfig(format="%(message)s", level=logging.WARNING)
log = logging.getLogger(__name__)

_STRIKE_RE = re.compile(r"Sell\\s*([0-9.]+)\\s*/\\s*Buy\\s*([0-9.]+)")


def _extract_width(trade_str: str) -> float:
    m = _STRIKE_RE.search(str(trade_str))
    if not m:
        return 0.0
    a, b = float(m.group(1)), float(m.group(2))
    return abs(a - b)


def _extract_sell_strike(trade_str: str) -> float:
    m = _STRIKE_RE.search(str(trade_str))
    return float(m.group(1)) if m else 0.0


def scan_iron_condor(chains: dict,
                     spot_price: float,
                     expiry: str,
                     dte: int,
                     T: float,
                     max_width: float,
                     max_loss: float,
                     min_pop: float,
                     raw_mode: bool,
                     contracts: int = 1,
                     pricing_mode: str = "mid",
                     custom_limit: float | None = None) -> pd.DataFrame:
    """
    Scan Iron Condor setups by pairing Bull Put and Bear Call spreads.
    """
    puts = chains.get("puts")
    calls = chains.get("calls")
    if puts is None or calls is None:
        log.info("Iron Condor received missing puts/calls.")
        return pd.DataFrame()

    df_puts = scan_bull_put(
        puts, spot_price, expiry, dte, T,
        max_width, max_loss, min_pop if not raw_mode else 0, True,
        contracts,
        pricing_mode=pricing_mode,
        custom_limit=custom_limit,
    )
    df_calls = scan_bear_call(
        calls, spot_price, expiry, dte, T,
        max_width, max_loss, min_pop if not raw_mode else 0, True,
        contracts,
        pricing_mode=pricing_mode,
        custom_limit=custom_limit,
    )

    if df_puts is None or df_puts.empty or df_calls is None or df_calls.empty:
        return pd.DataFrame()

    trades = []

    for _, put_leg in df_puts.iterrows():
        for _, call_leg in df_calls.iterrows():
            put_trade = str(put_leg["Trade"])
            call_trade = str(call_leg["Trade"])

            width_put = _extract_width(put_trade)
            width_call = _extract_width(call_trade)
            if width_put <= 0 or width_call <= 0:
                continue

            credit_put_pc = float(put_leg["Credit (Realistic)"])
            credit_call_pc = float(call_leg["Credit (Realistic)"])
            total_credit_pc = round(credit_put_pc + credit_call_pc, 2)
            total_credit = round(total_credit_pc * contracts, 2)

            max_width_leg = max(width_put, width_call)
            max_loss_pc = (max_width_leg * 100) - total_credit_pc
            max_loss_total = round(max_loss_pc * contracts, 2)

            pop_put = float(put_leg["POP %"])
            pop_call = float(call_leg["POP %"])
            pop = round((pop_put + pop_call) / 2.0, 1)

            short_put = _extract_sell_strike(put_trade)
            short_call = _extract_sell_strike(call_trade)
            credit_per_share = total_credit_pc / 100.0
            lower_be = round(short_put - credit_per_share, 2)
            upper_be = round(short_call + credit_per_share, 2)
            be_str = f"{lower_be} / {upper_be}"

            dist_put = abs(spot_price - short_put) / spot_price * 100.0 if spot_price else 0.0
            dist_call = abs(short_call - spot_price) / spot_price * 100.0 if spot_price else 0.0
            distance_pct = round(min(dist_put, dist_call), 2)

            if not raw_mode:
                if max_loss_total > float(max_loss) or pop < float(min_pop):
                    continue

            trades.append({
                "Strategy": "Iron Condor",
                "Expiry": expiry,
                "DTE": int(dte),
                "Put Spread": put_trade,
                "Call Spread": call_trade,
                "Credit (Realistic)": total_credit_pc,
                "Total Credit ($)": total_credit,
                "Max Loss ($)": max_loss_total,
                "POP %": pop,
                "Breakeven": be_str,
                "Distance %": distance_pct,
                "Contracts": int(contracts),
                "Spot": round(float(spot_price), 2),
            })

    return pd.DataFrame(trades)
