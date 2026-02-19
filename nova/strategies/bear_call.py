import logging
import pandas as pd

from nova.core.math_utils import calc_credit, calc_max_loss, calc_breakeven, calc_pop, bs_delta

logging.basicConfig(format="%(message)s", level=logging.WARNING)
log = logging.getLogger(__name__)


def scan_bear_call(chain: pd.DataFrame,
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
    Scan Bear Call vertical spreads:
      - Sell lower strike CALL, buy higher strike CALL.
      - POP uses Black-Scholes when IV/T are available.
    """
    trades = []

    if chain is None or chain.empty:
        log.info("Empty option chain received.")
        return pd.DataFrame(trades)

    chain = chain.sort_values("strike", ascending=True).reset_index(drop=True)

    for i in range(len(chain) - 1):
        sell_leg = chain.iloc[i]
        buy_leg = chain.iloc[i + 1]

        if float(sell_leg["strike"]) > float(buy_leg["strike"]):
            sell_leg, buy_leg = buy_leg, sell_leg

        width = abs(float(sell_leg["strike"]) - float(buy_leg["strike"]))
        if width <= 0 or width > float(max_width):
            continue

        try:
            sell_bid = float(sell_leg["bid"])
            sell_ask = float(sell_leg["ask"])
            buy_bid = float(buy_leg["bid"])
            buy_ask = float(buy_leg["ask"])
        except Exception:
            continue

        if sell_bid <= 0 or sell_ask <= 0 or buy_bid <= 0 or buy_ask <= 0:
            continue

        if float(sell_leg["strike"]) <= float(spot_price):
            continue

        sell_mid = (sell_bid + sell_ask) / 2
        buy_mid = (buy_bid + buy_ask) / 2
        credit_mid = calc_credit(sell_mid, buy_mid)
        credit_natural = calc_credit(sell_bid, buy_ask)

        if pricing_mode == "natural":
            credit_per_contract = credit_natural
        elif pricing_mode == "custom" and custom_limit is not None:
            credit_per_contract = max(float(custom_limit) * 100, 0.0)
        else:
            credit_per_contract = credit_mid
        total_credit = round(credit_per_contract * contracts, 2)

        max_loss_val = calc_max_loss(width, total_credit, contracts)
        max_loss_per_contract = calc_max_loss(width, credit_per_contract, 1)
        breakeven = calc_breakeven(float(sell_leg["strike"]), credit_per_contract, "call", 1)

        iv_raw = sell_leg.get("impliedVolatility_raw", sell_leg.get("impliedVolatility", 0))
        try:
            iv_calc = float(iv_raw)
        except (TypeError, ValueError):
            iv_calc = 0.0
        if iv_calc > 1:
            iv_calc = iv_calc / 100

        delta_raw = sell_leg.get("delta_raw", sell_leg.get("delta"))
        delta_calc = delta_raw
        if isinstance(delta_calc, (int, float)) and abs(delta_calc) > 1:
            delta_calc = None
        if delta_calc is None and iv_calc > 0 and T > 0:
            delta_calc = bs_delta(
                S=float(spot_price),
                K=float(sell_leg["strike"]),
                T=T,
                r=0.02,
                sigma=iv_calc,
                option_type="call",
            )

        pop = calc_pop(
            short_strike=float(sell_leg["strike"]),
            spot=float(spot_price),
            width=width,
            credit=credit_per_contract,
            max_loss=max_loss_per_contract,
            opt_type="call",
            delta=delta_calc,
            contracts=1,
            iv=iv_calc,
            T=T,
            r=0.02,
        )

        if not raw_mode:
            if max_loss_val > float(max_loss) or float(pop) < float(min_pop):
                continue

        trades.append({
            "Strategy": "Bear Call Vertical",
            "Expiry": expiry,
            "DTE": int(dte),
            "Trade": f"Sell {sell_leg['strike']} / Buy {buy_leg['strike']} CALL",
            "Credit (Realistic)": round(credit_per_contract, 2),
            "Credit (Mid $)": round(credit_mid, 2),
            "Credit (Natural $)": round(credit_natural, 2),
            "Total Credit ($)": total_credit,
            "Max Loss ($)": round(max_loss_val, 2),
            "POP %": round(float(pop), 1),
            "Breakeven": round(float(breakeven), 2),
            "Distance %": round(abs(float(sell_leg['strike']) - float(spot_price)) / float(spot_price) * 100, 2),
            "Delta": delta_raw if delta_raw is not None else delta_calc,
            "Implied Vol": round(iv_raw, 3) if isinstance(iv_raw, (int, float)) else iv_raw,
            "Contracts": int(contracts),
            "Spot": round(float(spot_price), 2),
        })

    return pd.DataFrame(trades)
