import logging
import pandas as pd

from nova.core.math_utils import calc_credit, calc_max_loss, calc_breakeven, calc_pop, bs_delta

logging.basicConfig(format="%(message)s", level=logging.WARNING)
log = logging.getLogger(__name__)


def scan_bull_put(chain: pd.DataFrame,
                  spot_price: float,
                  expiry: str,
                  dte: int,
                  T: float,
                  max_width: float,
                  max_loss: float,
                  min_pop: float,
                  raw_mode: bool,
                  contracts: int = 1) -> pd.DataFrame:
    """
    Scan Bull Put vertical spreads:
      - Sell higher strike PUT, buy lower strike PUT.
      - POP uses Black-Scholes when IV/T are available.
    """
    trades = []

    if chain is None or chain.empty:
        log.info("Empty option chain received.")
        return pd.DataFrame(trades)

    chain = chain.sort_values("strike", ascending=True).reset_index(drop=True)

    for i in range(len(chain) - 1):
        sell_leg = chain.iloc[i + 1]
        buy_leg = chain.iloc[i]

        if float(sell_leg["strike"]) < float(buy_leg["strike"]):
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

        if float(sell_leg["strike"]) >= float(spot_price):
            continue

        sell_mid = (sell_bid + sell_ask) / 2
        buy_mid = (buy_bid + buy_ask) / 2

        credit_per_contract = calc_credit(sell_mid, buy_mid)
        total_credit = round(credit_per_contract * contracts, 2)

        max_loss_val = calc_max_loss(width, credit_per_contract, contracts)
        breakeven = calc_breakeven(float(sell_leg["strike"]), credit_per_contract, "put", contracts)

        iv = float(sell_leg.get("impliedVolatility", 0))
        iv = iv / 100 if iv > 1 else iv
        delta = sell_leg.get("delta")
        if delta is None and iv > 0 and T > 0:
            delta = bs_delta(
                S=float(spot_price),
                K=float(sell_leg["strike"]),
                T=T,
                r=0.02,
                sigma=iv,
                option_type="put",
            )

        pop = calc_pop(
            short_strike=float(sell_leg["strike"]),
            spot=float(spot_price),
            width=width,
            credit=credit_per_contract,
            max_loss=max_loss_val,
            opt_type="put",
            delta=delta,
            contracts=contracts,
            iv=iv,
            T=T,
            r=0.02,
        )

        if not raw_mode:
            if max_loss_val > float(max_loss) or float(pop) < float(min_pop):
                continue

        trades.append({
            "Strategy": "Bull Put Vertical",
            "Expiry": expiry,
            "DTE": int(dte),
            "Trade": f"Sell {sell_leg['strike']} / Buy {buy_leg['strike']} PUT",
            "Credit (Realistic)": round(credit_per_contract, 2),
            "Total Credit ($)": total_credit,
            "Max Loss ($)": round(max_loss_val, 2),
            "POP %": round(float(pop), 1),
            "Breakeven": round(float(breakeven), 2),
            "Distance %": round(abs(float(sell_leg['strike']) - float(spot_price)) / float(spot_price) * 100, 2),
            "Delta": delta,
            "Implied Vol": round(iv, 3),
            "Contracts": int(contracts),
            "Spot": round(float(spot_price), 2),
        })

    return pd.DataFrame(trades)
