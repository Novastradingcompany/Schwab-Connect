# =======================================
# strategies/bear_call.py — Nova POP + TOS-style + Logging Debug
# =======================================
import pandas as pd
import logging
from core.math_utils import calc_credit, calc_max_loss, calc_breakeven, calc_pop, bs_delta

# ----------------------------
# Logging setup (visible in terminal)
# ----------------------------
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
                   contracts: int = 1) -> pd.DataFrame:
    """
    Scan Bear Call vertical spreads:
      - Sell lower strike CALL, buy higher strike CALL.
      - Uses Nova-style POP from Black–Scholes (N(d2)) when IV/T are available.
      - Includes logging for POP and delta verification.
    """
    trades = []

    if chain is None or chain.empty:
        log.info("DEBUG: Empty option chain received.")
        return pd.DataFrame(trades)

    # Ensure chain sorted ascending by strike so spread pairing is consistent
    chain = chain.sort_values("strike", ascending=True).reset_index(drop=True)

    for i in range(len(chain) - 1):
        sell_leg = chain.iloc[i]
        buy_leg = chain.iloc[i + 1]

        # Correct order: sell lower strike, buy higher strike
        if float(sell_leg["strike"]) > float(buy_leg["strike"]):
            sell_leg, buy_leg = buy_leg, sell_leg

        width = abs(float(sell_leg["strike"]) - float(buy_leg["strike"]))
        if width <= 0 or width > float(max_width):
            continue

        try:
            sell_bid = float(sell_leg["bid"]); sell_ask = float(sell_leg["ask"])
            buy_bid  = float(buy_leg["bid"]);  buy_ask  = float(buy_leg["ask"])
        except Exception:
            continue

        # Skip invalid quotes
        if sell_bid <= 0 or sell_ask <= 0 or buy_bid <= 0 or buy_ask <= 0:
            continue

        # Only OTM short calls (bearish bias)
        # Skip if short strike is below or equal to spot (ITM)
        if float(sell_leg["strike"]) <= float(spot_price):
            continue

        # --- Midpoints per share
        sell_mid = (sell_bid + sell_ask) / 2
        buy_mid  = (buy_bid + buy_ask) / 2

        # --- Credit per contract ($)
        credit_per_contract = calc_credit(sell_mid, buy_mid)
        total_credit = round(credit_per_contract * contracts, 2)

        # --- Risk + metrics
        max_loss_val = calc_max_loss(width, credit_per_contract, contracts)
        breakeven = calc_breakeven(float(sell_leg["strike"]), credit_per_contract, "call", contracts)

        # --- Compute delta using BS if missing
        iv = float(sell_leg.get("impliedVolatility", 0)) / 100 if float(sell_leg.get("impliedVolatility", 0)) > 1 else float(sell_leg.get("impliedVolatility", 0))
        delta = sell_leg.get("delta")
        if delta is None and iv > 0 and T > 0:
            delta = bs_delta(
                S=float(spot_price),
                K=float(sell_leg["strike"]),
                T=T,
                r=0.02,
                sigma=iv,
                option_type="call"
            )

        # --- Debug output (visible in terminal)
        log.info(
            f"\nDEBUG POP INPUTS:\n"
            f"Strike={sell_leg['strike']} | Spot={spot_price} | Width={width}\n"
            f"Credit={credit_per_contract} | MaxLoss={max_loss_val} | Delta={delta} | IV={iv}"
        )

        # --- POP calculation using Nova formula (Black–Scholes N(d2))
        pop = calc_pop(
            short_strike=float(sell_leg["strike"]),
            spot=float(spot_price),
            width=width,
            credit=credit_per_contract,
            max_loss=max_loss_val,
            opt_type="call",
            delta=delta,
            contracts=contracts,
            iv=iv,
            T=T,
            r=0.02
        )

        # --- Filter out unqualified trades
        if not raw_mode:
            if max_loss_val > float(max_loss) or float(pop) < float(min_pop):
                continue

        trades.append({
            "Strategy": "Bear Call Vertical",
            "Expiry": expiry,
            "DTE": int(dte),
            "Trade": f"Sell {sell_leg['strike']} / Buy {buy_leg['strike']} CALL",
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

    # Log if no trades found
    if not trades:
        log.info("DEBUG: No valid bear call spreads found for this chain.")

    return pd.DataFrame(trades)







