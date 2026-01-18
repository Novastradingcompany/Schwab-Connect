import pandas as pd
import math
from nova_math import calc_credit, calc_max_loss, calc_breakeven, calc_pop

# ----------------------------
# Black-Scholes Delta helper
# ----------------------------
def bs_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return None
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if option_type == "call":
            return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        else:
            return -0.5 * (1 - math.erf(d1 / math.sqrt(2)))
    except Exception:
        return None

# ----------------------------
# Vertical spread scanner
# ----------------------------
def scan_verticals(chain, spot_price, expiry, dte, T,
                   max_width, max_loss, min_pop,
                   raw_mode, opt_type, contracts=1):

    trades = []
    for i in range(len(chain) - 1):
        sell_leg = chain.iloc[i]
        buy_leg = chain.iloc[i + 1]

        # --- enforce correct strike order ---
        if opt_type == "put":
            if sell_leg["strike"] < buy_leg["strike"]:
                sell_leg, buy_leg = buy_leg, sell_leg
        elif opt_type == "call":
            if sell_leg["strike"] > buy_leg["strike"]:
                sell_leg, buy_leg = buy_leg, sell_leg

        width = abs(buy_leg["strike"] - sell_leg["strike"])
        if width > max_width:
            continue

        # --- mid-prices ---
        sell_mid = (sell_leg["bid"] + sell_leg["ask"]) / 2 if sell_leg["ask"] > 0 else sell_leg["bid"]
        buy_mid  = (buy_leg["bid"] + buy_leg["ask"]) / 2 if buy_leg["ask"] > 0 else buy_leg["ask"]

        if sell_mid <= 0 or buy_mid < 0:
            continue

        # --- OTM requirement ---
        if opt_type == "put" and sell_leg["strike"] >= spot_price:
            continue
        if opt_type == "call" and sell_leg["strike"] <= spot_price:
            continue

        # --- credit calculations ---
        credit_mid = (sell_mid - buy_mid) * contracts * 100
        credit_realistic = max((sell_leg["bid"] - buy_leg["ask"]) * contracts * 100, 0)
        if credit_realistic <= 0 and credit_mid <= 0:
            continue

        effective_credit = credit_realistic if credit_realistic > 0 else credit_mid
        max_loss_val = calc_max_loss(width, effective_credit, contracts)
        breakeven = calc_breakeven(sell_leg["strike"], effective_credit, opt_type)

        # --- POP ---
        pop = calc_pop(
            short_strike=sell_leg["strike"],
            spot=spot_price,
            width=width,
            credit=effective_credit,
            max_loss=max_loss_val,
            opt_type=opt_type,
            delta=sell_leg.get("delta", None)
        )

        if not raw_mode:
            if max_loss_val > max_loss or pop < min_pop:
                continue

        trades.append({
            "Strategy": f"{'Bull Put' if opt_type == 'put' else 'Bear Call'} Vertical",
            "Expiry": expiry,
            "DTE": dte,
            "Trade": f"Sell {sell_leg['strike']} / Buy {buy_leg['strike']} {opt_type.upper()}",
            "Credit ($)": round(credit_mid, 2),
            "Credit (Realistic)": round(credit_realistic, 2),
            "Max Loss ($)": max_loss_val,
            "POP %": pop,
            "Breakeven": breakeven,
            "Distance %": round((abs(sell_leg['strike'] - spot_price) / spot_price) * 100, 2),
            "Delta": sell_leg.get("delta", None),
            "Contracts": contracts,
            "Spot": round(spot_price, 2)
        })

    return pd.DataFrame(trades), None

# ----------------------------
# Iron Condor scanner
# ----------------------------
def scan_condors(chain, spot_price, expiry, dte, T,
                 max_width, max_loss, min_pop,
                 raw_mode, contracts=1):
    """
    Build iron condors by combining a bull put and bear call spread.
    Accepts either a yfinance OptionChain object or a dict with 'puts' and 'calls'.
    """
    # ✅ handle both dict and yfinance OptionChain
    if isinstance(chain, dict):
        puts = chain["puts"].reset_index(drop=True)
        calls = chain["calls"].reset_index(drop=True)
    else:
        puts = chain.puts.reset_index(drop=True)
        calls = chain.calls.reset_index(drop=True)

    condors = []

    # Generate bull put legs
    put_trades, _ = scan_verticals(
        puts, spot_price, expiry, dte, T,
        max_width, max_loss, min_pop,
        True, "put", contracts
    )
    # Generate bear call legs
    call_trades, _ = scan_verticals(
        calls, spot_price, expiry, dte, T,
        max_width, max_loss, min_pop,
        True, "call", contracts
    )

    for _, put_row in put_trades.iterrows():
        for _, call_row in call_trades.iterrows():
            total_credit = put_row["Credit ($)"] + call_row["Credit ($)"]
            total_max_loss = max(put_row["Max Loss ($)"], call_row["Max Loss ($)"])
            avg_pop = (put_row["POP %"] + call_row["POP %"]) / 2

            if not raw_mode:
                if total_max_loss > max_loss or avg_pop < min_pop:
                    continue

            condors.append({
                "Strategy": "Iron Condor",
                "Expiry": expiry,
                "DTE": dte,
                "Trade": f"{put_row['Trade']} + {call_row['Trade']}",
                "Credit ($)": total_credit,
                "Max Loss ($)": total_max_loss,
                "POP %": avg_pop,
                "Contracts": contracts,
                "Spot": round(spot_price, 2)
            })

    return pd.DataFrame(condors), None

# ----------------------------
# Style table
# ----------------------------
def style_table(df, min_pop_val):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    if "Credit ($)" in df.columns:
        df["Credit ($)"] = df["Credit ($)"].map(lambda x: f"${x:,.2f}")
    if "Credit (Realistic)" in df.columns:
        df["Credit (Realistic)"] = df["Credit (Realistic)"].map(lambda x: f"${x:,.2f}")
    if "Max Loss ($)" in df.columns:
        df["Max Loss ($)"] = df["Max Loss ($)"].map(lambda x: f"${x:,.2f}")
    if "Breakeven" in df.columns:
        df["Breakeven"] = df["Breakeven"].map(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
    if "POP %" in df.columns:
        df["POP %"] = df["POP %"].map(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    if "Distance %" in df.columns:
        df["Distance %"] = df["Distance %"].map(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    if "Spot" in df.columns:
        df["Spot"] = df["Spot"].apply(
            lambda x: f"${float(x):,.2f}" if str(x).replace('.', '', 1).isdigit() else x
        )

    return df.style
























































