# =======================================
# core/trade_utils.py
# =======================================
import pandas as pd


# --------------------------------------------------
# Realistic Credit (per contract, not multiplied)
# --------------------------------------------------
def calc_realistic_credit(sell_leg: pd.Series, buy_leg: pd.Series) -> float:
    """
    Calculates a realistic credit PER CONTRACT (not total).
    Uses midpoint pricing minus a small slippage to approximate fills.
    Example:
        sell_mid = (bid + ask) / 2 for the short leg
        buy_mid  = (bid + ask) / 2 for the long leg
        credit   = (sell_mid - buy_mid - slippage) * 100
    """
    try:
        sell_bid = float(sell_leg.get("bid", 0))
        sell_ask = float(sell_leg.get("ask", 0))
        buy_bid = float(buy_leg.get("bid", 0))
        buy_ask = float(buy_leg.get("ask", 0))

        # Midpoints
        sell_mid = (sell_bid + sell_ask) / 2
        buy_mid = (buy_bid + buy_ask) / 2

        # Apply minor slippage for realism
        slippage = 0.01
        credit = (sell_mid - buy_mid - slippage) * 100  # per contract
        return round(max(credit, 0.0), 2)

    except Exception:
        return 0.0


# --------------------------------------------------
# Midpoint Credit (for reference / diagnostic)
# --------------------------------------------------
def calc_mid_credit(sell_leg: pd.Series, buy_leg: pd.Series) -> float:
    """
    Pure midpoint credit PER CONTRACT (no slippage, not total).
    """
    try:
        sell_mid = (float(sell_leg["bid"]) + float(sell_leg["ask"])) / 2
        buy_mid = (float(buy_leg["bid"]) + float(buy_leg["ask"])) / 2
        credit = (sell_mid - buy_mid) * 100
        return round(max(credit, 0.0), 2)
    except Exception:
        return 0.0


# --------------------------------------------------
# Formatting Helper for Auto Take and Reports
# --------------------------------------------------
def format_credit_summary(per_contract_credit: float, contracts: int) -> str:
    """
    Returns formatted text for Nova’s Auto Take and reports.

    Example:
        Credit Received: $22.50 (Total: $67.50 for 3 contracts)
    """
    try:
        total_credit = per_contract_credit * contracts
        return (
            f"Credit Received: ${per_contract_credit:,.2f} "
            f"(Total: ${total_credit:,.2f} for {contracts} contract"
            f"{'s' if contracts > 1 else ''})"
        )
    except Exception:
        return "Credit Received: $0.00"


