# ============================
# core/style_utils.py
# ============================

import pandas as pd


def style_table(df: pd.DataFrame, min_pop_val: int = 0):
    """
    Apply consistent formatting, conditional color bands, and derived metrics to the trade DataFrame.

    Features:
    - Adds Premium Yield % (credit / width * 100)
    - Formats $ and % columns
    - Color-codes POP% by risk level
    - Sorts by POP descending
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ============================
    # ✅ Calculate Premium Yield %
    # ============================
    if "Credit ($)" in df.columns and "Trade" in df.columns:
        def _extract_width(trade_str: str):
            """Extract strike width from the trade string (e.g., 'Sell 240 / Buy 237.5 PUT')."""
            try:
                parts = trade_str.replace("Sell", "").replace("Buy", "").split("/")
                strikes = [float(s.strip().split()[0]) for s in parts if s.strip()]
                if len(strikes) == 2:
                    return abs(strikes[0] - strikes[1])
                return None
            except Exception:
                return None

        df["Width"] = df["Trade"].apply(_extract_width)
        def _to_float(val):
            try:
              return float(str(val).replace("$", "").replace(",", "").strip())
            except:
              return 0.0

        df["Premium Yield %"] = df.apply(
            lambda x: round((_to_float(x.get("Credit ($)", 0)) / (float(x.get("Width", 0)) * 100)) * 100, 1)
            if x.get("Width") not in [None, 0] else None,
            axis=1
   )

        

    # ============================
    # ✅ Format numeric columns
    # ============================
    money_cols = ["Credit ($)", "Credit (Realistic)", "Max Loss ($)", "Breakeven", "Spot"]
    percent_cols = ["POP %", "Distance %", "Premium Yield %"]

    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"${float(x):,.2f}" if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else x
            )

    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{float(x):.1f}%" if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else x
            )

    # ============================
    # ✅ Sort by POP descending
    # ============================
    if "POP %" in df.columns:
        try:
            df["POP_numeric"] = df["POP %"].astype(str).str.replace("%", "").astype(float)
            df = df.sort_values(by="POP_numeric", ascending=False)
            df = df.drop(columns=["POP_numeric"])
        except Exception:
            pass

    # ============================
    # ✅ POP color bands
    # ============================
    def _pop_color(val):
        try:
            num = float(str(val).replace("%", ""))
            if num >= 80:
                return "background-color: #1e4620; color: #b8f7b1;"  # Green (safe)
            elif num >= 70:
                return "background-color: #5a4b1e; color: #f3e7a4;"  # Yellow (medium)
            else:
                return "background-color: #4b1e1e; color: #f7b1b1;"  # Red (risky)
        except Exception:
            return ""

    styled = df.style
    if "POP %" in df.columns:
        # ✅ Use new pandas Styler.map if available, else fallback to applymap
        if hasattr(styled, "map"):
            styled = styled.map(_pop_color, subset=["POP %"])
        else:
            styled = styled.applymap(_pop_color, subset=["POP %"])

        styled = styled.set_properties(subset=["POP %"], **{"font-weight": "bold"})

    styled.format(precision=2)

    return styled



