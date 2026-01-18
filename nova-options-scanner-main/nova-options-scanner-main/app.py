import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# 🔗 Nova shared modules
from nova_rules import NOVA_RULES, get_max_loss_threshold
from core.style_utils import style_table
from core.export_utils import download_excel_button
from core.pdf_utils import download_pdf_button
from core.trade_utils import format_credit_summary

from strategies.bull_put import scan_bull_put
from strategies.bear_call import scan_bear_call
from strategies.iron_condor import scan_iron_condor


# =========================================================
# ⚙️ Page + API init
# =========================================================
st.set_page_config(
    page_title="Nova Options Scanner",
    page_icon="favicon.png",
    layout="centered",
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================================================
# 🧠 Session State boot
# =========================================================
def _boot_state():
    defaults = {
        "nova_chat": [],
        "last_trades_df": None,
        "last_trades_records": None,
        "last_meta": None,
        "auto_summary": None,
        "last_query": None,
        "total_credit_summary": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_results():
    st.session_state.last_trades_df = None
    st.session_state.last_trades_records = None
    st.session_state.last_meta = None
    st.session_state.auto_summary = None
    st.session_state.total_credit_summary = ""


_boot_state()


# =========================================================
# 🧹 Clean option chain data
# =========================================================
def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["bid"] >= 0.01) & (df["ask"] >= 0.01)]
    df = df[df["ask"] >= df["bid"]]
    if "openInterest" in df.columns:
        df = df[df["openInterest"] > 0]
    return df


# =========================================================
# 🧭 Scanner UI
# =========================================================
st.title("📊 Nova Options Scanner")
st.caption("Version 10-24-2025")

ticker_input = st.text_input("Enter Ticker Symbol", "NVDA", key="ticker").upper()
exp_dates, spot_price, ticker_obj = [], None, None

# 💰 Account Cash → dynamic Max Loss
cash_balance = st.number_input("Account Cash ($)", min_value=100, value=2000, step=50, key="account_cash")

if ticker_input:
    try:
        ticker_obj = yf.Ticker(ticker_input)
        exp_dates = [
            datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%y")
            for d in ticker_obj.options
        ]
        spot_price = ticker_obj.history(period="1d")["Close"].iloc[-1]
        st.subheader(f"📈 {ticker_input} Current Price: ${spot_price:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Could not fetch ticker data: {e}")
        exp_dates, spot_price = [], None

expiry_selected = (
    st.selectbox("Select Expiration Date", exp_dates, key="expiry")
    if exp_dates
    else None
)
if not exp_dates:
    st.warning("⚠️ Could not fetch expiration dates for this ticker.")

# --- Sliders ---
max_width = st.slider("Max Spread Width ($)", 0.5, 5.0, 2.5, 0.5, key="width")

max_loss_default = get_max_loss_threshold(cash_balance)
max_loss = st.slider("Max Loss ($)", 50, 1000, max_loss_default, 50, key="max_loss_scanner")

min_pop = st.slider("Minimum POP (%)", 0, 95, 70, 1, key="pop")
contracts = st.slider("Contracts", 1, 10, 1, key="contracts")

raw_mode = st.checkbox("🔎 Show Raw (ignore filters)", key="raw")
spread_type = st.radio("Choose Strategy", ["Bull Put", "Bear Call", "Iron Condor"], key="strategy")

# --- Query tracking ---
current_query = {
    "ticker": ticker_input,
    "expiry": expiry_selected,
    "strategy": spread_type,
    "width": float(max_width),
    "loss": int(max_loss),
    "pop": int(min_pop),
    "contracts": int(contracts),
    "raw": bool(raw_mode),
}

if st.session_state.last_query and st.session_state.last_query != current_query:
    _reset_results()


# =========================================================
# 🚀 Run Scan
# =========================================================
just_scanned = False

if st.button("Scan", key="main_scan"):
    if not expiry_selected or not ticker_obj or spot_price is None:
        _reset_results()
        st.warning("⚠️ Please select a valid ticker and expiration before scanning.")
    else:
        _reset_results()
        st.info("🔍 Scan started — please wait...")
        try:
            expiry_obj = datetime.strptime(expiry_selected, "%m/%d/%y")
            expiry_iso = expiry_obj.strftime("%Y-%m-%d")
            opt_chain = ticker_obj.option_chain(expiry_iso)
            dte = (expiry_obj - datetime.now()).days
            T = dte / 365.0

            trades = pd.DataFrame()

            if spread_type == "Bull Put":
                puts = clean_chain(opt_chain.puts)
                trades = scan_bull_put(puts, spot_price, expiry_selected, dte, T,
                                       max_width, max_loss, min_pop, raw_mode, contracts)
            elif spread_type == "Bear Call":
                calls = clean_chain(opt_chain.calls)
                trades = scan_bear_call(calls, spot_price, expiry_selected, dte, T,
                                        max_width, max_loss, min_pop, raw_mode, contracts)
            elif spread_type == "Iron Condor":
                puts = clean_chain(opt_chain.puts)
                calls = clean_chain(opt_chain.calls)
                trades = scan_iron_condor({"puts": puts, "calls": calls}, spot_price,
                                          expiry_selected, dte, T, max_width,
                                          max_loss, min_pop, raw_mode, contracts)

            trades_df = trades[0] if isinstance(trades, tuple) else trades
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                trades_df["Symbol"] = ticker_input
                st.session_state.last_trades_df = trades_df
                st.session_state.last_trades_records = trades_df.to_dict(orient="records")
                st.session_state.last_meta = {
                    "ticker": ticker_input,
                    "expiry": expiry_selected,
                    "dte": int(dte),
                    "min_pop": int(min_pop),
                    "spot": float(spot_price),
                    "strategy": spread_type,
                    "contracts": int(contracts),
                    "loss": int(max_loss),
                }
                st.session_state.last_query = current_query
                just_scanned = True
            else:
                _reset_results()
                st.warning("⚠️ No valid trades found for the given filters.")
        except Exception as e:
            _reset_results()
            st.error(f"❌ Scan failed: {e}")


# =========================================================
# 🧾 Display Results & Export Tools
# =========================================================
def render_results(trades_df: pd.DataFrame, min_pop_val: int):
    if trades_df is None or trades_df.empty:
        st.warning("⚠️ No trades found to display.")
        return

    if "Credit (Realistic)" in trades_df.columns:
        trades_df["Total Credit ($)"] = (
            trades_df["Credit (Realistic)"].astype(float) * trades_df.get("Contracts", 1)
        )

    if "POP %" in trades_df.columns:
        trades_df = trades_df.sort_values(by="POP %", ascending=False)

    df_display = trades_df.copy()

    money_cols = ["Credit (Realistic)", "Total Credit ($)", "Max Loss ($)", "Breakeven", "Spot"]
    percent_cols = ["POP %", "Distance %"]
    for col in money_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"${float(x):,.2f}" if str(x).replace('.', '', 1).isdigit() else x
            )
    for col in percent_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{float(x):.1f}%" if str(x).replace('.', '', 1).isdigit() else x
            )

    total_credit_pc_sum = pd.to_numeric(
        trades_df["Credit (Realistic)"].astype(str).str.replace("$", "").str.replace(",", ""),
        errors="coerce"
    ).fillna(0).sum()

    live_contracts = int(st.session_state.last_meta.get("contracts", 1))
    total_credit_live = total_credit_pc_sum * live_contracts
    st.session_state.total_credit_summary = format_credit_summary(total_credit_live, live_contracts)

    st.success(f"✅ Found {len(df_display)} {st.session_state.last_meta['strategy']} candidates")
    st.dataframe(style_table(df_display, min_pop_val), width="stretch")

    st.markdown("#### 📁 Export Options")
    download_excel_button(df_display, filename_prefix=f"NovaScan_{st.session_state.last_meta['ticker']}")
    download_pdf_button(df_display, st.session_state.last_meta, st.session_state.auto_summary or "")


# =========================================================
# 🧾 Summary Prompt Builder (Nova Auto Take)
# =========================================================
def build_summary_prompt():
    meta = st.session_state.last_meta
    recs = st.session_state.last_trades_records
    if not meta or not recs:
        return "You are Nova. No scan results available."

    summary_line = st.session_state.total_credit_summary or ""
    current_max_loss = meta.get("loss", 0)

    lines = []
    for i, r in enumerate(recs, 1):
        credit_total = float(r.get("Total Credit ($)", r.get("Credit (Realistic)", 0)))
        max_loss_val = float(r.get("Max Loss ($)", 0))

        lines.append(
            "----------------------------------------\n"
            f"Stock Symbol: {r.get('Symbol')}\n"
            f"Strategy Type: {r.get('Strategy')}\n"
            f"Strike Prices: {r.get('Trade')}\n"
            f"Expiration Date: {r.get('Expiry')}\n"
            f"Credit Received: ${credit_total:,.2f} for {meta.get('contracts', 1)} contract"
            f"{'s' if meta.get('contracts', 1) > 1 else ''}\n"
            f"Max Loss: ${max_loss_val:,.2f}\n"
            f"Probability of Profit (POP): {r.get('POP %', 0)}%\n"
            f"Breakeven Price: ${r.get('Breakeven', 0)}\n"
            f"Risk Management Notes: Monitor as usual for defined-risk management.\n"
        )

    cleaned_block = "\n".join(lines)
    dynamic_rules = f"{NOVA_RULES}\n\n💡 Current Max Loss Threshold: ${current_max_loss:.0f}"

    return (
        dynamic_rules
         + "\n---\n"
         + f"{summary_line}\n\n"
         + f"Ticker: {meta['ticker']} | Strategy: {meta['strategy']} | "
         + f"Expiry: {meta['expiry']} ({meta['dte']} DTE) | "
         + f"Contracts: {meta['contracts']} | Min POP: {meta['min_pop']}% | "
         + f"Spot: ${meta['spot']:.2f}\n\n"
         + cleaned_block
         + "\n\nSummarize the best opportunities succinctly and reference 'Credit Received' "
           "as total premium. Use the active max-loss threshold for qualification."
    )


# =========================================================
# 🧾 Render & Auto Summary
# =========================================================
if just_scanned and st.session_state.last_trades_df is not None:
    render_results(st.session_state.last_trades_df, min_pop)
elif st.session_state.last_trades_df is not None and st.session_state.last_meta is not None:
    if st.session_state.last_query == current_query:
        render_results(st.session_state.last_trades_df, st.session_state.last_meta["min_pop"])


# =========================================================
# 🧠 Nova’s Auto Take
# =========================================================
if st.session_state.last_trades_df is not None and st.session_state.last_meta is not None:
    st.subheader("🧠 Nova’s Auto Take")
    if just_scanned:
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": build_summary_prompt()}],
            )
            st.session_state.auto_summary = r.choices[0].message.content
        except Exception as e:
            st.session_state.auto_summary = f"(Nova API error: {e})"

    if st.session_state.auto_summary:
        st.text(st.session_state.auto_summary)


# =========================================================
# 💬 Nova Chat Interface
# =========================================================
st.markdown("---")
st.header("💬 Talk with Nova")

for msg in st.session_state.nova_chat:
    who = "You" if msg["role"] == "user" else "Nova"
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.text(f"{who}: {msg['content']}")

user_msg = st.chat_input("Type to Nova…")
if user_msg:
    st.session_state.nova_chat.append({"role": "user", "content": user_msg})
    if st.session_state.last_trades_records and st.session_state.last_meta:
        trades_context = st.session_state.last_trades_records[:20]
        meta = st.session_state.last_meta
        messages = [
            {"role": "system", "content": NOVA_RULES},
            {"role": "system", "content": f"Scan meta: {meta}"},
            {"role": "system", "content": f"Latest scanner results (up to 20 rows): {trades_context}"},
        ] + st.session_state.nova_chat
    else:
        messages = [{"role": "system", "content": "You are Nova. No scan results available. Ask Denny to run a scan first."}]
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        reply = r.choices[0].message.content
    except Exception as e:
        reply = f"(Nova API error: {e})"

    st.session_state.nova_chat.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.text(f"Nova: {reply}")














































































































































