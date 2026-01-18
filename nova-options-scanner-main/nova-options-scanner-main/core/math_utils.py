# =======================================
# core/math_utils.py — Nova POP + BS Delta + TOS-safe Credit
# =======================================
import math

# ----------------------------
# Credit calculation
# ----------------------------
def calc_credit(sell_mid, buy_mid, contracts=1):
    """Net credit collected per spread (in dollars)."""
    credit_per_contract = max(sell_mid - buy_mid, 0) * 100
    return round(credit_per_contract * contracts, 2)


# ----------------------------
# Max loss calculation
# ----------------------------
def calc_max_loss(width, credit, contracts=1):
    """
    Max loss = (width * 100 * contracts) - credit collected,
    capped so credit cannot exceed 99% of total width value.
    Prevents negative or unrealistic losses.
    """
    max_credit_allowed = width * 100 * contracts * 0.99
    credit = min(credit, max_credit_allowed)
    max_loss_total = (width * 100 * contracts) - credit
    return round(max(max_loss_total, 0), 2)


# ----------------------------
# Breakeven calculation
# ----------------------------
def calc_breakeven(short_strike, credit, opt_type, contracts=1):
    """Breakeven for a vertical spread."""
    credit_per_share = credit / (100 * contracts) if contracts > 0 else 0
    if opt_type == "put":
        return short_strike - credit_per_share
    elif opt_type == "call":
        return short_strike + credit_per_share
    return None


# ----------------------------
# Black–Scholes Delta
# ----------------------------
def bs_delta(S, K, T, r, sigma, option_type="call"):
    """
    Compute Black–Scholes delta for a call or put.
    S = Spot price, K = Strike, T = time (years), r = risk-free rate, sigma = IV (decimal).
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        N_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        if option_type == "call":
            return round(N_d1, 4)
        elif option_type == "put":
            return round(N_d1 - 1, 4)
        return None
    except Exception:
        return None


# ----------------------------
# Nova POP Calculation (Black–Scholes Probability)
# ----------------------------
def calc_pop(short_strike, spot, width, credit, max_loss,
             opt_type, delta=None, contracts=1,
             iv=None, T=None, r=0.02):
    """
    Probability of Profit (POP) using Nova's Black–Scholes approach:
      POP = N(d2)
      where d2 = [ln(S/K) + (r - 0.5σ²)T] / (σ√T)
      Falls back to delta or credit/loss ratio if IV or T are missing.
    """
    # 1️⃣ If we can compute d2 from IV, use true Nova POP
    if iv and T and iv > 0 and T > 0 and spot > 0 and short_strike > 0:
        try:
            d2 = (math.log(spot / short_strike) + (r - 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
            N_d2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))

            if opt_type == "put":
                pop = N_d2 * 100
            elif opt_type == "call":
                pop = (1 - N_d2) * 100
            else:
                pop = N_d2 * 100

            return round(min(max(pop, 0), 99.9), 1)
        except Exception:
            pass

    # 2️⃣ Delta-based fallback
    if delta is not None and isinstance(delta, (int, float)):
        prob_itm = abs(delta)
        prob_otm = 1 - prob_itm
        return round(prob_otm * 100, 1)

    # 3️⃣ Credit/Loss ratio fallback
    credit_per_share = credit / (100 * contracts) if contracts > 0 else 0
    max_loss_per_share = max_loss / (100 * contracts) if contracts > 0 else 0
    denom = credit_per_share + max_loss_per_share
    if denom > 0:
        pop = (credit_per_share / denom) * 100
        return round(min(max(pop, 0), 95), 1)

    # Default fallback if all else fails
    return 50.0










