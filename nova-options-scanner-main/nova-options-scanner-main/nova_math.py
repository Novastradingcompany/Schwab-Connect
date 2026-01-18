import math

# ----------------------------
# Credit calculation
# ----------------------------
def calc_credit(sell_mid, buy_mid, contracts=1):
    """
    Net credit collected per spread (in dollars).
    Assumes sell_mid and buy_mid are per-share option prices.
    Returns total dollar credit for all contracts.
    """
    credit_per_contract = max(sell_mid - buy_mid, 0)
    return credit_per_contract * 100 * contracts

# ----------------------------
# Max loss calculation
# ----------------------------
def calc_max_loss(width, credit, contracts=1):
    """
    Max loss = (width * 100 * contracts) - credit collected.
    Width is in dollars (strike difference).
    Credit is total dollar credit for all contracts.
    """
    max_loss_total = (width * 100 * contracts) - credit
    return max(max_loss_total, 0)

# ----------------------------
# Breakeven calculation
# ----------------------------
def calc_breakeven(short_strike, credit, opt_type, contracts=1):
    """
    Breakeven for simple vertical spread.
    Credit is total dollars; divide back by 100*contracts to get per-share.
    """
    credit_per_share = credit / (100 * contracts) if contracts > 0 else 0

    if opt_type == "put":
        return short_strike - credit_per_share
    elif opt_type == "call":
        return short_strike + credit_per_share
    return None

# ----------------------------
# POP (Probability of Profit) calculation
# ----------------------------
def calc_pop(short_strike, spot, width, credit, max_loss, opt_type, delta=None, contracts=1):
    """
    POP estimation hierarchy:
      1. Use delta if available.
      2. Else, use OTM distance vs. spot.
      3. Else, use crude credit/(credit+max_loss).
    Returns POP (%) rounded to 1 decimal.
    """

    # Convert credit/max_loss to per-share values for POP math
    credit_per_share = credit / (100 * contracts) if contracts > 0 else 0
    max_loss_per_share = max_loss / (100 * contracts) if contracts > 0 else 0

    # 1. Use delta (prob of ITM) → POP = 1 - prob(ITM)
    if delta is not None and isinstance(delta, (int, float)):
        if opt_type == "put":
            prob_itm = -delta  # put deltas are negative
        else:
            prob_itm = delta
        prob_itm = max(0.0, min(1.0, prob_itm))  # clamp to [0,1]
        return round((1 - prob_itm) * 100, 1)

    # 2. Distance-to-spot heuristic
    distance = abs(short_strike - spot)
    if spot > 0 and width > 0:
        est_pop = (distance / width) * 10  # crude scaling
        return round(max(0, min(est_pop, 95)), 1)

    # 3. Fallback: credit vs max loss
    denom = credit_per_share + max_loss_per_share
    if denom > 0:
        return round(100 * (credit_per_share / denom), 1)

    return 0.0







