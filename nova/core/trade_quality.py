def _score_band(value, low, high, points):
    if value is None:
        return 0.0
    value = float(value)
    if value <= low:
        return 0.0
    if value >= high:
        return float(points)
    return ((value - low) / (high - low)) * float(points)


def evaluate_trade_quality(
    *,
    pop,
    return_on_risk,
    distance_pct,
    credit_width_pct,
    dte,
    natural_credit=None,
    mid_credit=None,
):
    score = 0.0
    notes = []

    score += _score_band(pop, 60, 85, 25)
    score += _score_band(return_on_risk, 8, 25, 30)
    score += _score_band(distance_pct, 1.5, 8, 20)
    score += _score_band(credit_width_pct, 8, 25, 15)

    dte_val = float(dte or 0)
    if 14 <= dte_val <= 45:
        score += 10
    elif 7 <= dte_val <= 60:
        score += 5
        notes.append("DTE workable, not ideal")
    else:
        notes.append("DTE outside preferred range")

    if return_on_risk < 10:
        notes.append("Reward too small for risk")
    elif return_on_risk < 15:
        notes.append("Reward/risk is marginal")
    else:
        notes.append("Reward/risk acceptable")

    if credit_width_pct < 10:
        notes.append("Credit is thin for spread width")
    if distance_pct < 3:
        notes.append("Short strike is close to spot")
    if pop < 70:
        notes.append("POP below signal threshold")

    if natural_credit is not None and mid_credit:
        fill_quality = float(natural_credit) / float(mid_credit) if float(mid_credit) > 0 else 0.0
        if fill_quality < 0.55:
            score -= 10
            notes.append("Bid/ask fill quality is weak")
        elif fill_quality >= 0.75:
            score += 5

    score = round(max(0.0, min(score, 100.0)), 1)

    if (
        score >= 80
        and return_on_risk >= 15
        and pop >= 70
        and distance_pct >= 3
        and credit_width_pct >= 10
    ):
        signal = "TAKE"
    elif score >= 55 and return_on_risk >= 10 and pop >= 65:
        signal = "WATCH"
    else:
        signal = "PASS"

    return {
        "Signal": signal,
        "Quality Score": score,
        "Return/Risk %": round(float(return_on_risk), 1),
        "Credit/Width %": round(float(credit_width_pct), 1),
        "Quality Notes": "; ".join(notes[:3]),
    }
