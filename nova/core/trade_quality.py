def _score_band(value, low, high, points):
    if value is None:
        return 0.0
    value = float(value)
    if value <= low:
        return 0.0
    if value >= high:
        return float(points)
    return ((value - low) / (high - low)) * float(points)


def _gap(value, target):
    return max(float(target) - float(value or 0), 0.0)


def _fmt_pct(value):
    return f"{float(value):.1f}%"


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
    pop = float(pop or 0)
    return_on_risk = float(return_on_risk or 0)
    distance_pct = float(distance_pct or 0)
    credit_width_pct = float(credit_width_pct or 0)
    score = 0.0
    blockers = []
    positives = []

    score += _score_band(pop, 60, 85, 25)
    score += _score_band(return_on_risk, 8, 25, 30)
    score += _score_band(distance_pct, 1.5, 8, 20)
    score += _score_band(credit_width_pct, 8, 25, 15)

    dte_val = float(dte or 0)
    if 14 <= dte_val <= 45:
        score += 10
        positives.append("DTE in preferred range")
    elif 7 <= dte_val <= 60:
        score += 5
        blockers.append("DTE workable, not ideal")
    else:
        blockers.append("DTE outside preferred range")

    if return_on_risk < 15:
        gap = _gap(return_on_risk, 15)
        blockers.append(f"Needs +{_fmt_pct(gap)} return/risk")
    else:
        positives.append("Reward/risk acceptable")

    if credit_width_pct < 10:
        gap = _gap(credit_width_pct, 10)
        blockers.append(f"Needs +{_fmt_pct(gap)} credit/width")
    elif credit_width_pct >= 15:
        positives.append("Credit is solid for width")

    if distance_pct < 3:
        gap = _gap(distance_pct, 3)
        blockers.append(f"Needs +{_fmt_pct(gap)} more distance")
    elif distance_pct >= 5:
        positives.append("Strike has good cushion")

    if pop < 70:
        gap = _gap(pop, 70)
        blockers.append(f"Needs +{_fmt_pct(gap)} POP")
    elif pop >= 80:
        positives.append("POP is strong")

    if natural_credit is not None and mid_credit:
        fill_quality = float(natural_credit) / float(mid_credit) if float(mid_credit) > 0 else 0.0
        if fill_quality < 0.55:
            score -= 10
            blockers.append("Bid/ask fill quality is weak")
        elif fill_quality >= 0.75:
            score += 5
            positives.append("Bid/ask fill quality is acceptable")

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

    if blockers:
        notes = blockers[:3]
    elif positives:
        notes = positives[:3]
    else:
        notes = ["No major quality edge"]

    return {
        "Signal": signal,
        "Quality Score": score,
        "Return/Risk %": round(float(return_on_risk), 1),
        "Credit/Width %": round(float(credit_width_pct), 1),
        "Quality Notes": "; ".join(notes[:3]),
    }
