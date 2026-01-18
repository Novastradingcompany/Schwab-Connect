def get_max_loss_threshold(cash):
    """
    Dynamically determine the max loss allowed per trade based on available cash.
    """
    if cash < 300:
        return 0
    if cash < 500:
        return 100
    if cash < 700:
        return 150
    if cash < 999:
        return 200
    if cash < 1999:
        return 250
    if cash < 2999:
        return 350
    if cash < 3999:
        return 450
    if cash < 4999:
        return 550
    return 650


NOVA_RULES = (
    "Capital-Based Scaling Rules:\n"
    "- If cash < $300: no trades.\n"
    "- If cash $300-$499: max loss $100.\n"
    "- If cash $500-$699: max loss $150.\n"
    "- If cash $700-$999: max loss $200.\n"
    "- If cash $1,000-$1,999: max loss up to $250.\n"
    "- If cash $2,000-$2,999: max loss up to $350.\n"
    "- If cash $3,000-$3,999: max loss up to $450.\n"
    "- If cash $4,000-$4,999: max loss up to $550.\n"
    "- If cash $5,000+: max loss up to $650.\n"
)
