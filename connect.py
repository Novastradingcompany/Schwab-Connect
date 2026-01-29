import json
import logging
import os
from dotenv import load_dotenv
from schwab.auth import client_from_manual_flow, client_from_token_file

def _summarize_accounts(payload):
    summaries = []
    for entry in payload:
        account = entry.get("securitiesAccount", {})
        balances = account.get("currentBalances", {})
        summaries.append(
            {
                "accountNumber": account.get("accountNumber"),
                "type": account.get("type"),
                "liquidationValue": balances.get("liquidationValue"),
                "cashBalance": balances.get("cashBalance"),
                "equity": balances.get("equity"),
            }
        )
    return summaries

def _write_accounts(client):
    response = client.get_accounts()
    response.raise_for_status()
    data = response.json()
    with open("accounts.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    logging.info("Wrote account payload to accounts.json")
    summary = _summarize_accounts(data)
    logging.info("Account summary:\n%s", json.dumps(summary, indent=2))
    print(json.dumps(data, indent=2))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    api_key = os.getenv("SCHWAB_API_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL")
    token_path = "token.json"

    if not api_key or not app_secret or not callback_url:
        raise RuntimeError("Missing SCHWAB_API_KEY, SCHWAB_APP_SECRET, or SCHWAB_CALLBACK_URL")

    try:
        if os.path.exists(token_path):
            client = client_from_token_file(
                token_path,
                api_key,
                app_secret,
                enforce_enums=False,
            )
            _write_accounts(client)
            return
    except Exception:
        logging.warning("Token refresh failed. Starting manual auth flow.")

    client = client_from_manual_flow(
        api_key,
        app_secret,
        callback_url,
        token_path,
        enforce_enums=False,
    )
    _write_accounts(client)

if __name__ == "__main__":
    main()
