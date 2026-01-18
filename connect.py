from schwab.auth import client_from_manual_flow, client_from_token_file
import json
import logging
import os
from dotenv import load_dotenv

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

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    try:
        api_key = os.getenv("SCHWAB_API_KEY")
        app_secret = os.getenv("SCHWAB_APP_SECRET")
        callback_url = os.getenv("SCHWAB_CALLBACK_URL")
        token_path = "token.json"

        if os.path.exists(token_path):
            client = client_from_token_file(token_path, api_key, app_secret, enforce_enums=False)
        else:
            client = client_from_manual_flow(api_key, app_secret, callback_url, token_path)

        response = client.get_accounts()
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()

        data = response.json()
        with open("accounts.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        logging.info("Wrote account payload to accounts.json")

        summary = _summarize_accounts(data)
        logging.info("Account summary:\n%s", json.dumps(summary, indent=2))

        print(json.dumps(data, indent=2))
    except Exception:
        logging.exception("Failed to fetch accounts")
        raise

if __name__ == "__main__":
    main()
