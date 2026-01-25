import json
import logging
import os
from dotenv import load_dotenv
from schwab.auth import easy_client

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

    api_key = os.getenv("SCHWAB_API_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL")  # should be https://schwab-connect.onrender.com

    token_path = "token.json"

    try:
        # easy_client handles:
        # - local HTTPS callback server
        # - OAuth login
        # - token refresh
        # - token storage
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=callback_url,
            token_path=token_path,
            enforce_enums=False
        )

        response = client.get_accounts()
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

