import json
import logging
import os
import time
import urllib.parse
from dotenv import load_dotenv
from schwab.auth import client_from_token_file, OAuth2Client, __fetch_and_register_token_from_redirect

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

    oauth = OAuth2Client(api_key, redirect_uri=callback_url)
    authorization_url, state = oauth.create_authorization_url(
        "https://api.schwabapi.com/v1/oauth/authorize"
    )

    print("\n**************************************************************\n")
    print("Manual login and token creation flow.")
    print("1) Open this URL in your browser:\n")
    print("   " + authorization_url + "\n")
    print("2) Log in, approve access, then you will be redirected.")
    print("3) Paste ONLY the redirected URL OR just the code parameter below.")
    print("   Never share the full URL or code publicly.\n")
    print("**************************************************************\n")

    user_input = input("Redirect URL or code> ").strip()
    if "http://" in user_input or "https://" in user_input:
        redirected_url = user_input
    else:
        # Accept either a raw code or a query string like code=...&state=...
        parsed = urllib.parse.parse_qs(user_input, keep_blank_values=True)
        if "code" in parsed:
            code = parsed.get("code", [user_input])[0]
        else:
            # If user pasted "CODE&session=...&state=..." without "code=",
            # extract everything before the first "&".
            code = user_input.split("&", 1)[0]
        redirect_query = urllib.parse.urlencode({"code": code, "state": state})
        redirected_url = f"{callback_url}?{redirect_query}"

    # Best-effort check for expired codes to avoid unnecessary 400s
    try:
        parsed_url = urllib.parse.urlparse(redirected_url)
        params = urllib.parse.parse_qs(parsed_url.query)
        code = params.get("code", [None])[0]
        if code:
            # Schwab codes are short-lived. Warn if user input took too long.
            # We can't read actual expiry, so we just prompt for speed.
            print("Note: authorization codes expire quickly. If this fails, retry immediately.")
    except Exception:
        pass

    client = __fetch_and_register_token_from_redirect(
        oauth,
        redirected_url,
        api_key,
        app_secret,
        token_path,
        token_write_func=None,
        asyncio=False,
        enforce_enums=False,
    )
    _write_accounts(client)

if __name__ == "__main__":
    main()
