import base64
import json
from pathlib import Path


def main():
    token_path = Path("token.json")
    if not token_path.exists():
        raise SystemExit("token.json not found. Run `python connect.py` first.")

    try:
        payload = json.loads(token_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"token.json is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("token.json root must be a JSON object.")

    token = payload.get("token", payload)
    if not isinstance(token, dict) or not token.get("refresh_token"):
        raise SystemExit("token.json does not contain a usable Schwab refresh token.")

    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    print(base64.b64encode(canonical.encode("utf-8")).decode("ascii"))


if __name__ == "__main__":
    main()
