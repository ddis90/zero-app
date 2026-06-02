"""
inject_token.py — Manual Kite auth token injection (CAPTCHA fallback).

Run this locally on each trading morning when headless auth is blocked by CAPTCHA:

    python scripts/inject_token.py

Flow:
  1. Opens the Kite Connect OAuth login URL in your browser
  2. You log in normally (user/pass + TOTP)
  3. Kite redirects to localhost:5000/callback with request_token
  4. Script exchanges it for an access_token via Kite API
  5. Access token is saved to Azure Key Vault as KITE-ACCESS-TOKEN
  6. The cloud trading agent (ca-zero-agent) picks it up at 08:45 AM IST

Requirements:
  - .env file in project root with KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, etc.
  - AZURE_KEY_VAULT_URL set (from .env or environment)
  - az login done (or AZURE_CLIENT_ID/AZURE_TENANT_ID/AZURE_CLIENT_SECRET for service principal)
  - Kite Connect redirect URL registered as: http://127.0.0.1:5000/callback
"""

import json
import logging
import os
import sys
import threading
import webbrowser
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional

from src.utils.auth import KiteAuth

logging.basicConfig(level=logging.WARNING, format="%(message)s")

IST = ZoneInfo("Asia/Kolkata")

app = Flask(__name__)
app.logger.disabled = True

_token_event = threading.Event()
_result: dict = {}


@app.route("/callback")
def callback():
    request_token = request.args.get("request_token")
    status = request.args.get("status")

    if status != "success" or not request_token:
        _result["error"] = "Login failed or cancelled"
        _token_event.set()
        return "<h2>Login failed or cancelled. Close this tab and try again.</h2>", 400

    try:
        auth = KiteAuth()
        session = auth.authenticate_with_request_token(request_token)
        _result["token_data"] = {
            "access_token": session["access_token"],
            "api_key": auth.api_key,
            "user_id": auth.user_id,
            "login_time": datetime.now(IST).isoformat(),
        }
        _token_event.set()
        return (
            "<h2 style='color:green'>&#10003; Login successful!</h2>"
            "<p>Access token saved to Azure Key Vault. You can close this tab.<br>"
            "The trading agent will pick it up at 08:45 AM IST.</p>"
        )
    except Exception as e:
        _result["error"] = str(e)
        _token_event.set()
        return f"<h2 style='color:red'>Error: {e}</h2>", 500


def _save_to_keyvault(token_data: dict) -> bool:
    """Save the access token dict to Key Vault as KITE-ACCESS-TOKEN."""
    vault_url = os.getenv("AZURE_KEY_VAULT_URL", "")
    if not vault_url:
        print("  ERROR: AZURE_KEY_VAULT_URL not set in .env")
        return False
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())
        client.set_secret("KITE-ACCESS-TOKEN", json.dumps(token_data))
        print(f"  Token saved to Key Vault: {vault_url}")
        return True
    except Exception as e:
        print(f"  Key Vault save failed: {e}")
        return False


def main():
    auth = KiteAuth()
    login_url = auth.get_login_url()

    print("=" * 60)
    print("  Zero Agent — Manual Token Injection")
    print(f"  Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("=" * 60)
    print(f"\n  Redirect URL (must be registered in Kite Console):")
    print(f"    http://127.0.0.1:5000/callback")
    print(f"\n  Opening browser for Kite login...")
    print("  Log in normally — the token will be captured automatically.\n")

    # Start Flask callback server in a daemon thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()

    webbrowser.open(login_url)
    print("  Waiting for browser login... (timeout: 5 min)")

    if not _token_event.wait(timeout=300):
        print("\n  Timeout — no login detected in 5 minutes. Run again.")
        sys.exit(1)

    if "error" in _result:
        print(f"\n  Login failed: {_result['error']}")
        sys.exit(1)

    token_data = _result["token_data"]
    print(f"\n  Authenticated as: {token_data['user_id']}")
    print(f"  Access token: {token_data['access_token'][:8]}...")

    if _save_to_keyvault(token_data):
        print("\n  Done. The cloud trading agent will use this token at 08:45 AM IST.")
        print("  (If auth fails again, you'll get a Telegram alert to re-run this script.)\n")
    else:
        print("\n  Key Vault save failed. As a fallback, run this Azure CLI command:")
        print(f"    az keyvault secret set \\")
        print(f"      --vault-name kv-zero-trading \\")
        print(f"      --name KITE-ACCESS-TOKEN \\")
        print(f"      --value '{json.dumps(token_data)}'")
        print(f"\n  Then the 09:05 AM retry will pick it up automatically.")
        print()


if __name__ == "__main__":
    main()
