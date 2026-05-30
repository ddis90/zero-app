"""
Lightweight Flask server to handle Kite Connect OAuth redirect.
Run this on login day to capture the request_token automatically.

Usage:
    python -m src.utils.login_server
    
Then open the login URL in browser. After login, the server captures
the token and saves it for the trading session.
"""

import os
import sys
import logging
import webbrowser

from flask import Flask, request, jsonify
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.auth import KiteAuth

app = Flask(__name__)
logger = logging.getLogger(__name__)

auth = KiteAuth()


@app.route("/callback")
def callback():
    """Handle Kite Connect redirect with request_token."""
    request_token = request.args.get("request_token")
    status = request.args.get("status")

    if status != "success" or not request_token:
        return jsonify({"error": "Login failed or cancelled"}), 400

    try:
        session = auth.authenticate_with_request_token(request_token)
        auth.save_token()
        profile = auth.get_profile()
        margins = auth.get_margins()

        return jsonify({
            "status": "success",
            "user": profile.get("user_name", ""),
            "user_id": profile.get("user_id", ""),
            "equity_margin": margins.get("equity", {}).get("available", {}).get("live_balance", 0),
            "message": "Authentication successful! You can close this window. Trading agent will use saved token."
        })
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "authenticated": auth.is_authenticated})


def main():
    """Start login server and open browser for authentication."""
    print("=" * 60)
    print("  Kite Connect Login Server")
    print("=" * 60)
    print(f"\n  Login URL: {auth.get_login_url()}")
    print(f"  Redirect: http://127.0.0.1:5000/callback")
    print("\n  Opening browser for login...")
    print("  After login, the token will be saved automatically.")
    print("=" * 60)

    webbrowser.open(auth.get_login_url())
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
