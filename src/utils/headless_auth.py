"""
Headless Kite Connect Authentication.
Uses Playwright (headless Chromium) to auto-login to Zerodha without a visible browser.
Designed for cloud deployment where no GUI is available.

Flow:
1. Navigate to Kite login
2. Enter user ID + password
3. Auto-fill TOTP
4. Capture redirect with request_token
5. Generate access token via API
6. Save token to persistent storage

Requires: playwright install chromium
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyotp
import yaml

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

STATE_DIR = os.getenv("STATE_DIR", ".")
TOKEN_PATH = Path(STATE_DIR) / ".kite_token"


def _load_secrets_from_keyvault() -> dict:
    """Load secrets from Azure Key Vault using Managed Identity."""
    vault_url = os.getenv("AZURE_KEY_VAULT_URL", "")
    if not vault_url:
        return {}

    try:
        from azure.identity import ManagedIdentityCredential, DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        # Use ManagedIdentity in cloud, DefaultAzure for local dev
        if os.getenv("AZURE_DEPLOYMENT"):
            credential = ManagedIdentityCredential()
        else:
            credential = DefaultAzureCredential()

        client = SecretClient(vault_url=vault_url, credential=credential)
        secrets = {}
        secret_names = {
            "KITE-API-KEY": "KITE_API_KEY",
            "KITE-API-SECRET": "KITE_API_SECRET",
            "KITE-USER-ID": "KITE_USER_ID",
            "KITE-PASSWORD": "KITE_PASSWORD",
            "KITE-TOTP-SECRET": "KITE_TOTP_SECRET",
            "OPENAI-API-KEY": "OPENAI_API_KEY",
            "TELEGRAM-BOT-TOKEN": "TELEGRAM_BOT_TOKEN",
            "TELEGRAM-CHAT-ID": "TELEGRAM_CHAT_ID",
        }
        for kv_name, env_name in secret_names.items():
            try:
                secret = client.get_secret(kv_name)
                secrets[env_name] = secret.value
            except Exception:
                pass

        logger.info(f"Loaded {len(secrets)} secrets from Key Vault")
        return secrets
    except ImportError:
        logger.warning("azure-identity/azure-keyvault-secrets not installed")
        return {}
    except Exception as e:
        logger.warning(f"Key Vault access failed: {e}")
        return {}


class HeadlessAuth:
    """
    Automated Kite Connect login using Playwright headless browser.
    No human interaction needed — perfect for cloud/cron deployment.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)

        # Try Azure Key Vault first, then fall back to env vars
        kv_secrets = {}
        if os.getenv("AZURE_KEY_VAULT_URL"):
            kv_secrets = _load_secrets_from_keyvault()

        self.api_key = kv_secrets.get("KITE_API_KEY") or os.getenv("KITE_API_KEY", "")
        self.api_secret = kv_secrets.get("KITE_API_SECRET") or os.getenv("KITE_API_SECRET", "")
        self.user_id = kv_secrets.get("KITE_USER_ID") or os.getenv("KITE_USER_ID", "")
        self.password = kv_secrets.get("KITE_PASSWORD") or os.getenv("KITE_PASSWORD", "")
        self.totp_secret = kv_secrets.get("KITE_TOTP_SECRET") or os.getenv("KITE_TOTP_SECRET", "")

        # Also inject into env so other modules (notifier, analyst) can read them
        for key in ["OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
            val = kv_secrets.get(key)
            if val and not os.getenv(key):
                os.environ[key] = val

        if not all([self.api_key, self.api_secret, self.user_id, self.password, self.totp_secret]):
            raise ValueError(
                "Missing credentials. Set environment variables or configure Azure Key Vault: "
                "KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET"
            )

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def login(self, max_retries: int = 3) -> dict:
        """
        Perform headless login and return access token.
        
        Returns:
            dict with keys: access_token, user_id, login_time
            
        Raises:
            RuntimeError on login failure after retries
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Headless login attempt {attempt}/{max_retries}")
                request_token = self._get_request_token()
                if request_token:
                    token_data = self._generate_access_token(request_token)
                    self._save_token(token_data)
                    logger.info(f"Login successful! Token valid until 6 AM tomorrow.")
                    return token_data
            except Exception as e:
                logger.error(f"Login attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(35)  # Wait for new TOTP window (30s cycle)

        raise RuntimeError(f"Headless login failed after {max_retries} attempts")

    def _start_callback_server(self):
        """Start a local HTTP server to receive the Kite callback redirect."""
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler

        captured = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                if "request_token" in params:
                    captured["request_token"] = params["request_token"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body>Login successful. You can close this window.</body></html>")

            def log_message(self, format, *args):
                pass  # Suppress server logs

        server = HTTPServer(("127.0.0.1", 5000), CallbackHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, captured

    def _get_request_token(self) -> str:
        """Use Playwright to navigate login flow and capture request_token."""
        from playwright.sync_api import sync_playwright

        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"

        # Start local callback server to receive the redirect
        callback_server, callback_captured = self._start_callback_server()
        logger.info("Callback server started on 127.0.0.1:5000")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = context.new_page()

                try:
                    # Step 1: Navigate to login
                    page.goto(login_url, wait_until="networkidle", timeout=30000)
                    logger.info("Login page loaded")

                    # Step 2: Enter User ID and password
                    page.fill("input[type='text']", self.user_id)
                    page.fill("input[type='password']", self.password)
                    page.click("button[type='submit']")
                    logger.info("Credentials submitted")

                    # Step 3: Wait for TOTP page
                    time.sleep(3)
                    totp_input = page.wait_for_selector(
                        "input[label='External TOTP'], input[type='text'], input[type='number']",
                        timeout=15000
                    )

                    # Step 4: Generate and enter TOTP
                    totp = pyotp.TOTP(self.totp_secret)
                    otp_code = totp.now()
                    logger.info(f"TOTP generated: {otp_code[:2]}****")
                    totp_input.fill("")
                    totp_input.type(otp_code, delay=50)
                    logger.info("TOTP entered")
                    time.sleep(2)
                    submit_btn = page.query_selector("button[type='submit']")
                    if submit_btn and submit_btn.is_visible():
                        submit_btn.click()
                        logger.info("TOTP submit clicked")

                    # Step 5: Handle authorize page
                    time.sleep(3)
                    logger.info(f"Post-TOTP URL: {page.url[:120]}")

                    # Check if callback server already got the token
                    if callback_captured.get("request_token"):
                        logger.info("Callback server captured request_token!")
                        return callback_captured["request_token"]

                    # Click authorize button if on consent page
                    if "request_token" not in page.url:
                        try:
                            btn = page.query_selector("button[type='submit']")
                            if btn and btn.is_visible():
                                btn.click()
                                logger.info("Clicked authorize button")
                        except Exception as e:
                            logger.warning(f"Authorize click failed: {e}")

                    # Wait for callback server to receive the token
                    for _ in range(30):
                        if callback_captured.get("request_token"):
                            break
                        time.sleep(0.5)

                    if callback_captured.get("request_token"):
                        logger.info("Login redirect received by callback server")
                        return callback_captured["request_token"]

                    # Fallback: check page URL
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(page.url)
                    params = parse_qs(parsed.query)
                    token = params.get("request_token", [None])[0]
                    if token:
                        return token

                    raise RuntimeError(f"No request_token received. Final URL: {page.url[:150]}")

                except Exception as e:
                    screenshot_path = "logs/login_error.png"
                    os.makedirs("logs", exist_ok=True)
                    page.screenshot(path=screenshot_path)
                    logger.error(f"Login error (screenshot: {screenshot_path}): {e}")
                    raise
                finally:
                    browser.close()
        finally:
            callback_server.shutdown()

    def _generate_access_token(self, request_token: str) -> dict:
        """Exchange request_token for access_token via Kite API."""
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=self.api_key)
        data = kite.generate_session(request_token, api_secret=self.api_secret)

        return {
            "access_token": data["access_token"],
            "user_id": data.get("user_id", self.user_id),
            "login_time": datetime.now(IST).isoformat(),
            "api_key": self.api_key,
        }

    def _save_token(self, token_data: dict):
        """Save token to file (mounted Azure Files in cloud)."""
        import json
        TOKEN_PATH.write_text(json.dumps(token_data))
        logger.info(f"Token saved to {TOKEN_PATH}")

    def load_saved_token(self) -> dict:
        """Load previously saved token if still valid."""
        import json

        if not TOKEN_PATH.exists():
            return None

        try:
            data = json.loads(TOKEN_PATH.read_text())
            login_time = datetime.fromisoformat(data["login_time"])

            # Tokens expire at 6 AM IST next day
            now = datetime.now(IST)
            login_date = login_time.date() if login_time.tzinfo else login_time.date()
            if login_date == now.date() and now.hour < 6:
                return data
            if login_date == now.date() and now.hour >= 6:
                return data  # Same day, token valid until next 6 AM
            if (now.date() - login_date).days == 0:
                return data

            return None  # Expired (different day)
        except Exception:
            return None

    def ensure_authenticated(self) -> dict:
        """
        Ensure we have a valid token. Load from file or login fresh.
        This is the main entry point for the orchestrator.
        """
        saved = self.load_saved_token()
        if saved:
            logger.info("Using saved token (still valid)")
            return saved

        logger.info("Token expired or not found. Performing headless login...")
        return self.login()


def main():
    """CLI entry point for manual headless login."""
    logging.basicConfig(level=logging.INFO)
    
    print("Zero Agent — Headless Kite Login")
    print("-" * 40)

    try:
        auth = HeadlessAuth()
        token = auth.login()
        print(f"✅ Login successful!")
        print(f"   User: {token['user_id']}")
        print(f"   Time: {token['login_time']}")
        print(f"   Token saved to: {TOKEN_PATH}")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
