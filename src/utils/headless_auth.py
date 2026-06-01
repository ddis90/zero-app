"""
Headless Kite Connect Authentication.
Uses Kite's internal HTTP APIs to auto-login to Zerodha without a visible browser.
Designed for cloud deployment where no GUI is available.

Flow:
1. POST /api/login with credentials
2. POST /api/twofa with TOTP
3. GET connect/login URL with session cookies → follow redirect for request_token
4. Generate access token via Kite Connect API
5. Save token to persistent storage
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
                # Don't retry if CAPTCHA locked - it will only get worse
                if "CAPTCHA" in str(e).upper():
                    logger.error("CAPTCHA detected - stopping retries to avoid escalation")
                    break
                if attempt < max_retries:
                    time.sleep(35)  # Wait for new TOTP window (30s cycle)

        raise RuntimeError(f"Headless login failed after {max_retries} attempts")

    def _get_request_token(self) -> str:
        """Use Kite's internal HTTP APIs to login and get request_token.
        
        Flow:
        1. POST /api/login → get request_id
        2. POST /api/twofa → complete 2FA with TOTP
        3. GET connect/login URL with session cookies → follow redirect to get request_token
        """
        import requests
        from urllib.parse import urlparse, parse_qs

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://kite.zerodha.com",
            "Referer": "https://kite.zerodha.com/connect/login?v=3&api_key=" + self.api_key,
            "X-Kite-Version": "3.0.0",
        })

        # Step 1: Visit login page first to get session cookies
        logger.info("Step 1: Getting session cookies...")
        session.get(f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}")

        # Step 2: Login with credentials
        logger.info("Step 2: Logging in with credentials...")
        login_resp = session.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": self.user_id, "password": self.password},
        )
        login_data = login_resp.json()
        logger.info(f"Login response status: {login_resp.status_code}, keys: {list(login_data.get('data', {}).keys())}")

        if login_data.get("status") != "success":
            msg = login_data.get("message", str(login_data))
            if "captcha" in str(login_data.get("data", {})).lower() or "captcha" in msg.lower():
                raise RuntimeError(f"CAPTCHA required - account temporarily locked. Wait and retry later.")
            raise RuntimeError(f"Login failed: {msg}")

        request_id = login_data["data"]["request_id"]
        logger.info(f"Got request_id: {request_id[:8]}...")

        # Step 3: Submit TOTP (2FA)
        totp = pyotp.TOTP(self.totp_secret)
        otp_code = totp.now()
        logger.info(f"Step 3: Submitting TOTP: {otp_code[:2]}****")

        twofa_resp = session.post(
            "https://kite.zerodha.com/api/twofa",
            data={
                "user_id": self.user_id,
                "request_id": request_id,
                "twofa_value": otp_code,
                "twofa_type": "totp",
            },
        )
        twofa_data = twofa_resp.json()
        logger.info(f"2FA response status: {twofa_resp.status_code}, status: {twofa_data.get('status')}")

        if twofa_data.get("status") != "success":
            raise RuntimeError(f"2FA failed: {twofa_data.get('message', twofa_data)}")

        # Step 4: Now visit the connect/login URL with authenticated session
        # This should redirect to our callback URL with request_token
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"
        logger.info("Step 4: Visiting connect/login to get authorize redirect...")

        auth_resp = session.get(login_url, allow_redirects=False)
        logger.info(f"Auth response: status={auth_resp.status_code}, location={auth_resp.headers.get('Location', 'none')[:150]}")

        # Follow redirects manually to catch the one with request_token
        max_redirects = 10
        for i in range(max_redirects):
            if auth_resp.status_code in (301, 302, 303, 307, 308):
                redirect_loc = auth_resp.headers.get("Location", "")
                logger.info(f"Redirect {i+1}: {redirect_loc[:200]}")

                if "request_token" in redirect_loc:
                    parsed = urlparse(redirect_loc)
                    params = parse_qs(parsed.query)
                    token = params.get("request_token", [None])[0]
                    if token:
                        logger.info("Got request_token from redirect!")
                        return token

                # Follow the redirect
                auth_resp = session.get(redirect_loc, allow_redirects=False)
            else:
                break

        # Check final URL in case it landed on callback with token
        final_url = auth_resp.url if hasattr(auth_resp, 'url') else ""
        if "request_token" in final_url:
            parsed = urlparse(final_url)
            params = parse_qs(parsed.query)
            token = params.get("request_token", [None])[0]
            if token:
                return token

        # If we're on an authorize page, try POSTing to authorize endpoint
        if auth_resp.status_code == 200 and "authorize" in auth_resp.text.lower():
            logger.info("On authorize page, attempting to POST authorize...")
            # Extract any hidden form fields
            import re
            action_match = re.search(r'action="([^"]+)"', auth_resp.text)
            authorize_url = action_match.group(1) if action_match else "https://kite.zerodha.com/connect/authorize"

            authorize_resp = session.post(
                authorize_url if authorize_url.startswith("http") else f"https://kite.zerodha.com{authorize_url}",
                data={"api_key": self.api_key},
                allow_redirects=False,
            )
            logger.info(f"Authorize POST: status={authorize_resp.status_code}, location={authorize_resp.headers.get('Location', 'none')[:150]}")

            if authorize_resp.status_code in (301, 302, 303, 307, 308):
                redirect_loc = authorize_resp.headers.get("Location", "")
                if "request_token" in redirect_loc:
                    parsed = urlparse(redirect_loc)
                    params = parse_qs(parsed.query)
                    token = params.get("request_token", [None])[0]
                    if token:
                        logger.info("Got request_token from authorize POST redirect!")
                        return token

        logger.error(f"Failed to get request_token. Final status: {auth_resp.status_code}, body: {auth_resp.text[:300]}")
        raise RuntimeError(f"No request_token received after HTTP login flow")

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
