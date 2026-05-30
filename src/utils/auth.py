"""
Kite Connect Authentication with automated TOTP login.
Handles daily re-authentication required by Zerodha.
"""

import os
import time
import logging
from datetime import datetime, timedelta

import pyotp
import yaml
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)


class KiteAuth:
    """Manages Kite Connect authentication with automated TOTP."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.api_key = os.getenv("KITE_API_KEY", self.config["broker"]["api_key"])
        self.api_secret = os.getenv("KITE_API_SECRET", self.config["broker"]["api_secret"])
        self.totp_secret = os.getenv("KITE_TOTP_SECRET", self.config["broker"]["totp_secret"])
        self.user_id = os.getenv("KITE_USER_ID", self.config["broker"]["user_id"])
        self.kite = KiteConnect(api_key=self.api_key)
        self._access_token = None
        self._token_expiry = None

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @property
    def access_token(self) -> str | None:
        return self._access_token

    @property
    def is_authenticated(self) -> bool:
        if not self._access_token:
            return False
        if self._token_expiry and datetime.now() > self._token_expiry:
            return False
        return True

    def get_login_url(self) -> str:
        """Get the Kite login URL for manual/automated login."""
        return self.kite.login_url()

    def generate_totp(self) -> str:
        """Generate current TOTP for automated login."""
        totp = pyotp.TOTP(self.totp_secret)
        return totp.now()

    def authenticate_with_request_token(self, request_token: str) -> dict:
        """
        Complete authentication using the request_token from login redirect.
        Returns user session data.
        """
        try:
            session = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.api_secret
            )
            self._access_token = session["access_token"]
            self.kite.set_access_token(self._access_token)
            # Kite tokens expire at 6 AM next day
            tomorrow = datetime.now().replace(hour=6, minute=0, second=0)
            if tomorrow < datetime.now():
                tomorrow += timedelta(days=1)
            self._token_expiry = tomorrow
            logger.info(f"Authentication successful for user: {self.user_id}")
            return session
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def authenticate_automated(self, request_token: str) -> dict:
        """
        Automated authentication flow.
        In production, this is triggered after the login redirect captures the request_token.
        
        For fully automated daily login, use the login server (see login_server.py).
        """
        return self.authenticate_with_request_token(request_token)

    def load_saved_token(self, token_file: str = ".kite_token") -> bool:
        """Load a previously saved access token if still valid."""
        try:
            if not os.path.exists(token_file):
                return False
            with open(token_file, "r") as f:
                data = yaml.safe_load(f)
            saved_time = datetime.fromisoformat(data["saved_at"])
            # Token valid until 6 AM next day
            expiry = saved_time.replace(hour=6, minute=0, second=0) + timedelta(days=1)
            if datetime.now() < expiry:
                self._access_token = data["access_token"]
                self._token_expiry = expiry
                self.kite.set_access_token(self._access_token)
                logger.info("Loaded saved access token successfully")
                return True
            else:
                logger.info("Saved token has expired")
                return False
        except Exception as e:
            logger.warning(f"Could not load saved token: {e}")
            return False

    def save_token(self, token_file: str = ".kite_token"):
        """Save current access token to file for reuse."""
        if self._access_token:
            data = {
                "access_token": self._access_token,
                "saved_at": datetime.now().isoformat(),
                "user_id": self.user_id,
            }
            with open(token_file, "w") as f:
                yaml.dump(data, f)
            logger.info("Access token saved to file")

    def get_kite(self) -> KiteConnect:
        """Get authenticated KiteConnect instance."""
        if not self.is_authenticated:
            raise RuntimeError(
                "Not authenticated. Call authenticate_with_request_token() first "
                "or load a saved token with load_saved_token()."
            )
        return self.kite

    def get_profile(self) -> dict:
        """Get user profile to verify authentication."""
        return self.kite.profile()

    def get_margins(self) -> dict:
        """Get account margins/funds available."""
        return self.kite.margins()
