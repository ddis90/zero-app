"""
Telegram notification bot for trade alerts and daily summaries.
"""

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml
import requests

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class TelegramNotifier:
    """Sends trade alerts and summaries via Telegram bot."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        config = self._load_config(config_path)
        notif_config = config["notifications"]["telegram"]
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", notif_config.get("bot_token", ""))
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", notif_config.get("chat_id", ""))
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            logger.warning("Telegram notifications disabled (missing bot_token or chat_id)")

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def send_message(self, message: str, parse_mode: str = "HTML"):
        """Send a message via Telegram."""
        if not self.enabled:
            logger.info(f"[TELEGRAM DISABLED] {message}")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Telegram send failed: {response.text}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def notify_trade_executed(self, trade_details: dict):
        """Notify when a trade is executed."""
        msg = (
            f"🔔 <b>Trade Executed</b>\n"
            f"{'🟢 BUY' if trade_details.get('type') == 'BUY' else '🔴 SELL'} "
            f"<b>{trade_details.get('symbol', 'N/A')}</b>\n"
            f"Price: ₹{trade_details.get('price', 0):.1f}\n"
            f"Qty: {trade_details.get('quantity', 0)}\n"
            f"SL: ₹{trade_details.get('stop_loss', 0):.1f}\n"
            f"Target: ₹{trade_details.get('target', 0):.1f}\n"
            f"Strategy: {trade_details.get('strategy', 'N/A')}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        self.send_message(msg)

    def notify_stop_loss_hit(self, symbol: str, loss: float):
        """Notify when stop loss is triggered."""
        msg = (
            f"🛑 <b>Stop Loss Hit</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Loss: ₹{abs(loss):.0f}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        self.send_message(msg)

    def notify_target_hit(self, symbol: str, profit: float):
        """Notify when target is achieved."""
        msg = (
            f"🎯 <b>Target Hit!</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Profit: ₹{profit:.0f}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        self.send_message(msg)

    def notify_daily_summary(self, summary: dict):
        """Send end-of-day P&L summary with strategy breakdown."""
        pnl = summary.get("daily_pnl", 0)
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        weekly_pnl = summary.get("weekly_pnl", 0)
        theta_pnl = summary.get("theta_pnl", 0)
        theta_trades = summary.get("theta_trades", 0)
        total_capital = summary.get("total_capital", 200000)
        day_return_pct = (pnl / total_capital * 100) if total_capital > 0 else 0

        msg = (
            f"{pnl_emoji} <b>Daily Summary</b> | {datetime.now(IST).strftime('%d-%b-%Y (%A)')}\n"
            f"{'─' * 30}\n"
            f"<b>P&L: {'₹' + f'{pnl:,.0f}' if pnl >= 0 else '-₹' + f'{abs(pnl):,.0f}'}</b> "
            f"({day_return_pct:+.2f}%)\n\n"
            f"📊 <b>Theta Strategy</b>\n"
            f"  Trades: {theta_trades} | P&L: ₹{theta_pnl:,.0f}\n"
            f"  Winners: {summary.get('winners', 0)} | Losers: {summary.get('losers', 0)}\n\n"
            f"💼 <b>Portfolio</b>\n"
            f"  Capital: ₹{total_capital:,.0f}\n"
            f"  Open Positions: {summary.get('open_positions', 0)}\n"
            f"  Deployed: ₹{summary.get('capital_deployed', 0):,.0f}\n"
            f"{'─' * 30}\n"
            f"📅 Weekly P&L: ₹{weekly_pnl:,.0f}\n"
            f"Status: {summary.get('status', 'Active')}"
        )
        self.send_message(msg)

    def notify_risk_breach(self, reason: str):
        """Notify when a risk limit is breached."""
        msg = (
            f"⚠️ <b>RISK ALERT</b>\n"
            f"Trading PAUSED: {reason}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}\n"
            f"Action: Auto-shutdown activated"
        )
        self.send_message(msg)

    def notify_system_error(self, error: str):
        """Notify on critical system error."""
        msg = (
            f"🚨 <b>SYSTEM ERROR</b>\n"
            f"{error[:200]}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        self.send_message(msg)

    def notify_system_start(self):
        """Notify when trading system starts."""
        msg = (
            f"🚀 <b>Trading Agent Started</b>\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S %d-%b-%Y')}\n"
            f"Mode: {'PAPER' if os.getenv('PAPER_TRADE', '').lower() == 'true' else 'LIVE'}"
        )
        self.send_message(msg)

    # =========================================================================
    # PARAMETER APPROVAL FLOW (Learner Integration)
    # =========================================================================

    def send_approval_request(self, proposal: dict) -> bool:
        """
        Send a parameter change proposal with inline keyboard for approval.
        User taps Approve/Reject directly in Telegram.
        
        Returns True if message was sent successfully.
        """
        change_dir = "📈" if proposal.get("new_value", 0) > proposal.get("old_value", 0) else "📉"
        msg = (
            f"🔧 <b>Strategy Adjustment Proposal</b>\n"
            f"{'─' * 25}\n"
            f"Parameter: <code>{proposal.get('key', 'unknown')}</code>\n"
            f"Current: {proposal.get('old_value', '?')}\n"
            f"Proposed: {proposal.get('new_value', '?')} {change_dir} "
            f"({proposal.get('change_pct', 0):+.1f}%)\n"
            f"{'─' * 25}\n"
            f"Reason: {proposal.get('reason', 'No reason given')}\n"
            f"Proposed at: {proposal.get('proposed_at', 'unknown')}"
        )

        if not self.enabled:
            logger.info(f"[TELEGRAM DISABLED] Approval request: {msg}")
            return False

        # Send with inline keyboard (approve/reject buttons)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        proposal_id = f"{proposal.get('key', '')}_{proposal.get('proposed_at', '')}"
        payload = {
            "chat_id": self.chat_id,
            "text": msg,
            "parse_mode": "HTML",
            "reply_markup": {
                "inline_keyboard": [
                    [
                        {"text": "✅ Approve", "callback_data": f"approve:{proposal_id}"},
                        {"text": "❌ Reject", "callback_data": f"reject:{proposal_id}"},
                    ]
                ]
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Approval request send failed: {e}")
            return False

    def check_approval_responses(self) -> list[dict]:
        """
        Poll for callback query responses (approval/rejection).
        Call this periodically to check if user has responded.
        
        Returns list of responses: [{"proposal_id": ..., "action": "approve"|"reject"}]
        """
        if not self.enabled:
            return []

        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        try:
            response = requests.get(url, params={"timeout": 1, "allowed_updates": ["callback_query"]}, timeout=5)
            if response.status_code != 200:
                return []

            data = response.json()
            results = []
            update_ids = []

            for update in data.get("result", []):
                callback = update.get("callback_query")
                if callback:
                    callback_data = callback.get("data", "")
                    update_ids.append(update["update_id"])

                    if ":" in callback_data:
                        action, proposal_id = callback_data.split(":", 1)
                        results.append({"proposal_id": proposal_id, "action": action})

                        # Acknowledge the callback
                        self._answer_callback(callback["id"], f"{'Approved ✅' if action == 'approve' else 'Rejected ❌'}")

            # Mark updates as processed
            if update_ids:
                max_id = max(update_ids)
                requests.get(url, params={"offset": max_id + 1}, timeout=5)

            return results
        except Exception as e:
            logger.error(f"Check approvals failed: {e}")
            return []

    def _answer_callback(self, callback_query_id: str, text: str):
        """Acknowledge a callback query to remove loading state."""
        url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
        try:
            requests.post(url, json={"callback_query_id": callback_query_id, "text": text}, timeout=5)
        except Exception:
            pass

    def notify_weekly_review(self, summary: str):
        """Send the weekly strategy review summary from the Learner."""
        self.send_message(f"📊 <b>Weekly Strategy Review</b>\n\n{summary}")

    def notify_regime_change(self, old_regime: str, new_regime: str, recommendation: dict):
        """Notify when market regime changes."""
        msg = (
            f"🔄 <b>Regime Change Detected</b>\n"
            f"{old_regime.replace('_', ' ').title()} → {new_regime.replace('_', ' ').title()}\n\n"
            f"Recommendations:\n"
        )
        for strategy, action in recommendation.items():
            emoji = "✅" if action == "active" else "⚠️" if action in ("reduced", "cautious") else "🚫"
            msg += f"  {emoji} {strategy}: {action}\n"
        self.send_message(msg)
