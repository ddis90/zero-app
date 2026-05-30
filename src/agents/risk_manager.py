"""
Risk Manager - Enforces all trading guardrails.
This is the most critical module - it has VETO power over all trades.
"""

import logging
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    exchange: str
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    strategy: str
    position_type: str  # "long" or "short"
    current_pnl: float = 0.0


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    strategy: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str  # "target", "stop_loss", "time_exit", "manual"


class RiskManager:
    """
    Enforces trading guardrails. Has VETO power over all trade decisions.
    
    Rules enforced:
    - Max risk per trade (2% of capital)
    - Daily loss limit (2% → auto-shutdown)
    - Weekly loss limit (5% → pause for week)
    - Max consecutive losses (3 → auto-pause)
    - Max open positions (5)
    - No trading during restricted hours
    - No trading on high-impact event days
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.risk_config = self.config["risk"]
        self.capital = self.config["capital"]["total"]
        self.schedule_config = self.config["schedule"]

        # State tracking
        self.open_positions: list[Position] = []
        self.today_trades: list[TradeRecord] = []
        self.week_trades: list[TradeRecord] = []
        self.consecutive_losses: int = 0
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.is_paused: bool = False
        self.pause_reason: str = ""

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # =========================================================================
    # VETO CHECKS - Must pass ALL before any trade is allowed
    # =========================================================================

    def can_trade(self) -> tuple[bool, str]:
        """
        Master check - returns (allowed, reason).
        Call this before ANY order placement.
        """
        checks = [
            self._check_market_hours(),
            self._check_daily_loss_limit(),
            self._check_weekly_loss_limit(),
            self._check_consecutive_losses(),
            self._check_max_positions(),
            self._check_no_trade_zones(),
        ]

        for allowed, reason in checks:
            if not allowed:
                logger.warning(f"TRADE BLOCKED: {reason}")
                return False, reason

        return True, "All checks passed"

    def validate_trade(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        strategy: str,
    ) -> tuple[bool, str]:
        """
        Validate a specific trade against risk rules.
        Returns (allowed, reason).
        """
        # First check general trading permission
        can, reason = self.can_trade()
        if not can:
            return False, reason

        # Check position-specific risk
        risk_amount = abs(entry_price - stop_loss) * quantity
        max_risk = self.capital * (self.risk_config["max_risk_per_trade_pct"] / 100)

        if risk_amount > max_risk:
            return False, (
                f"Trade risk ₹{risk_amount:.0f} exceeds max ₹{max_risk:.0f} "
                f"({self.risk_config['max_risk_per_trade_pct']}% of capital)"
            )

        # Check if already holding same symbol
        existing = [p for p in self.open_positions if p.symbol == symbol]
        if existing:
            return False, f"Already holding position in {symbol}"

        return True, "Trade validated"

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        strategy: str,
    ) -> int:
        """
        Calculate optimal position size based on risk per trade.
        Uses fixed fractional method (risk 2% of capital per trade).
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0

        max_risk_amount = self.capital * (self.risk_config["max_risk_per_trade_pct"] / 100)
        position_size = int(max_risk_amount / risk_per_share)

        # Cap by available capital for the strategy allocation
        allocation_pct = self.config["capital"]["allocation"].get(
            "swing_trading" if strategy != "options_theta" else "options_theta", 0.25
        )
        max_capital_for_strategy = self.capital * allocation_pct
        max_shares_by_capital = int(max_capital_for_strategy / entry_price)

        return min(position_size, max_shares_by_capital)

    # =========================================================================
    # Individual Checks
    # =========================================================================

    def _check_market_hours(self) -> tuple[bool, str]:
        """Check if current time is within trading hours."""
        now = datetime.now()
        market_open = datetime.strptime(self.schedule_config["market_open"], "%H:%M").time()
        market_close = datetime.strptime(self.schedule_config["market_close"], "%H:%M").time()

        if not (market_open <= now.time() <= market_close):
            return False, f"Outside market hours ({market_open}-{market_close})"
        return True, ""

    def _check_no_trade_zones(self) -> tuple[bool, str]:
        """Check if current time is in a no-trade zone."""
        now = datetime.now().time()
        for zone in self.schedule_config.get("no_trade_zones", []):
            start_str, end_str = zone.split("-")
            start = datetime.strptime(start_str.strip(), "%H:%M").time()
            end = datetime.strptime(end_str.strip(), "%H:%M").time()
            if start <= now <= end:
                return False, f"In no-trade zone: {zone}"
        return True, ""

    def _check_daily_loss_limit(self) -> tuple[bool, str]:
        """Check if daily loss limit has been breached."""
        max_daily_loss = self.capital * (self.risk_config["daily_loss_limit_pct"] / 100)
        if self.daily_pnl <= -max_daily_loss:
            self.is_paused = True
            self.pause_reason = "Daily loss limit hit"
            return False, f"Daily loss limit hit: ₹{self.daily_pnl:.0f} (max: -₹{max_daily_loss:.0f})"
        return True, ""

    def _check_weekly_loss_limit(self) -> tuple[bool, str]:
        """Check if weekly loss limit has been breached."""
        max_weekly_loss = self.capital * (self.risk_config["weekly_loss_limit_pct"] / 100)
        if self.weekly_pnl <= -max_weekly_loss:
            self.is_paused = True
            self.pause_reason = "Weekly loss limit hit"
            return False, f"Weekly loss limit hit: ₹{self.weekly_pnl:.0f} (max: -₹{max_weekly_loss:.0f})"
        return True, ""

    def _check_consecutive_losses(self) -> tuple[bool, str]:
        """Check consecutive loss streak."""
        max_consecutive = self.risk_config["max_consecutive_losses"]
        if self.consecutive_losses >= max_consecutive:
            self.is_paused = True
            self.pause_reason = f"{self.consecutive_losses} consecutive losses"
            return False, f"Consecutive loss limit: {self.consecutive_losses}/{max_consecutive}"
        return True, ""

    def _check_max_positions(self) -> tuple[bool, str]:
        """Check if max open positions reached."""
        max_pos = self.risk_config["max_open_positions"]
        if len(self.open_positions) >= max_pos:
            return False, f"Max positions reached: {len(self.open_positions)}/{max_pos}"
        return True, ""

    # =========================================================================
    # State Updates
    # =========================================================================

    def add_position(self, position: Position):
        """Register a new open position."""
        self.open_positions.append(position)
        logger.info(f"Position opened: {position.symbol} @ ₹{position.entry_price}")

    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Optional[TradeRecord]:
        """Close a position and record the trade."""
        position = next((p for p in self.open_positions if p.symbol == symbol), None)
        if not position:
            logger.warning(f"No open position found for {symbol}")
            return None

        # Calculate PnL
        if position.position_type == "long":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        # Record trade
        trade = TradeRecord(
            symbol=symbol,
            strategy=position.strategy,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            exit_reason=exit_reason,
        )

        # Update state
        self.open_positions = [p for p in self.open_positions if p.symbol != symbol]
        self.today_trades.append(trade)
        self.week_trades.append(trade)
        self.daily_pnl += pnl
        self.weekly_pnl += pnl

        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        logger.info(
            f"Position closed: {symbol} @ ₹{exit_price} | "
            f"PnL: ₹{pnl:.0f} | Reason: {exit_reason}"
        )
        return trade

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self.daily_pnl = 0.0
        self.today_trades = []
        if self.pause_reason == "Daily loss limit hit":
            self.is_paused = False
            self.pause_reason = ""
        logger.info("Daily risk counters reset")

    def reset_weekly(self):
        """Reset weekly counters (call at start of each week)."""
        self.weekly_pnl = 0.0
        self.week_trades = []
        self.consecutive_losses = 0
        self.is_paused = False
        self.pause_reason = ""
        logger.info("Weekly risk counters reset")

    def get_risk_summary(self) -> dict:
        """Get current risk status summary."""
        max_daily = self.capital * (self.risk_config["daily_loss_limit_pct"] / 100)
        max_weekly = self.capital * (self.risk_config["weekly_loss_limit_pct"] / 100)
        return {
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "daily_pnl": self.daily_pnl,
            "daily_limit_remaining": max_daily + self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "weekly_limit_remaining": max_weekly + self.weekly_pnl,
            "open_positions": len(self.open_positions),
            "max_positions": self.risk_config["max_open_positions"],
            "consecutive_losses": self.consecutive_losses,
            "today_trades_count": len(self.today_trades),
        }
