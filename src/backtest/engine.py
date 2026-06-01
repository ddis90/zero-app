"""
Backtest Engine — Portfolio simulation with realistic brokerage and slippage.
Tracks cash, positions, trade history, and enforces risk rules.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a completed backtest trade."""
    trade_id: str
    symbol: str
    strategy: str
    direction: str  # "long" or "short"
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    brokerage: float
    net_pnl: float
    exit_reason: str
    holding_days: int
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestPosition:
    """An open position during backtesting."""
    symbol: str
    strategy: str
    direction: str
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    target: float
    metadata: dict = field(default_factory=dict)


class BrokerageModel:
    """
    Zerodha brokerage model for realistic cost simulation.
    
    Zerodha charges:
    - Equity delivery: ₹0 (free)
    - Equity intraday: ₹20/order or 0.03% (whichever is lower)
    - F&O: ₹20/order
    - STT: 0.0125% on sell side (options), 0.1% on sell (equity delivery)
    - Exchange charges: ~0.00325%
    - GST: 18% on brokerage + exchange charges
    - SEBI charges: ₹10 per crore
    - Stamp duty: 0.003% on buy side
    """

    def calculate_equity_charges(
        self, buy_price: float, sell_price: float, quantity: int, is_intraday: bool = False
    ) -> float:
        """Calculate total charges for an equity trade (buy + sell)."""
        buy_value = buy_price * quantity
        sell_value = sell_price * quantity
        turnover = buy_value + sell_value

        # Brokerage
        if is_intraday:
            brokerage = min(40, turnover * 0.0003)  # ₹20 each side or 0.03%
        else:
            brokerage = 0  # Free delivery

        # STT
        if is_intraday:
            stt = sell_value * 0.00025  # 0.025% on sell (intraday)
        else:
            stt = sell_value * 0.001  # 0.1% on sell (delivery)

        # Exchange charges
        exchange = turnover * 0.0000325

        # GST
        gst = (brokerage + exchange) * 0.18

        # SEBI charges
        sebi = turnover * 0.000001  # ₹10 per crore

        # Stamp duty (on buy side)
        stamp = buy_value * 0.00003

        total = brokerage + stt + exchange + gst + sebi + stamp
        return round(total, 2)

    def calculate_options_charges(
        self, premium_buy: float, premium_sell: float, quantity: int, lot_size: int
    ) -> float:
        """Calculate total charges for an options trade."""
        buy_value = premium_buy * quantity * lot_size
        sell_value = premium_sell * quantity * lot_size
        turnover = buy_value + sell_value

        # Brokerage: ₹20 per order (buy + sell = ₹40)
        brokerage = 40.0

        # STT: 0.0125% on sell side premium (for options)
        stt = sell_value * 0.000625  # 0.0625% on option sell (exercised)

        # Exchange charges
        exchange = turnover * 0.0005  # ~0.05% for NFO

        # GST
        gst = (brokerage + exchange) * 0.18

        # SEBI + stamp
        sebi = turnover * 0.000001
        stamp = buy_value * 0.00003

        total = brokerage + stt + exchange + gst + sebi + stamp
        return round(total, 2)


class BacktestEngine:
    """
    Portfolio simulation engine for backtesting.
    
    Features:
    - Realistic order fills with slippage
    - Position tracking with stop-loss and target
    - Risk management (max positions, daily loss, per-trade risk)
    - Brokerage deduction on every trade
    - Daily equity curve tracking
    """

    def __init__(
        self,
        initial_capital: float = 200000,
        max_risk_per_trade_pct: float = 2.0,
        max_positions: int = 5,
        max_daily_loss_pct: float = 2.0,
        max_weekly_loss_pct: float = 5.0,
        slippage_pct: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_positions = max_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.slippage_pct = slippage_pct / 100.0

        # State
        self.positions: list[BacktestPosition] = []
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[dict] = []
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.current_date: datetime = None
        self._trade_counter: int = 0

        # Brokerage model
        self.brokerage = BrokerageModel()

    def reset(self, capital: float = None):
        """Reset engine for a new backtest run."""
        self.capital = capital or self.initial_capital
        self.initial_capital = self.capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.consecutive_losses = 0
        self._trade_counter = 0

    def can_trade(self) -> tuple[bool, str]:
        """Check if new trades are allowed (risk rules)."""
        # Max positions
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        # Daily loss limit
        daily_loss_limit = self.capital * self.max_daily_loss_pct / 100
        if self.daily_pnl <= -daily_loss_limit:
            return False, f"Daily loss limit hit (₹{self.daily_pnl:.0f})"

        # Weekly loss limit
        weekly_loss_limit = self.capital * self.max_weekly_loss_pct / 100
        if self.weekly_pnl <= -weekly_loss_limit:
            return False, f"Weekly loss limit hit (₹{self.weekly_pnl:.0f})"

        # Consecutive losses
        if self.consecutive_losses >= 3:
            return False, f"3 consecutive losses — paused"

        return True, "OK"

    def calculate_position_size(
        self, entry_price: float, stop_loss: float, strategy: str = "swing"
    ) -> int:
        """Calculate position size based on risk per trade."""
        risk_amount = self.capital * self.max_risk_per_trade_pct / 100
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return 0

        quantity = int(risk_amount / risk_per_share)

        # Ensure we don't exceed available capital
        max_by_capital = int((self.capital * 0.3) / entry_price)  # Max 30% of capital per trade
        quantity = min(quantity, max_by_capital)

        return max(1, quantity)

    def open_position(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        date: datetime,
        price: float,
        quantity: int,
        stop_loss: float,
        target: float,
        metadata: dict = None,
    ) -> bool:
        """
        Open a new position with slippage applied.
        Returns True if position was opened.
        """
        can, reason = self.can_trade()
        if not can:
            return False

        # Apply slippage (worse fill)
        if direction == "long":
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)

        # Check capital sufficiency
        required_capital = fill_price * quantity
        if required_capital > self.capital * 0.9:  # Keep 10% buffer
            quantity = int((self.capital * 0.9) / fill_price)
            if quantity <= 0:
                return False

        position = BacktestPosition(
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            entry_date=date,
            entry_price=fill_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target,
            metadata=metadata or {},
        )
        self.positions.append(position)
        return True

    def close_position(
        self,
        position: BacktestPosition,
        date: datetime,
        price: float,
        exit_reason: str,
    ) -> BacktestTrade:
        """Close an open position and record the trade."""
        # Apply slippage (worse fill on exit)
        if position.direction == "long":
            fill_price = price * (1 - self.slippage_pct)
        else:
            fill_price = price * (1 + self.slippage_pct)

        # Calculate P&L
        if position.direction == "long":
            gross_pnl = (fill_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - fill_price) * position.quantity

        # Calculate brokerage
        is_intraday = (date.date() == position.entry_date.date())
        brokerage = self.brokerage.calculate_equity_charges(
            position.entry_price, fill_price, position.quantity, is_intraday
        )

        net_pnl = gross_pnl - brokerage
        holding_days = (date - position.entry_date).days

        self._trade_counter += 1
        trade = BacktestTrade(
            trade_id=f"BT_{self._trade_counter:04d}",
            symbol=position.symbol,
            strategy=position.strategy,
            direction=position.direction,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=fill_price,
            quantity=position.quantity,
            gross_pnl=gross_pnl,
            brokerage=brokerage,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            holding_days=holding_days,
            metadata=position.metadata,
        )

        # Update state
        self.capital += net_pnl
        self.daily_pnl += net_pnl
        self.weekly_pnl += net_pnl
        self.trades.append(trade)

        # Track consecutive losses
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Remove from open positions
        self.positions = [p for p in self.positions if p is not position]

        return trade

    def check_exits(self, date: datetime, market_data: dict) -> list[BacktestTrade]:
        """
        Check all open positions for exit conditions (stop-loss, trailing stop, target, time).
        
        Args:
            date: Current date
            market_data: Dict mapping symbol -> {open, high, low, close, volume, rsi, atr, adx}
        """
        closed = []
        for position in list(self.positions):
            symbol_data = market_data.get(position.symbol)
            if symbol_data is None:
                continue

            high = symbol_data.get("high", 0)
            low = symbol_data.get("low", 0)
            close = symbol_data.get("close", 0)

            exit_price = None
            exit_reason = None

            if position.direction == "long":
                # --- Trailing Stop Logic ---
                # Move stop to breakeven after 1x ATR profit
                atr = position.metadata.get("atr", 0)
                if atr > 0 and close > position.entry_price + atr:
                    # Trail stop to breakeven + 0.5 ATR (lock in small profit)
                    trailing_stop = position.entry_price + 0.5 * atr
                    if trailing_stop > position.stop_loss:
                        position.stop_loss = trailing_stop

                # Stop loss hit (use low of the day)
                if low <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                # Target hit (use high of the day)
                elif high >= position.target:
                    exit_price = position.target
                    exit_reason = "target"
                # Time-based exit (20 days max for mean-reversion, 15 for breakout)
                elif position.metadata.get("entry_type") == "mean_reversion" and (date - position.entry_date).days >= 20:
                    exit_price = close
                    exit_reason = "time_exit"
                elif (date - position.entry_date).days >= 15:
                    exit_price = close
                    exit_reason = "time_exit"
                # RSI overbought exit
                elif symbol_data.get("rsi", 50) > 75 and close > position.entry_price:
                    exit_price = close
                    exit_reason = "rsi_overbought"
            else:
                # Short position exits (for options)
                if high >= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif low <= position.target:
                    exit_price = position.target
                    exit_reason = "target"

            if exit_price is not None:
                trade = self.close_position(position, date, exit_price, exit_reason)
                closed.append(trade)

        return closed

    def record_daily_equity(self, date: datetime, market_data: dict = None):
        """Record end-of-day equity (including unrealized P&L)."""
        unrealized_pnl = 0.0
        for pos in self.positions:
            if market_data and pos.symbol in market_data:
                current_price = market_data[pos.symbol].get("close", pos.entry_price)
                if pos.direction == "long":
                    unrealized_pnl += (current_price - pos.entry_price) * pos.quantity
                else:
                    unrealized_pnl += (pos.entry_price - current_price) * pos.quantity

        self.equity_curve.append({
            "date": date,
            "capital": self.capital,
            "unrealized_pnl": unrealized_pnl,
            "total_equity": self.capital + unrealized_pnl,
            "open_positions": len(self.positions),
            "daily_pnl": self.daily_pnl,
        })

    def reset_daily(self):
        """Reset daily P&L counter (call at start of each day)."""
        self.daily_pnl = 0.0

    def reset_weekly(self):
        """Reset weekly P&L counter (call on Monday)."""
        self.weekly_pnl = 0.0
        self.consecutive_losses = 0  # Reset consecutive losses weekly

    def open_spread_position(
        self,
        underlying: str,
        strategy: str,
        date: datetime,
        net_premium: float,
        max_loss: float,
        lot_size: int,
        num_lots: int,
        metadata: dict = None,
    ) -> bool:
        """
        Open a credit spread position (for theta selling backtest).
        
        The position tracks net premium collected and max potential loss.
        """
        can, reason = self.can_trade()
        if not can:
            return False

        # For spreads: capital blocked = max_loss per lot * num_lots
        capital_required = max_loss * lot_size * num_lots
        if capital_required > self.capital * 0.3:
            return False

        position = BacktestPosition(
            symbol=f"{underlying}_SPREAD",
            strategy=strategy,
            direction="short",  # Credit spread = short premium
            entry_date=date,
            entry_price=net_premium,
            quantity=num_lots * lot_size,
            stop_loss=net_premium * 1.5,  # Stop at 50% of max loss
            target=net_premium * 0.5,  # Target 50% of premium collected
            metadata={
                **(metadata or {}),
                "net_premium": net_premium,
                "max_loss": max_loss,
                "lot_size": lot_size,
                "num_lots": num_lots,
                "capital_blocked": capital_required,
            },
        )
        self.positions.append(position)
        return True

    def close_spread_at_expiry(
        self,
        position: BacktestPosition,
        date: datetime,
        settlement_loss: float,
    ) -> BacktestTrade:
        """
        Close a spread position at expiry based on settlement value.
        
        Args:
            settlement_loss: How much the spread lost (0 = full profit, max_loss = worst case)
        """
        net_premium = position.metadata.get("net_premium", 0)
        lot_size = position.metadata.get("lot_size", 1)
        num_lots = position.metadata.get("num_lots", 1)
        total_qty = lot_size * num_lots

        gross_pnl = (net_premium - settlement_loss) * total_qty
        brokerage = self.brokerage.calculate_options_charges(
            premium_buy=settlement_loss, premium_sell=net_premium,
            quantity=num_lots, lot_size=lot_size
        )
        net_pnl = gross_pnl - brokerage

        self._trade_counter += 1
        trade = BacktestTrade(
            trade_id=f"BT_{self._trade_counter:04d}",
            symbol=position.symbol,
            strategy=position.strategy,
            direction="short",
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=net_premium,
            exit_price=settlement_loss,
            quantity=total_qty,
            gross_pnl=gross_pnl,
            brokerage=brokerage,
            net_pnl=net_pnl,
            exit_reason="expiry" if settlement_loss == 0 else "expiry_loss",
            holding_days=(date - position.entry_date).days,
            metadata=position.metadata,
        )

        self.capital += net_pnl
        self.daily_pnl += net_pnl
        self.weekly_pnl += net_pnl
        self.trades.append(trade)

        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.positions = [p for p in self.positions if p is not position]
        return trade

    def get_summary(self) -> dict:
        """Get overall backtest performance summary."""
        if not self.trades:
            return {"total_trades": 0, "net_pnl": 0}

        pnls = [t.net_pnl for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        equity_df = pd.DataFrame(self.equity_curve)
        max_equity = equity_df["total_equity"].cummax() if not equity_df.empty else pd.Series([self.initial_capital])
        drawdown = (equity_df["total_equity"] - max_equity) / max_equity if not equity_df.empty else pd.Series([0])

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.capital,
            "total_return_pct": ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            "total_trades": len(self.trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(self.trades) if self.trades else 0,
            "total_pnl": sum(pnls),
            "avg_winner": np.mean(winners) if winners else 0,
            "avg_loser": np.mean(losers) if losers else 0,
            "largest_winner": max(winners) if winners else 0,
            "largest_loser": min(losers) if losers else 0,
            "profit_factor": abs(sum(winners) / sum(losers)) if losers else float("inf"),
            "total_brokerage": sum(t.brokerage for t in self.trades),
            "max_drawdown_pct": abs(drawdown.min()) * 100 if not drawdown.empty else 0,
            "avg_holding_days": np.mean([t.holding_days for t in self.trades]),
        }
