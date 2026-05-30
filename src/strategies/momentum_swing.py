"""
Momentum Swing Trading Strategy.
Trades quality large-cap stocks on 3-15 day timeframes.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import yaml

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class MomentumSwingStrategy(BaseStrategy):
    """
    Swing trading strategy based on momentum and mean-reversion signals.
    
    Entry Signals:
    1. Momentum Breakout: Price breaks above 20-day high with volume surge
    2. Mean Reversion: RSI < 35 + price at lower Bollinger Band + above 50 DMA
    3. Volume Surge: 2x average volume with bullish candle
    
    Exit Signals:
    1. Target hit (based on ATR multiple)
    2. Trailing stop loss
    3. Time-based exit (max holding period)
    4. RSI overbought (>75)
    """

    def __init__(self, fetcher: DataFetcher, config_path: str = "config/settings.yaml"):
        config = self._load_full_config(config_path)
        super().__init__(name="momentum_swing", config=config["swing"])
        self.fetcher = fetcher
        self.risk_config = config["risk"]
        self.capital_allocation = config["capital"]["total"] * config["capital"]["allocation"]["swing_trading"]

    def _load_full_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def generate_signals(self, market_data: dict) -> list[Signal]:
        """
        Generate swing trading signals from screened stocks.
        
        Args:
            market_data: Dict with key 'candidates' containing screened DataFrame
        """
        signals = []
        candidates = market_data.get("candidates", pd.DataFrame())

        if candidates.empty:
            return signals

        for _, stock in candidates.iterrows():
            symbol = stock["symbol"]

            # Fetch detailed data for signal generation
            df = self.fetcher.get_historical_data(symbol=symbol, interval="day", days=100)
            if df.empty or len(df) < 50:
                continue

            # Calculate indicators
            df = self._add_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Check for momentum breakout
            breakout_signal = self._check_breakout(symbol, df, latest, prev)
            if breakout_signal:
                signals.append(breakout_signal)
                continue

            # Check for mean reversion
            reversion_signal = self._check_mean_reversion(symbol, df, latest)
            if reversion_signal:
                signals.append(reversion_signal)

        return signals

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()
        df["rsi"] = self._calculate_rsi(df["close"])
        df["atr"] = self._calculate_atr(df)
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()
        df["bb_upper"] = df["sma_20"] + 2 * df["close"].rolling(20).std()
        df["bb_lower"] = df["sma_20"] - 2 * df["close"].rolling(20).std()
        return df

    def _check_breakout(
        self, symbol: str, df: pd.DataFrame, latest: pd.Series, prev: pd.Series
    ) -> Optional[Signal]:
        """
        Check for momentum breakout signal.
        Criteria: Close above 20-day high + volume surge + above 50 DMA
        """
        min_volume_surge = self.config["min_volume_surge_multiplier"]

        # Breakout conditions
        price_breakout = latest["close"] > prev["high_20"]
        volume_surge = latest["volume_ratio"] >= min_volume_surge
        above_50dma = latest["close"] > latest["sma_50"]
        rsi_not_extreme = latest["rsi"] < 75  # Not already overbought

        if price_breakout and volume_surge and above_50dma and rsi_not_extreme:
            entry_price = latest["close"]
            atr = latest["atr"]
            stop_loss = entry_price - (2 * atr)  # 2 ATR stop
            target = entry_price + (3 * atr)  # 3 ATR target (1.5 R:R)

            confidence = min(0.85, 
                0.3 * (latest["volume_ratio"] / 3)  # Volume strength
                + 0.3 * (latest["rsi"] / 70)  # RSI momentum
                + 0.2 * (1 if latest["close"] > latest["sma_200"] else 0.5)  # Trend
                + 0.2  # Base confidence for breakout
            )

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                exchange="NSE",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                quantity=0,  # Will be calculated by risk manager
                strategy_name=self.name,
                confidence=confidence,
                reason=(
                    f"Momentum Breakout: {symbol} broke 20-day high "
                    f"(₹{prev['high_20']:.1f}) with {latest['volume_ratio']:.1f}x volume. "
                    f"RSI: {latest['rsi']:.0f}"
                ),
                metadata={
                    "entry_type": "momentum_breakout",
                    "atr": atr,
                    "volume_ratio": latest["volume_ratio"],
                    "rsi": latest["rsi"],
                },
            )
        return None

    def _check_mean_reversion(
        self, symbol: str, df: pd.DataFrame, latest: pd.Series
    ) -> Optional[Signal]:
        """
        Check for mean reversion signal.
        Criteria: RSI < 35 + near lower BB + above 50 DMA (still uptrend)
        """
        # Mean reversion conditions
        oversold = latest["rsi"] < 35
        near_bb_lower = latest["close"] <= latest["bb_lower"] * 1.02
        in_uptrend = latest["close"] > latest["sma_50"]

        if oversold and near_bb_lower and in_uptrend:
            entry_price = latest["close"]
            atr = latest["atr"]
            stop_loss = entry_price - (1.5 * atr)  # Tighter stop for reversals
            target = latest["sma_20"]  # Target = reversion to mean

            confidence = min(0.75,
                0.3 * ((35 - latest["rsi"]) / 35)  # How oversold
                + 0.3 * (1 if in_uptrend else 0.3)  # Uptrend bonus
                + 0.2 * (1 - (latest["close"] - latest["bb_lower"]) / latest["close"])
                + 0.2  # Base
            )

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                exchange="NSE",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                quantity=0,
                strategy_name=self.name,
                confidence=confidence,
                reason=(
                    f"Mean Reversion: {symbol} oversold (RSI: {latest['rsi']:.0f}) "
                    f"near lower BB (₹{latest['bb_lower']:.1f}). "
                    f"Target: SMA20 @ ₹{latest['sma_20']:.1f}"
                ),
                metadata={
                    "entry_type": "mean_reversion",
                    "atr": atr,
                    "rsi": latest["rsi"],
                    "bb_lower": latest["bb_lower"],
                    "target_sma20": latest["sma_20"],
                },
            )
        return None

    def should_exit(self, position: dict, current_data: dict) -> Optional[Signal]:
        """
        Check exit conditions for swing position.
        """
        entry_price = position["entry_price"]
        entry_time = position["entry_time"]
        current_price = current_data.get("ltp", entry_price)
        current_rsi = current_data.get("rsi", 50)
        atr = current_data.get("atr", 0)
        holding_days = (datetime.now() - entry_time).days

        # Exit 1: Target hit
        if current_price >= position.get("target", float("inf")):
            return self._make_exit_signal(
                position, current_price, f"Target hit @ ₹{current_price:.1f}"
            )

        # Exit 2: Stop loss hit
        if current_price <= position.get("stop_loss", 0):
            return self._make_exit_signal(
                position, current_price, f"Stop loss hit @ ₹{current_price:.1f}"
            )

        # Exit 3: Trailing stop (if in profit)
        if current_price > entry_price:
            trailing_stop_pct = self.risk_config["trailing_stop_pct"] / 100
            trailing_stop = current_price * (1 - trailing_stop_pct)
            if trailing_stop > position.get("stop_loss", 0):
                # Update stop loss (done externally)
                pass

        # Exit 4: Time-based exit
        max_holding = self.config["holding_period_days"][1]
        if holding_days >= max_holding:
            return self._make_exit_signal(
                position, current_price,
                f"Max holding period ({max_holding} days) reached"
            )

        # Exit 5: RSI overbought (take profit signal)
        if current_rsi > 75 and current_price > entry_price:
            return self._make_exit_signal(
                position, current_price,
                f"RSI overbought ({current_rsi:.0f}) - taking profit"
            )

        return None

    def _make_exit_signal(self, position: dict, price: float, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.EXIT,
            symbol=position["symbol"],
            exchange="NSE",
            entry_price=price,
            stop_loss=0,
            target=0,
            quantity=position["quantity"],
            strategy_name=self.name,
            confidence=1.0,
            reason=reason,
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
