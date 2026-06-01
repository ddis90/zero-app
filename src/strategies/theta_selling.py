"""
Theta Selling Strategy - Weekly options credit spreads on Nifty/BankNifty.
Primary income generator via time decay (theta).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yaml

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class ThetaSellingStrategy(BaseStrategy):
    """
    Sells OTM credit spreads on Nifty/BankNifty weekly options.
    
    Logic:
    - Sell OTM puts/calls with 3-5 days to expiry
    - Buy further OTM options as hedge (spread)
    - Profit from theta decay as options expire worthless
    - Exit at 50% of max profit or if stop-loss hit
    
    Entry Conditions:
    - India VIX < threshold (low volatility = safer selling)
    - Nifty/BankNifty in a defined range (not trending strongly)
    - Adequate distance from current price (3%+ OTM)
    
    Risk Management:
    - Always use spreads (never naked selling)
    - Max loss = spread width - premium collected
    - Stop loss at 50% of max profit (i.e., exit if losing 50% of collected premium)
    """

    def __init__(self, fetcher: DataFetcher, config_path: str = "config/settings.yaml"):
        config = self._load_full_config(config_path)
        super().__init__(name="theta_selling", config=config["options"])
        self.fetcher = fetcher
        self.risk_config = config["risk"]
        self.capital_allocation = config["capital"]["total"] * config["capital"]["allocation"]["options_theta"]
        self.adaptive_otm = config["options"].get("adaptive_otm", {})
        self.max_lots = config["options"].get("max_lots", 2)

    def _load_full_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def generate_signals(self, market_data: dict) -> list[Signal]:
        """
        Generate credit spread signals for Nifty/BankNifty.
        
        Args:
            market_data: Dict with keys:
                - 'nifty_spot': Current Nifty spot price
                - 'banknifty_spot': Current BankNifty spot price
                - 'india_vix': Current India VIX value
                - 'nifty_chain': Option chain DataFrame for Nifty
                - 'banknifty_chain': Option chain DataFrame for BankNifty
        """
        signals = []

        india_vix = market_data.get("india_vix", 15)
        vix_threshold = self.config["vix_threshold"]

        # Don't sell options when VIX is high (market uncertain)
        if india_vix > vix_threshold:
            logger.info(f"VIX ({india_vix:.1f}) > threshold ({vix_threshold}). No theta trades.")
            return signals

        for underlying in self.config["instruments"]:
            spot_key = f"{underlying.lower()}_spot"
            chain_key = f"{underlying.lower()}_chain"

            spot_price = market_data.get(spot_key)
            chain = market_data.get(chain_key)

            if spot_price is None or chain is None or chain.empty:
                continue

            # Generate put credit spread (bullish bias)
            put_signal = self._find_put_credit_spread(underlying, spot_price, chain, india_vix)
            if put_signal:
                signals.append(put_signal)

            # Generate call credit spread (bearish bias) - only if VIX is very low
            if india_vix < vix_threshold * 0.8:
                call_signal = self._find_call_credit_spread(underlying, spot_price, chain, india_vix)
                if call_signal:
                    signals.append(call_signal)

        return signals

    def _find_put_credit_spread(
        self,
        underlying: str,
        spot_price: float,
        chain: pd.DataFrame,
        vix: float,
    ) -> Optional[Signal]:
        """
        Find optimal put credit spread (bull put spread).
        Sell higher strike put, buy lower strike put.
        Uses adaptive OTM distance based on VIX level.
        """
        # Adaptive OTM: tighter in low VIX (higher probability), wider in high VIX (safety)
        if self.adaptive_otm:
            low_thresh = self.adaptive_otm.get("low_vix_threshold", 13)
            mid_thresh = self.adaptive_otm.get("mid_vix_threshold", 18)
            if vix < low_thresh:
                otm_distance = self.adaptive_otm.get("low_vix_otm_pct", 1.5) / 100
            elif vix < mid_thresh:
                otm_distance = self.adaptive_otm.get("mid_vix_otm_pct", 2.5) / 100
            else:
                otm_distance = self.adaptive_otm.get("high_vix_otm_pct", 3.0) / 100
        else:
            otm_distance = self.config["otm_distance_pct"] / 100
        target_strike = spot_price * (1 - otm_distance)

        # Get put options
        puts = chain[chain["instrument_type"] == "PE"].copy()
        if puts.empty:
            return None

        # Find sell strike (closest to target, must be OTM)
        puts["distance"] = abs(puts["strike"] - target_strike)
        sell_put = puts.loc[puts["distance"].idxmin()]

        if sell_put["strike"] >= spot_price:
            return None  # Would be ITM

        # Find buy strike (1-2 strikes below sell strike for defined risk)
        spread_width = self._get_spread_width(underlying)
        buy_strike = sell_put["strike"] - spread_width
        buy_put_candidates = puts[puts["strike"] == buy_strike]

        if buy_put_candidates.empty:
            # Find nearest available strike below
            lower_puts = puts[puts["strike"] < sell_put["strike"]].sort_values("strike", ascending=False)
            if lower_puts.empty:
                return None
            buy_put = lower_puts.iloc[0]
        else:
            buy_put = buy_put_candidates.iloc[0]

        # Calculate premium and risk
        premium_collected = sell_put["ltp"] - buy_put["ltp"]
        max_loss = (sell_put["strike"] - buy_put["strike"]) - premium_collected
        max_profit = premium_collected

        if premium_collected <= 0 or max_loss <= 0:
            return None

        # Check if premium/risk ratio is acceptable
        premium_risk_ratio = max_loss / premium_collected
        if premium_risk_ratio > self.config["max_premium_risk_ratio"]:
            return None

        # Calculate confidence based on distance + VIX
        distance_pct = (spot_price - sell_put["strike"]) / spot_price * 100
        confidence = min(0.9, (distance_pct / 5) * 0.5 + ((20 - vix) / 20) * 0.5)

        lot_size = self._get_lot_size(underlying)
        num_lots = min(self.max_lots, max(1, int(self.capital_allocation / 100000)))

        # Check minimum premium threshold
        min_premium = self.config.get("min_premium", 8)
        if premium_collected < min_premium:
            return None

        return Signal(
            signal_type=SignalType.SELL,
            symbol=f"{underlying}_PUT_SPREAD",
            exchange="NFO",
            entry_price=premium_collected,
            stop_loss=premium_collected * self.config.get("stop_loss_multiplier", 2.0),
            target=premium_collected * 0.5,  # Exit at 50% profit
            quantity=lot_size * num_lots,
            strategy_name=self.name,
            confidence=confidence,
            reason=(
                f"Bull Put Spread: Sell {sell_put['strike']}PE @ ₹{sell_put['ltp']:.1f}, "
                f"Buy {buy_put['strike']}PE @ ₹{buy_put['ltp']:.1f} | "
                f"Premium: ₹{premium_collected:.1f} | Max Loss: ₹{max_loss:.1f} | "
                f"VIX: {vix:.1f} | Distance: {distance_pct:.1f}% | Lots: {num_lots}"
            ),
            metadata={
                "sell_strike": sell_put["strike"],
                "buy_strike": buy_put["strike"],
                "sell_symbol": sell_put["tradingsymbol"],
                "buy_symbol": buy_put["tradingsymbol"],
                "premium_collected": premium_collected,
                "max_loss": max_loss,
                "max_profit": max_profit,
                "spread_type": "bull_put_spread",
                "underlying": underlying,
            },
        )

    def _find_call_credit_spread(
        self,
        underlying: str,
        spot_price: float,
        chain: pd.DataFrame,
        vix: float,
    ) -> Optional[Signal]:
        """
        Find optimal call credit spread (bear call spread).
        Sell lower strike call, buy higher strike call.
        """
        otm_distance = self.config["otm_distance_pct"] / 100
        target_strike = spot_price * (1 + otm_distance)

        calls = chain[chain["instrument_type"] == "CE"].copy()
        if calls.empty:
            return None

        calls["distance"] = abs(calls["strike"] - target_strike)
        sell_call = calls.loc[calls["distance"].idxmin()]

        if sell_call["strike"] <= spot_price:
            return None  # Would be ITM

        spread_width = self._get_spread_width(underlying)
        buy_strike = sell_call["strike"] + spread_width
        buy_call_candidates = calls[calls["strike"] == buy_strike]

        if buy_call_candidates.empty:
            higher_calls = calls[calls["strike"] > sell_call["strike"]].sort_values("strike")
            if higher_calls.empty:
                return None
            buy_call = higher_calls.iloc[0]
        else:
            buy_call = buy_call_candidates.iloc[0]

        premium_collected = sell_call["ltp"] - buy_call["ltp"]
        max_loss = (buy_call["strike"] - sell_call["strike"]) - premium_collected
        max_profit = premium_collected

        if premium_collected <= 0 or max_loss <= 0:
            return None

        premium_risk_ratio = max_loss / premium_collected
        if premium_risk_ratio > self.config["max_premium_risk_ratio"]:
            return None

        distance_pct = (sell_call["strike"] - spot_price) / spot_price * 100
        confidence = min(0.9, (distance_pct / 5) * 0.5 + ((20 - vix) / 20) * 0.5)

        lot_size = self._get_lot_size(underlying)

        return Signal(
            signal_type=SignalType.SELL,
            symbol=f"{underlying}_CALL_SPREAD",
            exchange="NFO",
            entry_price=premium_collected,
            stop_loss=premium_collected + (max_loss * self.config["stop_loss_pct"] / 100),
            target=premium_collected * 0.5,
            quantity=lot_size,
            strategy_name=self.name,
            confidence=confidence,
            reason=(
                f"Bear Call Spread: Sell {sell_call['strike']}CE @ ₹{sell_call['ltp']:.1f}, "
                f"Buy {buy_call['strike']}CE @ ₹{buy_call['ltp']:.1f} | "
                f"Premium: ₹{premium_collected:.1f} | Max Loss: ₹{max_loss:.1f} | "
                f"VIX: {vix:.1f} | Distance: {distance_pct:.1f}%"
            ),
            metadata={
                "sell_strike": sell_call["strike"],
                "buy_strike": buy_call["strike"],
                "sell_symbol": sell_call["tradingsymbol"],
                "buy_symbol": buy_call["tradingsymbol"],
                "premium_collected": premium_collected,
                "max_loss": max_loss,
                "max_profit": max_profit,
                "spread_type": "bear_call_spread",
                "underlying": underlying,
            },
        )

    def should_exit(self, position: dict, current_data: dict) -> Optional[Signal]:
        """
        Check if an options spread position should be exited.
        
        Exit conditions:
        1. Profit target hit (50% of max profit)
        2. Stop loss hit (loss exceeds 50% of max profit)
        3. Expiry approaching (< 1 day to expiry, close regardless)
        """
        current_premium = current_data.get("current_spread_value", 0)
        entry_premium = position.get("entry_premium", 0)
        max_profit = position.get("max_profit", 0)
        days_to_expiry = current_data.get("days_to_expiry", 5)

        # Current PnL (for credit spreads, profit = entry premium - current value)
        current_pnl = entry_premium - current_premium

        # Exit 1: Profit target (50% of max profit captured)
        if current_pnl >= max_profit * 0.5:
            return Signal(
                signal_type=SignalType.EXIT,
                symbol=position["symbol"],
                exchange="NFO",
                entry_price=current_premium,
                stop_loss=0,
                target=0,
                quantity=position["quantity"],
                strategy_name=self.name,
                confidence=1.0,
                reason=f"Profit target hit: ₹{current_pnl:.0f} (50% of max ₹{max_profit:.0f})",
            )

        # Exit 2: Stop loss
        stop_loss_amount = max_profit * (self.config["stop_loss_pct"] / 100)
        if current_pnl <= -stop_loss_amount:
            return Signal(
                signal_type=SignalType.EXIT,
                symbol=position["symbol"],
                exchange="NFO",
                entry_price=current_premium,
                stop_loss=0,
                target=0,
                quantity=position["quantity"],
                strategy_name=self.name,
                confidence=1.0,
                reason=f"Stop loss hit: Loss ₹{abs(current_pnl):.0f} > limit ₹{stop_loss_amount:.0f}",
            )

        # Exit 3: Close to expiry (less than 0.5 days)
        if days_to_expiry < 0.5:
            return Signal(
                signal_type=SignalType.EXIT,
                symbol=position["symbol"],
                exchange="NFO",
                entry_price=current_premium,
                stop_loss=0,
                target=0,
                quantity=position["quantity"],
                strategy_name=self.name,
                confidence=1.0,
                reason=f"Expiry approaching ({days_to_expiry:.1f} days). Closing to avoid assignment risk.",
            )

        return None

    def _get_lot_size(self, underlying: str) -> int:
        """Get lot size for underlying."""
        lot_sizes = {"NIFTY": 25, "BANKNIFTY": 15}
        return lot_sizes.get(underlying, 25)

    def _get_spread_width(self, underlying: str) -> float:
        """Get spread width (distance between strikes)."""
        spread_widths = {"NIFTY": 100, "BANKNIFTY": 100}
        return spread_widths.get(underlying, 100)
