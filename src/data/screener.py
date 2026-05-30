"""
Stock Universe Screener - filters the market for quality stocks.
Eliminates penny stocks, applies fundamental + technical filters.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yaml

from src.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class StockScreener:
    """Screens stocks based on fundamental and technical criteria."""

    def __init__(self, fetcher: DataFetcher, config_path: str = "config/settings.yaml"):
        self.fetcher = fetcher
        self.config = self._load_config(config_path)
        self.universe_config = self.config["universe"]
        self._cached_universe: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get_nifty500_symbols(self) -> list[str]:
        """
        Get Nifty 500 constituents from NSE instruments.
        Filters for equity segment, active trading.
        """
        instruments = self.fetcher.get_instruments("NSE")
        # Filter for equities (EQ segment)
        equities = instruments[
            (instruments["segment"] == "NSE")
            & (instruments["instrument_type"] == "EQ")
        ].copy()
        return equities["tradingsymbol"].tolist()

    def apply_market_cap_filter(self, symbols: list[str]) -> list[str]:
        """
        Filter stocks by market cap using quote data.
        Removes anything below min_market_cap_cr threshold.
        
        Note: Kite doesn't provide market cap directly.
        We use average volume * price as a proxy, or maintain a 
        pre-fetched list from NSE/Screener.in
        """
        min_cap = self.universe_config["min_market_cap_cr"]
        min_volume = self.universe_config["min_avg_volume_cr"]

        # Fetch quotes in batches
        qualified = []
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            quotes = self.fetcher.get_quote(batch)
            for key, data in quotes.items():
                symbol = key.split(":")[1]
                ltp = data.get("last_price", 0)
                volume = data.get("volume", 0)
                # Daily turnover as proxy for liquidity
                daily_turnover_cr = (ltp * volume) / 1e7  # Convert to crores
                if daily_turnover_cr >= min_volume:
                    qualified.append(symbol)

        logger.info(f"Volume filter: {len(symbols)} -> {len(qualified)} stocks")
        return qualified

    def apply_technical_filters(self, symbols: list[str]) -> pd.DataFrame:
        """
        Apply technical filters:
        - Above 200 DMA
        - RSI in acceptable range
        - Not in overbought/oversold extremes for entry
        """
        results = []

        for symbol in symbols:
            try:
                df = self.fetcher.get_historical_data(
                    symbol=symbol,
                    interval="day",
                    days=250,  # ~1 year for 200 DMA
                )
                if df.empty or len(df) < 200:
                    continue

                # Calculate indicators
                df["sma_200"] = df["close"].rolling(window=200).mean()
                df["sma_50"] = df["close"].rolling(window=50).mean()
                df["rsi"] = self._calculate_rsi(df["close"], period=14)
                df["avg_volume_20"] = df["volume"].rolling(window=20).mean()

                latest = df.iloc[-1]

                # Filters
                above_200dma = latest["close"] > latest["sma_200"]
                rsi_range = self.universe_config["technical_filters"]["rsi_entry_range"]
                rsi_ok = rsi_range[0] <= latest["rsi"] <= rsi_range[1]

                if above_200dma and (self.universe_config["technical_filters"]["above_200dma"]):
                    results.append({
                        "symbol": symbol,
                        "close": latest["close"],
                        "sma_200": latest["sma_200"],
                        "sma_50": latest["sma_50"],
                        "rsi": latest["rsi"],
                        "above_200dma": above_200dma,
                        "rsi_in_range": rsi_ok,
                        "avg_volume_20": latest["avg_volume_20"],
                        "distance_from_200dma_pct": (
                            (latest["close"] - latest["sma_200"]) / latest["sma_200"] * 100
                        ),
                    })
            except Exception as e:
                logger.debug(f"Skipping {symbol}: {e}")
                continue

        result_df = pd.DataFrame(results)
        logger.info(f"Technical filter: {len(symbols)} -> {len(result_df)} stocks")
        return result_df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_swing_candidates(self, max_stocks: int = 20) -> pd.DataFrame:
        """
        Get filtered stock candidates for swing trading.
        Applies all filters and ranks by momentum/quality score.
        """
        # Step 1: Get universe
        all_symbols = self.get_nifty500_symbols()
        logger.info(f"Total NSE equities: {len(all_symbols)}")

        # Step 2: Volume/liquidity filter
        liquid_symbols = self.apply_market_cap_filter(all_symbols)

        # Step 3: Technical filters
        screened = self.apply_technical_filters(liquid_symbols)

        if screened.empty:
            return pd.DataFrame()

        # Step 4: Rank by composite score
        # Higher RSI momentum + closer to 50 DMA = strong trend
        screened["momentum_score"] = (
            screened["rsi"] / 100 * 0.4
            + (screened["distance_from_200dma_pct"].clip(0, 30) / 30) * 0.3
            + (screened["close"] > screened["sma_50"]).astype(float) * 0.3
        )

        # Sort by score and return top candidates
        screened = screened.sort_values("momentum_score", ascending=False)
        return screened.head(max_stocks).reset_index(drop=True)

    def get_mean_reversion_candidates(self, max_stocks: int = 10) -> pd.DataFrame:
        """
        Find stocks that have pulled back to support levels.
        Good for mean-reversion swing entries.
        """
        all_symbols = self.get_nifty500_symbols()
        liquid_symbols = self.apply_market_cap_filter(all_symbols)

        results = []
        for symbol in liquid_symbols:
            try:
                df = self.fetcher.get_historical_data(symbol=symbol, interval="day", days=100)
                if df.empty or len(df) < 50:
                    continue

                df["sma_50"] = df["close"].rolling(50).mean()
                df["sma_20"] = df["close"].rolling(20).mean()
                df["rsi"] = self._calculate_rsi(df["close"])
                df["bb_lower"] = df["sma_20"] - 2 * df["close"].rolling(20).std()

                latest = df.iloc[-1]

                # Mean reversion criteria:
                # RSI < 35 (oversold) AND price near lower Bollinger Band
                # AND above 50 DMA (still in uptrend)
                if (
                    latest["rsi"] < 35
                    and latest["close"] <= latest["bb_lower"] * 1.02
                    and latest["close"] > latest["sma_50"]
                ):
                    results.append({
                        "symbol": symbol,
                        "close": latest["close"],
                        "rsi": latest["rsi"],
                        "bb_lower": latest["bb_lower"],
                        "sma_50": latest["sma_50"],
                        "bounce_potential_pct": (latest["sma_20"] - latest["close"]) / latest["close"] * 100,
                    })
            except Exception as e:
                logger.debug(f"Skipping {symbol}: {e}")
                continue

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values("bounce_potential_pct", ascending=False)
        return result_df.head(max_stocks).reset_index(drop=True)

    def load_blacklist(self) -> list[str]:
        """Load blacklisted symbols from instruments config."""
        with open("config/instruments.yaml", "r") as f:
            instruments_config = yaml.safe_load(f)
        return instruments_config.get("blacklist", [])

    def filter_blacklist(self, symbols: list[str]) -> list[str]:
        """Remove blacklisted symbols."""
        blacklist = self.load_blacklist()
        return [s for s in symbols if s not in blacklist]
