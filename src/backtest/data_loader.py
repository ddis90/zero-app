"""
Historical data loader for backtesting.
Fetches via Kite Connect API and caches as parquet files locally.
Falls back to yfinance for index data when Kite is unavailable.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/historical")


class DataLoader:
    """
    Loads and caches historical market data for backtesting.
    
    Data sources (in priority order):
    1. Local parquet cache (fastest)
    2. Kite Connect API (authoritative for Indian markets)
    3. yfinance (fallback for indices, VIX, global data)
    """

    def __init__(self, kite=None, cache_dir: str = "data/historical"):
        self.kite = kite
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_equity_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        start_date: datetime = None,
        end_date: datetime = None,
        interval: str = "day",
    ) -> pd.DataFrame:
        """
        Load OHLCV data for an equity instrument.
        
        Returns DataFrame with columns: [open, high, low, close, volume]
        Indexed by datetime.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()

        cache_key = f"{exchange}_{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        # Check cache
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.debug(f"Loaded {symbol} from cache ({len(df)} bars)")
            return df

        # Fetch from Kite
        if self.kite:
            df = self._fetch_from_kite(symbol, exchange, start_date, end_date, interval)
            if not df.empty:
                df.to_parquet(cache_path)
                logger.info(f"Cached {symbol}: {len(df)} bars → {cache_path.name}")
                return df

        # Fallback to yfinance for known symbols
        df = self._fetch_from_yfinance(symbol, exchange, start_date, end_date)
        if not df.empty:
            df.to_parquet(cache_path)
            logger.info(f"Cached {symbol} (yfinance): {len(df)} bars")
        return df

    def load_nifty_data(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """Load Nifty 50 index data with technical indicators pre-computed."""
        df = self.load_equity_data("NIFTY 50", "NSE", start_date, end_date)
        if df.empty:
            df = self._fetch_from_yfinance("^NSEI", "NSE", start_date, end_date)
        return self._add_indicators(df) if not df.empty else df

    def load_banknifty_data(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """Load Bank Nifty index data."""
        df = self.load_equity_data("NIFTY BANK", "NSE", start_date, end_date)
        if df.empty:
            df = self._fetch_from_yfinance("^NSEBANK", "NSE", start_date, end_date)
        return self._add_indicators(df) if not df.empty else df

    def load_vix_data(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """Load India VIX historical data."""
        df = self.load_equity_data("INDIA VIX", "NSE", start_date, end_date)
        if df.empty:
            df = self._fetch_from_yfinance("^INDIAVIX", "NSE", start_date, end_date)
        return df

    def load_nifty500_universe(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        top_n: int = 50,
    ) -> dict[str, pd.DataFrame]:
        """
        Load historical data for top N Nifty 500 stocks.
        Returns: {symbol: DataFrame}
        """
        # Nifty 50 constituents (representative large-caps for backtesting)
        nifty50_symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
            "SUNPHARMA", "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO",
            "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL",
            "TECHM", "INDUSINDBK", "NESTLEIND", "JSWSTEEL", "ADANIENT",
            "BAJAJFINSV", "ONGC", "COALINDIA", "GRASIM", "CIPLA",
            "DRREDDY", "APOLLOHOSP", "EICHERMOT", "HEROMOTOCO", "DIVISLAB",
            "BPCL", "TATACONSUM", "BRITANNIA", "SBILIFE", "HINDALCO",
            "HDFCLIFE", "BAJAJ-AUTO", "SHRIRAMFIN", "LTIM", "ADANIPORTS",
        ]

        symbols = nifty50_symbols[:top_n]
        universe = {}

        for symbol in symbols:
            df = self.load_equity_data(symbol, "NSE", start_date, end_date)
            if not df.empty and len(df) >= 50:
                universe[symbol] = self._add_indicators(df)

        logger.info(f"Loaded {len(universe)}/{top_n} stocks for backtest universe")
        return universe

    def _fetch_from_kite(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "day",
    ) -> pd.DataFrame:
        """Fetch historical data from Kite Connect API."""
        try:
            # Get instrument token
            instruments = self.kite.instruments(exchange)
            token = None
            for inst in instruments:
                if inst["tradingsymbol"] == symbol:
                    token = inst["instrument_token"]
                    break

            if token is None:
                logger.warning(f"Instrument not found: {exchange}:{symbol}")
                return pd.DataFrame()

            # Kite allows max 2000 candles per request for daily
            # For 2 years of daily data, single request is fine (~500 candles)
            data = self.kite.historical_data(
                instrument_token=token,
                from_date=start_date,
                to_date=end_date,
                interval=interval,
            )

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
            return df

        except Exception as e:
            logger.error(f"Kite fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_from_yfinance(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """Fallback: fetch data from yfinance."""
        try:
            import yfinance as yf

            # Map NSE symbols to yfinance format
            yf_symbol = symbol
            if exchange == "NSE" and not symbol.startswith("^"):
                yf_symbol = f"{symbol}.NS"

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            if df.empty:
                return pd.DataFrame()

            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            return df

        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using pandas-ta."""
        if len(df) < 50:
            return df

        # Moving averages
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["sma_200"] = ta.sma(df["close"], length=200)

        # RSI
        df["rsi"] = ta.rsi(df["close"], length=14)

        # ATR
        atr_result = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr"] = atr_result

        # Bollinger Bands
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None:
            df["bb_upper"] = bbands.iloc[:, 2]  # BBU
            df["bb_lower"] = bbands.iloc[:, 0]  # BBL
            df["bb_mid"] = bbands.iloc[:, 1]    # BBM

        # Volume
        df["volume_sma_20"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # ADX (trend strength)
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_result is not None:
            df["adx"] = adx_result.iloc[:, 0]  # ADX_14

        # Rolling high/low
        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()

        return df

    def get_cache_stats(self) -> dict:
        """Return statistics about cached data."""
        parquet_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)
        return {
            "cached_files": len(parquet_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }

    def clear_cache(self):
        """Clear all cached data."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        logger.info("Cache cleared")
