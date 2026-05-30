"""
Data fetcher module - handles historical and live market data from Kite Connect.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from kiteconnect import KiteConnect, KiteTicker

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches historical and real-time market data via Kite Connect."""

    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self._instruments_cache: dict = {}
        self._ticker: Optional[KiteTicker] = None

    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        """Fetch all tradeable instruments for an exchange."""
        if exchange not in self._instruments_cache:
            instruments = self.kite.instruments(exchange)
            self._instruments_cache[exchange] = pd.DataFrame(instruments)
        return self._instruments_cache[exchange]

    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Get instrument token for a symbol."""
        instruments = self.get_instruments(exchange)
        match = instruments[instruments["tradingsymbol"] == symbol]
        if match.empty:
            raise ValueError(f"Symbol {symbol} not found on {exchange}")
        return int(match.iloc[0]["instrument_token"])

    def get_historical_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "day",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "NIFTY 50")
            exchange: Exchange (NSE, BSE, NFO, MCX)
            interval: Candle interval - minute, 3minute, 5minute, 15minute, 
                      30minute, 60minute, day, week, month
            from_date: Start date (default: `days` ago)
            to_date: End date (default: today)
            days: Number of days of history if from_date not specified
        """
        instrument_token = self.get_instrument_token(symbol, exchange)

        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=days)

        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )
            df = pd.DataFrame(data)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df["symbol"] = symbol
            return df
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_quote(self, symbols: list[str], exchange: str = "NSE") -> dict:
        """
        Get live quotes for multiple symbols.
        
        Returns dict with LTP, OHLC, volume, OI etc.
        """
        instrument_keys = [f"{exchange}:{s}" for s in symbols]
        try:
            return self.kite.quote(instrument_keys)
        except Exception as e:
            logger.error(f"Failed to fetch quotes: {e}")
            return {}

    def get_ltp(self, symbols: list[str], exchange: str = "NSE") -> dict:
        """Get last traded price for symbols."""
        instrument_keys = [f"{exchange}:{s}" for s in symbols]
        try:
            return self.kite.ltp(instrument_keys)
        except Exception as e:
            logger.error(f"Failed to fetch LTP: {e}")
            return {}

    def get_option_chain(
        self,
        underlying: str,
        expiry_date: datetime,
        exchange: str = "NFO",
    ) -> pd.DataFrame:
        """
        Get option chain for an underlying (NIFTY/BANKNIFTY).
        Fetches all strikes for a given expiry.
        """
        instruments = self.get_instruments(exchange)

        # Filter for the underlying and expiry
        mask = (
            (instruments["name"] == underlying)
            & (instruments["expiry"] == expiry_date.date())
            & (instruments["instrument_type"].isin(["CE", "PE"]))
        )
        chain = instruments[mask].copy()

        if chain.empty:
            logger.warning(f"No options found for {underlying} expiry {expiry_date.date()}")
            return pd.DataFrame()

        # Get LTPs for the chain
        tokens = chain["instrument_token"].tolist()
        token_to_symbol = dict(zip(chain["instrument_token"], chain["tradingsymbol"]))

        # Fetch quotes in batches (Kite allows ~500 per call)
        all_quotes = {}
        batch_size = 400
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            instrument_keys = [f"{exchange}:{token_to_symbol[t]}" for t in batch]
            try:
                quotes = self.kite.quote(instrument_keys)
                all_quotes.update(quotes)
            except Exception as e:
                logger.error(f"Failed to fetch option quotes batch: {e}")

        # Enrich chain with live data
        chain["ltp"] = chain["tradingsymbol"].apply(
            lambda s: all_quotes.get(f"{exchange}:{s}", {}).get("last_price", 0)
        )
        chain["oi"] = chain["tradingsymbol"].apply(
            lambda s: all_quotes.get(f"{exchange}:{s}", {}).get("oi", 0)
        )
        chain["volume"] = chain["tradingsymbol"].apply(
            lambda s: all_quotes.get(f"{exchange}:{s}", {}).get("volume", 0)
        )

        return chain.sort_values("strike").reset_index(drop=True)

    def get_upcoming_expiries(self, underlying: str, exchange: str = "NFO") -> list:
        """Get list of upcoming expiry dates for an underlying."""
        instruments = self.get_instruments(exchange)
        mask = (
            (instruments["name"] == underlying)
            & (instruments["instrument_type"].isin(["CE", "PE"]))
        )
        expiries = instruments[mask]["expiry"].unique()
        today = datetime.now().date()
        upcoming = sorted([e for e in expiries if e >= today])
        return upcoming

    def start_ticker(
        self,
        tokens: list[int],
        on_tick: callable,
        on_connect: callable = None,
        on_close: callable = None,
    ):
        """
        Start WebSocket ticker for real-time streaming data.
        
        Args:
            tokens: List of instrument tokens to subscribe
            on_tick: Callback function(ws, ticks) for each tick
            on_connect: Callback on connection established
            on_close: Callback on connection close
        """
        access_token = self.kite.access_token
        api_key = self.kite.api_key

        self._ticker = KiteTicker(api_key, access_token)

        def _on_connect(ws, response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            logger.info(f"Ticker connected, subscribed to {len(tokens)} instruments")
            if on_connect:
                on_connect(ws, response)

        def _on_close(ws, code, reason):
            logger.warning(f"Ticker closed: {code} - {reason}")
            if on_close:
                on_close(ws, code, reason)

        self._ticker.on_ticks = on_tick
        self._ticker.on_connect = _on_connect
        self._ticker.on_close = _on_close
        self._ticker.connect(threaded=True)

    def stop_ticker(self):
        """Stop the WebSocket ticker."""
        if self._ticker:
            self._ticker.close()
            self._ticker = None
            logger.info("Ticker stopped")
