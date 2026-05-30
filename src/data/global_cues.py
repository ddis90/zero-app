"""
Global market cues fetcher.
Provides overnight/pre-market context from international markets,
commodities, currencies, and macro indicators.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GlobalCues:
    """Snapshot of global market conditions."""
    timestamp: datetime

    # US Markets (overnight)
    sp500_change_pct: float = 0.0
    nasdaq_change_pct: float = 0.0
    dow_change_pct: float = 0.0

    # US Futures (pre-market)
    sp500_futures_change_pct: float = 0.0

    # Asian Markets
    sgx_nifty_change_pct: float = 0.0  # Gift Nifty proxy
    nikkei_change_pct: float = 0.0
    hang_seng_change_pct: float = 0.0

    # Commodities
    crude_oil_price: float = 0.0
    crude_oil_change_pct: float = 0.0
    gold_price: float = 0.0
    gold_change_pct: float = 0.0

    # Currencies
    usdinr: float = 0.0
    usdinr_change_pct: float = 0.0
    dxy_value: float = 0.0
    dxy_change_pct: float = 0.0

    # Bonds
    us_10yr_yield: float = 0.0
    india_10yr_yield: float = 0.0

    @property
    def overall_sentiment(self) -> str:
        """Quick overall global sentiment assessment."""
        bullish_signals = 0
        bearish_signals = 0

        if self.sp500_change_pct > 0.3:
            bullish_signals += 1
        elif self.sp500_change_pct < -0.3:
            bearish_signals += 1

        if self.nasdaq_change_pct > 0.3:
            bullish_signals += 1
        elif self.nasdaq_change_pct < -0.3:
            bearish_signals += 1

        if self.crude_oil_change_pct < -1:
            bullish_signals += 1  # Lower crude = bullish for India
        elif self.crude_oil_change_pct > 2:
            bearish_signals += 1

        if self.dxy_change_pct < -0.3:
            bullish_signals += 1  # Weak dollar = bullish for EM
        elif self.dxy_change_pct > 0.5:
            bearish_signals += 1

        if self.sgx_nifty_change_pct > 0.3:
            bullish_signals += 1
        elif self.sgx_nifty_change_pct < -0.3:
            bearish_signals += 1

        if bullish_signals >= 3:
            return "bullish"
        elif bearish_signals >= 3:
            return "bearish"
        return "neutral"

    @property
    def risk_score(self) -> float:
        """Risk score 0 (safe) to 1 (dangerous). Used for position sizing."""
        risk = 0.0
        # High DXY movement = risk-off
        risk += min(abs(self.dxy_change_pct) / 2, 0.2)
        # Oil spike = inflation risk
        risk += min(max(self.crude_oil_change_pct, 0) / 5, 0.2)
        # Big US drop = global contagion risk
        risk += min(max(-self.sp500_change_pct, 0) / 3, 0.3)
        # INR depreciation
        risk += min(max(self.usdinr_change_pct, 0) / 2, 0.15)
        # Bond yield spike
        risk += min(max(self.us_10yr_yield - 4.5, 0) / 2, 0.15)
        return min(risk, 1.0)

    def to_context_string(self) -> str:
        """Format as a string for LLM context injection."""
        return (
            f"Global Cues ({self.timestamp.strftime('%d-%b %H:%M')}):\n"
            f"  US: S&P500 {self.sp500_change_pct:+.1f}%, Nasdaq {self.nasdaq_change_pct:+.1f}%\n"
            f"  Asia: SGX Nifty {self.sgx_nifty_change_pct:+.1f}%, Nikkei {self.nikkei_change_pct:+.1f}%\n"
            f"  Crude: ${self.crude_oil_price:.1f} ({self.crude_oil_change_pct:+.1f}%)\n"
            f"  Gold: ${self.gold_price:.1f} ({self.gold_change_pct:+.1f}%)\n"
            f"  USD/INR: {self.usdinr:.2f} ({self.usdinr_change_pct:+.2f}%)\n"
            f"  DXY: {self.dxy_value:.1f} ({self.dxy_change_pct:+.1f}%)\n"
            f"  Sentiment: {self.overall_sentiment.upper()} | Risk: {self.risk_score:.2f}"
        )


class GlobalCuesFetcher:
    """Fetches global market data using yfinance."""

    # Ticker mapping
    TICKERS = {
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dow": "^DJI",
        "sp500_futures": "ES=F",
        "nikkei": "^N225",
        "hang_seng": "^HSI",
        "crude_oil": "CL=F",
        "gold": "GC=F",
        "usdinr": "USDINR=X",
        "dxy": "DX-Y.NYB",
        "us_10yr": "^TNX",
    }

    def __init__(self):
        self._cache: Optional[GlobalCues] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_minutes = 15  # Refresh every 15 min

    def fetch_global_cues(self, force_refresh: bool = False) -> GlobalCues:
        """
        Fetch current global market cues.
        Caches results for 15 minutes to avoid excessive API calls.
        """
        if (
            not force_refresh
            and self._cache
            and self._cache_time
            and (datetime.now() - self._cache_time).seconds < self._cache_ttl_minutes * 60
        ):
            return self._cache

        cues = GlobalCues(timestamp=datetime.now())

        # Fetch each ticker
        changes = self._fetch_daily_changes()

        cues.sp500_change_pct = changes.get("sp500", 0)
        cues.nasdaq_change_pct = changes.get("nasdaq", 0)
        cues.dow_change_pct = changes.get("dow", 0)
        cues.sp500_futures_change_pct = changes.get("sp500_futures", 0)
        cues.nikkei_change_pct = changes.get("nikkei", 0)
        cues.hang_seng_change_pct = changes.get("hang_seng", 0)
        cues.crude_oil_change_pct = changes.get("crude_oil", 0)
        cues.gold_change_pct = changes.get("gold", 0)
        cues.usdinr_change_pct = changes.get("usdinr", 0)
        cues.dxy_change_pct = changes.get("dxy", 0)

        # Fetch absolute values for key items
        prices = self._fetch_current_prices()
        cues.crude_oil_price = prices.get("crude_oil", 0)
        cues.gold_price = prices.get("gold", 0)
        cues.usdinr = prices.get("usdinr", 0)
        cues.dxy_value = prices.get("dxy", 0)
        cues.us_10yr_yield = prices.get("us_10yr", 0)

        # SGX/Gift Nifty - approximate from S&P futures correlation
        cues.sgx_nifty_change_pct = cues.sp500_futures_change_pct * 0.8

        self._cache = cues
        self._cache_time = datetime.now()

        logger.info(f"Global cues: {cues.overall_sentiment} | Risk: {cues.risk_score:.2f}")
        return cues

    def _fetch_daily_changes(self) -> dict[str, float]:
        """Fetch daily % changes for all tickers."""
        changes = {}
        tickers_str = " ".join(self.TICKERS.values())

        try:
            data = yf.download(
                tickers_str,
                period="2d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                return changes

            for name, ticker in self.TICKERS.items():
                try:
                    if len(self.TICKERS) > 1:
                        close = data["Close"][ticker]
                    else:
                        close = data["Close"]

                    if len(close.dropna()) >= 2:
                        prev = close.dropna().iloc[-2]
                        curr = close.dropna().iloc[-1]
                        if prev > 0:
                            changes[name] = ((curr - prev) / prev) * 100
                except (KeyError, IndexError):
                    continue

        except Exception as e:
            logger.error(f"yfinance daily changes fetch failed: {e}")

        return changes

    def _fetch_current_prices(self) -> dict[str, float]:
        """Fetch current absolute prices."""
        prices = {}
        key_tickers = {
            "crude_oil": "CL=F",
            "gold": "GC=F",
            "usdinr": "USDINR=X",
            "dxy": "DX-Y.NYB",
            "us_10yr": "^TNX",
        }

        for name, ticker in key_tickers.items():
            try:
                tk = yf.Ticker(ticker)
                info = tk.fast_info
                prices[name] = info.last_price if hasattr(info, "last_price") else 0
            except Exception:
                continue

        return prices

    def get_fii_dii_data(self) -> dict:
        """
        Get FII/DII data from NSE.
        Note: NSE doesn't have a public API for this.
        In production, scrape from NSE or use a data provider.
        
        Returns approximate FII/DII data structure.
        """
        # Placeholder - in production, scrape NSE's FII/DII page
        # URL: https://www.nseindia.com/reports/fii-dii
        logger.info("FII/DII data: Using placeholder (implement NSE scraper for production)")
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "fii_net": 0,  # Positive = buying, Negative = selling (₹ Cr)
            "dii_net": 0,
            "source": "placeholder",
        }

    def get_advance_decline(self) -> dict:
        """
        Get NSE advance-decline ratio.
        Indicates market breadth.
        """
        # Placeholder - scrape from NSE in production
        return {
            "advances": 0,
            "declines": 0,
            "unchanged": 0,
            "ratio": 1.0,  # > 1.5 = strong breadth, < 0.5 = weak
        }
