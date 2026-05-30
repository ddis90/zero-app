"""
Real-time news and market cues ingestion pipeline.
Fetches news from RSS feeds, parses articles, and provides structured data
for sentiment analysis and strategy context.
"""

import logging
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import feedparser
import requests

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """A single news item from any source."""
    title: str
    summary: str
    source: str
    url: str
    published: datetime
    category: str  # "market", "sector", "global", "policy", "corporate"
    symbols_mentioned: list[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1 to 1, filled by analyst
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(f"{self.url}{self.title}".encode()).hexdigest()[:12]


# RSS Feed sources for Indian markets
RSS_FEEDS = {
    "et_markets": {
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "category": "market",
        "source": "Economic Times",
    },
    "et_stocks": {
        "url": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "category": "market",
        "source": "Economic Times",
    },
    "moneycontrol_market": {
        "url": "https://www.moneycontrol.com/rss/marketreports.xml",
        "category": "market",
        "source": "Moneycontrol",
    },
    "moneycontrol_business": {
        "url": "https://www.moneycontrol.com/rss/business.xml",
        "category": "corporate",
        "source": "Moneycontrol",
    },
    "livemint_market": {
        "url": "https://www.livemint.com/rss/markets",
        "category": "market",
        "source": "LiveMint",
    },
    "livemint_money": {
        "url": "https://www.livemint.com/rss/money",
        "category": "market",
        "source": "LiveMint",
    },
    "rbi_press": {
        "url": "https://rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?output=rss",
        "category": "policy",
        "source": "RBI",
    },
}

# Keywords that indicate high-impact news
HIGH_IMPACT_KEYWORDS = [
    "rbi", "rate cut", "rate hike", "monetary policy",
    "sebi", "circular", "regulation",
    "budget", "fiscal deficit", "gst",
    "fii", "dii", "foreign institutional",
    "nifty", "sensex", "crash", "rally", "circuit",
    "earnings", "quarterly results", "profit",
    "ban", "restriction", "investigation",
    "merger", "acquisition", "takeover",
    "inflation", "cpi", "gdp",
]

# Sector keywords mapping
SECTOR_KEYWORDS = {
    "IT": ["tcs", "infosys", "wipro", "hcl", "tech mahindra", "it sector", "technology"],
    "BANK": ["sbi", "hdfc bank", "icici bank", "kotak", "banking", "npa", "credit growth"],
    "PHARMA": ["pharma", "drug", "fda", "usfda", "healthcare", "hospital"],
    "AUTO": ["auto", "ev", "electric vehicle", "maruti", "tata motors", "bajaj"],
    "FMCG": ["fmcg", "hindustan unilever", "itc", "nestle", "consumer goods"],
    "METAL": ["metal", "steel", "tata steel", "jsw", "aluminium", "copper"],
    "ENERGY": ["reliance", "ongc", "oil", "gas", "power", "energy", "coal"],
    "REALTY": ["real estate", "realty", "housing", "dlf", "godrej properties"],
}


class NewsFeed:
    """
    Ingests real-time market news from multiple RSS sources.
    Categorizes, deduplicates, and structures news for the AI analyst.
    """

    def __init__(self):
        self._seen_ids: set[str] = set()
        self._news_buffer: list[NewsItem] = []
        self._last_fetch_time: dict[str, datetime] = {}

    def fetch_all_feeds(self, max_age_hours: int = 6) -> list[NewsItem]:
        """
        Fetch news from all configured RSS feeds.
        Deduplicates and filters by age.
        """
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        for feed_id, feed_config in RSS_FEEDS.items():
            try:
                items = self._fetch_feed(
                    feed_url=feed_config["url"],
                    source=feed_config["source"],
                    category=feed_config["category"],
                    cutoff=cutoff_time,
                )
                all_news.extend(items)
            except Exception as e:
                logger.debug(f"Feed {feed_id} fetch failed: {e}")
                continue

        # Deduplicate
        new_items = []
        for item in all_news:
            if item.id not in self._seen_ids:
                self._seen_ids.add(item.id)
                new_items.append(item)

        # Sort by recency
        new_items.sort(key=lambda x: x.published, reverse=True)
        self._news_buffer.extend(new_items)

        # Keep buffer manageable (last 200 items)
        self._news_buffer = self._news_buffer[:200]

        logger.info(f"Fetched {len(new_items)} new items from {len(RSS_FEEDS)} feeds")
        return new_items

    def _fetch_feed(
        self, feed_url: str, source: str, category: str, cutoff: datetime
    ) -> list[NewsItem]:
        """Fetch and parse a single RSS feed."""
        feed = feedparser.parse(feed_url)
        items = []

        for entry in feed.entries[:20]:  # Max 20 per feed
            # Parse published date
            published = self._parse_date(entry)
            if published and published < cutoff:
                continue

            title = entry.get("title", "").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()
            # Remove HTML tags from summary
            summary = self._strip_html(summary)
            url = entry.get("link", "")

            if not title:
                continue

            # Detect mentioned symbols/sectors
            symbols = self._extract_symbols(title + " " + summary)

            item = NewsItem(
                title=title,
                summary=summary[:500],  # Cap summary length
                source=source,
                url=url,
                published=published or datetime.now(),
                category=category,
                symbols_mentioned=symbols,
            )
            items.append(item)

        return items

    def get_recent_news(self, hours: int = 2, category: str = None) -> list[NewsItem]:
        """Get recent news items, optionally filtered by category."""
        cutoff = datetime.now() - timedelta(hours=hours)
        items = [n for n in self._news_buffer if n.published >= cutoff]
        if category:
            items = [n for n in items if n.category == category]
        return items

    def get_high_impact_news(self, hours: int = 4) -> list[NewsItem]:
        """Get news items that match high-impact keywords."""
        recent = self.get_recent_news(hours=hours)
        high_impact = []
        for item in recent:
            text = (item.title + " " + item.summary).lower()
            if any(kw in text for kw in HIGH_IMPACT_KEYWORDS):
                high_impact.append(item)
        return high_impact

    def get_sector_news(self, sector: str, hours: int = 6) -> list[NewsItem]:
        """Get news relevant to a specific sector."""
        keywords = SECTOR_KEYWORDS.get(sector.upper(), [])
        if not keywords:
            return []

        recent = self.get_recent_news(hours=hours)
        return [
            item for item in recent
            if any(kw in (item.title + " " + item.summary).lower() for kw in keywords)
        ]

    def get_news_for_symbols(self, symbols: list[str], hours: int = 12) -> list[NewsItem]:
        """Get news mentioning specific symbols."""
        recent = self.get_recent_news(hours=hours)
        symbols_lower = [s.lower() for s in symbols]
        return [
            item for item in recent
            if any(s in [x.lower() for x in item.symbols_mentioned] for s in symbols_lower)
        ]

    def get_news_summary_text(self, max_items: int = 10) -> list[str]:
        """Get plain text summaries of recent news for LLM analysis."""
        recent = self.get_recent_news(hours=3)
        return [f"[{item.source}] {item.title}" for item in recent[:max_items]]

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract potential stock symbols mentioned in text."""
        # Common large-cap mentions
        symbol_map = {
            "reliance": "RELIANCE", "tcs": "TCS", "infosys": "INFY",
            "hdfc bank": "HDFCBANK", "icici bank": "ICICIBANK",
            "sbi": "SBIN", "kotak": "KOTAKBANK", "itc": "ITC",
            "wipro": "WIPRO", "hcl tech": "HCLTECH", "bajaj finance": "BAJFINANCE",
            "maruti": "MARUTI", "tata motors": "TATAMOTORS",
            "tata steel": "TATASTEEL", "sun pharma": "SUNPHARMA",
            "hindustan unilever": "HINDUNILVR", "asian paints": "ASIANPAINT",
            "larsen": "LT", "axis bank": "AXISBANK", "bharti airtel": "BHARTIARTL",
            "adani enterprises": "ADANIENT", "adani ports": "ADANIPORTS",
        }
        text_lower = text.lower()
        found = []
        for keyword, symbol in symbol_map.items():
            if keyword in text_lower:
                found.append(symbol)
        return list(set(found))

    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse date from RSS entry."""
        import time
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        return None

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        import re
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
