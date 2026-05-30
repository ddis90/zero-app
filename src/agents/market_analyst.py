"""
AI-powered Market Analyst Agent.
Uses LLM for news sentiment analysis, market regime detection,
and RAG-enhanced context from the knowledge base.

Enhanced with:
- Real-time news feed ingestion
- Global market cues integration
- Vector knowledge base (ChromaDB) for historical context
- Self-adjusting confidence via learner feedback
"""

import logging
import os
from datetime import datetime
from typing import Optional

import yaml

from src.data.news_feed import NewsFeed
from src.data.global_cues import GlobalCuesFetcher, GlobalCues
from src.data.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class MarketAnalyst:
    """
    AI agent that provides market context for trading decisions.
    
    Responsibilities:
    - News sentiment analysis (avoid trading during negative sentiment)
    - Market regime detection (trending vs. ranging)
    - Global cues integration (US, Asia, commodities, FX)
    - RAG-enhanced decisions from historical knowledge base
    - Earnings calendar awareness (no positions before earnings)
    - Event risk assessment
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        knowledge_base: KnowledgeBase = None,
    ):
        self.config = self._load_config(config_path)
        self.ai_config = self.config["ai"]
        self._client = None
        self._daily_tokens_used = 0

        # Adaptive components
        self.news_feed = NewsFeed()
        self.global_cues_fetcher = GlobalCuesFetcher()
        self.kb = knowledge_base or KnowledgeBase()
        self._latest_global_cues: Optional[GlobalCues] = None
        self._latest_sentiment: Optional[dict] = None

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _get_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            provider = self.ai_config["provider"]
            api_key = os.getenv("OPENAI_API_KEY", self.ai_config.get("api_key", ""))

            if provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            elif provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

        return self._client

    def analyze_sentiment(self, news_items: list[str]) -> dict:
        """
        Analyze market news sentiment using LLM.
        
        Returns:
            dict with keys: overall_sentiment (-1 to 1), key_events, trade_impact
        """
        if not news_items:
            return {"overall_sentiment": 0, "key_events": [], "trade_impact": "neutral"}

        if self._daily_tokens_used >= self.ai_config["max_tokens_per_day"]:
            logger.warning("Daily token limit reached. Using neutral sentiment.")
            return {"overall_sentiment": 0, "key_events": [], "trade_impact": "neutral"}

        prompt = f"""Analyze the following Indian market news for trading sentiment.
Rate overall sentiment from -1 (very bearish) to +1 (very bullish).
Identify any events that could impact Nifty/BankNifty options or large-cap stocks.

News items:
{chr(10).join(f'- {item}' for item in news_items[:10])}

Respond in JSON format:
{{
    "overall_sentiment": <float -1 to 1>,
    "key_events": [<list of impactful events>],
    "trade_impact": "bullish" | "bearish" | "neutral" | "avoid_trading",
    "affected_sectors": [<sectors impacted>],
    "reasoning": "<brief explanation>"
}}"""

        try:
            client = self._get_client()
            if self.ai_config["provider"] == "openai":
                response = client.chat.completions.create(
                    model=self.ai_config["model"],
                    messages=[
                        {"role": "system", "content": "You are a quantitative market analyst for Indian equity markets. Be concise and data-driven."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                self._daily_tokens_used += response.usage.total_tokens
                import json
                return json.loads(response.choices[0].message.content)
            elif self.ai_config["provider"] == "anthropic":
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                self._daily_tokens_used += response.usage.input_tokens + response.usage.output_tokens
                import json
                return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"overall_sentiment": 0, "key_events": [], "trade_impact": "neutral"}

    def detect_market_regime(self, nifty_data: dict) -> str:
        """
        Detect current market regime based on technical indicators.
        
        Returns: "trending_up", "trending_down", "ranging", "volatile"
        """
        try:
            close = nifty_data.get("close", 0)
            sma_20 = nifty_data.get("sma_20", close)
            sma_50 = nifty_data.get("sma_50", close)
            sma_200 = nifty_data.get("sma_200", close)
            atr_pct = nifty_data.get("atr_pct", 1.0)  # ATR as % of price
            adx = nifty_data.get("adx", 20)

            # High volatility regime
            if atr_pct > 2.0:
                return "volatile"

            # Trending up: Price > 20 SMA > 50 SMA > 200 SMA
            if close > sma_20 > sma_50 > sma_200 and adx > 25:
                return "trending_up"

            # Trending down
            if close < sma_20 < sma_50 < sma_200 and adx > 25:
                return "trending_down"

            # Ranging
            return "ranging"

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return "ranging"

    def get_strategy_recommendation(self, regime: str, sentiment: dict) -> dict:
        """
        Recommend strategy adjustments based on regime and sentiment.
        """
        recommendations = {
            "trending_up": {
                "theta_selling": "active",  # Safe to sell puts
                "swing_trading": "aggressive",  # Momentum works
                "bias": "bullish",
            },
            "trending_down": {
                "theta_selling": "reduced",  # Sell calls instead of puts
                "swing_trading": "defensive",  # Only mean reversion
                "bias": "bearish",
            },
            "ranging": {
                "theta_selling": "active",  # Best environment for theta
                "swing_trading": "selective",  # Only strong setups
                "bias": "neutral",
            },
            "volatile": {
                "theta_selling": "paused",  # Don't sell when VIX is high
                "swing_trading": "paused",  # Wait for clarity
                "bias": "avoid",
            },
        }

        rec = recommendations.get(regime, recommendations["ranging"])

        # Override if sentiment is strongly negative
        if sentiment.get("trade_impact") == "avoid_trading":
            rec = {
                "theta_selling": "paused",
                "swing_trading": "paused",
                "bias": "avoid",
            }

        return rec

    def check_earnings_calendar(self, symbols: list[str]) -> list[str]:
        """
        Check which symbols have upcoming earnings (avoid trading these).
        
        Returns list of symbols with earnings in next 3 days.
        Note: In production, integrate with NSE corporate actions API.
        """
        # Placeholder - in production, fetch from NSE/Screener.in
        # This would check BSE/NSE corporate actions calendar
        logger.info(f"Checking earnings calendar for {len(symbols)} symbols")
        return []  # Return symbols to avoid

    def get_daily_brief(self, market_data: dict) -> str:
        """Generate a daily market brief for notifications."""
        regime = self.detect_market_regime(market_data.get("nifty_indicators", {}))
        vix = market_data.get("india_vix", 0)
        nifty_change = market_data.get("nifty_change_pct", 0)

        return (
            f"📊 Market Brief | {datetime.now().strftime('%d-%b-%Y')}\n"
            f"Regime: {regime.replace('_', ' ').title()}\n"
            f"Nifty: {nifty_change:+.1f}% | VIX: {vix:.1f}\n"
            f"Strategy: {'Active' if regime != 'volatile' else '⚠️ Paused'}"
        )

    # =========================================================================
    # RAG-ENHANCED METHODS (New: Adaptive Layer)
    # =========================================================================

    def get_full_market_context(self, nifty_data: dict = None) -> dict:
        """
        Build comprehensive market context combining all data sources.
        This is the PRIMARY method strategies should call before trading.
        
        Returns a rich context dict with:
        - regime, VIX, sentiment, global cues, news, historical patterns
        """
        context = {}

        # 1. Regime detection
        regime = "ranging"
        if nifty_data:
            regime = self.detect_market_regime(nifty_data)
        context["regime"] = regime

        # 2. Global cues
        try:
            global_cues = self.global_cues_fetcher.fetch_global_cues()
            self._latest_global_cues = global_cues
            context["global_cues"] = {
                "sentiment": global_cues.overall_sentiment,
                "risk_score": global_cues.risk_score,
                "sp500_pct": global_cues.sp500_change_pct,
                "crude_pct": global_cues.crude_oil_change_pct,
                "dxy_pct": global_cues.dxy_change_pct,
                "usdinr": global_cues.usdinr,
                "context_string": global_cues.to_context_string(),
            }
        except Exception as e:
            logger.error(f"Global cues fetch failed: {e}")
            context["global_cues"] = {"sentiment": "neutral", "risk_score": 0.3}

        # 3. News sentiment
        try:
            news_items = self.news_feed.fetch_all_feeds(max_age_hours=4)
            news_text = self.news_feed.get_news_summary_text(max_items=10)
            high_impact = self.news_feed.get_high_impact_news(hours=2)

            if news_text:
                sentiment = self.analyze_sentiment(news_text)
            else:
                sentiment = {"overall_sentiment": 0, "trade_impact": "neutral", "key_events": []}

            self._latest_sentiment = sentiment
            context["sentiment"] = sentiment
            context["high_impact_news"] = [n.title for n in high_impact[:5]]
            context["news_count"] = len(news_items)
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            context["sentiment"] = {"overall_sentiment": 0, "trade_impact": "neutral"}

        # 4. RAG: Historical context from knowledge base
        try:
            vix = nifty_data.get("vix", 15) if nifty_data else 15
            rag_context = self.kb.build_rag_context(
                current_regime=regime,
                current_vix=vix,
                strategy="theta_selling",  # Primary strategy context
            )
            context["rag_context"] = rag_context
            context["vix"] = vix
        except Exception as e:
            logger.debug(f"RAG context unavailable: {e}")
            context["rag_context"] = "No historical context available yet."

        # 5. FII/DII data
        try:
            fii_dii = self.global_cues_fetcher.get_fii_dii_data()
            context["fii_net"] = fii_dii.get("fii_net", 0)
            context["dii_net"] = fii_dii.get("dii_net", 0)
        except Exception:
            context["fii_net"] = 0
            context["dii_net"] = 0

        # 6. Composite trading recommendation
        context["recommendation"] = self._compute_composite_recommendation(context)

        return context

    def _compute_composite_recommendation(self, context: dict) -> dict:
        """
        Compute a weighted composite recommendation from all signals.
        This is the final "should I trade?" decision helper.
        """
        scores = {
            "trade_allowed": True,
            "confidence_multiplier": 1.0,
            "reasons": [],
        }

        regime = context.get("regime", "ranging")
        sentiment = context.get("sentiment", {})
        global_cues = context.get("global_cues", {})
        risk_score = global_cues.get("risk_score", 0.3)

        # Check sentiment
        if sentiment.get("trade_impact") == "avoid_trading":
            scores["trade_allowed"] = False
            scores["reasons"].append("Negative news sentiment - avoid trading")

        # Check global risk
        if risk_score > 0.7:
            scores["confidence_multiplier"] *= 0.6
            scores["reasons"].append(f"High global risk ({risk_score:.1f})")
        elif risk_score > 0.5:
            scores["confidence_multiplier"] *= 0.8
            scores["reasons"].append(f"Elevated global risk ({risk_score:.1f})")

        # Regime adjustment
        if regime == "volatile":
            scores["confidence_multiplier"] *= 0.5
            scores["reasons"].append("Volatile regime - reduce exposure")

        # High impact news present
        if context.get("high_impact_news"):
            scores["confidence_multiplier"] *= 0.85
            scores["reasons"].append("High-impact news active")

        # FII heavy selling
        fii_net = context.get("fii_net", 0)
        if fii_net < -2000:  # FII selling > ₹2000 Cr
            scores["confidence_multiplier"] *= 0.8
            scores["reasons"].append(f"FII selling ₹{abs(fii_net):.0f} Cr")

        return scores

    def get_enhanced_daily_brief(self, market_data: dict) -> str:
        """Enhanced daily brief with global cues and news."""
        context = self.get_full_market_context(market_data.get("nifty_indicators"))

        regime = context["regime"]
        global_cues = context.get("global_cues", {})
        sentiment = context.get("sentiment", {})
        rec = context.get("recommendation", {})

        lines = [
            f"📊 <b>Market Brief</b> | {datetime.now().strftime('%d-%b-%Y %H:%M')}",
            f"{'─' * 30}",
            f"Regime: {regime.replace('_', ' ').title()}",
            f"VIX: {context.get('vix', 0):.1f}",
            f"Sentiment: {sentiment.get('trade_impact', 'neutral').upper()}",
            f"Global: {global_cues.get('sentiment', 'neutral').upper()} (Risk: {global_cues.get('risk_score', 0):.1f})",
            f"FII Net: ₹{context.get('fii_net', 0):.0f} Cr",
            f"{'─' * 30}",
            f"Confidence Multiplier: {rec.get('confidence_multiplier', 1):.0%}",
            f"Trade Allowed: {'✅' if rec.get('trade_allowed') else '🚫'}",
        ]

        if rec.get("reasons"):
            lines.append("Notes:")
            for reason in rec["reasons"]:
                lines.append(f"  • {reason}")

        if context.get("high_impact_news"):
            lines.append("\n📰 Key News:")
            for news in context["high_impact_news"][:3]:
                lines.append(f"  • {news[:60]}")

        return "\n".join(lines)
