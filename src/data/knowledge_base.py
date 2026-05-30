"""
Vector Knowledge Base using ChromaDB.
Stores trade outcomes, market context, strategy performance,
and enables RAG (Retrieval-Augmented Generation) for context-aware decisions.

Collections:
- market_context: Daily market snapshots (regime, VIX, FII, news)
- trade_outcomes: Every trade with full context (what worked/didn't)
- strategy_insights: Learned patterns and parameter adjustments
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    ChromaDB-powered vector knowledge base for the trading agent.
    Provides semantic search over past trades, market contexts,
    and strategy performance for RAG-enhanced decision making.
    """

    def __init__(self, persist_dir: str = "data/knowledge_base"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._init_collections()

    def _init_collections(self):
        """Initialize all collections."""
        self.market_context = self.client.get_or_create_collection(
            name="market_context",
            metadata={"description": "Daily market snapshots and conditions"},
        )
        self.trade_outcomes = self.client.get_or_create_collection(
            name="trade_outcomes",
            metadata={"description": "All trade outcomes with full market context"},
        )
        self.strategy_insights = self.client.get_or_create_collection(
            name="strategy_insights",
            metadata={"description": "Learned patterns, what works and what doesn't"},
        )

    # =========================================================================
    # STORE: Record data into knowledge base
    # =========================================================================

    def store_market_snapshot(
        self,
        date: str,
        regime: str,
        vix: float,
        nifty_change_pct: float,
        fii_net: float,
        global_sentiment: str,
        news_summary: str,
        additional_context: dict = None,
    ):
        """Store a daily market conditions snapshot."""
        doc_text = (
            f"Date: {date} | Regime: {regime} | VIX: {vix:.1f} | "
            f"Nifty: {nifty_change_pct:+.1f}% | FII Net: ₹{fii_net:.0f} Cr | "
            f"Global: {global_sentiment} | News: {news_summary}"
        )

        metadata = {
            "date": date,
            "regime": regime,
            "vix": vix,
            "nifty_change_pct": nifty_change_pct,
            "fii_net": fii_net,
            "global_sentiment": global_sentiment,
            "type": "daily_snapshot",
        }
        if additional_context:
            metadata.update({k: str(v) for k, v in additional_context.items()})

        self.market_context.upsert(
            ids=[f"market_{date}"],
            documents=[doc_text],
            metadatas=[metadata],
        )
        logger.debug(f"Stored market snapshot for {date}")

    def store_trade_outcome(
        self,
        trade_id: str,
        symbol: str,
        strategy: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        holding_days: int,
        exit_reason: str,
        market_conditions: dict,
        signal_confidence: float,
        entry_indicators: dict,
    ):
        """
        Store a completed trade with all context.
        This is the core learning data.
        """
        outcome = "WIN" if pnl > 0 else "LOSS"

        doc_text = (
            f"Trade {outcome}: {symbol} via {strategy} | "
            f"Entry: ₹{entry_price:.1f} → Exit: ₹{exit_price:.1f} | "
            f"P&L: ₹{pnl:.0f} ({pnl_pct:+.1f}%) | Held: {holding_days}d | "
            f"Exit reason: {exit_reason} | "
            f"Market: Regime={market_conditions.get('regime', 'unknown')}, "
            f"VIX={market_conditions.get('vix', 0):.1f}, "
            f"FII={market_conditions.get('fii_net', 'unknown')}, "
            f"Global={market_conditions.get('global_sentiment', 'unknown')} | "
            f"Confidence: {signal_confidence:.0%} | "
            f"Indicators: RSI={entry_indicators.get('rsi', 0):.0f}, "
            f"Volume ratio={entry_indicators.get('volume_ratio', 0):.1f}"
        )

        metadata = {
            "trade_id": trade_id,
            "symbol": symbol,
            "strategy": strategy,
            "outcome": outcome,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "exit_reason": exit_reason,
            "regime": market_conditions.get("regime", "unknown"),
            "vix": market_conditions.get("vix", 0),
            "global_sentiment": market_conditions.get("global_sentiment", "unknown"),
            "signal_confidence": signal_confidence,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "type": "trade_outcome",
        }

        self.trade_outcomes.upsert(
            ids=[trade_id],
            documents=[doc_text],
            metadatas=[metadata],
        )
        logger.info(f"Stored trade outcome: {trade_id} ({outcome} ₹{pnl:.0f})")

    def store_insight(
        self,
        insight_id: str,
        insight_text: str,
        category: str,  # "pattern", "parameter_change", "regime_observation"
        confidence: float,
        source: str,  # "weekly_review", "monthly_analysis", "live_observation"
        metadata: dict = None,
    ):
        """Store a learned insight or pattern."""
        full_metadata = {
            "category": category,
            "confidence": confidence,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "type": "insight",
        }
        if metadata:
            full_metadata.update({k: str(v) for k, v in metadata.items()})

        self.strategy_insights.upsert(
            ids=[insight_id],
            documents=[insight_text],
            metadatas=[full_metadata],
        )
        logger.info(f"Stored insight: {insight_id[:30]}...")

    # =========================================================================
    # QUERY: Retrieve relevant context (RAG)
    # =========================================================================

    def query_similar_conditions(
        self,
        current_conditions: str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Find past market conditions similar to current.
        Used to inform trading decisions based on historical patterns.
        """
        results = self.market_context.query(
            query_texts=[current_conditions],
            n_results=n_results,
        )
        return self._format_results(results)

    def query_similar_trades(
        self,
        trade_context: str,
        strategy: str = None,
        n_results: int = 10,
    ) -> list[dict]:
        """
        Find past trades in similar conditions.
        Returns win/loss patterns for the given market context.
        """
        where_filter = None
        if strategy:
            where_filter = {"strategy": strategy}

        results = self.trade_outcomes.query(
            query_texts=[trade_context],
            n_results=n_results,
            where=where_filter,
        )
        return self._format_results(results)

    def query_strategy_performance(
        self,
        strategy: str,
        regime: str = None,
    ) -> dict:
        """
        Get aggregated performance stats for a strategy.
        Optionally filtered by market regime.
        """
        where_filter = {"strategy": strategy}
        if regime:
            where_filter = {"$and": [{"strategy": strategy}, {"regime": regime}]}

        results = self.trade_outcomes.get(
            where=where_filter,
            include=["metadatas"],
        )

        if not results["metadatas"]:
            return {"total_trades": 0, "win_rate": 0, "avg_pnl": 0}

        trades = results["metadatas"]
        total = len(trades)
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0,
            "avg_pnl": total_pnl / total if total > 0 else 0,
            "total_pnl": total_pnl,
            "strategy": strategy,
            "regime": regime or "all",
        }

    def query_insights(
        self,
        context: str,
        category: str = None,
        n_results: int = 5,
    ) -> list[dict]:
        """Retrieve relevant strategy insights."""
        where_filter = None
        if category:
            where_filter = {"category": category}

        results = self.strategy_insights.query(
            query_texts=[context],
            n_results=n_results,
            where=where_filter,
        )
        return self._format_results(results)

    def get_win_rate_by_conditions(
        self,
        regime: str = None,
        vix_range: tuple = None,
        strategy: str = None,
    ) -> dict:
        """
        Calculate win rate for specific market conditions.
        Helps decide whether to trade or not.
        """
        # Build filter
        conditions = []
        if strategy:
            conditions.append({"strategy": strategy})
        if regime:
            conditions.append({"regime": regime})

        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        results = self.trade_outcomes.get(
            where=where_filter,
            include=["metadatas"],
        )

        if not results["metadatas"]:
            return {"win_rate": 0.5, "sample_size": 0, "sufficient_data": False}

        trades = results["metadatas"]

        # Filter by VIX range if specified
        if vix_range:
            trades = [
                t for t in trades
                if vix_range[0] <= t.get("vix", 0) <= vix_range[1]
            ]

        total = len(trades)
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")

        return {
            "win_rate": wins / total if total > 0 else 0.5,
            "sample_size": total,
            "sufficient_data": total >= 20,  # Need at least 20 trades for significance
            "wins": wins,
            "losses": total - wins,
        }

    # =========================================================================
    # RAG Context Builder
    # =========================================================================

    def build_rag_context(
        self,
        current_regime: str,
        current_vix: float,
        strategy: str,
        symbol: str = None,
    ) -> str:
        """
        Build a comprehensive RAG context string for the LLM/strategy.
        Combines similar conditions, past trades, and insights.
        """
        context_parts = []

        # 1. Similar market conditions
        condition_query = f"Regime: {current_regime}, VIX: {current_vix:.1f}"
        similar_conditions = self.query_similar_conditions(condition_query, n_results=3)
        if similar_conditions:
            context_parts.append("## Similar Past Conditions:")
            for cond in similar_conditions:
                context_parts.append(f"  - {cond['document']}")

        # 2. Strategy performance in this regime
        perf = self.query_strategy_performance(strategy, regime=current_regime)
        if perf["total_trades"] > 0:
            context_parts.append(
                f"\n## {strategy} in {current_regime} regime: "
                f"{perf['win_rate']:.0%} win rate ({perf['total_trades']} trades, "
                f"avg P&L: ₹{perf['avg_pnl']:.0f})"
            )

        # 3. Relevant insights
        insight_query = f"{strategy} in {current_regime} market with VIX {current_vix}"
        insights = self.query_insights(insight_query, n_results=3)
        if insights:
            context_parts.append("\n## Learned Insights:")
            for ins in insights:
                context_parts.append(f"  - {ins['document']}")

        # 4. Recent trade outcomes in similar conditions
        trade_query = f"{strategy} trade when regime={current_regime} VIX={current_vix:.0f}"
        recent_trades = self.query_similar_trades(trade_query, strategy=strategy, n_results=5)
        if recent_trades:
            wins = sum(1 for t in recent_trades if t.get("metadata", {}).get("outcome") == "WIN")
            context_parts.append(
                f"\n## Recent similar trades: {wins}/{len(recent_trades)} winners"
            )

        return "\n".join(context_parts) if context_parts else "No historical context available yet."

    # =========================================================================
    # Helpers
    # =========================================================================

    def _format_results(self, results: dict) -> list[dict]:
        """Format ChromaDB query results into a clean list."""
        formatted = []
        if not results or not results.get("documents"):
            return formatted

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            formatted.append({
                "document": doc,
                "metadata": meta,
                "relevance_score": 1 - dist,  # Convert distance to similarity
            })
        return formatted

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        return {
            "market_snapshots": self.market_context.count(),
            "trade_outcomes": self.trade_outcomes.count(),
            "strategy_insights": self.strategy_insights.count(),
        }
