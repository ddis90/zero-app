"""
Self-Learning Agent (Learner).
Analyzes trade performance, discovers patterns, and suggests strategy adjustments.
Implements the feedback loop: Trade → Record → Analyze → Adapt.

Learning cycles:
- Per-trade: Record outcome + context to vector DB
- Daily: Update market regime, summarize performance
- Weekly: LLM-powered pattern analysis, suggest parameter tweaks
- Monthly: Full optimization run (Optuna), A/B test evaluation
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import yaml
import numpy as np

from src.data.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class StrategyParams:
    """Mutable strategy parameters that can be adjusted by the learner."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.params = self._load_params()
        self.history: list[dict] = []  # Track all parameter changes

    def _load_params(self) -> dict:
        """Load current strategy parameters."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return {
            "options_otm_distance_pct": config["options"]["otm_distance_pct"],
            "options_vix_threshold": config["options"]["vix_threshold"],
            "options_stop_loss_pct": config["options"]["stop_loss_pct"],
            "swing_min_volume_surge": config["swing"]["min_volume_surge_multiplier"],
            "risk_max_per_trade_pct": config["risk"]["max_risk_per_trade_pct"],
            "risk_trailing_stop_pct": config["risk"]["trailing_stop_pct"],
            "min_confidence_threshold": 0.6,  # Minimum signal confidence to act
        }

    def get(self, key: str, default=None):
        return self.params.get(key, default)

    def propose_change(self, key: str, new_value: float, reason: str) -> dict:
        """
        Propose a parameter change (doesn't apply until approved).
        Returns the proposal for human review.
        """
        old_value = self.params.get(key)
        proposal = {
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "change_pct": ((new_value - old_value) / old_value * 100) if old_value else 0,
            "reason": reason,
            "proposed_at": datetime.now().isoformat(),
            "status": "pending",  # pending, approved, rejected
        }
        self.history.append(proposal)
        return proposal

    def apply_change(self, key: str, new_value: float):
        """Apply an approved parameter change."""
        old_value = self.params.get(key)
        self.params[key] = new_value
        logger.info(f"Parameter updated: {key} = {old_value} → {new_value}")

    def get_all_pending_proposals(self) -> list[dict]:
        """Get all pending parameter change proposals."""
        return [p for p in self.history if p["status"] == "pending"]


class Learner:
    """
    The self-learning agent. Analyzes performance and adapts strategies.
    
    Key principles:
    1. Never change parameters without human approval (for safety)
    2. Propose changes based on statistical evidence (not single trades)
    3. Track what worked in which market regime
    4. Gradually increase confidence as data accumulates
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config_path: str = "config/settings.yaml",
    ):
        self.kb = knowledge_base
        self.config_path = config_path
        self.strategy_params = StrategyParams(config_path)
        self._ai_client = None

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def _get_ai_client(self):
        """Lazy init LLM client for analysis."""
        if self._ai_client is None:
            from openai import OpenAI
            self._ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        return self._ai_client

    # =========================================================================
    # PER-TRADE LEARNING: Record context with every trade
    # =========================================================================

    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        strategy: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        exit_reason: str,
        signal_confidence: float,
        market_snapshot: dict,
        indicators: dict,
    ):
        """
        Record a completed trade with full context to the knowledge base.
        Called automatically after every trade closes.
        """
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        holding_days = market_snapshot.get("holding_days", 1)

        self.kb.store_trade_outcome(
            trade_id=trade_id,
            symbol=symbol,
            strategy=strategy,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            exit_reason=exit_reason,
            market_conditions=market_snapshot,
            signal_confidence=signal_confidence,
            entry_indicators=indicators,
        )

    # =========================================================================
    # DAILY LEARNING: End-of-day analysis
    # =========================================================================

    def daily_review(self, today_trades: list, market_data: dict):
        """
        End-of-day review. Stores daily snapshot and checks for patterns.
        """
        # Store market snapshot
        today = datetime.now().strftime("%Y-%m-%d")
        self.kb.store_market_snapshot(
            date=today,
            regime=market_data.get("regime", "unknown"),
            vix=market_data.get("vix", 0),
            nifty_change_pct=market_data.get("nifty_change_pct", 0),
            fii_net=market_data.get("fii_net", 0),
            global_sentiment=market_data.get("global_sentiment", "neutral"),
            news_summary=market_data.get("news_summary", ""),
        )

        # Quick pattern check: Are we in a losing streak?
        if len(today_trades) >= 2:
            losses_today = sum(1 for t in today_trades if t.get("pnl", 0) < 0)
            if losses_today >= 2:
                self.kb.store_insight(
                    insight_id=f"daily_loss_pattern_{today}",
                    insight_text=(
                        f"Multiple losses on {today}: {losses_today} losses. "
                        f"Market regime: {market_data.get('regime')}. "
                        f"VIX: {market_data.get('vix')}. "
                        f"Consider reducing size in similar conditions."
                    ),
                    category="pattern",
                    confidence=0.5,
                    source="daily_review",
                )

    # =========================================================================
    # WEEKLY LEARNING: Pattern discovery + parameter suggestions
    # =========================================================================

    def weekly_review(self) -> dict:
        """
        Weekly deep analysis using LLM.
        Discovers patterns across the week's trades and proposes adjustments.
        
        Returns: Summary dict with findings and proposals.
        """
        # Get this week's performance
        all_strategies = ["theta_selling", "momentum_swing"]
        findings = {"patterns": [], "proposals": [], "summary": ""}

        for strategy in all_strategies:
            perf = self.kb.query_strategy_performance(strategy)
            if perf["total_trades"] < 5:
                continue

            # Analyze by regime
            for regime in ["trending_up", "trending_down", "ranging", "volatile"]:
                regime_perf = self.kb.query_strategy_performance(strategy, regime=regime)
                if regime_perf["total_trades"] >= 3:
                    findings["patterns"].append({
                        "strategy": strategy,
                        "regime": regime,
                        "win_rate": regime_perf["win_rate"],
                        "trades": regime_perf["total_trades"],
                        "avg_pnl": regime_perf["avg_pnl"],
                    })

        # Use LLM to analyze patterns and suggest changes
        if findings["patterns"]:
            llm_analysis = self._llm_analyze_patterns(findings["patterns"])
            findings["llm_analysis"] = llm_analysis
            findings["proposals"] = llm_analysis.get("proposals", [])

            # Create proposals for each suggestion
            for proposal in findings["proposals"]:
                self.strategy_params.propose_change(
                    key=proposal.get("parameter", ""),
                    new_value=proposal.get("suggested_value", 0),
                    reason=proposal.get("reason", ""),
                )

        findings["summary"] = self._generate_weekly_summary(findings)
        return findings

    def _llm_analyze_patterns(self, patterns: list[dict]) -> dict:
        """Use LLM to find actionable patterns in trade data."""
        prompt = f"""You are a quantitative trading strategist reviewing weekly performance data.
Analyze these strategy performance patterns and suggest specific parameter adjustments.

Performance by strategy and market regime:
{json.dumps(patterns, indent=2)}

Current parameters:
{json.dumps(self.strategy_params.params, indent=2)}

Rules:
- Only suggest changes with clear statistical backing (win rate < 40% or > 70%)
- Keep changes small (max 20% adjustment per parameter)
- If a strategy has < 40% win rate in a regime, suggest pausing it in that regime
- If win rate > 70%, suggest slightly increasing position size
- Be conservative - this is real money

Respond in JSON:
{{
    "analysis": "<2-3 sentence summary of findings>",
    "proposals": [
        {{
            "parameter": "<exact parameter key>",
            "current_value": <number>,
            "suggested_value": <number>,
            "reason": "<one-line justification with data>"
        }}
    ],
    "regime_recommendations": {{
        "<regime>": "<strategy recommendation>"
    }}
}}"""

        try:
            client = self._get_ai_client()
            response = client.chat.completions.create(
                model=self.config["ai"]["model"],
                messages=[
                    {"role": "system", "content": "You are a conservative quantitative trading advisor. Safety first."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM pattern analysis failed: {e}")
            return {"analysis": "Analysis unavailable", "proposals": []}

    def _generate_weekly_summary(self, findings: dict) -> str:
        """Generate human-readable weekly summary."""
        lines = ["📊 Weekly Strategy Review"]
        lines.append("=" * 40)

        for pattern in findings.get("patterns", []):
            emoji = "✅" if pattern["win_rate"] > 0.6 else "⚠️" if pattern["win_rate"] > 0.4 else "❌"
            lines.append(
                f"{emoji} {pattern['strategy']} in {pattern['regime']}: "
                f"{pattern['win_rate']:.0%} win rate ({pattern['trades']} trades, "
                f"avg ₹{pattern['avg_pnl']:.0f})"
            )

        if findings.get("proposals"):
            lines.append("\n📝 Proposed Changes (need approval):")
            for p in findings["proposals"]:
                lines.append(f"  • {p.get('parameter')}: {p.get('reason', '')}")

        return "\n".join(lines)

    # =========================================================================
    # ADAPTIVE CONFIDENCE: Adjust signal confidence based on historical context
    # =========================================================================

    def adjust_confidence(
        self,
        base_confidence: float,
        strategy: str,
        current_regime: str,
        current_vix: float,
    ) -> float:
        """
        Adjust signal confidence based on historical performance in similar conditions.
        This is the core "self-learning" mechanism for real-time decisions.
        
        Returns adjusted confidence (higher if history supports, lower if not).
        """
        # Get historical win rate for this strategy + regime
        win_data = self.kb.get_win_rate_by_conditions(
            regime=current_regime,
            strategy=strategy,
        )

        if not win_data["sufficient_data"]:
            # Not enough data yet - use base confidence with slight penalty
            return base_confidence * 0.9

        historical_win_rate = win_data["win_rate"]

        # Adjust confidence based on historical success
        if historical_win_rate >= 0.7:
            # Strategy works well here - boost confidence
            adjusted = base_confidence * 1.15
        elif historical_win_rate >= 0.5:
            # Neutral - keep as is
            adjusted = base_confidence
        elif historical_win_rate >= 0.35:
            # Below average - reduce
            adjusted = base_confidence * 0.75
        else:
            # Poor performance - significantly reduce
            adjusted = base_confidence * 0.5

        # VIX penalty: Higher VIX = more uncertainty
        if current_vix > 20:
            adjusted *= 0.85
        elif current_vix > 25:
            adjusted *= 0.7

        # Clamp between 0 and 1
        return max(0.0, min(1.0, adjusted))

    def get_regime_recommendation(self, regime: str) -> dict:
        """
        Get strategy recommendations for the current regime
        based on accumulated performance data.
        """
        recommendations = {}
        for strategy in ["theta_selling", "momentum_swing"]:
            perf = self.kb.query_strategy_performance(strategy, regime=regime)
            if perf["total_trades"] >= 10:
                if perf["win_rate"] >= 0.6:
                    recommendations[strategy] = "active"
                elif perf["win_rate"] >= 0.4:
                    recommendations[strategy] = "reduced"
                else:
                    recommendations[strategy] = "paused"
            else:
                # Not enough data - default to cautious
                recommendations[strategy] = "cautious"

        return recommendations

    # =========================================================================
    # MONTHLY OPTIMIZATION
    # =========================================================================

    def monthly_optimization_report(self) -> dict:
        """
        Generate comprehensive monthly report with optimization suggestions.
        This feeds into the human review process.
        """
        report = {
            "period": datetime.now().strftime("%B %Y"),
            "generated_at": datetime.now().isoformat(),
            "knowledge_base_stats": self.kb.get_stats(),
            "strategy_performance": {},
            "regime_analysis": {},
            "top_insights": [],
            "parameter_proposals": self.strategy_params.get_all_pending_proposals(),
        }

        # Strategy performance
        for strategy in ["theta_selling", "momentum_swing"]:
            report["strategy_performance"][strategy] = self.kb.query_strategy_performance(strategy)

        # Regime analysis
        for regime in ["trending_up", "trending_down", "ranging", "volatile"]:
            report["regime_analysis"][regime] = {
                "theta": self.kb.query_strategy_performance("theta_selling", regime),
                "swing": self.kb.query_strategy_performance("momentum_swing", regime),
            }

        # Top insights
        insights = self.kb.query_insights("most important trading patterns", n_results=10)
        report["top_insights"] = [i["document"] for i in insights]

        return report
