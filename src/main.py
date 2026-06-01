"""
Main Orchestrator - Coordinates all agents for the trading session.
Entry point for the trading system.

Enhanced with adaptive learning loop:
- Pre-market: Full market context (news + global cues + RAG)
- Intraday: Confidence-adjusted signal execution
- Post-market: Record trades, daily review, approval processing
- Weekly: Pattern analysis, strategy parameter proposals
"""

import logging
import os
import sys
import time
import threading
import signal as sig
from datetime import datetime, timedelta

import yaml
import schedule
from zoneinfo import ZoneInfo
from flask import Flask

IST = ZoneInfo("Asia/Kolkata")

from src.utils.auth import KiteAuth
from src.utils.notifier import TelegramNotifier
from src.data.fetcher import DataFetcher
from src.data.screener import StockScreener
from src.data.news_feed import NewsFeed
from src.data.global_cues import GlobalCuesFetcher
from src.data.knowledge_base import KnowledgeBase
from src.agents.risk_manager import RiskManager
from src.agents.market_analyst import MarketAnalyst
from src.agents.executor import OrderExecutor
from src.agents.learner import Learner
from src.strategies.theta_selling import ThetaSellingStrategy
from src.strategies.momentum_swing import MomentumSwingStrategy
from src.utils.headless_auth import HeadlessAuth

# Setup logging
STATE_DIR = os.getenv("STATE_DIR", ".")
os.makedirs(os.path.join(STATE_DIR, "logs") if os.getenv("AZURE_DEPLOYMENT") else "logs", exist_ok=True)
LOG_DIR = os.path.join(STATE_DIR, "logs") if os.getenv("AZURE_DEPLOYMENT") else "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "system.log"), mode="a"),
    ],
)
logger = logging.getLogger("orchestrator")


# ============================================================
# Health Check Endpoint (for Azure Container Apps probes)
# ============================================================
health_app = Flask(__name__)
health_app.logger.setLevel(logging.WARNING)

_system_healthy = True
_system_started_at = datetime.now(IST).isoformat()


@health_app.route("/healthz")
def healthz():
    if _system_healthy:
        return {"status": "healthy", "started_at": _system_started_at}, 200
    return {"status": "unhealthy"}, 503


def _run_health_server():
    """Run health check server in background thread."""
    health_app.run(host="0.0.0.0", port=8080, threaded=True)


class TradingOrchestrator:
    """
    Main trading system orchestrator.
    
    Workflow:
    1. Pre-market: Authenticate, fetch data, analyze sentiment
    2. Market open: Run strategies, generate signals
    3. During market: Monitor positions, enforce guardrails
    4. Post-market: Summarize P&L, send notifications
    """

    def __init__(self, paper_trade: bool = True):
        self.paper_trade = paper_trade
        self.running = False
        self.config = self._load_config()

        # Core components
        self.auth = KiteAuth()
        self.notifier = TelegramNotifier()
        self.risk_manager = RiskManager()

        # Adaptive learning components
        self.knowledge_base = KnowledgeBase()
        self.analyst = MarketAnalyst(knowledge_base=self.knowledge_base)
        self.learner = Learner(knowledge_base=self.knowledge_base)
        self.news_feed = NewsFeed()
        self.global_cues_fetcher = GlobalCuesFetcher()

        # Current market state (refreshed pre-market)
        self._market_context: dict = {}
        self._current_regime: str = "ranging"

        # These require authentication
        self.fetcher: DataFetcher = None
        self.screener: StockScreener = None
        self.executor: OrderExecutor = None
        self.theta_strategy: ThetaSellingStrategy = None
        self.swing_strategy: MomentumSwingStrategy = None

    def _load_config(self) -> dict:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)

    def initialize(self) -> bool:
        """Initialize all components. Must be called after authentication."""
        try:
            is_cloud = os.getenv("AZURE_DEPLOYMENT", "").lower() == "true"

            if is_cloud:
                # In cloud: use HeadlessAuth - try saved token first, then login directly
                headless = HeadlessAuth()
                token_data = headless.ensure_authenticated()
                if not token_data:
                    logger.warning("No saved token, attempting headless login directly...")
                    token_data = headless.login(max_retries=3)
                if not token_data:
                    logger.error("Cloud auth failed - headless login failed")
                    return False
                from kiteconnect import KiteConnect as KC
                kite = KC(api_key=token_data["api_key"])
                kite.set_access_token(token_data["access_token"])
                self.auth._access_token = token_data["access_token"]
                self.auth.kite = kite
                logger.info(f"Authenticated via headless auth: {token_data['user_id']}")
            else:
                # Local: use KiteAuth with login server token
                if not self.auth.load_saved_token():
                    logger.error(
                        "No valid token found. Run login server first:\n"
                        "  python -m src.utils.login_server"
                    )
                    return False
                kite = self.auth.get_kite()
                logger.info(f"Authenticated as: {self.auth.user_id}")

            # Initialize components with authenticated kite instance
            self.fetcher = DataFetcher(kite)
            self.screener = StockScreener(self.fetcher)
            self.executor = OrderExecutor(kite, self.risk_manager, paper_trade=self.paper_trade)
            self.theta_strategy = ThetaSellingStrategy(self.fetcher)
            self.swing_strategy = MomentumSwingStrategy(self.fetcher)

            self.notifier.notify_system_start()
            logger.info(f"System initialized | Mode: {'PAPER' if self.paper_trade else 'LIVE'}")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.notifier.notify_system_error(f"Init failed: {e}")
            return False

    def run_pre_market(self):
        """Pre-market routine (runs at 9:00 AM)."""
        logger.info("=" * 60)
        logger.info("PRE-MARKET ROUTINE")
        logger.info("=" * 60)

        # Reset daily counters
        self.risk_manager.reset_daily()
        if datetime.now(IST).weekday() == 0:  # Monday
            self.risk_manager.reset_weekly()

        # Build full market context (news + global + RAG)
        try:
            nifty_data = self.fetcher.get_historical_data("NIFTY 50", "NSE", "day", days=250)
            nifty_indicators = {}
            if not nifty_data.empty:
                nifty_indicators = {
                    "close": nifty_data["close"].iloc[-1],
                    "sma_20": nifty_data["close"].rolling(20).mean().iloc[-1],
                    "sma_50": nifty_data["close"].rolling(50).mean().iloc[-1],
                    "sma_200": nifty_data["close"].rolling(200).mean().iloc[-1],
                }

            # Get full context with news, global cues, and RAG
            self._market_context = self.analyst.get_full_market_context(nifty_indicators)
            self._current_regime = self._market_context.get("regime", "ranging")

            # Send enhanced daily brief
            brief = self.analyst.get_enhanced_daily_brief({"nifty_indicators": nifty_indicators})
            self.notifier.send_message(brief)

            logger.info(f"Market regime: {self._current_regime}")
            logger.info(f"Trade allowed: {self._market_context.get('recommendation', {}).get('trade_allowed', True)}")

        except Exception as e:
            logger.error(f"Pre-market data fetch failed: {e}")
            self._market_context = {"regime": "ranging", "recommendation": {"trade_allowed": True, "confidence_multiplier": 0.7}}

        # Process pending approval responses from Telegram
        self._process_approvals()

    def run_options_scan(self):
        """Scan for options theta selling opportunities (runs at 9:30 AM)."""
        logger.info("Scanning options opportunities...")

        # Check if trading is allowed by both risk manager AND market context
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info(f"Trading blocked (risk): {reason}")
            return

        rec = self._market_context.get("recommendation", {})
        if not rec.get("trade_allowed", True):
            logger.info(f"Trading blocked (market context): {rec.get('reasons', [])}")
            return

        confidence_multiplier = rec.get("confidence_multiplier", 1.0)

        try:
            # Get India VIX
            vix_quote = self.fetcher.get_ltp(["INDIA VIX"], "NSE")
            india_vix = list(vix_quote.values())[0]["last_price"] if vix_quote else 15

            # Get spot prices
            nifty_quote = self.fetcher.get_ltp(["NIFTY 50"], "NSE")
            nifty_spot = list(nifty_quote.values())[0]["last_price"] if nifty_quote else 0

            banknifty_quote = self.fetcher.get_ltp(["NIFTY BANK"], "NSE")
            banknifty_spot = list(banknifty_quote.values())[0]["last_price"] if banknifty_quote else 0

            # Get option chains (nearest weekly expiry)
            nifty_expiries = self.fetcher.get_upcoming_expiries("NIFTY")
            if nifty_expiries:
                # Find expiry 3-5 days away
                target_expiry = None
                for exp in nifty_expiries:
                    days_to_exp = (exp - datetime.now(IST).date()).days
                    if 3 <= days_to_exp <= 5:
                        target_expiry = exp
                        break
                
                if target_expiry is None and nifty_expiries:
                    target_expiry = nifty_expiries[0]

                if target_expiry:
                    nifty_chain = self.fetcher.get_option_chain(
                        "NIFTY", datetime.combine(target_expiry, datetime.min.time())
                    )
                else:
                    nifty_chain = None
            else:
                nifty_chain = None

            # Generate signals
            market_data = {
                "nifty_spot": nifty_spot,
                "banknifty_spot": banknifty_spot,
                "india_vix": india_vix,
                "nifty_chain": nifty_chain,
            }

            signals = self.theta_strategy.generate_signals(market_data)

            for signal in signals:
                logger.info(f"Signal: {signal.reason}")
                # Adjust confidence using learner + market context
                adjusted_confidence = self.learner.adjust_confidence(
                    base_confidence=signal.confidence,
                    strategy="theta_selling",
                    current_regime=self._current_regime,
                    current_vix=india_vix,
                )
                adjusted_confidence *= confidence_multiplier
                logger.info(f"  Base conf: {signal.confidence:.2f} → Adjusted: {adjusted_confidence:.2f}")

                if adjusted_confidence >= 0.6:
                    result = self.executor.execute_signal(signal)
                    if result and result.get("status") != "no_position":
                        self.notifier.notify_trade_executed({
                            "type": signal.signal_type.value,
                            "symbol": signal.symbol,
                            "price": signal.entry_price,
                            "quantity": signal.quantity,
                            "stop_loss": signal.stop_loss,
                            "target": signal.target,
                            "strategy": signal.strategy_name,
                        })

        except Exception as e:
            logger.error(f"Options scan failed: {e}")
            self.notifier.notify_system_error(f"Options scan error: {e}")

    def run_swing_scan(self):
        """Scan for swing trading opportunities (runs at 10:00 AM)."""
        logger.info("Scanning swing trading opportunities...")

        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info(f"Trading blocked (risk): {reason}")
            return

        rec = self._market_context.get("recommendation", {})
        if not rec.get("trade_allowed", True):
            logger.info(f"Trading blocked (market context): {rec.get('reasons', [])}")
            return

        confidence_multiplier = rec.get("confidence_multiplier", 1.0)

        try:
            # Get screened candidates
            candidates = self.screener.get_swing_candidates(max_stocks=20)
            if candidates.empty:
                logger.info("No swing candidates found today")
                return

            # Filter blacklisted
            valid_symbols = self.screener.filter_blacklist(candidates["symbol"].tolist())
            candidates = candidates[candidates["symbol"].isin(valid_symbols)]

            # Generate signals
            market_data = {"candidates": candidates}
            signals = self.swing_strategy.generate_signals(market_data)

            for signal in signals:
                logger.info(f"Signal: {signal.reason}")
                # Adjust confidence using learner + market context
                adjusted_confidence = self.learner.adjust_confidence(
                    base_confidence=signal.confidence,
                    strategy="momentum_swing",
                    current_regime=self._current_regime,
                    current_vix=self._market_context.get("vix", 15),
                )
                adjusted_confidence *= confidence_multiplier
                logger.info(f"  Base conf: {signal.confidence:.2f} → Adjusted: {adjusted_confidence:.2f}")

                if adjusted_confidence >= 0.65:
                    result = self.executor.execute_signal(signal)
                    if result and result.get("order_id"):
                        self.notifier.notify_trade_executed({
                            "type": signal.signal_type.value,
                            "symbol": signal.symbol,
                            "price": signal.entry_price,
                            "quantity": signal.quantity,
                            "stop_loss": signal.stop_loss,
                            "target": signal.target,
                            "strategy": signal.strategy_name,
                        })

        except Exception as e:
            logger.error(f"Swing scan failed: {e}")

    def monitor_positions(self):
        """Monitor open positions for exit conditions (runs every 5 min)."""
        if not self.risk_manager.open_positions:
            return

        for position in self.risk_manager.open_positions:
            try:
                # Get current price
                quote = self.fetcher.get_ltp([position.symbol], position.exchange)
                if not quote:
                    continue

                current_price = list(quote.values())[0]["last_price"]
                position.current_pnl = (
                    (current_price - position.entry_price) * position.quantity
                    if position.position_type == "long"
                    else (position.entry_price - current_price) * position.quantity
                )

                # Check exit conditions based on strategy
                current_data = {"ltp": current_price, "rsi": 50}  # Simplified

                if position.strategy == "momentum_swing":
                    exit_signal = self.swing_strategy.should_exit(
                        {
                            "symbol": position.symbol,
                            "entry_price": position.entry_price,
                            "entry_time": position.entry_time,
                            "stop_loss": position.stop_loss,
                            "target": position.target,
                            "quantity": position.quantity,
                        },
                        current_data,
                    )
                    if exit_signal:
                        self.executor.execute_signal(exit_signal)
                        if position.current_pnl >= 0:
                            self.notifier.notify_target_hit(position.symbol, position.current_pnl)
                        else:
                            self.notifier.notify_stop_loss_hit(position.symbol, position.current_pnl)

            except Exception as e:
                logger.error(f"Position monitor error for {position.symbol}: {e}")

    def run_post_market(self):
        """Post-market routine (runs at 3:35 PM)."""
        logger.info("=" * 60)
        logger.info("POST-MARKET SUMMARY")
        logger.info("=" * 60)

        risk_summary = self.risk_manager.get_risk_summary()
        winners = sum(1 for t in self.risk_manager.today_trades if t.pnl > 0)
        losers = sum(1 for t in self.risk_manager.today_trades if t.pnl < 0)
        theta_trades = [t for t in self.risk_manager.today_trades if t.strategy == "theta_selling"]
        theta_pnl = sum(t.pnl for t in theta_trades)

        summary = {
            "daily_pnl": risk_summary["daily_pnl"],
            "weekly_pnl": risk_summary["weekly_pnl"],
            "total_trades": len(self.risk_manager.today_trades),
            "winners": winners,
            "losers": losers,
            "theta_trades": len(theta_trades),
            "theta_pnl": theta_pnl,
            "open_positions": risk_summary["open_positions"],
            "total_capital": self.config["capital"]["total"],
            "capital_deployed": sum(
                p.entry_price * p.quantity for p in self.risk_manager.open_positions
            ),
            "status": "Paused" if risk_summary["is_paused"] else "Active",
        }

        logger.info(f"Daily P&L: ₹{summary['daily_pnl']:.0f}")
        logger.info(f"Trades: {summary['total_trades']} (W:{winners} L:{losers})")
        self.notifier.notify_daily_summary(summary)

        # --- LEARNING: Daily Review ---
        self._run_daily_learning(summary)

    def _run_daily_learning(self, summary: dict):
        """Post-market daily learning cycle."""
        try:
            today_trades = []
            for trade in self.risk_manager.today_trades:
                today_trades.append({
                    "symbol": trade.symbol,
                    "pnl": trade.pnl,
                    "strategy": trade.strategy,
                    "exit_reason": trade.exit_reason,
                })

                # Record each trade to the knowledge base
                self.learner.record_trade(
                    trade_id=trade.trade_id,
                    symbol=trade.symbol,
                    strategy=trade.strategy,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    quantity=trade.quantity,
                    exit_reason=trade.exit_reason,
                    signal_confidence=getattr(trade, "signal_confidence", 0.6),
                    market_snapshot={
                        "regime": self._current_regime,
                        "vix": self._market_context.get("vix", 15),
                        "global_sentiment": self._market_context.get("global_cues", {}).get("sentiment", "neutral"),
                        "holding_days": getattr(trade, "holding_days", 1),
                    },
                    indicators=getattr(trade, "indicators", {}),
                )

            # Daily review
            market_data = {
                "regime": self._current_regime,
                "vix": self._market_context.get("vix", 15),
                "nifty_change_pct": self._market_context.get("global_cues", {}).get("sp500_pct", 0),
                "fii_net": self._market_context.get("fii_net", 0),
                "global_sentiment": self._market_context.get("global_cues", {}).get("sentiment", "neutral"),
                "news_summary": "; ".join(self._market_context.get("high_impact_news", [])),
            }
            self.learner.daily_review(today_trades, market_data)
            logger.info("Daily learning cycle completed")

        except Exception as e:
            logger.error(f"Daily learning failed: {e}")

    def run_weekly_review(self):
        """Weekly strategy review with LLM analysis (runs Friday 15:45)."""
        logger.info("=" * 60)
        logger.info("WEEKLY STRATEGY REVIEW")
        logger.info("=" * 60)

        try:
            findings = self.learner.weekly_review()
            summary = findings.get("summary", "No summary available")
            logger.info(summary)

            # Send to Telegram
            self.notifier.notify_weekly_review(summary)

            # Send any proposals for approval
            for proposal in self.learner.strategy_params.get_all_pending_proposals():
                self.notifier.send_approval_request(proposal)
                logger.info(f"Sent approval request: {proposal['key']}")

        except Exception as e:
            logger.error(f"Weekly review failed: {e}")

    def _process_approvals(self):
        """Process any pending approval/rejection responses from Telegram."""
        try:
            responses = self.notifier.check_approval_responses()
            for resp in responses:
                action = resp["action"]
                proposal_id = resp["proposal_id"]

                # Find matching proposal
                for proposal in self.learner.strategy_params.history:
                    pid = f"{proposal.get('key', '')}_{proposal.get('proposed_at', '')}"
                    if pid == proposal_id:
                        if action == "approve":
                            self.learner.strategy_params.apply_change(
                                proposal["key"], proposal["new_value"]
                            )
                            proposal["status"] = "approved"
                            logger.info(f"Approved: {proposal['key']} = {proposal['new_value']}")
                            self.notifier.send_message(f"✅ Applied: {proposal['key']} = {proposal['new_value']}")
                        else:
                            proposal["status"] = "rejected"
                            logger.info(f"Rejected: {proposal['key']}")
                        break
        except Exception as e:
            logger.error(f"Approval processing failed: {e}")

    def start(self):
        """Start the trading system with scheduled tasks."""
        # Start health endpoint for Container Apps liveness probe
        if os.getenv("AZURE_DEPLOYMENT"):
            health_thread = threading.Thread(target=_run_health_server, daemon=True)
            health_thread.start()
            logger.info("Health endpoint started on :8080/healthz")

        if not self.initialize():
            global _system_healthy
            _system_healthy = False
            sys.exit(1)

        self.running = True

        # Schedule tasks (Theta-Only Mode)
        schedule.every().day.at("09:00").do(self.run_pre_market)
        schedule.every().day.at("09:30").do(self.run_options_scan)
        # Swing scan disabled - negative expectancy confirmed in 2-year backtest
        # schedule.every().day.at("10:00").do(self.run_swing_scan)
        schedule.every(5).minutes.do(self.monitor_positions)
        schedule.every().day.at("15:35").do(self.run_post_market)

        # Learning loop tasks
        schedule.every().friday.at("15:45").do(self.run_weekly_review)
        schedule.every(30).minutes.do(self._process_approvals)  # Check Telegram approvals

        # Handle graceful shutdown
        sig.signal(sig.SIGINT, self._shutdown)
        sig.signal(sig.SIGTERM, self._shutdown)

        logger.info("Trading system started. Waiting for scheduled tasks...")

        while self.running:
            now = datetime.now(IST).time()
            market_start = datetime.strptime("08:50", "%H:%M").time()
            market_end = datetime.strptime("15:45", "%H:%M").time()

            if market_start <= now <= market_end:
                schedule.run_pending()

            time.sleep(30)  # Check every 30 seconds

    def _shutdown(self, signum, frame):
        """Graceful shutdown handler."""
        logger.info("Shutdown signal received. Stopping...")
        self.running = False
        self.notifier.send_message("🔴 Trading Agent Stopped")


def main():
    """Entry point."""
    # Create logs directory
    os.makedirs(LOG_DIR, exist_ok=True)

    paper_trade = os.getenv("PAPER_TRADE", "true").lower() == "true"
    is_cloud = os.getenv("AZURE_DEPLOYMENT", "").lower() == "true"
    
    print("=" * 60)
    print("  Zero Trading Agent")
    print(f"  Mode: {'📝 PAPER TRADING' if paper_trade else '💰 LIVE TRADING'}")
    print(f"  Env:  {'☁️ Azure Cloud' if is_cloud else '🖥️ Local'}")
    print(f"  Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("=" * 60)

    if not paper_trade and not is_cloud:
        confirm = input("\n⚠️  LIVE TRADING MODE! Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(0)

    orchestrator = TradingOrchestrator(paper_trade=paper_trade)
    orchestrator.start()


if __name__ == "__main__":
    main()
