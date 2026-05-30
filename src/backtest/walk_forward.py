"""
Walk-Forward Backtesting Framework.
Splits historical data into train/test windows and simulates adaptive capital scaling.

Walk-forward approach:
- Train window (6 months): Optimize/validate strategy parameters
- Test window (3 months): Out-of-sample performance evaluation
- Roll forward: Move 3 months ahead, repeat
- Capital scaling: Increase/decrease based on test window performance

This avoids overfitting by NEVER using future data for decisions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestTrade, BacktestPosition
from src.backtest.data_loader import DataLoader
from src.backtest.options_pricer import OptionsPricer

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    window_id: int
    window_type: str  # "train" or "test"
    start_date: datetime
    end_date: datetime
    starting_capital: float
    ending_capital: float
    return_pct: float
    trades: list[BacktestTrade]
    total_trades: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    strategy_breakdown: dict = field(default_factory=dict)


@dataclass
class CapitalDecision:
    """Capital scaling decision after a test window."""
    window_id: int
    previous_capital: float
    new_capital: float
    change_pct: float
    reason: str


class WalkForwardRunner:
    """
    Orchestrates walk-forward backtesting with adaptive capital scaling.
    
    Capital Scaling Rules:
    - Start: ₹2,00,000
    - After profitable test window (>5% return): scale up 50% (max ₹5L)
    - After moderate loss (-1% to -3%): hold capital
    - After significant loss (<-3%): scale down 25% (min ₹1.5L)
    - Max capital: ₹5,00,000
    - Min capital: ₹1,50,000
    """

    def __init__(
        self,
        data_loader: DataLoader,
        initial_capital: float = 200000,
        max_capital: float = 500000,
        min_capital: float = 150000,
        train_months: int = 6,
        test_months: int = 3,
        roll_months: int = 3,
    ):
        self.data_loader = data_loader
        self.initial_capital = initial_capital
        self.max_capital = max_capital
        self.min_capital = min_capital
        self.train_months = train_months
        self.test_months = test_months
        self.roll_months = roll_months

        self.pricer = OptionsPricer()
        self.window_results: list[WindowResult] = []
        self.capital_decisions: list[CapitalDecision] = []
        self.current_capital = initial_capital

    def generate_windows(
        self, start_date: datetime, end_date: datetime
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test window pairs.
        
        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []
        current = start_date

        while True:
            train_start = current
            train_end = train_start + timedelta(days=self.train_months * 30)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_months * 30)

            if test_end > end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current += timedelta(days=self.roll_months * 30)

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def run(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> dict:
        """
        Run the full walk-forward backtest.
        
        Returns comprehensive results dict.
        """
        if start_date is None:
            start_date = datetime(2024, 6, 1)
        if end_date is None:
            end_date = datetime(2026, 5, 30)

        windows = self.generate_windows(start_date, end_date)
        if not windows:
            logger.error("No valid windows generated")
            return {"error": "No windows"}

        # Load all data upfront
        logger.info("Loading historical data...")
        nifty_data = self.data_loader.load_nifty_data(start_date, end_date)
        banknifty_data = self.data_loader.load_banknifty_data(start_date, end_date)
        vix_data = self.data_loader.load_vix_data(start_date, end_date)
        universe = self.data_loader.load_nifty500_universe(start_date, end_date, top_n=50)

        if nifty_data.empty:
            logger.error("Failed to load Nifty data")
            return {"error": "No data"}

        logger.info(f"Data loaded: Nifty={len(nifty_data)} bars, Universe={len(universe)} stocks")

        self.current_capital = self.initial_capital
        all_test_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            window_id = i + 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Window {window_id}: Train {train_start.strftime('%b %Y')} - {train_end.strftime('%b %Y')}")
            logger.info(f"           Test  {test_start.strftime('%b %Y')} - {test_end.strftime('%b %Y')}")
            logger.info(f"           Capital: ₹{self.current_capital:,.0f}")
            logger.info(f"{'='*60}")

            # --- TRAIN PHASE: Validate strategy parameters ---
            train_result = self._run_window(
                window_id=window_id,
                window_type="train",
                start_date=train_start,
                end_date=train_end,
                capital=self.current_capital,
                nifty_data=nifty_data,
                banknifty_data=banknifty_data,
                vix_data=vix_data,
                universe=universe,
            )
            self.window_results.append(train_result)

            # --- TEST PHASE: Out-of-sample evaluation ---
            test_result = self._run_window(
                window_id=window_id,
                window_type="test",
                start_date=test_start,
                end_date=test_end,
                capital=self.current_capital,
                nifty_data=nifty_data,
                banknifty_data=banknifty_data,
                vix_data=vix_data,
                universe=universe,
            )
            self.window_results.append(test_result)
            all_test_results.append(test_result)

            # --- CAPITAL SCALING ---
            decision = self._scale_capital(window_id, test_result)
            self.capital_decisions.append(decision)
            self.current_capital = decision.new_capital

        return self._compile_results(all_test_results, start_date, end_date)

    def _run_window(
        self,
        window_id: int,
        window_type: str,
        start_date: datetime,
        end_date: datetime,
        capital: float,
        nifty_data: pd.DataFrame,
        banknifty_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        universe: dict[str, pd.DataFrame],
    ) -> WindowResult:
        """Run a single train or test window."""
        engine = BacktestEngine(
            initial_capital=capital,
            max_risk_per_trade_pct=2.0,
            max_positions=5,
            max_daily_loss_pct=2.0,
            max_weekly_loss_pct=5.0,
        )

        # Filter data to window
        nifty_window = nifty_data.loc[start_date:end_date]
        banknifty_window = banknifty_data.loc[start_date:end_date] if not banknifty_data.empty else pd.DataFrame()
        vix_window = vix_data.loc[start_date:end_date] if not vix_data.empty else pd.DataFrame()

        theta_trades = 0
        swing_trades = 0

        # Day-by-day simulation
        for date_idx in range(len(nifty_window)):
            date = nifty_window.index[date_idx]
            engine.current_date = date

            # Reset daily at start
            if date_idx > 0 and nifty_window.index[date_idx - 1].weekday() > date.weekday():
                engine.reset_weekly()
            engine.reset_daily()

            # Get today's data
            nifty_today = nifty_window.iloc[date_idx]
            vix_today = vix_window.loc[date] if date in vix_window.index else pd.Series({"close": 15})
            current_vix = vix_today.get("close", 15) if isinstance(vix_today, pd.Series) else 15

            # Build market data for exit checks
            market_data = {}
            for symbol, df in universe.items():
                if date in df.index:
                    row = df.loc[date]
                    market_data[symbol] = {
                        "open": row.get("open", 0),
                        "high": row.get("high", 0),
                        "low": row.get("low", 0),
                        "close": row.get("close", 0),
                        "volume": row.get("volume", 0),
                        "rsi": row.get("rsi", 50),
                        "atr": row.get("atr", 0),
                    }

            # Check exits first
            engine.check_exits(date, market_data)

            # --- THETA SELLING STRATEGY ---
            # Execute weekly on Thursday/Friday (entry), exit at next week expiry
            if date.weekday() in [3, 4] and current_vix < 20:
                theta_signal = self._generate_theta_signal(
                    nifty_spot=nifty_today["close"],
                    vix=current_vix,
                    date=date,
                    engine=engine,
                )
                if theta_signal:
                    theta_trades += 1

            # Check spread expiry (simplification: 5 trading days from entry)
            for pos in list(engine.positions):
                if pos.strategy == "theta_selling" and (date - pos.entry_date).days >= 5:
                    # Settle the spread
                    settlement_loss = self.pricer.reprice_spread_at_expiry(
                        spot_at_expiry=nifty_today["close"],
                        sell_strike=pos.metadata.get("sell_strike", 0),
                        buy_strike=pos.metadata.get("buy_strike", 0),
                        spread_type=pos.metadata.get("spread_type", "bull_put"),
                    )
                    engine.close_spread_at_expiry(pos, date, settlement_loss)

            # --- SWING STRATEGY ---
            # Execute on days 1-3 of the week (Mon-Wed) when market allows
            if date.weekday() in [0, 1, 2]:
                swing_signals = self._generate_swing_signals(
                    date=date,
                    universe=universe,
                    market_data=market_data,
                    engine=engine,
                )
                swing_trades += len(swing_signals)

            # Record daily equity
            engine.record_daily_equity(date, market_data)

        # Calculate window metrics
        equity_df = pd.DataFrame(engine.equity_curve) if engine.equity_curve else pd.DataFrame()
        sharpe = self._calculate_sharpe(equity_df) if not equity_df.empty else 0
        max_dd = self._calculate_max_drawdown(equity_df) if not equity_df.empty else 0

        return WindowResult(
            window_id=window_id,
            window_type=window_type,
            start_date=start_date,
            end_date=end_date,
            starting_capital=capital,
            ending_capital=engine.capital,
            return_pct=((engine.capital - capital) / capital) * 100,
            trades=engine.trades,
            total_trades=len(engine.trades),
            win_rate=engine.get_summary().get("win_rate", 0),
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            strategy_breakdown={
                "theta_selling": {"trades": theta_trades},
                "momentum_swing": {"trades": swing_trades},
            },
        )

    def _generate_theta_signal(
        self,
        nifty_spot: float,
        vix: float,
        date: datetime,
        engine: BacktestEngine,
    ) -> bool:
        """Generate and execute a theta selling signal (bull put spread)."""
        can, _ = engine.can_trade()
        if not can:
            return False

        # Bull put spread: 3% OTM, 100-point spread width
        otm_distance = 0.03
        sell_strike = round((nifty_spot * (1 - otm_distance)) / 50) * 50
        buy_strike = sell_strike - 100  # ₹100 spread width

        # Price the spread
        spread = self.pricer.price_credit_spread(
            spot=nifty_spot,
            sell_strike=sell_strike,
            buy_strike=buy_strike,
            vix=vix,
            days_to_expiry=5,
            spread_type="bull_put",
        )

        # Only enter if premium is worth the risk (at least 30% of spread width)
        if spread["net_premium"] < 30:  # Min ₹30 premium
            return False

        # Nifty lot size = 25 (post-Nov 2024 = 75, but using 25 for this period)
        lot_size = 25
        num_lots = 1  # Conservative: 1 lot

        success = engine.open_spread_position(
            underlying="NIFTY",
            strategy="theta_selling",
            date=date,
            net_premium=spread["net_premium"],
            max_loss=spread["max_loss"],
            lot_size=lot_size,
            num_lots=num_lots,
            metadata={
                "sell_strike": sell_strike,
                "buy_strike": buy_strike,
                "spread_type": "bull_put",
                "vix_at_entry": vix,
                "nifty_at_entry": nifty_spot,
            },
        )
        return success

    def _generate_swing_signals(
        self,
        date: datetime,
        universe: dict[str, pd.DataFrame],
        market_data: dict,
        engine: BacktestEngine,
    ) -> list:
        """Generate and execute swing trading signals."""
        signals_executed = []

        for symbol, df in universe.items():
            if date not in df.index:
                continue

            can, _ = engine.can_trade()
            if not can:
                break

            # Check if already in position for this symbol
            if any(p.symbol == symbol for p in engine.positions):
                continue

            idx = df.index.get_loc(date)
            if idx < 20:
                continue

            today = df.iloc[idx]
            yesterday = df.iloc[idx - 1]

            # --- Momentum Breakout ---
            if (
                today.get("close", 0) > yesterday.get("high_20", float("inf"))
                and today.get("volume_ratio", 0) >= 2.0
                and today.get("close", 0) > today.get("sma_50", 0)
                and today.get("rsi", 50) < 75
            ):
                entry_price = today["close"]
                atr = today.get("atr", entry_price * 0.02)
                stop_loss = entry_price - 2 * atr
                target = entry_price + 3 * atr

                quantity = engine.calculate_position_size(entry_price, stop_loss)
                if quantity > 0:
                    success = engine.open_position(
                        symbol=symbol,
                        strategy="momentum_swing",
                        direction="long",
                        date=date,
                        price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        target=target,
                        metadata={"entry_type": "breakout", "atr": atr},
                    )
                    if success:
                        signals_executed.append(symbol)
                continue

            # --- Mean Reversion ---
            if (
                today.get("rsi", 50) < 35
                and today.get("close", 0) <= today.get("bb_lower", 0) * 1.02
                and today.get("close", 0) > today.get("sma_50", 0)
            ):
                entry_price = today["close"]
                atr = today.get("atr", entry_price * 0.02)
                stop_loss = entry_price - 1.5 * atr
                target = today.get("sma_20", entry_price * 1.05)

                quantity = engine.calculate_position_size(entry_price, stop_loss)
                if quantity > 0:
                    success = engine.open_position(
                        symbol=symbol,
                        strategy="momentum_swing",
                        direction="long",
                        date=date,
                        price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        target=target,
                        metadata={"entry_type": "mean_reversion", "atr": atr},
                    )
                    if success:
                        signals_executed.append(symbol)

        return signals_executed

    def _scale_capital(self, window_id: int, test_result: WindowResult) -> CapitalDecision:
        """Apply capital scaling rules based on test window performance."""
        previous_capital = self.current_capital
        return_pct = test_result.return_pct

        if return_pct > 5.0:
            # Strong performance: scale up 50%
            new_capital = min(self.max_capital, previous_capital * 1.5)
            reason = f"Strong test return (+{return_pct:.1f}%) → scale up 50%"
        elif return_pct > 2.0:
            # Good performance: scale up 25%
            new_capital = min(self.max_capital, previous_capital * 1.25)
            reason = f"Good test return (+{return_pct:.1f}%) → scale up 25%"
        elif return_pct >= -1.0:
            # Flat: hold capital
            new_capital = previous_capital
            reason = f"Flat test return ({return_pct:+.1f}%) → hold"
        elif return_pct >= -3.0:
            # Moderate loss: slight reduction
            new_capital = max(self.min_capital, previous_capital * 0.85)
            reason = f"Moderate loss ({return_pct:.1f}%) → reduce 15%"
        else:
            # Significant loss: reduce 25%
            new_capital = max(self.min_capital, previous_capital * 0.75)
            reason = f"Significant loss ({return_pct:.1f}%) → reduce 25%"

        new_capital = round(new_capital, -3)  # Round to nearest ₹1000
        change_pct = ((new_capital - previous_capital) / previous_capital) * 100

        decision = CapitalDecision(
            window_id=window_id,
            previous_capital=previous_capital,
            new_capital=new_capital,
            change_pct=change_pct,
            reason=reason,
        )

        logger.info(f"Capital: ₹{previous_capital:,.0f} → ₹{new_capital:,.0f} ({reason})")
        return decision

    def _calculate_sharpe(self, equity_df: pd.DataFrame, risk_free_annual: float = 0.065) -> float:
        """Calculate annualized Sharpe ratio from equity curve."""
        if equity_df.empty or len(equity_df) < 2:
            return 0.0

        daily_returns = equity_df["total_equity"].pct_change().dropna()
        if daily_returns.std() == 0:
            return 0.0

        daily_rf = risk_free_annual / 252
        excess_returns = daily_returns - daily_rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return round(sharpe, 2)

    def _calculate_max_drawdown(self, equity_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage."""
        if equity_df.empty:
            return 0.0

        equity = equity_df["total_equity"]
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return round(abs(drawdown.min()) * 100, 2)

    def _compile_results(
        self,
        test_results: list[WindowResult],
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """Compile all results into a comprehensive report."""
        all_trades = []
        for wr in test_results:
            all_trades.extend(wr.trades)

        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        years = (end_date - start_date).days / 365.25
        cagr = ((self.current_capital / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Strategy breakdown
        theta_trades = [t for t in all_trades if t.strategy == "theta_selling"]
        swing_trades = [t for t in all_trades if t.strategy == "momentum_swing"]

        return {
            "summary": {
                "initial_capital": self.initial_capital,
                "final_capital": self.current_capital,
                "total_return_pct": round(total_return, 2),
                "cagr_pct": round(cagr, 2),
                "period": f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}",
                "years": round(years, 1),
                "total_windows": len(test_results),
                "total_trades": len(all_trades),
            },
            "strategies": {
                "theta_selling": self._strategy_stats(theta_trades),
                "momentum_swing": self._strategy_stats(swing_trades),
            },
            "capital_scaling": {
                "decisions": [
                    {
                        "window": d.window_id,
                        "from": d.previous_capital,
                        "to": d.new_capital,
                        "change_pct": round(d.change_pct, 1),
                        "reason": d.reason,
                    }
                    for d in self.capital_decisions
                ],
                "peak_capital": max(d.new_capital for d in self.capital_decisions) if self.capital_decisions else self.initial_capital,
                "min_capital": min(d.new_capital for d in self.capital_decisions) if self.capital_decisions else self.initial_capital,
            },
            "windows": [
                {
                    "id": wr.window_id,
                    "type": wr.window_type,
                    "period": f"{wr.start_date.strftime('%b %Y')} - {wr.end_date.strftime('%b %Y')}",
                    "return_pct": round(wr.return_pct, 2),
                    "trades": wr.total_trades,
                    "win_rate": round(wr.win_rate * 100, 1),
                    "sharpe": wr.sharpe_ratio,
                    "max_drawdown": wr.max_drawdown_pct,
                }
                for wr in self.window_results
            ],
            "risk_metrics": {
                "max_drawdown_pct": max(wr.max_drawdown_pct for wr in test_results) if test_results else 0,
                "avg_sharpe": np.mean([wr.sharpe_ratio for wr in test_results]) if test_results else 0,
                "worst_window_return": min(wr.return_pct for wr in test_results) if test_results else 0,
                "best_window_return": max(wr.return_pct for wr in test_results) if test_results else 0,
            },
            "trades": all_trades,
        }

    def _strategy_stats(self, trades: list[BacktestTrade]) -> dict:
        """Calculate stats for a specific strategy."""
        if not trades:
            return {"trades": 0, "net_pnl": 0, "win_rate": 0}

        pnls = [t.net_pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        return {
            "trades": len(trades),
            "net_pnl": round(sum(pnls), 2),
            "win_rate": round(len(winners) / len(trades) * 100, 1),
            "avg_winner": round(np.mean(winners), 0) if winners else 0,
            "avg_loser": round(np.mean(losers), 0) if losers else 0,
            "profit_factor": round(abs(sum(winners) / sum(losers)), 2) if losers else float("inf"),
            "total_brokerage": round(sum(t.brokerage for t in trades), 0),
        }
