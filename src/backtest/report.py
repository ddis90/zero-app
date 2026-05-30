"""
Backtest Performance Report Generator.
Produces analytics, visualizations, and export files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


class BacktestReport:
    """
    Generates comprehensive backtest performance reports.
    
    Outputs:
    - Console summary
    - JSON detailed report
    - CSV trade log
    - Equity curve plot (PNG)
    - Monthly returns heatmap (PNG)
    """

    def __init__(self, results: dict, output_dir: str = "reports"):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_full_report(self) -> str:
        """Generate all report artifacts and return the summary."""
        summary_text = self._print_summary()
        self._save_json_report()
        self._save_trade_log()
        self._plot_equity_curve()
        self._plot_monthly_returns()
        self._plot_capital_scaling()

        logger.info(f"Reports saved to: {self.output_dir}/")
        return summary_text

    def _print_summary(self) -> str:
        """Generate formatted console summary."""
        s = self.results.get("summary", {})
        r = self.results.get("risk_metrics", {})
        theta = self.results.get("strategies", {}).get("theta_selling", {})
        swing = self.results.get("strategies", {}).get("momentum_swing", {})

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║            WALK-FORWARD BACKTEST RESULTS                    ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"  Period:            {s.get('period', 'N/A')} ({s.get('years', 0)} years)",
            f"  Initial Capital:   ₹{s.get('initial_capital', 0):,.0f}",
            f"  Final Capital:     ₹{s.get('final_capital', 0):,.0f}",
            f"  Total Return:      {s.get('total_return_pct', 0):+.1f}%",
            f"  CAGR:              {s.get('cagr_pct', 0):.1f}%",
            f"  Total Trades:      {s.get('total_trades', 0)}",
            "",
            "  ─── Risk Metrics ───────────────────────────────────────────",
            f"  Max Drawdown:      {r.get('max_drawdown_pct', 0):.1f}%",
            f"  Avg Sharpe Ratio:  {r.get('avg_sharpe', 0):.2f}",
            f"  Worst Window:      {r.get('worst_window_return', 0):+.1f}%",
            f"  Best Window:       {r.get('best_window_return', 0):+.1f}%",
            "",
            "  ─── Theta Selling (Options) ────────────────────────────────",
            f"  Trades:            {theta.get('trades', 0)}",
            f"  Win Rate:          {theta.get('win_rate', 0):.0f}%",
            f"  Net P&L:           ₹{theta.get('net_pnl', 0):,.0f}",
            f"  Profit Factor:     {theta.get('profit_factor', 0):.2f}",
            f"  Avg Winner:        ₹{theta.get('avg_winner', 0):,.0f}",
            f"  Avg Loser:         ₹{theta.get('avg_loser', 0):,.0f}",
            "",
            "  ─── Momentum Swing ─────────────────────────────────────────",
            f"  Trades:            {swing.get('trades', 0)}",
            f"  Win Rate:          {swing.get('win_rate', 0):.0f}%",
            f"  Net P&L:           ₹{swing.get('net_pnl', 0):,.0f}",
            f"  Profit Factor:     {swing.get('profit_factor', 0):.2f}",
            f"  Avg Winner:        ₹{swing.get('avg_winner', 0):,.0f}",
            f"  Avg Loser:         ₹{swing.get('avg_loser', 0):,.0f}",
            "",
            "  ─── Capital Scaling ────────────────────────────────────────",
        ]

        scaling = self.results.get("capital_scaling", {})
        for d in scaling.get("decisions", []):
            arrow = "↑" if d["change_pct"] > 0 else "↓" if d["change_pct"] < 0 else "→"
            lines.append(
                f"  Window {d['window']}: ₹{d['from']:,.0f} {arrow} ₹{d['to']:,.0f} ({d['change_pct']:+.0f}%)"
            )

        lines.extend([
            "",
            f"  Peak Capital:      ₹{scaling.get('peak_capital', 0):,.0f}",
            f"  Min Capital:       ₹{scaling.get('min_capital', 0):,.0f}",
            "",
            "  ─── Walk-Forward Windows ───────────────────────────────────",
        ])

        for w in self.results.get("windows", []):
            if w["type"] == "test":
                status = "✅" if w["return_pct"] > 0 else "❌"
                lines.append(
                    f"  {status} W{w['id']} Test: {w['period']} | "
                    f"{w['return_pct']:+.1f}% | {w['trades']} trades | "
                    f"WR {w['win_rate']:.0f}% | Sharpe {w['sharpe']:.2f}"
                )

        lines.extend([
            "",
            "═" * 64,
            f"  Reports: {self.output_dir}/",
            "═" * 64,
            "",
        ])

        summary = "\n".join(lines)
        print(summary)
        return summary

    def _save_json_report(self):
        """Save detailed results as JSON."""
        output_path = self.output_dir / f"backtest_report_{self.timestamp}.json"

        # Convert trades to serializable format
        report_data = {
            "summary": self.results.get("summary", {}),
            "strategies": self.results.get("strategies", {}),
            "risk_metrics": self.results.get("risk_metrics", {}),
            "capital_scaling": self.results.get("capital_scaling", {}),
            "windows": self.results.get("windows", []),
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"JSON report: {output_path}")

    def _save_trade_log(self):
        """Save all trades as CSV."""
        trades = self.results.get("trades", [])
        if not trades:
            return

        output_path = self.output_dir / f"trade_log_{self.timestamp}.csv"

        rows = []
        for t in trades:
            rows.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "strategy": t.strategy,
                "direction": t.direction,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "gross_pnl": t.gross_pnl,
                "brokerage": t.brokerage,
                "net_pnl": t.net_pnl,
                "exit_reason": t.exit_reason,
                "holding_days": t.holding_days,
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Trade log: {output_path} ({len(rows)} trades)")

    def _plot_equity_curve(self):
        """Plot equity curve with drawdown."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend for server/cloud
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            output_path = self.output_dir / f"equity_curve_{self.timestamp}.png"

            # Build equity from trade results
            trades = self.results.get("trades", [])
            if not trades:
                return

            # Reconstruct equity curve from trades
            capital = self.results["summary"]["initial_capital"]
            equity_points = [{"date": trades[0].entry_date, "equity": capital}]

            for t in sorted(trades, key=lambda x: x.exit_date):
                capital += t.net_pnl
                equity_points.append({"date": t.exit_date, "equity": capital})

            eq_df = pd.DataFrame(equity_points)
            eq_df["date"] = pd.to_datetime(eq_df["date"])
            eq_df = eq_df.set_index("date")

            # Calculate drawdown
            peak = eq_df["equity"].cummax()
            drawdown = (eq_df["equity"] - peak) / peak * 100

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True)

            # Equity curve
            ax1.plot(eq_df.index, eq_df["equity"], color="#2196F3", linewidth=1.5, label="Portfolio Equity")
            ax1.axhline(y=self.results["summary"]["initial_capital"], color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
            ax1.fill_between(eq_df.index, self.results["summary"]["initial_capital"], eq_df["equity"],
                           where=eq_df["equity"] >= self.results["summary"]["initial_capital"],
                           alpha=0.1, color="green")
            ax1.fill_between(eq_df.index, self.results["summary"]["initial_capital"], eq_df["equity"],
                           where=eq_df["equity"] < self.results["summary"]["initial_capital"],
                           alpha=0.1, color="red")
            ax1.set_ylabel("Portfolio Value (₹)")
            ax1.set_title("Walk-Forward Backtest: Equity Curve")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))

            # Drawdown
            ax2.fill_between(drawdown.index, 0, drawdown, color="red", alpha=0.3)
            ax2.plot(drawdown.index, drawdown, color="red", linewidth=0.8)
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Equity curve: {output_path}")

        except ImportError:
            logger.warning("matplotlib not available — skipping equity curve plot")

    def _plot_monthly_returns(self):
        """Plot monthly returns heatmap."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            output_path = self.output_dir / f"monthly_returns_{self.timestamp}.png"

            trades = self.results.get("trades", [])
            if not trades:
                return

            # Build monthly P&L
            monthly_pnl = {}
            for t in trades:
                month_key = t.exit_date.strftime("%Y-%m")
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + t.net_pnl

            if not monthly_pnl:
                return

            # Create DataFrame for heatmap
            df = pd.DataFrame(list(monthly_pnl.items()), columns=["month", "pnl"])
            df["date"] = pd.to_datetime(df["month"] + "-01")
            df["year"] = df["date"].dt.year
            df["month_num"] = df["date"].dt.month

            pivot = df.pivot_table(index="year", columns="month_num", values="pnl", aggfunc="sum")
            pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".0f",
                cmap="RdYlGn",
                center=0,
                ax=ax,
                cbar_kws={"label": "P&L (₹)"},
            )
            ax.set_title("Monthly P&L Heatmap")
            ax.set_ylabel("Year")

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Monthly returns: {output_path}")

        except ImportError:
            logger.warning("matplotlib/seaborn not available — skipping heatmap")

    def _plot_capital_scaling(self):
        """Plot capital scaling over time."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            output_path = self.output_dir / f"capital_scaling_{self.timestamp}.png"

            decisions = self.results.get("capital_scaling", {}).get("decisions", [])
            if not decisions:
                return

            windows = [f"W{d['window']}" for d in decisions]
            capitals = [d["to"] for d in decisions]
            colors = ["green" if d["change_pct"] > 0 else "red" if d["change_pct"] < 0 else "gray" for d in decisions]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(windows, capitals, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.axhline(y=self.results["summary"]["initial_capital"], color="blue", linestyle="--", alpha=0.5, label="Initial Capital")
            ax.axhline(y=500000, color="green", linestyle=":", alpha=0.5, label="Max Capital (₹5L)")
            ax.axhline(y=150000, color="red", linestyle=":", alpha=0.5, label="Min Capital (₹1.5L)")

            ax.set_xlabel("Walk-Forward Window")
            ax.set_ylabel("Capital (₹)")
            ax.set_title("Adaptive Capital Scaling")
            ax.legend()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Capital scaling: {output_path}")

        except ImportError:
            logger.warning("matplotlib not available — skipping capital scaling plot")
