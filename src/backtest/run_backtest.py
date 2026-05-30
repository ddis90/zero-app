"""
Backtest CLI Entry Point.
Run walk-forward backtest with configurable parameters.

Usage:
    python -m src.backtest.run_backtest --capital 200000 --years 2 --walk-forward
    python -m src.backtest.run_backtest --capital 500000 --months 6
    python -m src.backtest.run_backtest --report-only
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtest.data_loader import DataLoader
from src.backtest.walk_forward import WalkForwardRunner
from src.backtest.report import BacktestReport

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/backtest.log", mode="a"),
    ],
)
logger = logging.getLogger("backtest")


def run_backtest(
    initial_capital: float = 200000,
    years: float = 2,
    months: int = None,
    walk_forward: bool = True,
    max_capital: float = 500000,
    min_capital: float = 150000,
    train_months: int = 6,
    test_months: int = 3,
    use_kite: bool = False,
):
    """
    Main backtest execution function.
    
    Args:
        initial_capital: Starting capital in ₹
        years: Number of years to backtest
        months: Override years with specific months
        walk_forward: Use walk-forward approach (True) or static (False)
        max_capital: Maximum capital after scaling
        min_capital: Minimum capital after scaling
        train_months: Training window length
        test_months: Test window length
        use_kite: Use Kite Connect for data (requires auth)
    """
    print("=" * 64)
    print("  Zero Trading Agent — Walk-Forward Backtest")
    print(f"  Capital: ₹{initial_capital:,.0f} → Max ₹{max_capital:,.0f}")
    print(f"  Mode: {'Walk-Forward' if walk_forward else 'Static'}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    print()

    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/historical", exist_ok=True)

    # Calculate date range
    end_date = datetime.now()
    if months:
        start_date = end_date - timedelta(days=months * 30)
    else:
        start_date = end_date - timedelta(days=int(years * 365.25))

    logger.info(f"Backtest period: {start_date.strftime('%b %Y')} → {end_date.strftime('%b %Y')}")

    # Initialize data loader
    kite = None
    if use_kite:
        try:
            from src.utils.auth import KiteAuth
            auth = KiteAuth()
            if auth.load_saved_token():
                kite = auth.get_kite()
                logger.info("Using Kite Connect for historical data")
            else:
                logger.warning("No Kite token found. Falling back to yfinance.")
        except Exception as e:
            logger.warning(f"Kite auth failed: {e}. Using yfinance.")

    data_loader = DataLoader(kite=kite)

    # Show cache status
    cache_stats = data_loader.get_cache_stats()
    if cache_stats["cached_files"] > 0:
        logger.info(f"Cache: {cache_stats['cached_files']} files, {cache_stats['total_size_mb']:.1f} MB")
    else:
        logger.info("No cached data. Will download from source (this may take a few minutes).")

    # Run walk-forward backtest
    if walk_forward:
        runner = WalkForwardRunner(
            data_loader=data_loader,
            initial_capital=initial_capital,
            max_capital=max_capital,
            min_capital=min_capital,
            train_months=train_months,
            test_months=test_months,
            roll_months=test_months,  # Roll by test window size
        )
        results = runner.run(start_date=start_date, end_date=end_date)
    else:
        # Simple static backtest (single window)
        runner = WalkForwardRunner(
            data_loader=data_loader,
            initial_capital=initial_capital,
            max_capital=max_capital,
            min_capital=min_capital,
            train_months=0,
            test_months=int((end_date - start_date).days / 30),
        )
        results = runner.run(start_date=start_date, end_date=end_date)

    if "error" in results:
        logger.error(f"Backtest failed: {results['error']}")
        print(f"\n❌ Backtest failed: {results['error']}")
        print("  Ensure you have data available (run with --use-kite for live data, or check yfinance connectivity)")
        sys.exit(1)

    # Generate report
    report = BacktestReport(results)
    summary = report.generate_full_report()

    # Final verdict
    total_return = results["summary"].get("total_return_pct", 0)
    cagr = results["summary"].get("cagr_pct", 0)
    max_dd = results["risk_metrics"].get("max_drawdown_pct", 0)

    print("\n" + "=" * 64)
    if total_return > 0 and max_dd < 20:
        print("  ✅ VERDICT: Strategy shows positive expectancy")
        print(f"     CAGR {cagr:.1f}% with max drawdown {max_dd:.1f}%")
        if cagr > 15:
            print("     → RECOMMENDED for live deployment with ₹2L start")
        else:
            print("     → Consider parameter tuning before live deployment")
    elif total_return > 0 and max_dd >= 20:
        print("  ⚠️ VERDICT: Profitable but high risk")
        print(f"     CAGR {cagr:.1f}% but drawdown {max_dd:.1f}% is concerning")
        print("     → Reduce position sizes or tighten stops before live")
    else:
        print("  ❌ VERDICT: Strategy needs improvement")
        print(f"     Return {total_return:+.1f}% over the period")
        print("     → Do NOT deploy live. Review strategy logic.")
    print("=" * 64)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Zero Trading Agent — Walk-Forward Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.backtest.run_backtest --capital 200000 --years 2
  python -m src.backtest.run_backtest --capital 200000 --months 6 --no-walk-forward
  python -m src.backtest.run_backtest --capital 500000 --use-kite
        """,
    )
    parser.add_argument("--capital", type=float, default=200000, help="Initial capital in ₹ (default: 200000)")
    parser.add_argument("--years", type=float, default=2, help="Years to backtest (default: 2)")
    parser.add_argument("--months", type=int, default=None, help="Override years with months")
    parser.add_argument("--walk-forward", action="store_true", default=True, help="Use walk-forward (default: True)")
    parser.add_argument("--no-walk-forward", action="store_true", help="Disable walk-forward (single window)")
    parser.add_argument("--max-capital", type=float, default=500000, help="Max capital after scaling (default: 500000)")
    parser.add_argument("--min-capital", type=float, default=150000, help="Min capital after scaling (default: 150000)")
    parser.add_argument("--train-months", type=int, default=6, help="Train window months (default: 6)")
    parser.add_argument("--test-months", type=int, default=3, help="Test window months (default: 3)")
    parser.add_argument("--use-kite", action="store_true", help="Use Kite Connect for data (requires valid token)")
    parser.add_argument("--report-only", action="store_true", help="Only regenerate report from last run")

    args = parser.parse_args()

    if args.report_only:
        print("Report-only mode not yet implemented. Run full backtest.")
        sys.exit(0)

    walk_forward = not args.no_walk_forward

    run_backtest(
        initial_capital=args.capital,
        years=args.years,
        months=args.months,
        walk_forward=walk_forward,
        max_capital=args.max_capital,
        min_capital=args.min_capital,
        train_months=args.train_months,
        test_months=args.test_months,
        use_kite=args.use_kite,
    )


if __name__ == "__main__":
    main()
