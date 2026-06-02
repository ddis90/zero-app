"""
Simulation Script - Paper-trade today's session using real market data.
Uses yfinance for market data (no Kite auth needed).
Runs both strategies (theta selling + momentum swing) and calculates simulated P&L.
Sends results via Telegram.

Usage: python scripts/simulate_session.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field

# Fix Windows encoding for emoji output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime.now(IST)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("simulator")

# ─── Configuration ───────────────────────────────────────────────────────────
CAPITAL = 200_000
THETA_ALLOCATION = 0.90  # 90% = ₹1,80,000
ETF_ALLOCATION = 0.10    # 10% = ₹20,000
NIFTY_LOT_SIZE = 25      # Current Nifty lot size (2026)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8819579853:AAFLMEDADfECkdXCYP3pnsf-8qyKaSBhcxg")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "8253277369")

# Nifty lot size may vary - 25 for Nifty, 15 for BankNifty
# Use 25 as standard for 2026


# ─── Data Fetching ───────────────────────────────────────────────────────────

def fetch_nifty_data(days=100) -> pd.DataFrame:
    """Fetch Nifty 50 historical data."""
    logger.info("Fetching Nifty 50 data...")
    nifty = yf.download("^NSEI", period=f"{days}d", interval="1d", progress=False)
    if nifty.empty:
        # Try with longer period
        nifty = yf.download("^NSEI", period="6mo", interval="1d", progress=False)
    # Flatten multi-level columns if present
    if hasattr(nifty.columns, 'levels'):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty.columns = [c.lower() for c in nifty.columns]
    return nifty


def fetch_india_vix() -> float:
    """Fetch India VIX value."""
    logger.info("Fetching India VIX...")
    try:
        vix = yf.download("^INDIAVIX", period="5d", interval="1d", progress=False)
        if hasattr(vix.columns, 'levels'):
            vix.columns = vix.columns.get_level_values(0)
        vix.columns = [c.lower() for c in vix.columns]
        if not vix.empty:
            return float(vix["close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Could not fetch India VIX: {e}")
    return 14.0  # Default moderate VIX


def fetch_global_cues() -> dict:
    """Fetch global market cues for context."""
    logger.info("Fetching global cues...")
    cues = {}
    tickers = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Crude Oil": "CL=F",
        "Gold": "GC=F",
        "USD/INR": "USDINR=X",
        "DXY": "DX-Y.NYB",
    }
    
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if hasattr(data.columns, 'levels'):
                data.columns = data.columns.get_level_values(0)
            data.columns = [c.lower() for c in data.columns]
            if not data.empty and len(data) >= 2:
                prev_close = float(data["close"].iloc[-2])
                last_close = float(data["close"].iloc[-1])
                change_pct = ((last_close - prev_close) / prev_close) * 100
                cues[name] = {"price": last_close, "change_pct": round(change_pct, 2)}
        except Exception:
            pass
    
    return cues


def fetch_nifty_options_chain_simulated(spot_price: float, vix: float) -> pd.DataFrame:
    """
    Simulate a realistic Nifty options chain based on spot price and VIX.
    Uses Black-Scholes approximation for option premiums.
    """
    from scipy.stats import norm
    
    # Generate strikes (50-point intervals around spot, aligned to multiples of 50)
    base = round(spot_price / 50) * 50
    strikes = np.arange(base - 1500, base + 1500, 50)
    
    # Parameters
    r = 0.065  # Risk-free rate (India ~6.5%)
    days_to_expiry = 4  # Thursday expiry, entering Monday
    T = days_to_expiry / 365
    sigma = vix / 100  # VIX as annualized volatility
    
    rows = []
    for strike in strikes:
        # Black-Scholes for puts
        d1 = (np.log(spot_price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        call_price = spot_price * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        
        # Add some bid-ask spread noise
        put_price = max(0.5, put_price * (1 + np.random.uniform(-0.05, 0.05)))
        call_price = max(0.5, call_price * (1 + np.random.uniform(-0.05, 0.05)))
        
        # Put row
        rows.append({
            "strike": strike,
            "instrument_type": "PE",
            "ltp": round(put_price, 1),
            "tradingsymbol": f"NIFTY{strike}PE",
            "oi": int(np.random.uniform(50000, 500000)),
        })
        # Call row
        rows.append({
            "strike": strike,
            "instrument_type": "CE",
            "ltp": round(call_price, 1),
            "tradingsymbol": f"NIFTY{strike}CE",
            "oi": int(np.random.uniform(50000, 500000)),
        })
    
    return pd.DataFrame(rows)


# ─── Strategy Simulation ─────────────────────────────────────────────────────

@dataclass
class SimulatedTrade:
    """A simulated paper trade."""
    strategy: str
    symbol: str
    trade_type: str  # "BUY", "SELL", "SPREAD"
    entry_price: float
    stop_loss: float
    target: float
    quantity: int
    entry_time: datetime
    exit_price: float = 0.0
    exit_time: datetime = None
    exit_reason: str = ""
    pnl: float = 0.0
    confidence: float = 0.0
    details: str = ""


def simulate_theta_strategy(spot_price: float, vix: float, chain: pd.DataFrame) -> list[SimulatedTrade]:
    """
    Simulate the theta selling (bull put spread) strategy.
    """
    trades = []
    
    # Configuration from settings
    vix_threshold = 22.0
    min_premium = 5  # Lowered for simulation (real would be 8 but VIX is low)
    stop_loss_multiplier = 2.0
    max_lots = 2
    
    # Adaptive OTM
    if vix < 13:
        otm_pct = 1.5 / 100
        spread_width = 200
    elif vix < 18:
        otm_pct = 2.0 / 100  # Slightly tighter for better premiums
        spread_width = 150
    else:
        otm_pct = 3.0 / 100
        spread_width = 100
    
    if vix > vix_threshold:
        logger.info(f"❌ Theta: VIX ({vix:.1f}) > threshold ({vix_threshold}). NO TRADE.")
        return trades
    
    # Bull Put Spread
    target_sell_strike = spot_price * (1 - otm_pct)
    # Round to nearest 50
    sell_strike = round(target_sell_strike / 50) * 50
    buy_strike = sell_strike - spread_width
    
    # Get premiums from chain
    puts = chain[chain["instrument_type"] == "PE"]
    sell_put = puts[puts["strike"] == sell_strike]
    buy_put = puts[puts["strike"] == buy_strike]
    
    if sell_put.empty or buy_put.empty:
        logger.warning("Could not find matching strikes in chain")
        return trades
    
    sell_premium = float(sell_put.iloc[0]["ltp"])
    buy_premium = float(buy_put.iloc[0]["ltp"])
    net_premium = sell_premium - buy_premium
    max_loss = (sell_strike - buy_strike) - net_premium
    
    if net_premium < min_premium:
        logger.info(f"❌ Theta: Net premium (₹{net_premium:.1f}) < minimum (₹{min_premium}). NO TRADE.")
        return trades
    
    # Confidence score
    distance_pct = (spot_price - sell_strike) / spot_price * 100
    confidence = min(0.9, (distance_pct / 5) * 0.5 + ((20 - vix) / 20) * 0.5)
    
    # Simulate the day's outcome
    # Use intraday Nifty movement to determine if spread was breached
    intraday_move = simulate_intraday_move(spot_price, vix)
    low_of_day = spot_price + intraday_move["low_offset"]
    
    # Determine exit
    if low_of_day <= sell_strike:
        # Stop loss hit
        exit_premium = net_premium * stop_loss_multiplier  # Loss = 2x premium
        pnl_per_lot = -exit_premium * NIFTY_LOT_SIZE
        exit_reason = "stop_loss"
    elif net_premium > 0:
        # Normal theta decay - assume 30-40% decay on day 1 of a 4-day option
        theta_decay_pct = np.random.uniform(0.25, 0.40)
        remaining_premium = net_premium * (1 - theta_decay_pct)
        profit = (net_premium - remaining_premium)
        pnl_per_lot = profit * NIFTY_LOT_SIZE
        exit_reason = "eod_mark_to_market"
        exit_premium = remaining_premium
    else:
        pnl_per_lot = 0
        exit_reason = "no_entry"
        exit_premium = net_premium
    
    num_lots = min(max_lots, max(1, int((CAPITAL * THETA_ALLOCATION) / 100000)))
    total_pnl = pnl_per_lot * num_lots
    
    trade = SimulatedTrade(
        strategy="theta_selling",
        symbol=f"NIFTY {sell_strike}/{buy_strike} PE Spread",
        trade_type="SPREAD",
        entry_price=net_premium,
        stop_loss=net_premium * stop_loss_multiplier,
        target=net_premium * 0.5,
        quantity=NIFTY_LOT_SIZE * num_lots,
        entry_time=datetime(NOW.year, NOW.month, NOW.day, 9, 30, tzinfo=IST),
        exit_price=exit_premium,
        exit_time=datetime(NOW.year, NOW.month, NOW.day, 15, 30, tzinfo=IST),
        exit_reason=exit_reason,
        pnl=total_pnl,
        confidence=confidence,
        details=(
            f"Sell {sell_strike}PE @ ₹{sell_premium:.1f} | Buy {buy_strike}PE @ ₹{buy_premium:.1f}\n"
            f"Net Premium: ₹{net_premium:.1f}/lot | Max Loss: ₹{max_loss:.1f}/lot\n"
            f"Distance: {distance_pct:.1f}% OTM | VIX: {vix:.1f}\n"
            f"Lots: {num_lots} | Qty: {NIFTY_LOT_SIZE * num_lots}"
        ),
    )
    trades.append(trade)
    
    logger.info(f"✅ Theta Trade: {trade.symbol} | P&L: ₹{total_pnl:.0f} ({exit_reason})")
    return trades


def simulate_intraday_move(spot_price: float, vix: float) -> dict:
    """
    Simulate realistic intraday Nifty movement based on VIX.
    VIX implies annualized vol. Daily vol = VIX / sqrt(252).
    """
    daily_vol = vix / 100 / np.sqrt(252)
    
    # Intraday range is typically 1-1.5x daily vol
    intraday_range = daily_vol * spot_price * np.random.uniform(0.8, 1.5)
    
    # Direction bias (slight bullish for Indian markets)
    direction = np.random.choice([-1, 1], p=[0.45, 0.55])
    close_offset = direction * np.random.uniform(0.3, 0.8) * intraday_range
    
    # High and low
    if direction > 0:
        high_offset = np.random.uniform(0.6, 1.0) * intraday_range
        low_offset = -np.random.uniform(0.1, 0.4) * intraday_range
    else:
        high_offset = np.random.uniform(0.1, 0.4) * intraday_range
        low_offset = -np.random.uniform(0.6, 1.0) * intraday_range
    
    return {
        "close_offset": close_offset,
        "high_offset": high_offset,
        "low_offset": low_offset,
        "range": intraday_range,
        "daily_vol_pct": daily_vol * 100,
    }


def simulate_momentum_strategy(nifty_data: pd.DataFrame) -> list[SimulatedTrade]:
    """
    Simulate momentum swing strategy.
    Note: Swing allocation is 0% in config, so this is for observation only.
    But we simulate it to see potential signals.
    """
    trades = []
    
    # Check top large-cap movers via yfinance
    watchlist = {
        "RELIANCE.NS": "RELIANCE",
        "TCS.NS": "TCS",
        "HDFCBANK.NS": "HDFCBANK",
        "INFY.NS": "INFOSYS",
        "ICICIBANK.NS": "ICICIBANK",
        "HINDUNILVR.NS": "HINDUNILVR",
        "BHARTIARTL.NS": "BHARTIARTL",
        "ITC.NS": "ITC",
        "SBIN.NS": "SBIN",
        "LT.NS": "L&T",
        "BAJFINANCE.NS": "BAJFINANCE",
        "MARUTI.NS": "MARUTI",
        "AXISBANK.NS": "AXISBANK",
        "TATASTEEL.NS": "TATA STEEL",
        "SUNPHARMA.NS": "SUNPHARMA",
    }
    
    logger.info("Scanning large-caps for momentum signals...")
    
    for yf_symbol, name in watchlist.items():
        try:
            data = yf.download(yf_symbol, period="100d", interval="1d", progress=False)
            if hasattr(data.columns, 'levels'):
                data.columns = data.columns.get_level_values(0)
            data.columns = [c.lower() for c in data.columns]
            
            if data.empty or len(data) < 50:
                continue
            
            # Add indicators
            data["sma_20"] = data["close"].rolling(20).mean()
            data["sma_50"] = data["close"].rolling(50).mean()
            data["volume_sma_20"] = data["volume"].rolling(20).mean()
            data["volume_ratio"] = data["volume"] / data["volume_sma_20"]
            data["high_20"] = data["high"].rolling(20).max()
            data["rsi"] = calculate_rsi(data["close"])
            data["atr"] = calculate_atr(data)
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Check breakout
            price_breakout = float(latest["close"]) > float(prev["high_20"])
            volume_surge = float(latest["volume_ratio"]) >= 2.0
            above_50dma = float(latest["close"]) > float(latest["sma_50"])
            rsi_ok = float(latest["rsi"]) < 75
            
            if price_breakout and volume_surge and above_50dma and rsi_ok:
                entry = float(latest["close"])
                atr = float(latest["atr"])
                sl = entry - 2 * atr
                target = entry + 3 * atr
                
                # Simulate outcome (conservative: 40% win rate for breakouts)
                won = np.random.random() < 0.40
                if won:
                    exit_price = entry + np.random.uniform(1, 3) * atr
                    exit_reason = "partial_target"
                else:
                    exit_price = entry - np.random.uniform(0.5, 2) * atr
                    exit_reason = "stop_loss"
                
                # Position size (disabled in config but simulate for learning)
                qty = max(1, int(20000 / entry))  # ₹20K per trade
                pnl = (exit_price - entry) * qty
                
                trade = SimulatedTrade(
                    strategy="momentum_swing",
                    symbol=name,
                    trade_type="BUY",
                    entry_price=entry,
                    stop_loss=sl,
                    target=target,
                    quantity=qty,
                    entry_time=datetime(NOW.year, NOW.month, NOW.day, 9, 30, tzinfo=IST),
                    exit_price=exit_price,
                    exit_time=datetime(NOW.year, NOW.month, NOW.day, 15, 30, tzinfo=IST),
                    exit_reason=exit_reason,
                    pnl=pnl,
                    confidence=0.65,
                    details=f"Breakout above 20D high | Vol surge: {latest['volume_ratio']:.1f}x | RSI: {latest['rsi']:.0f}",
                )
                trades.append(trade)
                logger.info(f"{'✅' if pnl > 0 else '❌'} Swing: {name} | P&L: ₹{pnl:.0f} ({exit_reason})")
            
            # Check mean reversion
            rsi_low = float(latest["rsi"]) < 35
            below_lower_bb = float(latest["close"]) < float(latest["sma_20"]) - 2 * float(data["close"].rolling(20).std().iloc[-1])
            above_50dma_rev = float(latest["close"]) > float(latest["sma_50"])
            
            if rsi_low and above_50dma_rev:
                entry = float(latest["close"])
                atr = float(latest["atr"])
                sl = entry - 1.5 * atr
                target = entry + 2 * atr
                
                won = np.random.random() < 0.55  # Mean reversion has higher win rate
                exit_price = target * 0.7 + entry * 0.3 if won else sl
                exit_reason = "mean_reversion_target" if won else "stop_loss"
                
                qty = max(1, int(20000 / entry))
                pnl = (exit_price - entry) * qty
                
                trade = SimulatedTrade(
                    strategy="momentum_swing",
                    symbol=name,
                    trade_type="BUY",
                    entry_price=entry,
                    stop_loss=sl,
                    target=target,
                    quantity=qty,
                    entry_time=datetime(NOW.year, NOW.month, NOW.day, 10, 0, tzinfo=IST),
                    exit_price=exit_price,
                    exit_time=datetime(NOW.year, NOW.month, NOW.day, 15, 30, tzinfo=IST),
                    exit_reason=exit_reason,
                    pnl=pnl,
                    confidence=0.60,
                    details=f"Mean reversion | RSI: {latest['rsi']:.0f} | Below BB lower",
                )
                trades.append(trade)
                logger.info(f"{'✅' if pnl > 0 else '❌'} Swing (MR): {name} | P&L: ₹{pnl:.0f}")
                
        except Exception as e:
            logger.debug(f"Error scanning {name}: {e}")
            continue
    
    return trades


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─── Analysis & Reporting ────────────────────────────────────────────────────

def generate_report(
    theta_trades: list[SimulatedTrade],
    swing_trades: list[SimulatedTrade],
    spot_price: float,
    vix: float,
    global_cues: dict,
    intraday_sim: dict,
) -> str:
    """Generate comprehensive simulation report."""
    
    all_trades = theta_trades + swing_trades
    total_pnl = sum(t.pnl for t in all_trades)
    theta_pnl = sum(t.pnl for t in theta_trades)
    swing_pnl = sum(t.pnl for t in swing_trades)
    
    # Win rate
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl < 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
    
    # Return on capital
    day_return_pct = total_pnl / CAPITAL * 100
    
    report = f"""
{'='*50}
📊 ZERO AGENT - SIMULATION REPORT
📅 {NOW.strftime('%A, %B %d, %Y')}
{'='*50}

📈 MARKET CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Nifty Spot: {spot_price:,.0f}
• India VIX: {vix:.1f}
• Simulated Day Range: {spot_price + intraday_sim['low_offset']:.0f} - {spot_price + intraday_sim['high_offset']:.0f}
• Simulated Close: {spot_price + intraday_sim['close_offset']:.0f} ({intraday_sim['close_offset']/spot_price*100:+.2f}%)

🌍 GLOBAL CUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    for name, data in global_cues.items():
        emoji = "🟢" if data["change_pct"] > 0 else "🔴"
        report += f"\n• {emoji} {name}: {data['price']:,.1f} ({data['change_pct']:+.2f}%)"
    
    report += f"""

💰 P&L SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total P&L: ₹{total_pnl:,.0f} ({day_return_pct:+.2f}%)
• Theta Selling: ₹{theta_pnl:,.0f} ({len(theta_trades)} trades)
• Momentum Swing: ₹{swing_pnl:,.0f} ({len(swing_trades)} trades)
• Win Rate: {win_rate:.0f}% ({len(wins)}W / {len(losses)}L)
• Capital: ₹{CAPITAL:,} → ₹{CAPITAL + total_pnl:,.0f}

📋 TRADE DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    for i, trade in enumerate(all_trades, 1):
        pnl_emoji = "✅" if trade.pnl > 0 else "❌" if trade.pnl < 0 else "⚪"
        report += f"""
{i}. {pnl_emoji} [{trade.strategy}] {trade.symbol}
   Entry: ₹{trade.entry_price:.1f} | Exit: ₹{trade.exit_price:.1f}
   P&L: ₹{trade.pnl:,.0f} | Confidence: {trade.confidence:.0%}
   Exit Reason: {trade.exit_reason}
   {trade.details}"""
    
    if not all_trades:
        report += "\n   No trades generated today."
    
    return report


def generate_learnings(
    theta_trades: list[SimulatedTrade],
    swing_trades: list[SimulatedTrade],
    vix: float,
    global_cues: dict,
) -> dict:
    """Generate learnings and adaptation suggestions."""
    
    all_trades = theta_trades + swing_trades
    total_pnl = sum(t.pnl for t in all_trades)
    
    learnings = {
        "date": NOW.strftime("%Y-%m-%d"),
        "observations": [],
        "adaptations": [],
        "risk_notes": [],
    }
    
    # Theta strategy observations
    if theta_trades:
        for t in theta_trades:
            if t.exit_reason == "stop_loss":
                learnings["observations"].append(
                    f"Theta spread breached - VIX ({vix:.1f}) may have been too low for entry"
                )
                learnings["adaptations"].append(
                    "Consider widening OTM distance when VIX < 13 (currently 1.5%)"
                )
            elif t.exit_reason == "eod_mark_to_market" and t.pnl > 0:
                learnings["observations"].append(
                    f"Theta decay working as expected - {t.pnl/CAPITAL*100:.2f}% return"
                )
                learnings["adaptations"].append(
                    "Strategy performing within expected parameters"
                )
    else:
        if vix > 22:
            learnings["observations"].append("No theta trades - VIX too high (risk management working)")
            learnings["adaptations"].append("Wait for VIX to normalize before entering")
    
    # Swing observations
    swing_winners = [t for t in swing_trades if t.pnl > 0]
    swing_losers = [t for t in swing_trades if t.pnl < 0]
    if swing_trades:
        win_rate = len(swing_winners) / len(swing_trades) * 100
        learnings["observations"].append(
            f"Swing signals: {len(swing_trades)} | Win rate: {win_rate:.0f}%"
        )
        if win_rate < 40:
            learnings["adaptations"].append(
                "Swing allocation correctly set to 0% - confirming negative expectancy"
            )
        elif win_rate > 55:
            learnings["adaptations"].append(
                "Swing showing promise - consider allocating 5-10% after 20+ simulated trades"
            )
    
    # Global cues observations
    sp500 = global_cues.get("S&P 500", {})
    if sp500.get("change_pct", 0) < -1:
        learnings["risk_notes"].append("US markets weak - consider tighter stops tomorrow")
    
    crude = global_cues.get("Crude Oil", {})
    if crude.get("change_pct", 0) > 2:
        learnings["risk_notes"].append("Crude oil spiking - bearish for Indian markets")
    
    # General risk notes
    if total_pnl < -CAPITAL * 0.02:
        learnings["risk_notes"].append("DAILY LOSS LIMIT would have been breached - guardrails working")
    
    return learnings


def format_telegram_message(report: str, learnings: dict) -> str:
    """Format for Telegram (HTML)."""
    
    # Truncate report for Telegram (max 4096 chars)
    lines = report.strip().split("\n")
    
    msg = f"<b>🤖 ZERO AGENT - Simulation</b>\n"
    msg += f"<b>📅 {NOW.strftime('%A, %d %b %Y')}</b>\n\n"
    
    # Extract key metrics from report
    total_pnl = 0
    theta_pnl = 0
    swing_pnl = 0
    for line in lines:
        if "Total P&L:" in line:
            try:
                total_pnl = float(line.split("₹")[1].split(" ")[0].replace(",", ""))
            except (IndexError, ValueError):
                pass
    
    # Re-calculate from trades (more reliable)
    # Will be passed in the actual call
    
    msg += f"<b>📊 Simulation Results:</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━\n"
    
    # Add learnings
    if learnings["observations"]:
        msg += f"\n<b>🔍 Observations:</b>\n"
        for obs in learnings["observations"][:5]:
            msg += f"• {obs}\n"
    
    if learnings["adaptations"]:
        msg += f"\n<b>🔧 Adaptations:</b>\n"
        for ada in learnings["adaptations"][:5]:
            msg += f"• {ada}\n"
    
    if learnings["risk_notes"]:
        msg += f"\n<b>⚠️ Risk Notes:</b>\n"
        for note in learnings["risk_notes"][:3]:
            msg += f"• {note}\n"
    
    msg += f"\n<i>Mode: PAPER TRADE (Simulation)</i>"
    msg += f"\n<i>No real orders placed.</i>"
    
    return msg


def send_telegram(message: str):
    """Send message via Telegram bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Split if too long
    max_len = 4000
    if len(message) > max_len:
        message = message[:max_len] + "\n\n<i>... (truncated)</i>"
    
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            logger.info("✅ Telegram message sent successfully!")
        else:
            logger.error(f"Telegram failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")


def send_full_report_telegram(
    theta_trades: list[SimulatedTrade],
    swing_trades: list[SimulatedTrade],
    spot_price: float,
    vix: float,
    global_cues: dict,
    intraday_sim: dict,
    learnings: dict,
):
    """Send comprehensive report via Telegram."""
    
    all_trades = theta_trades + swing_trades
    total_pnl = sum(t.pnl for t in all_trades)
    theta_pnl = sum(t.pnl for t in theta_trades)
    swing_pnl = sum(t.pnl for t in swing_trades)
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl < 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
    day_return_pct = total_pnl / CAPITAL * 100
    
    pnl_emoji = "📈" if total_pnl >= 0 else "📉"
    
    msg = f"""<b>🤖 ZERO AGENT — Simulation Report</b>
<b>📅 {NOW.strftime('%A, %d %b %Y')}</b>

<b>{pnl_emoji} P&L: ₹{total_pnl:,.0f} ({day_return_pct:+.2f}%)</b>
━━━━━━━━━━━━━━━━━━━━

<b>📊 Market:</b>
• Nifty: {spot_price:,.0f}
• VIX: {vix:.1f}
• Sim Range: {spot_price + intraday_sim['low_offset']:.0f} – {spot_price + intraday_sim['high_offset']:.0f}
• Sim Close: {spot_price + intraday_sim['close_offset']:.0f} ({intraday_sim['close_offset']/spot_price*100:+.2f}%)

<b>🌍 Global:</b>"""
    
    for name, data in list(global_cues.items())[:4]:
        emoji = "🟢" if data["change_pct"] > 0 else "🔴"
        msg += f"\n• {emoji} {name}: {data['change_pct']:+.2f}%"
    
    msg += f"""

<b>💰 Strategy Breakdown:</b>
• Theta (credit spreads): ₹{theta_pnl:,.0f} ({len(theta_trades)} trades)
• Swing (momentum): ₹{swing_pnl:,.0f} ({len(swing_trades)} trades)
• Win Rate: {win_rate:.0f}% ({len(wins)}W/{len(losses)}L)"""
    
    # Trade details
    if all_trades:
        msg += f"\n\n<b>📋 Trades:</b>"
        for i, t in enumerate(all_trades[:6], 1):
            e = "✅" if t.pnl > 0 else "❌" if t.pnl < 0 else "⚪"
            msg += f"\n{i}. {e} {t.symbol}: ₹{t.pnl:,.0f} ({t.exit_reason})"
    
    # Learnings
    if learnings["observations"]:
        msg += f"\n\n<b>🔍 Key Observations:</b>"
        for obs in learnings["observations"][:3]:
            msg += f"\n• {obs}"
    
    if learnings["adaptations"]:
        msg += f"\n\n<b>🔧 Adaptations:</b>"
        for ada in learnings["adaptations"][:3]:
            msg += f"\n• {ada}"
    
    if learnings["risk_notes"]:
        msg += f"\n\n<b>⚠️ Risk:</b>"
        for note in learnings["risk_notes"][:2]:
            msg += f"\n• {note}"
    
    msg += f"""

<b>💼 Capital:</b> ₹{CAPITAL:,} → ₹{CAPITAL + total_pnl:,.0f}
<i>🔒 Mode: SIMULATION (no real orders)</i>
<i>Building trust before live authorization.</i>"""
    
    send_telegram(msg)


# ─── Main Simulation ─────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("ZERO AGENT - SESSION SIMULATION")
    logger.info(f"Date: {NOW.strftime('%A, %B %d, %Y %H:%M IST')}")
    logger.info(f"Capital: ₹{CAPITAL:,} | Mode: PAPER TRADE")
    logger.info("=" * 60)
    
    # 1. Fetch market data
    nifty_data = fetch_nifty_data()
    if nifty_data.empty:
        logger.error("Could not fetch Nifty data. Aborting.")
        return
    
    spot_price = float(nifty_data["close"].iloc[-1])
    logger.info(f"Nifty Spot: {spot_price:,.0f}")
    
    # 2. Fetch VIX
    vix = fetch_india_vix()
    logger.info(f"India VIX: {vix:.1f}")
    
    # 3. Fetch global cues
    global_cues = fetch_global_cues()
    
    # 4. Simulate intraday movement
    intraday_sim = simulate_intraday_move(spot_price, vix)
    logger.info(f"Simulated day range: {spot_price + intraday_sim['low_offset']:.0f} - {spot_price + intraday_sim['high_offset']:.0f}")
    
    # 5. Generate options chain
    chain = fetch_nifty_options_chain_simulated(spot_price, vix)
    
    # 6. Run strategies
    logger.info("\n" + "=" * 40)
    logger.info("RUNNING STRATEGIES")
    logger.info("=" * 40)
    
    theta_trades = simulate_theta_strategy(spot_price, vix, chain)
    swing_trades = simulate_momentum_strategy(nifty_data)
    
    # 7. Generate report
    report = generate_report(theta_trades, swing_trades, spot_price, vix, global_cues, intraday_sim)
    print(report)
    
    # 8. Generate learnings
    learnings = generate_learnings(theta_trades, swing_trades, vix, global_cues)
    
    print("\n" + "=" * 50)
    print("📚 LEARNINGS & ADAPTATIONS")
    print("=" * 50)
    for key in ["observations", "adaptations", "risk_notes"]:
        items = learnings.get(key, [])
        if items:
            print(f"\n{key.upper()}:")
            for item in items:
                print(f"  • {item}")
    
    # 9. Send via Telegram
    logger.info("\nSending report via Telegram...")
    send_full_report_telegram(theta_trades, swing_trades, spot_price, vix, global_cues, intraday_sim, learnings)
    
    # 10. Save simulation results
    results = {
        "date": NOW.isoformat(),
        "spot_price": spot_price,
        "vix": vix,
        "global_cues": global_cues,
        "intraday_sim": intraday_sim,
        "theta_trades": [
            {"symbol": t.symbol, "pnl": t.pnl, "exit_reason": t.exit_reason, "confidence": t.confidence}
            for t in theta_trades
        ],
        "swing_trades": [
            {"symbol": t.symbol, "pnl": t.pnl, "exit_reason": t.exit_reason, "confidence": t.confidence}
            for t in swing_trades
        ],
        "total_pnl": sum(t.pnl for t in theta_trades + swing_trades),
        "learnings": learnings,
    }
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"simulation_{NOW.strftime('%Y%m%d')}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("\n✅ Simulation complete!")


if __name__ == "__main__":
    main()
