"""
Trading Dashboard - Streamlit app for monitoring P&L and positions.
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(page_title="Zero Trading Agent", page_icon="📈", layout="wide")

st.title("📈 Zero Trading Agent Dashboard")

# Sidebar
st.sidebar.header("System Status")
mode = os.getenv("PAPER_TRADE", "true")
st.sidebar.metric("Mode", "PAPER" if mode == "true" else "🔴 LIVE")
st.sidebar.metric("Status", "Running" if True else "Stopped")

# Main content
col1, col2, col3, col4 = st.columns(4)

# Load trade log if exists
trade_log_path = "logs/trades.json"
trades = []
if os.path.exists(trade_log_path):
    with open(trade_log_path, "r") as f:
        for line in f:
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue

daily_pnl = sum(t.get("pnl", 0) for t in trades if t.get("date") == datetime.now().strftime("%Y-%m-%d"))
total_pnl = sum(t.get("pnl", 0) for t in trades)

col1.metric("Today's P&L", f"₹{daily_pnl:,.0f}")
col2.metric("Total P&L", f"₹{total_pnl:,.0f}")
col3.metric("Total Trades", len(trades))
col4.metric("Win Rate", f"{sum(1 for t in trades if t.get('pnl', 0) > 0) / max(len(trades), 1) * 100:.0f}%")

st.divider()

# Trade history
st.subheader("Recent Trades")
if trades:
    df = pd.DataFrame(trades[-20:])  # Last 20 trades
    st.dataframe(df, use_container_width=True)
else:
    st.info("No trades recorded yet. Start the agent to begin trading.")

# Risk Status
st.subheader("Risk Status")
st.json({
    "daily_loss_limit": "2% of capital",
    "weekly_loss_limit": "5% of capital",
    "max_positions": 5,
    "consecutive_losses_allowed": 3,
})
