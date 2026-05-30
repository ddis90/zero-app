# Zero Trading Agent

AI-powered trading agent for Indian markets via Zerodha Kite Connect.

## Features

- **Options Theta Selling**: Weekly credit spreads on Nifty/BankNifty
- **Momentum Swing Trading**: Quality large-cap stocks (Nifty 500 only)
- **AI Market Analysis**: News sentiment + regime detection via LLM
- **Strict Risk Management**: Daily/weekly loss limits, position sizing, auto-shutdown
- **No Penny Stocks**: Minimum ₹5,000 Cr market cap, volume filters
- **Telegram Alerts**: Real-time trade notifications

## Quick Start

### 1. Setup

```bash
# Clone and install
cd zero
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy env template and fill in your credentials
copy .env.example .env
# Edit .env with your Kite API key, TOTP secret, Telegram bot token
```

### 3. Authenticate (Daily)

```bash
# Run login server - opens browser for Kite login
python -m src.utils.login_server
```

### 4. Run (Paper Trading)

```bash
# Paper trading mode (default - no real orders)
set PAPER_TRADE=true
python -m src.main
```

### 5. Run (Live Trading)

```bash
# ⚠️ CAUTION: Real money!
set PAPER_TRADE=false
python -m src.main
```

## Architecture

```
src/
├── agents/
│   ├── risk_manager.py    # Guardrails & position sizing (VETO power)
│   ├── market_analyst.py  # AI sentiment & regime detection
│   └── executor.py        # Order placement via Kite Connect
├── strategies/
│   ├── base.py            # Abstract strategy interface
│   ├── theta_selling.py   # Options credit spreads
│   └── momentum_swing.py  # Large-cap swing trades
├── data/
│   ├── fetcher.py         # Historical + live data from Kite
│   └── screener.py        # Stock universe filtering
├── utils/
│   ├── auth.py            # Kite Connect auth + TOTP
│   ├── login_server.py    # OAuth redirect handler
│   └── notifier.py        # Telegram alerts
└── main.py                # Orchestrator / scheduler
```

## Risk Guardrails

| Rule | Value |
|------|-------|
| Max risk per trade | 2% of capital |
| Daily loss limit | 2% → auto-shutdown |
| Weekly loss limit | 5% → pause for week |
| Max consecutive losses | 3 → auto-pause |
| Max open positions | 5 |
| Minimum market cap | ₹5,000 Cr |
| Stop-loss | Mandatory on every trade (GTT) |

## Deployment (Cloud)

```bash
# Docker deployment
cd deploy
docker-compose up -d

# Or AWS EC2 (Mumbai region)
# See deploy/ folder for crontab scheduling
```

## Cost Breakdown

| Item | Monthly |
|------|---------|
| Kite Connect API | ₹500 |
| Cloud VPS (AWS Mumbai) | ₹1,500 |
| OpenAI API | ₹2,500 |
| **Total** | **~₹5,000** |

## Disclaimer

This software is for educational purposes. Trading involves risk of loss.
Past performance does not guarantee future results. Use at your own risk.
Always start with paper trading and validate strategies before using real capital.

## License

Private / Personal Use
