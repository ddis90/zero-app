# Zero Trading Agent — Setup Guide

> Complete step-by-step guide to get the trading agent running from scratch.

---

## Prerequisites Checklist

| # | Item | Time | Cost |
|---|------|------|------|
| 1 | Zerodha Kite Connect API | 10 min | ₹500/month |
| 2 | TOTP Secret (auto-login) | 5 min | Free |
| 3 | Telegram Bot | 5 min | Free |
| 4 | OpenAI API Key | 5 min | ₹500-2000/month |
| 5 | Azure Account | 10 min | ₹800-1200/month |
| 6 | Python 3.11+ Environment | 10 min | Free |
| 7 | GitHub Account (CI/CD) | 5 min | Free |

---

## Step 1: Zerodha Kite Connect API

1. Go to **https://kite.trade** (Zerodha developer portal)
2. Login with your Zerodha credentials
3. Click **"Subscribe"** → Choose **Kite Connect** plan (₹500/month)
4. Click **"Create new app"**:
   - **App Name:** `ZeroAgent`
   - **Type:** Connect
   - **Redirect URL:** `http://127.0.0.1:5000/callback`
   - **Description:** Personal automated trading agent
5. After creation, note down:
   ```
   API Key:    xxxxxxxxxxxxxxxxxx
   API Secret: xxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
6. Your **User ID** is your Zerodha client ID (e.g., `AB1234`)
   - Found on: Kite → Profile → Client ID

---

## Step 2: Enable TOTP (Required for Auto-Login)

The system uses TOTP to login automatically without human intervention.

1. Open **Kite mobile app** → Settings → Security
2. Go to **Two-factor authentication** → Enable **TOTP**
3. When shown the QR code, click **"Can't scan the QR code?"**
4. Copy the **32-character secret key** shown (e.g., `JBSWY3DPEHPK3PXP7USNCZ2MKFHG5HT`)
5. Complete setup by entering the 6-digit code from Google Authenticator / Authy
6. **IMPORTANT:** Keep this 32-char secret safe — it's needed for `.env`

### Verify TOTP works:
```python
import pyotp
totp = pyotp.TOTP("YOUR_32_CHAR_SECRET")
print(totp.now())  # Should match your authenticator app
```

---

## Step 3: Create Telegram Bot

Telegram is used for:
- Real-time trade alerts on your phone
- Daily P&L summaries
- Approve/reject strategy parameter changes (inline buttons)

### Create the bot:
1. Open Telegram → Search for **@BotFather**
2. Send: `/newbot`
3. Enter name: `Zero Trading Bot`
4. Enter username: `zero_trading_XXXX_bot` (must end with `bot`, must be unique)
5. BotFather responds with your **Bot Token:**
   ```
   123456789:ABCdefGHIjklMNOpqrSTUVwxyz
   ```

### Get your Chat ID:
1. Send any message to your new bot (e.g., "hello")
2. Open this URL in browser (replace YOUR_TOKEN):
   ```
   https://api.telegram.org/botYOUR_TOKEN/getUpdates
   ```
3. Find the `"chat":{"id":` value — this is your **Chat ID** (a number like `987654321`)

### Verify:
```bash
curl -X POST "https://api.telegram.org/botYOUR_TOKEN/sendMessage" \
  -d "chat_id=YOUR_CHAT_ID" \
  -d "text=Hello from Zero Agent!"
```
You should receive "Hello from Zero Agent!" on Telegram.

---

## Step 4: OpenAI API Key

Used for: sentiment analysis, weekly strategy review, pattern discovery.

1. Go to **https://platform.openai.com/api-keys**
2. Click **"Create new secret key"**
3. Name: `ZeroAgent`
4. Copy the key (starts with `sk-...`)
5. Go to **Billing** → Add ₹500 credit (gpt-4o-mini is very cheap: ~₹0.10 per call)

### Verify:
```python
from openai import OpenAI
client = OpenAI(api_key="sk-YOUR-KEY")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello"}],
    max_tokens=10,
)
print(resp.choices[0].message.content)
```

---

## Step 5: Azure Account Setup

### 5a. Create Azure Account
1. Go to **https://azure.microsoft.com/free**
2. Sign up (free tier gives ₹14,500 credits for 30 days)
3. After trial, pay-as-you-go is ~₹800-1200/month for our setup

### 5b. Install Azure CLI
```bash
# Windows (PowerShell as Admin)
winget install -e --id Microsoft.AzureCLI

# Verify
az --version
az login
```

### 5c. Create Resource Group
```bash
az group create --name rg-zero-trading --location centralindia
```

### 5d. Create Key Vault (stores all secrets)
```bash
az keyvault create \
  --name kv-zero-trading \
  --resource-group rg-zero-trading \
  --location centralindia

# Store secrets
az keyvault secret set --vault-name kv-zero-trading --name KITE-API-KEY --value "your-key"
az keyvault secret set --vault-name kv-zero-trading --name KITE-API-SECRET --value "your-secret"
az keyvault secret set --vault-name kv-zero-trading --name KITE-TOTP-SECRET --value "your-totp"
az keyvault secret set --vault-name kv-zero-trading --name KITE-USER-ID --value "AB1234"
az keyvault secret set --vault-name kv-zero-trading --name KITE-PASSWORD --value "your-password"
az keyvault secret set --vault-name kv-zero-trading --name OPENAI-API-KEY --value "sk-xxx"
az keyvault secret set --vault-name kv-zero-trading --name TELEGRAM-BOT-TOKEN --value "123:ABC"
az keyvault secret set --vault-name kv-zero-trading --name TELEGRAM-CHAT-ID --value "987654321"
```

---

## Step 6: Python Environment

```bash
# Ensure Python 3.11+
python --version

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright (for headless Kite login in cloud)
playwright install chromium
```

> **Note:** We use `pandas-ta` (pure Python). No C compilation or ta-lib system install needed.

---

## Step 7: Configure Local Environment

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` with your actual values:
```env
# Zerodha Kite Connect
KITE_API_KEY=your_api_key_from_step_1
KITE_API_SECRET=your_api_secret_from_step_1
KITE_TOTP_SECRET=your_32_char_totp_from_step_2
KITE_USER_ID=AB1234
KITE_PASSWORD=your_zerodha_password

# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# Telegram
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNO
TELEGRAM_CHAT_ID=987654321

# Trading mode (ALWAYS start with paper!)
PAPER_TRADE=true
```

---

## Step 8: First Login & Token Generation

### Local (development):
```bash
python -m src.utils.login_server
```
- Browser opens → Zerodha login page
- TOTP auto-filled → Callback captured
- Token saved to `.kite_token`
- Output: `"Login successful! Token saved."`

### Cloud (headless, no browser):
```bash
python -m src.utils.headless_auth
```
- Playwright opens headless Chromium
- Navigates to kite.trade, fills credentials + TOTP
- Captures redirect token automatically
- No human interaction needed

---

## Step 9: Run Locally (Paper Trade)

```bash
python -m src.main
```

Expected:
```
============================================================
  Zero Trading Agent
  Mode: 📝 PAPER TRADING
  Time: 2026-05-30 09:00:00
============================================================
Trading system started. Waiting for scheduled tasks...
```

### What happens during market hours (9:00 - 15:45 IST):
| Time | Event |
|------|-------|
| 09:00 | Pre-market brief → Telegram |
| 09:30 | Options scan (theta selling) |
| 10:00 | Swing scan (momentum/mean-reversion) |
| Every 5 min | Position monitoring |
| 15:35 | Daily P&L summary → Telegram |
| 15:45 (Fri) | Weekly review + parameter proposals |

---

## Step 10: Run Backtest (No Live Credentials Needed)

```bash
# Full 2-year walk-forward backtest
python -m src.backtest.run_backtest --capital 200000 --years 2 --walk-forward

# Quick 6-month test
python -m src.backtest.run_backtest --capital 200000 --months 6

# Generate report only (if data already cached)
python -m src.backtest.run_backtest --capital 200000 --report-only
```

Output: Equity curve PNG + JSON report + CSV trade log in `reports/` folder.

---

## Step 11: Deploy to Azure Container Apps

```bash
# One-command deployment (after Azure setup)
az containerapp up \
  --name zero-trading-agent \
  --resource-group rg-zero-trading \
  --location centralindia \
  --source . \
  --env-vars "PAPER_TRADE=true"
```

Or use the CI/CD pipeline (push to `main` → auto-deploys).

---

## Step 12: Go Live

> **Only after:** 5+ days of successful paper trading AND positive backtest results.

1. Change `PAPER_TRADE=false` in Azure Key Vault or env vars
2. Restart the container app
3. Start with ₹2,00,000 capital
4. Scale to ₹5,00,000 after 4-6 weeks of profitable live trading

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `pip install` fails | Ensure Python 3.11+. Run `pip install --upgrade pip` first |
| Login timeout | Check TOTP secret (32 chars, base32 encoded). Verify with `pyotp` |
| No Telegram messages | Send `/start` to your bot first. Verify chat_id is a number |
| Token expired (6 AM) | Normal. Headless auth re-runs at 8:50 AM automatically |
| ChromaDB error | `pip install chromadb --upgrade`. Needs SQLite 3.35+ |
| Options data missing | Historical option chains aren't free. Backtester uses Black-Scholes synthetic pricing |
| Azure deploy fails | Run `az login` first. Check resource group exists |

---

## Monthly Cost Summary

| Item | Cost |
|------|------|
| Kite Connect API | ₹500 |
| Azure Container Apps (scale-to-zero) | ₹800-1,200 |
| Azure Key Vault + Storage | ₹70 |
| OpenAI gpt-4o-mini | ₹500-2,000 |
| **Total** | **₹1,870-3,770/month** |

---

## Security Notes

- Never commit `.env` or `.kite_token` to git (already in `.gitignore`)
- All secrets stored in Azure Key Vault (encrypted at rest, audit-logged)
- Kite password is needed for headless login — store ONLY in Key Vault
- Paper trade mode is the default — live requires explicit confirmation
- TOTP secret + password together can access your Zerodha account — treat them as critically sensitive
