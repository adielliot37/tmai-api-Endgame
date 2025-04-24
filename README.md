# ArithmoAI

ArithmoAI is your Gen-Z friendly, quant-powered, AI-supercharged crypto trading companion. Built as a Telegram bot, it unifies multi-exchange portfolio management, AI chat, quantitative metrics, technical analysis, and automated futures strategies into a single, easy-to-use interface.

---

## 🚀 Features

- **Multi-CEX Management**  
  - Connect Binance (and other CEX) via API keys  
  - View spot & futures balances, open orders, positions  
  - Place market buy/sell orders directly from Telegram

- **Natural-Language Interface (NLP)**  
  - Classify messages into intents (`portfolio_health`, `market_sentiment`, `coin_analysis`, `should_buy`, `buy_token`, `sell_token`, `replace_coin`, `portfolio_diversity`, `tax_implications`, etc.)  
  - Extract symbols & amounts automatically  

- **QuantMetrics Integration**  
  - Fetch Sharpe ratio, volatility, max drawdown, daily return from TokenMetrics `/quantmetrics` API  
  - Fetch long-form AI analyses via `/ai-reports`  

- **AI Chat & Summaries**  
  - TokenMetrics AI Chat (`/tmai`) for open-ended questions (“next 100× coin?”)  
  - OpenAI GPT for summarization & enrichment  

- **Technical Analysis (TA-Lib)**  
  - **ATR** (Average True Range) → dynamic stop-loss & take-profit  
  - **RSI**, **MACD**, **Bollinger Bands**, **MA** (SMA/EMA) for entry/exit signals  
  - Entry: MA crossovers, volatility breakouts, divergence checks  

- **Risk & Money Management**  
  - **Stop-Loss**: `entry_price – 1.5 × ATR`  
  - **Take-Profit**: `entry_price + 2.0 × ATR`  
  - **Position Sizing** via Kelly Criterion:  
    ```python
    f_star = (win_rate * (avg_win_loss + 1) - 1) / avg_win_loss
    ```  

- **Automated Futures Trading**  
  - Trend-following, mean-reversion hooks  
  - Scheduled checks & market orders  

- **Portfolio Analysis & Rebalancing**  
  - Real-time USD valuation via CCXT/Binance  
  - Allocation %, portfolio Sharpe & volatility, risk score  
  - Identify weakest Sharpe coin and recommend replacements from top QuantMetrics picks  

- **Market Sentiment Dashboard**  
  - Pull headlines from TokenMetrics `sentiments` API  
  - Summarize & tag tokens as bullish or bearish  

- **Tax & Compliance Aid**  
  - Generate tailored tax-reporting guidance based on portfolio composition  

---

## 🏗️ Architecture

User ↔ Telegram Bot ├─ Intent Classification → OpenAI GPT ├─ QuantMetrics APIs → /quantmetrics, /ai-reports, /tmai ├─ CCXT Library → Binance spot & futures ├─ TA-Lib Library → ATR, RSI, MACD, Bollinger, MA… └─ OpenAI → Summarization & Q&A enhancements
---

## 🔧 Installation

1. **Clone & enter**  
   ```bash
   git clone https://github.com/yourname/arithmoai.git
   cd arithmoai
   ```
2. **Create & activate virtualenv**

    ```bash
    
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies

    ```bash
    
    pip install -r requirements.txt
    ```
4. **Environment variables**
   
   Copy .env.example to .env and set:

    ```bash
    TELEGRAM_BOT_TOKEN=your_telegram_token
    OPENAI_API_KEY=your_openai_key
    TOKENMETRICS_API_KEY=your_tokenmetrics_key
    
    ```
5. **Run**

    ```bash
    
    python bot.py
    ```

## ⚙️ Configuration & Commands

### `/start`
Connect your Binance account if not already connected.  
If already connected, it shows demo queries you can try like:
- `portfolio health`
- `buy ETH 10`
- `replace weakest coin`

### `/cex`
List currently connected exchanges and their connection status.

### `/auto`
Trigger the automated futures strategy using quant + TA indicators.

### `/positions`
Display your current open positions (spot + futures).

## 💬 Usage Examples

| Command               | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `portfolio health`    | 📊 Detailed portfolio analytics & risk assessment             |
| `market sentiment`    | 📰 News summary + per-token bullish/bearish outlook           |
| `analyze BTC`         | 🔍 Coin-specific Quant & AI analysis                         |
| `should I buy SOL?`   | ✅/❌ Quant signal + metrics breakdown                         |
| `which coin to buy`   | ✨ Top 5 QuantMetrics picks                                    |
| `buy ETH 10`          | 💸 Market-buy $10 USDT of ETH                                 |
| `sell ADA all`        | 💸 Market-sell your full ADA position                         |
| `replace weakest coin`| 🔄 Suggest replacement for lowest Sharpe token                |
| `portfolio diversity` | 🔄 HHI concentration score & diversity rating                 |
| `tax implications`    | 💰 Custom crypto tax guidance                                 |
| `/auto`               | 🤖 Run automated futures strategy with dynamic leverage        |
| `/positions`          | 🤖 Show current futures & spot positions                      |

---

## 🧠 Technical Deep-Dive

### 🔌 APIs

#### TokenMetrics

- `GET /quantmetrics?symbol=...`  
  → Fetches detailed quantitative metrics like Sharpe ratio, volatility, max drawdown, and average daily return.

- `GET /ai-reports?symbol=...`  
  → Returns natural-language investment analysis generated by TokenMetrics AI.

- `POST /tmai`  
  → Sends user prompts to TokenMetrics AI Chatbot for conversational insights and recommendations.

- `GET /trading-signals?symbol=...`  
  → Retrieves AI-generated buy/sell/neutral signals used for auto-futures strategies and short-term trading.

- `GET /sentiments`  
  → Provides aggregated market sentiment and recent news summaries with bullish/bearish classifications.

#### OpenAI GPT-3.5
- Intent classification from user input
- Summarization of AI insights

#### CCXT
- Binance spot & futures order placement
- Portfolio balance fetching

---
## 📊 TA-Lib Indicators

| Indicator         | Technical Role                                           | Strategy Weightage |
|-------------------|----------------------------------------------------------|---------------------|
| **ATR**           | Measures volatility to size stop-loss & take-profit     | 🟢 High (SL/TP calc) |
| **RSI**           | Identifies overbought/oversold momentum zones           | 🟡 Medium            |
| **MACD**          | Confirms trend direction + momentum crossover signals   | 🟢 High (entry logic)|
| **Bollinger Bands**| Volatility envelopes for breakout & mean-reversion     | 🟡 Medium            |
| **SMA/EMA**       | Defines short/long trend bias and crossover strategies  | 🟢 High (entry logic)|
| **Stochastic RSI**| Combines RSI + momentum for micro timing                | 🟡 Medium-Low        |
| **CCI**           | Detects price divergence vs mean (momentum filtering)   | ⚪ Optional           |
| **WMA**           | Weighted moving average for smoother trend reactions    | ⚪ Optional           |
| **ADX**           | Measures trend strength to filter weak signals          | ⚪ Optional           |

> ⚙️ *Only high-confidence indicators (like ATR, MACD, SMA/EMA) are directly used in auto-trading entries and stop logic. Others enhance edge in confluence zones.*

## 📈 Kelly Criterion (Position Sizing)

The **Kelly Criterion** is a formula used to determine the optimal bet size for maximizing long-term capital growth while minimizing risk. It balances **win probability** and **risk-reward ratio**.

**Formula:**
  - f* = (p × (b + 1) − 1) / b

Where:
- `f*` is the optimal fraction of capital to risk per trade  
- `p` is the probability of a win  
- `b` is the average win-to-loss ratio (`avg_win / avg_loss`)

In ArithmoAI, this is used to scale position sizes based on past performance and volatility-adjusted edge.



## 📉 Risk Management

- **Entry Signal:**  
  - Moving average crossover  
  - Volatility breakout (Bollinger Band spike)

- **Stop-Loss:**  
  - SL = Entry Price - (1.5 × ATR)
 
- **Take-Profit:**
  - TP = Entry Price + (2.0 × ATR)


- **Position Sizing (Kelly Criterion):**
```python
win_rate     = wins / total_trades
avg_win_loss = avg_profit / avg_loss
f_star       = (win_rate * (avg_win_loss + 1) - 1) / avg_win_loss
```

---

## 🔄 Portfolio Rebalancing

- 📥 Fetch current holdings using **CCXT**
- 💵 Compute USD value and % allocation of each token
- 📉 Rank tokens by **Sharpe Ratio** to identify the weakest asset
- 🔁 Recommend replacement from the top QuantMetrics picks

---

## 🤝 Contributing

1. **Fork** this repository
2. **Create** a feature branch (`git checkout -b feature-name`)
3. **Implement & test** your changes
4. **Push** your branch (`git push origin feature-name`)
5. **Open a Pull Request** describing your enhancement

---
---

## 📄 License

This project is licensed under the **MIT License**.

© Aditya Chaplot – Feel free to use, fork, and build on it.

---


    

