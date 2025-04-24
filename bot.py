import os
import logging
import requests
import datetime
import ccxt
import json
import re
from auto_trader import auto_trade, show_positions
from openai import OpenAI
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler, CallbackQueryHandler
)


load_dotenv()
logging.basicConfig(level=logging.INFO)


USERS_FILE = 'users.json'
try:
    with open(USERS_FILE, 'r') as f:
        user_binance_keys = json.load(f)
except FileNotFoundError:
    user_binance_keys = {}


bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
tm_api_key = os.getenv("TOKENMETRICS_API_KEY")
openai = OpenAI() 
API_KEY, API_SECRET = range(2)


def save_users():
    with open(USERS_FILE, 'w') as f:
        json.dump(user_binance_keys, f)


def _tm_request(endpoint, params=None):
    url = f"https://api.tokenmetrics.com/v2/{endpoint}"
    headers = {"api_key": tm_api_key}
    r = requests.get(url, headers=headers, params=params)
    logging.debug("TM URL: %s", r.request.url)
    if r.status_code != 200:
        logging.error(f"TM {endpoint} {r.status_code}: {r.text}")
        return []
    try:
        return r.json().get("data", [])
    except ValueError:
        logging.error(f"TM {endpoint} JSON error")
        return []

def fetch_tmai(prompt: str):
    url = "https://api.tokenmetrics.com/v2/tmai"
    headers = {"api_key": tm_api_key, "content-type": "application/json"}
    body = {"messages": [{"user": prompt}]}
    r = requests.post(url, headers=headers, json=body)
    if r.status_code != 200:
        logging.error(f"TMAI failed {r.status_code}: {r.text}")
        return {}
    return r.json()


def fetch_sentiment(limit=1):
    arr = _tm_request("sentiments", {"limit": limit, "page": 0})
    return arr[0] if arr else {}

def fetch_quantmetrics(symbol=None, marketcap=None, volume=None, fdv=None, limit=10, page=0):
    params = {"limit": limit, "page": page}
    if symbol:    params["symbol"]    = symbol.upper()
    if marketcap: params["marketcap"] = marketcap
    if volume:    params["volume"]    = volume
    if fdv:       params["fdv"]       = fdv
    return _tm_request("quantmetrics", params)

def fetch_ai_report(symbol, limit=1):
    arr = _tm_request("ai-reports", {"symbol": symbol.upper(), "limit": limit, "page": 0})
    return arr[0] if arr else {}


def recommend_coins(marketcap=100_000_000, volume=100_000_000, fdv=100_000_000, top_n=5):
    data = fetch_quantmetrics(marketcap=marketcap, volume=volume, fdv=fdv, limit=50)
    # Sort by Sharpe ratio and only include tokens with positive values
    valid_tokens = [t for t in data if t.get("SHARPE", 0) > 0]
    sorted_tokens = sorted(valid_tokens, key=lambda x: x.get("SHARPE", 0), reverse=True)
    return sorted_tokens[:top_n]


def analyze_portfolio(holdings):
    """
    Analyze a portfolio and return detailed metrics
    Returns: dict with analysis results
    """
    if not holdings:
        return {"error": "Empty portfolio"}
    
   
    coin_metrics = {}
    total_value_usd = 0
    
    for symbol, amount in holdings.items():
        metrics = fetch_quantmetrics(symbol=symbol)
        if not metrics:
            continue
            
        coin_data = metrics[0]
           
        try:
            ex = ccxt.binance({"enableRateLimit": True})
            ticker = ex.fetch_ticker(f"{symbol}/USDT")
            price_usd = ticker.get("last", 0) or 0
        except Exception as e:
            logging.error(f"Price fetch failed for {symbol}: {e}")
            price_usd = 0

        value_usd = price_usd * float(amount)
        total_value_usd += value_usd
        
        coin_metrics[symbol] = {
            "amount": amount,
            "price_usd": price_usd,
            "value_usd": value_usd,
            "sharpe": coin_data.get("SHARPE", 0),
            "volatility": coin_data.get("VOLATILITY", 0),
            "max_drawdown": coin_data.get("MAX_DRAWDOWN", 0),
            "daily_return": coin_data.get("DAILY_RETURN_AVG", 0),
        }
    
   
    if not coin_metrics:
        return {"error": "Unable to fetch metrics for any holdings"}
    
   
    for symbol in coin_metrics:
        if total_value_usd > 0:
            coin_metrics[symbol]["allocation"] = (coin_metrics[symbol]["value_usd"] / total_value_usd) * 100
        else:
            coin_metrics[symbol]["allocation"] = 0
    
  
    if coin_metrics:
        best_coin = max(coin_metrics.keys(), key=lambda x: coin_metrics[x]["sharpe"])
        worst_coin = min(coin_metrics.keys(), key=lambda x: coin_metrics[x]["sharpe"])
    else:
        best_coin = worst_coin = None
    

    portfolio_sharpe = sum(m["sharpe"] * m["allocation"]/100 for m in coin_metrics.values())
    portfolio_volatility = sum(m["volatility"] * m["allocation"]/100 for m in coin_metrics.values())
    
   
    risk_score = min(100, max(0, 50 + (portfolio_sharpe * 10) - (portfolio_volatility * 10)))
    
    return {
        "total_value_usd": total_value_usd,
        "coins": coin_metrics,
        "best_coin": best_coin,
        "worst_coin": worst_coin,
        "portfolio_sharpe": portfolio_sharpe,
        "portfolio_volatility": portfolio_volatility,
        "risk_score": risk_score
    }

def get_portfolio_health_message(analysis):
    """Format portfolio health analysis into a readable message"""
    if "error" in analysis:
        return f"❌ Error: {analysis['error']}"
    
    risk_level = "Low" if analysis["risk_score"] > 70 else "Medium" if analysis["risk_score"] > 40 else "High"
    
    msg = [
        f"📊 *Portfolio Health Analysis*",
        f"",
        f"💰 Total Value: ${analysis['total_value_usd']:.4f}",
        f"📈 Portfolio Sharpe Ratio: {analysis['portfolio_sharpe']:.2f}",
        f"📉 Portfolio Volatility: {analysis['portfolio_volatility']:.2f}",
        f"⚠️ Risk Score: {analysis['risk_score']:.0f}/100 ({risk_level} Risk)",
        f"",
        f"*Top Performers:*"
    ]
    
   
    sorted_coins = sorted(analysis['coins'].items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for symbol, data in sorted_coins[:3]:
        msg.append(f"• {symbol}: Sharpe={data['sharpe']:.2f}, Allocation={data['allocation']:.4f}%")
    
    msg.append("")
    msg.append("*Weakest Performers:*")
    
  
    for symbol, data in sorted_coins[-2:]:
        msg.append(f"• {symbol}: Sharpe={data['sharpe']:.2f}, Allocation={data['allocation']:.4f}%")
    
    msg.append("")
    msg.append("*Recommendations:*")
    
   
    if analysis['portfolio_sharpe'] < 0.5:
        msg.append("• Consider replacing low Sharpe assets with higher-quality alternatives")
    if analysis['portfolio_volatility'] > 1.5:
        msg.append("• Portfolio volatility is high - consider more stable assets")
    
 
    high_allocations = [s for s, d in analysis['coins'].items() if d['allocation'] > 25]
    if high_allocations:
        msg.append(f"• High concentration in {', '.join(high_allocations)} - consider diversifying")
    
    return "\n".join(msg)


def classify_intent(text: str):
    system = (
        "You are a crypto assistant.\n"
        "Classify into: portfolio_health, market_sentiment, coin_analysis, should_buy, replace_coin, portfolio_diversity, tax_implications, buy_token, sell_token.\n"
        "Use portfolio_health for questions about portfolio quality, performance, health scores.\n"
        "Use replace_coin for questions about which coins to remove/replace from portfolio.\n"
        "Use portfolio_diversity for questions about how diversified a portfolio is.\n"
        "Use tax_implications for questions about crypto taxation.\n"
        "Use buy_token for requests to purchase cryptocurrency.\n"
        "Use sell_token for requests to sell cryptocurrency.\n"
        "Extract a symbol if mentioned.\n"
        "For buy/sell, also extract an amount (number) if mentioned.\n"
        "Reply only with JSON: {\"intent\":...,\"symbol\":...,\"amount\":...}."
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ]
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logging.error(f"Intent JSON error: {raw}")
        return {"intent": "unknown", "symbol": None, "amount": None}

 
def get_binance_holdings(api_key: str, api_secret: str):
    ex = ccxt.binance({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
    bal = ex.fetch_balance()
    return {k: v for k, v in bal.get('total', {}).items() if v and k not in ('USDT', 'BUSD')}


def get_usdt_balance(api_key: str, api_secret: str):
    ex = ccxt.binance({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
    bal = ex.fetch_balance()
    return bal.get('total', {}).get('USDT', 0)


def buy_token(api_key: str, api_secret: str, symbol: str, amount_usdt: float):
    """
    Buy a token on Binance using market order
    Returns: Order details or error message
    """
    try:
        ex = ccxt.binance({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
        
      
        market_symbol = f"{symbol}/USDT"
        
        
        markets = ex.load_markets()
        if market_symbol not in markets:
            return {"error": f"Market {market_symbol} not available"}
        
       
        ticker = ex.fetch_ticker(market_symbol)
        current_price = ticker['last']
        
      
        quantity = amount_usdt / current_price
        
       
        order = ex.create_market_buy_order(
            symbol=market_symbol,
            amount=quantity
        )
        
        return {
            "success": True,
            "order_id": order.get('id'),
            "symbol": symbol,
            "amount_usdt": amount_usdt,
            "quantity": quantity,
            "price": current_price,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logging.error(f"Buy error: {str(e)}")
        return {"error": str(e)}


def sell_token(api_key: str, api_secret: str, symbol: str, quantity=None):
    """
    Sell a token on Binance using market order
    If quantity is None, sells all available balance
    Returns: Order details or error message
    """
    try:
        ex = ccxt.binance({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
        
       
        market_symbol = f"{symbol}/USDT"
        
        
        markets = ex.load_markets()
        if market_symbol not in markets:
            return {"error": f"Market {market_symbol} not available"}
        
       
        if quantity is None:
            balance = ex.fetch_balance()
            quantity = balance.get('free', {}).get(symbol, 0)
            
            if quantity <= 0:
                return {"error": f"No {symbol} available to sell"}
        
       
        ticker = ex.fetch_ticker(market_symbol)
        current_price = ticker['last']
        estimated_value = quantity * current_price
        
        
        order = ex.create_market_sell_order(
            symbol=market_symbol,
            amount=quantity
        )
        
        return {
            "success": True,
            "order_id": order.get('id'),
            "symbol": symbol,
            "quantity": quantity,
            "price": current_price,
            "estimated_value_usdt": estimated_value,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logging.error(f"Sell error: {str(e)}")
        return {"error": str(e)}


def should_buy_decision(quant: dict):
    sharpe = quant.get("SHARPE", 0)
    max_dd = quant.get("MAX_DRAWDOWN", -1)
    vol    = quant.get("VOLATILITY", 1)
    daily  = quant.get("DAILY_RETURN_AVG", 0)
    if sharpe > 1 and daily > 0 and vol < 1.5 and max_dd > -0.95:
        return True, "Quant metrics are strong: Sharpe >1, positive returns, controlled drawdown."
    return False, "Quant metrics suggest caution: check risk-adjusted performance."


def summarize(text: str, lines: int = 5):
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Summarize in {lines} lines:\n{text}"}]
    )
    return resp.choices[0].message.content.strip()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton("🔑 Connect Binance", callback_data="connect_binance")]]
    await update.message.reply_text(
        "👋 Welcome! Connect your Binance account to begin.",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    
    uid = str(update.effective_user.id)
    if uid in user_binance_keys:
        return await update.callback_query.message.reply_text("🔄 You already have Binance connected. Use /cex to view.")
    await update.callback_query.message.reply_text("Please send your Binance API Key:")
    return API_KEY

async def handle_api_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['api_key'] = update.message.text.strip()
    await update.message.reply_text("Now send your Binance API Secret:")
    return API_SECRET

async def handle_api_secret(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    user_binance_keys[uid] = {
        'key': context.user_data['api_key'],
        'secret': update.message.text.strip(),
        'username': update.effective_user.username
    }
    save_users()
    await update.message.reply_text("✅ Binance connected! Use /cex to view your connected exchanges.")
    return ConversationHandler.END

async def handle_cex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    text = []
    if uid in user_binance_keys:
        text.append("Binance: Connected")
    else:
        text.append("Binance: Disconnected")
    await update.message.reply_text("\n".join(text))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid    = str(update.effective_user.id)
    keys   = user_binance_keys.get(uid)
    data   = classify_intent(update.message.text)
    intent = data.get('intent')
    sym    = data.get('symbol')
    symbol = sym.upper() if sym else None
    amount = data.get('amount')

    
    if intent == 'buy_token':
        if not keys:
            return await update.message.reply_text("🔗 Connect Binance first via /start.")
        
        if not symbol:
            return await update.message.reply_text("❓ Please specify which token you want to buy.")
        
       
        amount_usdt = float(amount) if amount else 5.0
        
      
        usdt_balance = get_usdt_balance(keys['key'], keys['secret'])
        
        
        await update.message.reply_text(f"💸 Processing purchase of {symbol} for ${amount_usdt}...")
        
        
        if usdt_balance < amount_usdt:
            return await update.message.reply_text(
                f"❌ Insufficient USDT balance. You have ${usdt_balance:.2f} but need ${amount_usdt:.2f}."
            )
        
        
        result = buy_token(keys['key'], keys['secret'], symbol, amount_usdt)
        
        if "error" in result:
            return await update.message.reply_text(f"❌ Buy failed: {result['error']}")
        
        
        receipt = [
            f"✅ *Purchase Complete*",
            f"",
            f"🪙 Token: {result['symbol']}",
            f"💰 Amount paid: ${result['amount_usdt']:.2f} USDT",
            f"🔢 Quantity: {result['quantity']:.8f} {result['symbol']}",
            f"💱 Price: ${result['price']:.2f} per {result['symbol']}",
            f"🕒 Time: {result['timestamp']}",
            f"📝 Order ID: {result['order_id']}"
        ]
        
        return await update.message.reply_text("\n".join(receipt))
    
   
    if intent == 'sell_token':
        if not keys:
            return await update.message.reply_text("🔗 Connect Binance first via /start.")
        
        if not symbol:
            return await update.message.reply_text("❓ Please specify which token you want to sell.")
        
        
        holdings = get_binance_holdings(keys['key'], keys['secret'])
        
        if symbol not in holdings:
            return await update.message.reply_text(f"❌ You don't have any {symbol} in your portfolio.")
        
        
        
        available_amount = holdings[symbol]

        
       
        sell_quantity = None 
        if amount is not None and not (isinstance(amount, str) and amount.lower() == "all"):
            # amount should be numeric
            try:
                requested = float(amount)
            except (ValueError, TypeError):
                return await update.message.reply_text(f"❌ Invalid amount: {amount}")
            if requested > available_amount:
                return await update.message.reply_text(
                    f"❌ You only have {available_amount} {symbol}, but tried to sell {requested}."
                )
            sell_quantity = requested


        
        # Show processing message
        sell_text = f"all your {symbol}" if sell_quantity is None else f"{sell_quantity} {symbol}"
        await update.message.reply_text(f"💸 Processing sale of {sell_text}...")
        
        # Execute sell order
        result = sell_token(keys['key'], keys['secret'], symbol, sell_quantity)
        
        if "error" in result:
            return await update.message.reply_text(f"❌ Sell failed: {result['error']}")
        
        # Format receipt
        receipt = [
            f"✅ *Sale Complete*",
            f"",
            f"🪙 Token: {result['symbol']}",
            f"🔢 Quantity sold: {result['quantity']:.8f} {result['symbol']}",
            f"💱 Price: ${result['price']:.2f} per {result['symbol']}",
            f"💰 Value received: ${result['estimated_value_usdt']:.2f} USDT",
            f"🕒 Time: {result['timestamp']}",
            f"📝 Order ID: {result['order_id']}"
        ]
        
        return await update.message.reply_text("\n".join(receipt))

    # Portfolio health 
    if intent == 'portfolio_health':
        if not keys:
            return await update.message.reply_text("🔗 Connect Binance first via /start.")
        
        await update.message.reply_text("📊 Analyzing your portfolio health...")
        holdings = get_binance_holdings(keys['key'], keys['secret'])
        
        if not holdings:
            return await update.message.reply_text("Your portfolio appears empty.")
        
        # Perform detailed analysis
        analysis = analyze_portfolio(holdings)
        health_msg = get_portfolio_health_message(analysis)
        
        return await update.message.reply_text(health_msg)

    # Portfolio diversity 
    if intent == 'portfolio_diversity':
        if not keys:
            return await update.message.reply_text("🔗 Connect Binance first via /start.")
        
        await update.message.reply_text("🔄 Analyzing your portfolio diversity...")
        holdings = get_binance_holdings(keys['key'], keys['secret'])
        
        if not holdings:
            return await update.message.reply_text("Your portfolio appears empty.")
        
        # Get analysis for diversity calculations
        analysis = analyze_portfolio(holdings)
        
        # Calculate diversity metrics
        num_assets = len(analysis['coins'])
        allocations = [data['allocation'] for data in analysis['coins'].values()]
        
        # Herfindahl-Hirschman Index (HHI) - measure of concentration
  
        hhi = sum((a/100)**2 for a in allocations) * 10000
        
        # Rate portfolio diversity
        if num_assets < 3:
            diversity_rating = "Very Low"
        elif num_assets < 5:
            diversity_rating = "Low"
        elif num_assets < 8:
            diversity_rating = "Medium" 
        else:
            diversity_rating = "High"
            
        # Adjust for concentration
        if hhi > 3000:
            diversity_rating = "Low (High Concentration)"
        
        # Find highest allocation
        highest_coin = max(analysis['coins'].items(), key=lambda x: x[1]['allocation'])
        highest_alloc = highest_coin[1]['allocation']
        
        msg = [
            f"🔄 *Portfolio Diversity Analysis*",
            f"",
            f"Number of assets: {num_assets}",
            f"Diversity rating: {diversity_rating}",
            f"Concentration score: {hhi:.0f}/10000 (lower is better)",
            f"",
            f"*Asset Allocation:*"
        ]
        
       
        sorted_by_alloc = sorted(analysis['coins'].items(), key=lambda x: x[1]['allocation'], reverse=True)
        for symbol, data in sorted_by_alloc:
            msg.append(f"• {symbol}: {data['allocation']:.4f}%")
        
        # Add recommendations
        msg.append("")
        msg.append("*Recommendations:*")
        
        if highest_alloc > 40:
            msg.append(f"• Consider reducing {highest_coin[0]} position which is {highest_alloc:.4f}% of portfolio")
            
        if num_assets < 5:
            msg.append(f"• Add more assets to increase diversity (aim for 5-10 uncorrelated assets)")
            
        if hhi > 2500:
            msg.append(f"• Portfolio is too concentrated - distribute more evenly")
            
        return await update.message.reply_text("\n".join(msg))

    # Tax implications 
    if intent == 'tax_implications':
        if not keys:
            return await update.message.reply_text("🔗 Connect Binance first via /start.")
        
        await update.message.reply_text("💰 Generating tax implications info...")
        holdings = get_binance_holdings(keys['key'], keys['secret'])
        
        if not holdings:
            return await update.message.reply_text("Your portfolio appears empty.")
        
        # Get analysis for value calculations
        analysis = analyze_portfolio(holdings)
        
        # Generate tax info using OpenAI
        prompt = f"""
        Generate tax implications/considerations for a crypto portfolio valued at ${analysis['total_value_usd']:.2f}.
        Portfolio contains: {', '.join(analysis['coins'].keys())}.
        Include:
        1. General tax reporting requirements
        2. Implications of trading vs. holding
        3. Key considerations for tax efficiency
        Keep it brief and informative.
        """
        
        tax_info = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content.strip()
        
        msg = [
            f"💰 *Tax Implications for Your Portfolio*",
            f"",
            f"{tax_info}",
            f"",
            f"*Disclaimer:* This is general information, not tax advice. Consult a tax professional for your specific situation."
        ]
        
        return await update.message.reply_text("\n".join(msg))

    # Replace weakest coin
    if intent == 'replace_coin':
        if not keys:
            return await update.message.reply_text("🔗 Connect Binance first via /start.")
        
        # First get portfolio
        holdings = get_binance_holdings(keys['key'], keys['secret'])
        if not holdings:
            return await update.message.reply_text("Your portfolio appears empty.")
            
        # Get performance metrics for each holding
        await update.message.reply_text("🔍 Analyzing your portfolio performance...")
        
        sharpe_map = {}
        for s in holdings:
            qm = fetch_quantmetrics(symbol=s)
            if qm:
                sharpe_map[s] = qm[0].get('SHARPE', 0)
        
        if not sharpe_map:
            return await update.message.reply_text("❌ Unable to analyze portfolio metrics.")
        
        # Find worst performing asset by Sharpe ratio
        worst = min(sharpe_map, key=sharpe_map.get)
        
        # Get recommendations for replacement
        recs = recommend_coins()
        # Find first recommended coin not already in portfolio
        replacement = next((t['TOKEN_SYMBOL'] for t in recs if t['TOKEN_SYMBOL'] not in holdings), recs[0]['TOKEN_SYMBOL'])
        
        # Find metrics for the recommended replacement
        rep_qm = fetch_quantmetrics(symbol=replacement)
        rep_sharpe = rep_qm[0].get('SHARPE', 0) if rep_qm else 0
        
        response = (
            f"🔄 Analysis complete!\n\n"
            f"Weakest holding: {worst} (Sharpe ratio: {sharpe_map[worst]:.2f})\n\n"
            f"Recommended replacement: {replacement} (Sharpe ratio: {rep_sharpe:.2f})\n\n"
            f"This recommendation is based on comparing risk-adjusted returns (Sharpe ratios) "
            f"across your portfolio and current market opportunities."
        )
        
        return await update.message.reply_text(response)

    # Market sentiment
    if intent == 'market_sentiment':
        s  = fetch_sentiment()
        sm = summarize(s.get('NEWS_SUMMARY', 'No recent news.'))
        prompt = f"Extract each token from this summary and label bullish or bearish:\n\n{sm}"
        out = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}]
        ).choices[0].message.content.strip()
        return await update.message.reply_text(f"📰 Market Summary:\n{sm}\n\nToken Outlook:\n{out}")

    # Coin analysis
    if intent == 'coin_analysis' and symbol:
        qlist = fetch_quantmetrics(symbol=symbol)
        q = qlist[0] if qlist else {}
        if not q:
            # fallback to AI
            tmai = fetch_tmai(update.message.text)
            ans  = tmai.get('answer','')
            if not ans:
                return await update.message.reply_text("❌ Unable to fetch AI recommendation.")
            ai_sum = summarize(ans)
            toks   = set(re.findall(r"\b[A-Z]{2,5}\b", ans))
            mets   = []
            for t in toks:
                ql = fetch_quantmetrics(symbol=t)
                if ql:
                    qq = ql[0]
                    mets.append(f"{t}: Sharpe={qq.get('SHARPE')}, Vol={qq.get('VOLATILITY')}")
            reply = ai_sum
            if mets:
                reply += "\n\nTech Metrics:\n" + "\n".join(mets)
            return await update.message.reply_text(reply)
        a   = fetch_ai_report(symbol)
        rpt = (
            f"{symbol} Quant Metrics:\n"
            f"- Sharpe: {q.get('SHARPE')}\n"
            f"- Volatility: {q.get('VOLATILITY')}\n"
            f"- Drawdown: {q.get('MAX_DRAWDOWN')}\n"
            f"- Avg Daily Return: {q.get('DAILY_RETURN_AVG')}\n\n"
            f"AI Insights:\n{a.get('INVESTMENT_ANALYSIS','N/A')}"
        )
        return await update.message.reply_text(summarize(rpt))
    # Should buy
    if intent == 'should_buy':
        if symbol:
            qlist = fetch_quantmetrics(symbol=symbol)
            q     = qlist[0] if qlist else {}
            if not q:
                return await update.message.reply_text(f"❌ No data for {symbol}.")
            ok, reason = should_buy_decision(q)
            details   = f"Sharpe={q.get('SHARPE')}, Vol={q.get('VOLATILITY')}, Drawdown={q.get('MAX_DRAWDOWN')}"
            emoji     = "✅" if ok else "❌"
            return await update.message.reply_text(f"{emoji} {reason}\n{details}")
        # otherwise AI+quant fallback
        tmai = fetch_tmai(update.message.text)
        ans  = tmai.get('answer','')
        if not ans:
            return await update.message.reply_text("❌ Unable to fetch AI recommendation.")
        ai_sum = summarize(ans)
        toks   = set(re.findall(r"\b[A-Z]{2,5}\b", ans))
        mets   = []
        for t in toks:
            ql = fetch_quantmetrics(symbol=t)
            if ql:
                qq = ql[0]
                mets.append(f"{t}: Sharpe={qq.get('SHARPE')}, Vol={qq.get('VOLATILITY')}" )
        reply = ai_sum
        if mets:
            reply += "\n\nTech Metrics:\n" + "\n".join(mets)
        return await update.message.reply_text(reply)

    
    tmai = fetch_tmai(update.message.text)
    ans  = tmai.get('answer','')
    if not ans:
        return await update.message.reply_text("❌ Unable to fetch AI response.")
    ai_sum = summarize(ans)
    toks   = set(re.findall(r"\b[A-Z]{2,5}\b", ans))
    mets   = []
    for t in toks:
        ql = fetch_quantmetrics(symbol=t)
        if ql:
            qq = ql[0]
            mets.append(f"{t}: Sharpe={qq.get('SHARPE')}, Vol={qq.get('VOLATILITY')}" )
    reply = ai_sum
    if mets:
        reply += "\n\nTech Metrics:\n" + "\n".join(mets)
    return await update.message.reply_text(reply)


def main():
    app = ApplicationBuilder().token(bot_token).build()
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(on_button)],
        states={
            API_KEY:    [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_api_key)],
            API_SECRET:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_api_secret)]
        },
        fallbacks=[]
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cex", handle_cex))
    app.add_handler(conv)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("auto", auto_trade))
    app.add_handler(CommandHandler("positions", show_positions))

    logging.info("🤖 Bot is up and running.")
    app.run_polling()

if __name__ == "__main__":
    main()    