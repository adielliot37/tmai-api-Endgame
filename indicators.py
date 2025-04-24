import numpy as np
np.NaN = np.nan
import os
import requests
import ccxt
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from openai import OpenAI

from typing import Dict, Tuple, List, Optional



load_dotenv()
TM_API_KEY = os.getenv("TOKENMETRICS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai = OpenAI()  


RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
RISK_REWARD_RATIO = 2.0  
DEFAULT_WIN_RATE = 0.55


def fetch_ai_overview(symbol: str) -> str:
    """Fetch AI analysis from Token Metrics."""
    url = "https://api.tokenmetrics.com/v2/tmai"
    headers = {
        "accept": "application/json",
        "api_key": TM_API_KEY,
        "content-type": "application/json",
    }
    payload = {
        "messages": [
            {
                "user": (
                    f"Should I long or short on {symbol} now "
                    "if yes what should be my entry price and stop loss"
                )
            }
        ]
    }
    try:
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json().get("answer", "")
    except Exception as e:
        return f"Error fetching AI overview: {str(e)}"


def fetch_historical(symbol: str, timeframe="4h", limit=200) -> pd.DataFrame:
    """Fetch historical OHLCV data with error handling."""
    try:
        ex = ccxt.binance()
        bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts")
        
       
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = df[col].astype(float)
        
        return df
    except Exception as e:
        print(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()




def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate multiple technical indicators using pandas_ta."""
    if df.empty:
        return df
    
    
    df = df.copy()
    df.rename(columns={'vol': 'volume'}, inplace=True)
    
    
    MyStrategy = ta.Strategy(
        name="Multi-Indicator Strategy",
        description="SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, OBV, CCI",
        ta=[
            # Trend indicators
            {"kind": "sma", "length": 20},
            {"kind": "sma", "length": 50},
            {"kind": "ema", "length": 20},
            {"kind": "ema", "length": 50},
            
            # Momentum indicators
            {"kind": "rsi", "length": RSI_PERIOD},
            
            # MACD
            {"kind": "macd", "fast": MACD_FAST, "slow": MACD_SLOW, "signal": MACD_SIGNAL},
            
            # Bollinger Bands
            {"kind": "bbands", "length": BOLLINGER_PERIOD, "std": BOLLINGER_STD},
            
            # Volatility indicators
            {"kind": "atr", "length": ATR_PERIOD},
            
            # Stochastic
            {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3},
            
            # ADX - Trend strength
            {"kind": "adx", "length": 14},
            
            # On Balance Volume
            {"kind": "obv"},
            
            # CCI - Commodity Channel Index
            {"kind": "cci", "length": 14},
        ]
    )
    
  
    df.ta.strategy(MyStrategy)
    
    
    column_mapping = {
        'RSI_14': 'rsi',
        'ATR_14': 'atr',
        'SMA_20': 'sma20',
        'SMA_50': 'sma50',
        'EMA_20': 'ema20',
        'EMA_50': 'ema50',
        'BBL_20_2.0': 'bb_lower',
        'BBM_20_2.0': 'bb_middle',
        'BBU_20_2.0': 'bb_upper',
        'MACD_12_26_9': 'macd',
        'MACDs_12_26_9': 'macd_signal',
        'MACDh_12_26_9': 'macd_hist',
        'STOCHk_14_3_3': 'stoch_k',
        'STOCHd_14_3_3': 'stoch_d',
        'ADX_14': 'adx',
        'CCI_14': 'cci'
    }
    
    
    if 'ATR_14' not in df.columns:
        df['ATR_14'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=ATR_PERIOD)
    
    if 'CCI_14' not in df.columns:
        df['CCI_14'] = ta.cci(high=df['high'], low=df['low'], close=df['close'], length=14)
    
   
    columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
    df.rename(columns=columns_to_rename, inplace=True)
    
    
    print(f"Available indicators: {df.columns.tolist()}")
    
    return df

def analyze_signals(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze various indicators and return signal strengths."""
    if df.empty:
        return {'long': 0, 'short': 0}
    
    signals = {'long': 0, 'short': 0}
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # RSI signals
    if 'rsi' in current:
        if current['rsi'] < RSI_OVERSOLD:
            signals['long'] += 1
        elif current['rsi'] > RSI_OVERBOUGHT:
            signals['short'] += 1
    
    # MACD signals
    if 'macd_hist' in current and 'macd_hist' in previous:
        if current['macd_hist'] > 0 and previous['macd_hist'] < 0:  # Bullish crossover
            signals['long'] += 1
        elif current['macd_hist'] < 0 and previous['macd_hist'] > 0:  # Bearish crossover
            signals['short'] += 1
    
    # Moving average signals
    if all(col in current for col in ['close', 'ema20', 'ema50']):
        if current['close'] > current['ema20'] > current['ema50']:
            signals['long'] += 0.5
        elif current['close'] < current['ema20'] < current['ema50']:
            signals['short'] += 0.5
    
    # Bollinger Bands signals
    if all(col in current for col in ['close', 'bb_lower', 'bb_upper']):
        if current['close'] < current['bb_lower']:
            signals['long'] += 0.75  # Potential oversold
        elif current['close'] > current['bb_upper']:
            signals['short'] += 0.75  # Potential overbought
    
    # Stochastic signals
    if all(col in current for col in ['stoch_k', 'stoch_d']):
        if current['stoch_k'] < 20 and current['stoch_d'] < 20:
            signals['long'] += 0.5
        elif current['stoch_k'] > 80 and current['stoch_d'] > 80:
            signals['short'] += 0.5
    
    # ADX - Strong trend confirmation
    if 'adx' in current:
        if current['adx'] > 25:
            if signals['long'] > signals['short']:
                signals['long'] += 0.5
            elif signals['short'] > signals['long']:
                signals['short'] += 0.5
    
    # CCI signals
    if 'cci' in current:
        if current['cci'] < -100:
            signals['long'] += 0.5
        elif current['cci'] > 100:
            signals['short'] += 0.5
    
    return signals


def determine_signal(signals: Dict[str, float]) -> Tuple[str, float]:
    """Determine the final signal based on signal strengths."""
    long_strength = signals.get('long', 0)
    short_strength = signals.get('short', 0)
    
    signal_strength = abs(long_strength - short_strength)
    
    if long_strength > short_strength + 1:  # Requiring stronger confirmation
        return "Long", signal_strength
    elif short_strength > long_strength + 1:
        return "Short", signal_strength
    else:
        return "Neutral", signal_strength


def calculate_entry_exit(df: pd.DataFrame, signal: str) -> Tuple[float, float, float]:
    """Calculate entry, stop loss, and take profit based on signal and ATR."""
    if df.empty or signal == "Neutral":
        return 0, 0, 0
    
    current = df.iloc[-1]
    entry_price = current['close']
    
    # Check for valid entry price
    if entry_price <= 0:
        print("Warning: Entry price is zero or negative. Using alternative value.")
        return 0, 0, 0
    
    # Check if ATR exists, if not use a percentage of price
    if 'atr' in current and not pd.isna(current['atr']) and current['atr'] > 0:
        atr = current['atr']
    else:
        # Default to 1.5% of price as ATR substitute
        atr = entry_price * 0.015
    
    # Use ATR to determine stop loss distance
    atr_multiplier = 1.5 
    
    if signal == "Long":
        stop_loss = entry_price - (atr * atr_multiplier)
        take_profit = entry_price + ((entry_price - stop_loss) * RISK_REWARD_RATIO)
    else:  # Short
        stop_loss = entry_price + (atr * atr_multiplier)
        take_profit = entry_price - ((stop_loss - entry_price) * RISK_REWARD_RATIO)
    
    return entry_price, stop_loss, take_profit


def kelly_stop_loss(entry: float, win_rate=DEFAULT_WIN_RATE, rr=RISK_REWARD_RATIO) -> float:
    """
    Calculate stop loss using Kelly Criterion.
    f* = (win_rate*RR - (1-win_rate)) / RR
    SL = entry * (1 - f*) if f*>0 else entry
    """
    f = (win_rate * rr - (1 - win_rate)) / rr
    
   
    f = min(f, 0.25)
    
    if f <= 0:
        return entry * 0.95 
    
    if entry > 0:
        return entry * (1 - f)
    return entry


def adjust_win_rate(signal_strength: float) -> float:
    """Adjust win rate based on signal strength."""
   
    return min(DEFAULT_WIN_RATE + (signal_strength * 0.02), 0.65)


def detect_trend(df: pd.DataFrame) -> str:
    """Detect overall market trend."""
    if df.empty:
        return "Unknown"
    
   
    current = df.iloc[-1]
    
   
    if not all(col in current.index for col in ['close', 'sma50', 'sma20']):
        return "Unknown"
    
    if current['close'] > current['sma50'] and current['sma20'] > current['sma50']:
        return "Bullish"
    elif current['close'] < current['sma50'] and current['sma20'] < current['sma50']:
        return "Bearish"
    else:
        return "Sideways"

def summarize_via_gpt(system_prompt: str, user_prompt: str) -> str:
    """Generate a summary using OpenAI's GPT model."""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def generate_detailed_report(df: pd.DataFrame, symbol: str, signal: str, 
                            entry: float, stop_loss: float, take_profit: float,
                            signal_strength: float, ai_overview: str) -> str:
    """Generate a detailed report of the analysis."""
    if df.empty:
        return "No data available for analysis."
    
    current = df.iloc[-1]
    trend = detect_trend(df)
    
    
    def safe_get(column, default="N/A", format_str="{:.2f}"):
        if column in current and not pd.isna(current[column]):
            return format_str.format(current[column])
        return default
    
   
    def safe_percentage(a, b):
        if b == 0: 
            return "N/A"
        return f"{abs(a - b) / b * 100:.2f}%"
    
    report = f"""
### Technical Analysis Report for {symbol}

#### Overall Market Trend: {trend}

#### Key Indicators:
- RSI(14): {safe_get('rsi')} ({'Oversold' if current.get('rsi', 50) < 30 else 'Overbought' if current.get('rsi', 50) > 70 else 'Neutral'})
- MACD: {safe_get('macd_hist', format_str="{:.4f}")} ({'Bullish' if current.get('macd_hist', 0) > 0 else 'Bearish'})
- Stochastic: K={safe_get('stoch_k')}, D={safe_get('stoch_d')}
- ADX: {safe_get('adx')} ({'Strong Trend' if current.get('adx', 0) > 25 else 'Weak Trend'})
- CCI: {safe_get('cci')}
- ATR: {safe_get('atr', format_str="{:.4f}")}
- Price in relation to Bollinger Bands: {
    'Below Lower Band (Potential Oversold)' if 'bb_lower' in current and current['close'] < current['bb_lower'] 
    else 'Above Upper Band (Potential Overbought)' if 'bb_upper' in current and current['close'] > current['bb_upper'] 
    else 'Within Bands (Neutral)'
}

#### Signal Strength: {signal_strength:.2f}/5

#### Trade Recommendation:
- Signal: {signal}
- Entry Price: {f"{entry:.4f}" if entry != 0 else "N/A"}
"""

    # Only add stop loss and take profit if entry price is not zero
    if entry != 0:
        report += f"""- Stop Loss: {stop_loss:.4f} ({safe_percentage(stop_loss, entry)} from entry)
- Take Profit: {take_profit:.4f} ({safe_percentage(take_profit, entry)} from entry)
- Risk-Reward Ratio: {RISK_REWARD_RATIO:.1f}
"""
    else:
        report += "- Stop Loss: N/A\n- Take Profit: N/A\n- Risk-Reward Ratio: N/A\n"

    report += f"""
#### AI Market Sentiment:
{ai_overview}
"""
    return report


def main():
    #  input()/print() 
    symbol = input("Enter symbol (e.g. ETH/USDT): ").strip().upper()
    timeframe = input("Enter timeframe (default: 1h): ").strip() or "1h"
    #  CLI workflow

# gaurd
if __name__ == "__main__":
    main()