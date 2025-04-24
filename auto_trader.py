import json
import os
import datetime
import ccxt
from openai import OpenAI
from tokenmetrics import fetch_top_market_cap_tokens
from indicators import (
    fetch_historical,
    calculate_indicators,
    analyze_signals,
    determine_signal,
    calculate_entry_exit,
)

USERS_FILE = "users.json"
POS_FILE   = "positions.json"
openai     = OpenAI()  



def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        return json.load(open(USERS_FILE))
    return {}

def load_positions() -> list:
    if os.path.exists(POS_FILE):
        return json.load(open(POS_FILE))
    return []

def save_positions(positions: list):
    with open(POS_FILE, "w") as f:
        json.dump(positions, f, default=str, indent=2)



def get_binance_client(api_key: str, api_secret: str, futures: bool = False):
    cfg = {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
    if futures:
        cfg["options"] = {"defaultType": "future"}
    return ccxt.binance(cfg)



async def auto_trade(update, context):
    users   = load_users()
    uid     = str(update.effective_user.id)
    keys    = users.get(uid)
    if not keys:
        return await update.message.reply_text("🔗 Please connect Binance via /start first.")

    history = load_positions()
    # block if last trade still open
    if history and history[-1]["status"] == "open":
        last = history[-1]
        return await update.message.reply_text(
            f"🔒 You already have an open trade: {last['symbol']} {last['side']} (Order {last['order_id']})"
        )

    # 1) Top 15 by market cap (excluding USDT/USDC)
    try:
        top15 = fetch_top_market_cap_tokens(top_k=15, page=0)
    except Exception as e:
        return await update.message.reply_text(f"⚠️ Could not fetch top tokens: {e}")

    coins = [t["TOKEN_SYMBOL"] for t in top15 if t["TOKEN_SYMBOL"] not in ("USDT","USDC")]
    if not coins:
        return await update.message.reply_text("⚠️ No non-stablecoins in top list.")

    # 2) Scan for first actionable signal, collecting skip reasons
    skip_reasons = []
    for coin in coins:
        market = f"{coin}/USDT"
        df     = fetch_historical(market)
        df     = calculate_indicators(df)
        side, strength = determine_signal(analyze_signals(df))

        reason = f"{coin}: {side} (strength {strength:.2f})"
        if side in ("Long", "Short"):
            entry, sl, tp = calculate_entry_exit(df, side)
            if entry and sl and tp:
                
                sys  = "You are a crypto analyst. In 1–2 sentences explain why this setup makes sense."
                usr  = (
                    f"Coin: {coin}\n"
                    f"Side: {side}\n"
                    f"RSI: {df['rsi'].iloc[-1]:.2f}, "
                    f"MACD_hist: {df['macd_hist'].iloc[-1]:.4f}\n"
                    f"Entry: {entry:.4f}, EMA20: {df['ema20'].iloc[-1]:.4f}, EMA50: {df['ema50'].iloc[-1]:.4f}"
                )
                ai_summary = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role":"system","content":sys},
                        {"role":"user","content":usr}
                    ],
                ).choices[0].message.content.strip()

                # 3) Attempt to place a 6× $20 futures market order
                client = get_binance_client(keys["key"], keys["secret"], futures=True)
                try:
                    client.set_leverage(6, f"{coin}/USDT")
                    usdt_amt = 20.0
                    qty      = usdt_amt / entry
                    side_cmd = "BUY" if side=="Long" else "SELL"
                    order    = client.create_order(
                        symbol=f"{coin}/USDT",
                        type="MARKET",
                        side=side_cmd,
                        amount=qty
                    )
                except ccxt.InvalidOrder as e:
                    skip_reasons.append(f"{coin}: order failed ({e})")
                    continue   # ← SCAN NEXT COIN
                except Exception as e:
                    skip_reasons.append(f"{coin}: unexpected error ({e})")
                    continue   # ← SCAN NEXT COIN

                # 4) Record success and notify
                new_pos = {
                    "timestamp":    datetime.datetime.utcnow().isoformat(),
                    "order_id":     order["id"],
                    "symbol":       coin,
                    "side":         side,
                    "entry":        entry,
                    "stop_loss":    sl,
                    "take_profit":  tp,
                    "quantity":     qty,
                    "ai_summary":   ai_summary,
                    "status":       "open"
                }
                history.append(new_pos)
                save_positions(history)

                await update.message.reply_text(
                    "✅ Auto-trade executed:\n"
                    f"{coin} | {side}\n"
                    f"Entry: {entry:.4f}\n"
                    f"SL:    {sl:.4f}\n"
                    f"TP:    {tp:.4f}\n"
                    f"Qty:   {qty:.4f}\n"
                    f"OrderID: {order['id']}\n\n"
                    f"💡 Rationale: {ai_summary}"
                )
                return  # stop as soon as one trade succeeds
            else:
                reason += " → missing entry/SL/TP"

        skip_reasons.append(reason + " — skipped")

    
    msg = "🤷 No actionable signals in top 15.\n\n" + "\n".join(skip_reasons)
    return await update.message.reply_text(msg)



async def show_positions(update, context):
    users = load_users()
    uid   = str(update.effective_user.id)
    keys  = users.get(uid)
    if not keys:
        return await update.message.reply_text("🔗 Connect Binance via /start first.")

    history = load_positions()
    if not history:
        return await update.message.reply_text("📂 No trades found.")

    client = get_binance_client(keys["key"], keys["secret"], futures=True)

    # fetch current unrealized PnL for open positions
    try:
        all_pos = client.fapiPrivatev3_get_positionRisk()
    except Exception:
        all_pos = []

    out     = ["📊 All Positions:"]
    changed = False

    for pos in history:
        coin     = pos["symbol"]
        symbol   = f"{coin}/USDT"
        entry    = pos["entry"]
        oid      = pos["order_id"]
        side     = pos["side"]
        qty      = pos.get("quantity", 0.0)
        current  = pos.get("current", 0.0)
        upnl     = pos.get("unrealized_pnl", 0.0)
        pct      = pos.get("pnl_percentage", 0.0)
        status   = pos.get("status", "open")

        # 1 fetch actual order status & average fill price
        try:
            order      = client.fetch_order(oid, symbol)
            ord_status = order["status"]              
            avg_price  = order.get("average", 0.0) or 0.0
            print(f"orders     :{order}")
        except ccxt.OrderNotFound:
         
            rec = next(
                (r for r in all_pos
                 if r["symbol"] == symbol
                 and abs(float(r["positionAmt"])) > 0),
                None
            )
            ord_status = "open" if rec else "closed"
            avg_price  = 0.0
        except Exception:
            ord_status = status
            avg_price  = 0.0

        # if order is fully closed ⇒ realized PnL
        if ord_status == "closed":
            if side == "Long":
                pnl     = (avg_price - entry) * qty
            else:
                pnl     = (entry - avg_price) * qty
            pct     = (pnl / (entry * qty) * 100) if entry and qty else 0.0
            status  = "closed"
            upnl    = pnl
            current = avg_price

        # if still open, recalc unrealized from positionRisk
        elif ord_status == "open":
            rec = next((r for r in all_pos if r["symbol"] == symbol), None)
            status = "open"
            if rec and abs(float(rec["positionAmt"])) > 0:
                qty      = abs(float(rec["positionAmt"]))
                current  = float(rec["markPrice"])
                upnl     = float(rec["unRealizedProfit"])
                pct      = (upnl / (entry * qty) * 100) if entry and qty else 0.0

        # update stored record if changed
        if (pos.get("status")        != status
            or pos.get("current")     != current
            or pos.get("unrealized_pnl") != upnl):
            pos["status"]         = status
            pos["current"]        = current
            pos["unrealized_pnl"] = upnl
            pos["pnl_percentage"] = pct
            changed = True

        # output
        out.append(
            f"\n— Order {oid}\n"
            f"{coin} {side}\n"
            f"Entry:   {entry:.4f}\n"
            f"Current: {current:.4f}\n"
            f"Qty:     {qty:.4f}\n"
            f"PnL:     ${upnl:.2f} ({pct:.2f}%)\n"
            f"Status:  {status}\n"
            f"💡 Rationale: {pos['ai_summary']}"
        )

    if changed:
        save_positions(history)

    await update.message.reply_text("\n".join(out))
