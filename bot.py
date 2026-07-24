# BTC Bot – Relaxed SMC with Supply/Demand (15min)
import encodings.idna
import os, logging, requests, threading, numpy as np
from datetime import datetime, timezone
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")
CHAT_ID, RUN_SIGNALS = None, False

# -------- CONFIG ---------
TIMEFRAME = "15min"
PRICE_INTERVAL_SECONDS = 900          # 15 minutes
RISK_REWARD_MULTIPLIER = 2.0
MIN_STOP_POINTS = 200                # wider for BTC
MAX_DAILY_LOSSES = 6

ACTIVE_POSITIONS = []
STATS = {"total_signals":0,"tp1_hits":0,"tp2_hits":0,"sl_hits":0,"daily_losses":0}
SIGNAL_HISTORY = []

FREE_CHANNEL_ID = -1004410090098
VIP_CHANNEL_ID = -1004416190238
HISTORY_CHANNEL_ID = FREE_CHANNEL_ID

app = Flask(__name__)
@app.route('/')
def home():
    return "BTC Bot (15min Relaxed) is running!"

cached_candles = []
last_fetch_time = 0

def fetch_real_candles():
    global cached_candles, last_fetch_time
    now = datetime.now().timestamp()
    if cached_candles and (now - last_fetch_time) < 60:
        return cached_candles
    url = f"https://api.twelvedata.com/time_series?symbol=BTC/USD&interval={TIMEFRAME}&outputsize=30&apikey={TWELVE_DATA_KEY}"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if data.get("status") == "ok" and "values" in data:
            candles = []
            for bar in reversed(data["values"]):
                candles.append({
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "date": bar["datetime"]
                })
            cached_candles = candles
            last_fetch_time = now
            logger.info(f"Fetched {len(candles)} {TIMEFRAME} BTC candles. Price: ${candles[-1]['close']:.2f}")
            return candles
    except Exception as e:
        logger.error(f"API error: {e}")
    return cached_candles

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return MIN_STOP_POINTS
    tr_list = []
    for i in range(1, len(candles)):
        high, low, prev_close = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
    return np.mean(tr_list[-period:]) if tr_list else MIN_STOP_POINTS

def calculate_ema(closes, period=20):
    if len(closes) < period:
        return np.mean(closes) if closes else 0
    alpha = 2 / (period + 1)
    ema = np.mean(closes[:period])
    for price in closes[period:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def find_swing_levels(candles, lookback=20):
    if len(candles) < lookback + 2:
        return None, None
    highs = [c["high"] for c in candles[-lookback:]]
    lows = [c["low"] for c in candles[-lookback:]]
    swing_highs, swing_lows = [], []
    for i in range(1, len(highs)-1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append(lows[i])
    resistance = max(swing_highs[-3:]) if swing_highs else max(highs)
    support = min(swing_lows[-3:]) if swing_lows else min(lows)
    return resistance, support

def detect_fvg(candles):
    if len(candles) < 3:
        return None
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if c1["high"] < c3["low"] and c3["close"] > c3["open"]:
        body = abs(c3["close"] - c3["open"])
        rng = c3["high"] - c3["low"]
        if rng > 0 and (body / rng) > 0.25:
            return "BUY"
    if c1["low"] > c3["high"] and c3["close"] < c3["open"]:
        body = abs(c3["close"] - c3["open"])
        rng = c3["high"] - c3["low"]
        if rng > 0 and (body / rng) > 0.25:
            return "SELL"
    return None

def detect_order_blocks(candles, lookback=8):
    if len(candles) < lookback+2:
        return None, None
    bullish_ob, bearish_ob = None, None
    for i in range(len(candles)-lookback, len(candles)-1):
        if i+1 >= len(candles): continue
        c = candles[i]; nxt = candles[i+1]
        if c["close"] < c["open"] and nxt["close"] > nxt["open"] and nxt["close"] > c["high"]:
            bullish_ob = {"high":c["high"], "low":c["low"]}
        if c["close"] > c["open"] and nxt["close"] < nxt["open"] and nxt["close"] < c["low"]:
            bearish_ob = {"high":c["high"], "low":c["low"]}
    return bullish_ob, bearish_ob

def detect_choch(candles):
    if len(candles) < 8: return None
    highs = [c["high"] for c in candles[-8:]]
    lows = [c["low"] for c in candles[-8:]]
    swing_highs, swing_lows = [], []
    for i in range(1, len(highs)-1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append(lows[i])
    if len(swing_highs) < 2 or len(swing_lows) < 2: return None
    current = candles[-1]["close"]
    if len(swing_highs)>=2 and current > swing_highs[-2]: return "BULLISH"
    if len(swing_lows)>=2 and current < swing_lows[-2]: return "BEARISH"
    return None

def price_near_zone(price, zone, atr):
    if zone is None: return False
    return abs(price - zone) < atr

def process_signals():
    global RISK_REWARD_MULTIPLIER, STATS
    candles = fetch_real_candles()
    if not candles or len(candles) < 8: return None
    if STATS["daily_losses"] >= MAX_DAILY_LOSSES: return None

    closes = [c["close"] for c in candles]
    current_price = closes[-1]
    atr = calculate_atr(candles)

    resistance, support = find_swing_levels(candles)

    ema_fast = calculate_ema(closes, 10)
    ema_slow = calculate_ema(closes, 20)
    trend_up = ema_fast > ema_slow and current_price > ema_fast
    trend_down = ema_fast < ema_slow and current_price < ema_fast

    fvg = detect_fvg(candles)
    bullish_ob, bearish_ob = detect_order_blocks(candles)
    choch = detect_choch(candles)

    sig = None
    reason = ""
    grade = "C"
    score_val = 0

    # --- BUY setup ---
    if fvg == "BUY" or choch == "BULLISH" or (bullish_ob and price_near_zone(current_price, bullish_ob["low"], atr)):
        score = 0
        reasons = []
        if fvg == "BUY": score += 20; reasons.append("FVG")
        if choch == "BULLISH": score += 25; reasons.append("CHoCH")
        if bullish_ob and current_price <= bullish_ob["high"] and current_price >= bullish_ob["low"]:
            score += 10; reasons.append("OB")
        if trend_up: score += 10; reasons.append("Trend↑")
        if support and price_near_zone(current_price, support, atr):
            score += 15; reasons.append("DemandZone")
        last = candles[-1]
        rng = last["high"] - last["low"]
        if rng > 0:
            body_ratio = abs(last["close"] - last["open"]) / rng
            if body_ratio > 0.5: score += 5; reasons.append("StrongCandle")

        if score >= 45:
            stop_distance = max(atr * 1.5, MIN_STOP_POINTS)
            sig = "BUY"
            reason = " + ".join(reasons) + f" | ATR:{atr:.1f}"
            sl = current_price - stop_distance
            tp1 = current_price + (stop_distance * RISK_REWARD_MULTIPLIER)
            tp2 = current_price + (stop_distance * RISK_REWARD_MULTIPLIER * 2.0)
            grade = "A" if score >= 70 else ("B" if score >= 55 else "C")
            score_val = score

    # --- SELL setup ---
    elif fvg == "SELL" or choch == "BEARISH" or (bearish_ob and price_near_zone(current_price, bearish_ob["high"], atr)):
        score = 0
        reasons = []
        if fvg == "SELL": score += 20; reasons.append("FVG")
        if choch == "BEARISH": score += 25; reasons.append("CHoCH")
        if bearish_ob and current_price <= bearish_ob["high"] and current_price >= bearish_ob["low"]:
            score += 10; reasons.append("OB")
        if trend_down: score += 10; reasons.append("Trend↓")
        if resistance and price_near_zone(current_price, resistance, atr):
            score += 15; reasons.append("SupplyZone")
        last = candles[-1]
        rng = last["high"] - last["low"]
        if rng > 0:
            body_ratio = abs(last["close"] - last["open"]) / rng
            if body_ratio > 0.5: score += 5; reasons.append("StrongCandle")

        if score >= 45:
            stop_distance = max(atr * 1.5, MIN_STOP_POINTS)
            sig = "SELL"
            reason = " + ".join(reasons) + f" | ATR:{atr:.1f}"
            sl = current_price + stop_distance
            tp1 = current_price - (stop_distance * RISK_REWARD_MULTIPLIER)
            tp2 = current_price - (stop_distance * RISK_REWARD_MULTIPLIER * 2.0)
            grade = "A" if score >= 70 else ("B" if score >= 55 else "C")
            score_val = score

    if sig:
        STATS["total_signals"] += 1
        return {"type":sig,"reason":reason,"entry":current_price,"sl":sl,"tp1":tp1,"tp2":tp2,
                "status":"PENDING","grade":grade,"score":score_val}
    return None

async def monitor_positions(bot, price):
    global ACTIVE_POSITIONS, CHAT_ID, STATS, SIGNAL_HISTORY
    surv = []
    for p in ACTIVE_POSITIONS:
        if p["status"] == "PENDING":
            if (price <= p["entry"]) if p["type"] == "BUY" else (price >= p["entry"]):
                p["status"] = "ACTIVE"
                await bot.send_message(chat_id=CHAT_ID, text=f"✅ BTC {p['type']} EXECUTED at ${price:.2f}")
            surv.append(p); continue
        if p["type"] == "BUY":
            if price <= p["sl"]:
                STATS["sl_hits"] += 1; STATS["daily_losses"] += 1
                await bot.send_message(chat_id=CHAT_ID, text=f"🔴 BTC SL HIT ${p['sl']:.2f}")
                SIGNAL_HISTORY.append({"type":p["type"],"entry":p["entry"],"exit":price,"result":"SL","grade":p.get("grade","C"),"time":datetime.now(timezone.utc).strftime("%H:%M UTC")})
                await bot.send_message(chat_id=HISTORY_CHANNEL_ID, text=f"❌ BTC {p['type']} SL\nGrade: {p.get('grade','C')}\nEntry: ${p['entry']:.2f}\nExit: ${price:.2f}")
            elif price >= p["tp2"]:
                STATS["tp2_hits"] += 1
                await bot.send_message(chat_id=CHAT_ID, text=f"👑 BTC TP2 ${p['tp2']:.2f}")
                SIGNAL_HISTORY.append({"type":p["type"],"entry":p["entry"],"exit":price,"result":"TP2","grade":p.get("grade","C"),"time":datetime.now(timezone.utc).strftime("%H:%M UTC")})
                await bot.send_message(chat_id=HISTORY_CHANNEL_ID, text=f"✅ BTC {p['type']} TP2\nGrade: {p.get('grade','C')}\nEntry: ${p['entry']:.2f}\nExit: ${price:.2f}")
            elif price >= p["tp1"] and not p.get("tp1_hit"):
                p["tp1_hit"] = True; STATS["tp1_hits"] += 1
                p["sl"] = p["entry"]
                await bot.send_message(chat_id=CHAT_ID, text=f"💰 BTC TP1 ${p['tp1']:.2f} | SL→BE 🔒")
                surv.append(p)
            else: surv.append(p)
        elif p["type"] == "SELL":
            if price >= p["sl"]:
                STATS["sl_hits"] += 1; STATS["daily_losses"] += 1
                await bot.send_message(chat_id=CHAT_ID, text=f"🔴 BTC SL HIT ${p['sl']:.2f}")
                SIGNAL_HISTORY.append({"type":p["type"],"entry":p["entry"],"exit":price,"result":"SL","grade":p.get("grade","C"),"time":datetime.now(timezone.utc).strftime("%H:%M UTC")})
                await bot.send_message(chat_id=HISTORY_CHANNEL_ID, text=f"❌ BTC {p['type']} SL\nGrade: {p.get('grade','C')}\nEntry: ${p['entry']:.2f}\nExit: ${price:.2f}")
            elif price <= p["tp2"]:
                STATS["tp2_hits"] += 1
                await bot.send_message(chat_id=CHAT_ID, text=f"👑 BTC TP2 ${p['tp2']:.2f}")
                SIGNAL_HISTORY.append({"type":p["type"],"entry":p["entry"],"exit":price,"result":"TP2","grade":p.get("grade","C"),"time":datetime.now(timezone.utc).strftime("%H:%M UTC")})
                await bot.send_message(chat_id=HISTORY_CHANNEL_ID, text=f"✅ BTC {p['type']} TP2\nGrade: {p.get('grade','C')}\nEntry: ${p['entry']:.2f}\nExit: ${price:.2f}")
            elif price <= p["tp1"] and not p.get("tp1_hit"):
                p["tp1_hit"] = True; STATS["tp1_hits"] += 1
                p["sl"] = p["entry"]
                await bot.send_message(chat_id=CHAT_ID, text=f"💰 BTC TP1 ${p['tp1']:.2f} | SL→BE 🔒")
                surv.append(p)
            else: surv.append(p)
    ACTIVE_POSITIONS = surv

async def signal_loop(context: ContextTypes.DEFAULT_TYPE):
    global RUN_SIGNALS, CHAT_ID, ACTIVE_POSITIONS
    if not RUN_SIGNALS or not CHAT_ID: return
    candles = fetch_real_candles()
    if candles:
        live = candles[-1]["close"]
        if ACTIVE_POSITIONS: await monitor_positions(context.bot, live)
        sig = process_signals()
        if sig:
            ACTIVE_POSITIONS.append(sig)
            grade = sig.get("grade","C"); score = sig.get("score",0)
            vip_msg = (f"🟠 BTC {grade} {sig['type']} SIGNAL\n"
                       f"Score: {score}/100\nEntry: ${sig['entry']:.2f}\nSL: ${sig['sl']:.2f}\n"
                       f"TP1: ${sig['tp1']:.2f}\nTP2: ${sig['tp2']:.2f}\n"
                       f"Reason: {sig['reason']}\n⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n🔒 VIP Instant Signal")
            free_msg = (f"🟠 BTC {grade} {sig['type']} SIGNAL\nScore: {score}/100\nEntry: ${sig['entry']:.2f}\n"
                        f"SL: ${sig['sl']:.2f}\nTP1: ${sig['tp1']:.2f}\n\n⚡ Full details in VIP: /join_vip")
            await context.bot.send_message(chat_id=VIP_CHANNEL_ID, text=vip_msg)
            await context.bot.send_message(chat_id=FREE_CHANNEL_ID, text=free_msg)
            await context.bot.send_message(chat_id=CHAT_ID, text=vip_msg)

async def report_callback(context: ContextTypes.DEFAULT_TYPE):
    global CHAT_ID, STATS
    if not CHAT_ID: return
    total = STATS["tp1_hits"] + STATS["tp2_hits"] + STATS["sl_hits"]
    wr = ((STATS["tp1_hits"]+STATS["tp2_hits"])/total*100) if total>0 else 0
    await context.bot.send_message(chat_id=CHAT_ID, text=f"📅 DAILY BTC\nSignals: {STATS['total_signals']}\nTP1: {STATS['tp1_hits']} TP2: {STATS['tp2_hits']}\nSL: {STATS['sl_hits']}\nWin: {wr:.1f}%")
    STATS["total_signals"]=STATS["tp1_hits"]=STATS["tp2_hits"]=STATS["sl_hits"]=STATS["daily_losses"]=0

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CHAT_ID; CHAT_ID = update.effective_chat.id
    await update.message.reply_text(f"🟠 BTC SMC ({TIMEFRAME})\n/start_signals /stop_signals /status /report /history /join_vip")

async def start_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RUN_SIGNALS, CHAT_ID
    CHAT_ID = update.effective_chat.id
    if RUN_SIGNALS: await update.message.reply_text("Already running"); return
    RUN_SIGNALS = True
    context.job_queue.run_repeating(signal_loop, interval=PRICE_INTERVAL_SECONDS, name="btc_job")
    context.job_queue.run_repeating(report_callback, interval=86400, first=86400, name="report_job")
    await update.message.reply_text("🚀 BTC scanning started (15min)")

async def stop_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RUN_SIGNALS
    RUN_SIGNALS = False
    for j in context.job_queue.get_jobs_by_name("btc_job"): j.schedule_removal()
    for j in context.job_queue.get_jobs_by_name("report_job"): j.schedule_removal()
    await update.message.reply_text("⏸️ Stopped")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RUN_SIGNALS, ACTIVE_POSITIONS, STATS
    candles = fetch_real_candles()
    price = candles[-1]["close"] if candles else "N/A"
    count = len(candles) if candles else 0
    await update.message.reply_text(f"📊 BTC State: {'ACTIVE' if RUN_SIGNALS else 'IDLE'}\nPrice: ${price}\nCandles: {count}/30\nTrades: {len(ACTIVE_POSITIONS)}\nLosses: {STATS['daily_losses']}/{MAX_DAILY_LOSSES}")

async def manual_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global STATS
    total = STATS["tp1_hits"] + STATS["tp2_hits"] + STATS["sl_hits"]
    wr = ((STATS["tp1_hits"]+STATS["tp2_hits"])/total*100) if total>0 else 0
    await update.message.reply_text(f"📝 BTC Signals: {STATS['total_signals']}\nTP1: {STATS['tp1_hits']} TP2: {STATS['tp2_hits']}\nSL: {STATS['sl_hits']}\nWin: {wr:.1f}%")

async def signal_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SIGNAL_HISTORY
    if not SIGNAL_HISTORY: await update.message.reply_text("No closed trades yet."); return
    last10 = SIGNAL_HISTORY[-10:]
    msg = "📜 LAST 10 BTC TRADES\n\n"
    for t in reversed(last10):
        emoji = "✅" if t["result"] != "SL" else "❌"
        msg += f"{emoji} {t['type']} {t['result']} | {t['grade']} | {t['time']}\n"
    wins = sum(1 for t in SIGNAL_HISTORY if t["result"] != "SL")
    total = len(SIGNAL_HISTORY)
    wr = (wins/total*100) if total>0 else 0
    msg += f"\n📈 Win Rate: {wr:.0f}% ({wins}/{total})"
    await update.message.reply_text(msg)

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global PRICE_INTERVAL_SECONDS, RUN_SIGNALS
    if not context.args: await update.message.reply_text("/set_interval 900"); return
    val = int(context.args[0])
    if val < 60: await update.message.reply_text("Min 60s"); return
    PRICE_INTERVAL_SECONDS = val
    if RUN_SIGNALS:
        for j in context.job_queue.get_jobs_by_name("btc_job"): j.schedule_removal()
        context.job_queue.run_repeating(signal_loop, interval=val, name="btc_job")
    await update.message.reply_text(f"✅ {val}s")

async def set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RISK_REWARD_MULTIPLIER
    if not context.args: await update.message.reply_text("/set_risk 2.0"); return
    val = float(context.args[0])
    if val < 0.5: await update.message.reply_text("Min 0.5"); return
    RISK_REWARD_MULTIPLIER = val
    await update.message.reply_text(f"✅ RR: {val}x")

async def join_vip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔒 *XAUUSD VIP Signals*\n\n💰 *$25/month*\n💎 USDT (TRC20): `TFEYT12uggMhmhncqFSc8SAFzpdz6YfS2j`\n✅ Send screenshot to @pipzoe", parse_mode="Markdown")

application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("start_signals", start_signals))
application.add_handler(CommandHandler("stop_signals", stop_signals))
application.add_handler(CommandHandler("status", status))
application.add_handler(CommandHandler("report", manual_report))
application.add_handler(CommandHandler("history", signal_history))
application.add_handler(CommandHandler("set_interval", set_interval))
application.add_handler(CommandHandler("set_risk", set_risk))
application.add_handler(CommandHandler("join_vip", join_vip))

if __name__ == "__main__":
    def run_flask():
        port = int(os.getenv("PORT", "10000"))
        app.run(host="0.0.0.0", port=port, use_reloader=False)
    threading.Thread(target=run_flask, daemon=True).start()
    application.run_polling()