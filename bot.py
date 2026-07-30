# BTC/USD – Advanced MTF (4H+1H+15min), SMC, Liquidity, Key Levels, Candlestick Confirmation
import encodings.idna
import os, logging, requests, threading, numpy as np, asyncio
from datetime import datetime, timezone, timedelta, time
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")
CHAT_ID, RUN_SIGNALS = None, False

SYMBOL = "BTC/USD"
TIMEFRAME = "15min"
PRICE_INTERVAL_SECONDS = 900
RISK_REWARD_MULTIPLIER = 2.0
MIN_STOP_POINTS = 200
MAX_DAILY_LOSSES = 3
MIN_ATR_15M = 150

ACTIVE_POSITIONS = []
STATS = {"total_signals":0,"tp1_hits":0,"tp2_hits":0,"sl_hits":0,"daily_losses":0}
SIGNAL_HISTORY = []

FREE_CHANNEL_ID = -1004410090098      # @XAU_EDGE or your BTC free channel
VIP_CHANNEL_ID = -1004416190238
HISTORY_CHANNEL_ID = FREE_CHANNEL_ID

app = Flask(__name__)
@app.route('/')
def home():
    return "BTC Bot (Advanced MTF) is running!"

cached_candles_15m = []
last_fetch_time = 0

def fetch_real_candles():
    global cached_candles_15m, last_fetch_time
    now = datetime.now().timestamp()
    if cached_candles_15m and (now - last_fetch_time) < 60:
        return cached_candles_15m
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={TIMEFRAME}&outputsize=30&apikey={TWELVE_DATA_KEY}"
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
            cached_candles_15m = candles
            last_fetch_time = now
            logger.info(f"Fetched {len(candles)} 15m BTC candles. Price: ${candles[-1]['close']:.2f}")
            return candles
    except Exception as e:
        logger.error(f"API error: {e}")
    return cached_candles_15m

# ---------- Helper indicators ----------
def calculate_ema(closes, period=20):
    if len(closes) < period:
        return np.mean(closes) if closes else 0
    alpha = 2 / (period + 1)
    ema = np.mean(closes[:period])
    for price in closes[period:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return MIN_STOP_POINTS
    tr_list = []
    for i in range(1, len(candles)):
        high, low, prev_close = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
    return np.mean(tr_list[-period:]) if tr_list else MIN_STOP_POINTS

def calculate_rsi(prices, period=7):
    if len(prices) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        gains.append(diff if diff > 0 else 0)
        losses.append(abs(diff) if diff < 0 else 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    return 100 - (100 / (1 + avg_gain / avg_loss))

def find_swing_points(candles, lookback=20):
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

# ---------- SMC detection on any timeframe ----------
def detect_fvg(candles):
    if len(candles) < 3: return None
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if c1["high"] < c3["low"] and c3["close"] > c3["open"]:
        body = abs(c3["close"] - c3["open"])
        rng = c3["high"] - c3["low"]
        if rng > 0 and (body / rng) > 0.25: return "BUY"
    if c1["low"] > c3["high"] and c3["close"] < c3["open"]:
        body = abs(c3["close"] - c3["open"])
        rng = c3["high"] - c3["low"]
        if rng > 0 and (body / rng) > 0.25: return "SELL"
    return None

def detect_order_blocks(candles, lookback=8):
    if len(candles) < lookback+2: return None, None
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
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]: swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]: swing_lows.append(lows[i])
    if len(swing_highs) < 2 or len(swing_lows) < 2: return None
    current = candles[-1]["close"]
    if len(swing_highs)>=2 and current > swing_highs[-2]: return "BULLISH"
    if len(swing_lows)>=2 and current < swing_lows[-2]: return "BEARISH"
    return None

def detect_bos(candles):
    # similar to CHoCH but with larger lookback
    return detect_choch(candles)  # for simplicity we use same function

def detect_sr_bounce(candles, atr):
    if len(candles) < 3: return None, None
    resistance, support = find_swing_points(candles)
    if resistance is None or support is None: return None, None
    prev = candles[-2]; curr = candles[-1]
    price = curr["close"]
    if (abs(prev["low"] - support) < atr * 0.5 and
        curr["close"] > curr["open"] and curr["close"] > prev["close"]):
        return "SUPPORT", "BUY"
    if (abs(prev["high"] - resistance) < atr * 0.5 and
        curr["close"] < curr["open"] and curr["close"] < prev["close"]):
        return "RESISTANCE", "SELL"
    return None, None

# ---------- MTF data fetch ----------
def fetch_tf_candles(symbol, interval="4h", outputsize=30):
    api_key = os.getenv("TWELVE_DATA_KEY")
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
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
            return candles
    except:
        pass
    return []

# ---------- 4H Analysis ----------
def analyze_4h(candles_4h):
    if len(candles_4h) < 20:
        return {"direction": None, "key_resistance": None, "key_support": None, "near_zone": None}
    closes = [c["close"] for c in candles_4h]
    current = closes[-1]
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    direction = None
    if ema20 > ema50 and current > ema20:
        direction = "BULLISH"
    elif ema20 < ema50 and current < ema20:
        direction = "BEARISH"

    resistance, support = find_swing_points(candles_4h, lookback=20)
    near_zone = None
    atr4h = calculate_atr(candles_4h, 14)
    if support and abs(current - support) < atr4h * 1.5:
        near_zone = "SUPPORT"
    elif resistance and abs(current - resistance) < atr4h * 1.5:
        near_zone = "RESISTANCE"

    return {
        "direction": direction,
        "key_resistance": resistance,
        "key_support": support,
        "near_zone": near_zone
    }

# ---------- 1H Analysis ----------
def analyze_1h(candles_1h):
    if len(candles_1h) < 20:
        return {"trend": None, "fvg": None, "ob": None, "bos": None, "liquidity": None, "reversal": None}

    closes = [c["close"] for c in candles_1h]
    current = closes[-1]
    ema10 = calculate_ema(closes, 10)
    ema20 = calculate_ema(closes, 20)
    trend = None
    if ema10 > ema20 and current > ema10:
        trend = "BULLISH"
    elif ema10 < ema20 and current < ema10:
        trend = "BEARISH"

    fvg = detect_fvg(candles_1h)
    ob_bull, ob_bear = detect_order_blocks(candles_1h)
    ob = None
    if ob_bull:
        ob = {"type": "BULLISH", "high": ob_bull["high"], "low": ob_bull["low"]}
    elif ob_bear:
        ob = {"type": "BEARISH", "high": ob_bear["high"], "low": ob_bear["low"]}

    bos = detect_bos(candles_1h)

    # Liquidity sweep
    liquidity = None
    highs = [c["high"] for c in candles_1h[-10:]]
    lows = [c["low"] for c in candles_1h[-10:]]
    swing_highs, swing_lows = [], []
    for i in range(1, len(highs)-1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]: swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]: swing_lows.append(lows[i])
    if len(swing_highs) >= 2 and candles_1h[-1]["high"] > swing_highs[-1] and current < swing_highs[-1]:
        liquidity = "BULLISH"
    elif len(swing_lows) >= 2 and candles_1h[-1]["low"] < swing_lows[-1] and current > swing_lows[-1]:
        liquidity = "BEARISH"

    # Reversal
    reversal = None
    if bos == "BULLISH" and ob and ob["type"] == "BULLISH" and current <= ob["high"] and current >= ob["low"]:
        reversal = "BULLISH"
    elif bos == "BEARISH" and ob and ob["type"] == "BEARISH" and current <= ob["high"] and current >= ob["low"]:
        reversal = "BEARISH"
    elif bos == "BULLISH" and fvg == "BUY":
        reversal = "BULLISH"
    elif bos == "BEARISH" and fvg == "SELL":
        reversal = "BEARISH"

    return {
        "trend": trend,
        "fvg": fvg,
        "ob": ob,
        "bos": bos,
        "liquidity": liquidity,
        "reversal": reversal
    }

# ---------- 15min Confirmation ----------
def confirm_15m(candles, trade_type):
    if len(candles) < 3:
        return False
    prev = candles[-2]
    curr = candles[-1]
    o1, h1, l1, c1 = prev["open"], prev["high"], prev["low"], prev["close"]
    o2, h2, l2, c2 = curr["open"], curr["high"], curr["low"], curr["close"]

    pattern = False
    if trade_type == "BUY":
        if (c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1): pattern = True
        body = abs(c2 - o2)
        lower_wick = min(o2, c2) - l2
        upper_wick = h2 - max(o2, c2)
        total_range = h2 - l2
        if total_range > 0 and lower_wick > 2 * body and upper_wick < body and c2 > o2:
            pattern = True
    else:
        if (c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1): pattern = True
        body = abs(c2 - o2)
        lower_wick = min(o2, c2) - l2
        upper_wick = h2 - max(o2, c2)
        total_range = h2 - l2
        if total_range > 0 and upper_wick > 2 * body and lower_wick < body and c2 < o2:
            pattern = True

    if not pattern:
        return False

    prices = [c["close"] for c in candles]
    rsi = calculate_rsi(prices)
    if trade_type == "BUY" and rsi > 70:
        return False
    if trade_type == "SELL" and rsi < 30:
        return False

    ema10 = calculate_ema(prices, 10)
    if trade_type == "BUY" and curr["close"] < ema10:
        return False
    if trade_type == "SELL" and curr["close"] > ema10:
        return False

    return True

# ---------- Main signal engine ----------
def process_signals():
    global RISK_REWARD_MULTIPLIER, STATS
    candles_15m = fetch_real_candles()
    if not candles_15m or len(candles_15m) < 8:
        return None
    if STATS["daily_losses"] >= MAX_DAILY_LOSSES:
        return None

    atr_15m = calculate_atr(candles_15m)
    if atr_15m < MIN_ATR_15M:
        return None

    candles_4h = fetch_tf_candles(SYMBOL, "4h", 30)
    candles_1h = fetch_tf_candles(SYMBOL, "1h", 30)
    if not candles_4h or not candles_1h:
        logger.info("Missing higher timeframe data")
        return None

    h4 = analyze_4h(candles_4h)
    h1 = analyze_1h(candles_1h)

    if h4["direction"] is None:
        return None

    current_price = candles_15m[-1]["close"]

    def score_buy():
        if h4["direction"] == "BEARISH":
            return None
        score = 0
        reasons = []

        if h4["direction"] == "BULLISH":
            score += 15; reasons.append("4H↑")
        if h4["near_zone"] == "SUPPORT":
            score += 10; reasons.append("4H-Support")

        if h1["trend"] == "BULLISH":
            score += 10; reasons.append("1H↑")
        if h1["fvg"] == "BUY":
            score += 10; reasons.append("1H-FVG")
        if h1["ob"] and h1["ob"]["type"] == "BULLISH" and current_price <= h1["ob"]["high"] and current_price >= h1["ob"]["low"]:
            score += 10; reasons.append("1H-OB")
        if h1["bos"] == "BULLISH":
            score += 5; reasons.append("1H-BOS")
        if h1["liquidity"] == "BULLISH":
            score += 5; reasons.append("1H-Liq")
        if h1["reversal"] == "BULLISH":
            score += 10; reasons.append("1H-Rev")

        if not confirm_15m(candles_15m, "BUY"):
            return None

        fvg_15 = detect_fvg(candles_15m)
        ob_bull_15, _ = detect_order_blocks(candles_15m)
        choch_15 = detect_choch(candles_15m)
        bos_15 = detect_bos(candles_15m)

        if fvg_15 == "BUY": score += 10; reasons.append("15m-FVG")
        if ob_bull_15 and current_price <= ob_bull_15["high"] and current_price >= ob_bull_15["low"]:
            score += 8; reasons.append("15m-OB")
        if choch_15 == "BULLISH": score += 8; reasons.append("15m-CHoCH")
        if bos_15 == "BULLISH": score += 5; reasons.append("15m-BOS")
        score += 4; reasons.append("CandlePattern")

        prices = [c["close"] for c in candles_15m]
        rsi = calculate_rsi(prices)
        if 40 <= rsi <= 60:
            score += 3; reasons.append("RSI-mid")

        if score >= 75:
            return {"type": "BUY", "score": score, "reasons": reasons}
        return None

    def score_sell():
        if h4["direction"] == "BULLISH":
            return None
        score = 0
        reasons = []

        if h4["direction"] == "BEARISH":
            score += 15; reasons.append("4H↓")
        if h4["near_zone"] == "RESISTANCE":
            score += 10; reasons.append("4H-Resistance")

        if h1["trend"] == "BEARISH":
            score += 10; reasons.append("1H↓")
        if h1["fvg"] == "SELL":
            score += 10; reasons.append("1H-FVG")
        if h1["ob"] and h1["ob"]["type"] == "BEARISH" and current_price <= h1["ob"]["high"] and current_price >= h1["ob"]["low"]:
            score += 10; reasons.append("1H-OB")
        if h1["bos"] == "BEARISH":
            score += 5; reasons.append("1H-BOS")
        if h1["liquidity"] == "BEARISH":
            score += 5; reasons.append("1H-Liq")
        if h1["reversal"] == "BEARISH":
            score += 10; reasons.append("1H-Rev")

        if not confirm_15m(candles_15m, "SELL"):
            return None

        fvg_15 = detect_fvg(candles_15m)
        _, ob_bear_15 = detect_order_blocks(candles_15m)
        choch_15 = detect_choch(candles_15m)
        bos_15 = detect_bos(candles_15m)

        if fvg_15 == "SELL": score += 10; reasons.append("15m-FVG")
        if ob_bear_15 and current_price <= ob_bear_15["high"] and current_price >= ob_bear_15["low"]:
            score += 8; reasons.append("15m-OB")
        if choch_15 == "BEARISH": score += 8; reasons.append("15m-CHoCH")
        if bos_15 == "BEARISH": score += 5; reasons.append("15m-BOS")
        score += 4; reasons.append("CandlePattern")

        prices = [c["close"] for c in candles_15m]
        rsi = calculate_rsi(prices)
        if 40 <= rsi <= 60:
            score += 3; reasons.append("RSI-mid")

        if score >= 75:
            return {"type": "SELL", "score": score, "reasons": reasons}
        return None

    buy = score_buy()
    sell = score_sell()

    if buy and (not sell or buy["score"] >= sell.get("score", 0)):
        trade = buy
    elif sell:
        trade = sell
    else:
        return None

    stop_distance = max(atr_15m * 1.5, MIN_STOP_POINTS)
    sig = trade["type"]
    reason = " + ".join(trade["reasons"])
    sl = current_price - stop_distance if sig == "BUY" else current_price + stop_distance
    tp1 = current_price + (stop_distance * RISK_REWARD_MULTIPLIER) if sig == "BUY" else current_price - (stop_distance * RISK_REWARD_MULTIPLIER)
    tp2 = current_price + (stop_distance * RISK_REWARD_MULTIPLIER * 2) if sig == "BUY" else current_price - (stop_distance * RISK_REWARD_MULTIPLIER * 2)
    grade = "A" if trade["score"] >= 90 else "B"

    STATS["total_signals"] += 1
    return {
        "type": sig,
        "reason": reason,
        "entry": current_price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "status": "PENDING",
        "grade": grade,
        "score": trade["score"]
    }

# ---------- Position monitor & signal loop ----------
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
            grade = sig.get("grade","B"); score = sig.get("score",0)
            emoji = "🟢" if sig['type']=="BUY" else "🔴"

            vip_msg = (
                f"┌─────────────────────────────────┐\n"
                f"│  {emoji} {sig['type']} BTC/USD  │  {grade}  │  {score}%  │\n"
                f"└─────────────────────────────────┘\n"
                f"  Entry    ${sig['entry']:.2f}\n"
                f"  SL       ${sig['sl']:.2f} ({abs(sig['entry']-sig['sl']):.1f} pts)\n"
                f"  TP1      ${sig['tp1']:.2f} (+{abs(sig['tp1']-sig['entry']):.1f} pts)\n"
                f"  TP2      ${sig['tp2']:.2f} (+{abs(sig['tp2']-sig['entry']):.1f} pts)\n\n"
                f"  [{sig['reason']}]\n\n"
                f"  ⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )
            await context.bot.send_message(chat_id=VIP_CHANNEL_ID, text=vip_msg)

            free_msg = (
                f"┌─────────────────────────────────┐\n"
                f"│  {emoji} {sig['type']} BTC/USD  │  {grade}  │  {score}%  │\n"
                f"└─────────────────────────────────┘\n"
                f"  Entry    ${sig['entry']:.2f}\n"
                f"  SL       ${sig['sl']:.2f}\n"
                f"  TP1      ${sig['tp1']:.2f}\n\n"
                f"  ⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
                f"  ⚡ Full breakdown in VIP: /join_vip"
            )
            await context.bot.send_message(chat_id=FREE_CHANNEL_ID, text=free_msg)
            await context.bot.send_message(chat_id=CHAT_ID, text=vip_msg)

# ---------- Daily bias ----------
async def daily_bias(context: ContextTypes.DEFAULT_TYPE):
    if not RUN_SIGNALS or not CHAT_ID: return
    candles = fetch_real_candles()
    if not candles: return
    closes = [c["close"] for c in candles]
    ema20 = calculate_ema(closes, 20)
    current = closes[-1]
    bias = "🟢 BULLISH" if current > ema20 else "🔴 BEARISH"
    msg = f"📊 BTC/USD DAILY BIAS – {bias}\n⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    await context.bot.send_message(chat_id=FREE_CHANNEL_ID, text=msg)

async def report_callback(context: ContextTypes.DEFAULT_TYPE):
    global CHAT_ID, STATS
    if not CHAT_ID: return
    total = STATS["tp1_hits"] + STATS["tp2_hits"] + STATS["sl_hits"]
    wr = ((STATS["tp1_hits"]+STATS["tp2_hits"])/total*100) if total>0 else 0
    await context.bot.send_message(chat_id=CHAT_ID, text=f"📅 DAILY BTC\nSignals: {STATS['total_signals']}\nTP1: {STATS['tp1_hits']} TP2: {STATS['tp2_hits']}\nSL: {STATS['sl_hits']}\nWin: {wr:.1f}%")
    STATS["total_signals"]=STATS["tp1_hits"]=STATS["tp2_hits"]=STATS["sl_hits"]=STATS["daily_losses"]=0

# ---------- Commands ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CHAT_ID; CHAT_ID = update.effective_chat.id
    await update.message.reply_text("🟠 BTC/USD SMC (Advanced MTF)\n/start_signals /stop_signals /status /report /history /join_vip")

async def start_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RUN_SIGNALS, CHAT_ID
    CHAT_ID = update.effective_chat.id
    if RUN_SIGNALS: await update.message.reply_text("Already running"); return
    RUN_SIGNALS = True
    context.job_queue.run_repeating(signal_loop, interval=PRICE_INTERVAL_SECONDS, name="btc_job")
    context.job_queue.run_repeating(report_callback, interval=86400, first=86400, name="report_job")
    context.job_queue.run_daily(daily_bias, time=time(hour=8, minute=0, tzinfo=timezone.utc), name="bias_job")
    await update.message.reply_text("🚀 BTC Advanced MTF scanner started (15min) + daily bias")

async def stop_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RUN_SIGNALS
    RUN_SIGNALS = False
    for j in context.job_queue.get_jobs_by_name("btc_job"): j.schedule_removal()
    for j in context.job_queue.get_jobs_by_name("report_job"): j.schedule_removal()
    for j in context.job_queue.get_jobs_by_name("bias_job"): j.schedule_removal()
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
    await update.message.reply_text("🔒 *VIP Signals (all assets)*\n\n💰 *$25/month*\n💎 USDT (TRC20): `TFEYT12uggMhmhncqFSc8SAFzpdz6YfS2j`\n✅ Send screenshot to @XAU_EDGE", parse_mode="Markdown")

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