# xau_once.py — run-once checker for GitHub Actions (free)
# Strategy: EMA20/50 cross + RSI + MACD + MACD histogram (strict), TF=5m

import os, sys, requests
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ======= CONFIG (ปรับได้) =====================================
# รายการ "ชุดสัญลักษณ์" ที่จะเช็คทีละชุด (ตัวแรกคือหลัก, ตัวหลัง ๆ คือ fallback)
INSTRUMENTS = [
    ["GC=F", "XAUUSD=X"],  # ทอง: ใช้ GC=F เป็นหลัก, ดึงไม่ได้ค่อยลอง XAUUSD=X
    # ["SI=F"],            # เปิดเช็คเงินได้: เอา # ออกถ้าต้องการ
]

INTERVAL = "5m"
PERIOD   = "30d"
BARS     = 220

# ส่งแจ้งเตือนเฉพาะเมื่อแท่งล่าสุดเพิ่งปิดภายในกี่นาที (กันซ้ำ/กัน stale)
FRESH_WINDOW_MIN = 4.0

# เกณฑ์ “เข้มงวด” เพื่อลดสัญญาณหลอก
RSI_BUY_MIN  = 55.0   # เดิม 52 → เข้มขึ้น
RSI_SELL_MAX = 45.0   # เดิม 48 → เข้มขึ้น
MACD_HIST_MIN = 0.02  # ต้องมากกว่าค่านี้ (BUY) / น้อยกว่าลบค่านี้ (SELL)

FAST_EMA, SLOW_EMA = 20, 50
RSI_LEN,  ATR_LEN  = 14, 14
# ===============================================================

TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT  = os.getenv("TG_CHAT")
if not TG_TOKEN or not TG_CHAT:
    print("Missing TG_TOKEN or TG_CHAT", file=sys.stderr); sys.exit(1)
try:
    TG_CHAT = int(TG_CHAT)
except:
    pass

def send_tg(text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text}
        )
    except Exception as e:
        print("Telegram error:", e)

def fetch_df(candidates: list[str]) -> pd.DataFrame:
    last_err = None
    for sym in candidates:
        try:
            df = yf.download(sym, interval=INTERVAL, period=PERIOD,
                             progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                raise RuntimeError("empty")
            df = df.rename(columns=str.title).reset_index()
            # normalize columns
            rename = {}
            for c in df.columns:
                lc = str(c).lower()
                if lc == "datetime": rename[c] = "time"
                elif lc == "date":   rename[c] = "time"
                elif lc == "open":   rename[c] = "o"
                elif lc == "high":   rename[c] = "h"
                elif lc == "low":    rename[c] = "l"
                elif lc == "close":  rename[c] = "c"
                elif lc == "volume": rename[c] = "v"
            df = df.rename(columns=rename)
            for c in ["o","h","l","c","v"]:
                if c not in df.columns: raise RuntimeError(f"missing {c}")
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if "time" not in df.columns:
                df.rename(columns={df.columns[0]:"time"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.dropna().reset_index(drop=True)
            if len(df) > BARS: df = df.tail(BARS).copy()
            df.attrs["symbol"] = sym
            return df
        except Exception as e:
            last_err = e
            print(f"[fetch] {sym} failed: {e}")
    raise RuntimeError(f"all candidates failed: {last_err}")

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = EMAIndicator(out["c"], FAST_EMA).ema_indicator()
    out["ema_slow"] = EMAIndicator(out["c"], SLOW_EMA).ema_indicator()
    out["rsi"]      = RSIIndicator(out["c"], RSI_LEN).rsi()
    macd = MACD(out["c"])  # 12/26/9
    out["macd"]        = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"]   = out["macd"] - out["macd_signal"]
    out["atr"]         = AverageTrueRange(out["h"], out["l"], out["c"], ATR_LEN).average_true_range()
    return out.dropna().reset_index(drop=True)

def generate_signal(prev, now):
    cross_up = (prev["ema_fast"] <= prev["ema_slow"]) and (now["ema_fast"] >  now["ema_slow"])
    cross_dn = (prev["ema_fast"] >= prev["ema_slow"]) and (now["ema_fast"] <  now["ema_slow"])
    # ฟิลเตอร์เข้มขึ้นด้วย RSI และ MACD histogram
    buy_ok  = (now["rsi"] >= RSI_BUY_MIN)  and (now["macd_hist"] >= MACD_HIST_MIN) and (now["macd"] > now["macd_signal"])
    sell_ok = (now["rsi"] <= RSI_SELL_MAX) and (now["macd_hist"] <= -MACD_HIST_MIN) and (now["macd"] < now["macd_signal"])
    if cross_up and buy_ok:  return "BUY"
    if cross_dn and sell_ok: return "SELL"
    return "WAIT"

def last_bar_is_fresh(last_time_utc) -> bool:
    now_utc = datetime.now(timezone.utc)
    age_min = (now_utc - pd.to_datetime(last_time_utc)).total_seconds() / 60.0
    return 0.0 <= age_min <= FRESH_WINDOW_MIN

def run_for(candidates: list[str]):
    df  = add_indicators(fetch_df(candidates))
    if len(df) < 2: return
    prev, now = df.iloc[-2], df.iloc[-1]
    sym  = df.attrs.get("symbol", candidates[0])
    side = generate_signal(prev, now)

    bkk   = pd.to_datetime(now["time"]).tz_convert(ZoneInfo("Asia/Bangkok"))
    bkk_s = bkk.strftime("%Y-%m-%d %H:%M")

    if not last_bar_is_fresh(now["time"]):
        print(f"{bkk_s} TH | {sym}/5m close {now['c']:.2f} (stale) → {side}")
        return

    print(f"{bkk_s} TH | {sym}/5m close {now['c']:.2f} → {side}")
    if side in ("BUY","SELL"):
        msg = (f"[{sym} 5m] {side}\n"
               f"Time (TH) {bkk_s}\n"
               f"Close {now['c']:.2f}\n"
               f"EMA20/50 {now['ema_fast']:.2f}/{now['ema_slow']:.2f}\n"
               f"RSI {now['rsi']:.1f} | MACD {now['macd']:.4f}/{now['macd_signal']:.4f} "
               f"| HIST {now['macd_hist']:.4f}\n"
               f"ATR {now['atr']:.2f}")
        send_tg(msg)

def main():
    for group in INSTRUMENTS:
        try:
            run_for(group)
        except Exception as e:
            print(f"[run] {group} error: {e}")

if __name__ == "__main__":
    main()
