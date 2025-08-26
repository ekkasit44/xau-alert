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
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text},
            timeout=(5, 10)  # (connect timeout, read timeout) วินาที
        )
        r.raise_for_status()  # ถ้า HTTP != 200 จะ throw error ให้จับด้านล่าง
    except requests.Timeout:
        print("Telegram timeout: connect/read เกินกำหนดเวลา")
    except requests.RequestException as e:
        print("Telegram error:", e)

def fetch_df(candidates: list[str]) -> pd.DataFrame:
    def _flatten(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = ["_".join([str(x) for x in t if x not in ("", None)]) for t in df.columns]
        return df

    def _pick(df: pd.DataFrame, key: str) -> pd.Series:
        key = key.lower()
        # ตรงตัวก่อน
        for c in df.columns:
            if str(c).lower() == key:
                return df[c]
        # ชื่อคอลัมน์ที่ลงท้ายด้วย key (เช่น 'GC=F_Open' หรือ 'Open_GC=F')
        for c in df.columns:
            s = str(c).lower()
            if s.endswith(key):
                return df[c]
        # มีคำว่า key อยู่ข้างใน
        for c in df.columns:
            if key in str(c).lower():
                return df[c]
        raise KeyError(key)

    last_err = None
    for sym in candidates:
        try:
            # พยายามแบบ download ก่อน (บังคับ group_by="column" เพื่อลด MultiIndex แบบ ticker)
            df0 = yf.download(
                sym, interval=INTERVAL, period=PERIOD,
                progress=False, auto_adjust=False, threads=False, group_by="column"
            )
            if df0 is None or df0.empty:
                raise RuntimeError("empty")

            df0 = _flatten(df0).reset_index()

            # normalize เป็น o/h/l/c/v + time
            try:
                o = _pick(df0, "open")
                h = _pick(df0, "high")
                l = _pick(df0, "low")
                c = _pick(df0, "close")
            except KeyError:
                # บางเคสคอลัมน์เพี้ยนมาก ลอง history() อีกแบบ
                tkr = yf.Ticker(sym)
                dfh = tkr.history(period=PERIOD, interval=INTERVAL, auto_adjust=False)
                if dfh is None or dfh.empty:
                    raise RuntimeError("history empty")
                df0 = _flatten(dfh).reset_index()
                o = _pick(df0, "open")
                h = _pick(df0, "high")
                l = _pick(df0, "low")
                c = _pick(df0, "close")

            try:
                v = _pick(df0, "volume")
            except KeyError:
                # ไม่มี volume ก็สร้างศูนย์ให้
                v = pd.Series([0]*len(df0), index=df0.index, name="Volume")

            out = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v})
            # หา column เวลา
            if "Datetime" in df0.columns:
                out.insert(0, "time", pd.to_datetime(df0["Datetime"], utc=True))
            elif "Date" in df0.columns:
                out.insert(0, "time", pd.to_datetime(df0["Date"], utc=True))
            else:
                out.insert(0, "time", pd.to_datetime(df0.iloc[:, 0], utc=True))

            # ทำความสะอาด
            for col in ["o", "h", "l", "c", "v"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna().reset_index(drop=True)
            if len(out) > BARS:
                out = out.tail(BARS).copy()

            out.attrs["symbol"] = sym
            return out

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


