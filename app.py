# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
import pandas as pd

# --------- Config ---------
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_API_BACKUPS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]
INTERVAL = "5m"
RSI_PERIOD = 13
DEFAULT_REFRESH_SEC = 60
LOW_TH = 25.0
HIGH_TH = 75.0
TOP_N = 30

st.set_page_config(
    page_title="Top 30 Market Cap — Live RSI (5m, period 13)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Live RSI (5m) — Top 30 by Market Cap")
st.caption("منبع مارکت‌کپ: CoinGecko | کندل ۵دقیقه: Binance (در صورت خطا، CoinGecko OHLC)")

# --------- Sidebar ---------
with st.sidebar:
    st.header("تنظیمات")
    refresh_sec = st.slider("بازهٔ رفرش (ثانیه)", min_value=30, max_value=180, value=DEFAULT_REFRESH_SEC, step=10)
    st.caption("⏱️ برای رعایت محدودیت‌های API بهتر است کمتر از 30 ثانیه نباشد.")
    st.divider()
    st.subheader("آستانه‌های هشدار (RSI)")
    low_th = st.number_input("پایین", min_value=1.0, max_value=49.0, value=LOW_TH, step=0.5)
    high_th = st.number_input("بالا", min_value=51.0, max_value=99.0, value=HIGH_TH, step=0.5)
    st.divider()
    provider = st.selectbox("منبع کندل ۵دقیقه", ["Binance (پیش‌فرض)", "CoinGecko (جایگزین)"])

# Auto-refresh via query param tick
try:
    st.experimental_set_query_params(ts=str(int(time.time() // max(1, refresh_sec))))
except Exception:
    pass

# --------- HTTP sessions with Retry ---------
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET", "POST"},
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "Top30-RSI-Live/1.1",
        "Accept": "application/json",
    })
    return s

SESSION = make_session()

# --------- Helpers (CoinGecko) ---------
@st.cache_data(show_spinner=False, ttl=1800)
def cg_get_top_coins(n: int) -> List[Dict]:
    url = f"{COINGECKO_API}/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": n, "page": 1, "sparkline": "false"}
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=300)
def cg_fetch_ohlc(coin_id: str, days: int = 1) -> List[List[float]]:
    # days=1 ~ 5m candles on CG OHLC
    url = f"{COINGECKO_API}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    r = SESSION.get(url, params=params, timeout=30)
    if r.status_code == 429:
        time.sleep(3)
        r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# --------- Helpers (Binance) ---------
@st.cache_data(show_spinner=False, ttl=3600)
def binance_spot_usdt_set(base_url: str) -> set:
    r = SESSION.get(f"{base_url}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    data = r.json()
    usdt = set()
    for s in data.get("symbols", []):
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT" and s.get("isSpotTradingAllowed", True):
            usdt.add(s["symbol"])
    return usdt

@st.cache_data(show_spinner=False, ttl=120)
def binance_fetch_klines(symbol: str, base_url: str, interval: str = "5m", limit: int = 200) -> List[List]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = SESSION.get(f"{base_url}/api/v3/klines", params=params, timeout=30)
    if r.status_code == 429:
        time.sleep(2)
        r = SESSION.get(f"{base_url}/api/v3/klines", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# --------- Common Helpers ---------
def map_to_binance_symbol(symbol_text: str, usdt_set: set) -> Optional[str]:
    cand = f"{symbol_text.upper()}USDT"
    if cand in usdt_set:
        return cand
    aliases = {"TONCOIN": "TON", "WBTC": "WBTC", "STETH": "STETH", "BCH": "BCH", "PEPE": "PEPE",
               "SHIB": "SHIB", "DOGE": "DOGE", "TRX": "TRX", "ADA": "ADA", "XRP": "XRP", "SOL": "SOL"}
    up = symbol_text.upper()
    alt = aliases.get(up, up)
    cand2 = f"{alt}USDT"
    return cand2 if cand2 in usdt_set else None

def compute_rsi_from_closes(closes: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if abs(avg_loss) < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# --------- Data Pull ---------
with st.spinner("دریافت ۳۰ کوین اول مارکت‌کپ از CoinGecko..."):
    try:
        top_list = cg_get_top_coins(TOP_N)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        st.error(f"❌ دریافت لیست کوین‌ها از CoinGecko ناموفق بود (HTTP {status}).")
        st.stop()
    except Exception:
        st.error("❌ خطای غیرمنتظره هنگام دریافت لیست کوین‌ها.")
        st.stop()

rows: List[Dict] = []
alerts: List[Tuple] = []

# Try Binance if selected; otherwise go CG directly
use_binance = provider.startswith("Binance")
base_url = None

if use_binance:
    # probe mirrors
    for cand in BINANCE_API_BACKUPS:
        try:
            _ = binance_spot_usdt_set(cand)
            base_url = cand
            break
        except Exception:
            continue
    if base_url is None:
        st.warning("بایننس در دسترس نیست؛ به CoinGecko OHLC سوییچ شد.")
        use_binance = False

if use_binance:
    try:
        usdt_set = binance_spot_usdt_set(base_url)
    except Exception:
        st.warning("بایننس در دسترس نیست؛ به CoinGecko OHLC سوییچ شد.")
        use_binance = False

with st.spinner("دریافت کندل‌های ۵ دقیقه‌ای و محاسبه RSI..."):
    if use_binance:
        mapped: List[Tuple[str, str, str]] = []  # (name, cg_symbol, binance_symbol)
        for coin in top_list:
            name = coin.get("name", "")
            cg_symbol = coin.get("symbol", "")
            bsym = map_to_binance_symbol(cg_symbol, usdt_set)
            if bsym:
                mapped.append((name, cg_symbol.upper(), bsym))
        if not mapped:
            st.warning("هیچ‌کدام از کوین‌ها جفت USDT در Binance نداشتند؛ سوییچ به CoinGecko.")
            use_binance = False

        if use_binance:
            for (name, cg_sym, binance_sym) in mapped:
                try:
                    kl = binance_fetch_klines(binance_sym, base_url, "5m", limit=200)
                    closes = [float(k[4]) for k in kl]
                    last_price = closes[-1] if closes else None
                    rsi = compute_rsi_from_closes(closes, RSI_PERIOD)
                except Exception:
                    last_price = None
                    rsi = None
                rows.append({
                    "Name": name,
                    "Symbol": cg_sym,
                    "Pair/ID": binance_sym,
                    "Price (USDT)": round(last_price, 6) if last_price is not None else None,
                    "RSI(13,5m)": round(rsi, 2) if rsi is not None else None,
                })
                if rsi is not None and (rsi <= low_th or rsi >= high_th):
                    alerts.append((name, cg_sym, binance_sym, rsi, last_price))
                time.sleep(0.12)

    if not use_binance:
        for coin in top_list:
            name = coin.get("name", "")
            cg_symbol = coin.get("symbol", "").upper()
            coin_id = coin.get("id")
            try:
                ohlc = cg_fetch_ohlc(coin_id, days=1)
                closes = [row[4] for row in ohlc if isinstance(row, list) and len(row) >= 5]
                last_price = closes[-1] if closes else None
                rsi = compute_rsi_from_closes(closes, RSI_PERIOD)
            except Exception:
                last_price = None
                rsi = None
            rows.append({
                "Name": name,
                "Symbol": cg_symbol,
                "Pair/ID": coin_id,
                "Price (USD)": round(last_price, 6) if last_price is not None else None,
                "RSI(13,5m)": round(rsi, 2) if rsi is not None else None,
            })
            if rsi is not None and (rsi <= low_th or rsi >= high_th):
                alerts.append((name, cg_symbol, coin_id, rsi, last_price))
            time.sleep(0.12)

df = pd.DataFrame(rows)

# --------- Live Ticker ---------
st.subheader("🔴 Live RSI Ticker — Top 30")
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.caption(f"آخرین به‌روزرسانی: {now} | رفرش خودکار هر {refresh_sec} ثانیه | منبع: {'Binance' if use_binance else 'CoinGecko'}")

def fmt(x):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.1f}"
    except Exception:
        return str(x)

if not df.empty:
    ticker_pairs = [f"{r['Symbol']}:{fmt(r['RSI(13,5m)'])}" for _, r in df.iterrows()]
    per_line = 12
    for i in range(0, len(ticker_pairs), per_line):
        st.code("  |  ".join(ticker_pairs[i:i+per_line]))
else:
    st.info("داده‌ای برای نمایش موجود نیست.")

st.divider()

# --------- Alerts Board ---------
st.subheader("📣 RSI Alerts (≤ پایین / ≥ بالا)")
if alerts:
    cols = st.columns(4)
    for idx, (name, sym, pair, rsi, price) in enumerate(sorted(alerts, key=lambda x: x[3])):
        with cols[idx % 4]:
            st.metric(label=f"{name} ({sym})", value=f"{rsi:.2f}", delta=f"{price:.6g}" if price is not None else None)
else:
    st.caption("فعلاً هیچ کوینی در محدودهٔ هشدار نیست.")

st.divider()

# --------- Full Table ---------
st.subheader("جدول کامل (Top 30)")
df_show = df.sort_values(by=["RSI(13,5m)"], ascending=True, na_position="last")
st.dataframe(df_show, use_container_width=True, height=520)

st.caption("⚠️ اگر Binance خطا داد، از سایدبار منبع را روی CoinGecko بگذارید یا صبر کنید تا محدودیت رفع شود.")