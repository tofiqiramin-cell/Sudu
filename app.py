# -*- coding: utf-8 -*-
"""
USD / USDT / GOLD / TON — Web Dashboard
هر ۶۰ ثانیه قیمت می‌گیرد و در دیتابیس ذخیره می‌کند.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "market_history.sqlite3"
API_URL = "https://raw.githubusercontent.com/iran-market/iran-market.github.io/main/data/popular.json"
COINGECKO_TON_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=the-open-network&vs_currencies=usd"
)

ASSETS = {
    "USD": {"symbol": "USD_IRR_FREE", "title": "دلار آزاد", "unit": "تومان"},
    "USDT": {"symbol": "USDT_IRR", "title": "تتر", "unit": "تومان"},
    "GOLD": {"symbol": "GOLD_18K_IRR", "title": "طلای ۱۸ عیار", "unit": "تومان / گرم"},
    "TON": {"symbol": "TON", "title": "تون کوین (TON)", "unit": "تومان (تقریبی)", "source": "coingecko"},
}

MIN_SAMPLES = 12
MAX_CADENCE_MIN = 180
LAGS = (1, 2, 3, 5, 8, 12, 20)
HORIZONS_HOURS = (1, 6, 12, 24)
POLL_SECONDS = 60

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_source_time(item, payload):
    candidates = (
        item.get("updated_at"), item.get("timestamp"),
        item.get("last_updated"), item.get("datetime"),
        payload.get("updated_at"), payload.get("timestamp"),
        payload.get("generated_at"),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            if isinstance(value, (int, float)):
                stamp = value / 1000 if value > 10_000_000_000 else value
                return datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="seconds")
            parsed = pd.to_datetime(value, utc=True, errors="coerce")
            if not pd.isna(parsed):
                return parsed.isoformat()
        except (ValueError, TypeError, OSError):
            pass
    return None


_db_lock = threading.Lock()


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            price REAL NOT NULL,
            observed_at TEXT NOT NULL,
            source_time TEXT,
            UNIQUE(asset, source_time)
        )
        """
    )
    conn.commit()
    return conn


def add_quote(asset: str, price: float, source_time) -> bool:
    observed = utc_now_iso()
    with _db_lock:
        conn = get_conn()
        try:
            if source_time:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO quotes(asset, price, observed_at, source_time) VALUES(?,?,?,?)",
                    (asset, float(price), observed, source_time),
                )
            else:
                row = conn.execute(
                    "SELECT price FROM quotes WHERE asset=? ORDER BY id DESC LIMIT 1",
                    (asset,),
                ).fetchone()
                if row and float(row[0]) == float(price):
                    return False
                cur = conn.execute(
                    "INSERT INTO quotes(asset, price, observed_at, source_time) VALUES(?,?,?,NULL)",
                    (asset, float(price), observed),
                )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def history(asset: str) -> pd.DataFrame:
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT observed_at, price, source_time FROM quotes WHERE asset=? ORDER BY id",
                (asset,),
            ).fetchall()
        finally:
            conn.close()
    frame = pd.DataFrame(rows, columns=["observed_at", "price", "source_time"])
    if not frame.empty:
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame = frame.dropna(subset=["price"]).drop_duplicates(subset=["observed_at"])
        frame = frame.sort_values("observed_at").reset_index(drop=True)
    return frame


def fetch_market_quotes() -> dict:
    headers = {"User-Agent": "USD-USDT-GOLD-TON-Web/2.0"}
    response = requests.get(API_URL, timeout=20, headers=headers)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("data", [])
    found = {}

    for asset, config in ASSETS.items():
        if config.get("source") == "coingecko":
            continue
        item = next((x for x in items if x.get("symbol") == config["symbol"]), None)
        if item is None:
            continue
        raw_price = item.get("price")
        if raw_price is None:
            continue
        price = float(str(raw_price).replace(",", "").strip())
        if not np.isfinite(price) or price <= 0:
            continue
        found[asset] = {
            "price": price,
            "source_time": parse_source_time(item, payload),
        }

    try:
        ton_resp = requests.get(COINGECKO_TON_URL, timeout=15, headers=headers)
        ton_resp.raise_for_status()
        ton_usd = float(ton_resp.json().get("the-open-network", {}).get("usd", 0) or 0)
        if ton_usd > 0 and "USD" in found:
            ton_toman = ton_usd * found["USD"]["price"]
            if ton_toman > 1000:
                found["TON"] = {"price": ton_toman, "source_time": None}
    except Exception:
        pass

    if not found:
        raise ValueError("هیچ نمادی از API دریافت نشد.")
    return found


def refresh_and_store():
    """قیمت آنلاین را بگیر و فقط در صورت تغییر ذخیره کن."""
    quotes = fetch_market_quotes()
    added = 0
    latest = {}
    for asset, q in quotes.items():
        latest[asset] = q["price"]
        if add_quote(asset, q["price"], q.get("source_time")):
            added += 1
    return latest, added



def make_feature_row(prices):
    """Features from past observations only (safe for short series)."""
    s = pd.Series(np.asarray(prices, dtype=float))
    n = len(s)
    last = float(s.iloc[-1]) if n else 0.0
    row = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = float(s.iloc[-lag]) if n >= lag else last
    for window in (5, 10, 15, 20):
        tail = s.tail(min(window, n)) if n else s
        if len(tail) == 0:
            row[f"mean_{window}"] = last
            row[f"std_{window}"] = 0.0
            row[f"momentum_{window}"] = 0.0
        else:
            row[f"mean_{window}"] = float(tail.mean())
            row[f"std_{window}"] = float(tail.std(ddof=0)) if len(tail) > 1 else 0.0
            row[f"momentum_{window}"] = float(tail.iloc[-1] - tail.iloc[0]) if len(tail) > 1 else 0.0
    row["return_1"] = float(s.iloc[-1] / s.iloc[-2] - 1) if n >= 2 else 0.0
    row["return_5"] = float(s.iloc[-1] / s.iloc[-5] - 1) if n >= 5 else 0.0
    return row


def build_supervised(prices, horizon):
    arr = np.asarray(prices, dtype=float)
    # با داده کم، به جای max(LAGS) از حداقل منطقی استفاده می‌کنیم
    max_lag = min(max(LAGS), max(8, len(arr) // 3))
    X, y = [], []
    for end in range(max_lag, len(arr) - horizon):
        hist = arr[:end]
        X.append(list(make_feature_row(hist).values()))
        y.append(arr[end + horizon - 1])
    if not X:
        return np.empty((0, 1)), np.empty((0,))
    return np.asarray(X), np.asarray(y)


def cadence_minutes(frame):
    if len(frame) < 3:
        return None
    diffs = frame["observed_at"].sort_values().diff().dropna().dt.total_seconds() / 60
    diffs = diffs[(diffs > 0) & (diffs < 24 * 60)]
    if len(diffs) == 0:
        return None
    return float(diffs.median())






def train_and_forecast(frame, hours):
    """
    با هر مقدار داده موجود در دیتابیس پیش‌بینی می‌کند.
    اگر برای افق کامل حرکت تاریخی کم باشد، پنجره را کوچک می‌کند
    یا از بازده تک‌قدم استفاده می‌کند — دیگر خطا نمی‌دهد.
    """
    n_obs = len(frame)
    if n_obs < MIN_SAMPLES:
        raise ValueError(
            f"داده کافی نیست: {n_obs} نمونه؛ حداقل {MIN_SAMPLES} لازم است."
        )

    cadence = cadence_minutes(frame)
    if cadence is None:
        # اگر زمان‌ها خراب بود، فاصله فرضی ۳۰ دقیقه
        cadence = 30.0
    cadence = float(max(cadence, 0.5))

    # کل سری از دیتابیس: ایندکس 0 تا -1 (اولین تا آخرین نمونه)
    frame = frame.sort_values("observed_at").reset_index(drop=True)
    prices = frame["price"].to_numpy(dtype=float)  # prices[0] ... prices[-1]
    # حذف نقاط پرت (مثلاً اگر TON یک‌بار به دلار و یک‌بار به تومان ذخیره شده)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) >= 5:
        med = float(np.median(prices))
        if med > 0:
            prices = prices[(prices > med / 20.0) & (prices < med * 20.0)]
    n = len(prices)
    if n < MIN_SAMPLES:
        raise ValueError(
            f"بعد از پاک‌سازی داده پرت، فقط {n} نمونه ماند؛ حداقل {MIN_SAMPLES} لازم است."
        )
    last = float(prices[-1])  # آخرین قیمت = مبنای پیش‌بینی
    if not np.isfinite(last) or last <= 0:
        raise ValueError("قیمت نامعتبر در دیتابیس.")

    log_p = np.log(np.maximum(prices, 1e-12))
    step_rets = np.diff(log_p)
    if len(step_rets) < 1:
        # فقط یک نقطه — همان قیمت را برگردان
        return {
            "prediction": last, "low": last * 0.99, "high": last * 1.01,
            "model": "قیمت فعلی", "scores": {}, "baseline_mae": float("nan"),
            "cadence": cadence, "samples": n_obs, "steps": 1, "steps_ideal": 1,
            "hours": hours, "capped": False, "quality": "flat",
            "drift_per_step": 0.0, "vol": 0.0, "direction": "خنثی",
        }

    steps_per_hour = 60.0 / cadence
    steps_ideal = max(1, int(round(float(hours) * steps_per_hour)))
    # پنجره را تا جایی کوچک کن که از داده موجود حرکت ساخته شود
    win = min(steps_ideal, max(1, n - 1))

    past_moves = np.array([], dtype=float)
    while win >= 1:
        if n > win:
            moves = log_p[win:] - log_p[:-win]
            moves = moves[np.isfinite(moves)]
            if len(moves) >= 1:
                past_moves = moves
                break
        win -= 1

    if len(past_moves) == 0:
        # fallback: بازده تک‌قدم × تعداد قدم هدف
        win = max(1, min(steps_ideal, len(step_rets)))
        mean_r = float(np.mean(step_rets[-min(len(step_rets), 20):]))
        past_moves = np.array([mean_r * win], dtype=float)

    # وزن روی حرکت‌های اخیر
    if len(past_moves) >= 2:
        idx = np.arange(len(past_moves), dtype=float)
        tau = max(len(past_moves) * 0.3, 2.0)
        w = np.exp((idx - idx[-1]) / tau)
        w /= w.sum()
        order = np.argsort(past_moves)
        sm, sw = past_moves[order], w[order]
        cdf = np.cumsum(sw)
        median_move = float(sm[min(np.searchsorted(cdf, 0.5), len(sm) - 1)])
        q1_move = float(sm[min(np.searchsorted(cdf, 0.25), len(sm) - 1)])
        q3_move = float(sm[min(np.searchsorted(cdf, 0.75), len(sm) - 1)])
    else:
        median_move = float(past_moves[0])
        q1_move = median_move
        q3_move = median_move

    # بازده متوسط هر قدم از کل سری + تاکید روی اخیر
    mu_step = float(np.mean(step_rets))
    recent_n = min(len(step_rets), max(8, len(step_rets) // 4))
    mu_recent = float(np.mean(step_rets[-recent_n:])) if recent_n else mu_step
    mu_step = 0.4 * mu_step + 0.6 * mu_recent  # اخیر مهم‌تر؛ منفی/مثبت هر دو مجاز

    vol_step = float(np.std(step_rets)) if len(step_rets) > 1 else 0.002
    vol_step = max(vol_step, 1e-8)

    # حرکت مورد انتظار برای این افق = mu × تعداد قدم همان افق
    # پس ۱س، ۶س، ۱۲س، ۲۴س عددهای متفاوت می‌گیرند مگر داده کاملاً تخت باشد
    expected_move = mu_step * float(steps_ideal)

    # اگر تاریخچه پنجره‌ای هم چیزی گفت، مخلوط کن
    expected_move = 0.55 * expected_move + 0.45 * (median_move * float(steps_ideal) / float(max(win, 1)))

    # سقف نرم از خود نوسان سری (جلوگیری از عدد پرت مثل TON)
    max_log = max(3.0 * vol_step * np.sqrt(float(steps_ideal)), 0.01)
    max_log = min(max_log, 0.25)  # در بدترین حالت حدود ±۲۵٪ لگاریتمی
    expected_move = float(np.clip(expected_move, -max_log, max_log))

    # بازه از نوسان واقعی × √افق
    sigma_h = min(vol_step * np.sqrt(float(steps_ideal)), max_log)
    q1_move = expected_move - 0.675 * sigma_h
    q3_move = expected_move + 0.675 * sigma_h
    median_move = expected_move

    prediction = float(last * np.exp(expected_move))
    low = float(last * np.exp(q1_move))
    high = float(last * np.exp(q3_move))

    # محدوده از خود توزیع بازده‌ها (نه درصد ثابت دستی)
    hist_min = float(np.min(past_moves)) if len(past_moves) else q1_move
    hist_max = float(np.max(past_moves)) if len(past_moves) else q3_move
    # مقیاس به افق
    scale = float(steps_ideal) / float(max(win, 1))
    hist_min *= scale
    hist_max *= scale
    # حداقل عرض بازه متناسب با نوسان
    if high <= low or (high - low) / max(abs(prediction), 1) < 0.002:
        pad = max(vol_step * np.sqrt(float(steps_ideal)), 0.002)
        low = float(last * np.exp(expected_move - pad))
        high = float(last * np.exp(expected_move + pad))

    low = float(min(low, prediction))
    high = float(max(high, prediction))
    if low >= prediction:
        low = prediction * 0.998
    if high <= prediction:
        high = prediction * 1.002

    steps = int(win)
    scores = {}
    baseline_mae = float("nan")
    model_choice = ""
    quality = "hist"
    ml_used = False

    # ML اختیاری
    if n_obs >= 50 and n > 20:
        max_lag_est = min(max(LAGS), max(6, n // 4))
        steps_ml = max(1, min(steps_ideal, max(1, n - max_lag_est - 8)))
        X, y = build_supervised(prices, steps_ml)
        if len(y) >= 12:
            split = max(8, int(len(y) * 0.75))
            if len(y) - split < 2:
                split = max(8, len(y) - 2)
            if split >= 8 and len(y) - split >= 2:
                X_tr, X_te = X[:split], X[split:]
                y_tr, y_te = y[:split], y[split:]
                models = {
                    "Random Forest": RandomForestRegressor(
                        n_estimators=min(160, 30 + n_obs),
                        max_depth=6, min_samples_leaf=max(2, n_obs // 50),
                        random_state=42, n_jobs=-1,
                    ),
                    "Gradient Boosting": HistGradientBoostingRegressor(
                        max_iter=min(120, 40 + n_obs // 2),
                        learning_rate=0.05, max_leaf_nodes=12,
                        l2_regularization=1.0, random_state=42,
                    ),
                }
                fitted, residuals = {}, {}
                for name, model in models.items():
                    model.fit(X_tr, y_tr)
                    pred_te = model.predict(X_te)
                    scores[name] = {
                        "mae": float(mean_absolute_error(y_te, pred_te)),
                        "rmse": float(np.sqrt(mean_squared_error(y_te, pred_te))),
                    }
                    fitted[name] = model
                    residuals[name] = y_te - pred_te
                base = np.full_like(
                    y_te,
                    prices[-(len(y_te) + steps_ml)]
                    if len(prices) > len(y_te) + steps_ml else prices[-1],
                )
                baseline_mae = float(mean_absolute_error(y_te, base))
                best = min(scores, key=lambda k: scores[k]["mae"])
                if scores[best]["mae"] < baseline_mae * 0.98:
                    feat = np.asarray([list(make_feature_row(prices).values())])
                    ml_pred = float(fitted[best].predict(feat)[0])
                    ml_pred = float(np.clip(
                        ml_pred, last * np.exp(hist_min), last * np.exp(hist_max)
                    ))
                    w_ml = min(0.5, 0.2 + (n_obs - 50) / 400.0)
                    prediction = float((1 - w_ml) * prediction + w_ml * ml_pred)
                    model_choice = ""
                    ml_used = True
                    steps = steps_ml
                    quality = "hybrid"

    return {
        "prediction": float(prediction),
        "low": float(low),
        "high": float(high),
        "model": model_choice,
        "scores": scores,
        "baseline_mae": baseline_mae,
        "cadence": float(cadence),
        "samples": n_obs,
        "steps": int(steps),
        "steps_ideal": int(steps_ideal),
        "hours": hours,
        "capped": False,
        "quality": quality,
        "drift_per_step": float(median_move / max(win, 1)),
        "vol": float(np.std(step_rets)) if len(step_rets) > 1 else 0.0,
        "direction": "صعودی" if prediction > last * 1.0005 else (
            "نزولی" if prediction < last * 0.9995 else "خنثی"
        ),
    }




# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# رفرش خودکار صفحه هر ۶۰ ثانیه → دوباره قیمت می‌گیرد و ذخیره می‌کند
st.markdown(
    f'<meta http-equiv="refresh" content="{POLL_SECONDS}">',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f19; color: #e8eef7; }
    h1, h2, h3 { color: #f0c14b !important; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Market Analyzer")
st.caption("دلار · تتر · طلا · تون‌کوین  |  دریافت خودکار هر ۶۰ ثانیه  |  پیش‌بینی آزمایشی")

# ---- دریافت خودکار در هر بار لود/رفرش ----
auto_error = None
latest = st.session_state.get("latest", {})
added_count = 0
try:
    latest, added_count = refresh_and_store()
    st.session_state["latest"] = latest
    st.session_state["last_fetch_ok"] = utc_now_iso()
except Exception as exc:
    auto_error = str(exc)

with st.sidebar:
    st.header("وضعیت")
    if auto_error:
        st.error(f"خطای دریافت: {auto_error}")
    else:
        st.success("دریافت خودکار فعال است")
        st.write(f"نمونه جدید این دور: **{added_count}**")
        st.caption(f"آخرین دریافت موفق: {st.session_state.get('last_fetch_ok', '—')}")
    st.info(f"صفحه هر **{POLL_SECONDS} ثانیه** یک‌بار خودش رفرش می‌شود و قیمت را می‌گیرد.")
    if st.button("دریافت دستی همین الان", use_container_width=True):
        st.rerun()
    st.divider()
    asset = st.selectbox(
        "دارایی برای تحلیل",
        options=list(ASSETS.keys()),
        format_func=lambda a: f"{a} — {ASSETS[a]['title']}",
    )
    run_fc = st.button("محاسبه پیش‌بینی‌ها", use_container_width=True, type="primary")

# کارت قیمت‌ها
cols = st.columns(4)
for i, (key, cfg) in enumerate(ASSETS.items()):
    price = latest.get(key)
    if price is None:
        hist = history(key)
        price = float(hist["price"].iloc[-1]) if not hist.empty else None
    with cols[i]:
        if price is not None:
            st.metric(cfg["title"], f"{price:,.0f}")
        else:
            st.metric(cfg["title"], "—")

st.divider()
frame = history(asset)
st.subheader(f"نمودار — {ASSETS[asset]['title']}")
if frame.empty:
    st.warning("هنوز داده‌ای نیست؛ تا یک دقیقه صبر کن یا دکمه دستی را بزن.")
else:
    st.line_chart(frame.tail(500).set_index("observed_at")[["price"]], height=320)
    st.caption(f"تعداد نمونه یکتا در دیتابیس: **{len(frame)}** (از ایندکس ۰ تا آخر)")

st.subheader("پیش‌بینی")
if run_fc:
    if frame.empty:
        st.warning("داده نیست.")
    else:
        fc_cols = st.columns(4)
        for i, hours in enumerate(HORIZONS_HOURS):
            with fc_cols[i]:
                try:
                    result = train_and_forecast(frame, hours)
                    st.markdown(f"**{hours} ساعته**")
                    st.markdown(f"### {result['prediction']:,.0f}")
                    st.markdown(
                        f"<div style='font-size:1.05rem;color:#c5d0dc'>"
                        f"بازه  {result['low']:,.0f}  —  {result['high']:,.0f}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.markdown(f"**{hours} ساعته**")
                    st.warning("نامعتبر")
                    st.caption(str(exc)[:140])
else:
    st.info("دارایی را انتخاب کن و «محاسبه پیش‌بینی‌ها» را بزن.")

with st.expander("تاریخچه اخیر"):
    if frame.empty:
        st.write("خالی")
    else:
        show = frame.tail(40).iloc[::-1].copy()
        show["observed_at"] = show["observed_at"].astype(str)
        show["price"] = show["price"].map(lambda x: f"{x:,.0f}")
        st.dataframe(show[["observed_at", "price"]], use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "منبع: Iran Market Data + CoinGecko | "
    "هر ۶۰ثانیه قیمت چک می‌شود؛ فقط در صورت تغییر در دیتابیس ذخیره می‌شود | "
    "پیش‌بینی آزمایشی است و توصیه مالی نیست."
)
