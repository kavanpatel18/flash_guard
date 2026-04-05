"""
FlashGuard v4 — Upstox-First Real-Time API Server
===================================================
• Primary data source: Upstox v2 REST API (hardcoded token or per-request)
• Fallback: yFinance (delayed)  →  demo data
• Model: improved_minute_model.keras  (30 timesteps × 10 features)
• Attention layer weights: W=(u,u)  b=(u,)  V=(u,)  [flat]

Run:
    pip install -r requirements.txt
    python api_server.py
    → http://localhost:5000
"""

import io, sys, traceback, time, random, json, logging, os
from datetime import date, timedelta, datetime
from pathlib import Path
from functools import wraps

# ── Load .env before anything else ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; environment vars may already be set

import numpy  as np
import pandas as pd
import requests

from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

_LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL     = getattr(logging, _LOG_LEVEL_STR, logging.INFO)
_LOG_FILE      = os.getenv("LOG_FILE", "")

_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
if _LOG_FILE:
    _handlers.append(logging.FileHandler(_LOG_FILE, encoding="utf-8"))

logging.basicConfig(
    level   = _LOG_LEVEL,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers= _handlers,
)

logger = logging.getLogger("flashguard")

# ── Keras ─────────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    import keras
    from keras.models import load_model as _keras_load
    KERAS_OK = True
    logger.info("TensorFlow %s ready", tf.__version__)
except ImportError:
    KERAS_OK = False
    logger.warning("TensorFlow not installed — model inference disabled")

if KERAS_OK:
    @keras.saving.register_keras_serializable(package="custom_layers")
    class Attention(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            u = input_shape[-1]
            self.W = self.add_weight(shape=(u,u), initializer="glorot_uniform", name="att_weight")
            self.b = self.add_weight(shape=(u,),  initializer="zeros",          name="att_bias")
            self.V = self.add_weight(shape=(u,),  initializer="glorot_uniform", name="att_v")
            super().build(input_shape)
        def call(self, x):
            s = tf.nn.tanh(tf.tensordot(x, self.W, axes=[[2],[0]]) + self.b)
            w = tf.nn.softmax(tf.reduce_sum(s * self.V, axis=-1, keepdims=True), axis=1)
            return tf.reduce_sum(x * w, axis=1)
        def get_config(self):
            return super().get_config()

    @keras.saving.register_keras_serializable(package="custom_layers")
    class TemporalAttention(keras.layers.Layer):
        def __init__(self, units=64, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.W = keras.layers.Dense(units, use_bias=False)
            self.v = keras.layers.Dense(1, use_bias=False)
        def call(self, hidden_states, training=False):
            score   = self.v(tf.nn.tanh(self.W(hidden_states)))
            weights = tf.nn.softmax(score, axis=1)
            context = tf.reduce_sum(weights * hidden_states, axis=1)
            return context
        def get_config(self):
            cfg = super().get_config()
            cfg.update({"units": self.units})
            return cfg

    _CUSTOM = {"Attention": Attention, "TemporalAttention": TemporalAttention}
else:
    _CUSTOM = {}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (environment-driven)
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
FRONTEND = BASE_DIR / "frontend"

FLASK_HOST      = os.getenv("FLASK_HOST",      "0.0.0.0")
FLASK_PORT      = int(os.getenv("FLASK_PORT",  "5000"))
FLASK_DEBUG     = os.getenv("FLASK_DEBUG",     "false").lower() == "true"
DEFAULT_TOKEN   = os.getenv("UPSTOX_TOKEN",    "")
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.20"))
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL",   "msa_gru_best.keras")

logger.info("Config loaded | host=%s port=%d debug=%s threshold=%.2f default_model=%s",
            FLASK_HOST, FLASK_PORT, FLASK_DEBUG, MODEL_THRESHOLD, DEFAULT_MODEL)

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
CORS(app)

# Suppress Flask/Werkzeug access logs unless DEBUG is on
if not FLASK_DEBUG:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST TIMING MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

@app.before_request
def _start_timer():
    g.start_time = time.perf_counter()

@app.after_request
def _log_request(response):
    duration_ms = (time.perf_counter() - g.start_time) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method, request.path, response.status_code, duration_ms,
    )
    return response

# ═══════════════════════════════════════════════════════════════════════════════
# UPSTOX CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

UPSTOX_BASE = "https://api.upstox.com/v2"

NSE_MAP = {
    "RELIANCE":   "NSE_EQ|INE002A01018",
    "TCS":        "NSE_EQ|INE467B01029",
    "HDFCBANK":   "NSE_EQ|INE040A01034",
    "INFY":       "NSE_EQ|INE009A01021",
    "ICICIBANK":  "NSE_EQ|INE090A01021",
    "SBIN":       "NSE_EQ|INE062A01020",
    "WIPRO":      "NSE_EQ|INE075A01022",
    "BAJFINANCE": "NSE_EQ|INE296A01032",
    "AXISBANK":   "NSE_EQ|INE238A01034",
    "MARUTI":     "NSE_EQ|INE585B01010",
    "ITC":        "NSE_EQ|INE154A01025",
    "ADANIENT":   "NSE_EQ|INE423A01024",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "KOTAKBANK":  "NSE_EQ|INE237A01036",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "SUNPHARMA":  "NSE_EQ|INE044A01036",
    "TITAN":      "NSE_EQ|INE280A01028",
    "LT":         "NSE_EQ|INE018A01030",
    "HCLTECH":    "NSE_EQ|INE860A01027",
    "NIFTY50":    "NSE_INDEX|Nifty 50",
    "BANKNIFTY":  "NSE_INDEX|Nifty Bank",
    "SENSEX":     "BSE_INDEX|SENSEX",
}

for k in list(NSE_MAP.keys()):
    NSE_MAP[k + ".NS"] = NSE_MAP[k]

IV_MAP = {
    "1m":  "1minute",
    "30m": "30minute",
    "1d":  "day",
    "1wk": "week",
    "1mo": "month",
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PRIORITY = [
    DEFAULT_MODEL,
    "msa_gru_best.keras",
    "improved_minute_model.keras",
    "improved_flash_crash_model.keras",
]
# Deduplicate while preserving order
_seen = set()
MODEL_PRIORITY = [m for m in MODEL_PRIORITY if not (m in _seen or _seen.add(m))]

_loaded: dict = {}

def _discover():
    return [m for m in MODEL_PRIORITY if (BASE_DIR / m).exists()]

def _load(name: str):
    if name in _loaded:
        return _loaded[name]
    fp = BASE_DIR / name
    if not fp.exists():
        raise FileNotFoundError(f"Model not found: {name}")
    if not KERAS_OK:
        raise RuntimeError("TensorFlow not installed")
    logger.info("Loading model: %s", name)
    m = _keras_load(str(fp), compile=False, custom_objects=_CUSTOM)
    _loaded[name] = m
    logger.info("Model loaded: %s  input_shape=%s", name, m.input_shape)
    return m

def _sig(model):
    s = model.input_shape
    return int(s[1]), int(s[2])

def _best():
    f = _discover()
    if not f:
        raise RuntimeError("No model files found next to api_server.py")
    return f[0]

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

FEATURES_10 = [
    "return", "log_return",
    "volatility_5", "volatility_10", "volatility_20",
    "momentum_5", "momentum_10",
    "high_low_spread", "open_close_return", "price_acceleration",
]
FEATURES_14 = [
    "Open","High","Low","Close","Volume","VWAP","return",
    "volatility","momentum","volume_change","vwap_diff",
    "high_low_spread","open_close_return","turnover_change",
]
FEATURES_5 = ["return","volume_change","volatility_5","volatility_10","momentum_5"]
FEATURES_MSA = [
    "return_1", "return_5", "return_10",
    "vol_10", "vol_30", "vol_60",
    "ema_ratio_10_30", "ema_ratio_10_60",
    "rsi_14", "spread", "vol_zscore",
    "intraday_range", "momentum_10", "momentum_30",
]

def _pick_features(n, model_name=""):
    if "msa" in model_name.lower(): return FEATURES_MSA
    if n == 10: return FEATURES_10
    if n == 14: return FEATURES_14
    if n == 5:  return FEATURES_5
    return FEATURES_10[:n]

def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _norm(df):
    m = {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume","date":"Date"}
    return df.rename(columns={c: m.get(str(c).strip().lower(), str(c).strip()) for c in df.columns})

def _engineer(df):
    o = _norm(_flatten(df.copy()))
    o["return"]             = o["Close"].pct_change()
    o["log_return"]         = np.log(o["Close"] / o["Close"].shift(1))
    o["volume_change"]      = o["Volume"].pct_change()
    o["volatility_5"]       = o["return"].rolling(5).std()
    o["volatility_10"]      = o["return"].rolling(10).std()
    o["volatility_20"]      = o["return"].rolling(20).std()
    o["volatility"]         = o["volatility_10"]
    o["momentum_5"]         = o["Close"].pct_change(5)
    o["momentum_10"]        = o["Close"].pct_change(10)
    o["momentum"]           = o["Close"] - o["Close"].shift(5)
    o["VWAP"]               = (o["High"] + o["Low"] + o["Close"]) / 3.0
    o["vwap_diff"]          = (o["Close"] - o["VWAP"]) / o["VWAP"].replace(0, np.nan)
    o["high_low_spread"]    = (o["High"] - o["Low"]) / o["Close"].replace(0, np.nan)
    o["open_close_return"]  = (o["Close"] - o["Open"]) / o["Open"].replace(0, np.nan)
    o["turnover_change"]    = (o["Close"] * o["Volume"]).pct_change()
    o["price_acceleration"] = o["return"].diff()

    # -- MSA GRU Features --
    o["return_1"] = o["Close"].pct_change(1)
    o["return_5"] = o["Close"].pct_change(5)
    o["return_10"] = o["Close"].pct_change(10)
    o["vol_10"] = o["return_1"].rolling(10).std()
    o["vol_30"] = o["return_1"].rolling(30).std()
    o["vol_60"] = o["return_1"].rolling(60).std()
    o["ema_10"] = o["Close"].ewm(span=10, adjust=False).mean()
    o["ema_30"] = o["Close"].ewm(span=30, adjust=False).mean()
    o["ema_60"] = o["Close"].ewm(span=60, adjust=False).mean()
    o["ema_ratio_10_30"] = o["ema_10"] / o["ema_30"] - 1
    o["ema_ratio_10_60"] = o["ema_10"] / o["ema_60"] - 1
    
    delta = o["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    o["rsi_14"] = 100 - 100 / (1 + rs)
    
    o["spread"] = (o["High"] - o["Low"]) / o["Close"].replace(0, np.nan)
    o["vol_zscore"] = (o["Volume"] - o["Volume"].rolling(30).mean()) / o["Volume"].rolling(30).std().replace(0, np.nan)
    o["intraday_range"] = (o["High"] - o["Low"]) / o["Open"].replace(0, np.nan)
    o["momentum_30"] = o["Close"] / o["Close"].shift(30) - 1

    return o

def _build_seq(df, timesteps, n_feat, model_name=""):
    eng   = _engineer(df)
    feats = _pick_features(n_feat, model_name)
    ff    = eng[feats].replace([np.inf,-np.inf], np.nan).bfill().ffill().fillna(0)
    if len(ff) < timesteps:
        raise ValueError(f"Need {timesteps} rows, only got {len(ff)}. Use a longer period.")
    sc  = StandardScaler()
    ff  = pd.DataFrame(sc.fit_transform(ff), columns=feats, index=ff.index)
    return np.expand_dims(ff.tail(timesteps).to_numpy(np.float32), 0), eng

# ═══════════════════════════════════════════════════════════════════════════════
# UPSTOX DATA FETCHER (PRIMARY)
# ═══════════════════════════════════════════════════════════════════════════════

def _upstox_headers(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}

def _upstox_candles(instrument_key, interval, token, period="6mo"):
    enc = instrument_key.replace("|", "%7C")
    hdr = _upstox_headers(token)
    all_candles = []

    days_map = {"5d":7,"1mo":35,"3mo":95,"6mo":185,"1y":370,"2y":740,"5y":1825}

    if interval in ("1minute", "30minute"):
        h_from = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        h_to   = date.today().strftime("%Y-%m-%d")
        url    = f"{UPSTOX_BASE}/historical-candle/{enc}/{interval}/{h_to}/{h_from}"
    else:
        days   = days_map.get(period, 185)
        fd     = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        td_    = date.today().strftime("%Y-%m-%d")
        url    = f"{UPSTOX_BASE}/historical-candle/{enc}/{interval}/{td_}/{fd}"

    try:
        r = requests.get(url, headers=hdr, timeout=12)
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                all_candles = d.get("data", {}).get("candles", [])
        else:
            logger.warning("Upstox historical: HTTP %d for %s", r.status_code, instrument_key)
    except Exception as e:
        logger.error("Upstox historical error: %s", e)

    if interval in ("1minute", "30minute"):
        try:
            url2 = f"{UPSTOX_BASE}/historical-candle/intraday/{enc}/{interval}"
            r2   = requests.get(url2, headers=hdr, timeout=8)
            if r2.status_code == 200:
                d2 = r2.json()
                if d2.get("status") == "success":
                    all_candles += d2.get("data", {}).get("candles", [])
        except Exception as e:
            logger.error("Upstox intraday error: %s", e)

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles,
                      columns=["Datetime","Open","High","Low","Close","Volume","OI"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = (df.drop_duplicates("Datetime")
            .set_index("Datetime")
            .sort_index())
    return df[["Open","High","Low","Close","Volume"]].astype(float)


def _upstox_quote(instrument_key, token):
    try:
        hdr = _upstox_headers(token)
        enc = instrument_key.replace("|", "%7C")
        url = f"{UPSTOX_BASE}/market-quote/quotes?instrument_key={enc}"
        r   = requests.get(url, headers=hdr, timeout=8)
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                data = d.get("data", {})
                key  = list(data.keys())[0] if data else None
                if key:
                    q = data[key]
                    return {
                        "ltp":        q.get("last_price", 0),
                        "open":       q.get("ohlc", {}).get("open", 0),
                        "high":       q.get("ohlc", {}).get("high", 0),
                        "low":        q.get("ohlc", {}).get("low", 0),
                        "prev_close": q.get("ohlc", {}).get("close", 0),
                        "volume":     q.get("volume", 0),
                        "change":     q.get("net_change", 0),
                        "change_pct": q.get("net_change", 0) / max(q.get("ohlc",{}).get("close",1),1) * 100,
                    }
    except Exception as e:
        logger.error("Upstox quote error: %s", e)
    return None


def _market_overview(token):
    tickers = {
        "NIFTY 50":   "NSE_INDEX|Nifty 50",
        "BANK NIFTY": "NSE_INDEX|Nifty Bank",
        "RELIANCE":   "NSE_EQ|INE002A01018",
        "TCS":        "NSE_EQ|INE467B01029",
        "HDFC BANK":  "NSE_EQ|INE040A01034",
        "INFOSYS":    "NSE_EQ|INE009A01021",
        "ICICI BANK": "NSE_EQ|INE090A01021",
        "SBI":        "NSE_EQ|INE062A01020",
    }
    results = []
    keys_encoded = ",".join(v.replace("|","%7C") for v in tickers.values())
    try:
        hdr = _upstox_headers(token)
        url = f"{UPSTOX_BASE}/market-quote/quotes?instrument_key={keys_encoded}"
        r   = requests.get(url, headers=hdr, timeout=10)
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                data = d.get("data", {})
                for name, ikey in tickers.items():
                    lookup = ikey.replace("|",":")
                    q = data.get(lookup) or data.get(ikey) or {}
                    if not q:
                        for dk in data.keys():
                            if ikey.split("|")[1] in dk:
                                q = data[dk]; break
                    ltp   = q.get("last_price", 0)
                    ohlc  = q.get("ohlc", {})
                    pclose= ohlc.get("close", ltp) or ltp
                    chg   = ltp - pclose
                    chgp  = (chg / pclose * 100) if pclose else 0
                    results.append({
                        "name":    name,
                        "ltp":     round(ltp, 2),
                        "change":  round(chg, 2),
                        "change_pct": round(chgp, 2),
                        "volume":  q.get("volume", 0),
                        "high":    round(ohlc.get("high", 0), 2),
                        "low":     round(ohlc.get("low", 0), 2),
                    })
        else:
            logger.warning("Market overview HTTP %d", r.status_code)
    except Exception as e:
        logger.error("Market overview error: %s", e)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def _demo(period="6mo", interval="1d", seed=42):
    td = {"5d":5,"1mo":22,"3mo":66,"6mo":132,"1y":252,"2y":504}.get(period,132)
    if interval in {"1m","30m"}:
        mins = {"1m":1,"30m":30}[interval]; rows = max(60,(td*390)//mins); freq=f"{mins}min"
    else:
        rows = max(30,td); freq="B"
    rng   = np.random.default_rng(seed)
    ts    = pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq=freq)
    ret   = rng.normal(0,0.008,rows)
    close = 2450.0 * np.cumprod(1+ret)
    open_ = np.r_[close[0],close[:-1]]*(1+rng.normal(0,0.001,rows))
    high  = np.maximum(open_,close)*(1+rng.uniform(0.001,0.01,rows))
    low   = np.minimum(open_,close)*(1-rng.uniform(0.001,0.01,rows))
    vol   = rng.integers(800_000,8_000_000,rows).astype(float)
    return pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Volume":vol},index=ts)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FETCH FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch(ticker, period="6mo", interval="1d", token=None):
    """
    1. Try Upstox (if token provided)
    2. Try yFinance
    3. Fall back to demo data
    Returns (DataFrame, source_string)
    """
    clean = ticker.upper().replace(".NS","").strip()
    ikey  = NSE_MAP.get(ticker.upper()) or NSE_MAP.get(clean)
    up_iv = IV_MAP.get(interval, "day")

    # ── 1. Upstox ──────────────────────────────────────────────────────────────
    if token and ikey:
        try:
            df = _upstox_candles(ikey, up_iv, token, period)
            if df is not None and len(df) > 5:
                logger.info("Upstox: %d candles for %s", len(df), ticker)
                return df, "upstox"
            else:
                logger.warning("Upstox returned empty data for %s — trying fallback", ticker)
        except Exception as e:
            logger.error("Upstox fetch error for %s: %s", ticker, e)

    # ── 2. yFinance ────────────────────────────────────────────────────────────
    try:
        import yfinance as yf
        sym = ticker if ".NS" in ticker else (ticker + ".NS" if ticker not in ["NIFTY50","BANKNIFTY","SENSEX"] else "^NSEI")
        df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
        if df is not None and not df.empty:
            df = _flatten(df)
            df.index = pd.to_datetime(df.index)
            logger.info("yFinance: %d rows for %s", len(df), ticker)
            return df, "yfinance"
    except Exception as e:
        logger.error("yFinance error for %s: %s", ticker, e)

    # ── 3. Demo ────────────────────────────────────────────────────────────────
    logger.warning("Using demo data for %s", ticker)
    return _demo(period, interval), "demo"


def _risk_band(prob, threshold=None):
    if threshold is None:
        threshold = MODEL_THRESHOLD
    if prob >= threshold:        return "HIGH RISK"
    if prob >= threshold * 0.65: return "ELEVATED"
    return "STABLE"


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    for d in (FRONTEND, BASE_DIR):
        if (d/"index.html").exists(): return send_from_directory(str(d),"index.html")
    return "index.html not found", 404

@app.route("/dashboard.html")
def dashboard():
    for d in (FRONTEND, BASE_DIR):
        if (d/"dashboard.html").exists(): return send_from_directory(str(d),"dashboard.html")
    return "dashboard.html not found", 404

@app.route("/<path:f>")
def static_files(f):
    for d in (FRONTEND, BASE_DIR):
        if (d/f).exists(): return send_from_directory(str(d), f)
    return "Not found", 404

# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","keras":KERAS_OK,
                    "models":_discover(),"loaded":list(_loaded.keys())})

@app.route("/api/models")
def list_models():
    out = []
    for name in _discover():
        try:
            m = _load(name); ts,nf = _sig(m)
            out.append({"name":name,"timesteps":ts,"features":nf,"params":int(m.count_params())})
        except Exception as e:
            logger.error("Failed to load model %s: %s", name, e)
            out.append({"name":name,"error":str(e).split("\n")[0]})
    return jsonify(out)


@app.route("/api/market-overview", methods=["POST"])
def market_overview():
    try:
        d     = request.json or {}
        token = d.get("token") or DEFAULT_TOKEN
        if not token:
            return jsonify({"error":"Upstox token required","data":[]}), 400
        data = _market_overview(token)
        return jsonify({"data": data, "timestamp": datetime.now().strftime("%H:%M:%S")})
    except Exception as e:
        logger.error("Market overview route error: %s", e, exc_info=True)
        return jsonify({"error":str(e),"data":[]}), 400


@app.route("/api/quote", methods=["POST"])
def quote():
    try:
        d      = request.json or {}
        ticker = d.get("ticker","RELIANCE")
        token  = d.get("token") or DEFAULT_TOKEN
        clean  = ticker.upper().replace(".NS","")
        ikey   = NSE_MAP.get(ticker.upper()) or NSE_MAP.get(clean)
        if not ikey:
            return jsonify({"error":f"Ticker {ticker} not in instrument map"}), 400
        q = _upstox_quote(ikey, token)
        if q:
            return jsonify(q)
        return jsonify({"error":"Quote unavailable"}), 400
    except Exception as e:
        logger.error("Quote route error: %s", e, exc_info=True)
        return jsonify({"error":str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        d          = request.json or {}
        ticker     = d.get("ticker","RELIANCE.NS")
        model_name = d.get("model") or _best()
        period     = d.get("period","6mo")
        interval   = d.get("interval","1d")
        threshold  = float(d.get("threshold", MODEL_THRESHOLD))
        token      = d.get("token") or d.get("upstox_token") or DEFAULT_TOKEN

        logger.info("Predict request: ticker=%s model=%s period=%s interval=%s",
                    ticker, model_name, period, interval)

        model      = _load(model_name)
        ts, nf     = _sig(model)
        df, source = _fetch(ticker, period, interval, token=token)
        seq, eng   = _build_seq(df, ts, nf, model_name)

        t0   = time.perf_counter()
        prob = float(np.clip(model.predict(seq,verbose=0).ravel()[0],0,1))
        inf_ms = (time.perf_counter() - t0) * 1000

        logger.info("Inference done: ticker=%s prob=%.4f band=%s latency=%.1fms",
                    ticker, prob, _risk_band(prob, threshold), inf_ms)

        band = _risk_band(prob, threshold)

        chart_df = _flatten(df).tail(200)
        ohlc = [
            {"date":  str(idx)[:19],
             "open":  round(float(r["Open"]),2),
             "high":  round(float(r["High"]),2),
             "low":   round(float(r["Low"]),2),
             "close": round(float(r["Close"]),2),
             "volume":int(r.get("Volume",0))}
            for idx, r in chart_df.iterrows()
        ]

        live_price = None
        if token:
            clean = ticker.upper().replace(".NS","")
            ikey  = NSE_MAP.get(ticker.upper()) or NSE_MAP.get(clean)
            if ikey:
                q = _upstox_quote(ikey, token)
                if q: live_price = q.get("ltp")

        latest_close = live_price or round(float(df["Close"].iloc[-1]),2)

        return jsonify({
            "ticker":      ticker,
            "model":       model_name,
            "probability": round(prob,6),
            "risk_pct":    round(prob*100,2),
            "band":        band,
            "threshold":   threshold,
            "timesteps":   ts,
            "features":    nf,
            "ohlc":        ohlc,
            "latest_close":latest_close,
            "latest_date": str(df.index[-1])[:10],
            "source":      source,
            "live_price":  live_price,
        })
    except Exception as e:
        logger.error("Predict route error: %s", e, exc_info=True)
        return jsonify({"error":str(e)}), 400


@app.route("/api/portfolio", methods=["POST"])
def portfolio():
    try:
        d          = request.json or {}
        tickers    = d.get("tickers",["RELIANCE","TCS","HDFCBANK"])
        model_name = d.get("model") or _best()
        period     = d.get("period","6mo")
        interval   = d.get("interval","1d")
        threshold  = float(d.get("threshold", MODEL_THRESHOLD))
        token      = d.get("token") or d.get("upstox_token") or DEFAULT_TOKEN

        logger.info("Portfolio request: tickers=%s model=%s", tickers, model_name)

        model  = _load(model_name); ts,nf = _sig(model)
        results = []
        for t in tickers:
            try:
                df,source = _fetch(t,period,interval,token=token)
                seq,_     = _build_seq(df,ts,nf, model_name)
                prob      = float(np.clip(model.predict(seq,verbose=0).ravel()[0],0,1))
                results.append({"ticker":t,"probability":round(prob,6),
                                 "risk_pct":round(prob*100,2),"band":_risk_band(prob,threshold),
                                 "latest_close":round(float(df["Close"].iloc[-1]),2),"source":source})
            except Exception as e:
                logger.warning("Portfolio error for %s: %s", t, e)
                results.append({"ticker":t,"error":str(e)})
        results.sort(key=lambda x:x.get("probability",0),reverse=True)
        return jsonify({"results":results,"model":model_name,"threshold":threshold})
    except Exception as e:
        logger.error("Portfolio route error: %s", e, exc_info=True)
        return jsonify({"error":str(e)}), 400


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file uploaded"}), 400
        f          = request.files["file"]
        model_name = request.form.get("model") or _best()
        threshold  = float(request.form.get("threshold", MODEL_THRESHOLD))
        model      = _load(model_name); ts,nf = _sig(model)
        df         = _norm(pd.read_csv(io.BytesIO(f.read())))
        seq,eng    = _build_seq(df,ts,nf, model_name)
        prob       = float(np.clip(model.predict(seq,verbose=0).ravel()[0],0,1))
        feats      = _pick_features(nf, model_name)
        snap       = {k:round(float(v),6) for k,v in eng[feats].dropna().tail(1).iloc[0].to_dict().items()}
        logger.info("CSV upload: model=%s rows=%d prob=%.4f", model_name, len(df), prob)
        return jsonify({"probability":round(prob,6),"risk_pct":round(prob*100,2),
                         "band":_risk_band(prob,threshold),"threshold":threshold,
                         "model":model_name,"rows_loaded":len(df),"features":snap})
    except Exception as e:
        logger.error("Upload route error: %s", e, exc_info=True)
        return jsonify({"error":str(e)}), 400


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 62)
    logger.info("  ⚡ FlashGuard v4 — Upstox Real-Time Server")
    logger.info("=" * 62)
    found = _discover()
    logger.info("  Models   : %s", found)
    logger.info("  Frontend : %s", FRONTEND)
    logger.info("  URL      : http://%s:%d", FLASK_HOST, FLASK_PORT)
    logger.info("  Dashboard: http://%s:%d/dashboard.html", FLASK_HOST, FLASK_PORT)
    logger.info("=" * 62)
    for name in found:
        try:   _load(name)
        except Exception as e: logger.error("  Failed to pre-load %s: %s", name, e)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
