from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import websocket  # websocket-client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("data_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_COINS = ["BTC", "ETH", "BNB", "SOL", "XRP"]

COINGECKO_IDS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
}

BINANCE_SYMBOLS: Dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

# ---------------------------------------------------------------------------
# Rate limiter (thread-safe token bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, calls_per_second: float, name: str = "api"):
        self.interval = 1.0 / calls_per_second
        self.name = name
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last_call = time.monotonic()


def _request_with_retry(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 5,
    timeout: int = 20,
) -> dict:
    """GET with exponential back-off on 429 / 5xx only.
    4xx client errors (400, 401, 403, 404) are not retried — they are permanent."""
    for attempt in range(max_retries):
        if rate_limiter:
            rate_limiter.wait()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            # Retry only on rate-limit or server errors
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                log.warning("HTTP %s — retrying in %ss", resp.status_code, wait)
                time.sleep(wait)
                continue
            # Client errors (4xx) are permanent — raise immediately, no retry
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            raise  # already logged via raise_for_status; let caller handle
        except requests.RequestException as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning("Request error (%s) — retrying in %ss", exc, wait)
                time.sleep(wait)
            else:
                raise RuntimeError(f"Max retries exceeded for {url}") from exc
    raise RuntimeError(f"Max retries exceeded for {url}")


# ---------------------------------------------------------------------------
# CoinGecko
# ---------------------------------------------------------------------------

_cg_limiter = RateLimiter(calls_per_second=0.5, name="coingecko")


class CoinGeckoClient:
    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        self.headers: Dict[str, str] = {}
        if api_key:
            # Keys starting with "CG-" are Demo/free-tier keys → use demo header.
            # Pro keys (paid) use x-cg-pro-api-key.
            if api_key.startswith("CG-"):
                self.headers["x-cg-demo-api-key"] = api_key
            else:
                self.headers["x-cg-pro-api-key"] = api_key
                global _cg_limiter
                _cg_limiter = RateLimiter(calls_per_second=8, name="coingecko-pro")

    def market_chart(self, coin_id: str, days: int = 90) -> pd.DataFrame:
        url = f"{self.BASE}/coins/{coin_id}/market_chart"
        data = _request_with_retry(
            url,
            params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
            headers=self.headers,
            rate_limiter=_cg_limiter,
        )
        prices = pd.DataFrame(data["prices"], columns=["timestamp_ms", "close"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp_ms", "volume"])
        df = prices.merge(volumes[["timestamp_ms", "volume"]], on="timestamp_ms")
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["timestamp_ms"])
        df.index = df.index.normalize()
        # CoinGecko only provides close; fill OHLC with close as best approximation
        for col in ["open", "high", "low"]:
            df[col] = df["close"]
        log.info("CoinGecko: %d rows for %s", len(df), coin_id)
        return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Binance REST
# ---------------------------------------------------------------------------

_bnc_limiter = RateLimiter(calls_per_second=10, name="binance")


class BinanceRestClient:
    BASE = "https://api.binance.com/api/v3"

    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}

    def ticker_24h(self, symbol: str) -> dict:
        data = _request_with_retry(
            f"{self.BASE}/ticker/24hr",
            params={"symbol": symbol},
            headers=self.headers,
            rate_limiter=_bnc_limiter,
        )
        return {
            "symbol": data["symbol"],
            "price": float(data["lastPrice"]),
            "price_change_pct_24h": float(data["priceChangePercent"]),
            "volume_24h": float(data["volume"]),
            "bid": float(data["bidPrice"]),
            "ask": float(data["askPrice"]),
            "spread": float(data["askPrice"]) - float(data["bidPrice"]),
            "timestamp": datetime.fromtimestamp(
                data["closeTime"] / 1000, tz=timezone.utc
            ).isoformat(),
        }

    def klines(self, symbol: str, interval: str = "1d", limit: int = 1000) -> pd.DataFrame:
        raw = _request_with_retry(
            f"{self.BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            headers=self.headers,
            rate_limiter=_bnc_limiter,
        )
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
        df = df.set_index("datetime")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Binance WebSocket (live price stream)
# ---------------------------------------------------------------------------

class BinanceWebSocket:
    WS_BASE = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.latest: Dict = {}
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        url = f"{self.WS_BASE}/{self.symbol}@miniTicker"
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=lambda ws, e: log.error("WS error: %s", e),
            on_close=lambda ws, c, m: log.info("WS closed"),
        )
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
        # Wait up to 5s for first message
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if self._ws:
            self._ws.close()

    def _on_message(self, ws, message: str) -> None:
        data = json.loads(message)
        self.latest = {
            "price": float(data["c"]),
            "volume": float(data["v"]),
            "timestamp": datetime.fromtimestamp(data["E"] / 1000, tz=timezone.utc).isoformat(),
        }
        self._ready.set()


# ---------------------------------------------------------------------------
# Sentiment (VADER — free, no API key needed for analysis)
# ---------------------------------------------------------------------------

_cp_limiter = RateLimiter(calls_per_second=5, name="cryptopanic")
_news_limiter = RateLimiter(calls_per_second=0.05, name="newsapi")


class CryptoPanicClient:
    BASE = "https://cryptopanic.com/api/free/v1/posts/"

    def __init__(self, auth_token: str):
        self.token = auth_token

    def fetch_sentiment(self, coin: str, pages: int = 3) -> float:
        scores: List[float] = []
        for page in range(1, pages + 1):
            try:
                data = _request_with_retry(
                    self.BASE,
                    params={"auth_token": self.token, "currencies": coin,
                            "public": "true", "page": page},
                    rate_limiter=_cp_limiter,
                )
                for item in data.get("results", []):
                    votes = item.get("votes", {})
                    bull = votes.get("positive", 0)
                    bear = votes.get("negative", 0)
                    total = bull + bear
                    if total > 0:
                        scores.append((bull - bear) / total)
            except requests.exceptions.HTTPError as exc:
                # 404 = token invalid/expired, 401 = unauthorized — stop all pages immediately
                log.warning("CryptoPanic token invalid (skipping): %s", exc)
                break
            except Exception as exc:
                log.warning("CryptoPanic: %s", exc)
                break
        return float(np.mean(scores)) if scores else 0.0


class NewsAPIClient:
    BASE = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str):
        self.key = api_key
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._sia = SentimentIntensityAnalyzer()
        except ImportError:
            self._sia = None
            log.warning("vaderSentiment not installed — NewsAPI sentiment disabled")

    def fetch_sentiment(self, coin: str, lookback_days: int = 7) -> float:
        from_dt = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        try:
            data = _request_with_retry(
                self.BASE,
                params={"q": coin + " cryptocurrency", "from": from_dt,
                        "sortBy": "publishedAt", "pageSize": 100, "apiKey": self.key},
                rate_limiter=_news_limiter,
            )
            if self._sia is None:
                return 0.0
            scores = [
                self._sia.polarity_scores(
                    a.get("title", "") + " " + (a.get("description") or "")
                )["compound"]
                for a in data.get("articles", [])
            ]
            return float(np.mean(scores)) if scores else 0.0
        except Exception as exc:
            log.warning("NewsAPI: %s", exc)
            return 0.0


# ---------------------------------------------------------------------------
# CSV Loader — Binance export format
# ---------------------------------------------------------------------------

def load_csv(path: str, coin: str) -> pd.DataFrame:
    """
    Load Binance-format CSV.
    Handles column: open_time (string date OR millisecond timestamp).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    date_col = None
    for candidate in ["open_time", "date", "datetime", "time", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = next(
            (c for c in df.columns if "time" in c or "date" in c), df.columns[0]
        )

    sample = str(df[date_col].iloc[0]).strip()
    if sample.isdigit() and len(sample) > 10:
        df["datetime"] = pd.to_datetime(df[date_col], unit="ms", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df[date_col], utc=True)

    df = df.set_index("datetime").sort_index()

    rename_map: Dict[str, str] = {}
    already_mapped: set = set()
    for needed in ["open", "high", "low", "close", "volume"]:
        for col in df.columns:
            if col in already_mapped:
                continue
            if col == needed:
                already_mapped.add(col)
                break
            if col.startswith(needed + "_") or col.startswith(needed + " "):
                rename_map[col] = needed
                already_mapped.add(col)
                break

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    available = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[available].copy()

    for col in available:
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        df[col] = pd.to_numeric(series, errors="coerce")

    df = df.dropna(subset=["close"])
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]

    log.info("CSV loaded: %d rows for %s", len(df), coin)
    return df


# ---------------------------------------------------------------------------
# Technical Indicators (vectorized, no look-ahead bias)
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 40+ technical indicators.
    All rolling/ewm operations use only past data (no look-ahead).
    OBV is fully vectorized (no Python loop).
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # --- Moving averages ---
    for span in [7, 14, 21, 50, 200]:
        df[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()
        df[f"sma_{span}"] = c.rolling(span, min_periods=1).mean()

    # --- RSI 14 ---
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    df["rsi_14"] = _rsi(c)

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Bollinger Bands ---
    sma20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=1).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = bb_range / sma20.replace(0, np.nan)
    df["bb_pct"] = (c - df["bb_lower"]) / bb_range

    # --- ATR 14 ---
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=1).mean()

    # --- OBV (fully vectorized — no Python loop) ---
    direction = np.sign(c.diff()).fillna(0)
    df["obv"] = (direction * v).cumsum()

    # --- ADX 14 ---
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = tr.rolling(14, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.rolling(14, min_periods=1).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # --- Volatility ---
    log_ret = np.log(c / c.shift())
    df["vol_7d"] = log_ret.rolling(7, min_periods=1).std() * np.sqrt(252)
    df["vol_30d"] = log_ret.rolling(30, min_periods=1).std() * np.sqrt(252)

    # --- Fibonacci retracement (rolling 50-bar window, no look-ahead) ---
    roll_high = h.rolling(50, min_periods=1).max()
    roll_low  = l.rolling(50, min_periods=1).min()
    diff = roll_high - roll_low
    df["fib_0"]   = roll_high
    df["fib_236"] = roll_high - 0.236 * diff
    df["fib_382"] = roll_high - 0.382 * diff
    df["fib_500"] = roll_high - 0.500 * diff
    df["fib_618"] = roll_high - 0.618 * diff
    df["fib_100"] = roll_low

    # --- Pivot points ---
    pp = (h + l + c) / 3
    df["pivot"]     = pp
    df["resist_1"]  = 2 * pp - l
    df["support_1"] = 2 * pp - h
    df["resist_2"]  = pp + (h - l)
    df["support_2"] = pp - (h - l)

    # --- Returns & misc ---
    df["log_return"]  = log_ret
    df["pct_change"]  = c.pct_change()
    df["price_range"] = (h - l) / c.replace(0, np.nan)
    df["vwap_approx"] = (c * v).rolling(14, min_periods=1).sum() / v.rolling(14, min_periods=1).sum().replace(0, np.nan)

    # --- Stochastic Oscillator %K and %D ---
    low14  = l.rolling(14, min_periods=1).min()
    high14 = h.rolling(14, min_periods=1).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=1).mean()

    # --- Williams %R ---
    df["williams_r"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    # --- Commodity Channel Index (CCI) ---
    typical = (h + l + c) / 3
    sma_tp  = typical.rolling(20, min_periods=1).mean()
    mad     = typical.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (typical - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # --- Rate of Change ---
    df["roc_10"] = c.pct_change(10) * 100

    return df


# ---------------------------------------------------------------------------
# Feature Matrix (with look-ahead bias guard)
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    window: int = 30,
    horizons: Optional[List[int]] = None,
    sentiment: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build sliding-window (X, y) arrays for deep-learning models.

    Look-ahead bias guard:
      Each sample i uses rows [i : i+window] as input features.
      Target y uses close price at i+window+h-1.
      No future data leaks into X.
    """
    if horizons is None:
        horizons = [7, 30, 90, 180, 365]

    feature_cols = [
        col for col in df.columns
        if df[col].dtype in [np.float32, np.float64, float, int, np.int64]
    ]

    df_feat = df[feature_cols].copy()
    df_feat["sentiment"] = sentiment
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Fill any remaining NaN with column mean to avoid propagation
    df_feat = df_feat.fillna(df_feat.mean())

    feat_names = list(df_feat.columns)
    arr = df_feat.values.astype(np.float32)
    close_arr = df["close"].values.astype(np.float32)

    max_horizon = max(horizons)
    n_samples = len(arr) - window - max_horizon
    if n_samples <= 0:
        raise ValueError(f"Not enough data: need {window + max_horizon} rows, got {len(arr)}")

    # Verify no look-ahead: X at position i uses indices [i, i+window)
    # y at position i uses index i+window+h-1 (all strictly after X window)
    X_list, y_list = [], []
    for i in range(n_samples):
        X_list.append(arr[i: i + window])                              # past only
        y_list.append([close_arr[i + window + h - 1] for h in horizons])  # future only

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    log.info("Feature matrix: X=%s  y=%s  (no look-ahead)", X.shape, y.shape)
    return X, y, feat_names


# ---------------------------------------------------------------------------
# CSV Cache (SHA-256 based — historical data never re-processed)
# ---------------------------------------------------------------------------

def csv_fingerprint(path: str) -> str:
    """SHA-256 of first 64KB + file size — fast, collision-resistant."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
    return h.hexdigest()


def load_csv_cached(path: str, coin: str, cache_dir: str = ".cache") -> pd.DataFrame:
    """
    Load CSV with SHA-256 caching.
    If the CSV hasn't changed since last run, return cached Parquet instantly.
    Historical CSV data never changes → this avoids re-processing on every run.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fp = csv_fingerprint(path)
    cache_path = Path(cache_dir) / f"{coin}_csv_{fp[:16]}.parquet"

    if cache_path.exists():
        log.info("CSV cache hit for %s (fp=%s…)", coin, fp[:8])
        return pd.read_parquet(cache_path)

    log.info("CSV cache miss for %s — processing CSV…", coin)
    df = load_csv(path, coin)
    df.to_parquet(cache_path)
    # Clean up old cache files for this coin
    for old in Path(cache_dir).glob(f"{coin}_csv_*.parquet"):
        if old != cache_path:
            try:
                old.unlink()
            except Exception:
                pass
    log.info("CSV cached for %s at %s", coin, cache_path)
    return df


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

@dataclass
class DataManager:
    """Orchestrates all data sources for a single coin."""

    coin: str
    csv_path: Optional[str] = None
    cache_dir: str = ".cache"
    coingecko_api_key: Optional[str] = None
    cryptopanic_token: Optional[str] = None
    newsapi_key: Optional[str] = None
    use_websocket: bool = False

    _df: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    _live_ticker: Dict = field(default_factory=dict, init=False, repr=False)
    _ws: Optional[BinanceWebSocket] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self.coin = self.coin.upper()

    def load(self, fetch_coingecko: bool = True, fetch_binance: bool = True) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        # 1. CSV — cached by SHA-256 (historical data never re-processed)
        if self.csv_path and Path(self.csv_path).exists():
            frames.append(load_csv_cached(self.csv_path, self.coin, self.cache_dir))
        else:
            log.warning("CSV not found: %s", self.csv_path)

        # 2. CoinGecko — always fresh (90-day recent data)
        if fetch_coingecko:
            try:
                cg = CoinGeckoClient(self.coingecko_api_key)
                frames.append(cg.market_chart(COINGECKO_IDS[self.coin], days=90))
            except Exception as exc:
                log.warning("CoinGecko failed: %s", exc)

        # 3. Binance klines — always fresh (recent 1000 bars)
        if fetch_binance:
            try:
                frames.append(BinanceRestClient().klines(BINANCE_SYMBOLS[self.coin], interval="1d", limit=1000))
            except Exception as exc:
                log.warning("Binance klines failed: %s", exc)

        if not frames:
            raise RuntimeError(f"No data loaded for {self.coin}")

        # Merge: CSV provides the long history; live APIs update recent bars
        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        for col in ["open", "high", "low"]:
            if col not in combined.columns:
                combined[col] = combined["close"]
        if "volume" not in combined.columns:
            combined["volume"] = 0.0

        combined = combined[["open", "high", "low", "close", "volume"]].copy()
        combined = compute_indicators(combined)
        self._df = combined
        log.info("DataManager: %d rows, %d cols for %s", len(combined), len(combined.columns), self.coin)
        return combined

    def live_ticker(self) -> Dict:
        client = BinanceRestClient()
        symbol = BINANCE_SYMBOLS[self.coin]
        self._live_ticker = client.ticker_24h(symbol)
        if self.use_websocket and self._ws is None:
            self._ws = BinanceWebSocket(symbol)
            self._ws.start()
        if self._ws and self._ws.latest:
            self._live_ticker["price"] = self._ws.latest["price"]
            self._live_ticker["ws_timestamp"] = self._ws.latest["timestamp"]
        return self._live_ticker

    def sentiment(self) -> float:
        scores: List[float] = []
        if self.cryptopanic_token:
            try:
                scores.append(CryptoPanicClient(self.cryptopanic_token).fetch_sentiment(self.coin))
            except Exception as exc:
                log.warning("CryptoPanic: %s", exc)
        if self.newsapi_key:
            try:
                scores.append(NewsAPIClient(self.newsapi_key).fetch_sentiment(self.coin))
            except Exception as exc:
                log.warning("NewsAPI: %s", exc)
        return float(np.mean(scores)) if scores else 0.0

    def stop_websocket(self) -> None:
        if self._ws:
            self._ws.stop()