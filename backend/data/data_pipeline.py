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
import websocket

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
    for attempt in range(max_retries):
        if rate_limiter:
            rate_limiter.wait()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                log.warning("HTTP %s — retrying in %ss", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            raise
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
        prices  = pd.DataFrame(data["prices"],        columns=["timestamp_ms", "close"])
        volumes = pd.DataFrame(data["total_volumes"],  columns=["timestamp_ms", "volume"])
        df = prices.merge(volumes[["timestamp_ms", "volume"]], on="timestamp_ms")
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["timestamp_ms"])
        df.index = df.index.normalize()
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

    def klines(
        self,
        symbol:   str,
        interval: str = "1d",
        limit:    int = 1000,
        start_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch klines.  When start_ms is provided, only rows at or after that
        timestamp are requested (used by the incremental fetch path).
        """
        params: Dict = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms

        raw = _request_with_retry(
            f"{self.BASE}/klines",
            params=params,
            headers=self.headers,
            rate_limiter=_bnc_limiter,
        )
        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

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
# Binance WebSocket
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
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if self._ws:
            self._ws.close()

    def _on_message(self, ws, message: str) -> None:
        data = json.loads(message)
        self.latest = {
            "price":     float(data["c"]),
            "volume":    float(data["v"]),
            "timestamp": datetime.fromtimestamp(data["E"] / 1000, tz=timezone.utc).isoformat(),
        }
        self._ready.set()


# ---------------------------------------------------------------------------
# Fear & Greed Index — free, no API key required
# ---------------------------------------------------------------------------

_fg_limiter = RateLimiter(calls_per_second=0.2, name="feargreed")


def fetch_fear_greed_series(limit: int = 700) -> pd.Series:
    try:
        _fg_limiter.wait()
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": limit, "format": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        records = {
            pd.Timestamp(int(item["timestamp"]), unit="s", tz="UTC").normalize(): float(item["value"])
            for item in data
        }
        series = pd.Series(records, name="fear_greed").sort_index()
        log.info("Fear & Greed: %d data points loaded", len(series))
        return series
    except Exception as exc:
        log.warning("Fear & Greed fetch failed (using neutral 50): %s", exc)
        return pd.Series(dtype=float, name="fear_greed")


# ---------------------------------------------------------------------------
# Sentiment (VADER)
# ---------------------------------------------------------------------------

_cp_limiter   = RateLimiter(calls_per_second=5,    name="cryptopanic")
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
                    bull  = votes.get("positive", 0)
                    bear  = votes.get("negative", 0)
                    total = bull + bear
                    if total > 0:
                        scores.append((bull - bear) / total)
            except requests.exceptions.HTTPError as exc:
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
# CSV Loader
# ---------------------------------------------------------------------------

def load_csv(path: str, coin: str) -> pd.DataFrame:
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


def csv_fingerprint(path: str) -> str:
    h    = hashlib.sha256()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
    return h.hexdigest()


def load_csv_cached(path: str, coin: str, cache_dir: str = ".cache") -> pd.DataFrame:
    """Load CSV with SHA-256 fingerprint cache.  Cache is permanent — CSVs never change."""
    os.makedirs(cache_dir, exist_ok=True)
    fp         = csv_fingerprint(path)
    cache_path = Path(cache_dir) / f"{coin}_csv_{fp[:16]}.parquet"

    if cache_path.exists():
        log.info("CSV cache hit for %s (fp=%s…)", coin, fp[:8])
        return pd.read_parquet(cache_path)

    log.info("CSV cache miss for %s — processing CSV…", coin)
    df = load_csv(path, coin)
    df.to_parquet(cache_path)
    for old in Path(cache_dir).glob(f"{coin}_csv_*.parquet"):
        if old != cache_path:
            try:
                old.unlink()
            except Exception:
                pass
    log.info("CSV cached for %s at %s", coin, cache_path)
    return df


# ---------------------------------------------------------------------------
# INCREMENTAL OHLCV CACHE
# ---------------------------------------------------------------------------
# Problem this solves:
#   Previously every server restart merged CSV + Binance(1000 bars) + CoinGecko(90 days)
#   from scratch.  The fingerprint changed each day because the last date changed,
#   causing a full retrain of all models every single day.
#
# Solution:
#   1.  On FIRST load: fetch all sources, merge, save as
#       {coin}_ohlcv_{last_date_iso}.parquet  (the "base snapshot").
#   2.  On SUBSEQUENT loads: load the most recent snapshot, then fetch only
#       new daily candles from Binance AFTER the snapshot's last date, append
#       them, save a NEW snapshot, delete the old one.
#   3.  The training data is now the full base snapshot + today's new rows —
#       models only see genuinely new information when they retrain.
# ---------------------------------------------------------------------------

def _ohlcv_cache_path(coin: str, last_date: pd.Timestamp, cache_dir: str) -> Path:
    """Canonical path for a dated OHLCV snapshot."""
    date_str = last_date.strftime("%Y%m%d")
    return Path(cache_dir) / f"{coin}_ohlcv_{date_str}.parquet"


def _find_latest_ohlcv_snapshot(coin: str, cache_dir: str) -> Optional[Path]:
    """
    Return the path of the most recently dated OHLCV snapshot file for this
    coin, or None if no snapshot exists yet.
    """
    pattern = f"{coin}_ohlcv_*.parquet"
    candidates = sorted(Path(cache_dir).glob(pattern))
    return candidates[-1] if candidates else None


def load_ohlcv_incremental(
    coin:              str,
    csv_path:          Optional[str],
    cache_dir:         str,
    coingecko_api_key: Optional[str] = None,
    fetch_coingecko:   bool = True,
    fetch_binance:     bool = True,
) -> pd.DataFrame:
    """
    Smart OHLCV loader.

    FIRST RUN (no snapshot exists):
        - Loads CSV (cached by fingerprint — never re-parsed)
        - Fetches Binance last 1000 bars
        - Fetches CoinGecko last 90 days
        - Merges, deduplicates, saves full snapshot
        → This is the only slow data-fetch run.

    SUBSEQUENT RUNS (snapshot exists):
        - Loads the saved snapshot instantly from disk
        - Fetches ONLY the new daily candles from Binance since the
          snapshot's last date (usually 1-3 rows)
        - Appends new rows, saves a new snapshot, deletes the old one
        → This is fast: usually a single small API call.

    If running 2-3 days after the last snapshot the same logic applies —
    Binance returns all candles since the last snapshot date, so you never
    miss any days regardless of how long the server was offline.
    """
    os.makedirs(cache_dir, exist_ok=True)
    bnc = BinanceRestClient()
    symbol = BINANCE_SYMBOLS[coin]

    existing_snapshot = _find_latest_ohlcv_snapshot(coin, cache_dir)

    if existing_snapshot is not None:
        # ── Incremental path ────────────────────────────────────────────
        log.info("OHLCV snapshot found for %s: %s", coin, existing_snapshot.name)
        base_df = pd.read_parquet(existing_snapshot)
        last_known = base_df.index.max()

        # Fetch only candles strictly AFTER last known date
        # Add 1 day so we don't re-fetch the last known bar
        start_dt = last_known + pd.Timedelta(days=1)
        start_ms = int(start_dt.timestamp() * 1000)

        today_utc = pd.Timestamp.now(tz="UTC").normalize()
        if start_dt >= today_utc:
            log.info(
                "%s OHLCV already up-to-date (last: %s) — using snapshot as-is",
                coin, last_known.date(),
            )
            return base_df

        new_frames: List[pd.DataFrame] = []

        if fetch_binance:
            try:
                new_bars = bnc.klines(symbol, interval="1d", limit=1000, start_ms=start_ms)
                if not new_bars.empty:
                    new_frames.append(new_bars)
                    log.info(
                        "%s incremental Binance: %d new rows (%s → %s)",
                        coin, len(new_bars),
                        new_bars.index.min().date(), new_bars.index.max().date(),
                    )
                else:
                    log.info("%s Binance: no new rows since %s", coin, last_known.date())
            except Exception as exc:
                log.warning("Binance incremental fetch failed for %s: %s", coin, exc)

        if fetch_coingecko:
            try:
                days_needed = (today_utc - last_known).days + 2
                days_needed = min(days_needed, 90)
                cg = CoinGeckoClient(coingecko_api_key)
                cg_df = cg.market_chart(COINGECKO_IDS[coin], days=days_needed)
                # Keep only rows strictly after last_known
                cg_new = cg_df[cg_df.index > last_known]
                if not cg_new.empty:
                    new_frames.append(cg_new)
                    log.info(
                        "%s incremental CoinGecko: %d new rows",
                        coin, len(cg_new),
                    )
            except Exception as exc:
                log.warning("CoinGecko incremental fetch failed for %s: %s", coin, exc)

        if not new_frames:
            log.info("%s no new data found — snapshot is current", coin)
            return base_df

        # Merge new rows with base
        updated = pd.concat([base_df] + new_frames)
        updated = updated[~updated.index.duplicated(keep="last")].sort_index()
        updated = updated[["open", "high", "low", "close", "volume"]].copy()

        new_last = updated.index.max()
        new_path = _ohlcv_cache_path(coin, new_last, cache_dir)
        updated.to_parquet(new_path)
        log.info(
            "%s OHLCV snapshot updated: %d rows, last=%s → saved to %s",
            coin, len(updated), new_last.date(), new_path.name,
        )

        # Remove old snapshot(s) for this coin to save disk space
        for old in Path(cache_dir).glob(f"{coin}_ohlcv_*.parquet"):
            if old != new_path:
                try:
                    old.unlink()
                    log.info("Deleted old snapshot: %s", old.name)
                except Exception:
                    pass

        return updated

    else:
        # ── First-run path (full fetch) ──────────────────────────────────
        log.info("No OHLCV snapshot for %s — performing full initial load", coin)
        frames: List[pd.DataFrame] = []

        if csv_path and Path(csv_path).exists():
            frames.append(load_csv_cached(csv_path, coin, cache_dir))
        else:
            log.warning("CSV not found: %s", csv_path)

        if fetch_coingecko:
            try:
                cg = CoinGeckoClient(coingecko_api_key)
                frames.append(cg.market_chart(COINGECKO_IDS[coin], days=90))
            except Exception as exc:
                log.warning("CoinGecko failed: %s", exc)

        if fetch_binance:
            try:
                frames.append(bnc.klines(symbol, interval="1d", limit=1000))
            except Exception as exc:
                log.warning("Binance klines failed: %s", exc)

        if not frames:
            raise RuntimeError(f"No data loaded for {coin}")

        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        for col in ["open", "high", "low"]:
            if col not in combined.columns:
                combined[col] = combined["close"]
        if "volume" not in combined.columns:
            combined["volume"] = 0.0
        combined = combined[["open", "high", "low", "close", "volume"]].copy()

        # Save full snapshot
        last_date = combined.index.max()
        snap_path = _ohlcv_cache_path(coin, last_date, cache_dir)
        combined.to_parquet(snap_path)
        log.info(
            "%s full OHLCV snapshot saved: %d rows, last=%s → %s",
            coin, len(combined), last_date.date(), snap_path.name,
        )
        return combined


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, fear_greed: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Compute 60+ technical indicators + lag/return features.
    All operations use only past data — no look-ahead bias.
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    for span in [7, 14, 21, 50, 200]:
        df[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()
        df[f"sma_{span}"] = c.rolling(span, min_periods=1).mean()

    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    df["rsi_14"] = _rsi(c)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    sma20  = c.rolling(20, min_periods=1).mean()
    std20  = c.rolling(20, min_periods=1).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    bb_range       = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = bb_range / sma20.replace(0, np.nan)
    df["bb_pct"]   = (c - df["bb_lower"]) / bb_range

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=1).mean()

    direction = np.sign(c.diff()).fillna(0)
    df["obv"]  = (direction * v).cumsum()

    up_move   = h.diff()
    down_move = -l.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14     = tr.rolling(14, min_periods=1).mean()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).rolling(14, min_periods=1).mean() / atr14.replace(0, np.nan)
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr14.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"]   = dx.rolling(14, min_periods=1).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di

    log_ret        = np.log(c / c.shift())
    df["vol_7d"]   = log_ret.rolling(7,  min_periods=1).std() * np.sqrt(252)
    df["vol_30d"]  = log_ret.rolling(30, min_periods=1).std() * np.sqrt(252)
    df["vol_regime"] = (df["vol_7d"] > df["vol_30d"]).astype(float)

    roll_high = h.rolling(50, min_periods=1).max()
    roll_low  = l.rolling(50, min_periods=1).min()
    diff      = roll_high - roll_low
    df["fib_0"]   = roll_high
    df["fib_236"] = roll_high - 0.236 * diff
    df["fib_382"] = roll_high - 0.382 * diff
    df["fib_500"] = roll_high - 0.500 * diff
    df["fib_618"] = roll_high - 0.618 * diff
    df["fib_100"] = roll_low

    pp              = (h + l + c) / 3
    df["pivot"]     = pp
    df["resist_1"]  = 2 * pp - l
    df["support_1"] = 2 * pp - h
    df["resist_2"]  = pp + (h - l)
    df["support_2"] = pp - (h - l)

    df["log_return"]  = log_ret
    df["pct_change"]  = c.pct_change()
    df["price_range"] = (h - l) / c.replace(0, np.nan)
    df["vwap_approx"] = (
        (c * v).rolling(14, min_periods=1).sum()
        / v.rolling(14, min_periods=1).sum().replace(0, np.nan)
    )

    low14  = l.rolling(14, min_periods=1).min()
    high14 = h.rolling(14, min_periods=1).max()
    df["stoch_k"]    = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"]    = df["stoch_k"].rolling(3, min_periods=1).mean()
    df["williams_r"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    typical = (h + l + c) / 3
    sma_tp  = typical.rolling(20, min_periods=1).mean()
    mad     = typical.rolling(20, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    df["cci"]    = (typical - sma_tp) / (0.015 * mad.replace(0, np.nan))
    df["roc_10"] = c.pct_change(10) * 100

    # Lag return features
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"return_lag_{lag}"]     = c.pct_change(lag)
        df[f"log_return_lag_{lag}"] = np.log(c / c.shift(lag))

    df["vol_change_7d"] = v.pct_change(7)
    df["vol_ratio_30d"] = v / v.rolling(30, min_periods=1).mean().replace(0, np.nan)
    df["vol_ma_7"]      = v.rolling(7, min_periods=1).mean()

    df["price_dist_sma50"]  = (c - df["sma_50"])  / df["sma_50"].replace(0, np.nan)
    df["price_dist_sma200"] = (c - df["sma_200"]) / df["sma_200"].replace(0, np.nan)
    df["price_dist_ema21"]  = (c - df["ema_21"])  / df["ema_21"].replace(0, np.nan)

    df["momentum_7"]  = c / c.shift(7).replace(0, np.nan)  - 1
    df["momentum_14"] = c / c.shift(14).replace(0, np.nan) - 1
    df["momentum_30"] = c / c.shift(30).replace(0, np.nan) - 1

    df["return_skew_30"] = log_ret.rolling(30, min_periods=10).skew()
    df["return_kurt_30"] = log_ret.rolling(30, min_periods=10).kurt()

    # Fear & Greed Index (free)
    if fear_greed is not None and len(fear_greed) > 0:
        fg_aligned            = fear_greed.reindex(df.index, method="ffill").fillna(50.0)
        df["fear_greed_norm"] = fg_aligned.values / 100.0
        df["fear_greed_ma7"]  = (
            pd.Series(fg_aligned.values, index=df.index)
            .rolling(7, min_periods=1).mean() / 100.0
        )
        log.info("Fear & Greed Index merged into features")
    else:
        df["fear_greed_norm"] = 0.5
        df["fear_greed_ma7"]  = 0.5

    return df


# ---------------------------------------------------------------------------
# Feature Matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df:        pd.DataFrame,
    window:    int = 60,
    horizons:  Optional[List[int]] = None,
    sentiment: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build sliding-window (X, y) arrays.

    TARGET: log-return from current close to each future horizon.
      y[i][h] = log(close[i+window+h-1] / close[i+window-1])
    At inference: predicted_price = current_price * exp(log_return)
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
    df_feat = df_feat.fillna(df_feat.mean())

    feat_names = list(df_feat.columns)
    arr        = df_feat.values.astype(np.float32)
    close_arr  = df["close"].values.astype(np.float32)

    max_horizon = max(horizons)
    n_samples   = len(arr) - window - max_horizon
    if n_samples <= 0:
        raise ValueError(f"Not enough data: need {window + max_horizon} rows, got {len(arr)}")

    X_list, y_list = [], []
    for i in range(n_samples):
        X_list.append(arr[i: i + window])
        current_close = float(close_arr[i + window - 1])
        y_list.append([
            float(np.log(max(close_arr[i + window + h - 1], 1e-8) / max(current_close, 1e-8)))
            for h in horizons
        ])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    log.info(
        "Feature matrix: X=%s  y=%s  (log-return targets, no look-ahead)",
        X.shape, y.shape,
    )
    return X, y, feat_names


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

@dataclass
class DataManager:
    """Orchestrates all data sources for a single coin."""

    coin:              str
    csv_path:          Optional[str]  = None
    cache_dir:         str            = ".cache"
    coingecko_api_key: Optional[str]  = None
    cryptopanic_token: Optional[str]  = None
    newsapi_key:       Optional[str]  = None
    use_websocket:     bool           = False
    fetch_fear_greed:  bool           = True

    _df:          pd.DataFrame          = field(default_factory=pd.DataFrame, init=False, repr=False)
    _live_ticker: Dict                  = field(default_factory=dict,          init=False, repr=False)
    _ws:          Optional[BinanceWebSocket] = field(default=None,             init=False, repr=False)
    _fear_greed:  Optional[pd.Series]   = field(default=None,                  init=False, repr=False)

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self.coin = self.coin.upper()

    def load(self, fetch_coingecko: bool = True, fetch_binance: bool = True) -> pd.DataFrame:
        """
        Load OHLCV data using the incremental snapshot cache.
        On first call: full load + save snapshot (~same time as before).
        On subsequent calls: load snapshot + fetch only new rows (fast).
        """
        # Raw OHLCV — incremental cache handles CSV + Binance + CoinGecko
        raw_ohlcv = load_ohlcv_incremental(
            coin=self.coin,
            csv_path=self.csv_path,
            cache_dir=self.cache_dir,
            coingecko_api_key=self.coingecko_api_key,
            fetch_coingecko=fetch_coingecko,
            fetch_binance=fetch_binance,
        )

        # Fear & Greed Index (free, no key needed)
        if self.fetch_fear_greed:
            try:
                self._fear_greed = fetch_fear_greed_series(limit=700)
            except Exception as exc:
                log.warning("Fear & Greed: %s", exc)
                self._fear_greed = None

        combined  = compute_indicators(raw_ohlcv, fear_greed=self._fear_greed)
        self._df  = combined
        log.info(
            "DataManager: %d rows, %d cols for %s",
            len(combined), len(combined.columns), self.coin,
        )
        return combined

    def live_ticker(self) -> Dict:
        client    = BinanceRestClient()
        symbol    = BINANCE_SYMBOLS[self.coin]
        self._live_ticker = client.ticker_24h(symbol)
        if self.use_websocket and self._ws is None:
            self._ws = BinanceWebSocket(symbol)
            self._ws.start()
        if self._ws and self._ws.latest:
            self._live_ticker["price"]        = self._ws.latest["price"]
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