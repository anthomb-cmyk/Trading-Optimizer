"""
Multi-timeframe data loader with disk caching and indicator pre-computation.

Fetches OHLCV data directly from Yahoo Finance's V8 chart API (no yfinance package).
SSL verification is disabled to support corporate-proxy environments.
Cached as Parquet. Adds ATR, EMA, and swing high/low columns used across all strategies.
"""
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import urllib3
import numpy as np
import pandas as pd
import requests
from ta.volatility import AverageTrueRange
from ta.trend     import EMAIndicator
from ta.momentum  import RSIIndicator

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config.logger import get_logger
from config.settings import (
    CACHE_EXPIRY_HOURS,
    DATA_DIR,
    INSTRUMENTS,
    LOOKBACK_DAYS,
    SMC,
    TIMEFRAMES,
    VOLATILITY,
    WARMUP_BARS,
)

log = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_path(ticker: str, tf: str) -> Path:
    key = hashlib.md5(f"{ticker}_{tf}".encode()).hexdigest()[:8]
    return DATA_DIR / f"{ticker.replace('=', '')}_{tf}_{key}.parquet"


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age > CACHE_EXPIRY_HOURS * 3600


# Futures tickers have no direct Yahoo chart history; proxy them to ETFs.
# Stooq format equivalents: SPY.US, QQQ.US — kept as reference comments.
_FUTURES_PROXY_MAP: dict = {
    "MES=F": "SPY",   # Micro E-mini S&P 500 → SPY (stooq: SPY.US)
    "ES=F":  "SPY",   # E-mini S&P 500       → SPY (stooq: SPY.US)
    "MNQ=F": "QQQ",   # Micro E-mini Nasdaq  → QQQ (stooq: QQQ.US)
    "NQ=F":  "QQQ",   # E-mini Nasdaq        → QQQ (stooq: QQQ.US)
    "MGC=F": "GLD",   # Micro Gold Futures   → GLD (backtest proxy)
    "YG=F":  "GLD",   # E-mini Gold Futures  → GLD (backtest proxy)
}

# Yahoo Finance V8 interval strings. 3m and 4h are not natively supported;
# approximate with the nearest available interval.
_YF_INTERVAL_MAP: dict = {
    "1m":  "1m",
    "3m":  "5m",    # Yahoo has no 3m; use 5m
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "1h",    # Yahoo has no 4h; use 1h
    "1d":  "1d",
    "1w":  "1wk",
}

# Yahoo V8 intraday limits (days of history available per interval)
_YF_INTRADAY_LIMIT: dict = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "1h": 730,
}

_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://finance.yahoo.com",
}


def _yahoo_interval(tf: str) -> str:
    return _YF_INTERVAL_MAP.get(tf, "1d")


def _fetch_yahoo(
    ticker:          str,
    tf:              str,
    days:            int,
    retries:         int = 5,
    retry_delay_s:   int = 15,
    fallback_ticker: str = "",
) -> pd.DataFrame:
    interval = _yahoo_interval(tf)

    # Cap days for intraday intervals based on Yahoo's history limits
    if interval in _YF_INTRADAY_LIMIT:
        days = min(days, _YF_INTRADAY_LIMIT[interval])

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    p1    = int(start.timestamp())
    p2    = int(end.timestamp())

    # Resolve futures tickers to ETF proxies for Yahoo chart API
    primary  = _FUTURES_PROXY_MAP.get(ticker, ticker)
    fallback = _FUTURES_PROXY_MAP.get(fallback_ticker, fallback_ticker) if fallback_ticker else ""

    tickers_to_try = [primary]
    if fallback and fallback != primary:
        tickers_to_try.append(fallback)

    for current_ticker in tickers_to_try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{current_ticker}"
            f"?interval={interval}&period1={p1}&period2={p2}&events=history"
        )
        for attempt in range(1, retries + 1):
            log.info(
                "Fetching %s %s (%d days) via Yahoo V8, attempt %d/%d…",
                current_ticker, tf, days, attempt, retries,
            )
            try:
                resp = requests.get(
                    url,
                    headers=_YF_HEADERS,
                    timeout=30,
                    verify=False,  # corporate proxy / untrusted CA
                )
                data = resp.json()
            except Exception as exc:
                log.warning("Yahoo V8 request error attempt %d/%d for %s: %s",
                            attempt, retries, current_ticker, exc)
                data = {}

            result = (data.get("chart") or {}).get("result")
            if result:
                if current_ticker != primary:
                    log.info("Using fallback ticker %s for %s.",
                             current_ticker, ticker)
                r        = result[0]
                ts       = r["timestamp"]
                q        = r["indicators"]["quote"][0]
                adjclose = (
                    r["indicators"].get("adjclose", [{}])[0].get("adjclose")
                    or q["close"]
                )
                df = pd.DataFrame(
                    {
                        "open":   q["open"],
                        "high":   q["high"],
                        "low":    q["low"],
                        "close":  adjclose,
                        "volume": q["volume"],
                    },
                    index=pd.to_datetime(ts, unit="s", utc=True),
                )
                df.index.name = "datetime"
                return df.dropna()

            if attempt < retries:
                log.warning(
                    "Empty/error response for %s — retrying in %ds… (%d/%d)",
                    current_ticker, retry_delay_s, attempt, retries,
                )
                time.sleep(retry_delay_s)

        log.warning("All %d attempts failed for Yahoo ticker %s.", retries, current_ticker)

    raise ValueError(
        f"No data returned for {ticker} (Yahoo: {primary}, "
        f"fallback: {fallback or 'none'}) {tf} after {retries} attempts each. "
        "Yahoo Finance V8 API unreachable — check network connectivity."
    )


# ── Indicator layer ────────────────────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ATR, EMAs, swing pivots, and VWAP-style columns."""
    atr_period = VOLATILITY["atr_period"]
    df["atr"]     = AverageTrueRange(df["high"], df["low"], df["close"], window=atr_period).average_true_range()
    df["atr_avg"] = df["atr"].rolling(50).mean()

    # EMAs for HTF bias
    for length in (20, 50, 200):
        df[f"ema_{length}"] = EMAIndicator(df["close"], window=length).ema_indicator()

    # RSI for confluence
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    # Volume moving average
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["vol_ma"] = df["volume"].rolling(20).mean()
    else:
        df["vol_ma"] = np.nan

    # Swing highs / lows (vectorized)
    lb = SMC["swing_lookback"]
    df["swing_high"] = _swing_highs(df["high"].values, lb)
    df["swing_low"]  = _swing_lows(df["low"].values, lb)

    # Body size
    df["body"]  = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Candle direction
    df["bullish"] = df["close"] > df["open"]

    return df


def _swing_highs(highs: np.ndarray, lookback: int) -> np.ndarray:
    n      = len(highs)
    result = np.zeros(n, dtype=bool)
    lb     = lookback
    for i in range(lb, n - lb):
        window = highs[i - lb : i + lb + 1]
        if highs[i] == window.max():
            result[i] = True
    return result


def _swing_lows(lows: np.ndarray, lookback: int) -> np.ndarray:
    n      = len(lows)
    result = np.zeros(n, dtype=bool)
    lb     = lookback
    for i in range(lb, n - lb):
        window = lows[i - lb : i + lb + 1]
        if lows[i] == window.min():
            result[i] = True
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def load_bars(
    symbol:      str = "MES",
    timeframe:   str = "15m",
    days:        int = LOOKBACK_DAYS,
    use_cache:   bool = True,
    add_indicators: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV bars for *symbol* at *timeframe*.

    Returns a DataFrame with DatetimeIndex (UTC), lowercase OHLCV columns,
    and pre-computed indicator columns (ATR, EMAs, swings, etc.).
    Data is fetched from Yahoo Finance V8 chart API (no yfinance package).
    Futures tickers (MES=F, MNQ=F) are proxied to SPY/QQQ for history.
    The first WARMUP_BARS rows are kept in the returned frame but
    flagged via `df.attrs['warmup_end_idx']` so strategies can skip them.
    """
    instrument = INSTRUMENTS.get(symbol)
    if instrument is None:
        raise ValueError(f"Unknown symbol '{symbol}'. Valid: {list(INSTRUMENTS)}")

    ticker          = instrument["ticker"]
    fallback_ticker = instrument.get("fallback_ticker", "")
    cache           = _cache_path(ticker, timeframe)

    if use_cache and not _is_stale(cache):
        log.debug("Loading %s %s from cache: %s", symbol, timeframe, cache.name)
        df = pd.read_parquet(cache)
    else:
        df = _fetch_yahoo(ticker, timeframe, days, fallback_ticker=fallback_ticker)
        if add_indicators:
            df = _add_indicators(df)
        df.to_parquet(cache, engine="pyarrow", compression="snappy")
        log.info("Cached %s %s → %s (%d bars)", symbol, timeframe, cache.name, len(df))

    if add_indicators and "atr" not in df.columns:
        df = _add_indicators(df)

    df.attrs["symbol"]        = symbol
    df.attrs["timeframe"]     = timeframe
    df.attrs["warmup_end_idx"] = WARMUP_BARS
    return df


def load_multi_timeframe(
    symbol:     str = "MES",
    timeframes: Optional[list] = None,
    days:       int = LOOKBACK_DAYS,
    use_cache:  bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple timeframes for *symbol*.
    Returns dict keyed by timeframe string.
    """
    if timeframes is None:
        timeframes = ["1d", "4h", "1h", "15m", "5m"]

    result = {}
    for tf in timeframes:
        try:
            result[tf] = load_bars(symbol, tf, days=days, use_cache=use_cache)
        except Exception as exc:
            log.warning("Failed to load %s %s: %s", symbol, tf, exc)
    return result


def align_htf_bias(
    primary: pd.DataFrame,
    htf:     pd.DataFrame,
    col:     str = "ema_200",
) -> pd.Series:
    """
    Forward-fill *col* from *htf* onto the index of *primary*.
    Returns a Series of bias values aligned to primary's index.
    Used for higher timeframe trend filter.
    """
    if col not in htf.columns:
        raise KeyError(f"Column '{col}' not in HTF dataframe.")

    htf_series = htf[col].reindex(primary.index, method="ffill")
    return htf_series


def get_session_mask(df: pd.DataFrame, session: str) -> pd.Series:
    """
    Return boolean mask where True = bar falls inside *session*.
    Sessions defined in config.settings.SESSIONS (UTC).
    """
    from config.settings import SESSIONS
    sess = SESSIONS.get(session)
    if sess is None:
        raise ValueError(f"Unknown session '{session}'")

    idx  = df.index
    hour = idx.hour + idx.minute / 60.0
    start_h = _time_to_float(sess["start"])
    end_h   = _time_to_float(sess["end"])
    return pd.Series((hour >= start_h) & (hour < end_h), index=idx)


def get_kill_zone_mask(df: pd.DataFrame, zone: str) -> pd.Series:
    """Boolean mask for kill zone *zone* (see KILL_ZONES in settings)."""
    from config.settings import KILL_ZONES
    kz = KILL_ZONES.get(zone)
    if kz is None:
        raise ValueError(f"Unknown kill zone '{zone}'")

    idx   = df.index
    hour  = idx.hour + idx.minute / 60.0
    start = _time_to_float(kz["start"])
    end   = _time_to_float(kz["end"])
    return pd.Series((hour >= start) & (hour < end), index=idx)


def _time_to_float(t: str) -> float:
    h, m = t.split(":")
    return int(h) + int(m) / 60.0


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "MES"
    tf  = sys.argv[2] if len(sys.argv) > 2 else "15m"
    df  = load_bars(sym, tf, use_cache=False)
    print(df.tail(5).to_string())
    print(f"\n{len(df)} bars loaded. Columns: {list(df.columns)}")
